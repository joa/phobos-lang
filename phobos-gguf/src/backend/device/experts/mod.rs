// Expert streaming: a block's experts live in a pinned host mirror, a
// share of them in device cache slots, and every miss crosses the bus once,
// at the block's sync point, into the slot it will be read from.
//
// A slab per stack kind and format (a file mixes formats per block, and a
// slab has one row stride), each `[slots * n, rb]` in the grouped layout the
// decode matvec reads, with an `f16` scale plane apiece. A block owns
// `per_block` consecutive slots of each of its three slabs plus one zeroed
// spare, from a base of its own; a slot holds one expert of one block across
// the three. Eviction is least recently used within the block, stamped by
// routing step.

mod grouped;
mod mirror;
mod op;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use cust::event::Event;
use cust::memory::{DeviceBuffer, LockedBuffer};
use cust::stream::Stream;
use phobos_kernels::cuda_ok;

use super::DeviceBackend;
use super::kernels::MOE_USED;
use crate::backend::ExpertsBuf;
use crate::experts::{ExpertSet, ExpertStack};
use crate::quant::Quant;
use mirror::Mirror;

/// Bytes a slab may not exceed: the decode matvec indexes in 32 bits.
const SLAB_LIMIT: usize = 1 << 31;

/// Rows one `moe` call may carry: twice the runtime's prompt batch, since
/// the checks feed longer passes than the runtime does.
pub(super) const MAX_ROWS: usize = 1024;

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum Kind {
    Gate,
    Up,
    Down,
}

pub(super) const KINDS: [Kind; 3] = [Kind::Gate, Kind::Up, Kind::Down];

impl Kind {
    pub(super) fn stack(self, set: &ExpertSet) -> &ExpertStack {
        match self {
            Kind::Gate => &set.gate,
            Kind::Up => &set.up,
            Kind::Down => &set.down,
        }
    }
}

pub(super) struct Experts {
    pub(super) per_block: usize,
    pub(super) blocks: Vec<BlockExperts>,
    /// By stack kind and format; `None` until the first `moe`.
    pub(super) slabs: Option<HashMap<(Kind, Quant), Slab>>,
    /// Bytes and expert sets `budget_streamed` gave, once it did.
    budget: Option<(usize, usize)>,
    /// A cap the user put on the cache, in bytes.
    pub(super) limit: Option<usize>,
    /// The iota row the top-k carries, by expert count.
    pub(super) iota: HashMap<usize, DeviceBuffer<f32>>,
    /// Routing steps so far, the stamp a slot takes when used: one a row a
    /// block, so a prompt pass's rows do not all stamp alike.
    tick: u64,
    pub(super) stats: ExpertStats,
}

#[derive(Default, Clone, Copy)]
pub(super) struct ExpertStats {
    pub(super) hits: u64,
    pub(super) misses: u64,
    pub(super) bytes: u64,
    pub(super) pinned_bytes: u64,
    pub(super) prefetches: u64,
    pub(super) prefetch_hits: u64,
    pub(super) cpu_misses: u64,
    pub(super) cpu_nanos: u64,
}

pub(super) struct BlockExperts {
    pub(super) set: Arc<ExpertSet>,
    pub(super) mirror: Mirror,
    /// `expert -> slot within the block`, `NONE` where not resident.
    slot_of: Vec<u32>,
    /// Per slot: the expert held and the tick it was last used at.
    held: Vec<(u32, u64)>,
    /// Whether a slot's expert arrived by prefetch and is unread since.
    prefetched: Vec<bool>,
    /// The block's first slot in each of its three slabs.
    pub(super) base: [usize; 3],
    /// `SLOT[rows, 8]`, `TOPK[rows, 8]` and `W[rows, 8]`, pointer-stable so
    /// a segment's graph needs no patching, with page-locked mirrors of
    /// the first two.
    pub(super) slot_table: DeviceBuffer<i32>,
    slot_host: LockedBuffer<i32>,
    pub(super) topk: DeviceBuffer<i32>,
    pub(super) topk_host: LockedBuffer<i32>,
    pub(super) weights: DeviceBuffer<f32>,
    /// The previous block's prediction for this one, and the event its
    /// copies complete at.
    pub(super) look_topk: DeviceBuffer<i32>,
    pub(super) look_host: LockedBuffer<i32>,
    pub(super) look_w: DeviceBuffer<f32>,
    pub(super) fetched: Option<Event>,
}

pub(super) struct Slab {
    pub(super) bytes: DeviceBuffer<u8>,
    pub(super) d: DeviceBuffer<u16>,
    pub(super) n: usize,
    pub(super) nb: usize,
    pub(super) rb: usize,
}

impl Slab {
    fn new(slots: usize, stack: &ExpertStack) -> Result<Slab> {
        let (n, nb) = (stack.n(), stack.blocks_per_row());
        let rb = nb * stack.quant().device_block().1;
        let bytes = slots * n * rb;
        ensure!(bytes < SLAB_LIMIT, "a slab of {slots} slots would be {bytes} bytes, past the 32-bit index");
        // SAFETY: every slot is written before a kernel reads it.
        let bytes = unsafe { DeviceBuffer::<u8>::uninitialized(bytes)? };
        let d = DeviceBuffer::from_slice(&vec![0u16; slots * n * nb])?;
        Ok(Slab { bytes, d, n, nb, rb })
    }

    /// Device addresses of slot `s`'s rows and of their scale plane.
    pub(super) fn at(&self, s: usize) -> (u64, u64) {
        (
            self.bytes.as_device_ptr().as_raw() + (s * self.n * self.rb) as u64,
            self.d.as_device_ptr().as_raw() + (s * self.n * self.nb * 2) as u64,
        )
    }

    pub(super) fn slot_bytes(&self) -> usize {
        self.n * self.rb
    }
}

/// An asynchronous host-to-device copy of `len` bytes.
///
/// # Safety
/// `src` is page-locked and both sides live until the stream reaches the
/// copy.
unsafe fn copy_async(dst: u64, src: *const std::ffi::c_void, len: usize, stream: &Stream) -> Result<()> {
    cuda_ok(unsafe { cust::sys::cuMemcpyHtoDAsync_v2(dst, src, len, stream.as_inner()) }, "copying to the device")
}

impl BlockExperts {
    /// Row `r`'s chosen experts, as the last sync point read them back.
    pub(super) fn chosen(&self, r: usize) -> Result<[usize; MOE_USED]> {
        let n = self.set.count() as i32;
        let mut out = [0; MOE_USED];
        for (o, &id) in out.iter_mut().zip(&self.topk_host.as_slice()[r * MOE_USED..(r + 1) * MOE_USED]) {
            ensure!((0..n).contains(&id), "the router chose expert {id} of {n}");
            *o = id as usize;
        }
        Ok(out)
    }

    fn stamp(&mut self, ids: &[usize], tick: u64) {
        for &e in ids {
            let s = self.slot_of[e];
            if s != NONE {
                self.held[s as usize].1 = tick;
            }
        }
    }

    /// The least recently used slot not stamped `tick`.
    fn victim(&self, tick: u64) -> Option<usize> {
        (0..self.held.len()).filter(|&s| self.held[s].1 != tick).min_by_key(|&s| self.held[s].1)
    }

    /// Publishes row `r`'s slot table from its pinned mirror.
    pub(super) fn publish(&mut self, r: usize, slots: &[i32; MOE_USED], stream: &Stream) -> Result<()> {
        let at = r * MOE_USED;
        self.slot_host.as_mut_slice()[at..at + MOE_USED].copy_from_slice(slots);
        let dst = self.slot_table.as_device_ptr().as_raw() + (at * size_of::<i32>()) as u64;
        // SAFETY: the row's region is not written again before the next
        // sync point.
        unsafe { copy_async(dst, self.slot_host.as_slice()[at..].as_ptr().cast(), size_of_val(slots), stream) }
    }
}

impl Experts {
    pub(super) fn new() -> Experts {
        Experts {
            per_block: 0,
            blocks: Vec::new(),
            slabs: None,
            budget: None,
            limit: None,
            iota: HashMap::new(),
            tick: 0,
            stats: ExpertStats::default(),
        }
    }

    pub(super) fn slab(&self, b: usize, kind: Kind) -> &Slab {
        let quant = kind.stack(&self.blocks[b].set).quant();
        &self.slabs.as_ref().expect("laid out")[&(kind, quant)]
    }

    /// A new routing step: `ids` of `block` stamped, the tick returned.
    pub(super) fn step(&mut self, block: usize, ids: &[usize]) -> u64 {
        self.tick += 1;
        self.blocks[block].stamp(ids, self.tick);
        self.tick
    }

    pub(super) fn resident(&self, block: usize, e: usize) -> bool {
        self.blocks[block].slot_of[e] != NONE
    }

    /// Expert `e` of `block` in a slot: the one it is in, or the least
    /// recently used one, filled from the mirror on `stream`. `Ok(None)`
    /// when every slot is stamped `tick`.
    pub(super) fn place(&mut self, block: usize, e: usize, tick: u64, stream: &Stream) -> Result<Option<usize>> {
        let b = &mut self.blocks[block];
        let s = b.slot_of[e];
        if s != NONE {
            self.stats.hits += 1;
            if std::mem::take(&mut b.prefetched[s as usize]) {
                self.stats.prefetch_hits += 1;
            }
            return Ok(Some(s as usize));
        }
        self.stats.misses += 1;
        let Some(victim) = b.victim(tick) else {
            return Ok(None);
        };
        self.fill(block, e, victim, tick, stream)?;
        Ok(Some(victim))
    }

    /// Expert `e` copied into `victim`'s slot of `block` on `stream`.
    pub(super) fn fill(&mut self, block: usize, e: usize, victim: usize, tick: u64, stream: &Stream) -> Result<()> {
        let slabs = self.slabs.as_ref().expect("laid out");
        let b = &mut self.blocks[block];
        let (old, _) = b.held[victim];
        if old != NONE {
            b.slot_of[old as usize] = NONE;
        }
        let src = b.mirror.expert(e);
        for (k, kind) in KINDS.into_iter().enumerate() {
            let slab = &slabs[&(kind, kind.stack(&b.set).quant())];
            let (bytes_at, d_at) = slab.at(b.base[k] + victim);
            ensure!(
                src[k].1 == slab.slot_bytes() && src[3 + k].1 == slab.n * slab.nb * 2,
                "mirror and slab disagree on an expert's size: {} against {}",
                src[k].1,
                slab.slot_bytes()
            );
            for (dst, (ptr, len)) in [(bytes_at, src[k]), (d_at, src[3 + k])] {
                // SAFETY: the mirror is pinned and outlives the copy.
                unsafe { copy_async(dst, ptr, len, stream)? };
                self.stats.bytes += len as u64;
            }
        }
        b.held[victim] = (e as u32, tick);
        b.slot_of[e] = victim as u32;
        b.prefetched[victim] = false;
        Ok(())
    }

    pub(super) fn mark_prefetched(&mut self, block: usize, slot: usize) {
        self.blocks[block].prefetched[slot] = true;
        self.stats.prefetches += 1;
    }
}

impl DeviceBackend {
    pub(super) fn set_streamed_budget(&self, resident_bytes: usize, sets: usize) -> Result<()> {
        if self.experts.borrow().budget.is_some() {
            return Ok(());
        }
        let (free, _) = cust::memory::mem_get_info()?;
        let room = free.saturating_sub(resident_bytes).saturating_sub(crate::runtime::RESERVE_BYTES);
        let budget = match self.experts.borrow().limit {
            Some(asked) if asked > room => {
                phobos_base::log::emit(
                    phobos_base::log::Level::Info,
                    format_args!(
                        "expert cache: {} MiB asked for, {} MiB left after the resident weights and the reserve; taking what is left",
                        asked >> 20,
                        room >> 20
                    ),
                );
                room
            }
            Some(asked) => asked,
            None => room,
        };
        ensure!(budget > 0 && sets > 0, "no device memory is left for an expert cache after {resident_bytes} bytes of resident weights");
        self.experts.borrow_mut().budget = Some((budget, sets));
        Ok(())
    }

    /// The block's mirror, built now; the slabs wait for the first `moe`,
    /// by which time every block has registered.
    pub(super) fn register_experts(&self, key: &str, set: &Arc<ExpertSet>) -> Result<ExpertsBuf> {
        if let Some(&buf) = self.expert_keys.borrow().get(key) {
            return Ok(buf);
        }
        let mut experts = self.experts.borrow_mut();
        ensure!(experts.slabs.is_none(), "an expert set registered after the cache was laid out");
        let n_expert = set.count();
        let mirror = Mirror::build(set).with_context(|| format!("mirroring the experts of {key}"))?;
        experts.stats.pinned_bytes += mirror.bytes() as u64;
        experts.iota.entry(n_expert).or_insert_with(|| {
            DeviceBuffer::from_slice(&(0..n_expert).map(|i| i as f32).collect::<Vec<_>>()).expect("a few hundred floats")
        });
        let ints = |len: usize| DeviceBuffer::from_slice(&vec![0i32; len]);
        let table = MAX_ROWS * MOE_USED;
        experts.blocks.push(BlockExperts {
            set: Arc::clone(set),
            mirror,
            slot_of: vec![NONE; n_expert],
            held: Vec::new(),
            prefetched: Vec::new(),
            base: [0; 3],
            slot_table: ints(table)?,
            slot_host: LockedBuffer::new(&0i32, table)?,
            topk: ints(table)?,
            topk_host: LockedBuffer::new(&0i32, table)?,
            weights: DeviceBuffer::from_slice(&vec![0f32; table])?,
            look_topk: ints(MOE_USED)?,
            look_host: LockedBuffer::new(&0i32, MOE_USED)?,
            look_w: DeviceBuffer::from_slice(&[0f32; MOE_USED])?,
            fetched: None,
        });
        let buf = ExpertsBuf(experts.blocks.len() - 1);
        self.expert_keys.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    /// The slabs, laid out at the first `moe`: the budget shared equally
    /// over the blocks, each with at least a token's worth plus a zeroed
    /// spare, one slab per stack kind and format.
    pub(super) fn ensure_slabs(&self, experts: &mut Experts) -> Result<()> {
        if experts.slabs.is_some() {
            return Ok(());
        }
        let (budget, sets) = experts.budget.context("the expert cache has no budget: budget_streamed was not called")?;
        let blocks = experts.blocks.len();
        ensure!(blocks == sets, "{blocks} expert sets registered of the {sets} the model has");
        let slot_bytes = experts
            .blocks
            .iter()
            .map(|b| KINDS.iter().map(|k| k.stack(&b.set).grouped_bytes()).sum::<usize>())
            .max()
            .context("no expert set is registered")?;
        let mut counts: HashMap<(Kind, Quant), usize> = HashMap::new();
        for b in &experts.blocks {
            for kind in KINDS {
                *counts.entry((kind, kind.stack(&b.set).quant())).or_default() += 1;
            }
        }
        // The spare slot a block gets comes out of the budget too.
        let mut per_block = (budget / slot_bytes / blocks).saturating_sub(1);
        for (&(kind, quant), &count) in &counts {
            let stack = experts.blocks.iter().map(|b| kind.stack(&b.set)).find(|s| s.quant() == quant).expect("counted");
            per_block = per_block.min(SLAB_LIMIT / stack.grouped_bytes() / count - 1);
        }
        ensure!(
            per_block >= MOE_USED,
            "the expert cache budget of {} MiB holds {per_block} experts a block; a token needs {MOE_USED}",
            budget >> 20
        );

        let stride = per_block + 1;
        let mut slabs = HashMap::new();
        let mut next: HashMap<(Kind, Quant), usize> = HashMap::new();
        for b in &mut experts.blocks {
            for (i, kind) in KINDS.into_iter().enumerate() {
                let stack = kind.stack(&b.set);
                let key = (kind, stack.quant());
                if let std::collections::hash_map::Entry::Vacant(v) = slabs.entry(key) {
                    v.insert(Slab::new(stride * counts[&key], stack)?);
                }
                let at = next.entry(key).or_default();
                b.base[i] = *at;
                *at += stride;
                let slab = &slabs[&key];
                // SAFETY: the spare slot is inside the slab just made.
                cuda_ok(unsafe { cust::sys::cuMemsetD8_v2(slab.at(b.base[i] + per_block).0, 0, slab.slot_bytes()) }, "zeroing a block's spare slot")?;
            }
            b.held = vec![(NONE, 0); per_block];
            b.prefetched = vec![false; per_block];
        }
        let total: usize = slabs.values().map(|s| s.bytes.len()).sum();
        experts.per_block = per_block;
        experts.slabs = Some(slabs);
        phobos_base::log::emit(
            phobos_base::log::Level::Info,
            format_args!(
                "expert cache: {} slots of up to {:.2} MiB, {per_block} a block, {:.0} MiB in {} slabs",
                per_block * blocks,
                slot_bytes as f64 / (1 << 20) as f64,
                total as f64 / (1 << 20) as f64,
                counts.len()
            ),
        );
        Ok(())
    }

    pub(super) fn expert_stats(&self) -> ExpertStats {
        self.experts.borrow().stats
    }
}
