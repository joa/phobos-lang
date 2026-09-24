// Expert streaming: a block's experts live in a pinned host mirror and a
// share of them in device cache slots. A decode step's misses are computed
// on the host from the mirror (`handoff.rs`); a prompt pass's cross the bus
// once, at the block's sync point, into the slot they will be read from,
// or go to the host too (`grouped.rs`).

mod grouped;
mod handoff;
mod mapped;
mod mirror;
mod op;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use cust::event::{Event, EventFlags};
use cust::memory::{DeviceBuffer, DeviceCopy, LockedBuffer};
use cust::stream::Stream;
use phobos_kernels::cuda_ok;

use super::DeviceBackend;
use super::kernels::MOE_USED;
use crate::backend::ExpertsBuf;
use crate::experts::{ExpertSet, ExpertStack, Stack};
use crate::quant::Quant;
use crate::simd::StartedRow;
use handoff::Handoff;
use mapped::Mapped;
use mirror::Mirror;

/// Bytes a slab may not exceed: the decode matvec indexes in 32 bits.
const SLAB_LIMIT: usize = 1 << 31;

/// Rows one `moe` call may carry: twice the runtime's prompt batch, since
/// the checks feed longer passes than the runtime does.
const MAX_ROWS: usize = 1024;

const NONE: u32 = u32::MAX;

pub(super) struct Experts {
    per_block: usize,
    blocks: Vec<BlockExperts>,
    /// By stack and format; `None` until the first `moe`.
    slabs: Option<HashMap<(Stack, Quant), Slab>>,
    /// Bytes and expert sets `budget_streamed` gave, once it did.
    budget: Option<(usize, usize)>,
    /// A cap the user put on the cache, in bytes.
    pub(super) limit: Option<usize>,
    /// The iota row the top-k carries, by expert count.
    iota: HashMap<usize, DeviceBuffer<f32>>,
    /// Routing steps so far, the stamp a slot takes when used: one a row a
    /// block, so a prompt pass's rows do not all stamp alike.
    tick: u64,
    stats: ExpertStats,
    /// The grouped prompt path's tables, once it has run.
    grouped: Option<grouped::GroupedScratch>,
    /// The share of a wide pass's experts the host takes, moved toward
    /// balance block by block; see `grouped.rs`.
    host_share: f32,
    /// Misses to copy into slots once the host has time: a block, the
    /// expert and the tick it was routed at.
    refills: Vec<(usize, usize, u64)>,
    /// The last block's misses, still running on the host.
    started: Option<StartedRow>,
}

#[derive(Default, Clone, Copy)]
pub(super) struct ExpertStats {
    /// Lookups by a one-row pass, a decode step, per expert a token routes to.
    pub(super) hits: u64,
    pub(super) misses: u64,
    /// Lookups by a wider pass, once per expert a group of rows routes to.
    pub(super) prompt_hits: u64,
    pub(super) prompt_misses: u64,
    pub(super) bytes: u64,
    pub(super) prefetches: u64,
    pub(super) prefetch_hits: u64,
    pub(super) cpu_misses: u64,
    pub(super) cpu_nanos: u64,
}

/// A pinned host buffer and its device twin, for tables many programs
/// read: what the host writes goes up asynchronously in stream order.
struct Staged<T: DeviceCopy> {
    host: LockedBuffer<T>,
    dev: DeviceBuffer<T>,
}

impl<T: DeviceCopy + Default> Staged<T> {
    fn new(len: usize) -> Result<Staged<T>> {
        Ok(Staged { host: LockedBuffer::new(&T::default(), len)?, dev: DeviceBuffer::from_slice(&vec![T::default(); len])? })
    }

    fn dev(&self) -> u64 {
        self.dev.as_device_ptr().as_raw()
    }

    /// Copies `values` in at `at` and returns their device address.
    fn push(&mut self, at: usize, values: &[T], stream: &Stream) -> Result<u64> {
        self.host.as_mut_slice()[at..at + values.len()].copy_from_slice(values);
        let dst = self.dev() + (at * size_of::<T>()) as u64;
        // SAFETY: the region is not written again before the next sync
        // point.
        unsafe { copy_async(dst, self.host.as_slice()[at..].as_ptr().cast(), size_of_val(values), stream)? };
        Ok(dst)
    }
}

struct BlockExperts {
    set: Arc<ExpertSet>,
    mirror: Mirror,
    /// `expert -> slot within the block`, `NONE` where not resident.
    slot_of: Vec<u32>,
    /// Per slot: the expert held and the tick it was last used at.
    held: Vec<(u32, u64)>,
    /// Whether a slot's expert arrived by prefetch and is unread since.
    prefetched: Vec<bool>,
    /// The block's first slot in each of its three slabs.
    base: [usize; 3],
    /// `SLOT[rows, 8]`, `TOPK[rows, 8]` and `W[rows, 8]`, pointer-stable so
    /// a segment's graph needs no patching.
    slots: Mapped<i32>,
    topk: Mapped<i32>,
    weights: Mapped<f32>,
    /// The previous block's prediction for this one, and the event its
    /// copies complete at. The prediction's weights are the kernel's
    /// mandatory output and nothing reads them.
    look: Mapped<i32>,
    look_w: DeviceBuffer<f32>,
    fetched: Option<Event>,
    /// A decode row's misses on their way to the host and back.
    handoff: Handoff,
}

struct Slab {
    bytes: DeviceBuffer<u8>,
    d: DeviceBuffer<u16>,
    n: usize,
    nb: usize,
    rb: usize,
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
    fn at(&self, s: usize) -> (u64, u64) {
        (
            self.bytes.as_device_ptr().as_raw() + (s * self.n * self.rb) as u64,
            self.d.as_device_ptr().as_raw() + (s * self.n * self.nb * 2) as u64,
        )
    }

    fn slot_bytes(&self) -> usize {
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
    fn chosen(&self, r: usize) -> Result<[usize; MOE_USED]> {
        let n = self.set.count() as i32;
        let mut out = [0; MOE_USED];
        for (o, &id) in out.iter_mut().zip(&self.topk.host()[r * MOE_USED..(r + 1) * MOE_USED]) {
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
}

impl Experts {
    pub(super) fn new() -> Experts {
        Experts {
            per_block: 0,
            blocks: Vec::new(),
            slabs: None,
            budget: None,
            limit: None,
            grouped: None,
            host_share: grouped::HOST_SHARE_START,
            iota: HashMap::new(),
            tick: 0,
            stats: ExpertStats::default(),
            refills: Vec::new(),
            started: None,
        }
    }

    fn slab(&self, block: usize, stack: Stack) -> &Slab {
        let quant = self.blocks[block].set.stack(stack).quant();
        &self.slabs.as_ref().expect("laid out")[&(stack, quant)]
    }

    /// The slab operands of `block`'s `stack` for a slot kernel: its rows
    /// and their scale plane from the block's base, over the block's share
    /// without its spare slot, since the kernels' grouped addressing reads
    /// the extent.
    fn slab_operands(&self, block: usize, stack: Stack) -> [(u64, [i64; 2]); 2] {
        let slab = self.slab(block, stack);
        let ns = (self.per_block * slab.n) as i64;
        let (bytes, d) = slab.at(self.blocks[block].base[stack as usize]);
        [(bytes, [ns, slab.rb as i64]), (d, [ns, slab.nb as i64])]
    }

    /// A new routing step: `ids` of `block` stamped, the tick returned.
    fn step(&mut self, block: usize, ids: &[usize]) -> u64 {
        self.tick += 1;
        self.blocks[block].stamp(ids, self.tick);
        self.tick
    }

    fn resident(&self, block: usize, e: usize) -> bool {
        self.blocks[block].slot_of[e] != NONE
    }

    /// Expert `e` of `block` in a slot: the one it is in, or the least
    /// recently used one, filled from the mirror on `stream`. `Ok(None)`
    /// when every slot is stamped `tick`. `prompt` says which counters the
    /// lookup goes to.
    fn place(&mut self, block: usize, e: usize, tick: u64, stream: &Stream, prompt: bool) -> Result<Option<usize>> {
        let b = &mut self.blocks[block];
        let s = b.slot_of[e];
        let stats = &mut self.stats;
        let (hits, misses) = if prompt { (&mut stats.prompt_hits, &mut stats.prompt_misses) } else { (&mut stats.hits, &mut stats.misses) };
        if s != NONE {
            *hits += 1;
            if std::mem::take(&mut b.prefetched[s as usize]) {
                stats.prefetch_hits += 1;
            }
            return Ok(Some(s as usize));
        }
        *misses += 1;
        let Some(victim) = b.victim(tick) else {
            return Ok(None);
        };
        self.fill(block, e, victim, tick, stream)?;
        Ok(Some(victim))
    }

    /// Expert `e` copied into `victim`'s slot of `block` on `stream`.
    fn fill(&mut self, block: usize, e: usize, victim: usize, tick: u64, stream: &Stream) -> Result<()> {
        let slabs = self.slabs.as_ref().expect("laid out");
        let b = &mut self.blocks[block];
        let (old, _) = b.held[victim];
        if old != NONE {
            b.slot_of[old as usize] = NONE;
        }
        let src = b.mirror.expert(e);
        for (k, stack) in Stack::ALL.into_iter().enumerate() {
            let slab = &slabs[&(stack, b.set.stack(stack).quant())];
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

    /// Experts `ids` of `block` that are not resident copied into least
    /// recently used slots on `stream`, the second one, and the event the
    /// block's kernels wait on before they read them. A slot stamped
    /// `tick` is never a victim.
    fn fetch(&mut self, block: usize, ids: impl IntoIterator<Item = usize>, tick: u64, stream: &Stream) -> Result<()> {
        let mut copied = false;
        for e in ids {
            if self.resident(block, e) {
                continue;
            }
            let Some(victim) = self.blocks[block].victim(tick) else {
                break;
            };
            self.fill(block, e, victim, tick, stream)?;
            self.blocks[block].prefetched[victim] = true;
            self.stats.prefetches += 1;
            copied = true;
        }
        if copied {
            let event = Event::new(EventFlags::DISABLE_TIMING)?;
            event.record(stream)?;
            self.blocks[block].fetched = Some(event);
        }
        Ok(())
    }

    /// The refills queued since the last call copied into slots on
    /// `stream`. A decode step queues one a block, its most heavily
    /// weighted miss: the host computes a miss in a quarter of a copy, so
    /// copying them all would outrun the bus and have a later token wait on
    /// the copies, and a fixed number keeps the cache, and so what the host
    /// computes, the same from run to run for a seed. Issuing a copy costs
    /// the host several microseconds, so this runs while the device is busy.
    fn refill(&mut self, stream: &Stream) -> Result<()> {
        for (block, expert, tick) in std::mem::take(&mut self.refills) {
            self.fetch(block, [expert], tick, stream)?;
        }
        Ok(())
    }
}

impl Experts {
    /// Waits for the misses started on the host, if any, and counts their
    /// time.
    fn join_started(&mut self) -> Result<()> {
        if let Some(started) = self.started.take() {
            self.stats.cpu_nanos += started.join()?;
        }
        Ok(())
    }
}

impl Drop for Experts {
    /// Misses started on the host read a block's mirror, which goes with
    /// this.
    fn drop(&mut self) {
        let _ = self.join_started();
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
        experts.iota.entry(n_expert).or_insert_with(|| {
            DeviceBuffer::from_slice(&(0..n_expert).map(|i| i as f32).collect::<Vec<_>>()).expect("a few hundred floats")
        });
        let (table, d) = (MAX_ROWS * MOE_USED, set.gate.k());
        experts.blocks.push(BlockExperts {
            set: Arc::clone(set),
            mirror,
            slot_of: vec![NONE; n_expert],
            held: Vec::new(),
            prefetched: Vec::new(),
            base: [0; 3],
            slots: Mapped::new(table)?,
            topk: Mapped::new(table)?,
            weights: Mapped::new(table)?,
            look: Mapped::new(MOE_USED)?,
            look_w: DeviceBuffer::from_slice(&[0f32; MOE_USED])?,
            fetched: None,
            handoff: Handoff::new(d)?,
        });
        let buf = ExpertsBuf(experts.blocks.len() - 1);
        self.expert_keys.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    /// The slabs, laid out at the first `moe`: the budget, less a prompt
    /// pass's scratch, shared equally over the blocks, each with at least
    /// a token's worth plus a zeroed spare, one slab per stack and format.
    fn ensure_slabs(&self, experts: &mut Experts) -> Result<()> {
        if experts.slabs.is_some() {
            return Ok(());
        }
        let (budget, sets) = experts.budget.context("the expert cache has no budget: budget_streamed was not called")?;
        let blocks = experts.blocks.len();
        ensure!(blocks == sets, "{blocks} expert sets registered of the {sets} the model has");
        let slot_bytes = experts
            .blocks
            .iter()
            .map(|b| Stack::ALL.iter().map(|&s| b.set.stack(s).grouped_bytes()).sum::<usize>())
            .max()
            .context("no expert set is registered")?;
        let mut counts: HashMap<(Stack, Quant), usize> = HashMap::new();
        for b in &experts.blocks {
            for stack in Stack::ALL {
                *counts.entry((stack, b.set.stack(stack).quant())).or_default() += 1;
            }
        }
        // The scratch comes out of the budget, or the card pages once the
        // pass allocates it; the widest group it can want is bound by the
        // slots the whole budget would buy.
        let set = &experts.blocks[0].set;
        let held = grouped::scratch_bytes(MAX_ROWS, set.gate.k(), set.gate.n(), budget / slot_bytes / blocks);
        let budget = budget.saturating_sub(held);
        phobos_base::log::emit(
            phobos_base::log::Level::Info,
            format_args!("expert cache: {} MiB held back for a prompt pass's scratch", held >> 20),
        );
        // The spare slot a block gets comes out of the budget too.
        let mut per_block = (budget / slot_bytes / blocks).saturating_sub(1);
        for (&(stack, quant), &count) in &counts {
            let found = experts.blocks.iter().map(|b| b.set.stack(stack)).find(|s| s.quant() == quant).expect("counted");
            per_block = per_block.min(SLAB_LIMIT / found.grouped_bytes() / count - 1);
        }
        ensure!(
            per_block >= MOE_USED,
            "the expert cache budget of {} MiB holds {per_block} experts a block; a token needs {MOE_USED}",
            budget >> 20
        );

        let stride = per_block + 1;
        let mut slabs = HashMap::new();
        let mut next: HashMap<(Stack, Quant), usize> = HashMap::new();
        for b in &mut experts.blocks {
            for (i, stack) in Stack::ALL.into_iter().enumerate() {
                let key = (stack, b.set.stack(stack).quant());
                if let std::collections::hash_map::Entry::Vacant(v) = slabs.entry(key) {
                    v.insert(Slab::new(stride * counts[&key], b.set.stack(stack))?);
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
