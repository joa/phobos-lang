// Expert streaming. A model's experts live in the file mapping, as many of
// them as the host budget buys in a pinned host cache in the slabs' layout
// (`host_cache.rs`), all of them where they fit, and some of them in device
// cache slots.
//
// A decode step computes its misses on the host, from the host cache or
// the mapping (`handoff.rs`). A prompt pass copies its misses into slots at the
// block's sync point, or runs some of them on the host instead
// (`grouped.rs`).
//
// There is one slab per stack and format, since a file can mix formats per
// block and a slab has one row stride. Each slab is `[slots * n, rb]` in the
// grouped layout the decode matvec reads, plus an `f16` scale plane. A block
// owns `per_block` consecutive slots in each of its three slabs, plus one
// zeroed spare. A slot holds one expert across the three slabs. Eviction is
// least recently used within the block.

mod grouped;
mod handoff;
mod headroom;
mod host_cache;
mod mapped;
mod op;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use cust::event::{Event, EventFlags, EventStatus};
use cust::memory::{DeviceBuffer, DeviceCopy, LockedBuffer};
use cust::stream::Stream;
use phobos_kernels::cuda_ok;

use super::DeviceBackend;
use super::kernels::MOE_USED;
use super::mem::card_memory;
use crate::backend::ExpertsBuf;
use crate::experts::{ExpertSet, ExpertStack, Stack};
use crate::quant::Quant;
use crate::simd::StartedRow;
pub use grouped::moe_qgemm_src;
use handoff::Handoff;
use mapped::Mapped;
use host_cache::{AHEAD, HostCache, HostSource, Submit, Written};

/// Upper bound on one block's span of a slab. A kernel takes the slab from the
/// block's first slot on and indexes it in 32 bits; the slab itself may hold
/// more.
const SPAN_LIMIT: usize = 1 << 31;

/// Most rows one `moe` call may carry, and so the widest prompt pass the
/// runtime's batch tuner tries on a model with experts.
pub(crate) const MAX_ROWS: usize = 1024;

const NONE: u32 = u32::MAX;

/// Host memory pinned experts leave to everything else, at least: their
/// pages stay locked for the session, so the rest of the system cannot
/// reclaim them as it grows.
const HOST_RESERVE_BYTES: u64 = 4 << 30;

/// The share of the host's memory left to everything else, when larger.
const HOST_RESERVE_SHARE: u64 = 4;

/// The fewest experts the host cache holds, whatever the budget.
const HOST_SLOTS_MIN: usize = 256;

pub(super) struct Experts {
    per_block: usize,
    blocks: Vec<BlockExperts>,
    /// By stack and format; `None` until the first `moe`.
    slabs: Option<HashMap<(Stack, Quant), Slab>>,
    /// Cache bytes and expert set count, set by `set_streamed_budget`.
    budget: Option<(usize, usize)>,
    /// The user's cap on the cache, in bytes.
    pub(super) limit: Option<usize>,
    /// The user's cap on the host memory the experts are held in, in bytes.
    pub(super) host_limit: Option<usize>,
    /// The iota row the top-k carries, by expert count.
    iota: HashMap<usize, DeviceBuffer<f32>>,
    /// Routing step counter, stamped on a slot when it is used. It advances
    /// once per row per block, so a prompt pass's rows get distinct stamps.
    tick: u64,
    stats: ExpertStats,
    /// The grouped prompt path's tables, once it has run.
    grouped: Option<grouped::GroupedScratch>,
    /// The share of a wide pass's experts the host takes. It moves toward
    /// balance block by block; see `grouped.rs`.
    host_share: f32,
    /// Misses to copy into slots later, as (block, expert, routing tick).
    refills: Vec<(usize, usize, u64)>,
    /// The last block's misses, still running on the host, and the host
    /// cache slots they read.
    started: Option<StartedRow>,
    started_holds: Vec<usize>,
    /// The most slots a block may have while the card is short of memory.
    per_block_cap: Option<usize>,

    /// The experts' host cache, made at the first layout.
    host: Option<HostCache>,
    /// The most slots a block the budget buys, which a grown cache returns to.
    most_per_block: usize,
    /// Passes since free memory was last queried.
    since_checked: usize,
}

#[derive(Default, Clone, Copy)]
pub(super) struct ExpertStats {
    /// Decode lookups, one per expert a token routes to.
    pub(super) hits: u64,
    pub(super) misses: u64,
    /// Prompt pass lookups, one per expert a group of rows routes to.
    pub(super) prompt_hits: u64,
    pub(super) prompt_misses: u64,
    pub(super) bytes: u64,
    pub(super) prefetches: u64,
    pub(super) prefetch_hits: u64,
    pub(super) cpu_misses: u64,
    pub(super) cpu_nanos: u64,
    /// Host misses the host cache did not hold, read from the mapping.
    pub(super) cpu_mapped: u64,
}

/// A pinned host buffer and its device twin. Host writes are copied up
/// asynchronously in stream order.
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
    /// `expert -> slot within the block`, `NONE` where not resident.
    slot_of: Vec<u32>,
    /// Per slot: the expert held and the tick it was last used at.
    held: Vec<(u32, u64)>,
    /// Whether a slot's expert arrived by prefetch and is unread since.
    prefetched: Vec<bool>,
    /// The block's first slot in each of its three slabs.
    base: [usize; 3],
    /// `SLOT[rows, 8]`, `TOPK[rows, 8]` and `W[rows, 8]`. Their pointers
    /// never change, so a segment's graph needs no patching.
    slots: Mapped<i32>,
    topk: Mapped<i32>,
    weights: Mapped<f32>,
    /// The previous block's prediction for this one. The kernel must write
    /// `look_w`, but nothing reads it.
    look: Mapped<i32>,
    look_w: DeviceBuffer<f32>,
    /// The event the last prefetch's copies complete at, and the slots they
    /// fill. A chosen expert in one of those slots is a hit only once the
    /// event has passed.
    fetched: Option<Event>,
    arriving: Vec<usize>,
    /// Whether a prefetch was attempted for this block since its last row.
    predicted: bool,
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
        let rb = nb * stack.unit_bytes();
        let bytes = slots * n * rb;
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

    /// Device bytes the slab holds, rows and scales.
    fn held_bytes(&self) -> usize {
        self.bytes.len() + self.d.len() * size_of::<u16>()
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

    /// Whether a prefetch into slot `s` is still copying. Once the copies
    /// have completed, nothing is in flight until the next prefetch.
    fn in_flight(&mut self, s: usize) -> Result<bool> {
        let Some(event) = &self.fetched else {
            return Ok(false);
        };
        if event.query()? == EventStatus::Ready {
            self.fetched = None;
            self.arriving.clear();
            return Ok(false);
        }
        Ok(self.arriving.contains(&s))
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
            host_limit: None,
            grouped: None,
            host_share: grouped::HOST_SHARE_START,
            iota: HashMap::new(),
            tick: 0,
            stats: ExpertStats::default(),
            refills: Vec::new(),
            started: None,
            started_holds: Vec::new(),
            per_block_cap: None,
            host: None,
            most_per_block: 0,
            since_checked: 0,
        }
    }

    fn slab(&self, block: usize, stack: Stack) -> &Slab {
        let quant = self.blocks[block].set.stack(stack).quant();
        &self.slabs.as_ref().expect("laid out")[&(stack, quant)]
    }

    /// The slab operands of `block`'s `stack` for a slot kernel: rows and
    /// scale plane, starting at the block's base.
    ///
    /// The extent covers the block's slots without the spare, because the
    /// kernels' grouped addressing reads it.
    fn slab_operands(&self, block: usize, stack: Stack) -> [(u64, [i64; 2]); 2] {
        let slab = self.slab(block, stack);
        let ns = (self.per_block * slab.n) as i64;
        let (bytes, d) = slab.at(self.blocks[block].base[stack as usize]);
        [(bytes, [ns, slab.rb as i64]), (d, [ns, slab.nb as i64])]
    }

    /// Starts a new routing step, stamps `ids` of `block` and returns the
    /// tick.
    fn step(&mut self, block: usize, ids: &[usize]) -> u64 {
        self.tick += 1;
        self.blocks[block].stamp(ids, self.tick);
        self.tick
    }

    fn resident(&self, block: usize, e: usize) -> bool {
        self.blocks[block].slot_of[e] != NONE
    }

    /// Returns the slot holding expert `e` of `block`. On a miss, fills the
    /// least recently used slot from the mirror on `stream`.
    ///
    /// Returns `Ok(None)` when every slot is stamped `tick`. `prompt` picks
    /// which counters the lookup goes to.
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
        self.fill(block, e, victim, tick, stream, None)?;
        Ok(Some(victim))
    }

    /// Tells the host cache which of `ids` of `block` the next fills will
    /// be, the ones not resident, so it can write the ones it lacks ahead.
    /// Call once the ids are stamped, so no fill evicts another of them.
    fn plan_fills(&mut self, block: usize, ids: &[usize]) {
        let slot_of = &self.blocks[block].slot_of;
        if let Some(host) = &mut self.host {
            host.plan(block, ids.iter().copied().filter(|&e| slot_of[e] == NONE));
        }
    }

    /// Copies expert `e` into slot `victim` of `block` on `stream`, from
    /// host cache slot `from`, or from the one holding it, written now if
    /// none does.
    fn fill(&mut self, block: usize, e: usize, victim: usize, tick: u64, stream: &Stream, from: Option<usize>) -> Result<()> {
        let slabs = self.slabs.as_ref().expect("laid out");
        let host = self.host.as_mut().expect("made with the slabs");
        let from = match from {
            Some(s) => s,
            None => host.ready(block, e)?,
        };
        let src = host.regions(from, block);
        let b = &mut self.blocks[block];
        let (old, _) = b.held[victim];
        if old != NONE {
            b.slot_of[old as usize] = NONE;
        }
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
                // SAFETY: the host cache is pinned, and a slot is not
                // rewritten until its copies have run.
                unsafe { copy_async(dst, ptr, len, stream)? };
                self.stats.bytes += len as u64;
            }
        }
        host.issued(from, stream)?;
        b.held[victim] = (e as u32, tick);
        b.slot_of[e] = victim as u32;
        b.prefetched[victim] = false;
        Ok(())
    }

    /// Copies the first `most` non-resident experts in `ids` of `block`
    /// into least recently used slots on `stream`. Records the event the
    /// copies complete at and the slots they fill, which stay in flight
    /// until a row finds the event passed.
    ///
    /// A slot stamped `tick` is never evicted.
    fn fetch(&mut self, block: usize, ids: impl IntoIterator<Item = usize>, most: usize, tick: u64, stream: &Stream) -> Result<()> {
        let mut filled = Vec::new();
        for e in ids {
            if filled.len() == most {
                break;
            }
            if self.resident(block, e) {
                continue;
            }
            let Some(victim) = self.blocks[block].victim(tick) else {
                break;
            };
            self.fill(block, e, victim, tick, stream, None)?;
            self.blocks[block].prefetched[victim] = true;
            self.stats.prefetches += 1;
            filled.push(victim);
        }
        self.arriving(block, filled, stream)
    }

    /// Records the event the copies into `filled` of `block` complete at;
    /// see [`Experts::fetch`].
    fn arriving(&mut self, block: usize, filled: Vec<usize>, stream: &Stream) -> Result<()> {
        if !filled.is_empty() {
            // Copies on one stream complete in order, so the newest event
            // covers every slot still arriving.
            let event = Event::new(EventFlags::DISABLE_TIMING)?;
            event.record(stream)?;
            let b = &mut self.blocks[block];
            b.fetched = Some(event);
            b.arriving.extend(filled);
        }
        Ok(())
    }

    /// Copies the refills queued since the last call into slots on `stream`.
    ///
    /// A decode step queues one refill per block, its most heavily weighted
    /// miss. The host computes a miss faster than the bus copies it, so
    /// copying every miss would stall later tokens. A fixed count also
    /// keeps the cache contents deterministic for a seed.
    ///
    /// Issuing copies costs host time, so call this while the device is
    /// busy.
    ///
    /// A refill the host cache does not hold is written on its worker and
    /// copied at a sync point after it is done. The slots in use at a sync
    /// point are the last block's at most, whose refills are still with the
    /// worker, so a late copy never lands in a slot a kernel reads.
    fn refill(&mut self, stream: &Stream) -> Result<()> {
        let host = self.host.as_mut().expect("made with the slabs");
        for written in host.written() {
            if let Written { block, expert, slot, refill: Some(tick) } = written {
                self.refill_from(block, expert, tick, slot, stream)?;
            }
        }
        for (block, expert, tick) in std::mem::take(&mut self.refills) {
            if self.resident(block, expert) {
                continue;
            }
            match self.host.as_mut().expect("as above").submit(block, expert, Some(tick))? {
                Submit::Held(slot) => self.refill_from(block, expert, tick, slot, stream)?,
                Submit::Queued => {}
                Submit::Busy => self.fetch(block, [expert], 1, tick, stream)?,
            }
        }
        Ok(())
    }

    /// Copies a refill from host cache slot `slot` into a device slot on
    /// `stream`, as [`Experts::fetch`] would have, unless the expert arrived
    /// meanwhile or no device slot is free.
    fn refill_from(&mut self, block: usize, expert: usize, tick: u64, slot: usize, stream: &Stream) -> Result<()> {
        if self.resident(block, expert) {
            return Ok(());
        }
        let Some(victim) = self.blocks[block].victim(tick) else {
            return Ok(());
        };
        self.fill(block, expert, victim, tick, stream, Some(slot))?;
        self.blocks[block].prefetched[victim] = true;
        self.stats.prefetches += 1;
        self.arriving(block, vec![victim], stream)
    }
}

impl Experts {
    /// Waits for any misses running on the host, counts their time and
    /// lets go of the host cache slots they read.
    fn join_started(&mut self) -> Result<()> {
        if let Some(started) = self.started.take() {
            self.stats.cpu_nanos += started.join()?;
        }
        let holds = std::mem::take(&mut self.started_holds);
        self.release_holds(holds);
        Ok(())
    }

    /// Where the host kernels read `block`'s `experts`: the host cache slots
    /// holding them, kept until [`Experts::release_holds`], else the mapping.
    fn host_source(&mut self, block: usize, experts: impl IntoIterator<Item = usize>) -> (HostSource<'_>, Vec<usize>) {
        let host = self.host.as_mut().expect("made with the slabs");
        let (mut held, mut slots) = (Vec::new(), Vec::new());
        for e in experts {
            if let Some((s, regions)) = host.hold(block, e) {
                held.push((e, regions));
                slots.push(s);
            } else {
                self.stats.cpu_mapped += 1;
            }
        }
        (HostSource::new(&self.blocks[block].set, held), slots)
    }

    fn release_holds(&mut self, slots: Vec<usize>) {
        if let Some(host) = &mut self.host {
            for s in slots {
                host.release(s);
            }
        }
    }
}

impl Experts {
    /// Drops every resident expert. Counters keep running. The caller has
    /// synchronized the stream, so no copy is still filling a slot. The host
    /// cache keeps what it holds, which only decides how fast a fill is.
    pub(super) fn cool(&mut self) -> Result<()> {
        self.join_started()?;
        self.refills.clear();
        if let Some(host) = &mut self.host {
            host.settle()?;
        }
        for b in &mut self.blocks {
            b.slot_of.fill(NONE);
            b.held.fill((NONE, 0));
            b.prefetched.fill(false);
            b.fetched = None;
            b.arriving.clear();
            b.predicted = false;
        }
        Ok(())
    }
}

impl Drop for Experts {
    /// Host misses still read the host cache or the mapping, so join them
    /// before either is freed.
    fn drop(&mut self) {
        let _ = self.join_started();
    }
}

impl DeviceBackend {
    pub(super) fn set_streamed_budget(&self, resident_bytes: usize, sets: usize) -> Result<()> {
        if self.experts.borrow().budget.is_some() {
            return Ok(());
        }
        let (free, _) = card_memory()?;
        let room = free.saturating_sub(resident_bytes).saturating_sub(crate::runtime::RESERVE_BYTES);
        let budget = match self.experts.borrow().limit {
            Some(asked) if asked > room => {
                phobos_base::phinfo!(
                    "expert cache: {} MiB asked for, {} MiB left after the resident weights and the reserve; taking what is left",
                    asked >> 20,
                    room >> 20
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

    /// Registers a block's experts. The slabs and the host cache are laid out
    /// at the first `moe`, once every block has registered.
    pub(super) fn register_experts(&self, key: &str, set: &Arc<ExpertSet>) -> Result<ExpertsBuf> {
        if let Some(&buf) = self.expert_keys.borrow().get(key) {
            return Ok(buf);
        }
        let mut experts = self.experts.borrow_mut();
        ensure!(experts.slabs.is_none(), "an expert set registered after the cache was laid out");
        let n_expert = set.count();
        for stack in Stack::ALL {
            op::kernels(set.stack(stack).quant()).with_context(|| format!("streaming the experts of {key}"))?;
        }
        experts.iota.entry(n_expert).or_insert_with(|| {
            DeviceBuffer::from_slice(&(0..n_expert).map(|i| i as f32).collect::<Vec<_>>()).expect("a few hundred floats")
        });
        let (table, d) = (MAX_ROWS * MOE_USED, set.gate.k());
        experts.blocks.push(BlockExperts {
            set: Arc::clone(set),
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
            arriving: Vec::new(),
            predicted: false,
            handoff: Handoff::new(d)?,
        });
        let buf = ExpertsBuf(experts.blocks.len() - 1);
        self.expert_keys.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    /// Lays out the slabs at the first `moe`, one per stack and format.
    ///
    /// The budget, minus a prompt pass's scratch, is split equally over
    /// the blocks. Each block gets at least a token's worth of slots plus a
    /// zeroed spare.
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
        // Reserve the scratch from the budget, or the card pages when the
        // pass allocates it. The widest group is bounded by the slots the
        // whole budget would buy.
        let set = &experts.blocks[0].set;
        let held = grouped::scratch_bytes(MAX_ROWS, set.gate.k(), set.gate.n(), budget / slot_bytes / blocks);
        let budget = budget.saturating_sub(held);
        phobos_base::phinfo!("expert cache: {} MiB held back for a prompt pass's scratch", held >> 20);
        // Each block's spare slot comes out of the budget too. A block never
        // needs more slots than it has experts, which a card that holds the
        // whole model reaches.
        let experts_per_block = experts.blocks.iter().map(|b| b.set.count()).max().expect("a block registered");
        let mut most = (budget / slot_bytes / blocks).saturating_sub(1).min(experts_per_block);
        for &(stack, quant) in counts.keys() {
            let found = experts.blocks.iter().map(|b| b.set.stack(stack)).find(|s| s.quant() == quant).expect("counted");
            most = most.min(SPAN_LIMIT / found.grouped_bytes() - 1);
        }
        experts.most_per_block = most;
        let per_block = most.min(experts.per_block_cap.unwrap_or(usize::MAX));
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
        if experts.host.is_none() {
            experts.host = Some(host_cache(experts, most)?);
        }
        let total: usize = slabs.values().map(|s| s.bytes.len()).sum();
        experts.per_block = per_block;
        experts.slabs = Some(slabs);
        phobos_base::phinfo!(
            "expert cache: {} slots of up to {:.2} MiB, {per_block} a block, {:.0} MiB in {} slabs",
            per_block * blocks,
            slot_bytes as f64 / (1 << 20) as f64,
            total as f64 / (1 << 20) as f64,
            counts.len()
        );
        Ok(())
    }

    pub(super) fn expert_stats(&self) -> ExpertStats {
        self.experts.borrow().stats
    }
}

/// The host bytes the experts may be held in, and where that came from:
/// `limit` when the user set one, else what the host has available beside a
/// reserve, the larger of [`HOST_RESERVE_BYTES`] and a [`HOST_RESERVE_SHARE`]
/// of its memory.
fn host_budget(limit: Option<usize>) -> (usize, String) {
    let mib = |b: u64| b >> 20;
    match (limit, phobos_base::host::memory()) {
        (Some(limit), _) => (limit, "as asked".to_string()),
        (None, Some(memory)) => {
            let reserve = HOST_RESERVE_BYTES.max(memory.total_bytes / HOST_RESERVE_SHARE);
            let room = memory.available_bytes.saturating_sub(reserve);
            let why = format!("of {} MiB available, {} MiB held back", mib(memory.available_bytes), mib(reserve));
            (usize::try_from(room).unwrap_or(usize::MAX), why)
        }
        (None, None) => (usize::MAX, "with the host's memory unknown".to_string()),
    }
}

/// The host cache for `experts`' registered sets: as many slots as the
/// budget buys, at least enough for a group of `per_block` fills, a token's
/// refills and a row's host misses in flight together.
fn host_cache(experts: &Experts, per_block: usize) -> Result<HostCache> {
    let sets: Vec<Arc<ExpertSet>> = experts.blocks.iter().map(|b| Arc::clone(&b.set)).collect();
    let slot_bytes = HostCache::slot_bytes(&sets);
    let floor = HOST_SLOTS_MIN.max(2 * per_block + 2 * sets.len() + AHEAD + MOE_USED);
    let (budget, why) = host_budget(experts.host_limit);
    let slots = budget / slot_bytes;
    if let Some(limit) = experts.host_limit {
        ensure!(
            slots >= floor,
            "a host memory limit of {} MiB holds {slots} experts of this model; it takes at least {floor}, {} MiB",
            limit >> 20,
            (floor * slot_bytes) >> 20
        );
    }
    let total: usize = sets.iter().map(|s| s.count()).sum();
    let host = HostCache::new(sets, slots.max(floor))?;
    phobos_base::phinfo!(
        "expert host cache: {} of {total} experts, {} MiB pinned {why}{}",
        host.slots(),
        host.bytes() >> 20,
        if host.complete() { "" } else { "; the rest are read from the file" }
    );
    Ok(host)
}
