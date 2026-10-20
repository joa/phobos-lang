// The experts' host cache: pinned slots in the slabs' grouped layout, one
// expert each, as many as the host budget buys. When every expert fits, it
// is filled at load and never evicts. Otherwise a slot is filled on first
// use from the file mapping and the least recently used idle slot is
// reused; an expert outside the cache is read from the mapping, which the
// system pages like any file.
//
// A device fill copies from a slot, one `cuMemcpyHtoDAsync` per region. A
// slot is busy while such a copy has not run, while a host job reads it,
// and while the worker writes it; eviction skips a busy slot.
//
// A slot is written on the recording thread, several at once on the host
// pool when a pass names its fills ahead (`plan`), or on the worker for a
// decode step's refill, which is for later tokens and is copied at a later
// sync point.

use std::collections::{BTreeSet, VecDeque};
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use anyhow::{Context, Result, ensure};
use cust::event::{Event, EventFlags, EventStatus};
use cust::stream::Stream;
use phobos_kernels::cuda_ok;
use rayon::prelude::*;

use crate::experts::{ExpertSet, Stack};
use crate::simd::{Shape, Source, Weight};

/// An expert's six regions in a slot: host address and bytes.
pub(super) type Regions = [(*const c_void, usize); 6];

/// Alignment of a slot, which keeps each scale plane u16-aligned.
const SLOT_ALIGN: usize = 256;

/// Bytes of one pinned allocation: a few hundred MiB each rather than one
/// request for the whole budget.
const CHUNK_BYTES: usize = 512 << 20;

/// Planned fills written at once, so one batch is written while the copies
/// of the one before it run.
pub(super) const AHEAD: usize = 16;

const FREE: u32 = u32::MAX;

/// Bytes one expert of `set` takes in each of the six regions: the gate, up
/// and down blocks, then the gate, up and down scale planes.
fn per_expert(set: &ExpertSet) -> [usize; 6] {
    let mut per = [0usize; 6];
    for (i, stack) in Stack::ALL.into_iter().enumerate() {
        per[i] = set.stack(stack).grouped_bytes();
        per[3 + i] = set.stack(stack).grouped_scales() * 2;
    }
    per
}

/// Expert `set`'s regions in the slot at `base`. The three block regions
/// are of even length from an aligned start, so each scale plane is
/// u16-aligned.
fn regions_at(base: usize, set: &ExpertSet) -> Regions {
    let per = per_expert(set);
    let mut at = base;
    std::array::from_fn(|i| {
        let region = (at as *const c_void, per[i]);
        at += per[i];
        region
    })
}

/// Writes expert `e` of `set` in the grouped layout into the slot at `base`.
///
/// # Safety
/// `base` is a slot no copy or host job reads and nothing else writes.
unsafe fn write_expert(base: usize, set: &ExpertSet, e: usize) {
    for (i, (ptr, len)) in regions_at(base, set).into_iter().enumerate() {
        // SAFETY: the caller's contract.
        let out = unsafe { std::slice::from_raw_parts_mut(ptr.cast_mut().cast::<u8>(), len) };
        let stack = set.stack(Stack::ALL[i % 3]);
        if i < 3 {
            stack.grouped_into(e, out);
        } else {
            // SAFETY: the plane is u16-aligned (see `regions_at`) and whole
            // halves.
            let halves = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u16>(), out.len() / 2) };
            stack.grouped_scales_into(e, halves);
        }
    }
}

struct Slot {
    /// The block and expert held, `None` while empty.
    owner: Option<(usize, usize)>,
    /// When it was last used, its key in the LRU order.
    used: u64,
    /// When the device copies reading it have run.
    done: Option<Event>,
    /// Host jobs reading it.
    holds: u32,
    /// Out with the worker.
    writing: bool,
}

/// A slot the worker has written, and for a refill the routing step it was
/// queued at.
pub(super) struct Written {
    pub(super) block: usize,
    pub(super) expert: usize,
    pub(super) slot: usize,
    pub(super) refill: Option<u64>,
}

struct Job {
    set: Arc<ExpertSet>,
    base: usize,
    written: Written,
}

/// What [`HostCache::submit`] did with an expert.
pub(super) enum Submit {
    /// Already held, in this slot.
    Held(usize),
    /// With the worker.
    Queued,
    /// Every idle slot is in use; nothing was done.
    Busy,
}

pub(super) struct HostCache {
    chunks: Vec<*mut c_void>,
    per_chunk: usize,
    slot_bytes: usize,
    sets: Vec<Arc<ExpertSet>>,
    slots: Vec<Slot>,
    /// `[block][expert] -> slot`, `FREE` where not held.
    slot_of: Vec<Vec<u32>>,
    /// `(used, slot)`, least recently used first.
    lru: BTreeSet<(u64, usize)>,
    clock: u64,
    /// Every expert is held, so nothing is evicted or written after load.
    complete: bool,
    /// A pass's next fills that are not held, in order.
    planned: VecDeque<(usize, usize)>,
    jobs: Option<Sender<Job>>,
    results: Receiver<Written>,
    worker: Option<JoinHandle<()>>,
    /// Jobs out with the worker.
    out: usize,
    /// Slots the worker wrote, received while waiting for another.
    received: Vec<Written>,
}

// SAFETY: the pinned chunks are owned here and only touched through
// `&mut self`, or by the worker for a slot it was handed.
unsafe impl Send for HostCache {}

impl HostCache {
    /// Bytes a slot takes: the largest expert of `sets`.
    pub(super) fn slot_bytes(sets: &[Arc<ExpertSet>]) -> usize {
        sets.iter().map(|s| per_expert(s).iter().sum::<usize>()).max().unwrap_or(0).next_multiple_of(SLOT_ALIGN)
    }

    /// A cache of `slots` slots for `sets`, one per block, filled at once
    /// when it holds every expert.
    pub(super) fn new(sets: Vec<Arc<ExpertSet>>, slots: usize) -> Result<HostCache> {
        let slot_bytes = Self::slot_bytes(&sets);
        let total: usize = sets.iter().map(|s| s.count()).sum();
        let slots = slots.min(total);
        let (jobs, inbox) = channel::<Job>();
        let (done, results) = channel();
        let worker = std::thread::Builder::new()
            .name("phobos-experts".into())
            .spawn(move || {
                for job in inbox {
                    // SAFETY: the slot is this job's until it is sent back.
                    unsafe { write_expert(job.base, &job.set, job.written.expert) };
                    if done.send(job.written).is_err() {
                        return;
                    }
                }
            })
            .context("spawning the expert host cache's worker")?;
        let mut cache = HostCache {
            chunks: Vec::new(),
            per_chunk: (CHUNK_BYTES / slot_bytes.max(1)).max(1),
            slot_bytes,
            slot_of: sets.iter().map(|s| vec![FREE; s.count()]).collect(),
            sets,
            slots: (0..slots).map(|_| Slot { owner: None, used: 0, done: None, holds: 0, writing: false }).collect(),
            lru: (0..slots).map(|s| (0, s)).collect(),
            clock: 0,
            complete: false,
            planned: VecDeque::new(),
            jobs: Some(jobs),
            results,
            worker: Some(worker),
            out: 0,
            received: Vec::new(),
        };
        for c in 0..slots.div_ceil(cache.per_chunk) {
            let n = cache.per_chunk.min(slots - c * cache.per_chunk);
            let mut host: *mut c_void = std::ptr::null_mut();
            // Pinned so copies from it are asynchronous. Not mapped, since
            // no kernel reads it in place.
            cuda_ok(unsafe { cust::sys::cuMemHostAlloc(&mut host, n * slot_bytes, 0) }, "pinning the expert host cache")?;
            cache.chunks.push(host);
        }
        if slots == total {
            cache.fill_all();
        }
        Ok(cache)
    }

    /// Writes every expert into its own slot, across the host pool.
    fn fill_all(&mut self) {
        let mut writes = Vec::with_capacity(self.slots.len());
        for (block, set) in self.sets.iter().enumerate() {
            for e in 0..set.count() {
                let s = writes.len();
                self.slots[s].owner = Some((block, e));
                self.slot_of[block][e] = s as u32;
                writes.push((self.base(s), block, e));
            }
        }
        let sets = &self.sets;
        crate::simd::pool().install(|| {
            writes.par_iter().for_each(|&(base, block, e)| {
                // SAFETY: each write has its own slot, which nothing reads yet.
                unsafe { write_expert(base, &sets[block], e) };
            });
        });
        self.complete = true;
    }

    pub(super) fn slots(&self) -> usize {
        self.slots.len()
    }

    pub(super) fn bytes(&self) -> usize {
        self.slots.len() * self.slot_bytes
    }

    pub(super) fn complete(&self) -> bool {
        self.complete
    }

    fn base(&self, s: usize) -> usize {
        self.chunks[s / self.per_chunk] as usize + s % self.per_chunk * self.slot_bytes
    }

    /// Slot `s`'s regions, holding an expert of `block`.
    pub(super) fn regions(&self, s: usize, block: usize) -> Regions {
        regions_at(self.base(s), &self.sets[block])
    }

    fn touch(&mut self, s: usize) {
        if self.complete {
            return;
        }
        let slot = &mut self.slots[s];
        self.lru.remove(&(slot.used, s));
        self.clock += 1;
        slot.used = self.clock;
        self.lru.insert((slot.used, s));
    }

    /// The slot holding `block`'s expert `e`, written, if any.
    pub(super) fn lookup(&mut self, block: usize, e: usize) -> Option<usize> {
        let s = self.slot_of[block][e];
        if s == FREE || self.slots[s as usize].writing {
            return None;
        }
        self.touch(s as usize);
        Some(s as usize)
    }

    /// Gives `block`'s expert `e` the least recently used idle slot, once
    /// that slot's last copies have run, waiting for them if every idle
    /// slot is still being copied from. The slot's contents are left to the
    /// caller to write. `None` when every slot is held or out with the
    /// worker.
    fn claim(&mut self, block: usize, e: usize) -> Result<Option<usize>> {
        let (mut found, mut pending) = (None, None);
        for &(_, s) in &self.lru {
            let slot = &self.slots[s];
            if slot.holds > 0 || slot.writing {
                continue;
            }
            match &slot.done {
                Some(event) if event.query()? != EventStatus::Ready => {
                    pending.get_or_insert(s);
                }
                _ => {
                    found = Some(s);
                    break;
                }
            }
        }
        let Some(s) = found.or(pending) else {
            return Ok(None);
        };
        if let Some(event) = self.slots[s].done.take() {
            event.synchronize()?;
        }
        if let Some((b, old)) = self.slots[s].owner.replace((block, e)) {
            self.slot_of[b][old] = FREE;
        }
        self.slot_of[block][e] = s as u32;
        self.touch(s);
        Ok(Some(s))
    }

    /// [`HostCache::claim`], waiting for the worker when every slot is
    /// busy.
    fn claim_waiting(&mut self, block: usize, e: usize) -> Result<usize> {
        loop {
            if let Some(s) = self.claim(block, e)? {
                return Ok(s);
            }
            ensure!(self.out > 0, "every slot of the expert host cache is in use");
            self.receive_one()?;
        }
    }

    fn receive_one(&mut self) -> Result<()> {
        let written = self.results.recv().context("the expert host cache's worker stopped")?;
        self.slots[written.slot].writing = false;
        self.out -= 1;
        self.received.push(written);
        Ok(())
    }

    /// The slot holding `block`'s expert `e`, written now if it is not
    /// held: with the next planned fills when it is the first of them,
    /// else on its own.
    pub(super) fn ready(&mut self, block: usize, e: usize) -> Result<usize> {
        if let Some(s) = self.lookup(block, e) {
            return Ok(s);
        }
        let s = self.slot_of[block][e];
        if s != FREE {
            while self.slots[s as usize].writing {
                self.receive_one()?;
            }
            self.touch(s as usize);
            return Ok(s as usize);
        }
        if self.planned.front() == Some(&(block, e)) {
            self.write_ahead()?;
            if let Some(s) = self.lookup(block, e) {
                return Ok(s);
            }
        }
        // Off the plan, which no longer says what comes next.
        self.planned.clear();
        let s = self.claim_waiting(block, e)?;
        // SAFETY: a claimed slot is idle and this thread's.
        unsafe { write_expert(self.base(s), &self.sets[block], e) };
        Ok(s)
    }

    /// Names the experts of `block` the next fills will be, in order, so
    /// the ones not held can be written ahead together. Touches the held
    /// ones, so writing the others does not evict them first.
    pub(super) fn plan(&mut self, block: usize, experts: impl IntoIterator<Item = usize>) {
        self.planned.clear();
        for e in experts {
            if self.lookup(block, e).is_none() && self.slot_of[block][e] == FREE {
                self.planned.push_back((block, e));
            }
        }
    }

    /// Writes the next [`AHEAD`] planned experts, across the host pool.
    fn write_ahead(&mut self) -> Result<()> {
        let count = self.planned.len().min(AHEAD);
        let mut writes = Vec::with_capacity(count);
        for (block, e) in self.planned.drain(..count).collect::<Vec<_>>() {
            let s = self.claim_waiting(block, e)?;
            writes.push((self.base(s), block, e));
        }
        let sets = &self.sets;
        crate::simd::pool().install(|| {
            writes.par_iter().for_each(|&(base, block, e)| {
                // SAFETY: each write has its own claimed slot.
                unsafe { write_expert(base, &sets[block], e) };
            });
        });
        Ok(())
    }

    /// Hands `block`'s expert `e` to the worker, unless it is held or on its
    /// way already. `refill` is the routing step of a refill, which comes
    /// back from [`HostCache::written`] to be copied.
    pub(super) fn submit(&mut self, block: usize, e: usize, refill: Option<u64>) -> Result<Submit> {
        if let Some(s) = self.lookup(block, e) {
            return Ok(Submit::Held(s));
        }
        if self.slot_of[block][e] != FREE {
            return Ok(Submit::Busy);
        }
        let Some(s) = self.claim(block, e)? else {
            return Ok(Submit::Busy);
        };
        self.slots[s].writing = true;
        let job = Job { set: Arc::clone(&self.sets[block]), base: self.base(s), written: Written { block, expert: e, slot: s, refill } };
        self.jobs.as_ref().expect("open until drop").send(job).map_err(|_| anyhow::anyhow!("the expert host cache's worker stopped"))?;
        self.out += 1;
        Ok(Submit::Queued)
    }

    /// The slots the worker has written since the last call.
    pub(super) fn written(&mut self) -> Vec<Written> {
        for written in self.results.try_iter() {
            self.slots[written.slot].writing = false;
            self.out -= 1;
            self.received.push(written);
        }
        std::mem::take(&mut self.received)
    }

    /// Waits for every job out with the worker and forgets the refills
    /// among them; what they wrote stays held.
    pub(super) fn settle(&mut self) -> Result<()> {
        while self.out > 0 {
            self.receive_one()?;
        }
        self.received.clear();
        Ok(())
    }

    /// Marks slot `s` as read by the copies queued on `stream` so far. A
    /// complete cache never rewrites a slot, so it need not know.
    pub(super) fn issued(&mut self, s: usize, stream: &Stream) -> Result<()> {
        if self.complete {
            return Ok(());
        }
        let event = Event::new(EventFlags::DISABLE_TIMING)?;
        event.record(stream)?;
        self.slots[s].done = Some(event);
        Ok(())
    }

    /// The slot holding `block`'s expert `e` for a host job, which keeps it
    /// until [`HostCache::release`]. `None` when it is not held.
    pub(super) fn hold(&mut self, block: usize, e: usize) -> Option<(usize, Regions)> {
        let s = self.lookup(block, e)?;
        self.slots[s].holds += 1;
        Some((s, self.regions(s, block)))
    }

    pub(super) fn release(&mut self, s: usize) {
        self.slots[s].holds -= 1;
    }
}

impl Drop for HostCache {
    fn drop(&mut self) {
        // The worker finishes the slot it holds before the chunks go, and a
        // copy still reading a slot runs first.
        self.jobs = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        for event in self.slots.iter().filter_map(|s| s.done.as_ref()) {
            let _ = event.synchronize();
        }
        for &host in &self.chunks {
            // SAFETY: allocated by cuMemHostAlloc in `new`, freed once.
            unsafe {
                cust::sys::cuMemFreeHost(host);
            }
        }
    }
}

/// Where the host kernels read a block's experts for one job: the slots it
/// holds, else the file mapping in place.
pub(super) struct HostSource<'a> {
    set: &'a ExpertSet,
    held: Vec<(usize, Regions)>,
}

// SAFETY: a held slot is not written while the job runs, and the mapping
// is read-only.
unsafe impl Sync for HostSource<'_> {}
unsafe impl Send for HostSource<'_> {}

impl<'a> HostSource<'a> {
    /// `held` pairs an expert with its slot's regions.
    pub(super) fn new(set: &'a ExpertSet, held: Vec<(usize, Regions)>) -> HostSource<'a> {
        HostSource { set, held }
    }
}

impl Source for HostSource<'_> {
    fn shape(&self, stack: Stack) -> Shape {
        self.set.shape(stack)
    }

    fn weight(&self, stack: Stack, e: usize) -> Weight<'_> {
        let Some((_, regions)) = self.held.iter().find(|(held, _)| *held == e) else {
            return self.set.weight(stack, e);
        };
        let i = stack as usize;
        let (blocks, plane) = (regions[i], regions[3 + i]);
        // SAFETY: both regions are inside a held slot, which stays written
        // while the job runs, and the plane is u16-aligned.
        let (bytes, scales) = unsafe {
            (
                std::slice::from_raw_parts(blocks.0.cast::<u8>(), blocks.1),
                std::slice::from_raw_parts(plane.0.cast::<u16>(), plane.1 / 2),
            )
        };
        let s = self.set.stack(stack);
        Weight::Grouped { bytes, block_bytes: s.unit_bytes(), nb: s.blocks_per_row(), scales }
    }
}
