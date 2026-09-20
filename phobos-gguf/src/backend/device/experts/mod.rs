// Expert streaming: a block's experts live in a pinned host mirror, a
// share of them in device cache slots, and every miss crosses the bus once,
// at the block's sync point, into the slot it will be read from.
//
// A slab per stack kind and format (a file mixes formats per block, and a
// slab has one row stride), each `[slots * n, rb]` in the grouped layout the
// decode matvec reads, with an `f16` scale plane apiece that the K-quant
// kernels take and never read. A block owns `per_block` consecutive slots
// of each of its three slabs, from a base of its own; a slot holds one
// expert of one block across the three, and the block's map says which.
// Eviction is least recently used within the block, stamped by routing step.

mod grouped;
mod mirror;
mod op;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use cust::event::Event;
use cust::memory::{DeviceBuffer, LockedBuffer};

use super::DeviceBackend;
use crate::backend::ExpertsBuf;
use crate::experts::{ExpertSet, ExpertStack};
use crate::quant::Quant;
use mirror::Mirror;

/// Slots a block must have: a token's experts have to fit at once.
const MIN_SLOTS_PER_BLOCK: usize = super::kernels::MOE_USED;

/// Bytes a slab may not exceed: the decode matvec indexes in 32 bits.
const SLAB_LIMIT: usize = 1 << 31;

/// Rows one `moe` call may carry: twice the runtime's prompt batch, since
/// the checks feed longer passes than the runtime does.
pub(super) const MAX_ROWS: usize = 1024;

/// Which of a block's three stacks.
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

/// What the device holds of every block's experts.
pub(super) struct Experts {
    /// Slots a block owns of each of its slabs.
    pub(super) per_block: usize,
    pub(super) blocks: Vec<BlockExperts>,
    /// By stack kind and format; `None` until the first `moe`.
    pub(super) slabs: Option<HashMap<(Kind, Quant), Slab>>,
    /// The budget `budget_streamed` set, in bytes, and the expert sets it
    /// said would register, or `None` before it did.
    budget: Option<(usize, usize)>,
    /// The iota row the top-k carries, `[0, n_expert)` as f32.
    pub(super) iota: HashMap<usize, DeviceBuffer<f32>>,
    /// Routing steps so far, the stamp a slot takes when it is used: one a
    /// row a block, so the rows of a prompt pass do not all stamp alike and
    /// leave a block with no victim.
    pub(super) tick: u64,
    pub(super) stats: ExpertStats,
}

#[derive(Default, Clone, Copy)]
pub(super) struct ExpertStats {
    pub(super) hits: u64,
    pub(super) misses: u64,
    pub(super) bytes: u64,
    /// Host memory the mirrors pin, for the report.
    pub(super) pinned_bytes: u64,
    /// Experts the lookahead copied early, and how many of those a block
    /// then wanted.
    pub(super) prefetches: u64,
    pub(super) prefetch_hits: u64,
    /// Misses computed on the host, and the host time they took.
    pub(super) cpu_misses: u64,
    pub(super) cpu_nanos: u64,
}

/// One block's experts: where they are on the host, and which slots hold
/// which of them.
pub(super) struct BlockExperts {
    pub(super) set: Arc<ExpertSet>,
    pub(super) mirror: Mirror,
    /// `expert -> slot within the block`, `NONE` where not resident.
    pub(super) slot_of: Vec<u32>,
    /// Per slot of the block: the expert held and the tick it was last used
    /// at; `NONE` while empty.
    pub(super) held: Vec<(u32, u64)>,
    /// The block's first slot in each of its three slabs.
    pub(super) base: [usize; 3],
    /// The block's `SLOT[rows, 8]` table, `TOPK[rows, 8]` and `W[rows, 8]`,
    /// the kernels' operands, pointer-stable so a segment's graph needs no
    /// patching, with page-locked host mirrors of the first two so the
    /// table goes up and the choice comes back without a bounce.
    pub(super) slot_table: DeviceBuffer<i32>,
    pub(super) slot_host: LockedBuffer<i32>,
    pub(super) topk: DeviceBuffer<i32>,
    pub(super) topk_host: LockedBuffer<i32>,
    pub(super) weights: DeviceBuffer<f32>,
    /// What the previous block's lookahead predicted for this one, the
    /// logits it selected from, and the event its copies on the copy
    /// stream complete at; `None` when nothing is on its way.
    pub(super) look_logits: DeviceBuffer<f32>,
    pub(super) look_topk: DeviceBuffer<i32>,
    pub(super) look_host: LockedBuffer<i32>,
    pub(super) look_w: DeviceBuffer<f32>,
    pub(super) fetched: Option<Event>,
    /// Per slot, whether what it holds arrived by prefetch and has not
    /// been read since, for the report.
    pub(super) prefetched: Vec<bool>,
}

pub(super) const NONE: u32 = u32::MAX;

pub(super) struct Slab {
    pub(super) bytes: DeviceBuffer<u8>,
    pub(super) d: DeviceBuffer<u16>,
    /// Rows an expert takes, blocks a row, bytes a row.
    pub(super) n: usize,
    pub(super) nb: usize,
    pub(super) rb: usize,
}

impl Slab {
    fn new(slots: usize, stack: &ExpertStack) -> Result<Slab> {
        let (n, nb) = (stack.n(), stack.blocks_per_row());
        let rb = nb * stack.quant().device_block().1;
        let bytes = slots * n * rb;
        ensure!(
            bytes < SLAB_LIMIT,
            "a slab of {slots} slots would be {bytes} bytes, past the 32-bit index"
        );
        // SAFETY: every slot is written before any kernel reads it. The
        // scale plane is never read by the K-quant kernels, zeroed anyway.
        let bytes = unsafe { DeviceBuffer::<u8>::uninitialized(bytes)? };
        let d = DeviceBuffer::from_slice(&vec![0u16; slots * n * nb])?;
        Ok(Slab { bytes, d, n, nb, rb })
    }

    /// Device address of slot `s`'s rows, and of their scale plane.
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

impl Experts {
    pub(super) fn new() -> Experts {
        Experts {
            per_block: 0,
            blocks: Vec::new(),
            slabs: None,
            budget: None,
            iota: HashMap::new(),
            tick: 0,
            stats: ExpertStats::default(),
        }
    }

    /// The slab block `b` reads for `kind`.
    pub(super) fn slab(&self, b: usize, kind: Kind) -> &Slab {
        let quant = kind.stack(&self.blocks[b].set).quant();
        &self.slabs.as_ref().expect("slabs exist once a pass has begun")[&(kind, quant)]
    }
}

impl DeviceBackend {
    /// [`crate::backend::Backend::budget_streamed`]: what is left of the
    /// card once the resident weights and the pass's reserve are taken is
    /// the expert cache.
    pub(super) fn set_streamed_budget(&self, resident_bytes: usize, sets: usize) -> Result<()> {
        if self.experts.borrow().budget.is_some() {
            return Ok(());
        }
        let (free, _) = cust::memory::mem_get_info()?;
        let reserve = crate::runtime::RESERVE_BYTES;
        let budget = free.saturating_sub(resident_bytes).saturating_sub(reserve);
        ensure!(
            budget > 0 && sets > 0,
            "no device memory is left for an expert cache after {resident_bytes} bytes of resident weights"
        );
        self.experts.borrow_mut().budget = Some((budget, sets));
        Ok(())
    }

    /// [`crate::backend::Backend::constant_experts`]: the block's mirror,
    /// built now. The slabs wait for the first `moe`, by which time the
    /// model has registered every block.
    pub(super) fn register_experts(&self, key: &str, set: &Arc<ExpertSet>) -> Result<ExpertsBuf> {
        if let Some(&buf) = self.expert_keys.borrow().get(key) {
            return Ok(buf);
        }
        let mut experts = self.experts.borrow_mut();
        ensure!(
            experts.slabs.is_none(),
            "an expert set registered after the cache was laid out: every block registers before the first pass"
        );
        let n_expert = set.count();
        let mirror = Mirror::build(set).with_context(|| format!("mirroring the experts of {key}"))?;
        experts.stats.pinned_bytes += mirror.bytes() as u64;
        let ints = |len: usize| DeviceBuffer::from_slice(&vec![0i32; len]);
        let table = MAX_ROWS * super::kernels::MOE_USED;
        experts.iota.entry(n_expert).or_insert_with(|| {
            DeviceBuffer::from_slice(&(0..n_expert).map(|i| i as f32).collect::<Vec<_>>())
                .expect("a few hundred floats")
        });
        experts.blocks.push(BlockExperts {
            set: Arc::clone(set),
            mirror,
            slot_of: vec![NONE; n_expert],
            held: Vec::new(),
            base: [0; 3],
            slot_table: ints(table)?,
            slot_host: LockedBuffer::new(&0i32, table)?,
            topk: ints(table)?,
            topk_host: LockedBuffer::new(&0i32, table)?,
            weights: DeviceBuffer::from_slice(&vec![0f32; table])?,
            look_logits: DeviceBuffer::from_slice(&vec![0f32; n_expert])?,
            look_topk: ints(super::kernels::MOE_USED)?,
            look_host: LockedBuffer::new(&0i32, super::kernels::MOE_USED)?,
            look_w: DeviceBuffer::from_slice(&[0f32; super::kernels::MOE_USED])?,
            fetched: None,
            prefetched: Vec::new(),
        });
        let buf = ExpertsBuf(experts.blocks.len() - 1);
        self.expert_keys.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    /// The slabs, laid out at the first `moe` over every registered block:
    /// the budget shared equally, each block with at least a token's worth,
    /// one slab per stack kind and format holding its blocks' slots back to
    /// back.
    pub(super) fn ensure_slabs(&self, experts: &mut Experts) -> Result<()> {
        if experts.slabs.is_some() {
            return Ok(());
        }
        let (budget, sets) = experts
            .budget
            .context("the expert cache has no budget: budget_streamed was not called")?;
        let blocks = experts.blocks.len();
        ensure!(
            blocks == sets,
            "{blocks} expert sets registered of the {sets} the model has"
        );
        // A slot is sized for the widest block's expert, so every block's
        // share is the same count.
        let slot_bytes = experts
            .blocks
            .iter()
            .map(|b| KINDS.iter().map(|k| k.stack(&b.set).grouped_bytes()).sum::<usize>())
            .max()
            .unwrap_or(0);
        ensure!(slot_bytes > 0, "no expert set is registered");
        let mut per_block = budget / slot_bytes / blocks;
        // Each slab under the 32-bit index: the widest kind at its block count.
        let mut counts: HashMap<(Kind, Quant), usize> = HashMap::new();
        for b in &experts.blocks {
            for kind in KINDS {
                *counts.entry((kind, kind.stack(&b.set).quant())).or_default() += 1;
            }
        }
        for (&(kind, quant), &count) in &counts {
            let stack = experts
                .blocks
                .iter()
                .map(|b| kind.stack(&b.set))
                .find(|s| s.quant() == quant)
                .expect("counted");
            per_block = per_block.min(SLAB_LIMIT / stack.grouped_bytes() / count);
        }
        ensure!(
            per_block >= MIN_SLOTS_PER_BLOCK,
            "the expert cache budget of {} MiB holds {per_block} experts a block; a token needs {MIN_SLOTS_PER_BLOCK}",
            budget >> 20
        );

        // A block's share plus one zero slot, for a miss served elsewhere.
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
                let (zero_at, _) = slab.at(b.base[i] + per_block);
                // SAFETY: the zero slot is inside the slab just made.
                phobos_kernels::cuda_ok(
                    unsafe { cust::sys::cuMemsetD8_v2(zero_at, 0, slab.slot_bytes()) },
                    "zeroing a block's spare slot",
                )?;
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
