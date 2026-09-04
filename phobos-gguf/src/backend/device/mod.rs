use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;

use anyhow::{Context, Result, bail, ensure};
use cust::memory::{CopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

use phobos_base::half::f16_to_f32;
use phobos_kernels::pool::Pool;
use phobos_kernels::{
    Variants, compile, compile_parallel, compile_shared, cuda_ok, push_descriptor,
};

use super::fuse::{self, Bound, Chain, ChainKey, Plan, Scratch};
use phobos_kernels::launch::{CTA_THREADS, STATIC_SHARED_LIMIT, persistent_grid};

use crate::quant::Quant;

use super::{
    Attn, Backend, Buf, DeltaMix, Fused, FusedAttnOut, FusedMlp, FusedProject, HBuf, HPlane,
    Packed, Plane, Q8_BLOCK, QAct, QBuf, RawBuf, Rope,
};

mod arena;
mod argmax;
mod attn;
mod backend;
mod delta;
mod dense;
mod elem;
mod fused;
mod graph;
use graph::{PassGraph, PassOp, Recorded};
mod init;
mod kernels;
mod launch;
mod matmul;
mod mem;
mod qmma_raw;
mod raw;
mod residency;

use fused::*;

// The catalog is written as one flat namespace of sources and tile sizes, and
// the impls below reach for them unqualified; see kernels/mod.rs.
use kernels::*;

pub use kernels::{ATTN_GEMM_TILE, ATTN_SOFT_TILE, attn_gemm_src};

/// Whether an opt-in `PHOBOS_*` toggle is set. Opt-out toggles (a default-on
/// mechanism) negate the complementary spelling instead; see `attn_persist`.
/// The formats `PHOBOS_RAW_QMMA` names, or all of them when it is just a flag.
///
/// Which formats pay was a real question and the answer moved. Attributed one
/// card state at a time, `-p 128 -n 128 -r 1`, each row twice:
///
/// | fused | pp128, a slot a projection | pp128, a ring |
/// | --- | ---: | ---: |
/// | none | 71.8, 79.1 | 79.2, 79.0 |
/// | IQ1_S | 114.3, 110.2 | |
/// | IQ1_S, IQ2_XXS | 149.6, 149.4 | 151.8, 157.9 |
/// | **all four** | 135.3, 107.0 | **188.0, 183.8** |
///
/// IQ2_S and IQ2_XS *lost* while every fused projection took an activation slot
/// of its own, and win by a wide margin once those share a ring: 143 MiB of
/// slots across the model was more than their kernels were worth. See
/// [`DeviceBackend::act_slot_transient`]. It is worth keeping that the negative
/// result was real and was about residency, not about the kernels.
fn qmma_formats() -> Vec<Quant> {
    const ALL: [Quant; 6] = [
        Quant::IQ1_S,
        Quant::IQ2_XXS,
        Quant::IQ2_S,
        Quant::IQ2_XS,
        Quant::IQ3_XXS,
        Quant::IQ3_S,
    ];
    let Ok(value) = std::env::var("PHOBOS_RAW_QMMA") else {
        return ALL.to_vec();
    };
    if !value.contains(',') && ALL.iter().all(|q| !value.eq_ignore_ascii_case(q.name())) {
        return ALL.to_vec();
    }
    value
        .split(',')
        .filter_map(|want| {
            ALL.iter()
                .copied()
                .find(|q| q.name().eq_ignore_ascii_case(want.trim()))
        })
        .collect()
}

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1" | "on" | "yes" | "true")
    )
}

/// The same for a flag that is on unless it is turned off.
fn env_flag_on(name: &str) -> bool {
    !matches!(
        std::env::var(name).as_deref(),
        Ok("0" | "off" | "no" | "false")
    )
}

/// A device-resident Q8_0 weight: signed bytes, per-block scales in both
/// orders (`q8_mma` wants `[block, out]`, `qdot_t` wants `[out, block]`),
/// and the output width it was uploaded with.
/// The buffers behind a [`DeviceQuant`] or [`DeviceRaw`] that is not in the
/// arena. Held only to keep the allocation alive; nothing reads them.
type OwnedQuant = (DeviceBuffer<i8>, DeviceBuffer<f32>, DeviceBuffer<f32>);
type OwnedRaw = (
    DeviceBuffer<i8>,
    DeviceBuffer<u16>,
    Option<DeviceBuffer<u16>>,
);

/// A device-resident Q8_0 weight: where its int8 blocks and its two scale
/// layouts landed in the [`arena::Arena`], and the output width they went up
/// with. Pointers rather than buffers for the reason `arena.rs` gives.
struct DeviceQuant {
    qs: u64,
    scales: u64,
    row_scales: u64,
    n: usize,
}

use raw::DeviceRaw;

/// Output width, `k`, and split count: what [`kernels::q8_qmma_split_src`]'s
/// generated text is a function of.
type QmmaSplitKey = (usize, usize, usize);

/// Heads, head dimension, taps, head stride, normalize, query scale, rows, and
/// rows per program: everything [`delta_conv_src`] bakes in.
type ConvKey = (usize, usize, usize, usize, usize, bool, u32, usize, usize);

/// Head count, group size, head dimension and query group: the shape
/// [`DeviceBackend::attn_persist_plan`] settles a grid and split count for.
/// The split count itself is derived from occupancy, not part of the key.
type AttnPersistKey = (usize, usize, usize, usize);

/// The compiled module for one [`AttnPersistKey`], with the block count and
/// the persist-specific split count [`DeviceBackend::attn_persist_plan`]
/// settled on.
type AttnPersistEntry = (Module, u32, usize);

/// A device-resident backend for GGUF models. A whole decode step stays in
/// device memory; it synchronizes once, to read the logits.
pub struct DeviceBackend {
    stream: Stream,
    matmul: Variants,
    /// The tensor-core band [`DeviceBackend::matmul`] takes first for `m >=
    /// TC_TILE_M`; the plain `matmul` above finishes whatever doesn't fit a
    /// whole 64x64 tile.
    matmul_tc: Module,
    /// The same ladder over an f16 weight, which is what a `_qdecode` strip
    /// is. See `kernels/matmul.rs`'s `matmul_f16w_src`.
    matmul_f16w: Variants,
    matmul_tc_f16w: Module,
    matvec: Variants,
    q8_dp4a: Variants,
    q8_mma: Variants,
    q8_qmma: Module,
    q8_qmma_deep: HashMap<usize, Module>,
    /// The split-K variant of the deep tile and its reduction, keyed by output
    /// width, `k` and split count. Built when the unsplit grid is declined.
    q8_qmma_split: RefCell<HashMap<QmmaSplitKey, (Module, Module)>>,
    /// The narrow-CTA variant of the deep tile: same `qmma_t` kernel, half
    /// the threads and column tile, same per-warp patch. Compiled lazily,
    /// only when [`Self::qmma_narrow`] asks for it.
    q8_qmma_narrow: RefCell<Option<Module>>,
    q8_split: Variants,
    q8_qdot: Module,
    q8_qdot_add: Module,
    /// The persistent matvec, keyed by iteration count and whether it
    /// accumulates. Only built when `PHOBOS_PERSIST_QDOT` asks for it; see
    /// [`q8_qdot_persist_src`].
    q8_qdot_persist: RefCell<HashMap<(usize, bool), Module>>,
    /// Blocks a persistent kernel may use here, from the occupancy API, and zero
    /// until the first one is compiled and can be asked about.
    persist_blocks: Cell<u32>,
    persist_qdot: bool,
    /// Contract the IQ decode in `dp4a` against an int8 activation. **On**: 2x
    /// the float matvec, and worth tg128 11.20 -> 18.06 on Qwen3.8-27B with
    /// prefill unchanged.
    ///
    /// It quantizes the activation where the f32 host reference does not, so
    /// device against host cannot judge it -- those rows disagree by the size
    /// of an 8-bit activation however right the kernel is, which is why this
    /// waited on a check for so long. `backend_check` now runs the `m = 1` raw
    /// rows both ways on the device and compares them against each other, the
    /// way `fuse_check` already does for the fused decode path, and keeps the
    /// host comparison for the float path it replaces.
    ///
    /// It quantizes the activation where the f32 host reference does not, so
    /// `backend_check`'s `m = 1` rows move by the size of an 8-bit activation
    /// and device-against-host cannot settle it -- that bound is already too
    /// wide, which is why `fuse_check` exists. Device against device can:
    /// `argmax_check` on this model produces **identical tokens for all 32
    /// greedy decode steps on each of three prompts**, factual, narrative and
    /// code. The kernels themselves agree with the float bodies they replace to
    /// 1.096e-6 for IQ1_S and exactly 0 for IQ2_XXS and IQ3_XXS, and
    /// quantizing the activation is already what every Q8_0 projection here
    /// does. `PHOBOS_IQ1S_DP4A=1` opts in.
    iq1s_dp4a: Cell<bool>,
    /// Whether `q8_qmma`'s deep tile takes the split-K path on a starved grid.
    /// `PHOBOS_QMMA_SPLIT=1` opts in; off by default, a net wall-clock loss.
    qmma_split: bool,
    /// Whether `q8_qmma`'s deep tile takes the narrow-CTA path instead of
    /// the unsplit launch. `PHOBOS_QMMA_NARROW=1` opts in; default off, see
    /// [`kernels::q8_qmma_narrow_eligible`] for the row-count caveat.
    qmma_narrow: bool,
    /// Fused kernels the pass has emitted, with the plan that says what to bind
    /// to each. See [`fuse`].
    fused_plans: RefCell<HashMap<ChainKey, (Module, Plan)>>,
    fused_mlp: bool,
    fused_project: bool,
    /// Whether the delta net's convolution and gates join the projection's
    /// kernel. Separate from [`Self::fused_project`] since this one costs a
    /// barrier and has to be measurable against the projection alone.
    fused_mix: bool,
    /// Whether attention's output epilogue (quantizing the mixed heads, then
    /// the output projection) joins the fused-chain path. Default on.
    fused_attn_out: bool,
    /// Whether attention's key and value writes into the cache land in one
    /// launch instead of two. Default on, `PHOBOS_FUSED_STORE2D=0` off.
    fused_store2d: bool,
    /// Blocks a fused kernel is launched with, which unlike [`persist_blocks`]
    /// has to be exact: a block of the grid that is not resident never reaches
    /// the barrier. Zero until such a kernel exists to be asked about.
    fused_blocks: Cell<u32>,
    /// Storage for the values a plan found crossing a nest, by the plan's own
    /// scratch index, each grown to the largest a chain has asked for.
    fused_scratch: RefCell<Vec<(DeviceBuffer<i8>, DeviceBuffer<f32>)>>,
    /// A fused kernel's arrival counter and release generation, zeroed once. The
    /// barrier leaves both as it found them, so every launch of every layer
    /// reuses it.
    fused_barrier: RefCell<Option<DeviceBuffer<i32>>>,
    /// Operands for a fused launch, reused so a fused step allocates nothing.
    fused_operands: RefCell<Vec<(u64, [i64; 2])>>,
    functions: RefCell<HashMap<(usize, usize), cust::sys::CUfunction>>,
    func_shared: RefCell<HashMap<usize, u32>>,
    /// Per kernel, from its declared maxntid. See `threads_of`.
    func_threads: RefCell<HashMap<usize, u32>>,
    eager: RefCell<Recorded>,
    recording: Cell<bool>,
    flushed: Cell<bool>,
    pending: RefCell<Vec<Recorded>>,
    recorded_len: Cell<usize>,
    pass: RefCell<Option<PassGraph>>,
    /// Which replay to report on, and the launches seen so far in the current
    /// one. Zero when `PHOBOS_PASS_REPORT` is unset, which costs nothing.
    report_pass: Cell<usize>,
    report: RefCell<Vec<PassOp>>,
    /// Replays so far, so the report can name which one it is describing.
    reported: Cell<usize>,
    quantize: Module,
    quantize_wide: Module,
    pointwise: Module,
    pointwise_wide: Module,
    /// RMSNorm kernels, keyed by width and the epsilon's bit pattern.
    norms: RefCell<HashMap<(usize, u32), Module>>,
    quant_norms: RefCell<HashMap<(usize, u32), Module>>,
    gated_norms: RefCell<HashMap<(usize, u32), Module>>,
    gated_swiglu: RefCell<HashMap<usize, Module>>,
    /// A SwiGLU reading its two operands as planes of a wider buffer, by width.
    swiglu_planes: RefCell<HashMap<usize, Module>>,
    /// Keyed by head count and head dimension. Both are tile extents, so both
    /// have to be compile-time constants; a model only ever uses one pair.
    deltas: RefCell<HashMap<(usize, usize), Module>>,
    chunks: RefCell<HashMap<(usize, usize), Module>>,
    identities: RefCell<HashMap<usize, DeviceBuffer<f32>>>,
    /// Keyed by [`ConvKey`]. A model compiles three, one per plane, since the
    /// epilogue differs.
    convs: RefCell<HashMap<ConvKey, Module>>,
    /// Gate kernels, keyed by head count.
    gates: RefCell<HashMap<usize, Module>>,
    /// Strided copy kernels, keyed by direction, the width they copy, and
    /// whether the pitches let the promise be made.
    splits: RefCell<HashMap<(Strided, usize, bool), Module>>,
    /// [`Backend::store_2d_pair`] kernels, keyed by the width both windows
    /// share and whether the pitches let the alignment promise be made.
    store_pairs: RefCell<HashMap<(usize, bool), Module>>,
    /// Rotary kernels, keyed by head count and half the rotary width.
    ropes: RefCell<HashMap<(usize, usize), Module>>,
    /// [`Backend::rope_gather`] kernels, keyed by head count, half the
    /// rotary width, the source's heads-per-physical-row stride and the head
    /// dimension (the passthrough tail's width follows from the last two).
    rope_gathers: RefCell<HashMap<(usize, usize, usize, usize), Module>>,
    /// Attention kernels, keyed by query heads, group size and head dimension.
    attentions: RefCell<HashMap<(usize, usize, usize), Module>>,
    split_attn: RefCell<HashMap<(usize, usize, usize), Module>>,
    /// Whether [`DeviceBackend::attention_decode`] takes the persistent
    /// split-plus-merge path. Default on; `PHOBOS_ATTN_PERSIST=0` switches
    /// it off. Kept outside [`fused_stage`]'s `PHOBOS_FUSED` fallback since
    /// this gates a `grid_barrier`, not a launch chain: a grid this flag
    /// alone let through would hang rather than run slow, so it's
    /// `attn_persist_plan`'s occupancy check, not the flag, keeping that safe.
    attn_persist: bool,
    /// The persistent attention kernel, keyed by [`AttnPersistKey`], with its
    /// settled block and split count. `None` means occupancy couldn't fit
    /// the split phase in one pass, so the persistent path is declined.
    attn_persist_modules: RefCell<HashMap<AttnPersistKey, Option<AttnPersistEntry>>>,
    /// Per-split partial accumulators, and their running maxima and sums.
    attn_partials: RefCell<Option<(DeviceBuffer<f32>, DeviceBuffer<f32>)>>,
    /// Page-locked staging for the one readback a pass makes.
    readback: RefCell<Option<LockedBuffer<f32>>>,
    /// [`DeviceBackend::argmax`]'s reduction kernel, keyed by chunk width
    /// ([`argmax_chunk_width`]); stays a single entry in practice, since a
    /// vocabulary's size never changes for the life of the backend.
    argmax_reduce: RefCell<HashMap<usize, Module>>,
    /// [`argmax_reduce`]'s partial-value column, `[0, W)`, one per lane;
    /// added to each chunk's base index to recover a vocabulary position.
    argmax_iota: RefCell<HashMap<usize, DeviceBuffer<f32>>>,
    /// [`ARGMAX_FINISH_SRC`], compiled once: it takes no shape baked in.
    argmax_finish: Module,
    /// [`argmax_reduce`]'s per-block partials and [`argmax_finish`]'s
    /// answer, reused across calls since [`Backend::argmax`] is hot-path.
    argmax_scratch: RefCell<Option<(DeviceBuffer<f32>, DeviceBuffer<f32>)>>,
    /// The blocked attention kernels, keyed the same way.
    blocked: RefCell<HashMap<(usize, usize, usize), Module>>,
    attn_gemm: RefCell<HashMap<usize, Module>>,
    /// Addressed by [`Buf`]; a slot is `None` while free.
    slots: RefCell<Vec<Option<mem::Slot>>>,
    free_slots: RefCell<Vec<usize>>,
    /// Released allocations, handed out again rather than going back to the
    /// driver. See [`Backend::alloc`], [`DeviceBackend::alloc_written_now`].
    pool: Pool,
    /// The weight strip a prompt pass dequantizes into, and the output the
    /// matmul over it writes. One of each for the whole model rather than a
    /// pooled pair per weight. See [`DeviceBackend::dense_scratch`].
    dense_scratch: [Cell<Option<Buf>>; 2],
    /// Set by a pass that filled the scratch, cleared by the trim that follows
    /// it. See [`DeviceBackend::trim_after_dense`].
    drop_scratch: Cell<bool>,
    /// Passes begun, so [`DeviceBackend::mark_pass_vram`] can thin its output.
    pass_marks: Cell<usize>,
    /// Every distinct allocation length this backend has asked the pool for,
    /// and how many are outstanding. `PHOBOS_VRAM=1` only: the pool keys on
    /// exact length, so this is what its free list is made of.
    alloc_hist: RefCell<HashMap<usize, isize>>,
    constants: RefCell<HashMap<String, Buf>>,
    /// Addressed by [`QBuf`]: bytes, per-block scales, and the output width
    /// they went up with. Constants, so never released.
    quants: RefCell<Vec<DeviceQuant>>,
    q_constants: RefCell<HashMap<String, QBuf>>,
    /// Addressed by [`RawBuf`]: raw block bytes, the two header planes, the
    /// output width and the super-blocks per row they went up with.
    raw_quants: RefCell<Vec<DeviceRaw>>,
    /// The slabs every raw weight's bytes and header planes live in.
    arena: arena::Arena,
    /// The same for the small hot constants, in slabs a thirty-second the size.
    /// See [`arena::HOT_SLAB_BYTES`].
    hot: arena::Arena,
    /// And for a sequence's recurrent state, which is released when the
    /// sequence ends rather than never; `state_live` counts the regions so the
    /// slabs can go back once the last one does.
    state_arena: arena::Arena,
    state_live: Cell<usize>,
    /// Whether that arena is used at all. Off: it loses, and it is the third
    /// measurement saying the same thing. The arena's win is that a *cold* 15
    /// MiB weight stops being its own allocation, so the driver cannot single
    /// it out; data that is read or written every token is resident anyway,
    /// and pooling it only makes one slab's eviction cost more. tg128 reads
    /// 11.20 with each layer's state on its own and 9.98 pooled, the same
    /// direction as the Q8_0 planes and the f32 constants.
    /// `PHOBOS_STATE_ARENA=1` re-measures it.
    state_arena_on: bool,
    /// Whether the small, hot constants share the arena with the bulk weights.
    /// Off, because measured they should not: the arena's win is that a cold
    /// 15 MiB weight stops being its own allocation, and a hot 0.1 MiB scale
    /// plane put in a 128 MiB slab instead drags the whole slab resident.
    /// tg128 reads 9.74 with only the raw weights arena'd, 8.65 once the Q8_0
    /// planes join them and 7.82 once the f32 constants do.
    /// `PHOBOS_ARENA_CONST=1` puts them back in.
    arena_const: bool,
    /// Whether the bulk weights go in the arena at all. `PHOBOS_ARENA=0` gives
    /// every tensor its own allocation again, which is what this replaced.
    arena_weights: bool,
    /// Buffers a constant owns when it is not in the arena, kept alive so the
    /// pointer in its [`DeviceQuant`] or [`DeviceRaw`] stays good.
    owned_quants: RefCell<Vec<OwnedQuant>>,
    owned_raw: RefCell<Vec<OwnedRaw>>,
    raw_constants: RefCell<HashMap<String, RawBuf>>,
    /// Every raw format's matvec, compiled once in parallel (see
    /// `compile_parallel`); each shape is a kernel parameter, not baked in.
    q2k_matvec: Module,
    q3k_matvec: Module,
    iq1s_matvec: Module,
    iq2xxs_matvec: Module,
    iq1m_matvec: Module,
    iq2s_matvec: Module,
    iq2xs_matvec: Module,
    iq3xxs_matvec: Module,
    iq3s_matvec: Module,
    iq4xs_matvec: Module,
    /// Each format's `m == 1` decode folded into one `*_qdot_t` call, no
    /// per-lane shared-memory staging. Needs `N` to be a whole number of its
    /// own `TN`; `project_raw` falls back to the plain matvec otherwise.
    iq1s_qdot_matvec: Module,
    /// The dp4a decode matvec, wide tile then narrow; see `I8_NARROW_TN`.
    /// The dp4a decode matvecs: wide tile, then narrow for a ragged `n`.
    iq1s_qdot_i8: [Module; 2],
    iq3s_qdot_i8: [Module; 2],
    iq3xxs_qdot_i8: [Module; 2],
    iq2xxs_qdot_i8: [Module; 2],
    iq1m_qdot_i8: [Module; 2],
    iq2xs_qdot_i8: [Module; 2],
    iq2s_qdot_i8: [Module; 2],
    iq2xxs_qdot_matvec: Module,
    iq1m_qdot_matvec: Module,
    iq2s_qdot_matvec: Module,
    iq2xs_qdot_matvec: Module,
    iq3xxs_qdot_matvec: Module,
    iq3s_qdot_matvec: Module,
    iq4xs_qdot_matvec: Module,
    q2k_qdot_matvec: Module,
    q3k_qdot_matvec: Module,
    /// Every raw format's own decode minus the per-row reduction: writes a
    /// `[K, N]` strip of dequantized weight for
    /// [`DeviceBackend::project_raw_dense`] to run a batched matmul against.
    /// `m == 1` still uses the matching `_matvec` kernel unchanged.
    iq1s_dequant: Module,
    iq2xxs_dequant: Module,
    iq1m_dequant: Module,
    iq2s_dequant: Module,
    iq2xs_dequant: Module,
    iq3xxs_dequant: Module,
    iq3s_dequant: Module,
    iq4xs_dequant: Module,
    /// The warp-collective form of the `_dequant` kernels above: one
    /// `*_qdecode_t` call, nothing staged, no barrier. Needs a strip that is a
    /// whole number of its own `TN`.
    iq1s_qdecode: Module,
    iq2xxs_qdecode: Module,
    iq1m_qdecode: Module,
    iq2s_qdecode: Module,
    iq2xs_qdecode: Module,
    iq3xxs_qdecode: Module,
    iq3s_qdecode: Module,
    /// The same seven writing an f16 strip, launched only where the matmul
    /// reading it is entirely tensor-core. See `project_raw_dense`.
    iq1s_qdecode_f16: Module,
    iq2xxs_qdecode_f16: Module,
    iq1m_qdecode_f16: Module,
    iq2s_qdecode_f16: Module,
    iq2xs_qdecode_f16: Module,
    iq3xxs_qdecode_f16: Module,
    iq3s_qdecode_f16: Module,
    /// Q2_K's own dequant, same purpose as the block above. Q3_K has none:
    /// it is the LM head, run only at `m == 1`, so it never takes
    /// `project_raw_dense`'s batched path.
    q2k_dequant: Module,
    /// IQ1_S's grid, uploaded once and shared by every IQ1_S weight:
    /// [`crate::quant::iq1s_flat_grid`] unpacked to one `i32` lane a slot.
    /// IQ1_M shares this same grid; see `quant/iq1_m.rs`'s module doc.
    iq1s_grid: DeviceBuffer<i32>,
    /// IQ2_XXS's magnitude grid and sign table, flattened like IQ1_S's and
    /// shared by every IQ2_XXS weight.
    iq2xxs_grid: DeviceBuffer<i32>,
    iq2xxs_signs: DeviceBuffer<i32>,
    /// IQ2_S's own magnitude grid and sign table; wider than IQ2_XXS's and
    /// keyed by raw byte rather than parity index, so not shared with it.
    iq2s_grid: DeviceBuffer<i32>,
    iq2s_signs: DeviceBuffer<i32>,
    /// IQ2_XS's own magnitude grid (wider than IQ2_XXS's), but shares its
    /// sign mechanism, so it reuses `iq2xxs_signs`.
    iq2xs_grid: DeviceBuffer<i32>,
    /// IQ3_XXS's own magnitude grid (four `i32` lanes an entry, `u32`
    /// entries); its sign field matches IQ2_XXS's `aux32`, so it reuses
    /// `iq2xxs_signs`.
    iq3xxs_grid: DeviceBuffer<i32>,
    /// IQ3_S's own magnitude grid (also four `i32` lanes an entry); its
    /// sign byte matches IQ2_S's, so it reuses `iq2s_signs`.
    iq3s_grid: DeviceBuffer<i32>,
    /// IQ4_XS's fixed sixteen-value codebook every nibble indexes directly
    /// ([`crate::quant::iq4xs_flat_codebook`]); needs no per-lane unpacking,
    /// and its `gather` covers a whole run rather than a lane.
    iq4xs_codebook: DeviceBuffer<i32>,
    /// The same tables at one `i8` a slot, which every `*_qdot_t`/`*_qdecode_t`
    /// reads: a lane's entry is then one vector load, and the table a quarter
    /// the size. The i32 copies stay for the `gather`-based fallbacks.
    iq1s_grid_packed: DeviceBuffer<i8>,
    /// The same table with the delta folded in and both signs laid out, which
    /// is what makes an IQ1_S weight an exact `i8`. See `quant::iq1s_signed_grid`.
    iq1s_signed_grid: DeviceBuffer<i8>,
    /// IQ1_S's prompt projection: the decode contracted on the integer tensor
    /// cores, so no expanded weight is ever written. See `qmma_raw.rs`.
    iq1s_qmma: Module,
    /// IQ2_XXS's, the second largest item in a prompt pass, and the two that
    /// share its decode but scale per sixteen elements.
    iq2xxs_qmma: Module,
    iq2s_qmma: Module,
    iq2xs_qmma: Module,
    /// The two whose grid entry is four bytes rather than eight.
    iq3xxs_qmma: Module,
    iq3s_qmma: Module,
    /// Whether that projection is used. On now, and `PHOBOS_RAW_QMMA=0` turns
    /// it off. It was off while it cost decode more than it bought prefill,
    /// which the activation ring and the 32 MiB dequant scratch between them
    /// undid: at the current default, each row twice, it measures **pp128
    /// 192.8, 192.9 against 58.4, 58.2 and tg128 18.05, 18.02 against 18.04,
    /// 18.03**. Prefill 3.3x, decode unchanged.
    raw_qmma: Cell<bool>,
    /// Which formats it covers. `PHOBOS_RAW_QMMA` takes a comma-separated
    /// list as well as a flag -- `iq1s,iq2xxs` -- so the formats can be
    /// attributed one at a time. Fusing one trades its expansion for an
    /// activation slot a projection, and on a card past its residency cliff
    /// that trade can go either way, so which formats pay is a measurement
    /// rather than a given.
    raw_qmma_formats: Vec<Quant>,
    /// IQ1_S's staged projection, when asked for. See `qmma_raw.rs`.
    qgemm: qmma_raw::Qgemm,
    /// `PHOBOS_DENSE_SCRATCH=0` goes back to a pooled pair per weight and
    /// `PHOBOS_TRIM=1` to handing the whole free list back after a dense pass.
    /// Both are what the prompt path used to do, kept so the pair can be
    /// measured in one session; on Qwen3.8-27B they cost **pp128 82.3 against
    /// 18.8 and tg128 8.19 against 7.52**. See `residency.rs`.
    dense_scratch_shared: bool,
    trim_after_dense: bool,
    iq2xxs_grid_packed: DeviceBuffer<i8>,
    /// +/-1 for the float decode, then the same signs as a 0/-1 mask for the
    /// dp4a one, which applies them with `and`.
    iq2xxs_signs_packed: DeviceBuffer<i8>,
    iq2s_grid_packed: DeviceBuffer<i8>,
    iq2s_signs_packed: DeviceBuffer<i8>,
    iq2xs_grid_packed: DeviceBuffer<i8>,
    iq3xxs_grid_packed: DeviceBuffer<i8>,
    iq3s_grid_packed: DeviceBuffer<i8>,
    /// `[0, 1, .., 7]`: the row offsets a grid-coded raw kernel's batched
    /// `gather` broadcasts against, so one call reads a whole eight-wide lane.
    iota8: DeviceBuffer<i32>,
    /// One entry per `quantize_act` in a pass, grown to the largest
    /// projection seen. Slot by slot, not one arena: several are live at
    /// once, and a pass asks for them in the same order every time.
    act_scratch: RefCell<Vec<(DeviceBuffer<i8>, DeviceBuffer<f32>)>>,
    /// Which of the first few `act_scratch` slots the next transient
    /// activation takes. See [`DeviceBackend::act_slot_transient`].
    act_ring: Cell<usize>,
    act_next: Cell<usize>,
    /// Split-K partial sums, `[splits, n]`, grown to the largest asked for.
    split_scratch: RefCell<Option<DeviceBuffer<f32>>>,
    /// Must be last: Rust drops in declaration order and every allocation
    /// above has to be released while the context is still alive.
    _ctx: cust::context::Context,
}

impl DeviceBackend {
    /// Turns every fused stage on or off, whatever the environment asked
    /// for. Needed because [`fused_stage`] reads its env vars once at
    /// construction, but the equivalence harness wants both paths in one
    /// process; see `examples/fuse_check.rs`.
    /// Turn the `dp4a` decode matvecs on or off after construction, so a check
    /// can run the same projection both ways in one session and compare them
    /// against each other. Device against host cannot judge this path: the
    /// host reference does not quantize the activation, so it disagrees by the
    /// size of an 8-bit one however right the kernel is.
    pub fn set_iq_dp4a(&self, on: bool) {
        self.iq1s_dp4a.set(on);
    }

    /// The same for the fused prompt projection. It quantizes its activation
    /// too, so the host cannot judge it either; `backend_check` runs the shapes
    /// it takes both ways on the device.
    pub fn set_raw_qmma(&self, on: bool) {
        self.raw_qmma.set(on);
    }

    pub fn set_fused(&mut self, on: bool) {
        self.fused_mlp = on;
        self.fused_project = on;
        self.fused_mix = on;
        self.fused_attn_out = on;
        self.fused_store2d = on;
    }
}
