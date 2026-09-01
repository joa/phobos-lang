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
mod elem;
mod fused;
mod graph;
use graph::{PassGraph, PassOp, Recorded};
mod kernels;
mod launch;
mod matmul;
mod mem;
mod qmma_raw;
mod residency;

use fused::*;

// The catalog is written as one flat namespace of sources and tile sizes, and
// the impls below reach for them unqualified; see kernels/mod.rs.
use kernels::*;

pub use kernels::{ATTN_GEMM_TILE, ATTN_SOFT_TILE, attn_gemm_src};

/// Whether an opt-in `PHOBOS_*` toggle is set. Opt-out toggles (a default-on
/// mechanism) negate the complementary spelling instead; see `attn_persist`.
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

/// A device-resident raw-block weight: where its file bytes and `f16` header
/// plane(s) landed in the [`arena::Arena`] (`dmin` absent for a format with no
/// minimum term), output width, super-blocks per row, and which format's
/// kernel decodes it.
///
/// Device pointers rather than buffers because the arena owns the allocation:
/// see `arena.rs` for why the weights share a dozen of those rather than
/// taking one each.
struct DeviceRaw {
    bytes: u64,
    d: u64,
    dmin: Option<u64>,
    n: usize,
    nb: usize,
    quant: Quant,
}

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
    /// The dp4a decode matvec, wide tile then narrow; see `I8_NARROW_TN`.
    iq2xxs_qdot_i8: [Module; 2],
    /// The dp4a decode matvec, wide tile then narrow; see `I8_NARROW_TN`.
    iq1m_qdot_i8: [Module; 2],
    /// The dp4a decode matvec, wide tile then narrow; see `I8_NARROW_TN`.
    iq2xs_qdot_i8: [Module; 2],
    /// The dp4a decode matvec, wide tile then narrow; see `I8_NARROW_TN`.
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
    /// Whether that projection is used. Off until it stops costing decode what
    /// it buys prefill: `PHOBOS_RAW_QMMA=1` measures **pp128 101.8 against
    /// 82.3 and tg128 6.23 against 8.19**, and the decode side is the
    /// activation slots it takes, one a projection, not the kernel.
    raw_qmma: Cell<bool>,
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

    pub fn new() -> Result<DeviceBackend> {
        let _ctx = cust::quick_init().context("initializing CUDA")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        residency::vram_mark("cuda context up");

        let matmul = Variants::compile(
            MATMUL_SRC,
            &[("TILE_M", TILE_M), ("TILE_N", TILE_N), ("TILE_K", TILE_K)],
            "matmul",
            ("@aligned(M = TILE_M, N = TILE_N)", ""),
        )?;
        let matmul_tc = compile(
            MATMUL_TC_SRC,
            &[
                ("TILE_M", TC_TILE_M),
                ("TILE_N", TC_TILE_N),
                ("TILE_K", TC_TILE_K),
            ],
            "matmul_tc",
        )?;
        let matmul_f16w_body = matmul_f16w_src();
        let matmul_f16w = Variants::compile(
            &matmul_f16w_body,
            &[("TILE_M", TILE_M), ("TILE_N", TILE_N), ("TILE_K", TILE_K)],
            "matmul",
            ("@aligned(M = TILE_M, N = TILE_N)", ""),
        )?;
        let matmul_tc_f16w = compile(
            &matmul_tc_f16w_src(),
            &[
                ("TILE_M", TC_TILE_M),
                ("TILE_N", TC_TILE_N),
                ("TILE_K", TC_TILE_K),
            ],
            "matmul_tc",
        )?;
        let matvec = Variants::compile(
            MATVEC_SRC,
            &[("TILE_N", MV_TN), ("TILE_K", TILE_K)],
            "matvec",
            ("@aligned(N = TILE_N)", ""),
        )?;
        // matmul_quant requires k to be a whole number of Q8_0 blocks, so the k
        // loop never has a remainder to split off.
        let q8_dp4a = Variants::compile(
            Q8_DP4A_SRC,
            &[("TN", Q8_TN)],
            "q8_dp4a",
            ("@aligned(N = TN, K = 32)", "@aligned(K = 32)"),
        )?;
        let q8_mma = Variants::compile(
            Q8_MMA_SRC,
            &[("TM", Q8_MMA_TM), ("TN", Q8_MMA_TN)],
            "q8_mma",
            ("@aligned(M = TM, N = TN, K = 32)", "@aligned(K = 32)"),
        )?;
        let qmma_src = q8_qmma_src(Q8_QMMA_CTA);
        let q8_qmma = compile(
            &qmma_src,
            &[("TM", Q8_QMMA_SHALLOW), ("TN", Q8_QMMA_TN)],
            "q8_qmma",
        )?;
        let mut q8_qmma_deep = HashMap::new();
        for tn in Q8_QMMA_WIDTHS {
            let module = compile(&qmma_src, &[("TM", Q8_QMMA_TM), ("TN", tn)], "q8_qmma")?;
            q8_qmma_deep.insert(tn, module);
        }
        let q8_split = Variants::compile(
            Q8_SPLIT_SRC,
            &[("TN", Q8_TN), ("RT", Q8_REDUCE_TN)],
            "q8_split",
            ("@aligned(N = TN, K = 32)", "@aligned(K = 32)"),
        )?;
        let q8_qdot = compile(Q8_QDOT_SRC, &[("TN", Q8_QDOT_TN)], "q8_qdot")?;
        let q8_qdot_add = compile(Q8_QDOT_ADD_SRC, &[("TN", Q8_QDOT_TN)], "q8_qdot_add")?;
        let quantize = compile(QUANTIZE_SRC, &[("TB", QUANT_TB)], "quantize")?;
        let quantize_wide = compile(QUANTIZE_SRC, &[("TB", QUANT_TB_WIDE)], "quantize")?;
        let pointwise = compile(POINTWISE_SRC, &[("TILE", ELEM_TILE)], "pointwise")?;
        let pointwise_wide = compile(POINTWISE_SRC, &[("TILE", ELEM_TILE_WIDE)], "pointwise")?;
        let argmax_finish = compile(ARGMAX_FINISH_SRC, &[], "argmax_finish")?;
        // Independent kernels compiled in parallel (`compile_parallel`), so
        // MLIR-to-PTX lowering runs concurrently. `Vec::remove(0)` keeps the
        // destructure in the jobs' own order without cloning `Module`s.
        let q2k_src = q2k_matvec_src(Q2K_TN);
        let q3k_src = q3k_matvec_src(Q3K_TN);
        let iq1s_src = iq1s_matvec_src(IQ1S_TN);
        let iq2xxs_src = iq2xxs_matvec_src(IQ2XXS_TN);
        let iq1m_src = iq1m_matvec_src(IQ1M_TN);
        let iq2s_src = iq2s_matvec_src(IQ2S_TN);
        let iq2xs_src = iq2xs_matvec_src(IQ2XS_TN);
        let iq3xxs_src = iq3xxs_matvec_src(IQ3XXS_TN);
        let iq3s_src = iq3s_matvec_src(IQ3S_TN);
        let iq4xs_src = iq4xs_matvec_src(IQ4XS_TN);
        let iq1s_dequant_body = iq1s_dequant_src(IQ1S_TN);
        let iq2xxs_dequant_body = iq2xxs_dequant_src(IQ2XXS_TN);
        let iq1m_dequant_body = iq1m_dequant_src(IQ1M_TN);
        let iq2s_dequant_body = iq2s_dequant_src(IQ2S_TN);
        let iq2xs_dequant_body = iq2xs_dequant_src(IQ2XS_TN);
        let iq3xxs_dequant_body = iq3xxs_dequant_src(IQ3XXS_TN);
        let iq3s_dequant_body = iq3s_dequant_src(IQ3S_TN);
        let iq4xs_dequant_body = iq4xs_dequant_src(IQ4XS_TN);
        let q2k_dequant_body = q2k_dequant_src(Q2K_TN);
        let iq1s_qmma_body = iq1s_qmma_src(IQ1S_QMMA_CTA, IQ1S_QMMA_TM, IQ1S_QMMA_TN);
        let iq2xxs_qmma_body = iq2xxs_qmma_src(IQ1S_QMMA_CTA, IQ1S_QMMA_TM, IQ1S_QMMA_TN);
        let iq2s_qmma_body = iq2s_qmma_src(IQ1S_QMMA_CTA, IQ1S_QMMA_TM, IQ1S_QMMA_TN);
        let iq2xs_qmma_body = iq2xs_qmma_src(IQ1S_QMMA_CTA, IQ1S_QMMA_TM, IQ1S_QMMA_TN);
        let iq1s_qdecode_body = iq1s_qdecode_src(IQ1S_TN);
        let iq2xxs_qdecode_body = iq2xxs_qdecode_src(IQ2XXS_TN);
        let iq1m_qdecode_body = iq1m_qdecode_src(IQ1M_TN);
        let iq2s_qdecode_body = iq2s_qdecode_src(IQ2S_TN);
        let iq2xs_qdecode_body = iq2xs_qdecode_src(IQ2XS_TN);
        let iq3xxs_qdecode_body = iq3xxs_qdecode_src(IQ3XXS_TN);
        let iq3s_qdecode_body = iq3s_qdecode_src(IQ3S_TN);
        // The f16 strip is the same kernel with a narrower destination.
        let f16_scratch =
            |src: &str| src.replace("SCRATCH: tensor<f32>[K, N]", "SCRATCH: tensor<f16>[K, N]");
        let iq1s_qdecode_f16_body = f16_scratch(&iq1s_qdecode_body);
        let iq2xxs_qdecode_f16_body = f16_scratch(&iq2xxs_qdecode_body);
        let iq1m_qdecode_f16_body = f16_scratch(&iq1m_qdecode_body);
        let iq2s_qdecode_f16_body = f16_scratch(&iq2s_qdecode_body);
        let iq2xs_qdecode_f16_body = f16_scratch(&iq2xs_qdecode_body);
        let iq3xxs_qdecode_f16_body = f16_scratch(&iq3xxs_qdecode_body);
        let iq3s_qdecode_f16_body = f16_scratch(&iq3s_qdecode_body);
        let iq1s_qdot_body = iq1s_qdot_matvec_src(IQ1S_TN);
        let iq2xxs_qdot_body = iq2xxs_qdot_matvec_src(IQ2XXS_TN);
        let iq1m_qdot_body = iq1m_qdot_matvec_src(IQ1M_TN);
        let iq2s_qdot_body = iq2s_qdot_matvec_src(IQ2S_TN);
        let iq2xs_qdot_body = iq2xs_qdot_matvec_src(IQ2XS_TN);
        let iq3xxs_qdot_body = iq3xxs_qdot_matvec_src(IQ3XXS_TN);
        let iq3s_qdot_body = iq3s_qdot_matvec_src(IQ3S_TN);
        let iq4xs_qdot_body = iq4xs_qdot_matvec_src(IQ4XS_TN);
        let q2k_qdot_body = q2k_qdot_matvec_src(Q2K_TN);
        let q3k_qdot_body = q3k_qdot_matvec_src(Q3K_TN);
        // The dp4a decode matvecs, wide tile then narrow. One table drives the
        // sources, the compile entries and the `remove`s below, so their order
        // cannot drift apart.
        let i8: [I8Row; 7] = [
            (
                iq1s_qdot_i8_matvec_src,
                IQ1S_I8_TN,
                IQ1S_I8_NARROW_TN,
                "iq1s_qdot_i8_matvec",
            ),
            (
                iq3s_qdot_i8_matvec_src,
                IQ3S_I8_TN,
                IQ3S_I8_NARROW_TN,
                "iq3s_qdot_i8_matvec",
            ),
            (
                iq3xxs_qdot_i8_matvec_src,
                IQ3XXS_I8_TN,
                IQ3XXS_I8_NARROW_TN,
                "iq3xxs_qdot_i8_matvec",
            ),
            (
                iq2xxs_qdot_i8_matvec_src,
                IQ2XXS_I8_TN,
                IQ2XXS_I8_NARROW_TN,
                "iq2xxs_qdot_i8_matvec",
            ),
            (
                iq1m_qdot_i8_matvec_src,
                IQ1M_I8_TN,
                IQ1M_I8_NARROW_TN,
                "iq1m_qdot_i8_matvec",
            ),
            (
                iq2xs_qdot_i8_matvec_src,
                IQ2XS_I8_TN,
                IQ2XS_I8_NARROW_TN,
                "iq2xs_qdot_i8_matvec",
            ),
            (
                iq2s_qdot_i8_matvec_src,
                IQ2S_I8_TN,
                IQ2S_I8_NARROW_TN,
                "iq2s_qdot_i8_matvec",
            ),
        ];
        let i8_srcs: Vec<OwnedEntry> = i8
            .iter()
            .flat_map(|&(src, w, n, name)| {
                [(src(w), [("TN", w)], name), (src(n), [("TN", n)], name)]
            })
            .collect();

        let mut raw_entries: Vec<Entry> = vec![
            (q2k_src.as_str(), &[("TN", Q2K_TN)], "q2k_matvec"),
            (q3k_src.as_str(), &[("TN", Q3K_TN)], "q3k_matvec"),
            (iq1s_src.as_str(), &[("TN", IQ1S_TN)], "iq1s_matvec"),
            (iq2xxs_src.as_str(), &[("TN", IQ2XXS_TN)], "iq2xxs_matvec"),
            (iq1m_src.as_str(), &[("TN", IQ1M_TN)], "iq1m_matvec"),
            (iq2s_src.as_str(), &[("TN", IQ2S_TN)], "iq2s_matvec"),
            (iq2xs_src.as_str(), &[("TN", IQ2XS_TN)], "iq2xs_matvec"),
            (iq3xxs_src.as_str(), &[("TN", IQ3XXS_TN)], "iq3xxs_matvec"),
            (iq3s_src.as_str(), &[("TN", IQ3S_TN)], "iq3s_matvec"),
            (iq4xs_src.as_str(), &[("TN", IQ4XS_TN)], "iq4xs_matvec"),
            (
                iq1s_dequant_body.as_str(),
                &[("TN", IQ1S_TN)],
                "iq1s_dequant",
            ),
            (
                iq2xxs_dequant_body.as_str(),
                &[("TN", IQ2XXS_TN)],
                "iq2xxs_dequant",
            ),
            (
                iq1m_dequant_body.as_str(),
                &[("TN", IQ1M_TN)],
                "iq1m_dequant",
            ),
            (
                iq2s_dequant_body.as_str(),
                &[("TN", IQ2S_TN)],
                "iq2s_dequant",
            ),
            (
                iq2xs_dequant_body.as_str(),
                &[("TN", IQ2XS_TN)],
                "iq2xs_dequant",
            ),
            (
                iq3xxs_dequant_body.as_str(),
                &[("TN", IQ3XXS_TN)],
                "iq3xxs_dequant",
            ),
            (
                iq3s_dequant_body.as_str(),
                &[("TN", IQ3S_TN)],
                "iq3s_dequant",
            ),
            (
                iq4xs_dequant_body.as_str(),
                &[("TN", IQ4XS_TN)],
                "iq4xs_dequant",
            ),
            (q2k_dequant_body.as_str(), &[("TN", Q2K_TN)], "q2k_dequant"),
            (
                iq1s_qdecode_body.as_str(),
                &[("TN", IQ1S_TN)],
                "iq1s_qdecode",
            ),
            (
                iq2xxs_qdecode_body.as_str(),
                &[("TN", IQ2XXS_TN)],
                "iq2xxs_qdecode",
            ),
            (
                iq1m_qdecode_body.as_str(),
                &[("TN", IQ1M_TN)],
                "iq1m_qdecode",
            ),
            (
                iq2s_qdecode_body.as_str(),
                &[("TN", IQ2S_TN)],
                "iq2s_qdecode",
            ),
            (
                iq2xs_qdecode_body.as_str(),
                &[("TN", IQ2XS_TN)],
                "iq2xs_qdecode",
            ),
            (
                iq3xxs_qdecode_body.as_str(),
                &[("TN", IQ3XXS_TN)],
                "iq3xxs_qdecode",
            ),
            (
                iq3s_qdecode_body.as_str(),
                &[("TN", IQ3S_TN)],
                "iq3s_qdecode",
            ),
            (
                iq1s_qdecode_f16_body.as_str(),
                &[("TN", IQ1S_TN)],
                "iq1s_qdecode",
            ),
            (
                iq2xxs_qdecode_f16_body.as_str(),
                &[("TN", IQ2XXS_TN)],
                "iq2xxs_qdecode",
            ),
            (
                iq1m_qdecode_f16_body.as_str(),
                &[("TN", IQ1M_TN)],
                "iq1m_qdecode",
            ),
            (
                iq2s_qdecode_f16_body.as_str(),
                &[("TN", IQ2S_TN)],
                "iq2s_qdecode",
            ),
            (
                iq2xs_qdecode_f16_body.as_str(),
                &[("TN", IQ2XS_TN)],
                "iq2xs_qdecode",
            ),
            (
                iq3xxs_qdecode_f16_body.as_str(),
                &[("TN", IQ3XXS_TN)],
                "iq3xxs_qdecode",
            ),
            (
                iq3s_qdecode_f16_body.as_str(),
                &[("TN", IQ3S_TN)],
                "iq3s_qdecode",
            ),
            (
                iq1s_qdot_body.as_str(),
                &[("TN", IQ1S_TN)],
                "iq1s_qdot_matvec",
            ),
            (
                iq2xxs_qdot_body.as_str(),
                &[("TN", IQ2XXS_TN)],
                "iq2xxs_qdot_matvec",
            ),
            (
                iq1m_qdot_body.as_str(),
                &[("TN", IQ1M_TN)],
                "iq1m_qdot_matvec",
            ),
            (
                iq2s_qdot_body.as_str(),
                &[("TN", IQ2S_TN)],
                "iq2s_qdot_matvec",
            ),
            (
                iq2xs_qdot_body.as_str(),
                &[("TN", IQ2XS_TN)],
                "iq2xs_qdot_matvec",
            ),
            (
                iq3xxs_qdot_body.as_str(),
                &[("TN", IQ3XXS_TN)],
                "iq3xxs_qdot_matvec",
            ),
            (
                iq3s_qdot_body.as_str(),
                &[("TN", IQ3S_TN)],
                "iq3s_qdot_matvec",
            ),
            (
                iq4xs_qdot_body.as_str(),
                &[("TN", IQ4XS_TN)],
                "iq4xs_qdot_matvec",
            ),
            (q2k_qdot_body.as_str(), &[("TN", Q2K_TN)], "q2k_qdot_matvec"),
            (q3k_qdot_body.as_str(), &[("TN", Q3K_TN)], "q3k_qdot_matvec"),
        ];
        raw_entries.extend(
            i8_srcs
                .iter()
                .map(|(b, d, n)| (b.as_str(), d.as_slice(), *n)),
        );
        raw_entries.push((
            iq1s_qmma_body.as_str(),
            &[("TM", IQ1S_QMMA_TM), ("TN", IQ1S_QMMA_TN)],
            "iq1s_qmma",
        ));
        raw_entries.push((
            iq2xxs_qmma_body.as_str(),
            &[("TM", IQ1S_QMMA_TM), ("TN", IQ1S_QMMA_TN)],
            "iq2xxs_qmma",
        ));
        raw_entries.push((
            iq2s_qmma_body.as_str(),
            &[("TM", IQ1S_QMMA_TM), ("TN", IQ1S_QMMA_TN)],
            "iq2s_qmma",
        ));
        raw_entries.push((
            iq2xs_qmma_body.as_str(),
            &[("TM", IQ1S_QMMA_TM), ("TN", IQ1S_QMMA_TN)],
            "iq2xs_qmma",
        ));
        let mut raw_matvecs = compile_parallel(&raw_entries)?;
        let q2k_matvec = raw_matvecs.remove(0);
        let q3k_matvec = raw_matvecs.remove(0);
        let iq1s_matvec = raw_matvecs.remove(0);
        let iq2xxs_matvec = raw_matvecs.remove(0);
        let iq1m_matvec = raw_matvecs.remove(0);
        let iq2s_matvec = raw_matvecs.remove(0);
        let iq2xs_matvec = raw_matvecs.remove(0);
        let iq3xxs_matvec = raw_matvecs.remove(0);
        let iq3s_matvec = raw_matvecs.remove(0);
        let iq4xs_matvec = raw_matvecs.remove(0);
        let iq1s_dequant = raw_matvecs.remove(0);
        let iq2xxs_dequant = raw_matvecs.remove(0);
        let iq1m_dequant = raw_matvecs.remove(0);
        let iq2s_dequant = raw_matvecs.remove(0);
        let iq2xs_dequant = raw_matvecs.remove(0);
        let iq3xxs_dequant = raw_matvecs.remove(0);
        let iq3s_dequant = raw_matvecs.remove(0);
        let iq4xs_dequant = raw_matvecs.remove(0);
        let q2k_dequant = raw_matvecs.remove(0);
        let iq1s_qdecode = raw_matvecs.remove(0);
        let iq2xxs_qdecode = raw_matvecs.remove(0);
        let iq1m_qdecode = raw_matvecs.remove(0);
        let iq2s_qdecode = raw_matvecs.remove(0);
        let iq2xs_qdecode = raw_matvecs.remove(0);
        let iq3xxs_qdecode = raw_matvecs.remove(0);
        let iq3s_qdecode = raw_matvecs.remove(0);
        let iq1s_qdecode_f16 = raw_matvecs.remove(0);
        let iq2xxs_qdecode_f16 = raw_matvecs.remove(0);
        let iq1m_qdecode_f16 = raw_matvecs.remove(0);
        let iq2s_qdecode_f16 = raw_matvecs.remove(0);
        let iq2xs_qdecode_f16 = raw_matvecs.remove(0);
        let iq3xxs_qdecode_f16 = raw_matvecs.remove(0);
        let iq3s_qdecode_f16 = raw_matvecs.remove(0);
        let iq1s_qdot_matvec = raw_matvecs.remove(0);
        let iq2xxs_qdot_matvec = raw_matvecs.remove(0);
        let iq1m_qdot_matvec = raw_matvecs.remove(0);
        let iq2s_qdot_matvec = raw_matvecs.remove(0);
        let iq2xs_qdot_matvec = raw_matvecs.remove(0);
        let iq3xxs_qdot_matvec = raw_matvecs.remove(0);
        let iq3s_qdot_matvec = raw_matvecs.remove(0);
        let iq4xs_qdot_matvec = raw_matvecs.remove(0);
        let q2k_qdot_matvec = raw_matvecs.remove(0);
        let q3k_qdot_matvec = raw_matvecs.remove(0);
        let iq1s_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq3s_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq3xxs_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq2xxs_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq1m_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq2xs_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq2s_qdot_i8 = [raw_matvecs.remove(0), raw_matvecs.remove(0)];
        let iq1s_qmma = raw_matvecs.remove(0);
        let iq2xxs_qmma = raw_matvecs.remove(0);
        let iq2s_qmma = raw_matvecs.remove(0);
        let iq2xs_qmma = raw_matvecs.remove(0);
        let iq1s_grid = DeviceBuffer::from_slice(&crate::quant::iq1s_flat_grid())?;
        let iq2xxs_grid = DeviceBuffer::from_slice(&crate::quant::iq2xxs_flat_grid())?;
        let iq2xxs_signs = DeviceBuffer::from_slice(&crate::quant::iq2xxs_flat_signs())?;
        let iq2s_grid = DeviceBuffer::from_slice(&crate::quant::iq2s_flat_grid())?;
        let iq2s_signs = DeviceBuffer::from_slice(&crate::quant::iq2s_flat_signs())?;
        let iq2xs_grid = DeviceBuffer::from_slice(&crate::quant::iq2xs_flat_grid())?;
        let iq3xxs_grid = DeviceBuffer::from_slice(&crate::quant::iq3xxs_flat_grid())?;
        let iq3s_grid = DeviceBuffer::from_slice(&crate::quant::iq3s_flat_grid())?;
        let iq4xs_codebook = DeviceBuffer::from_slice(&crate::quant::iq4xs_flat_codebook())?;
        let iq1s_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq1s_packed_grid())?;
        let iq1s_signed_grid = DeviceBuffer::from_slice(&crate::quant::iq1s_signed_grid())?;
        let iq2xxs_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq2xxs_packed_grid())?;
        let iq2xxs_signs_packed = DeviceBuffer::from_slice(
            &[
                crate::quant::iq2xxs_packed_signs(),
                crate::quant::iq2xxs_sign_masks(),
            ]
            .concat(),
        )?;
        let iq2s_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq2s_packed_grid())?;
        let iq2s_signs_packed = DeviceBuffer::from_slice(
            &[
                crate::quant::iq2s_packed_signs(),
                crate::quant::iq2s_sign_masks(),
            ]
            .concat(),
        )?;
        let iq2xs_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq2xs_packed_grid())?;
        let iq3xxs_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq3xxs_packed_grid())?;
        let iq3s_grid_packed = DeviceBuffer::from_slice(&crate::quant::iq3s_packed_grid())?;
        let iota: Vec<i32> = (0..8).collect();
        let iota8 = DeviceBuffer::from_slice(&iota)?;

        residency::vram_mark("kernels compiled and loaded");
        Ok(DeviceBackend {
            stream,
            matmul,
            matmul_tc,
            matmul_f16w,
            matmul_tc_f16w,
            matvec,
            q8_dp4a,
            q8_mma,
            q8_qmma,
            q8_qmma_deep,
            q8_qmma_split: RefCell::new(HashMap::new()),
            q8_qmma_narrow: RefCell::new(None),
            q8_split,
            q8_qdot,
            q8_qdot_add,
            q8_qdot_persist: RefCell::new(HashMap::new()),
            persist_blocks: Cell::new(0),
            persist_qdot: std::env::var_os("PHOBOS_PERSIST_QDOT").is_some(),
            iq1s_dp4a: Cell::new(env_flag_on("PHOBOS_IQ1S_DP4A")),
            qmma_split: env_flag("PHOBOS_QMMA_SPLIT"),
            qmma_narrow: env_flag("PHOBOS_QMMA_NARROW"),
            fused_plans: RefCell::new(HashMap::new()),
            fused_mlp: fused_stage("PHOBOS_FUSED_MLP"),
            fused_project: fused_stage("PHOBOS_FUSED_PROJ"),
            fused_mix: fused_stage("PHOBOS_FUSED_MIX"),
            fused_attn_out: fused_stage("PHOBOS_FUSED_ATTN_OUT"),
            fused_store2d: fused_stage("PHOBOS_FUSED_STORE2D"),
            fused_blocks: Cell::new(0),
            fused_scratch: RefCell::new(Vec::new()),
            fused_barrier: RefCell::new(None),
            fused_operands: RefCell::new(Vec::new()),
            functions: RefCell::new(HashMap::new()),
            func_shared: RefCell::new(HashMap::new()),
            func_threads: RefCell::new(HashMap::new()),
            eager: RefCell::new(Recorded::default()),
            recording: Cell::new(false),
            flushed: Cell::new(false),
            pending: RefCell::new(Vec::new()),
            recorded_len: Cell::new(0),
            pass: RefCell::new(None),
            // The fourth replay by default: past the prefill and past the
            // warmup passes whose scratch arenas are still growing.
            report_pass: Cell::new(match std::env::var("PHOBOS_PASS_REPORT") {
                Ok(v) if v.is_empty() => 4,
                Ok(v) => v.parse().unwrap_or(4),
                Err(_) => 0,
            }),
            report: RefCell::new(Vec::new()),
            reported: Cell::new(0),
            quantize,
            quantize_wide,
            pointwise,
            pointwise_wide,
            norms: RefCell::new(HashMap::new()),
            quant_norms: RefCell::new(HashMap::new()),
            gated_norms: RefCell::new(HashMap::new()),
            gated_swiglu: RefCell::new(HashMap::new()),
            swiglu_planes: RefCell::new(HashMap::new()),
            deltas: RefCell::new(HashMap::new()),
            chunks: RefCell::new(HashMap::new()),
            identities: RefCell::new(HashMap::new()),
            convs: RefCell::new(HashMap::new()),
            gates: RefCell::new(HashMap::new()),
            splits: RefCell::new(HashMap::new()),
            store_pairs: RefCell::new(HashMap::new()),
            ropes: RefCell::new(HashMap::new()),
            rope_gathers: RefCell::new(HashMap::new()),
            attentions: RefCell::new(HashMap::new()),
            split_attn: RefCell::new(HashMap::new()),
            attn_persist: !matches!(
                std::env::var("PHOBOS_ATTN_PERSIST").as_deref(),
                Ok("0" | "off" | "no" | "false")
            ),
            attn_persist_modules: RefCell::new(HashMap::new()),
            attn_partials: RefCell::new(None),
            readback: RefCell::new(None),
            argmax_reduce: RefCell::new(HashMap::new()),
            argmax_iota: RefCell::new(HashMap::new()),
            argmax_finish,
            argmax_scratch: RefCell::new(None),
            blocked: RefCell::new(HashMap::new()),
            attn_gemm: RefCell::new(HashMap::new()),
            slots: RefCell::new(Vec::new()),
            free_slots: RefCell::new(Vec::new()),
            pool: Pool::new(),
            dense_scratch: [const { Cell::new(None) }; 2],
            drop_scratch: Cell::new(false),
            pass_marks: Cell::new(0),
            alloc_hist: RefCell::new(HashMap::new()),
            constants: RefCell::new(HashMap::new()),
            quants: RefCell::new(Vec::new()),
            q_constants: RefCell::new(HashMap::new()),
            raw_quants: RefCell::new(Vec::new()),
            arena: arena::Arena::default(),
            hot: arena::Arena::with_slab(arena::HOT_SLAB_BYTES),
            state_arena: arena::Arena::default(),
            state_live: Cell::new(0),
            state_arena_on: env_flag("PHOBOS_STATE_ARENA"),
            arena_const: env_flag("PHOBOS_ARENA_CONST"),
            arena_weights: env_flag_on("PHOBOS_ARENA"),
            owned_quants: RefCell::new(Vec::new()),
            owned_raw: RefCell::new(Vec::new()),
            raw_constants: RefCell::new(HashMap::new()),
            q2k_matvec,
            q3k_matvec,
            iq1s_matvec,
            iq2xxs_matvec,
            iq1m_matvec,
            iq2s_matvec,
            iq2xs_matvec,
            iq3xxs_matvec,
            iq3s_matvec,
            iq4xs_matvec,
            iq1s_dequant,
            iq1s_qdecode,
            iq2xxs_qdecode,
            iq1m_qdecode,
            iq2s_qdecode,
            iq2xs_qdecode,
            iq3xxs_qdecode,
            iq3s_qdecode,
            iq1s_qdecode_f16,
            iq2xxs_qdecode_f16,
            iq1m_qdecode_f16,
            iq2s_qdecode_f16,
            iq2xs_qdecode_f16,
            iq3xxs_qdecode_f16,
            iq3s_qdecode_f16,
            iq2xxs_dequant,
            iq1m_dequant,
            iq2s_dequant,
            iq2xs_dequant,
            iq3xxs_dequant,
            iq3s_dequant,
            iq4xs_dequant,
            q2k_dequant,
            iq1s_qdot_matvec,
            iq1s_qdot_i8,
            iq3s_qdot_i8,
            iq3xxs_qdot_i8,
            iq2xxs_qdot_i8,
            iq1m_qdot_i8,
            iq2xs_qdot_i8,
            iq2s_qdot_i8,
            iq2xxs_qdot_matvec,
            iq1m_qdot_matvec,
            iq2s_qdot_matvec,
            iq2xs_qdot_matvec,
            iq3xxs_qdot_matvec,
            iq3s_qdot_matvec,
            iq4xs_qdot_matvec,
            q2k_qdot_matvec,
            q3k_qdot_matvec,
            iq1s_grid,
            iq2xxs_grid,
            iq2xxs_signs_packed,
            iq2s_grid,
            iq2s_signs_packed,
            iq2xs_grid,
            iq3xxs_grid,
            iq3s_grid,
            iq4xs_codebook,
            iq1s_grid_packed,
            iq1s_signed_grid,
            iq1s_qmma,
            iq2xxs_qmma,
            iq2s_qmma,
            iq2xs_qmma,
            raw_qmma: Cell::new(env_flag("PHOBOS_RAW_QMMA")),
            dense_scratch_shared: env_flag_on("PHOBOS_DENSE_SCRATCH"),
            trim_after_dense: env_flag("PHOBOS_TRIM"),
            iq2xxs_grid_packed,
            iq2xxs_signs,
            iq2s_grid_packed,
            iq2s_signs,
            iq2xs_grid_packed,
            iq3xxs_grid_packed,
            iq3s_grid_packed,
            iota8,
            act_scratch: RefCell::new(Vec::new()),
            act_next: Cell::new(0),
            split_scratch: RefCell::new(None),
            _ctx,
        })
    }
}
