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
    Attn, Backend, Buf, DeltaMix, Fused, FusedAttnOut, FusedMlp, FusedMlpRaw, FusedProject,
    HADAMARD_BLOCK, HBuf, HPlane, HeadPerm, Packed, Plane, Q8_BLOCK, QAct, QBuf, RawBuf, Rope,
};

mod arena;
mod experts;
mod argmax;
mod attn;
mod backend;
mod delta;
mod dense;
mod elem;
mod formats;
mod fused;
mod graph;
mod hadamard;
use graph::{PassGraph, PassOp, Recorded};
mod init;
mod kernels;
mod launch;
mod matmul;
mod matmul_f32;
mod mem;
mod norm;
mod q50;
mod qmma_raw;
mod raw;
mod residency;

use fused::*;

// The kernel catalog is one flat namespace of sources and tile sizes, used
// unqualified below. See kernels/mod.rs.
use kernels::*;

pub use kernels::{ATTN_GEMM_TILE, ATTN_SOFT_TILE, attn_gemm_src};

/// The formats `PHOBOS_RAW_QMMA` enables for the fused prompt projection.
/// It takes a comma-separated list of format names, or a plain on/off flag.
/// Unset means all of them.
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
        return if env_flag_on("PHOBOS_RAW_QMMA") { ALL.to_vec() } else { Vec::new() };
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

/// The tree's two toggle spellings, see `ENV.md`.
use phobos_base::env::{flag as env_flag, flag_on as env_flag_on};

/// The buffers behind a [`DeviceQuant`] or [`DeviceRaw`] outside the arena.
/// Held only to keep the allocation alive.
type OwnedQuant = (DeviceBuffer<i8>, DeviceBuffer<f32>, DeviceBuffer<f32>);
type OwnedRaw = (
    DeviceBuffer<i8>,
    DeviceBuffer<u16>,
    Option<DeviceBuffer<u16>>,
);

/// Quant bytes from which a Q8_0-family weight goes into the bulk arena
/// rather than allocations of its own: a 4096 by 13696 down projection, not
/// the attention projections of the small models.
const BULK_QUANT_BYTES: usize = 32 << 20;

/// A device-resident Q8_0 weight: the addresses of its int8 blocks and its
/// two scale layouts, and its output width. Raw pointers, because the
/// [`arena::Arena`] or `owned_quants` owns the memory.
struct DeviceQuant {
    qs: u64,
    scales: u64,
    row_scales: u64,
    n: usize,
    /// `qs` holds Q5_0 blocks less their scale rather than one byte an
    /// element; see `q50.rs`.
    q50: bool,
}

use raw::DeviceRaw;

/// Output width, `k`, and split count: the parameters of
/// [`kernels::q8_qmma_split_src`].
type QmmaSplitKey = (usize, usize, usize);

/// Heads, grouped heads, head dimension, taps, head stride, normalize, query
/// scale and rows per program: everything [`delta_conv_src`] bakes in. The
/// row count is a launch parameter, so a new prompt length compiles nothing.
type ConvKey = (usize, usize, usize, usize, usize, bool, u32, usize);

/// Head count, group size, head dimension and query group: the shape
/// [`DeviceBackend::attn_persist_plan`] settles a grid and split count for.
type AttnPersistKey = (usize, usize, usize, usize);

/// The compiled module for one [`AttnPersistKey`], with its settled block
/// count and split count.
type AttnPersistEntry = (Module, u32, usize);

/// A device-resident backend for GGUF models. A decode step stays in device
/// memory and synchronizes once, to read the logits.
pub struct DeviceBackend {
    stream: Stream,
    /// Copies that run beside the compute stream, such as prefetched
    /// experts. See `experts/`.
    copy_stream: Stream,
    /// Whether the MoE path runs the next block's router early on the
    /// current residual and prefetches the predicted experts.
    /// `PHOBOS_MOE_LOOKAHEAD` opts in.
    moe_lookahead: bool,
    /// Whether a prompt pass runs the lightest experts on the host while the
    /// device runs the rest. `PHOBOS_MOE_HOST=0` opts out.
    moe_host: bool,
    /// Whether a decode step computes cache misses on the host while the
    /// device runs the hits. A zero slot stands in for each miss.
    /// `PHOBOS_MOE_HOST_DECODE=0` opts out.
    moe_host_decode: bool,
    /// Whether a prompt pass runs its routed feed-forward as grouped GEMMs
    /// over rows sorted by expert. `PHOBOS_MOE_GROUPED=0` opts out.
    moe_grouped: bool,
    matmul: Variants,
    /// The tensor-core matmul for whole tiles when `m >= TC_TILE_M`. The
    /// plain `matmul` above handles the remainder.
    matmul_tc: Module,
    /// The same pair over an f16 weight, such as a `_qdecode` strip. See
    /// `matmul_f16w_src`.
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
    /// The narrow-CTA variant of the deep tile: half the threads and column
    /// tile, same per-warp patch. Compiled lazily when
    /// [`Self::qmma_narrow`] is set.
    q8_qmma_narrow: RefCell<Option<Module>>,
    q8_split: Variants,
    q8_qdot: Module,
    q8_qdot_add: Module,
    /// The persistent matvec, keyed by iteration count and whether it
    /// accumulates. Only built with `PHOBOS_PERSIST_QDOT`. See
    /// [`q8_qdot_persist_src`].
    q8_qdot_persist: RefCell<HashMap<(usize, bool), Module>>,
    /// The Q5_0 projections, compiled on first use: `q50_qmma` keyed by its
    /// tile's depth and width, `q50_qdot` by zero and whether it adds.
    q50_kernels: RefCell<HashMap<(usize, usize), Module>>,
    /// Blocks a persistent matvec may use, from the occupancy API. Zero until
    /// first compiled.
    persist_blocks: Cell<u32>,
    persist_qdot: bool,
    /// Whether the raw decode matvecs contract in `dp4a` against an int8
    /// activation. `PHOBOS_IQ1S_DP4A=0` opts out.
    ///
    /// The host reference does not quantize the activation, so it cannot
    /// judge this path. `backend_check` compares both device paths instead.
    iq1s_dp4a: Cell<bool>,
    /// Whether `q8_qmma`'s deep tile takes the split-K path on a starved grid.
    /// On by default; `PHOBOS_QMMA_SPLIT=0` opts out.
    qmma_split: bool,
    /// Whether `q8_qmma`'s deep tile takes the narrow-CTA path.
    /// `PHOBOS_QMMA_NARROW=1` opts in. See
    /// [`kernels::q8_qmma_narrow_eligible`].
    qmma_narrow: bool,
    /// Fused kernels the pass has emitted, each with the plan that says what
    /// to bind. See [`fuse`].
    fused_plans: RefCell<HashMap<ChainKey, (Module, Plan)>>,
    fused_mlp: bool,
    fused_project: bool,
    /// Whether the delta net's convolution and gates join the projection's
    /// fused kernel. Separate from [`Self::fused_project`], since it adds a
    /// barrier.
    fused_mix: bool,
    /// Whether attention's output epilogue (quantizing the mixed heads, then
    /// the output projection) joins the fused-chain path. Default on.
    fused_attn_out: bool,
    /// Whether attention's key and value writes into the cache land in one
    /// launch instead of two. Default on, `PHOBOS_FUSED_STORE2D=0` off.
    fused_store2d: bool,
    /// Blocks a fused kernel is launched with. Unlike [`persist_blocks`] this
    /// must be exact, since a non-resident block never reaches the barrier.
    /// Zero until the first fused kernel is compiled.
    fused_blocks: Cell<u32>,
    /// Storage for values that cross a loop nest in a fused kernel, by the
    /// plan's scratch index. Each only grows.
    fused_scratch: RefCell<Vec<(DeviceBuffer<i8>, DeviceBuffer<f32>)>>,
    /// A fused kernel's arrival counter and release generation, zeroed once.
    /// The barrier restores both, so every launch reuses them.
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
    /// The recorded pass's graphs, one per segment. Each sync point starts a
    /// new segment.
    pass: RefCell<Vec<PassGraph>>,
    /// Segments replayed so far in the current pass.
    segment: Cell<usize>,
    /// Replays left until the one to report on, and that pass's launches.
    /// Zero when `PHOBOS_PASS_REPORT` is unset.
    report_pass: Cell<usize>,
    report: RefCell<Vec<PassOp>>,
    /// Replays so far, so the report can name its replay.
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
    /// SwiGLU over two planes of a wider buffer, keyed by tile width.
    swiglu_planes: RefCell<HashMap<usize, Module>>,
    /// Delta rule kernels, keyed by head count and head dimension. Both are
    /// compile-time tile extents.
    deltas: RefCell<HashMap<(usize, usize), Module>>,
    chunks: RefCell<HashMap<(usize, usize), Module>>,
    identities: RefCell<HashMap<usize, DeviceBuffer<f32>>>,
    /// Delta convolution kernels, keyed by [`ConvKey`].
    convs: RefCell<HashMap<ConvKey, Module>>,
    /// Gate kernels, keyed by head count.
    gates: RefCell<HashMap<usize, Module>>,
    /// Hadamard transforms, keyed by width, head regrouping and form.
    hadamards: RefCell<HashMap<HadamardKey, Module>>,
    /// The row-major dense contractions, by tile rows and outputs.
    rows_matmuls: RefCell<HashMap<RowsKernel, Module>>,
    /// Strided copy kernels, keyed by direction, copy width, and whether the
    /// pitches are aligned to the width.
    splits: RefCell<HashMap<(Strided, usize, bool), Module>>,
    /// [`Backend::store_2d_pair`] kernels, keyed by width and whether all
    /// pitches are aligned to it.
    store_pairs: RefCell<HashMap<(usize, bool), Module>>,
    /// Rotary kernels, keyed by head count and half the rotary width.
    ropes: RefCell<HashMap<(usize, usize), Module>>,
    /// [`Backend::rope_gather`] kernels, keyed by head count, half the
    /// rotary width, the source's heads per row, and the head dimension.
    rope_gathers: RefCell<HashMap<(usize, usize, usize, usize), Module>>,
    /// Attention kernels, keyed by query heads, group size and head dimension.
    attentions: RefCell<HashMap<(usize, usize, usize), Module>>,
    split_attn: RefCell<HashMap<(usize, usize, usize), Module>>,
    /// Whether [`DeviceBackend::attention_decode`] takes the persistent path.
    /// Default on, `PHOBOS_ATTN_PERSIST=0` turns it off. Not controlled by
    /// `PHOBOS_FUSED`.
    attn_persist: bool,
    /// The persistent attention kernel, keyed by [`AttnPersistKey`], with its
    /// settled block and split count. `None` means the persistent path was
    /// declined for that shape.
    attn_persist_modules: RefCell<HashMap<AttnPersistKey, Option<AttnPersistEntry>>>,
    /// Per-split partial accumulators, and their running maxima and sums.
    attn_partials: RefCell<Option<(DeviceBuffer<f32>, DeviceBuffer<f32>)>>,
    /// Page-locked staging for the one readback a pass makes.
    readback: RefCell<Option<LockedBuffer<f32>>>,
    /// [`DeviceBackend::argmax`]'s reduction kernel, keyed by chunk width
    /// ([`argmax_chunk_width`]).
    argmax_reduce: RefCell<HashMap<usize, Module>>,
    /// `[0, W)` as floats, added to a chunk's base index to recover a
    /// vocabulary position. Keyed by chunk width.
    argmax_iota: RefCell<HashMap<usize, DeviceBuffer<f32>>>,
    /// [`argmax_finish_src`], compiled once since it has no shape parameter.
    argmax_finish: Module,
    /// [`argmax_reduce`]'s per-block partials and [`argmax_finish`]'s
    /// result, reused across calls.
    argmax_scratch: RefCell<Option<(DeviceBuffer<f32>, DeviceBuffer<f32>)>>,
    /// The blocked attention kernels, keyed like `attentions`.
    blocked: RefCell<HashMap<(usize, usize, usize), Module>>,
    attn_gemm: RefCell<HashMap<usize, Module>>,
    /// Addressed by [`Buf`]; a slot is `None` while free.
    slots: RefCell<Vec<Option<mem::Slot>>>,
    free_slots: RefCell<Vec<usize>>,
    /// Released allocations, reused instead of returned to the driver. See
    /// [`Backend::alloc`], [`DeviceBackend::alloc_written_now`].
    pool: Pool,
    /// Module cache hits and compiles. See [`DeviceBackend::with_kernel`].
    kernels_reused: Cell<u64>,
    kernels_compiled: Cell<u64>,
    /// The weight strip a prompt pass dequantizes into, and the matmul's
    /// output. One of each for the whole model. See
    /// [`DeviceBackend::dense_scratch`].
    dense_scratch: [Cell<Option<Buf>>; 2],
    /// Set by a pass that used the prompt scratch, cleared by the next trim.
    /// See [`DeviceBackend::trim_after_dense`].
    drop_scratch: Cell<bool>,
    /// Rows of the pass before this one, for [`DeviceBackend::trim_after_prompt`].
    last_rows: Cell<usize>,
    /// Passes begun, so [`DeviceBackend::mark_pass_vram`] can thin its output.
    pass_marks: Cell<usize>,
    /// Outstanding allocations per length. Only tracked with `PHOBOS_VRAM`.
    alloc_hist: RefCell<HashMap<usize, isize>>,
    constants: RefCell<HashMap<String, Buf>>,
    /// The streamed experts. See `experts/`.
    experts: RefCell<experts::Experts>,
    expert_keys: RefCell<HashMap<String, crate::backend::ExpertsBuf>>,
    /// The MoE kernels. Top-k is keyed by expert count. The slot matvec is
    /// keyed by format, width, and whether every program reads activation
    /// row zero.
    moe_topk: RefCell<HashMap<usize, Module>>,
    moe_qdot: RefCell<HashMap<(&'static str, usize, bool), Module>>,
    /// Gate, up and the SwiGLU in one, by format and width.
    moe_gateup: RefCell<HashMap<(&'static str, usize), Module>>,
    moe_combine: RefCell<HashMap<(), Module>>,
    /// Adds the host-computed share of a decode row into the device result.
    host_add: RefCell<Option<Module>>,
    /// The grouped prompt path's kernels: the row permutation, the GEMM
    /// (keyed by format and width), and the gathering combine.
    moe_permute: RefCell<HashMap<(&'static str, usize), Module>>,
    moe_qgemm: RefCell<HashMap<(&'static str, usize), Module>>,
    moe_gather_add: RefCell<HashMap<(), Module>>,
    moe_shared: RefCell<HashMap<(), Module>>,
    /// Q8_0 weights, addressed by [`QBuf`]. Never released.
    quants: RefCell<Vec<DeviceQuant>>,
    q_constants: RefCell<HashMap<String, QBuf>>,
    /// Raw-format weights, addressed by [`RawBuf`].
    raw_quants: RefCell<Vec<DeviceRaw>>,
    /// The slabs the bulk weights live in.
    arena: arena::Arena,
    /// Smaller slabs for the small, hot constants. See
    /// [`arena::HOT_SLAB_BYTES`].
    hot: arena::Arena,
    /// Slabs for recurrent state. `state_live` counts the live regions, and
    /// the slabs are freed when it reaches zero.
    state_arena: arena::Arena,
    state_live: Cell<usize>,
    /// Whether recurrent state goes in `state_arena`. State is resident
    /// anyway, so the arena brings no benefit. `PHOBOS_STATE_ARENA=1` turns
    /// it on.
    state_arena_on: bool,
    /// Whether the small, hot constants go in the bulk arena instead of
    /// `hot`. A hot plane in a bulk slab keeps the whole slab resident.
    /// `PHOBOS_ARENA_CONST=1` turns it on.
    arena_const: bool,
    /// Whether the bulk weights go in the arena. `PHOBOS_ARENA=0` gives each
    /// tensor its own allocation.
    arena_weights: bool,
    /// Buffers of constants outside the arena, kept alive so the pointers in
    /// [`DeviceQuant`] and [`DeviceRaw`] stay valid.
    owned_quants: RefCell<Vec<OwnedQuant>>,
    owned_raw: RefCell<Vec<OwnedRaw>>,
    raw_constants: RefCell<HashMap<String, RawBuf>>,
    /// The raw-format kernels, by format, for the formats the model holds.
    /// Never changed after construction, since a launch keys its function
    /// cache on the module's address.
    formats: HashMap<Quant, formats::FormatKernels>,
    /// IQ1_S's grid, [`crate::quant::iq1s_flat_grid`] at one `i32` per
    /// lane. Shared by every IQ1_S and IQ1_M weight.
    iq1s_grid: DeviceBuffer<i32>,
    /// IQ2_XXS's magnitude grid and sign table, flattened like IQ1_S's.
    iq2xxs_grid: DeviceBuffer<i32>,
    iq2xxs_signs: DeviceBuffer<i32>,
    /// IQ2_S's magnitude grid and sign table. The signs are keyed by raw
    /// byte rather than parity index, unlike IQ2_XXS's.
    iq2s_grid: DeviceBuffer<i32>,
    iq2s_signs: DeviceBuffer<i32>,
    /// IQ2_XS's magnitude grid. Its signs use `iq2xxs_signs`.
    iq2xs_grid: DeviceBuffer<i32>,
    /// IQ3_XXS's magnitude grid, four `i32` lanes per entry. Its signs use
    /// `iq2xxs_signs`.
    iq3xxs_grid: DeviceBuffer<i32>,
    /// IQ3_S's magnitude grid, four `i32` lanes per entry. Its signs use
    /// `iq2s_signs`.
    iq3s_grid: DeviceBuffer<i32>,
    /// IQ4_XS's fixed sixteen-value codebook, indexed directly by each
    /// nibble. See [`crate::quant::iq4xs_flat_codebook`].
    iq4xs_codebook: DeviceBuffer<i32>,
    /// The same tables at one `i8` per slot, read by every `*_qdot_t` and
    /// `*_qdecode_t`. The i32 copies above serve the `gather`-based
    /// fallbacks.
    iq1s_grid_packed: DeviceBuffer<i8>,
    /// The IQ1_S grid with the delta and signs folded in, giving an exact
    /// `i8` weight. See `quant::iq1s_signed_grid`.
    iq1s_signed_grid: DeviceBuffer<i8>,
    /// Whether the fused prompt projection is used. On by default,
    /// `PHOBOS_RAW_QMMA=0` turns it off.
    raw_qmma: Cell<bool>,
    /// Which formats [`raw_qmma`] covers, from `PHOBOS_RAW_QMMA`, which also
    /// takes a comma-separated list such as `iq1s,iq2xxs`.
    raw_qmma_formats: Vec<Quant>,
    /// The staged prompt projections. See `qmma_raw.rs`.
    qgemm: qmma_raw::Qgemm,
    /// `PHOBOS_DENSE_SCRATCH=0` uses a pooled scratch pair per weight
    /// instead of one shared pair. `PHOBOS_TRIM=1` also frees the scratch
    /// and the pool's free list after a dense pass. See `residency.rs`.
    dense_scratch_shared: bool,
    trim_after_dense: bool,
    iq2xxs_grid_packed: DeviceBuffer<i8>,
    /// The signs as +/-1 for the float decode, followed by the same signs as
    /// a 0/-1 mask for the dp4a decode.
    iq2xxs_signs_packed: DeviceBuffer<i8>,
    iq2s_grid_packed: DeviceBuffer<i8>,
    iq2s_signs_packed: DeviceBuffer<i8>,
    iq2xs_grid_packed: DeviceBuffer<i8>,
    iq3xxs_grid_packed: DeviceBuffer<i8>,
    iq3s_grid_packed: DeviceBuffer<i8>,
    /// `[0, 1, .., 7]`, the offsets a grid-coded raw kernel's `gather`
    /// broadcasts against to read an eight-wide lane in one call.
    iota8: DeviceBuffer<i32>,
    /// One entry per `quantize_act` in a pass, each grown to the largest
    /// projection seen. Several are live at once, and a pass takes them in
    /// the same order every time.
    act_scratch: RefCell<Vec<(DeviceBuffer<i8>, DeviceBuffer<f32>)>>,
    /// Which of the first few `act_scratch` slots the next transient
    /// activation takes. See [`DeviceBackend::act_slot_transient`].
    act_ring: Cell<usize>,
    /// The next slot of the shared ring, see [`DeviceBackend::act_slot_shared`].
    act_shared: Cell<usize>,
    act_next: Cell<usize>,
    /// Split-K partial sums, `[splits, n]`. Only grows.
    split_scratch: RefCell<Option<DeviceBuffer<f32>>>,
    /// Must be last. Fields drop in declaration order, and every allocation
    /// above must be freed while the context is alive.
    _ctx: cust::context::Context,
}

impl DeviceBackend {
    /// Turns the `dp4a` decode matvecs on or off, so a check can compare
    /// both paths in one session. The host reference does not quantize the
    /// activation, so it cannot judge this path.
    pub fn set_iq_dp4a(&self, on: bool) {
        self.iq1s_dp4a.set(on);
    }

    /// The same for the fused prompt projection, which also quantizes its
    /// activation.
    pub fn set_raw_qmma(&self, on: bool) {
        self.raw_qmma.set(on);
    }

    /// Turns every fused stage on or off, overriding the environment, so one
    /// process can compare both paths. See `examples/fuse_check.rs`.
    pub fn set_fused(&mut self, on: bool) {
        self.fused_mlp = on;
        self.fused_project = on;
        self.fused_mix = on;
        self.fused_attn_out = on;
        self.fused_store2d = on;
    }
}
