use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;

use anyhow::{Context, Result, ensure};
use cust::memory::{CopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

use phobos_base::half::f16_to_f32;
use phobos_kernels::pool::Pool;
use phobos_kernels::{Variants, compile, compile_shared, cuda_ok, push_descriptor};

use super::fuse::{self, Bound, Chain, ChainKey, Plan, Scratch};
use phobos_kernels::launch::{CTA_THREADS, STATIC_SHARED_LIMIT, persistent_grid};

use super::{
    Attn, Backend, Buf, DeltaMix, Fused, FusedAttnOut, FusedMlp, FusedProject, HBuf, HPlane,
    Packed, Plane, Q8_BLOCK, QAct, QBuf, Rope,
};

mod argmax;
mod attn;
mod backend;
mod delta;
mod elem;
mod fused;
mod graph;
mod kernels;
mod launch;
mod matmul;
mod mem;

use fused::*;

// The catalog is written as one flat namespace of sources and tile sizes, and
// the impls below reach for them unqualified; see kernels/mod.rs.
use kernels::*;

pub use kernels::{ATTN_GEMM_TILE, ATTN_SOFT_TILE, attn_gemm_src};

/// A device-resident Q8_0 weight: signed bytes, per-block scales in both
/// orders, and the output width it was uploaded with. `q8_mma` wants
/// `[block, out]` and `qdot_t` wants `[out, block]`, neither can cheaply
/// transpose, and a scale is one f32 per 32 bytes, so the duplicate is cheap.
type DeviceQuant = (
    DeviceBuffer<i8>,
    DeviceBuffer<f32>,
    DeviceBuffer<f32>,
    usize,
);

/// Heads, head dimension, taps, head stride, normalize, query scale, rows, and
/// rows per program: everything [`delta_conv_src`] bakes in.
type ConvKey = (usize, usize, usize, usize, bool, u32, usize, usize);

/// Head count, group size, head dimension and query group: the shape
/// [`DeviceBackend::attn_persist_plan`] settles a grid and a split count for.
/// The split count is not part of the key: it is derived from the settled
/// grid (as many whole units as the grid holds exactly), not chosen by the
/// caller, so it is already a function of the other four fields plus the
/// device's own occupancy answer.
type AttnPersistKey = (usize, usize, usize, usize);

/// The compiled module for one [`AttnPersistKey`], with the block count and
/// the persist-specific split count [`DeviceBackend::attn_persist_plan`]
/// settled on.
type AttnPersistEntry = (Module, u32, usize);

#[derive(Default)]
struct Recorded {
    func: cust::sys::CUfunction,
    grid: (u32, u32, u32),
    /// Zero for the kernels whose tiles are static globals.
    shared: u32,
    threads: u32,
    /// The exploded-memref ABI's argument words, one per kernel parameter.
    /// See [`push_descriptor`]
    slots: Vec<u64>,
}

impl Recorded {
    fn params(&self, argv: &mut Vec<*mut c_void>) -> cust::sys::CUDA_KERNEL_NODE_PARAMS {
        // Borrows slots for the pointer array. The driver copies the values out
        // during the call, so neither outlives it.
        argv.clear();
        argv.extend(
            self.slots
                .iter()
                .map(|s| s as *const u64 as *mut u64 as *mut c_void),
        );
        cust::sys::CUDA_KERNEL_NODE_PARAMS {
            func: self.func,
            gridDimX: self.grid.0,
            gridDimY: self.grid.1,
            gridDimZ: self.grid.2,
            blockDimX: self.threads,
            blockDimY: 1,
            blockDimZ: 1,
            sharedMemBytes: self.shared,
            kernelParams: argv.as_mut_ptr(),
            extra: std::ptr::null_mut(),
        }
    }

    fn same(&self, other: &Recorded) -> bool {
        self.func == other.func && self.grid == other.grid && self.slots == other.slots
    }
}

/// enabled via `PHOBOS_PASS_REPORT`
struct PassOp {
    name: &'static str,
    func: cust::sys::CUfunction,
    blocks: u32,
    threads: u32,
    shared: u32,
}

struct PassGraph {
    graph: cust::sys::CUgraph,
    exec: cust::sys::CUgraphExec,
    nodes: Vec<cust::sys::CUgraphNode>,
    recorded: Vec<Recorded>,
}

impl Drop for PassGraph {
    fn drop(&mut self) {
        // SAFETY: both handles were created by this type and are dropped once.
        unsafe {
            cust::sys::cuGraphExecDestroy(self.exec);
            cust::sys::cuGraphDestroy(self.graph);
        }
    }
}

/// A device-resident backend for GGUF models. A whole decode step stays in
/// device memory: the residual stream, the projections, the pointwise work and
/// both mixers. A step synchronizes once, to read the logits.
pub struct DeviceBackend {
    stream: Stream,
    matmul: Variants,
    matvec: Variants,
    q8_dp4a: Variants,
    q8_mma: Variants,
    q8_qmma: Module,
    q8_qmma_deep: HashMap<usize, Module>,
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
    /// Fused kernels the pass has emitted, with the plan that says what to bind
    /// to each. See [`fuse`].
    fused_plans: RefCell<HashMap<ChainKey, (Module, Plan)>>,
    fused_mlp: bool,
    fused_project: bool,
    /// Whether the delta net's convolution and gates join the projection's
    /// kernel. Separate from [`Self::fused_project`] because unlike everything
    /// fused so far this one costs a barrier, so it has to be able to be
    /// measured against the projection alone.
    fused_mix: bool,
    /// Whether attention's output epilogue (quantizing the mixed heads, then
    /// the output projection) joins the fused-chain path. Measured a wash
    /// against the launched pair before `warp_partial` cut attention's own
    /// share of a step; re-measured after and a clean win
    /// (`autoresearch/beams/launch-bound-headroom.md`), so this now joins
    /// the default-on `fused_stage` trio above. `PHOBOS_FUSED_ATTN_OUT=0`
    /// switches it off.
    fused_attn_out: bool,
    /// Whether attention's key and value writes into the cache land in one
    /// launch instead of two. Same history as [`Self::fused_attn_out`]:
    /// default on, `PHOBOS_FUSED_STORE2D=0` switches it off.
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
    /// Attention kernels, keyed by query heads, group size and head dimension.
    attentions: RefCell<HashMap<(usize, usize, usize), Module>>,
    split_attn: RefCell<HashMap<(usize, usize, usize), Module>>,
    /// Whether [`DeviceBackend::attention_decode`] takes the persistent
    /// split-plus-merge path. `PHOBOS_ATTN_PERSIST`, unset by default: see
    /// that function's doc comment for why this is opt-in rather than a
    /// `fused_stage`-style default-on switch.
    attn_persist: bool,
    /// The persistent attention kernel, keyed by [`AttnPersistKey`], with the
    /// block count and the persist-specific split count it was settled at.
    /// `None` means the shape's occupancy could not fit the split phase in
    /// one grid-strided pass and the persistent path is declined for it. See
    /// `DeviceBackend::attn_persist_plan` in `attn.rs`.
    attn_persist_modules: RefCell<HashMap<AttnPersistKey, Option<AttnPersistEntry>>>,
    /// Per-split partial accumulators, and their running maxima and sums.
    attn_partials: RefCell<Option<(DeviceBuffer<f32>, DeviceBuffer<f32>)>>,
    /// Page-locked staging for the one readback a pass makes.
    readback: RefCell<Option<LockedBuffer<f32>>>,
    /// [`DeviceBackend::argmax`]'s reduction kernel, keyed by chunk width
    /// ([`argmax_chunk_width`]): every model uses one width for the life of
    /// the backend, since a vocabulary's size does not change, so this stays
    /// a single entry in practice.
    argmax_reduce: RefCell<HashMap<usize, Module>>,
    /// [`argmax_reduce`]'s partial-value column, one per lane, `[0, W)`;
    /// uploaded once per chunk width and added to each chunk's own base
    /// index to map a lane back to a vocabulary position.
    argmax_iota: RefCell<HashMap<usize, DeviceBuffer<f32>>>,
    /// [`ARGMAX_FINISH_SRC`], compiled once: it takes no shape baked in.
    argmax_finish: Module,
    /// [`argmax_reduce`]'s per-block partials and [`argmax_finish`]'s answer.
    /// Both are a handful of floats; reused across calls rather than
    /// allocated fresh, since [`Backend::argmax`] is on the hot decode path.
    argmax_scratch: RefCell<Option<(DeviceBuffer<f32>, DeviceBuffer<f32>)>>,
    /// The blocked attention kernels, keyed the same way.
    blocked: RefCell<HashMap<(usize, usize, usize), Module>>,
    attn_gemm: RefCell<HashMap<usize, Module>>,
    /// Addressed by [`Buf`]; a slot is `None` while free.
    slots: RefCell<Vec<Option<DeviceBuffer<f32>>>>,
    free_slots: RefCell<Vec<usize>>,
    /// Released allocations, handed out again rather than going back to the
    /// driver. See [`Backend::alloc`] and [`DeviceBackend::alloc_written_now`].
    pool: Pool,
    constants: RefCell<HashMap<String, Buf>>,
    /// Addressed by [`QBuf`]: bytes, per-block scales, and the output width
    /// they went up with. Constants, so never released.
    quants: RefCell<Vec<DeviceQuant>>,
    q_constants: RefCell<HashMap<String, QBuf>>,
    /// One entry per `quantize_act` in a pass, each grown to the largest
    /// projection it has seen. Slot by slot rather than one shared arena:
    /// several are live at once, and a pass asks for them in the same order
    /// every time, which keeps their addresses out of the graph's changing
    /// nodes.
    act_scratch: RefCell<Vec<(DeviceBuffer<i8>, DeviceBuffer<f32>)>>,
    act_next: Cell<usize>,
    /// Split-K partial sums, `[splits, n]`, grown to the largest asked for.
    split_scratch: RefCell<Option<DeviceBuffer<f32>>>,
    /// Must be last: Rust drops in declaration order and every allocation
    /// above has to be released while the context is still alive.
    _ctx: cust::context::Context,
}

impl DeviceBackend {
    /// Turns every fused stage on or off whatever the environment asked for.
    /// [`fused_stage`] reads its variables once, at construction, which cannot
    /// express what the equivalence harness needs: both paths in one process,
    /// over one upload of the weights and one context. See
    /// `examples/fuse_check.rs`.
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

        let matmul = Variants::compile(
            MATMUL_SRC,
            &[("TILE_M", TILE_M), ("TILE_N", TILE_N), ("TILE_K", TILE_K)],
            "matmul",
            ("@aligned(M = TILE_M, N = TILE_N)", ""),
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

        Ok(DeviceBackend {
            stream,
            matmul,
            matvec,
            q8_dp4a,
            q8_mma,
            q8_qmma,
            q8_qmma_deep,
            q8_split,
            q8_qdot,
            q8_qdot_add,
            q8_qdot_persist: RefCell::new(HashMap::new()),
            persist_blocks: Cell::new(0),
            persist_qdot: std::env::var_os("PHOBOS_PERSIST_QDOT").is_some(),
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
            attentions: RefCell::new(HashMap::new()),
            split_attn: RefCell::new(HashMap::new()),
            attn_persist: std::env::var_os("PHOBOS_ATTN_PERSIST").is_some(),
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
            constants: RefCell::new(HashMap::new()),
            quants: RefCell::new(Vec::new()),
            q_constants: RefCell::new(HashMap::new()),
            act_scratch: RefCell::new(Vec::new()),
            act_next: Cell::new(0),
            split_scratch: RefCell::new(None),
            _ctx,
        })
    }
}
