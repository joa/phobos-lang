use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;

use anyhow::{Context, Result, ensure};
use cust::memory::{CopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

use phobos_kernels::pool::Pool;
use phobos_kernels::{Variants, compile, compile_shared, cuda_ok, matmul, push_descriptor};

use super::{Attn, Backend, Buf, DeltaMix, Plane, Q8_BLOCK, QAct, QBuf, Rope, check_q8_shape};

/// Output tile and k-slice for the tiled matmul.
const TILE_M: usize = matmul::TILE_M;
const TILE_N: usize = matmul::TILE_N;
const TILE_K: usize = matmul::TILE_K;

/// Output tile along N for the single-row matmul.
const MV_TN: usize = 128;

/// Output tile along N for the quantized single-row matmul.
const Q8_TN: usize = 32;

/// Output tile for the quantized tensor-core matmul. The rows are the depth of
/// the tensor core's m8 fragment, the smallest tile that reaches it at all, and
/// the columns give the CTA's eight warps an 8x8 tile each, so no warp reads a
/// weight byte another warp read.
///
/// Deeper row tiles measured the same on this model and raise the row count a
/// prefill needs to reach the tensor cores at all. They would cut the weight
/// traffic, since every row block re-reads the whole weight tile, but at 512
/// tokens that is a few percent of the pass. They also run out of room: four
/// [TM, TN] f32 temporaries carry the accumulator chain, which at 64 by 64 is
/// the whole of Turing's 64K of shared memory.
const Q8_MMA_TM: usize = 8;
const Q8_MMA_TN: usize = 64;

/// Blocks quantized per CTA in the activation-quantization kernel.
const QUANT_TB: usize = 4;

/// The same for a prompt, whose activation is thousands of blocks rather than
/// thirty-two.
///
/// Four rows is 128 values on a 256-thread CTA, which idles half of them and
/// spends more dispatching the block than running it: a 512-token activation is
/// 16384 rows, so 4096 blocks. Sixteen rows is four times fewer and still two
/// CTAs deep in shared memory, which 64 is not, since the tile chain is six
/// `[TB, 32]` f32 buffers.
const QUANT_TB_WIDE: usize = 16;

/// Elements per block for the pointwise kernels.
const ELEM_TILE: usize = 128;

/// Pointwise elements a program takes when there are enough of them.
///
/// 128 suits a decode step, where the whole call is a few thousand elements and
/// what matters is having any blocks at all. A prompt's SwiGLU is 1.8 million,
/// which at 128 is 14336 blocks each using half a CTA for one tile; at 1024 it
/// is 1792 and the kernel halves. Shared memory is the ceiling, since `swiglu`
/// holds about six tiles at once.
const ELEM_TILE_WIDE: usize = 1024;

/// Below this the narrow tile still wins: enough wide tiles to fill the card
/// several times over.
const WIDE_FLOOR: usize = 192 * ELEM_TILE_WIDE;

use phobos_kernels::launch::{CTA_THREADS, STATIC_SHARED_LIMIT, WARP_THREADS};

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

/// A device-resident Q8_0 weight: signed bytes, per-block scales in both
/// orders, and the output width it was uploaded with.
///
/// Both scale copies exist because the two kernels want opposite orders and
/// neither can cheaply transpose. `q8_mma` reads a row of scales, one per
/// output, to shade a `[TM, TN]` accumulator by column, so it wants
/// `[block, out]`. `qdot_t` has a lane per `k` chunk of one output, so it wants
/// `[out, block]`, and the other order costs it a sector per lane. A scale is
/// one f32 per 32 weight bytes, so the duplicate is an eighth of a weight.
type DeviceQuant = (
    DeviceBuffer<i8>,
    DeviceBuffer<f32>,
    DeviceBuffer<f32>,
    usize,
);

/// Everything [`delta_conv_src`] bakes in: head count and dimension, tap count,
/// the distance between heads, whether the plane normalizes, the scale it
/// carries, and the positions in the call against those a program takes.
type ConvKey = (usize, usize, usize, usize, bool, u32, usize, usize);

const MATMUL_SRC: &str = matmul::TEMPLATE;

/// The single-row specialization decoding needs.
///
/// NOTE: Always reads row zero: a caller wanting row `r` offsets the operand pointers,
/// which keeps the kernel free of scalar arguments.
const MATVEC_SRC: &str = "\
@launch(256)
@autotune(TILE_N in [128], TILE_K in [16])
{ALIGNED}
kernel matvec(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  var acc: tile<f32>[1, TILE_N] = 0.0
  for kt in range(0, K, TILE_K) {
    var a = A[0 :+ 1, kt :+ TILE_K]
    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
  }
  C[0 :+ 1, pn * TILE_N :+ TILE_N] = acc
}
";

/// Quantize an activation to int8 with one scale per block of 32. The
/// activation is viewed as `[blocks, 32]`, so a row is one block and `rowmax`
/// gives its magnitude directly, and the epsilon keeps an all-zero block from
/// dividing by zero.
///
/// The rounding is the hardware's own, ties to even, matching the host
/// reference exactly. Biasing into `[1.5, 255.5]` and truncating gets a
/// round-half-up without a rounding instruction, but adding 128 to a value that
/// can reach 127 costs seven bits of mantissa, enough at the top of the range
/// to carry a value across a boundary.
const QUANTIZE_SRC: &str = "\
@launch(256)
@autotune(TB in [8])
kernel quantize(X: tensor<f32>[R, 32], Q: tensor<i8>[R, 32], S: tensor<f32>[R, D]) {
  let p = program_id(0)
  var x = X[p * TB :+ TB, 0 :+ 32]
  var mx: tile<f32>[TB, 1] = rowmax(tmax(x, -x))
  var inv = 127.0 / (mx + 0.00000001)
  var y = x * inv
  Q[p * TB :+ TB, 0 :+ 32] = i8(i32(round(y)))
  S[p * TB :+ TB, 0 :+ 1] = mx / 127.0
}
";

/// The Q8_0 projection with both operands in int8, contracted by `dp4a`.
///
/// `dot_t` contracts the last axis of both operands, so the activation and the
/// weight row both walk `k` contiguously, which lets four bytes of each pack
/// into one instruction. Nothing is dequantized into shared memory, so the f32
/// weight tile is gone and with it the round trip that made the conversion
/// kernel occupancy bound. The two scales are constant across a block, so they
/// multiply the integer dot once per block per output.
const Q8_DP4A_SRC: &str = "\
@launch(256)
@autotune(TN in [32])
{ALIGNED}
kernel q8_dp4a(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
               C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  var acc: tile<f32>[1, TN] = 0.0
  for kt in range(0, K, 32) {
    let b = kt / 32
    let a = A[0 :+ 1, kt :+ 32]
    let w = W[pn * TN :+ TN, kt :+ 32]
    let ws = WS[b :+ 1, pn * TN :+ TN]
    let da = AS[0 :+ 1, b :+ 1]
    acc += f32(dot_t(a, w)) * ws * da
  }
  C[0 :+ 1, pn * TN :+ TN] = acc
}
";

/// The batched Q8_0 projection, on the integer tensor cores.
///
/// The same contraction as [`Q8_DP4A_SRC`] over an output tile in both
/// directions rather than one row, which is what lets a prefill batch: the rows
/// move from a host loop of launches into the grid. Widening the tile is also
/// what reaches `mma.sync`, whose smallest integer output tile is 8x8, and
/// `dot_t` hands it the four contiguous bytes per operand it wants.
const Q8_MMA_SRC: &str = "\
@launch(256)
@autotune(TM in [8], TN in [64])
{ALIGNED}
kernel q8_mma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
              W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
              C: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, K, 32) {
    let b = kt / 32
    let a = A[pm * TM :+ TM, kt :+ 32]
    let w = W[pn * TN :+ TN, kt :+ 32]
    let ws = WS[b :+ 1, pn * TN :+ TN]
    let da = AS[pm * TM :+ TM, b :+ 1]
    acc += f32(dot_t(a, w)) * ws * da
  }
  C[pm * TM :+ TM, pn * TN :+ TN] = acc
}
";

/// The Q8_0 projection as a single `qmma_t`, which is what a prompt pass runs.
///
/// [`Q8_MMA_SRC`] applies the block scales every 32 elements of `k`, which puts
/// its accumulator in shared memory and stages both operands there per block.
/// That holds it to a 2.3 TOPS no tile shape moves: a `[64, 64]` accumulator is
/// 16 KB on its own and does not build, and every shape that does measures
/// between 2.2 and 2.9.
///
/// `qmma_t` folds the scales in, so the whole of `k` is one operation. The
/// accumulators stay in registers across all of it, the operands are read from
/// global memory in the layout the `m8n8k16` fragments already want, and no
/// barrier is left in the loop. On the same shapes that is 12.6 TOPS, 4.9x,
/// taking the projections of a 512-token pass from 196 ms to 40.
fn q8_qmma_src(block: usize) -> String {
    format!(
        "@launch({block})
@autotune(TM in [64], TN in [64])
@aligned(M = TM, N = TN, K = 32)
kernel q8_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
               C: tensor<f32>[M, N]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm * TM :+ TM, pn * TN :+ TN] = qmma_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                           W[pn * TN :+ TN, :], WS[:, pn * TN :+ TN])
}}
"
    )
}

/// Output tiles of the `qmma_t` projection: the depth, and the widths it comes
/// in.
///
/// What pays for the operand loads and the scale arithmetic is how many tensor
/// core tiles a warp's patch holds, since both are per output element however
/// the tiles are arranged. The patch cannot grow past leaving every warp of the
/// CTA something to do, so the output tile bounds it, and once the result
/// stopped going through shared memory only the register file did: 128 by 256
/// measures 28.1 TOPS against 16.5 for 128 by 64 and 12.6 for 64 by 64.
///
/// A wider tile also shrinks the grid. At 512 rows and a width of 1024 the
/// widest tile is sixteen blocks, a third of the card, and measures worse than
/// the narrow one. [`qmma_width`] picks the width per projection; the depth
/// stays 128, with a 64-deep kernel for the rows a prompt leaves over.
const Q8_QMMA_TM: usize = 128;
const Q8_QMMA_SHALLOW: usize = 64;
const Q8_QMMA_WIDTHS: [usize; 2] = [128, 64];
const Q8_QMMA_TN: usize = 64;

/// Threads the projection's CTA carries: half of every other kernel here.
///
/// A warp of this one is bounded by the register file rather than the block,
/// since a patch is 128 live accumulators whatever the CTA, so a narrower block
/// buys the same warps per multiprocessor on twice the grid. It wins on every
/// shape a pass runs and moves the best tile from 128 by 256 to 128 by 128:
/// 24.91 TOPS to 30.64 on the widest projection, 11.96 to 13.20 on the
/// narrowest.
const Q8_QMMA_CTA: usize = 128;

/// The widest column tile that divides `n`, regardless of how many blocks that
/// leaves.
///
/// At 512 rows a 128-deep tile is four row tiles, so a 1024-wide projection
/// with a 256-wide tile has sixteen blocks on a 48-multiprocessor card, and it
/// still wins: 13.55 TOPS against 11.23 for a 64-wide tile at 64 blocks and
/// 9.87 for 128 wide at 32. The tile buys the warp's patch of tensor-core
/// tiles, which pays for the operand loads and the scale arithmetic, and that
/// beats filling the grid. Keeping the grid full was costing the two 1024-wide
/// projections about a quarter of their throughput.
///
/// The depth stays 128 for the same reason: 64 measures worse at every width.
fn qmma_width(n: usize) -> usize {
    Q8_QMMA_WIDTHS
        .iter()
        .copied()
        .find(|&tn| n.is_multiple_of(tn))
        .unwrap_or(Q8_QMMA_TN)
}

/// The Q8_0 projection as a single `qdot_t`, which is what decoding runs. The
/// tile is eight outputs, one per warp of the CTA.
///
/// [`Q8_DP4A_SRC`] and [`Q8_SPLIT_SRC`] both stop every 32 elements of `k` to
/// apply the block scales, and `dot_t` gives each output column to one thread:
/// five barriers and 32 scattered sectors per kilobyte of weight. `qdot_t`
/// folds the scales in so the contraction is one operation, and turns the
/// mapping around so a warp owns an output and its lanes divide `k`, which puts
/// a warp's reads on 512 contiguous bytes and leaves nothing to stage. Over the
/// seven projections a decode step runs that is 2.4x to 7.4x, and it wants no
/// k-split, since 32 lanes per output already fill the machine.
const Q8_QDOT_SRC: &str = "\
@launch(256)
@autotune(TN in [8])
@aligned(N = TN)
kernel q8_qdot(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[N, KB],
               C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = qdot_t(A[0 :+ 1, :], AS[0 :+ 1, :],
                                    W[pn * TN :+ TN, :], WS[pn * TN :+ TN, :])
}
";

/// [`Q8_QDOT_SRC`] adding into its destination, for the residual connection at
/// the end of every block.
const Q8_QDOT_ADD_SRC: &str = "@launch(256)
@autotune(TN in [8])
@aligned(N = TN)
kernel q8_qdot_add(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                   W: tensor<i8>[N, K], WS: tensor<f32>[N, KB],
                   C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] += qdot_t(A[0 :+ 1, :], AS[0 :+ 1, :],
                                     W[pn * TN :+ TN, :], WS[pn * TN :+ TN, :])
}
";

/// Outputs per CTA in [`Q8_QDOT_SRC`], one per warp.
const Q8_QDOT_TN: usize = 8;

/// The Q8_0 projection with the contraction split across the grid.
///
/// [`Q8_DP4A_SRC`] puts the whole of `k` in one program, so its grid is `n / TN`
/// blocks and nothing else, which binds at decode. Achieved bandwidth tracks
/// the block count and little else: 38 GB/s at 32 blocks, 47 at 64, 111 at 112,
/// 159 at 7760, against roughly 427 the card sustains. Most of a decode step's
/// projections are 1024 or 2048 wide, so 32 or 64 blocks on 48 SMs.
///
/// Widening the tile to do more work per barrier shrinks the grid further and
/// measures worse everywhere, as does rearranging the weight so a program reads
/// one contiguous run instead of `TN` scattered pieces. Splitting `k` is the
/// only one of the three that adds blocks.
///
/// Program `(pn, ps)` takes output tile `pn` and the `k` slice at `ps` and
/// writes its partial sum to its own row of `P`, which `q8_reduce` then sums.
/// The slice comes from the extents rather than being compiled in, so one
/// module serves every shape and split count a model uses.
const Q8_SPLIT_SRC: &str = "\
@launch(256)
@autotune(TN in [32])
{ALIGNED}
kernel q8_split(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
                P: tensor<f32>[S, N]) {
  let pn = program_id(0)
  let ps = program_id(1)
  let slice = K / S
  let from = ps * slice
  var acc: tile<f32>[1, TN] = 0.0
  for kt in range(from, from + slice, 32) {
    let b = kt / 32
    let a = A[0 :+ 1, kt :+ 32]
    let w = W[pn * TN :+ TN, kt :+ 32]
    let ws = WS[b :+ 1, pn * TN :+ TN]
    let da = AS[0 :+ 1, b :+ 1]
    acc += f32(dot_t(a, w)) * ws * da
  }
  P[ps :+ 1, pn * TN :+ TN] = acc
}

@launch(256)
@autotune(RT in [128])
kernel q8_reduce(P: tensor<f32>[S, N], C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  var acc: tile<f32>[1, RT] = 0.0
  for s in range(0, S, 1) {
    acc = acc + P[s :+ 1, pn * RT :+ RT]
  }
  C[0 :+ 1, pn * RT :+ RT] = acc
}
";

/// Output tile for the split-K reduction.
const Q8_REDUCE_TN: usize = 128;

/// Blocks the split aims for, and the most slices it will cut `k` into. The
/// target is a little over four times the card's SM count, where the measured
/// bandwidth curve flattens. The cap keeps a slice from shrinking to a block or
/// two, where the second pass costs more than the first saves.
const Q8_SPLIT_TARGET: usize = 256;
const Q8_SPLIT_MAX: usize = 16;

/// Slices to cut a `[1, k] x [k, n]` projection's contraction into. One means
/// the whole contraction in one program, [`Q8_DP4A_SRC`] with no second pass;
/// wide projections already fill the machine and stay there.
fn q8_splits(n: usize, k: usize) -> usize {
    let grid = n.div_ceil(Q8_TN).max(1);
    let blocks = k / Q8_BLOCK;
    let mut splits = (Q8_SPLIT_TARGET / grid).min(Q8_SPLIT_MAX);

    // The scales change at every Q8_0 block, so a slice has to be whole ones.
    while splits > 1 && !blocks.is_multiple_of(splits) {
        splits /= 2;
    }

    splits.max(1)
}

/// The gated delta rule, carrying the recurrent state in shared memory.
///
/// Positions are sequential, so the loop over them is inside the kernel, but
/// every column of the `[head_dim, head_dim]` state is independent: the decay,
/// the rank-one write and the readout all touch column `j` alone. That lets the
/// state tile, which matters because a whole head's state is 64K at this head
/// dimension, exactly Turing's shared memory. Splitting it `DELTA_TN` columns
/// wide leaves room for the operands and gives the grid a second axis.
///
/// `H` divides the row index rather than multiplying it: rows are
/// `[position, head]` with the head fastest, so program `h` walks its own head
/// by stepping `H` at a time from `h`.
///
/// The rank-one write is a broadcast product, not a `dot` of a `[D, 1]` by a
/// `[1, TN]`. The two are the same arithmetic and 12% of the kernel apart,
/// since the contraction is over one element and all `dot` adds is machinery
/// for a contraction that is not there.
const DELTA_SRC: &str = "\
@launch(256)
@autotune(H in [{H}], D in [{D}], TN in [{TN}])
@aligned(R = H, D = TN, SD = D)
kernel delta_rule(Q:   tensor<f32>[R, D],
                  K:   tensor<f32>[R, D],
                  V:   tensor<f32>[R, D],
                  DEC: tensor<f32>[R, 1],
                  BET: tensor<f32>[R, 1],
                  S:   tensor<f32>[SD, D],
                  O:   tensor<f32>[R, D]) {
  let h = program_id(0)
  let jn = program_id(1)
  var st: tile<f32>[D, TN] = S[h * D :+ D, jn * TN :+ TN]

  for t in range(0, R, H) {
    let r = t + h
    var k = K[r :+ 1, 0 :+ D]
    var q = Q[r :+ 1, 0 :+ D]
    var v = V[r :+ 1, jn * TN :+ TN]
    var dec = DEC[r :+ 1, 0 :+ 1]
    var bet = BET[r :+ 1, 0 :+ 1]

    st = st * dec
    var e: tile<f32>[1, TN] = dot(k, st)
    e = (v - e) * bet
    var kt: tile<f32>[D, 1] = transpose(k)
    st = st + kt * e
    O[r :+ 1, jn * TN :+ TN] = dot(q, st)
  }

  S[h * D :+ D, jn * TN :+ TN] = st
}
";

/// The gated delta rule in chunks, as two passes.
///
/// [`DELTA_SRC`] walks positions one at a time because the recurrence says to,
/// and for 512 of them that is 512 rounds of staging five operand tiles for two
/// thousand elements of work. Unrolling the dependency between positions turns
/// a chunk of `C` of them into matmuls.
///
/// The recurrence is `S_i = a_i (I - beta_i k_i^T k_i) S_{i-1} + beta_i k_i^T
/// v_i`, and the `- k_i^T k_i S` term is what makes position `i` depend on every
/// earlier write inside the chunk. Writing `b_i` for the cumulative log decay,
/// the chunk has one pseudo-value `u_i` per position with
///
///   u_i + beta_i sum_{j<i} exp(b_i - b_j) (k_i . k_j) u_j
///     = beta_i v_i - beta_i exp(b_i) k_i S_0
///
/// so `U = T diag(beta) (V - diag(exp b) K S_0)` for `T = (I + N)^-1`, `N` that
/// strictly lower triangular matrix, and then
///
///   O   = diag(exp b) Q S_0 + tril(D * (Q K^T)) U
///   S_C = exp(b_C) (S_0 + (diag(exp(b_C - b)) K)^T U)
///
/// with `D[i, j] = exp(b_i - b_j)`.
///
/// It is two kernels because of how the work divides. Everything above that
/// depends only on the keys, so `N`, its inverse and the intra-chunk attention,
/// is the same for every column of the state, while the rest has to be walked
/// chunk by chunk and splits over those columns to fill the grid. As one kernel
/// the key-only half is recomputed per column slice, and at the chunk shared
/// memory then allows that measured 77.0 ms a pass against the sequential
/// kernel's 53.1. Split, the key-only half is a grid of (head, chunk) that runs
/// once and the scan can take a wider slice.
///
/// Four details are not the obvious spelling:
///
/// - The decay rides as a matrix. Every implementation of this folds it into
///   the operands as `q_i exp(b_i)` and `k_j exp(-b_j)`, which is one multiply
///   cheaper and overflows f32 outright, since `exp(-b_j)` grows without bound
///   as the chunk decays. As `D[i, j] = exp(b_i - b_j)` every entry of the
///   triangle that is used is at most one. The upper triangle does overflow,
///   but `tril` selects rather than multiplies, so no infinity reaches an
///   arithmetic operand.
/// - `(I + N)^-1` is not solved. `N` is strictly lower triangular and so
///   nilpotent, and forward substitution is `C` sequential steps, the depth
///   this exists to remove. `(I - M)^-1 = prod_j (I + M^(2^j))` is
///   `log2(C) - 1` rounds of two `[C, C]` matmuls and no sequential depth.
/// - `(T diag(exp b) K) S_0` associates to the right. Left to right it builds a
///   `[C, head_dim]` intermediate; as `T diag(exp b) (K S_0)` the intermediate
///   is `[C, TN]`, smaller and less arithmetic.
/// - `D` has an exact 1 on its diagonal, so `tril(D) - I` is the strict lower
///   mask, which `tril` alone cannot give.
///
/// `E` is the identity, uploaded once, and serves as both the `I` of the
/// inversion and that diagonal. The three `[C, C]` matrices the first pass
/// leaves are laid out one row per position with the heads side by side, the
/// same reinterpretation the operands take.
fn delta_wy_src(heads: usize, head_dim: usize, chunk: usize) -> String {
    // `I - N` covers the first two powers and each round doubles the reach, so
    // `log2(C) - 1` rounds carry it to N^(C-1), where N vanishes. The squarings
    // alternate between two names because a matmul writes its target as it
    // goes, and `p = dot(p, p)` would read what it had overwritten.
    let rounds = chunk.trailing_zeros().max(2) - 1;
    let mut invert = String::from("  var p0: tile<f32>[C, C] = dot(nn, nn)\n");

    for round in 0..rounds {
        let (this, next) = if round % 2 == 0 {
            ("p0", "p1")
        } else {
            ("p1", "p0")
        };

        invert.push_str(&format!("  t = t + dot(t, {this})\n"));

        if round + 1 < rounds {
            let decl = if round == 0 {
                "var p1: tile<f32>[C, C] = "
            } else {
                ""
            };

            let assign = if round == 0 {
                String::new()
            } else {
                format!("{next} = ")
            };

            invert.push_str(&format!("  {decl}{assign}dot({this}, {this})\n"));
        }
    }

    format!(
        "@launch(256)
@autotune(H in [{heads}], D in [{head_dim}], C in [{chunk}])
@aligned(N = C, QW = D, WW = C)
kernel delta_wy(Q:   tensor<f32>[N, QW],
                K:   tensor<f32>[N, QW],
                DEC: tensor<f32>[N, GW],
                BET: tensor<f32>[N, GW],
                E:   tensor<f32>[EC, ED],
                W:   tensor<f32>[N, WW]) {{
  let h = program_id(0)
  let ci = program_id(1)
  let qc = h * D
  let wc = h * 4 * C
  let c = ci * C
  var eye: tile<f32>[C, C] = E[0 :+ C, 0 :+ C]

  var k = K[c :+ C, qc :+ D]
  var dec = DEC[c :+ C, h :+ 1]
  var bet = BET[c :+ C, h :+ 1]

  var g: tile<f32>[C, 1] = log(dec + 0.000000000000000000000000000001)
  var b: tile<f32>[C, 1] = cumsum(g)
  var eb = exp(b)
  var dl: tile<f32>[C, C] = tril(exp(b - transpose(b)))

  var nn: tile<f32>[C, C] = dot_t(k, k) * (dl - eye) * bet
  var t: tile<f32>[C, C] = eye - nn
{invert}
  var tb: tile<f32>[C, C] = t * transpose(bet)
  W[c :+ C, wc :+ C] = tb
  W[c :+ C, wc + C :+ C] = tb * transpose(eb)
  W[c :+ C, wc + 2 * C :+ C] = dot_t(Q[c :+ C, qc :+ D], k) * dl

  var total: tile<f32>[1, 1] = rowsum(transpose(g))
  var ef: tile<f32>[C, 1] = exp(total - b)
  W[c :+ C, wc + 3 * C :+ 1] = eb
  W[c :+ C, wc + 3 * C + 1 :+ 1] = ef
  W[c :+ C, wc + 3 * C + 2 :+ 1] = ef * eb
}}
"
    )
}

/// The state scan of the chunked delta rule; see [`delta_wy_src`].
///
/// The order of the body is not free. The key and query tiles are both
/// `[C, head_dim]`, and letting the key die before the query is read is what
/// fits this in 48 KB: the transpose is taken at the key's last use so the
/// query reuses its buffer, 42 KB against 50.
fn delta_scan_src(heads: usize, head_dim: usize, tile: usize, chunk: usize) -> String {
    format!(
        "@launch(256)
@autotune(H in [{heads}], D in [{head_dim}], TN in [{tile}], C in [{chunk}])
@aligned(N = C, QW = D, VW = TN, OW = TN, WW = C, SD = D)
@dynshared
kernel delta_scan(Q:   tensor<f32>[N, QW],
                  K:   tensor<f32>[N, QW],
                  V:   tensor<f32>[N, VW],
                  W:   tensor<f32>[N, WW],
                  S:   tensor<f32>[SD, D],
                  O:   tensor<f32>[N, OW]) {{
  let h = program_id(0)
  let jn = program_id(1)
  let qc = h * D
  let vc = h * D + jn * TN
  let wc = h * 4 * C
  var st: tile<f32>[D, TN] = S[h * D :+ D, jn * TN :+ TN]

  for c in range(0, N, C) {{
    var k = K[c :+ C, qc :+ D]
    var kst: tile<f32>[C, TN] = dot(k, st)
    var kt: tile<f32>[D, C] = transpose(k)
    var uk: tile<f32>[C, TN] = dot(W[c :+ C, wc + C :+ C], kst)
    var u: tile<f32>[C, TN] = dot(W[c :+ C, wc :+ C], V[c :+ C, vc :+ TN])
    u = u - uk
    var qs: tile<f32>[C, TN] = dot(Q[c :+ C, qc :+ D], st)
    var mu: tile<f32>[C, TN] = dot(W[c :+ C, wc + 2 * C :+ C], u)
    qs = qs * W[c :+ C, wc + 3 * C :+ 1]
    O[c :+ C, vc :+ TN] = qs + mu

    st = st * W[c :+ 1, wc + 3 * C + 2 :+ 1]
    st += dot(kt, u * W[c :+ C, wc + 3 * C + 1 :+ 1])
  }}

  S[h * D :+ D, jn * TN :+ TN] = st
}}
"
    )
}

/// Positions one chunk covers, and the state columns one scan program owns.
///
/// The scan does about twelve tile operations per (chunk, slice) where the
/// sequential kernel does six per position, so a chunk of 16 over four slices
/// is half the tile operations of 512 positions over eight. Both are what
/// shared memory allows: `[C, head_dim]` and `[head_dim, TN]` tiles at 16 and
/// 32 are 42 KB of the 48 static shared memory gives.
const DELTA_CHUNK: usize = 16;
const DELTA_CHUNK_TN: usize = 64;

/// Columns of the delta-rule state one program owns.
const DELTA_TN: usize = 16;

/// The delta net's causal depthwise convolution, fused with the activation, the
/// split into per-head planes, and their normalization.
///
/// One program owns one position's one head of one plane: `D` channels of the
/// convolved stream. That span is both the row the delta rule reads and the
/// span the L2 normalization covers, so the convolution never writes an
/// intermediate and the split is a choice of destination rather than a pass.
///
/// `X` already carries the previous call's trailing positions ahead of this
/// call's, so tap `k` of position `t` is row `t + k` with no boundary case. The
/// taps arrive as `[KS, C]`, transposed relative to the file, so one tap across
/// a run of channels is one contiguous load.
///
/// The plane rides the grid's third axis rather than being three launches: they
/// read the same two buffers a fixed stride apart and write the same packed
/// destination, and one launch of ninety-six blocks beats three of thirty-two
/// on a forty-eight multiprocessor card. The planes are evenly spaced in both
/// of the layouts a file might use, so the offset is a multiple, not a table.
///
/// The query and key are L2-normalized and the value is not, since it is
/// written into the state rather than matched against it, and the query carries
/// the `1/sqrt(d)` softmax scale. Those are the only differences, and they are
/// a gain the epilogue picks by plane. The tests are on the block index, so
/// they stay uniform across the CTA.
#[allow(clippy::too_many_arguments)]
fn delta_conv_src(
    heads: usize,
    head_dim: usize,
    kernel: usize,
    stride: usize,
    plane_stride: usize,
    rows: usize,
    batch: usize,
    normalize: bool,
    query_scale: f32,
) -> String {
    let normalized = |scale: f32| {
        format!(
            "    g = {scale:.9} / sqrt(rowsum(s * s) + 0.000000000001)
"
        )
    };

    let query = if normalize {
        normalized(query_scale)
    } else {
        format!(
            "    g = {query_scale:.9}
"
        )
    };

    let key = if normalize {
        normalized(1.0)
    } else {
        String::new()
    };

    let key_case = if key.is_empty() {
        String::new()
    } else {
        format!(
            "  if p == 1 {{
{key}  }}
"
        )
    };

    format!(
        "@launch(256)
@autotune(H in [{heads}], D in [{head_dim}], KS in [{kernel}], ST in [{stride}], PS in [{plane_stride}], R in [{rows}], TB in [{batch}])
@aligned(C = D, HD = D)
kernel delta_conv(X: tensor<f32>[PR, C], W: tensor<f32>[KS, C], O: tensor<f32>[R3, HD]) {{
  let t = program_id(0)
  let h = program_id(1)
  let p = program_id(2)
  let cb = p * PS + h * ST
  var acc: tile<f32>[TB, D] = 0.0
  for k in range(0, KS, 1) {{
    var x = X[t * TB + k :+ TB, cb :+ D]
    var w = W[k :+ 1, cb :+ D]
    acc = acc + x * w
  }}
  var s = acc / (1.0 + exp(-acc))
  var g: tile<f32>[TB, 1] = 1.0
  if p == 0 {{
{query}  }}
{key_case}  O[p * R + t * TB :+ TB, h * D :+ D] = s * g
}}
"
    )
}

/// Positions one program of [`delta_conv_src`] carries. One position of one
/// head is 24576 blocks of 256 threads for two multiplies each at a 512-token
/// prompt, which measured 512 microseconds a call, almost all of it dispatch.
/// Eight positions at a time is eight times fewer blocks.
const DELTA_CONV_ROWS: usize = 8;

/// The positions a call batches. The tile has no remainder, so this is the
/// largest power of two up to [`DELTA_CONV_ROWS`] dividing the call.
fn delta_conv_batch(rows: usize) -> usize {
    let mut batch = DELTA_CONV_ROWS;
    while batch > 1 && !rows.is_multiple_of(batch) {
        batch /= 2;
    }
    batch
}

/// The delta rule's per-head gates.
///
/// The softplus is `max(x, 0) + log(1 + exp(-|x|))` rather than the direct
/// `log(1 + exp(x))`. The two agree everywhere, but the direct form overflows
/// the exponential well inside the range the decay projection reaches, and the
/// host reference guards that with a branch a tile has no room for.
///
/// The per-head parameters broadcast down the row tile, so a program covers
/// `TR` positions at once. A row on its own is only `H` wide, which would leave
/// most of the block idle.
fn delta_gates_src(heads: usize, tile_rows: usize) -> String {
    format!(
        "@launch(256)
@autotune(H in [{heads}], TR in [{tile_rows}])
kernel delta_gates(A: tensor<f32>[R, H], B: tensor<f32>[R, H],
                   RATE: tensor<f32>[M, H], BIAS: tensor<f32>[M, H],
                   DEC: tensor<f32>[R, H], BET: tensor<f32>[R, H]) {{
  let p = program_id(0)
  var a = A[p * TR :+ TR, 0 :+ H]
  var bias = BIAS[0 :+ 1, 0 :+ H]
  var rate = RATE[0 :+ 1, 0 :+ H]
  var x = a + bias
  var zero: tile<f32>[TR, H] = 0.0
  var mag = tmax(x, -x)
  var sp = tmax(x, zero) + log(1.0 + exp(-mag))
  DEC[p * TR :+ TR, 0 :+ H] = exp(rate * sp)
  var b = B[p * TR :+ TR, 0 :+ H]
  BET[p * TR :+ TR, 0 :+ H] = 1.0 / (1.0 + exp(-b))
}}
"
    )
}

/// Pointwise kernels over a flat buffer viewed as one row, so the tail is a
/// masked tile.
const POINTWISE_SRC: &str = "\
@launch(256)
@autotune(TILE in [1024])
kernel add_into(A: tensor<f32>[M, N], B: tensor<f32>[M, N]) {
  let p = program_id(0)
  var a = A[0 :+ 1, p * TILE :+ TILE]
  var b = B[0 :+ 1, p * TILE :+ TILE]
  A[0 :+ 1, p * TILE :+ TILE] = a + b
}

@launch(256)
@autotune(TILE in [1024])
kernel swiglu(G: tensor<f32>[M, N], U: tensor<f32>[M, N], O: tensor<f32>[M, N]) {
  let p = program_id(0)
  var g = G[0 :+ 1, p * TILE :+ TILE]
  var u = U[0 :+ 1, p * TILE :+ TILE]
  var s = g / (1.0 + exp(-g))
  O[0 :+ 1, p * TILE :+ TILE] = s * u
}

@launch(256)
@autotune(TILE in [1024])
kernel copy(S: tensor<f32>[M, N], D: tensor<f32>[M, N]) {
  let p = program_id(0)
  D[0 :+ 1, p * TILE :+ TILE] = S[0 :+ 1, p * TILE :+ TILE]
}

@launch(256)
@autotune(TILE in [1024])
kernel gate_into(X: tensor<f32>[M, N], G: tensor<f32>[M, N]) {
  let p = program_id(0)
  var x = X[0 :+ 1, p * TILE :+ TILE]
  var g = G[0 :+ 1, p * TILE :+ TILE]
  X[0 :+ 1, p * TILE :+ TILE] = x / (1.0 + exp(-g))
}
";

/// Elements of one `[BC, head_dim]` tile in [`attention_src`].
///
/// Shared memory bounds the tile, not arithmetic: the key and value tiles are
/// `[BC, head_dim]` each and the codegen allocates a second pair for the masked
/// replay of the loop body, so four have to fit. Four `[32, 128]` tiles are 64K
/// and will not load; [`ATTN_BLOCK_ELEMS`] says why the ceiling is 48K rather
/// than the hardware's 64K.
const ATTN_TILE_ELEMS: usize = 2048;

/// Rows of the cache one pass of the scan covers, and the cap on the single-key
/// remainder that follows it.
fn attention_tile(head_dim: usize) -> usize {
    (ATTN_TILE_ELEMS / head_dim).clamp(1, 32)
}

/// Elements of one `[BR, head_dim]` tile in [`attention_block_src`].
///
/// The blocked kernel carries about ten of them against the row kernel's four:
/// the query block, the accumulator, the key and value tiles, and one per step
/// of the rescale chain, since the codegen stages each in shared memory rather
/// than registers.
///
/// Ten at 4 KB apiece is 40 KB, as far as this can go. The ceiling is 48 KB
/// rather than Turing's 64 because these are `memref.global`s in the shared
/// address space, so they become statically declared arrays, and static shared
/// memory caps at 48 KB on every architecture. The rest needs a dynamic
/// allocation and an opt-in at launch.
const ATTN_BLOCK_ELEMS: usize = 1024;

/// Queries one program of [`attention_block_src`] covers: four at a head
/// dimension of 256, which is what caps it.
fn attention_block_tile(head_dim: usize) -> usize {
    (ATTN_BLOCK_ELEMS / head_dim).clamp(1, 16)
}

/// Query rows, keys and contraction depth of one [`attn_gemm_src`] tile.
///
/// The square tile is what puts the causal mask on `tril`: with both sides 64,
/// the one tile straddling the diagonal has key `j` paired with query `i` at
/// the same tile coordinates, so keeping `j <= i` is the mask.
pub const ATTN_GEMM_TILE: usize = 64;
const ATTN_GEMM_STEP: usize = 32;

/// The same for the softmax and the mix, which shared memory bounds rather than
/// throughput.
///
/// The softmax's tile has to be square for the same reason the scores' does and
/// holds four at once, so 64 would be 66 KB. The mix takes the same row tile,
/// which is not a tuning choice: the softmax leaves a row block zeroed only as
/// far as that block's own last query reaches, so a mix contracting over a
/// deeper block would sum raw scores for the rows above. Its step is shallower
/// to fit the `[32, 64]` accumulator alongside.
pub const ATTN_SOFT_TILE: usize = 32;
const ATTN_MIX_STEP: usize = 16;

/// Rows one program of the key transpose carries.
const ATTN_KT_ROWS: usize = 8;

/// Causal attention for a prompt, as two matmuls with the scores in between.
///
/// [`attention_block_src`] keeps one query block's whole attention inside one
/// program, which is the right shape at a small head dimension and the wrong
/// one here. At 256 the query tile, the accumulator and the key and value tiles
/// are all `[BR, 256]` f32, so 48 KB of shared memory caps `BR` at four and the
/// score tile is `[4, 4]`: sixteen output elements on a 256-thread CTA, which
/// measures 0.1 TFLOP/s and 30 ms of a 512-token pass.
///
/// Materializing the scores costs a `[rows, keys]` buffer per head and removes
/// the cap, so both halves become ordinary tiled matmuls at 3.3 and 2.6
/// TFLOP/s. The score matrix is 1 MB a head here, and the three passes over it
/// are 480 microseconds against the 26 ms the tiling saves.
///
/// The tensor cores are left out. They do reach 6.0 and 4.8 TFLOP/s against 3.3
/// and 2.6, but by rounding both operands to f16, which costs the call three
/// digits of accuracy and measures nothing end to end: once attention is off
/// the critical path the pass is the same 145 ms either way.
///
/// Two things have to be arranged for those matmuls to be clean. `dot_t` cannot
/// accumulate in place and `acc = acc + dot_t(..)` builds a whole tile per step
/// instead, measuring 0.33 against 3.3 TFLOP/s, so the keys are transposed once
/// a head and the scores are a plain `dot`. And a head is a column window of a
/// wider tensor at an offset no promise can bound, so every operand slice would
/// carry a mask and lose the pipelined path: with the head on the grid's third
/// axis that measured 74.5 microseconds against 25.4. Each head is gathered
/// into a buffer of its own first instead.
///
/// The three kernels split at the two points a row of scores has to be whole:
/// the softmax needs its row's maximum before it can exponentiate anything and
/// its sum before it can normalize. The sum is left for the mix to divide by,
/// so the scores are read three times rather than four.
///
/// All three skip the blocks past the diagonal rather than masking them: the
/// scores kernel does not run them, the softmax does not read them, and the mix
/// stops its contraction at the diagonal. That is half the rectangle.
pub fn attn_gemm_src(head_dim: usize, tile: usize) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    let (step, soft, mix_step) = (ATTN_GEMM_STEP, ATTN_SOFT_TILE, ATTN_MIX_STEP);
    format!(
        "@launch(256)
@autotune(TM in [{tile}], TN in [{tile}], TK in [{step}], TB in [{ATTN_KT_ROWS}], HD in [{head_dim}])
@aligned(NK = TN, KW = HD, DK = TB)
kernel attn_kt(K: tensor<f32>[NK, KW], T: tensor<f32>[DK, NK]) {{
  let p = program_id(0)
  var t = K[p * TB :+ TB, 0 :+ HD]
  T[0 :+ HD, p * TB :+ TB] = transpose(t)
}}

@launch(256)
@autotune(TM in [{tile}], TN in [{tile}], TK in [{step}], HD in [{head_dim}])
@aligned(R = TM, NK = TN, DQ = TK, DK = TK)
kernel attn_scores(Q: tensor<f32>[R, DQ], KT: tensor<f32>[DK, NK], S: tensor<f32>[R, NK]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  if pn * TN < NK - R + pm * TM + TM {{
    var acc: tile<f32>[TM, TN] = 0.0
    for kt in range(0, DQ, TK) {{
      var a = Q[pm * TM :+ TM, kt :+ TK]
      var b = KT[kt :+ TK, pn * TN :+ TN]
      acc += dot(a, b)
    }}
    S[pm * TM :+ TM, pn * TN :+ TN] = acc * {scale:.9}
  }}
}}

@launch(256)
@autotune(TM in [{soft}], TN in [{soft}])
@aligned(R = TM, NK = TN)
kernel attn_softmax(S: tensor<f32>[R, NK], L: tensor<f32>[R, 1]) {{
  let p = program_id(0)
  let diag = NK - R + p * TM
  var m: tile<f32>[TM, 1] = -300000000.0
  for j in range(0, diag, TN) {{
    m = tmax(m, rowmax(S[p * TM :+ TM, j :+ TN]))
  }}
  var d = S[p * TM :+ TM, diag :+ TN]
  m = tmax(m, rowmax(d))

  var l: tile<f32>[TM, 1] = 0.0
  for j in range(0, diag, TN) {{
    var e = exp(S[p * TM :+ TM, j :+ TN] - m)
    l = l + rowsum(e)
    S[p * TM :+ TM, j :+ TN] = e
  }}
  var de = tril(exp(d - m))
  l = l + rowsum(de)
  S[p * TM :+ TM, diag :+ TN] = de
  L[p * TM :+ TM, 0 :+ 1] = l
}}

@launch(256)
@autotune(TM in [{soft}], TN in [{tile}], TK in [{mix_step}])
@aligned(R = TM, DV = TN, NK = TK)
kernel attn_mix(P: tensor<f32>[R, NK], V: tensor<f32>[NK, DV], L: tensor<f32>[R, 1],
                O: tensor<f32>[R, DV]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, NK - R + pm * TM + TM, TK) {{
    var a = P[pm * TM :+ TM, kt :+ TK]
    var b = V[kt :+ TK, pn * TN :+ TN]
    acc += dot(a, b)
  }}
  O[pm * TM :+ TM, pn * TN :+ TN] = acc / L[pm * TM :+ TM, 0 :+ 1]
}}
"
    )
}

/// Causal attention over a block of queries at once.
///
/// [`attention_src`] gives each program one query row, the right shape for
/// decoding and the wrong one for a prompt: the key and value tiles are re-read
/// per query, and a 512-token pass measured 0.09 TFLOP/s where this card's own
/// flash-attention kernel does 5.3. Here a program owns `BR` consecutive
/// positions of one head, so one pass over the cache serves all of them.
///
/// The query block needs no rearranging. `Q` is `[rows, n_head * head_dim]`,
/// the same memory as the `[rows * n_head, head_dim]` the norm and the rotary
/// want, so `BR` consecutive rows at the head's column window is a query block.
///
/// Causality is `tril` on the one tile straddling the diagonal, which is why
/// the key tile is the query block's size: at both `BR`, tile column `j` is key
/// `base + j` and tile row `i` is query `base + i`, so keeping `j <= i` is the
/// mask. It cleans up the end of the cache too, since a column past the last
/// key is only ever paired with a query row that is also past it, and that row
/// is not stored.
///
/// The mask goes on the probabilities rather than the scores, which is the same
/// thing: a masked score's `exp` is what softmax would have driven to zero. The
/// running maximum takes in masked entries too, which only shifts the
/// exponentials down and cannot overflow.
fn attention_block_src(n_head: usize, group: usize, head_dim: usize, tile: usize) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    format!(
        "@launch(256)
@autotune(NH in [{n_head}], G in [{group}], D in [{head_dim}], BR in [{tile}])
kernel attention_block(Q: tensor<f32>[R, QW], K: tensor<f32>[NK, KW],
                       V: tensor<f32>[NK, KW], O: tensor<f32>[R, QW]) {{
  let qt = program_id(0)
  let h = program_id(1)
  let qcol = h * D
  let kcol = h / G * D
  var q = Q[qt * BR :+ BR, qcol :+ D]

  var acc: tile<f32>[BR, D] = 0.0
  var m: tile<f32>[BR, 1] = -300000000.0
  var l: tile<f32>[BR, 1] = 0.0

  let base = NK - R + qt * BR
  for kt in range(0, base, BR) {{
    let k = K[kt :+ BR, kcol :+ D]
    let v = V[kt :+ BR, kcol :+ D]
    var s: tile<f32>[BR, BR] = dot_t(q, k)
    s = s * {scale:.9}
    var mn: tile<f32>[BR, 1] = rowmax(s)
    mn = tmax(m, mn)
    s = exp(s - mn)
    var corr: tile<f32>[BR, 1] = exp(m - mn)
    l = l * corr + rowsum(s)
    acc = acc * corr + dot(s, v)
    m = mn
  }}

  let dk = K[base :+ BR, kcol :+ D]
  let dv = V[base :+ BR, kcol :+ D]
  var ds: tile<f32>[BR, BR] = dot_t(q, dk)
  ds = ds * {scale:.9}
  var dm: tile<f32>[BR, 1] = rowmax(ds)
  dm = tmax(m, dm)
  ds = exp(ds - dm)
  ds = tril(ds)
  var dcorr: tile<f32>[BR, 1] = exp(m - dm)
  l = l * dcorr + rowsum(ds)
  acc = acc * dcorr + dot(ds, dv)

  O[qt * BR :+ BR, qcol :+ D] = acc / l
}}
"
    )
}

/// A strided block copy, the shape every fused projection's split takes.
///
/// The width is baked in rather than tiled, so a row is one tile and the column
/// loop has no remainder to mask. The pitches are the declared extents and the
/// starting corner is a pointer offset, so neither side needs a stride beyond
/// what the descriptor carries.
fn copy_2d_src(width: usize, aligned: bool) -> String {
    // Without the promise the compiler cannot know the declared pitch is at
    // least the width, so every element carries a bounds check and the copy
    // does not vectorize: seven predicates and no vector access against two and
    // two. The promise is that both pitches are whole multiples of the width,
    // which a gather of one head out of a row of them satisfies and a slice of
    // a fused projection does not, so it compiles both ways.
    let claim = if aligned {
        "@aligned(SW = W, DW = W)"
    } else {
        ""
    };

    format!(
        "@launch(256)
@autotune(W in [{width}])
{claim}
kernel copy_2d(S: tensor<f32>[R, SW], D: tensor<f32>[R, DW]) {{
  let r = program_id(0)
  D[r :+ 1, 0 :+ W] = S[r :+ 1, 0 :+ W]
}}
"
    )
}

/// A SwiGLU whose two operands are planes of a wider buffer.
///
/// Shaped like [`copy_2d_src`]: the pitches are the declared extents and each
/// plane's start is a pointer offset, so neither side needs a stride the
/// descriptor does not carry. Unlike the copy it takes a column tile rather
/// than a whole row, since it holds three tiles at once; see
/// [`swiglu_2d_tile`]. The promise is that every pitch is a whole number of
/// tiles, which the halves of a fused gate-and-up projection satisfy.
fn swiglu_2d_src(tile: usize) -> String {
    format!(
        "@launch(256)
@autotune(T in [{tile}])
@aligned(GW = T, UW = T, OW = T)
kernel swiglu_2d(G: tensor<f32>[R, GW], U: tensor<f32>[R, UW], O: tensor<f32>[R, OW]) {{
  let r = program_id(0)
  let c = program_id(1) * T
  var g = G[r :+ 1, c :+ T]
  O[r :+ 1, c :+ T] = (g / (1.0 + exp(-g))) * U[r :+ 1, c :+ T]
}}
"
    )
}

/// Columns one program of the strided SwiGLU covers.
///
/// The kernel holds three tiles of this width, the gate, the up half and the
/// result, in static shared memory, which caps at 48 KB whatever the card has.
/// A whole row does not always fit: MiniCPM5-1B's 4608-wide feed-forward wants
/// 54 KB and the JIT refuses the module. So the row splits into the widest tile
/// that both divides it and fits, and the grid takes a second axis. A width
/// that already fits stays one tile.
fn swiglu_2d_tile(width: usize) -> usize {
    const OPERANDS: usize = 3;
    let fits = STATIC_SHARED_LIMIT / (OPERANDS * size_of::<f32>());
    (1..=width.min(fits))
        .rev()
        .find(|t| width.is_multiple_of(*t))
        .unwrap_or(width)
}

/// The rotary embedding, in place.
///
/// The angles arrive as a table rather than being computed: the language has no
/// sine, and the hardware's approximate one loses accuracy across the range an
/// absolute position covers. The caller offsets `T` to the first row's
/// position, so `r / H` is the row's position within it.
///
/// Reading both halves into tiles before either store is what makes the update
/// safe in place; the second store still needs the pre-rotation values.
fn rope_src(heads: usize, half: usize) -> String {
    format!(
        "@launch(256)
@autotune(H in [{heads}])
kernel rope(X: tensor<f32>[R, D], T: tensor<f32>[P, RD]) {{
  let r = program_id(0)
  let p = r / H
  var a = X[r :+ 1, 0 :+ {half}]
  var b = X[r :+ 1, {half} :+ {half}]
  var c = T[p :+ 1, 0 :+ {half}]
  var s = T[p :+ 1, {half} :+ {half}]
  X[r :+ 1, 0 :+ {half}] = a * c - b * s
  X[r :+ 1, {half} :+ {half}] = a * s + b * c
}}
"
    )
}

/// Causal softmax attention over the whole cache, one query row per program,
/// with the key axis split across the grid and a pass folding the pieces back
/// together.
///
/// FlashAttention's online softmax at one query row: the running maximum and
/// denominator rescale the carried output as each key tile arrives, so the
/// scores are never materialized and the cache is read once. Causality is a
/// loop bound rather than a mask, since the keys a row may see are a prefix:
/// the tiled loop runs over that prefix's whole tiles and picks up the last few
/// keys one at a time, which costs at most `BC - 1` narrow steps and needs no
/// triangular mask, padding or bias tensor.
///
/// `SP` is not passed: the key extent is `start_pos + rows` and the query
/// extent is `rows * n_head`, so the kernel recovers the position from the two.
/// The caches keep every head of one position together, which makes appending
/// one contiguous copy and a head a column window, so `K` and `V` are indexed
/// by `col` rather than sliced by row.
///
/// Unsplit, one query row leaves a grid of `n_head` blocks, eight of this
/// card's forty-eight multiprocessors, each walking the whole cache in turn:
/// fifty microseconds a call and eight per cent of a decode step for a few
/// megabytes of reading. A split block covers a slice of the keys and writes
/// the running maximum, sum and unnormalized accumulator it reached. Merging
/// those is the same rescaling the online softmax already does between tiles,
/// so the arithmetic is unchanged and the pieces stop having to be visited in
/// order.
///
/// The merge takes the maximum over all the pieces first and then weighs each
/// once, rather than rescaling what it has at every step: the running form
/// carries the wide accumulator through a multiply and an add per piece, this
/// one touches it twice in total and its first pass is over single values. The
/// split count rides in the partial buffer's extent rather than being compiled
/// in, so one module serves every cache length.
fn attention_split_src(n_head: usize, group: usize, head_dim: usize, tile: usize) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    format!(
        "@launch(256)
@autotune(NH in [{n_head}], G in [{group}], D in [{head_dim}], BC in [{tile}])
kernel attention_split(Q: tensor<f32>[R, D], K: tensor<f32>[NK, KW],
                       V: tensor<f32>[NK, KW],
                       P: tensor<f32>[SH, D], ML: tensor<f32>[SH, 2]) {{
  let h = program_id(0)
  let s = program_id(1)
  let col = h / G * D
  var q = Q[h :+ 1, 0 :+ D]

  var acc: tile<f32>[1, D] = 0.0
  var m: tile<f32>[1, 1] = -300000000.0
  var l: tile<f32>[1, 1] = 0.0

  let per = (NK + SH / NH - 1) / (SH / NH)
  var lo = s * per
  if lo > NK {{
    lo = NK
  }}
  var hi = lo + per
  if hi > NK {{
    hi = NK
  }}
  let full = lo + (hi - lo) / BC * BC
  for kt in range(lo, full, BC) {{
    let k = K[kt :+ BC, col :+ D]
    let v = V[kt :+ BC, col :+ D]
    var sc: tile<f32>[1, BC] = dot_t(q, k)
    sc = sc * {scale:.9}
    var mn: tile<f32>[1, 1] = rowmax(sc)
    mn = tmax(m, mn)
    sc = exp(sc - mn)
    var corr: tile<f32>[1, 1] = exp(m - mn)
    l = l * corr + rowsum(sc)
    acc = acc * corr + dot(sc, v)
    m = mn
  }}
  for j in range(full, hi, 1) {{
    let k1 = K[j :+ 1, col :+ D]
    let v1 = V[j :+ 1, col :+ D]
    var s1: tile<f32>[1, 1] = dot_t(q, k1)
    s1 = s1 * {scale:.9}
    var mn: tile<f32>[1, 1] = tmax(m, s1)
    var p: tile<f32>[1, 1] = exp(s1 - mn)
    var corr: tile<f32>[1, 1] = exp(m - mn)
    l = l * corr + p
    acc = acc * corr + p * v1
    m = mn
  }}
  P[s * NH + h :+ 1, 0 :+ D] = acc
  ML[s * NH + h :+ 1, 0 :+ 1] = m
  ML[s * NH + h :+ 1, 1 :+ 1] = l
}}

@launch(256)
@autotune(NH in [{n_head}], D in [{head_dim}])
kernel attention_merge(P: tensor<f32>[SH, D], ML: tensor<f32>[SH, 2],
                       O: tensor<f32>[R, D]) {{
  let h = program_id(0)
  var m: tile<f32>[1, 1] = -300000000.0
  for s in range(0, SH / NH, 1) {{
    m = tmax(m, ML[s * NH + h :+ 1, 0 :+ 1])
  }}
  var acc: tile<f32>[1, D] = 0.0
  var l: tile<f32>[1, 1] = 0.0
  for s in range(0, SH / NH, 1) {{
    let at = s * NH + h
    var c: tile<f32>[1, 1] = exp(ML[at :+ 1, 0 :+ 1] - m)
    acc = acc + P[at :+ 1, 0 :+ D] * c
    l = l + ML[at :+ 1, 1 :+ 1] * c
  }}
  O[h :+ 1, 0 :+ D] = acc / l
}}
"
    )
}

/// Pieces the key axis is cut into while decoding.
///
/// Fixed rather than chosen from the cache length, which is what it wants to
/// be: a count growing with the cache changed the pass's shape every few dozen
/// tokens, and each change costs a graph rebuild and a step issued the slow
/// way. Eight pieces is 64 blocks at this head count, and a piece with no keys
/// costs a block that exits immediately.
const ATTN_SPLITS: usize = 8;

fn attention_src(n_head: usize, group: usize, head_dim: usize, tile: usize) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    format!(
        "@launch(256)
@autotune(NH in [{n_head}], G in [{group}], D in [{head_dim}], BC in [{tile}])
kernel attention(Q: tensor<f32>[R, D], K: tensor<f32>[NK, KW],
                 V: tensor<f32>[NK, KW], O: tensor<f32>[R, D]) {{
  let t = program_id(0)
  let h = program_id(1)
  let col = h / G * D
  let row = t * NH + h
  var q = Q[row :+ 1, 0 :+ D]

  var acc: tile<f32>[1, D] = 0.0
  var m: tile<f32>[1, 1] = -300000000.0
  var l: tile<f32>[1, 1] = 0.0

  let visible = NK - R / NH + t + 1
  let full = visible / BC * BC
  for kt in range(0, full, BC) {{
    let k = K[kt :+ BC, col :+ D]
    let v = V[kt :+ BC, col :+ D]
    var s: tile<f32>[1, BC] = dot_t(q, k)
    s = s * {scale:.9}
    var mn: tile<f32>[1, 1] = rowmax(s)
    mn = tmax(m, mn)
    s = exp(s - mn)
    var corr: tile<f32>[1, 1] = exp(m - mn)
    l = l * corr + rowsum(s)
    acc = acc * corr + dot(s, v)
    m = mn
  }}
  for j in range(full, visible, 1) {{
    let k1 = K[j :+ 1, col :+ D]
    let v1 = V[j :+ 1, col :+ D]
    var s1: tile<f32>[1, 1] = dot_t(q, k1)
    s1 = s1 * {scale:.9}
    var mn: tile<f32>[1, 1] = tmax(m, s1)
    var p: tile<f32>[1, 1] = exp(s1 - mn)
    var corr: tile<f32>[1, 1] = exp(m - mn)
    l = l * corr + p
    acc = acc * corr + p * v1
    m = mn
  }}
  O[row :+ 1, 0 :+ D] = acc / l
}}
"
    )
}

/// Root-mean-square normalization over rows of a fixed width, one block per
/// row, viewed as blocks of 32, and the same with the quantized copy the
/// projections want. The width has to be a compile-time constant for the tile,
/// so this is generated per width; a GGUF decoder only ever normalizes at the
/// model dimension, so it compiles once. Epsilon is a plain decimal, since the
/// grammar's float literal has no exponent form.
///
/// Two things come out of the reshape. A row reduction hands one row to a warp,
/// so a `[1, width]` tile sums on 32 threads of a 256-thread CTA; folded to
/// `[width / 32, 32]` it is a row per eight lanes and the partials fold once
/// more. And blocks of 32 are what a Q8_0 scale covers, so the maximum the
/// quantization needs is the same row reduction over the same tile.
///
/// A projection reads every one of these immediately afterwards, and the
/// separate quantizing pass was a launch for four kilobytes of work.
fn rms_norm_src(width: usize, eps: f32, form: NormForm) -> String {
    let blocks = width / RMS_LANE;
    let (gated, quantized) = (form.gated(), form.quantized());
    let mut params = String::new();
    let mut body = String::new();

    // The gate multiplies the normalized row, so it replaces the plain store.
    if gated {
        params.push_str(&format!(", Z: tensor<f32>[RB, {RMS_LANE}]"));
        body.push_str(&format!(
            "  var z = Z[r * NB :+ NB, 0 :+ {RMS_LANE}]
               var y: tile<f32>[NB, {RMS_LANE}] = n * (z / (1.0 + exp(-z)))
"
        ));
    } else {
        body.push_str(&format!(
            "  var y: tile<f32>[NB, {RMS_LANE}] = n
"
        ));
    }
    body.push_str(&format!(
        "  O[r * NB :+ NB, 0 :+ {RMS_LANE}] = y
"
    ));

    // A Q8_0 scale covers 32 values, the row this tile is already folded into,
    // so the maximum is one more row reduction over it.
    if quantized {
        params.push_str(&format!(
            ", Q: tensor<i8>[RB, {RMS_LANE}], S: tensor<f32>[RB, D1]"
        ));
        body.push_str(&format!(
            "  var mx: tile<f32>[NB, 1] = rowmax(tmax(y, -y))
               var q = y * (127.0 / (mx + 0.00000001))
               Q[r * NB :+ NB, 0 :+ {RMS_LANE}] = i8(i32(round(q)))
               S[r * NB :+ NB, 0 :+ 1] = mx / 127.0
"
        ));
    }

    format!(
        "@launch({cta})
@autotune(NB in [{blocks}])
kernel {name}(X: tensor<f32>[RB, {RMS_LANE}], G: tensor<f32>[MB, {RMS_LANE}],
              O: tensor<f32>[RB, {RMS_LANE}]{params}) {{
  let r = program_id(0)
  var x = X[r * NB :+ NB, 0 :+ {RMS_LANE}]
  var sq: tile<f32>[NB, 1] = rowsum(x * x)
  var tot: tile<f32>[1, 1] = rowsum(transpose(sq))
  var inv: tile<f32>[1, 1] = 1.0 / sqrt(tot / {width}.0 + {eps:.12})
  var g = G[0 :+ NB, 0 :+ {RMS_LANE}]
  var n: tile<f32>[NB, {RMS_LANE}] = x * inv * g
{body}}}
",
        cta = norm_cta(blocks),
        name = form.kernel()
    )
}

/// Threads a normalization's CTA carries: one per value of its tile, up to the
/// usual width.
///
/// The delta net normalizes one head of one position, 128 values, so on a
/// 256-thread CTA half the block had nothing to do while the grid ran 8192 of
/// them. Sizing the block to the tile is what walking several groups per
/// program was reaching for, and it does not lengthen the reduction: the group
/// still belongs to one program.
fn norm_cta(blocks: usize) -> usize {
    (blocks * RMS_LANE).clamp(WARP_THREADS, CTA_THREADS as usize)
}

/// Values per row of the reshaped normalization tile, and of a Q8_0 block.
const RMS_LANE: usize = 32;

/// `out = silu(gate) * up`, with the quantized copy the projection after it
/// reads. Folded into blocks of [`RMS_LANE`] for the same reason the
/// normalization is: that is the run a Q8_0 scale covers, so the maximum is a
/// row reduction over the tile the kernel already holds.
fn swiglu_q_src(blocks: usize) -> String {
    format!(
        "@launch(256)
@autotune(NB in [{blocks}])
kernel swiglu_q(G: tensor<f32>[RB, {RMS_LANE}], U: tensor<f32>[RB, {RMS_LANE}],
                O: tensor<f32>[RB, {RMS_LANE}], Q: tensor<i8>[RB, {RMS_LANE}],
                S: tensor<f32>[RB, D1]) {{
  let p = program_id(0)
  var g = G[p * NB :+ NB, 0 :+ {RMS_LANE}]
  var u = U[p * NB :+ NB, 0 :+ {RMS_LANE}]
  var y: tile<f32>[NB, {RMS_LANE}] = (g / (1.0 + exp(-g))) * u
  O[p * NB :+ NB, 0 :+ {RMS_LANE}] = y
  var mx: tile<f32>[NB, 1] = rowmax(tmax(y, -y))
  var q = y * (127.0 / (mx + 0.00000001))
  Q[p * NB :+ NB, 0 :+ {RMS_LANE}] = i8(i32(round(q)))
  S[p * NB :+ NB, 0 :+ 1] = mx / 127.0
}}
"
    )
}

/// Values of a SwiGLU one CTA takes, in [`RMS_LANE`] blocks.
const SWIGLU_Q_BLOCKS: usize = ELEM_TILE / RMS_LANE;

/// What a normalization kernel leaves behind besides the normalized row. Both
/// extras are epilogues on the same tile, so they compose: the delta net's
/// readout is gated and then read by a projection, and wants both.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum NormForm {
    Plain,
    /// The quantized copy the projections read.
    Quantized,
    /// Multiplied by a SiLU gate, what the delta net's readout wants.
    GatedQuantized,
}

impl NormForm {
    fn gated(self) -> bool {
        self == NormForm::GatedQuantized
    }

    fn quantized(self) -> bool {
        self != NormForm::Plain
    }

    fn kernel(self) -> &'static str {
        match self {
            NormForm::Plain => "rms_norm",
            NormForm::Quantized => "rms_norm_q",
            NormForm::GatedQuantized => "rms_norm_gated",
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
    /// Keyed by head count and head dimension. Both are tile extents, so like
    /// the norm width they have to be compile-time constants, and a model only
    /// ever uses one pair.
    deltas: RefCell<HashMap<(usize, usize), Module>>,
    chunks: RefCell<HashMap<(usize, usize), Module>>,
    identities: RefCell<HashMap<usize, DeviceBuffer<f32>>>,
    /// Keyed by [`ConvKey`]. A model compiles three, one per plane, since the
    /// epilogue differs.
    convs: RefCell<HashMap<ConvKey, Module>>,
    /// Gate kernels, keyed by head count.
    gates: RefCell<HashMap<usize, Module>>,
    /// Strided copy kernels, keyed by the width they copy.
    splits: RefCell<HashMap<(usize, bool), Module>>,
    /// Rotary kernels, keyed by head count and half the rotary width.
    ropes: RefCell<HashMap<(usize, usize), Module>>,
    /// Attention kernels, keyed by query heads, group size and head dimension.
    attentions: RefCell<HashMap<(usize, usize, usize), Module>>,
    split_attn: RefCell<HashMap<(usize, usize, usize), Module>>,
    /// Per-split partial accumulators, and their running maxima and sums.
    attn_partials: RefCell<Option<(DeviceBuffer<f32>, DeviceBuffer<f32>)>>,
    /// Page-locked staging for the one readback a pass makes.
    readback: RefCell<Option<LockedBuffer<f32>>>,
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
    /// projection it has seen.
    ///
    /// Slot by slot rather than one shared arena, because several are live at
    /// once and because a pass asks for them in the same order every time,
    /// which keeps their addresses out of the graph's changing nodes.
    act_scratch: RefCell<Vec<(DeviceBuffer<i8>, DeviceBuffer<f32>)>>,
    act_next: Cell<usize>,
    /// Split-K partial sums, `[splits, n]`, grown to the largest asked for.
    split_scratch: RefCell<Option<DeviceBuffer<f32>>>,
    /// Must be last: Rust drops in declaration order and every allocation
    /// above has to be released while the context is still alive.
    _ctx: cust::context::Context,
}

impl DeviceBackend {
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
        // matmul_q8 requires k to be a whole number of Q8_0 blocks, so the k
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
            functions: RefCell::new(HashMap::new()),
            func_shared: RefCell::new(HashMap::new()),
            func_threads: RefCell::new(HashMap::new()),
            eager: RefCell::new(Recorded::default()),
            recording: Cell::new(false),
            flushed: Cell::new(false),
            pending: RefCell::new(Vec::new()),
            recorded_len: Cell::new(0),
            pass: RefCell::new(None),
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
            ropes: RefCell::new(HashMap::new()),
            attentions: RefCell::new(HashMap::new()),
            split_attn: RefCell::new(HashMap::new()),
            attn_partials: RefCell::new(None),
            readback: RefCell::new(None),
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

    fn store(&self, buffer: DeviceBuffer<f32>) -> Buf {
        if let Some(slot) = self.free_slots.borrow_mut().pop() {
            self.slots.borrow_mut()[slot] = Some(buffer);
            return Buf(slot);
        }
        let mut slots = self.slots.borrow_mut();
        slots.push(Some(buffer));
        Buf(slots.len() - 1)
    }

    /// The device pointer behind a handle, offset by `elements`.
    fn ptr(&self, buf: Buf, elements: usize) -> Result<u64> {
        let slots = self.slots.borrow();
        let buffer = slots
            .get(buf.0)
            .and_then(Option::as_ref)
            .context("use of a released buffer handle")?;
        ensure!(
            elements <= buffer.len(),
            "offset {elements} is past the buffer"
        );
        Ok(buffer.as_device_ptr().as_raw() + (elements * size_of::<f32>()) as u64)
    }

    fn len_of(&self, buf: Buf) -> Result<usize> {
        let slots = self.slots.borrow();
        Ok(slots
            .get(buf.0)
            .and_then(Option::as_ref)
            .context("use of a released buffer handle")?
            .len())
    }

    /// The kernels do not tolerate a destination sharing storage with a
    /// source, so fail loudly on one.
    fn check_distinct(&self, what: &str, dst: Buf, sources: &[Buf]) {
        if std::env::var_os("PHOBOS_CHECK_BUFS").is_none() {
            return;
        }
        for &src in sources {
            assert_ne!(dst.0, src.0, "{what}: destination aliases a source");
        }
    }

    /// Launch a kernel over tensor operands given as pointer and extents.
    ///
    /// Nothing here allocates. A pass is some seven hundred launches, and a
    /// fresh vector per descriptor cost most of a millisecond a step in host
    /// code alone, which the card spends idle waiting to be handed the pass.
    fn launch(
        &self,
        module: &Module,
        name: &'static str,
        operands: &[(u64, [i64; 2])],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let func = self.function(module, name)?;

        if self.recording.get() {
            let mut pending = self.pending.borrow_mut();
            let at = self.recorded_len.get();
            if at == pending.len() {
                pending.push(Recorded::default());
            }
            let slot = &mut pending[at];
            slot.func = func;
            slot.grid = grid;
            slot.shared = self.shared_of(func);
            slot.threads = self.threads_of(func)?;
            slot.slots.clear();
            for &(ptr, dims) in operands {
                push_descriptor(&mut slot.slots, ptr, dims);
            }
            self.recorded_len.set(at + 1);
            return Ok(());
        }

        let mut eager = self.eager.borrow_mut();
        eager.func = func;
        eager.grid = grid;
        eager.shared = self.shared_of(func);
        eager.threads = self.threads_of(func)?;
        eager.slots.clear();
        for &(ptr, dims) in operands {
            push_descriptor(&mut eager.slots, ptr, dims);
        }
        self.issue(&eager, name)
    }

    /// Zero unless the kernel is one of the `@dynshared` ones.
    #[inline]
    fn shared_of(&self, func: cust::sys::CUfunction) -> u32 {
        self.func_shared
            .borrow()
            .get(&(func as usize))
            .copied()
            .unwrap_or(0)
    }

    /// What `@launch` put in the kernel's `maxntid`, asked of the driver once
    /// per kernel.
    #[inline]
    fn threads_of(&self, func: cust::sys::CUfunction) -> Result<u32> {
        if let Some(&threads) = self.func_threads.borrow().get(&(func as usize)) {
            return Ok(threads);
        }
        let mut threads = 0i32;
        cuda_ok(
            // SAFETY: `func` came from cuModuleGetFunction and the module is
            // held for the backend's lifetime.
            unsafe {
                cust::sys::cuFuncGetAttribute(
                    &mut threads,
                    cust::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    func,
                )
            },
            "asking a kernel how wide a block it wants",
        )?;
        let threads = (threads as u32).min(CTA_THREADS);
        self.func_threads
            .borrow_mut()
            .insert(func as usize, threads);
        Ok(threads)
    }

    /// The kernel handle for `name`, resolved once. `cuModuleGetFunction` is a
    /// driver call, and the key is the two addresses rather than the name's
    /// text: every call site passes a literal, so the pointer identifies it
    /// without hashing or copying the string.
    #[inline]
    fn function(&self, module: &Module, name: &'static str) -> Result<cust::sys::CUfunction> {
        let key = (module as *const Module as usize, name.as_ptr() as usize);
        if let Some(&func) = self.functions.borrow().get(&key) {
            return Ok(func);
        }
        let func = module.get_function(name)?.to_raw();
        self.functions.borrow_mut().insert(key, func);
        Ok(func)
    }

    /// Puts one launch on the stream.
    fn issue(&self, launch: &Recorded, name: &'static str) -> Result<()> {
        let mut argv: Vec<*mut c_void> = launch
            .slots
            .iter()
            .map(|s| s as *const u64 as *mut u64 as *mut c_void)
            .collect();
        // SAFETY: argv points into launch.slots, which outlives the call; the
        // layout matches phobos-mlir's exploded-memref ABI, and every buffer
        // stays alive because the caller holds its handle.
        cuda_ok(
            unsafe {
                cust::sys::cuLaunchKernel(
                    launch.func,
                    launch.grid.0,
                    launch.grid.1,
                    launch.grid.2,
                    launch.threads,
                    1,
                    1,
                    launch.shared,
                    self.stream.as_inner(),
                    argv.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            name,
        )
    }

    /// Issues whatever has been recorded so far and stops recording this pass.
    /// Growing a scratch arena frees the buffer the recorded launches point at,
    /// so the recording has to be spent before the old pointer dies. The arenas
    /// grow only until they have seen the widest projection, so after the
    /// warmup no pass flushes and every pass is one graph.
    fn flush_pending(&self) -> Result<()> {
        if !self.recording.get() {
            return Ok(());
        }
        self.recording.set(false);
        self.flushed.set(true);
        self.issue_recorded("flushing a recorded launch")
    }

    /// Issues the recording as plain launches and empties it.
    fn issue_recorded(&self, what: &'static str) -> Result<()> {
        let pending = self.pending.borrow();
        for launch in &pending[..self.recorded_len.replace(0)] {
            self.issue(launch, what)?;
        }
        Ok(())
    }

    /// Replays the recorded pass, building or patching the graph first. A
    /// rebuild is only needed when the pass's shape changes, which in practice
    /// means the first decode step after a prefill and the reverse. Otherwise
    /// the topology is identical and the only nodes that moved are the ones
    /// reading the key/value cache, whose length grew by a token.
    fn replay(&self) -> Result<()> {
        let pending = self.pending.borrow();
        let recorded = &pending[..self.recorded_len.replace(0)];
        let mut cached = self.pass.borrow_mut();
        let reusable = cached.as_ref().is_some_and(|p| {
            p.recorded.len() == recorded.len()
                && p.recorded
                    .iter()
                    .zip(recorded)
                    .all(|(a, b)| a.func == b.func)
        });
        if !reusable {
            *cached = Some(Self::build_graph(recorded)?);
        } else {
            let pass = cached.as_mut().expect("reusable implies present");
            let mut argv = Vec::new();
            for (i, (was, now)) in pass.recorded.iter_mut().zip(recorded).enumerate() {
                if was.same(now) {
                    continue;
                }
                let params = now.params(&mut argv);
                // SAFETY: the node belongs to this exec and the parameter
                // block matches the function it was built with.
                cuda_ok(
                    unsafe {
                        cust::sys::cuGraphExecKernelNodeSetParams(pass.exec, pass.nodes[i], &params)
                    },
                    "updating a graph node",
                )?;
                was.func = now.func;
                was.grid = now.grid;
                was.slots.clear();
                was.slots.extend_from_slice(&now.slots);
            }
        }

        let exec = cached.as_ref().expect("built above").exec;
        // SAFETY: the exec outlives the launch, held by self.pass.
        cuda_ok(
            unsafe { cust::sys::cuGraphLaunch(exec, self.stream.as_inner()) },
            "launching the pass graph",
        )?;
        Ok(())
    }

    /// Instantiates a recorded pass as a chain of kernel nodes. A chain rather
    /// than a dependency analysis: these launches shared one stream, so serial
    /// order is the ordering they already relied on.
    fn build_graph(recorded: &[Recorded]) -> Result<PassGraph> {
        let mut graph: cust::sys::CUgraph = std::ptr::null_mut();
        // SAFETY: graph is written on success and destroyed by PassGraph.
        cuda_ok(
            unsafe { cust::sys::cuGraphCreate(&mut graph, 0) },
            "creating the pass graph",
        )?;

        let mut nodes: Vec<cust::sys::CUgraphNode> = Vec::with_capacity(recorded.len());
        let mut argv = Vec::new();
        for launch in recorded {
            let params = launch.params(&mut argv);
            let deps = nodes.last().copied();
            let mut node: cust::sys::CUgraphNode = std::ptr::null_mut();
            // SAFETY: deps points at the previous node, which belongs to graph.
            cuda_ok(
                unsafe {
                    cust::sys::cuGraphAddKernelNode(
                        &mut node,
                        graph,
                        deps.as_ref().map_or(std::ptr::null(), |d| d as *const _),
                        usize::from(deps.is_some()),
                        &params,
                    )
                },
                "adding a graph node",
            )?;
            nodes.push(node);
        }

        let mut exec: cust::sys::CUgraphExec = std::ptr::null_mut();
        // SAFETY: exec is written on success and destroyed by PassGraph.
        cuda_ok(
            unsafe {
                cust::sys::cuGraphInstantiate_v2(
                    &mut exec,
                    graph,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                )
            },
            "instantiating the pass graph",
        )?;

        Ok(PassGraph {
            graph,
            exec,
            nodes,
            recorded: recorded
                .iter()
                .map(|r| Recorded {
                    func: r.func,
                    grid: r.grid,
                    shared: r.shared,
                    threads: r.threads,
                    slots: r.slots.clone(),
                })
                .collect(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn project_q8(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        w: QBuf,
        n: usize,
        out: Buf,
        accumulate: bool,
    ) -> Result<()> {
        ensure!(
            k.is_multiple_of(Q8_BLOCK),
            "matmul_q8 needs k ({k}) to be a multiple of {Q8_BLOCK}"
        );
        let quants = self.quants.borrow();
        let (qs, scales, row_scales, stored_n) = quants
            .get(w.0)
            .context("use of an unknown quantized weight handle")?;
        ensure!(
            *stored_n == n,
            "quantized weight was uploaded with n = {stored_n}, used with n = {n}"
        );
        let (w_ptr, s_ptr) = (qs.as_device_ptr().as_raw(), scales.as_device_ptr().as_raw());
        let rs_ptr = row_scales.as_device_ptr().as_raw();
        let out_ptr = self.ptr(out, 0)?;
        let blocks = k / Q8_BLOCK;
        let (qa_ptr, das_ptr) = self.act_ptrs(act)?;

        let f32_bytes = size_of::<f32>() as u64;

        // The rows go through four kernels, deepest tile first, each taking
        // the whole tiles it can before handing the remainder on. That is the
        // difference between a prefill walking its rows on the host and one
        // spending them on the grid.
        //
        // `qmma_t` carries the scales itself, so it needs no bounds mask on the
        // weight and keeps its 64-wide column tile; a width that does not
        // divide by 64 goes to the kernel below. It comes in two depths because
        // the deeper one is 30% faster and a prompt is rarely a whole number of
        // 128-row tiles. Whole tensor-core tiles left over take `q8_mma` at 8
        // rows deep, and single rows finish on the matvec, as does decoding,
        // where every row is a leftover.
        let mut qmma_rows = 0;
        if n.is_multiple_of(Q8_QMMA_TN) {
            let wide = qmma_width(n);
            for (module, depth, tn) in [
                (&self.q8_qmma_deep[&wide], Q8_QMMA_TM, wide),
                (&self.q8_qmma, Q8_QMMA_SHALLOW, Q8_QMMA_TN),
            ] {
                let left = m - qmma_rows;
                let rows = left - left % depth;
                if rows == 0 {
                    continue;
                }
                self.launch(
                    module,
                    "q8_qmma",
                    &[
                        (qa_ptr + (qmma_rows * k) as u64, [rows as i64, k as i64]),
                        (
                            das_ptr + (qmma_rows * blocks) as u64 * f32_bytes,
                            [rows as i64, blocks as i64],
                        ),
                        (w_ptr, [n as i64, k as i64]),
                        (s_ptr, [blocks as i64, n as i64]),
                        (
                            out_ptr + (qmma_rows * n) as u64 * f32_bytes,
                            [rows as i64, n as i64],
                        ),
                    ],
                    ((rows / depth) as u32, (n / tn) as u32, 1),
                )?;
                qmma_rows += rows;
            }
        }

        let left = m - qmma_rows;
        let mma_rows = qmma_rows + left - left % Q8_MMA_TM;
        if mma_rows > qmma_rows {
            let rows = mma_rows - qmma_rows;
            self.launch(
                // The rows tile evenly by construction; only n can be ragged.
                self.q8_mma.pick(n.is_multiple_of(Q8_MMA_TN)),
                "q8_mma",
                &[
                    (qa_ptr + (qmma_rows * k) as u64, [rows as i64, k as i64]),
                    (
                        das_ptr + (qmma_rows * blocks) as u64 * f32_bytes,
                        [rows as i64, blocks as i64],
                    ),
                    (w_ptr, [n as i64, k as i64]),
                    (s_ptr, [blocks as i64, n as i64]),
                    (
                        out_ptr + (qmma_rows * n) as u64 * f32_bytes,
                        [rows as i64, n as i64],
                    ),
                ],
                ((rows / Q8_MMA_TM) as u32, n.div_ceil(Q8_MMA_TN) as u32, 1),
            )?;
        }

        let tiles_evenly = n.is_multiple_of(Q8_TN);
        let grid_n = n.div_ceil(Q8_TN) as u32;
        // A single row leaves the grid as short as the projection is wide,
        // which for most of a decode step is a fraction of the card. Splitting
        // the contraction fills it at the cost of a pass to sum the partials.
        // `qdot_t` fills it from the contraction instead, 32 lanes to an
        // output, and wants no split; the split kernels stay for the widths its
        // tile does not divide.
        let splits = q8_splits(n, k);
        let qdot = n.is_multiple_of(Q8_QDOT_TN);
        for row in mma_rows..m {
            let a_row = qa_ptr + (row * k) as u64;
            let as_row = das_ptr + (row * blocks) as u64 * f32_bytes;
            let c_row = out_ptr + (row * n) as u64 * f32_bytes;
            if qdot {
                let (module, name) = if accumulate {
                    (&self.q8_qdot_add, "q8_qdot_add")
                } else {
                    (&self.q8_qdot, "q8_qdot")
                };
                self.launch(
                    module,
                    name,
                    &[
                        (a_row, [1, k as i64]),
                        (as_row, [1, blocks as i64]),
                        (w_ptr, [n as i64, k as i64]),
                        (rs_ptr, [n as i64, blocks as i64]),
                        (c_row, [1, n as i64]),
                    ],
                    (n.div_ceil(Q8_QDOT_TN) as u32, 1, 1),
                )?;
                continue;
            }
            if splits == 1 {
                self.launch(
                    self.q8_dp4a.pick(tiles_evenly),
                    "q8_dp4a",
                    &[
                        (a_row, [1, k as i64]),
                        (as_row, [1, blocks as i64]),
                        (w_ptr, [n as i64, k as i64]),
                        (s_ptr, [blocks as i64, n as i64]),
                        (c_row, [1, n as i64]),
                    ],
                    (grid_n, 1, 1),
                )?;
                continue;
            }
            let partials = self.split_partials(splits * n)?;
            let module = self.q8_split.pick(tiles_evenly);
            self.launch(
                module,
                "q8_split",
                &[
                    (a_row, [1, k as i64]),
                    (as_row, [1, blocks as i64]),
                    (w_ptr, [n as i64, k as i64]),
                    (s_ptr, [blocks as i64, n as i64]),
                    (partials, [splits as i64, n as i64]),
                ],
                (grid_n, splits as u32, 1),
            )?;
            self.launch(
                module,
                "q8_reduce",
                &[
                    (partials, [splits as i64, n as i64]),
                    (c_row, [1, n as i64]),
                ],
                (n.div_ceil(Q8_REDUCE_TN) as u32, 1, 1),
            )?;
        }
        Ok(())
    }

    /// Scratch for `rows` split-attention partials of `head_dim` each, plus
    /// their maxima and sums.
    fn attn_scratch(&self, rows: usize, head_dim: usize) -> Result<(u64, u64)> {
        let too_small = self
            .attn_partials
            .borrow()
            .as_ref()
            .is_none_or(|(p, ml)| p.len() < rows * head_dim || ml.len() < rows * 2);
        if too_small {
            self.flush_pending()?;
            // SAFETY: the split kernel writes every row before the merge reads it.
            let grown = unsafe {
                (
                    DeviceBuffer::uninitialized(rows * head_dim)?,
                    DeviceBuffer::uninitialized(rows * 2)?,
                )
            };
            *self.attn_partials.borrow_mut() = Some(grown);
        }
        let scratch = self.attn_partials.borrow();
        let (p, ml) = scratch.as_ref().expect("filled above");
        Ok((p.as_device_ptr().as_raw(), ml.as_device_ptr().as_raw()))
    }

    /// The row count and width of the reshaped normalization tile.
    fn norm_shape(&self, width: usize) -> Result<(i64, i64)> {
        ensure!(
            width.is_multiple_of(RMS_LANE),
            "a normalization needs a width ({width}) that is a multiple of {RMS_LANE}"
        );
        Ok(((width / RMS_LANE) as i64, RMS_LANE as i64))
    }

    /// This pass's next quantized-activation slot, big enough for `m` rows of
    /// `k`, with its device pointers.
    fn act_slot(&self, m: usize, k: usize) -> Result<(QAct, u64, u64)> {
        ensure!(
            k.is_multiple_of(Q8_BLOCK),
            "a quantized activation needs k ({k}) to be a multiple of {Q8_BLOCK}"
        );
        let blocks = k / Q8_BLOCK;
        let at = self.act_next.get();
        let too_small = self
            .act_scratch
            .borrow()
            .get(at)
            .is_none_or(|(q, s)| q.len() < m * k || s.len() < m * blocks);
        if too_small {
            // Growing frees the buffer the recorded launches point at, so the
            // recording has to be spent first.
            self.flush_pending()?;
            // SAFETY: whatever fills the slot writes every element before the
            // projection reads it.
            let grown = unsafe {
                (
                    DeviceBuffer::uninitialized(m * k)?,
                    DeviceBuffer::uninitialized(m * blocks)?,
                )
            };
            let mut scratch = self.act_scratch.borrow_mut();
            if at == scratch.len() {
                scratch.push(grown);
            } else {
                scratch[at] = grown;
            }
        }
        self.act_next.set(at + 1);
        let (q, s) = self.act_ptrs(QAct(at))?;
        Ok((QAct(at), q, s))
    }

    /// Device pointers to the quantized activation behind a handle.
    fn act_ptrs(&self, act: QAct) -> Result<(u64, u64)> {
        let scratch = self.act_scratch.borrow();
        let (q, s) = scratch
            .get(act.0)
            .context("use of an unknown quantized activation handle")?;
        Ok((q.as_device_ptr().as_raw(), s.as_device_ptr().as_raw()))
    }

    /// Run `f` against a module compiled under `key` on first use.
    ///
    /// Several kernels take a tile extent that has to be a compile-time
    /// constant, so they are generated per shape. Every shape a model uses is
    /// fixed at load, so each cache holds a handful of entries and stops growing
    /// once decoding starts.
    fn with_kernel<K: Copy + Eq + std::hash::Hash>(
        &self,
        cache: &RefCell<HashMap<K, Module>>,
        key: K,
        what: &'static str,
        source: impl FnOnce() -> String,
        f: impl FnOnce(&Module) -> Result<()>,
    ) -> Result<()> {
        if !cache.borrow().contains_key(&key) {
            let module = self.compile_dynamic(&source(), what)?;
            cache.borrow_mut().insert(key, module);
        }
        let modules = cache.borrow();
        f(&modules[&key])
    }

    /// Causal attention for a prompt, one head at a time, as two matmuls with
    /// the scores materialized in between. See [`attn_gemm_src`].
    ///
    /// The gathers are what let the matmuls be clean: a head is a column window
    /// of a wider tensor, and slicing one inside the kernel costs every operand
    /// a bounds mask and with it the pipelined path. The keys are transposed on
    /// the way in for the same reason, since `dot_t` cannot accumulate in
    /// place. Both are strided copies of half a megabyte, one per head against
    /// 50 microseconds of matmul.
    fn attention_gemm(&self, q: Buf, keys: Buf, values: Buf, spec: Attn, out: Buf) -> Result<()> {
        let (rows, dim, nk) = (spec.rows, spec.head_dim, spec.total());
        let (qw, kw) = ((spec.n_head * dim) as i64, spec.kv_width());
        let tile = ATTN_GEMM_TILE;

        let head = self.alloc(rows * dim)?;
        let transposed = self.alloc(dim * nk)?;
        let value = self.alloc(nk * dim)?;
        let scores = self.alloc(rows * nk)?;
        let sums = self.alloc(rows)?;
        let mixed = self.alloc(rows * dim)?;

        let result = self.with_kernel(
            &self.attn_gemm,
            dim,
            "attn_gemm",
            || attn_gemm_src(dim, tile),
            |module| {
                let (r, d, n) = (rows as i64, dim as i64, nk as i64);
                for h in 0..spec.n_head {
                    // Consecutive query heads share a key head, so the
                    // transpose and the value gather only redo on a change.
                    if h.is_multiple_of(spec.group()) {
                        let at = h / spec.group() * dim;
                        self.launch(
                            module,
                            "attn_kt",
                            &[
                                (self.ptr(keys, at)?, [n, kw as i64]),
                                (self.ptr(transposed, 0)?, [d, n]),
                            ],
                            (nk.div_ceil(ATTN_KT_ROWS) as u32, 1, 1),
                        )?;
                        self.copy_2d(
                            Plane {
                                buf: values,
                                offset: at,
                                pitch: kw,
                            },
                            Plane {
                                buf: value,
                                offset: 0,
                                pitch: dim,
                            },
                            nk,
                            dim,
                        )?;
                    }
                    self.copy_2d(
                        Plane {
                            buf: q,
                            offset: h * dim,
                            pitch: spec.n_head * dim,
                        },
                        Plane {
                            buf: head,
                            offset: 0,
                            pitch: dim,
                        },
                        rows,
                        dim,
                    )?;
                    self.launch(
                        module,
                        "attn_scores",
                        &[
                            (self.ptr(head, 0)?, [r, d]),
                            (self.ptr(transposed, 0)?, [d, n]),
                            (self.ptr(scores, 0)?, [r, n]),
                        ],
                        ((rows / tile) as u32, nk.div_ceil(tile) as u32, 1),
                    )?;
                    self.launch(
                        module,
                        "attn_softmax",
                        &[(self.ptr(scores, 0)?, [r, n]), (self.ptr(sums, 0)?, [r, 1])],
                        ((rows / ATTN_SOFT_TILE) as u32, 1, 1),
                    )?;
                    self.launch(
                        module,
                        "attn_mix",
                        &[
                            (self.ptr(scores, 0)?, [r, n]),
                            (self.ptr(value, 0)?, [n, d]),
                            (self.ptr(sums, 0)?, [r, 1]),
                            (self.ptr(mixed, 0)?, [r, d]),
                        ],
                        ((rows / ATTN_SOFT_TILE) as u32, (dim / tile) as u32, 1),
                    )?;
                    self.copy_2d(
                        Plane {
                            buf: mixed,
                            offset: 0,
                            pitch: dim,
                        },
                        Plane {
                            buf: out,
                            offset: h * dim,
                            pitch: qw as usize,
                        },
                        rows,
                        dim,
                    )?;
                }
                Ok(())
            },
        );

        for buf in [head, transposed, value, scores, sums, mixed] {
            self.release(buf);
        }
        result
    }

    /// The chunked gated delta rule, in two passes; see [`delta_wy_src`].
    ///
    /// The packed operands are one row per (position, head) and a chunk wants
    /// `C` consecutive positions of one head, which sit `C` rows apart there.
    /// Read instead as one row per position with the heads side by side, the
    /// same memory, a chunk is a column window of consecutive rows and so an
    /// ordinary tile. The gates are the same reinterpretation one column wide,
    /// and the three `[C, C]` matrices the first pass leaves are laid out to
    /// match.
    fn delta_chunked(
        &self,
        packed: Buf,
        rows: usize,
        heads: usize,
        head_dim: usize,
        state: Buf,
        out: Buf,
    ) -> Result<()> {
        let (span, gates) = (rows * heads * head_dim, rows * heads);
        let (n, width) = (rows as i64, (heads * head_dim) as i64);
        let wy_width = heads * 4 * DELTA_CHUNK;
        let eye = self.identity(DELTA_CHUNK)?;
        let wy = self.alloc(rows * wy_width)?;
        let result = self.with_kernel(
            &self.chunks,
            (heads, head_dim),
            "delta_chunk",
            || {
                let mut source = delta_wy_src(heads, head_dim, DELTA_CHUNK);
                source.push_str(&delta_scan_src(
                    heads,
                    head_dim,
                    DELTA_CHUNK_TN,
                    DELTA_CHUNK,
                ));
                source
            },
            |module| {
                let wy_ptr = self.ptr(wy, 0)?;
                let wy_dims = [n, wy_width as i64];
                self.launch(
                    module,
                    "delta_wy",
                    &[
                        (self.ptr(packed, 0)?, [n, width]),
                        (self.ptr(packed, span)?, [n, width]),
                        (self.ptr(packed, 3 * span)?, [n, heads as i64]),
                        (self.ptr(packed, 3 * span + gates)?, [n, heads as i64]),
                        (eye, [DELTA_CHUNK as i64, DELTA_CHUNK as i64]),
                        (wy_ptr, wy_dims),
                    ],
                    (heads as u32, (rows / DELTA_CHUNK) as u32, 1),
                )?;
                self.launch(
                    module,
                    "delta_scan",
                    &[
                        (self.ptr(packed, 0)?, [n, width]),
                        (self.ptr(packed, span)?, [n, width]),
                        (self.ptr(packed, 2 * span)?, [n, width]),
                        (wy_ptr, wy_dims),
                        (
                            self.ptr(state, 0)?,
                            [(heads * head_dim) as i64, head_dim as i64],
                        ),
                        (self.ptr(out, 0)?, [n, width]),
                    ],
                    (heads as u32, (head_dim / DELTA_CHUNK_TN) as u32, 1),
                )
            },
        );
        self.release(wy);
        result
    }

    /// Compiles a module whose kernels may want dynamic shared memory,
    /// recording what each needs and raising those past the 48 KB ceiling.
    fn compile_dynamic(&self, source: &str, what: &'static str) -> Result<Module> {
        let (module, shared) = compile_shared(source, &[], what)?;
        for (name, bytes) in shared {
            let func = module.get_function(&name)?.to_raw();
            if bytes > STATIC_SHARED_LIMIT {
                // SAFETY: func comes from the module just loaded and outlives
                // this call; the attribute takes a plain byte count.
                cuda_ok(
                    unsafe {
                        cust::sys::cuFuncSetAttribute(
                            func,
                            cust::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            bytes as i32,
                        )
                    },
                    "raising a kernel's dynamic shared memory ceiling",
                )?;
            }
            self.func_shared
                .borrow_mut()
                .insert(func as usize, bytes as u32);
        }
        Ok(module)
    }

    /// An allocation for something that writes it immediately rather than from
    /// the stream, so it must not come out of the pool mid-pass.
    ///
    /// This is the only place that knows both halves: the pool cannot see
    /// whether a pass is recording, and the recorder does not allocate. See
    /// [`Pool::take_fresh`] for why reuse is unsafe while recording.
    ///
    /// Two callers reach this and both were wrong before it existed. A constant
    /// that grows mid-pass, as the rotary table does when a sequence passes its
    /// length: on `minicpm5-1b` the new table was exactly the size of the key
    /// cache the same block had just outgrown, so it landed on it and the copy
    /// carrying the cache forward read rotary angles instead. And a zero fill,
    /// which a delta net asks for once per generation, whose memset goes
    /// straight to the stream while the pass around it is only recorded.
    fn alloc_written_now(&self, len: usize) -> Result<Buf> {
        if !self.recording.get() {
            return self.alloc(len);
        }
        Ok(self.store(self.pool.take_fresh(len)?))
    }

    /// A device pointer to an `n` by `n` identity, uploaded once.
    fn identity(&self, n: usize) -> Result<u64> {
        if !self.identities.borrow().contains_key(&n) {
            let mut host = vec![0.0f32; n * n];
            for i in 0..n {
                host[i * n + i] = 1.0;
            }
            let buf = DeviceBuffer::from_slice(&host)?;
            self.identities.borrow_mut().insert(n, buf);
        }
        let identities = self.identities.borrow();
        Ok(identities[&n].as_device_ptr().as_raw())
    }

    /// A device pointer to scratch big enough for `len` split-K partial sums.
    fn split_partials(&self, len: usize) -> Result<u64> {
        if self
            .split_scratch
            .borrow()
            .as_ref()
            .is_none_or(|p| p.len() < len)
        {
            self.flush_pending()?;
            // SAFETY: q8_split writes every row before q8_reduce reads it.
            let grown = unsafe { DeviceBuffer::uninitialized(len)? };
            *self.split_scratch.borrow_mut() = Some(grown);
        }
        let scratch = self.split_scratch.borrow();
        Ok(scratch
            .as_ref()
            .expect("just filled")
            .as_device_ptr()
            .as_raw())
    }

    /// A pointwise launch over `len` elements of one or more flat buffers. Past
    /// enough of them to fill the card the wide kernel takes over; see
    /// [`ELEM_TILE_WIDE`].
    fn pointwise(&self, name: &'static str, buffers: &[(Buf, usize)], len: usize) -> Result<()> {
        let mut operands = Vec::with_capacity(buffers.len());
        for &(buf, offset) in buffers {
            operands.push((self.ptr(buf, offset)?, [1i64, len as i64]));
        }
        let (module, tile) = if len >= WIDE_FLOOR {
            (&self.pointwise_wide, ELEM_TILE_WIDE)
        } else {
            (&self.pointwise, ELEM_TILE)
        };
        self.launch(module, name, &operands, (len.div_ceil(tile) as u32, 1, 1))
    }
}

impl Backend for DeviceBackend {
    fn alloc(&self, len: usize) -> Result<Buf> {
        Ok(self.store(self.pool.take(len)?))
    }

    fn release(&self, buf: Buf) {
        let taken = self
            .slots
            .borrow_mut()
            .get_mut(buf.0)
            .and_then(Option::take);
        if let Some(buffer) = taken {
            self.pool.put(buffer);
            self.free_slots.borrow_mut().push(buf.0);
        }
    }

    fn upload(&self, data: &[f32]) -> Result<Buf> {
        let buf = self.alloc_written_now(data.len())?;
        let slots = self.slots.borrow();
        let buffer = slots
            .get(buf.0)
            .and_then(Option::as_ref)
            .context("upload lost its buffer")?;
        buffer.index(0..data.len()).copy_from(data)?;
        Ok(buf)
    }

    fn zeroed(&self, len: usize) -> Result<Buf> {
        let buf = self.alloc_written_now(len)?;
        let ptr = self.ptr(buf, 0)?;
        // On the stream, so it orders with the pass rather than forcing the
        // synchronization an upload's staging copy would.
        cuda_ok(
            // SAFETY: the allocation is at least `len` floats and the handle
            // holds it alive for the duration.
            unsafe { cust::sys::cuMemsetD32Async(ptr, 0, len, self.stream.as_inner()) },
            "zeroing a device allocation",
        )?;
        Ok(buf)
    }

    fn begin_pass(&self) -> Result<()> {
        self.act_next.set(0);
        self.recorded_len.set(0);
        self.flushed.set(false);
        self.recording.set(true);
        Ok(())
    }

    fn end_pass(&self) -> Result<()> {
        // A pass that had to flush is only partly recorded, so it goes out as
        // launches and leaves the cached graph alone. The next pass records
        // whole and replaces it.
        if self.flushed.replace(false) {
            self.recording.set(false);
            return self.issue_recorded("issuing the tail of a flushed pass");
        }
        self.recording.set(false);
        self.replay()
    }

    fn read(&self, buf: Buf, out: &mut [f32]) -> Result<()> {
        // The only synchronization point in a block: everything queued since
        // the last read has to land before the host can look at it.
        self.stream.synchronize()?;
        let slots = self.slots.borrow();
        let buffer = slots
            .get(buf.0)
            .and_then(Option::as_ref)
            .context("use of a released buffer handle")?;
        ensure!(
            buffer.len() >= out.len(),
            "reading {} elements from a {}-element buffer",
            out.len(),
            buffer.len()
        );
        // Through page-locked staging. The logits are the one thing a decode
        // step brings back, a megabyte a token at this vocabulary, and straight
        // into a Vec the driver bounces it through its own pinned staging a
        // page at a time, at about a third of the rate.
        let mut staging = self.readback.borrow_mut();
        let too_small = staging.as_ref().is_none_or(|s| s.len() < out.len());
        if too_small {
            *staging = Some(LockedBuffer::new(&0.0f32, out.len())?);
        }
        let pinned = staging.as_mut().expect("filled above");
        buffer
            .index(0..out.len())
            .copy_to(&mut pinned.as_mut_slice()[..out.len()])?;
        out.copy_from_slice(&pinned.as_slice()[..out.len()]);
        Ok(())
    }

    fn constant(&self, key: &str, data: &[f32]) -> Result<Buf> {
        if let Some(&buf) = self.constants.borrow().get(key) {
            return Ok(buf);
        }
        let buf = self.upload(data)?;
        self.constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    fn matmul(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()> {
        self.check_distinct("matmul", out, &[a, w]);
        let (a_ptr, w_ptr, out_ptr) = (self.ptr(a, 0)?, self.ptr(w, 0)?, self.ptr(out, 0)?);

        // The tiled kernel rounds a single row up to a whole TILE_M tile, so
        // decoding stays on the matvec specialization and anything wider tiles.
        // Either handles a ragged shape by masking the boundary tile, so the
        // choice is only which does less arithmetic.
        if m > 1 {
            let tiles_evenly = m.is_multiple_of(TILE_M) && n.is_multiple_of(TILE_N);
            return self.launch(
                self.matmul.pick(tiles_evenly),
                "matmul",
                &[
                    (a_ptr, [m as i64, k as i64]),
                    (w_ptr, [k as i64, n as i64]),
                    (out_ptr, [m as i64, n as i64]),
                ],
                (m.div_ceil(TILE_M) as u32, n.div_ceil(TILE_N) as u32, 1),
            );
        }

        self.launch(
            self.matvec.pick(n.is_multiple_of(MV_TN)),
            "matvec",
            &[
                (a_ptr, [1, k as i64]),
                (w_ptr, [k as i64, n as i64]),
                (out_ptr, [1, n as i64]),
            ],
            (n.div_ceil(MV_TN) as u32, 1, 1),
        )
    }

    fn constant_q8(
        &self,
        key: &str,
        qs: &[i8],
        scales: &[f32],
        k: usize,
        n: usize,
    ) -> Result<QBuf> {
        if let Some(&buf) = self.q_constants.borrow().get(key) {
            return Ok(buf);
        }
        check_q8_shape(qs, scales, k, n)?;
        let blocks = k / Q8_BLOCK;
        let mut row_scales = vec![0.0f32; scales.len()];
        for (b, row) in scales.chunks_exact(n).enumerate() {
            for (j, &s) in row.iter().enumerate() {
                row_scales[j * blocks + b] = s;
            }
        }
        let uploaded = (
            DeviceBuffer::from_slice(qs)?,
            DeviceBuffer::from_slice(scales)?,
            DeviceBuffer::from_slice(&row_scales)?,
            n,
        );
        let mut quants = self.quants.borrow_mut();
        quants.push(uploaded);
        let buf = QBuf(quants.len() - 1);
        drop(quants);
        self.q_constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }

    fn quantize_act(&self, a: Buf, m: usize, k: usize) -> Result<QAct> {
        let (act, qa_ptr, das_ptr) = self.act_slot(m, k)?;
        let blocks = k / Q8_BLOCK;
        let a_ptr = self.ptr(a, 0)?;
        let rows = m * blocks;
        let (module, tile) = if rows * Q8_BLOCK >= WIDE_FLOOR {
            (&self.quantize_wide, QUANT_TB_WIDE)
        } else {
            (&self.quantize, QUANT_TB)
        };
        self.launch(
            module,
            "quantize",
            &[
                (a_ptr, [rows as i64, Q8_BLOCK as i64]),
                (qa_ptr, [rows as i64, Q8_BLOCK as i64]),
                (das_ptr, [rows as i64, 1]),
            ],
            (rows.div_ceil(tile) as u32, 1, 1),
        )?;
        Ok(act)
    }

    fn matmul_q8_act(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        w: QBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        self.project_q8(act, m, k, w, n, out, false)
    }

    fn matmul_q8_add(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        w: QBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        // Only the single-row kernel accumulates. A prefill goes through the
        // tensor cores, where the residual add is a rounding error on the pass
        // rather than a launch that matters.
        if m != 1 || !n.is_multiple_of(Q8_QDOT_TN) {
            let temp = self.alloc(m * n)?;
            self.matmul_q8_act(act, m, k, w, n, temp)?;
            self.add_into(out, temp)?;
            self.release(temp);
            return Ok(());
        }
        self.project_q8(act, m, k, w, n, out, true)
    }

    fn rms_norm(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        out: Buf,
    ) -> Result<()> {
        self.check_distinct("rms_norm", out, &[x, gain]);
        let shape = self.norm_shape(width)?;
        self.with_kernel(
            &self.norms,
            (width, eps.to_bits()),
            "rms_norm",
            || rms_norm_src(width, eps, NormForm::Plain),
            |module| {
                self.launch(
                    module,
                    "rms_norm",
                    &[
                        (self.ptr(x, 0)?, [(rows as i64) * shape.0, shape.1]),
                        (self.ptr(gain, 0)?, [shape.0, shape.1]),
                        (self.ptr(out, 0)?, [(rows as i64) * shape.0, shape.1]),
                    ],
                    (rows as u32, 1, 1),
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rms_norm_gated(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        gate: Buf,
        gate_at: usize,
        out: Buf,
    ) -> Result<QAct> {
        self.check_distinct("rms_norm_gated", out, &[x, gain, gate]);
        let shape = self.norm_shape(width)?;
        let (act, qa_ptr, das_ptr) = self.act_slot(rows, width)?;
        self.with_kernel(
            &self.gated_norms,
            (width, eps.to_bits()),
            "rms_norm_gated",
            || rms_norm_src(width, eps, NormForm::GatedQuantized),
            |module| {
                let rb = (rows as i64) * shape.0;
                self.launch(
                    module,
                    "rms_norm_gated",
                    &[
                        (self.ptr(x, 0)?, [rb, shape.1]),
                        (self.ptr(gain, 0)?, [shape.0, shape.1]),
                        (self.ptr(out, 0)?, [rb, shape.1]),
                        (self.ptr(gate, gate_at)?, [rb, shape.1]),
                        (qa_ptr, [rb, shape.1]),
                        (das_ptr, [rb, 1]),
                    ],
                    (rows as u32, 1, 1),
                )
            },
        )?;
        Ok(act)
    }

    fn rms_norm_q(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        out: Buf,
    ) -> Result<QAct> {
        self.check_distinct("rms_norm_q", out, &[x, gain]);
        let shape = self.norm_shape(width)?;
        let (act, qa_ptr, das_ptr) = self.act_slot(rows, width)?;
        self.with_kernel(
            &self.quant_norms,
            (width, eps.to_bits()),
            "rms_norm_q",
            || rms_norm_src(width, eps, NormForm::Quantized),
            |module| {
                let rb = (rows as i64) * shape.0;
                self.launch(
                    module,
                    "rms_norm_q",
                    &[
                        (self.ptr(x, 0)?, [rb, shape.1]),
                        (self.ptr(gain, 0)?, [shape.0, shape.1]),
                        (self.ptr(out, 0)?, [rb, shape.1]),
                        (qa_ptr, [rb, shape.1]),
                        (das_ptr, [rb, 1]),
                    ],
                    (rows as u32, 1, 1),
                )
            },
        )?;
        Ok(act)
    }

    fn copy_2d(&self, src: Plane, dst: Plane, rows: usize, width: usize) -> Result<()> {
        self.check_distinct("copy_2d", dst.buf, &[src.buf]);
        let aligned = src.pitch.is_multiple_of(width) && dst.pitch.is_multiple_of(width);
        self.with_kernel(
            &self.splits,
            (width, aligned),
            "copy_2d",
            || copy_2d_src(width, aligned),
            |module| {
                self.launch(
                    module,
                    "copy_2d",
                    &[
                        (
                            self.ptr(src.buf, src.offset)?,
                            [rows as i64, src.pitch as i64],
                        ),
                        (
                            self.ptr(dst.buf, dst.offset)?,
                            [rows as i64, dst.pitch as i64],
                        ),
                    ],
                    (rows as u32, 1, 1),
                )
            },
        )
    }

    fn rope(&self, x: Buf, rows: usize, table: Buf, spec: Rope) -> Result<()> {
        let half = spec.rope_dim / 2;
        let (r, d) = ((rows * spec.heads) as i64, spec.head_dim as i64);
        self.with_kernel(
            &self.ropes,
            (spec.heads, half),
            "rope",
            || rope_src(spec.heads, half),
            |module| {
                self.launch(
                    module,
                    "rope",
                    &[
                        (self.ptr(x, 0)?, [r, d]),
                        (
                            // Offset to this call's first position, so the
                            // kernel indexes the table by row rather than
                            // taking the absolute position as an argument.
                            self.ptr(table, spec.start_pos * spec.rope_dim)?,
                            [rows as i64, spec.rope_dim as i64],
                        ),
                    ],
                    (r as u32, 1, 1),
                )
            },
        )
    }

    fn attention(&self, q: Buf, keys: Buf, values: Buf, spec: Attn, out: Buf) -> Result<()> {
        self.check_distinct("attention", out, &[q, keys, values]);
        let (r, d) = ((spec.rows * spec.n_head) as i64, spec.head_dim as i64);
        let (nk, kw) = (spec.total() as i64, spec.kv_width() as i64);
        let (qw, block) = (
            (spec.n_head * spec.head_dim) as i64,
            attention_block_tile(spec.head_dim),
        );
        // A prompt whose query block, cache and head dimension all tile evenly
        // goes through the two matmuls, 26 ms of a 512-token pass; see
        // `attn_gemm_src`. Everything else falls through to the kernels below,
        // which need no alignment.
        let gemm = ATTN_GEMM_TILE;
        if spec.rows >= gemm
            && spec.rows.is_multiple_of(gemm)
            && spec.start_pos.is_multiple_of(gemm)
            && spec.head_dim.is_multiple_of(gemm)
            && spec.total().is_multiple_of(ATTN_KT_ROWS)
        {
            return self.attention_gemm(q, keys, values, spec, out);
        }

        // The blocked kernel masks its one diagonal tile with `tril`, which is
        // the causal mask only when that tile starts where the query block
        // does. A prompt into an empty cache always does; a continuation whose
        // cache is not a whole number of blocks deep does not, and takes the
        // row kernel, which needs no alignment.
        if spec.rows > 1 && spec.start_pos.is_multiple_of(block) {
            return self.with_kernel(
                &self.blocked,
                (spec.n_head, spec.group(), spec.head_dim),
                "attention_block",
                || attention_block_src(spec.n_head, spec.group(), spec.head_dim, block),
                |module| {
                    self.launch(
                        module,
                        "attention_block",
                        &[
                            (self.ptr(q, 0)?, [spec.rows as i64, qw]),
                            (self.ptr(keys, 0)?, [nk, kw]),
                            (self.ptr(values, 0)?, [nk, kw]),
                            (self.ptr(out, 0)?, [spec.rows as i64, qw]),
                        ],
                        (spec.rows.div_ceil(block) as u32, spec.n_head as u32, 1),
                    )
                },
            );
        }
        // Decoding leaves the row kernel a grid of n_head blocks, a sixth of
        // this card, each walking the whole cache. Splitting the keys fills it.
        let splits = ATTN_SPLITS;
        if spec.rows == 1 {
            let (part, ml) = self.attn_scratch(splits * spec.n_head, spec.head_dim)?;
            return self.with_kernel(
                &self.split_attn,
                (spec.n_head, spec.group(), spec.head_dim),
                "attention_split",
                || {
                    attention_split_src(
                        spec.n_head,
                        spec.group(),
                        spec.head_dim,
                        attention_tile(spec.head_dim),
                    )
                },
                |module| {
                    let rows = (splits * spec.n_head) as i64;
                    self.launch(
                        module,
                        "attention_split",
                        &[
                            (self.ptr(q, 0)?, [r, d]),
                            (self.ptr(keys, 0)?, [nk, kw]),
                            (self.ptr(values, 0)?, [nk, kw]),
                            (part, [rows, d]),
                            (ml, [rows, 2]),
                        ],
                        (spec.n_head as u32, splits as u32, 1),
                    )?;
                    self.launch(
                        module,
                        "attention_merge",
                        &[
                            (part, [rows, d]),
                            (ml, [rows, 2]),
                            (self.ptr(out, 0)?, [r, d]),
                        ],
                        (spec.n_head as u32, 1, 1),
                    )
                },
            );
        }
        self.with_kernel(
            &self.attentions,
            (spec.n_head, spec.group(), spec.head_dim),
            "attention",
            || {
                attention_src(
                    spec.n_head,
                    spec.group(),
                    spec.head_dim,
                    attention_tile(spec.head_dim),
                )
            },
            |module| {
                self.launch(
                    module,
                    "attention",
                    &[
                        (self.ptr(q, 0)?, [r, d]),
                        (self.ptr(keys, 0)?, [nk, kw]),
                        (self.ptr(values, 0)?, [nk, kw]),
                        (self.ptr(out, 0)?, [r, d]),
                    ],
                    (spec.rows as u32, spec.n_head as u32, 1),
                )
            },
        )
    }

    fn gate_into(&self, x: Buf, gate: Buf) -> Result<()> {
        let len = self.len_of(x)?.min(self.len_of(gate)?);
        self.pointwise("gate_into", &[(x, 0), (gate, 0)], len)
    }

    fn delta_conv(&self, history: Buf, taps: Buf, mix: DeltaMix, packed: Buf) -> Result<()> {
        self.check_distinct("delta_conv", packed, &[history, taps]);
        let channels = mix.channels();
        let (pr, c) = ((mix.pad() + mix.rows) as i64, channels as i64);
        // The packed destination is one row per (position, head), the same
        // memory as one row per position with the heads side by side. Read that
        // way a program's `batch` positions of one head are a column window of
        // consecutive rows, so the strided store is an ordinary tile.
        let batch = delta_conv_batch(mix.rows);
        let (planes, width) = ((3 * mix.rows) as i64, (mix.heads * mix.head_dim) as i64);
        // Both fused layouts space the planes evenly, so the second's offset is
        // the whole spacing.
        let plane_stride = mix.planes[1];
        ensure!(
            mix.planes == [0, plane_stride, 2 * plane_stride],
            "delta_conv expects evenly spaced query, key and value planes"
        );
        let key = (
            mix.heads,
            mix.head_dim,
            mix.kernel,
            mix.head_stride,
            mix.normalize,
            mix.query_scale.to_bits(),
            mix.rows,
            batch,
        );
        self.with_kernel(
            &self.convs,
            key,
            "delta_conv",
            || {
                delta_conv_src(
                    mix.heads,
                    mix.head_dim,
                    mix.kernel,
                    mix.head_stride,
                    plane_stride,
                    mix.rows,
                    batch,
                    mix.normalize,
                    mix.query_scale,
                )
            },
            |module| {
                self.launch(
                    module,
                    "delta_conv",
                    &[
                        (self.ptr(history, 0)?, [pr, c]),
                        (self.ptr(taps, 0)?, [mix.kernel as i64, c]),
                        (self.ptr(packed, 0)?, [planes, width]),
                    ],
                    ((mix.rows / batch) as u32, mix.heads as u32, 3),
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn delta_gates(
        &self,
        decay_in: Buf,
        decay_at: usize,
        beta_in: Buf,
        beta_at: usize,
        rate: Buf,
        dt_bias: Buf,
        mix: DeltaMix,
        packed: Buf,
    ) -> Result<()> {
        self.check_distinct("delta_gates", packed, &[decay_in, beta_in, rate, dt_bias]);
        let (heads, span, gates) = (mix.heads, mix.span(), mix.gates());
        let tile_rows = (ELEM_TILE / 4 / heads).max(1);
        let (r, h) = (mix.rows as i64, heads as i64);
        self.with_kernel(
            &self.gates,
            heads,
            "delta_gates",
            || delta_gates_src(heads, tile_rows),
            |module| {
                self.launch(
                    module,
                    "delta_gates",
                    &[
                        (self.ptr(decay_in, decay_at)?, [r, h]),
                        (self.ptr(beta_in, beta_at)?, [r, h]),
                        (self.ptr(rate, 0)?, [1, h]),
                        (self.ptr(dt_bias, 0)?, [1, h]),
                        (self.ptr(packed, 3 * span)?, [r, h]),
                        (self.ptr(packed, 3 * span + gates)?, [r, h]),
                    ],
                    (mix.rows.div_ceil(tile_rows) as u32, 1, 1),
                )
            },
        )
    }

    fn delta_rule(
        &self,
        packed: Buf,
        rows: usize,
        heads: usize,
        head_dim: usize,
        state: Buf,
        out: Buf,
    ) -> Result<()> {
        self.check_distinct("delta_rule", out, &[packed, state]);
        ensure!(
            head_dim.is_multiple_of(DELTA_TN),
            "delta_rule needs a head dimension ({head_dim}) that is a multiple of {DELTA_TN}"
        );
        // The five operands are one allocation, each a descriptor over its own
        // window of it, which is the point of packing them.
        let (span, gates) = (rows * heads * head_dim, rows * heads);
        let (r, d) = ((rows * heads) as i64, head_dim as i64);

        // A prompt goes through the chunked form, which needs whole chunks and
        // a state slice dividing the head. Decoding is a single position, and
        // every other shape falls through to the sequential kernel.
        if rows.is_multiple_of(DELTA_CHUNK) && head_dim.is_multiple_of(DELTA_CHUNK_TN) {
            return self.delta_chunked(packed, rows, heads, head_dim, state, out);
        }

        self.with_kernel(
            &self.deltas,
            (heads, head_dim),
            "delta_rule",
            || {
                DELTA_SRC
                    .replace("{H}", &heads.to_string())
                    .replace("{D}", &head_dim.to_string())
                    .replace("{TN}", &DELTA_TN.to_string())
            },
            |module| {
                self.launch(
                    module,
                    "delta_rule",
                    &[
                        (self.ptr(packed, 0)?, [r, d]),
                        (self.ptr(packed, span)?, [r, d]),
                        (self.ptr(packed, 2 * span)?, [r, d]),
                        (self.ptr(packed, 3 * span)?, [r, 1]),
                        (self.ptr(packed, 3 * span + gates)?, [r, 1]),
                        (self.ptr(state, 0)?, [(heads * head_dim) as i64, d]),
                        (self.ptr(out, 0)?, [r, d]),
                    ],
                    (heads as u32, (head_dim / DELTA_TN) as u32, 1),
                )
            },
        )
    }

    fn add_into(&self, acc: Buf, add: Buf) -> Result<()> {
        let len = self.len_of(acc)?.min(self.len_of(add)?);
        self.pointwise("add_into", &[(acc, 0), (add, 0)], len)
    }

    fn swiglu_q(
        &self,
        gate: Buf,
        gate_at: usize,
        up: Buf,
        up_at: usize,
        out: Buf,
        len: usize,
    ) -> Result<QAct> {
        self.check_distinct("swiglu_q", out, &[gate, up]);
        ensure!(
            len.is_multiple_of(ELEM_TILE),
            "a quantizing SwiGLU needs a length ({len}) that is a multiple of {ELEM_TILE}"
        );
        let (act, qa_ptr, das_ptr) = self.act_slot(1, len)?;
        let rb = (len / RMS_LANE) as i64;
        let lane = RMS_LANE as i64;
        self.with_kernel(
            &self.gated_swiglu,
            SWIGLU_Q_BLOCKS,
            "swiglu_q",
            || swiglu_q_src(SWIGLU_Q_BLOCKS),
            |module| {
                self.launch(
                    module,
                    "swiglu_q",
                    &[
                        (self.ptr(gate, gate_at)?, [rb, lane]),
                        (self.ptr(up, up_at)?, [rb, lane]),
                        (self.ptr(out, 0)?, [rb, lane]),
                        (qa_ptr, [rb, lane]),
                        (das_ptr, [rb, 1]),
                    ],
                    ((len / ELEM_TILE) as u32, 1, 1),
                )
            },
        )?;
        Ok(act)
    }

    fn swiglu(
        &self,
        gate: Buf,
        gate_at: usize,
        up: Buf,
        up_at: usize,
        out: Buf,
        len: usize,
    ) -> Result<()> {
        self.check_distinct("swiglu", out, &[gate, up]);
        self.pointwise(
            "swiglu",
            &[(gate, gate_at), (up, up_at), (out, 0)],
            len.min(self.len_of(out)?),
        )
    }

    fn swiglu_planes(
        &self,
        gate: Plane,
        up: Plane,
        out: Buf,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        self.check_distinct("swiglu_planes", out, &[gate.buf, up.buf]);
        let tile = swiglu_2d_tile(width);
        self.with_kernel(
            &self.swiglu_planes,
            tile,
            "swiglu_2d",
            || swiglu_2d_src(tile),
            |module| {
                self.launch(
                    module,
                    "swiglu_2d",
                    &[
                        (
                            self.ptr(gate.buf, gate.offset)?,
                            [rows as i64, gate.pitch as i64],
                        ),
                        (self.ptr(up.buf, up.offset)?, [rows as i64, up.pitch as i64]),
                        (self.ptr(out, 0)?, [rows as i64, width as i64]),
                    ],
                    (rows as u32, (width / tile) as u32, 1),
                )
            },
        )
    }

    fn copy(
        &self,
        src: Buf,
        src_offset: usize,
        dst: Buf,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        self.check_distinct("copy", dst, &[src]);
        self.pointwise("copy", &[(src, src_offset), (dst, dst_offset)], len)
    }
}
