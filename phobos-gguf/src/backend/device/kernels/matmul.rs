// Dense matmul and matvec kernel sources, and their tile sizes.

use phobos_kernels::matmul;

/// Output tile and k-slice for the tiled matmul.
pub(crate) const TILE_M: usize = matmul::TILE_M;

pub(crate) const TILE_N: usize = matmul::TILE_N;

pub(crate) const TILE_K: usize = matmul::TILE_K;

/// Output tile along N for the single-row matmul.
pub(crate) const MV_TN: usize = 128;

pub(crate) const MATMUL_SRC: &str = matmul::TEMPLATE;

/// Tensor-core tile and source for [`super::DeviceBackend::matmul`]'s deep
/// band; see [`matmul::TC_TEMPLATE`]'s own doc for the shape reasoning.
pub(crate) const TC_TILE_M: usize = matmul::TC_TILE_M;

pub(crate) const TC_TILE_N: usize = matmul::TC_TILE_N;

pub(crate) const TC_TILE_K: usize = matmul::TC_TILE_K;

pub(crate) const MATMUL_TC_SRC: &str = matmul::TC_TEMPLATE;

/// The same two kernels reading an f16 weight, for the strip a `_qdecode`
/// writes. Free on the tensor-core path, which stages to f16 anyway.
pub(crate) fn matmul_f16w_src() -> String {
    MATMUL_SRC.replace("B: tensor<f32>[K, N]", "B: tensor<f16>[K, N]")
}

pub(crate) fn matmul_tc_f16w_src() -> String {
    MATMUL_TC_SRC.replace("B: tensor<f32>[K, N]", "B: tensor<f16>[K, N]")
}

/// The single-row specialization decoding needs. Always reads row zero: a
/// caller wanting row `r` offsets the operand pointers instead, which keeps the
/// kernel free of scalar arguments.
pub(crate) const MATVEC_SRC: &str = "\
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

@launch(256)
@autotune(TILE_N in [128], TILE_K in [16])
{ALIGNED}
kernel matvec_split(A: tensor<f32>[M, K], B: tensor<f32>[K, N], P: tensor<f32>[S, N]) {
  let pn = program_id(0)
  let ps = program_id(1)
  let slice = K / S
  let from = ps * slice
  var acc: tile<f32>[1, TILE_N] = 0.0
  for kt in range(from, from + slice, TILE_K) {
    var a = A[0 :+ 1, kt :+ TILE_K]
    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
  }
  P[ps :+ 1, pn * TILE_N :+ TILE_N] = acc
}
";

/// Programs a narrow dense matvec splits `k` to reach, and the shortest
/// slice it will cut. A projection a few dozen outputs wide is one program
/// unsplit and walks all of `k` alone.
pub(crate) const MV_SPLIT_TARGET: usize = 96;
const MV_SPLIT_MIN_SLICE: usize = 64;

/// Slices for a `[1, k] x [k, n]` dense projection: one where the output
/// grid already fills the card, else as many as reach
/// [`MV_SPLIT_TARGET`] programs with whole `TILE_K` steps a slice.
pub(crate) fn mv_splits(n: usize, k: usize) -> usize {
    let grid = n.div_ceil(MV_TN);
    let mut splits = (MV_SPLIT_TARGET / grid).min(k / MV_SPLIT_MIN_SLICE);
    while splits > 1 && !k.is_multiple_of(splits * TILE_K) {
        splits -= 1;
    }
    splits.max(1)
}

/// Rows a prompt's program of the row-major contraction takes; the rest go
/// a row a program.
pub(crate) const ROWS_TM: usize = 16;

/// Outputs a prompt's program takes at most: small tiles, many programs, as
/// a narrow projection over a short prompt has little else to spread over.
pub(crate) const ROWS_TN: usize = 16;

/// Programs below which a prompt's row-major contraction splits `k`, and the
/// most slices it cuts: every program walks its slice alone, so a short
/// prompt is latency bound without them, and a long one only pays the
/// extra pass over the partials.
pub(crate) const ROWS_SPLIT_TARGET: usize = 192;
pub(crate) const ROWS_MAX_SPLITS: usize = 8;

/// Widest `k` [`matvec_blocks_src`] stages whole: the row and one output's
/// weight row, beside the partial sums, within the static shared memory.
pub(crate) const BLOCKS_MAX_K: usize = 5120;

/// The row-major contractions, and the tile each is generated for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RowsKernel {
    /// [`matvec_blocks_src`], a decode row against a weight `k` wide.
    Blocks { k: usize, tn: usize },
    /// [`matmul_rows_src`], and for a prompt's band
    /// [`matmul_rows_split_src`] in the same module.
    Tiles { tm: usize, tn: usize },
}

/// `C = A W^T` for a weight held row-major as the file has it, `[n, k]`, a
/// `tm` by `tn` tile a program, the weight tile transposed in the
/// contraction.
pub(crate) fn matmul_rows_src(tm: usize, tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TM in [{tm}], TN in [{tn}], TK in [16])
@aligned(M = TM, N = TN, K = TK)
kernel matmul_rows(A: tensor<f32>[M, K], W: tensor<f32>[N, K], C: tensor<f32>[M, N]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, K, TK) {{
    var a = A[pm * TM :+ TM, kt :+ TK]
    var w = W[pn * TN :+ TN, kt :+ TK]
    acc += dot(a, transpose(w))
  }}
  C[pm * TM :+ TM, pn * TN :+ TN] = acc
}}
"
    )
}

/// [`matmul_rows_src`] over one of `S` slices of `k` a program, the partials
/// `[S * M, N]` for `q8_reduce` to sum.
pub(crate) fn matmul_rows_split_src(tm: usize, tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TM in [{tm}], TN in [{tn}], TK in [16])
@aligned(M = TM, N = TN, K = TK)
kernel matmul_rows_split(A: tensor<f32>[M, K], W: tensor<f32>[N, K], P: tensor<f32>[SM, N]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  let ps = program_id(2)
  let slice = K / (SM / M)
  let from = ps * slice
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(from, from + slice, TK) {{
    var a = A[pm * TM :+ TM, kt :+ TK]
    var w = W[pn * TN :+ TN, kt :+ TK]
    acc += dot(a, transpose(w))
  }}
  P[ps * M + pm * TM :+ TM, pn * TN :+ TN] = acc
}}
"
    )
}

/// One row against a `[n, k]` weight, `tn` outputs a program. The row and
/// each weight row are viewed `[k / 32, 32]`, so `k` runs across the threads
/// a 32-element block apiece and one closing sum per output ends it: a
/// contraction's own output tile is too narrow to keep the threads busy.
pub(crate) fn matvec_blocks_src(k: usize, tn: usize) -> String {
    let kb = k / 32;
    format!(
        "@launch(256)
@autotune(TN in [{tn}], KB in [{kb}])
@aligned(AB = KB, WB = KB, N = TN)
kernel matvec_blocks(A: tensor<f32>[AB, 32], W: tensor<f32>[WB, 32], C: tensor<f32>[N, D1]) {{
  let pn = program_id(0)
  var a = A[0 :+ KB, :]
  var parts: tile<f32>[KB, TN] = 0.0
  for j in range(0, TN) {{
    let o = pn * TN + j
    parts[:, j :+ 1] = rowsum(a * W[o * KB :+ KB, :])
  }}
  C[pn * TN :+ TN, 0 :+ 1] = rowsum(transpose(parts))
}}
"
    )
}
