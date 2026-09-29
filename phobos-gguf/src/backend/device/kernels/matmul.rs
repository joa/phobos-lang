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
/// band; see [`matmul::TC_TEMPLATE`] for the shape.
pub(crate) const TC_TILE_M: usize = matmul::TC_TILE_M;

pub(crate) const TC_TILE_N: usize = matmul::TC_TILE_N;

pub(crate) const TC_TILE_K: usize = matmul::TC_TILE_K;

pub(crate) const MATMUL_TC_SRC: &str = matmul::TC_TEMPLATE;

/// The matmul kernels reading an f16 weight, for the strip a `_qdecode`
/// writes. Free on the tensor-core path, which stages to f16 anyway.
pub(crate) fn matmul_f16w_src() -> String {
    MATMUL_SRC.replace("B: tensor<f32>[K, N]", "B: tensor<f16>[K, N]")
}

pub(crate) fn matmul_tc_f16w_src() -> String {
    MATMUL_TC_SRC.replace("B: tensor<f32>[K, N]", "B: tensor<f16>[K, N]")
}

/// The single-row matmul for decoding. It always reads row zero; a caller
/// wanting row `r` offsets the operand pointers, so the kernel needs no
/// scalar arguments.
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

/// Program count a narrow dense matvec splits `k` to reach, and the
/// shortest slice it cuts. Without a split, a projection a few dozen
/// outputs wide is one program walking all of `k`.
pub(crate) const MV_SPLIT_TARGET: usize = 96;
const MV_SPLIT_MIN_SLICE: usize = 64;

/// Slices of `k` for a `[1, k] x [k, n]` dense projection. One when the
/// output grid already fills the card, else enough to reach
/// [`MV_SPLIT_TARGET`] programs, with whole `TILE_K` steps per slice.
pub(crate) fn mv_splits(n: usize, k: usize) -> usize {
    let grid = n.div_ceil(MV_TN);
    let mut splits = (MV_SPLIT_TARGET / grid).min(k / MV_SPLIT_MIN_SLICE);
    while splits > 1 && !k.is_multiple_of(splits * TILE_K) {
        splits -= 1;
    }
    splits.max(1)
}

/// Rows per program of a prompt's row-major contraction. The leftover rows
/// take one program each.
pub(crate) const ROWS_TM: usize = 16;

/// Most outputs per program of a prompt's row-major contraction. Tiles are
/// small so a narrow projection over a short prompt still gets many
/// programs.
pub(crate) const ROWS_TN: usize = 16;

/// Program count below which a prompt's row-major contraction splits `k`,
/// and the most slices it cuts. Splitting helps a short prompt, which is
/// latency bound; a long one only pays an extra pass over the partials.
pub(crate) const ROWS_SPLIT_TARGET: usize = 192;
pub(crate) const ROWS_MAX_SPLITS: usize = 8;

/// Widest `k` [`matvec_blocks_src`] accepts. The row, one weight row and
/// the partial sums must fit in static shared memory.
pub(crate) const BLOCKS_MAX_K: usize = 5120;

/// The row-major contractions, and the tile each is generated for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RowsKernel {
    /// [`matvec_blocks_src`], a decode row against a weight `k` wide.
    Blocks { k: usize, tn: usize },
    /// [`matmul_rows_src`], or [`matmul_rows_split_src`] for a prompt's
    /// band.
    Tiles { tm: usize, tn: usize },
}

/// `C = A W^T` for a weight stored row-major as `[n, k]`, as in the file.
/// Each program computes a `tm` by `tn` tile.
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

/// [`matmul_rows_src`] with `k` split into `S` slices, one per program. The
/// partials, `[S * M, N]`, are summed by `q8_reduce`.
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

/// One row against a `[n, k]` weight, `tn` outputs per program.
///
/// The row and each weight row are viewed as `[k / 32, 32]`, so `k` is
/// spread across the threads, one 32-element block each, and a final sum
/// closes each output. The output tile alone is too narrow to keep the
/// threads busy.
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
