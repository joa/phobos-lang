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
