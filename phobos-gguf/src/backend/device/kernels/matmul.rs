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
";
