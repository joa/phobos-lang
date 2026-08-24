pub const TILE_M: usize = 32;
pub const TILE_N: usize = 32;
pub const TILE_K: usize = 16;

pub fn shapes() -> [(&'static str, usize); 3] {
    [("TILE_M", TILE_M), ("TILE_N", TILE_N), ("TILE_K", TILE_K)]
}

pub const TEMPLATE: &str = "\
@launch(256)
@autotune(TILE_M in [32], TILE_N in [32], TILE_K in [16])
{ALIGNED}
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TILE_M, TILE_N] = 0.0
  for kt in range(0, K, TILE_K) {
    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
  }
  C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}
";

pub fn src() -> String {
    TEMPLATE.replace("{ALIGNED}\n", "")
}

/// Tensor-core tile: a 64x64 output block over 4x4 sixteen-square WMMA
/// fragments, the smallest square `wmma_plan` accepts at `@launch(256)`'s
/// eight warps. Only launched over whole `TC_TILE_M`-row bands; a caller with
/// a ragged `m` hands the remainder to [`TEMPLATE`] above.
pub const TC_TILE_M: usize = 64;
pub const TC_TILE_N: usize = 64;
pub const TC_TILE_K: usize = 16;

pub const TC_TEMPLATE: &str = "\
@launch(256)
@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
@tensorcore
@aligned(M = TILE_M, N = TILE_N, K = TILE_K)
kernel matmul_tc(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TILE_M, TILE_N] = 0.0
  for kt in range(0, K, TILE_K) {
    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
  }
  C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}
";
