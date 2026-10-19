@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32], U in [1, 8], T in [256, 128], IB in [32, 64])
@launch(T)
@index_bits(IB)
@pipeline
@aligned(M = TILE_M, N = TILE_N, K = TILE_K, SM = TILE_M)
kernel gemm_split(A: tensor<f32>[M, K],
                  B: tensor<f32>[K, N],
                  P: tensor<f32>[SM, N]) {
  let pm = ctx.block.x
  let pn = ctx.block.y
  let ps = ctx.block.z
  let slice = K / TILE_K / (SM / M) * TILE_K
  let from = ps * slice

  var acc: tile<f32>[TILE_M, TILE_N] = 0.0

  for<unroll U> kt in range(from, from + slice, TILE_K) {
    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
  }

  P[ps * M + pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}

@autotune(RM in [32, 32], RN in [32, 32])
@launch(256)
@aligned(M = RM, N = RN, SM = RM)
kernel gemm_reduce(P: tensor<f32>[SM, N],
                   C: tensor<f32>[M, N],
                   alpha: f32,
                   beta: f32) {
  let pm = ctx.block.x
  let pn = ctx.block.y

  var acc: tile<f32>[RM, RN] = 0.0

  for s in range(0, SM / M, 1) {
    acc += P[s * M + pm * RM :+ RM, pn * RN :+ RN]
  }

  if beta == 0.0 {
    C[pm * RM :+ RM, pn * RN :+ RN] = alpha * acc
  } else {
    let c_old = C[pm * RM :+ RM, pn * RN :+ RN]
    C[pm * RM :+ RM, pn * RN :+ RN] = alpha * acc + beta * c_old
  }
}
