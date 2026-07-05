@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [16, 64])
@launch(256, 2)
@pipeline
@tensorcore
kernel gemm(A: tensor<f16>[M, K],
            B: tensor<f16>[K, N],
            C: tensor<f16>[M, N],
            alpha: f32,
            beta: f32) {
  let pm = program_id(0)
  let pn = program_id(1)

  var acc: tile<f16>[TILE_M, TILE_N] = 0.0
  
  for kt in range(0, K, TILE_K) {
    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
  }
  
  let c_old = C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N]
  C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = alpha * acc + beta * c_old
}
