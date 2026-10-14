// From website/src/content/docs/docs/attributes/swizzle.mdx.
@launch(128)
@swizzle
@aligned(M = 64, N = 64, K = 128)
kernel proj(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
            W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
            C: tensor<f32>[M, N], E: tensor<i8>[N, K]) {
  let pm = ctx.block.x
  let pn = ctx.block.y
  var w: tile<i8>[64, 128]
  var acc: tile<f32>[64, 64] = 0.0
  for kt in range(0, K, 128) {
    w = W[pn * 64 :+ 64, kt :+ 128]
    barrier()
    let a0 = A[pm * 64 :+ 64, kt :+ 32]
    let w0 = w[0 :+ 64, 0 :+ 32]
    let da = AS[pm * 64 :+ 64, kt / 32 :+ 1]
    let ws = WS[kt / 32 :+ 1, pn * 64 :+ 64]
    acc += f32(dot_t(a0, w0)) * ws * da
    E[pn * 64 :+ 64, kt :+ 128] = w
  }
  C[pm * 64 :+ 64, pn * 64 :+ 64] = acc
}
