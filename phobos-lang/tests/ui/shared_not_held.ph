@launch(256)
@persistent(BLOCKS)
@shared(V)
@autotune(BLOCKS in [8])
kernel k(X: tensor<f32>[1, 256], Y: tensor<f32>[1, 256]) {
  var V: tensor<f32>[1, 256]
  for i in range(0, 4) {
    let u = ctx.block.x + i * ctx.blocks.x
    if u < 32 {
      V[0 :+ 1, u * 8 :+ 8] = X[0 :+ 1, u * 8 :+ 8]
    }
  }
  for i in range(0, 4) {
    let u = ctx.block.x + i * ctx.blocks.x
    if u < 32 {
      Y[0 :+ 1, u * 8 :+ 8] = V[0 :+ 1, ((u + 1) % 32) * 8 :+ 8]
    }
  }
}
