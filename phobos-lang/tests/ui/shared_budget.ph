// A tile larger than the 48 KB a block may hold statically.
kernel big(X: tensor<f32>[N, 256], Y: tensor<f32>[N, 256]) {
  let r = ctx.block.x * 64
  var x: tile<f32>[64, 256] = X[r :+ 64, :]
  Y[r :+ 64, :] = x * 2.0
}
