kernel staged(X: tensor<f32>[N]) {
  let i = ctx.block.x
  X[i] = X[i] + 1.0
  grid_barrier()
}
