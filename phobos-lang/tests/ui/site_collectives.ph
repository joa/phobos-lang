// From website/src/content/docs/docs/ctx/collectives.mdx.
@launch(128)
kernel k(A: tensor<f32>[1, 128], O: tensor<f32>[1, 1]) {
  let x = A[0, ctx.thread]
  var r = 0.0
  if ctx.lane == 0 {
    r = sum<over ctx.lane>(x)
  }
  if ctx.thread == 0 {
    O[0, 0] = r
  }
}
