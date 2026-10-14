// From website/src/content/docs/docs/ctx/uniformity.mdx.
@launch(256)
kernel k(A: tensor<f32>[1, 32], O: tensor<f32>[1, 32]) {
  var t: tile<f32>[1, 32] = A[0 :+ 1, 0 :+ 32]
  if ctx.lane == 0 {
    t = t * 2.0
  }
  O[0 :+ 1, 0 :+ 32] = t
}
