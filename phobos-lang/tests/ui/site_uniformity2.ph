// From website/src/content/docs/docs/ctx/uniformity.mdx.
@launch(256)
kernel k(A: tensor<f32>[1, 32], O: tensor<f32>[1, 32]) {
  var u = A[0, ctx.lane :+ 1]
  O[0 :+ 1, 0 :+ 1] = u
}
