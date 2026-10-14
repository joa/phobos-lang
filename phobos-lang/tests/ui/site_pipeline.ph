// From website/src/content/docs/docs/attributes/pipeline.mdx.
@pipeline
kernel flat(A: tensor<f32>[N], C: tensor<f32>[N]) {
  let i = ctx.block.x
  C[i :+ 1] = A[i :+ 1] * 2.0
}
