// From website/src/content/docs/docs/attributes/index.mdx.
@unroll
@autotune(BLOCK in [1024])
kernel copy(x: tensor<f32>[N], y: tensor<f32>[N]) {
  let base = ctx.block.x * BLOCK
  y[base :+ BLOCK] = x[base :+ BLOCK]
}
