// From website/src/content/docs/docs/ctx/counts.mdx.
@launch(48)
kernel half_a_warp(O: tensor<i32>[1, 1]) {
  O[0, 0] = i32(ctx.warps)
}
