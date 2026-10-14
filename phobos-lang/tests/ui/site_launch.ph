// From website/src/content/docs/docs/attributes/launch.mdx.
@launch(48)
kernel k(O: tensor<i32>[1, 1]) {
  O[0, 0] = i32(ctx.threads)
}
