// Not a parse error: the build reports it, at the attribute.
@unroll(4)
kernel copy(X: tensor<f32>[N], Y: tensor<f32>[N]) {
    Y[ctx.thread] = X[ctx.thread]
}
