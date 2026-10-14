// Not a parse error: until spans reach the IR, a later error has no location.
@unroll(4)
kernel copy(X: tensor<f32>[N], Y: tensor<f32>[N]) {
    Y[ctx.thread] = X[ctx.thread]
}
