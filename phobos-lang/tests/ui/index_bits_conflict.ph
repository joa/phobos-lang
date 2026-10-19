// The kernels of one module share one index width.
@autotune(IB in [32, 64])
@index_bits(IB)
kernel first(X: tensor<f32>[N]) {
    X[ctx.thread] = 1.0
}

@index_bits(64)
kernel second(X: tensor<f32>[N]) {
    X[ctx.thread] = 2.0
}
