// `@tensorcore` takes mma.sync, which no 32-bit index reaches.
@index_bits(32)
@tensorcore
kernel scale(X: tensor<f32>[N], Y: tensor<f32>[N]) {
    Y[ctx.thread] = X[ctx.thread] * 2.0
}
