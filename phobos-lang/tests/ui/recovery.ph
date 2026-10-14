// Three mistakes that do not depend on each other, each reported once.
@launch(64)
kernel scale(X: tensor<f32>[1, 64], Y: tensor<f32>[1, 64]) {
    let s = 2.0 * * 3.0
    Y[0, ctx.thread] = X[0, ctx.thread] * s s
}

kernel copy(X: tensor<f32>[N], Y: tensor<f32>[N]) {
    let i = ctx.block.x * ctx.threads + ctx.thread
    if i < N {
        Y[i] = X[i] +
    }
}

kernel fill(Y tensor<f32>[N]) {
    Y[ctx.thread] = 0.0
}
