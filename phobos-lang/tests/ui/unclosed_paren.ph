kernel k(X: tensor<f32>[N], Y: tensor<f32>[N]) {
    Y[ctx.thread] = exp(X[ctx.thread] * 2.0
}
