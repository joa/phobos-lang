kernel k(X: tensor<f32>[N], Y: tensor<f32>[N]) {
    let i = ctx.thread
    if i < N {
        Y[i] = X[i]
}
