kernel k(Y: tensor<f32>[N]) {
    Y[ctx.thread] = 1.0 # 2.0
    let big = 99999999999999999999
    let e = 1e
}
