kernel k(Y: tensor<i32>[N]) {
    Y[ctx.thread] = ctx.lanez
    let ctx = 1
}
