// A name one typo away from a known one is offered in its place.
@launch(128)
kernel k(A: tensor<f32>[N], B: tensor<f32>[N]) {
    let scale = 2.0
    let x = A[ctx.thread] * scael
    B[ctx.thread] = sqtr(x)
}
