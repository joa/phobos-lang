// `min`, `max` and `pow` become operators as they parse, with two
// arguments; any other count is an error, not an unknown function.
kernel k(A: tensor<f32>[N], B: tensor<f32>[N]) {
    let x = A[ctx.thread]
    B[ctx.thread] = pow(x)
}
