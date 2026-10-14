// Three mistakes the build finds, each once: `y` uses `x`, whose
// declaration failed, and says nothing more.
kernel k(A: tensor<f32>[N], B: tensor<f32>[N]) {
    let x = nope + 1
    let y = x * 2
    let z = wobble(A)
    for i in range(0, N) {
        B[i] = A
    }
    B[ctx.thread] = A[ctx.thread] + y
}
