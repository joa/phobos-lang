// An error about a use of a name points back at its declaration: a `let`
// assigned to, and a store of the wrong type.
@launch(128)
kernel k(A: tensor<f32>[N], B: tensor<i32>[N]) {
    let x = A[ctx.thread]
    x = 2.0
    var m = 0
    m = x
    B[ctx.thread] = x
}
