// A generated tile is computed afresh at each read, so the function that
// computes it may read memory but not write it.
fn wq(j, q) {
    let v = word(W[j, 4 * q :+ 4], 0)
    C[0, 0] = 1.0
    v
}

@launch(128)
kernel k(A: tensor<i8>[128, 32], AS: tensor<f32>[128, 1],
         W: tensor<i8>[64, 32], WS: tensor<f32>[1, 64],
         C: tensor<f32>[128, 64]) {
    var acc: tile<f32>[128, 64] = 0.0
    for kt in range(0, 32, 32) {
        let a = A[0 :+ 128, 0 :+ 32]
        let w: tile<i8>[64, 32] = generate(wq)
        let ws = WS[0 :+ 1, 0 :+ 64]
        let da = AS[0 :+ 128, 0 :+ 1]
        acc += f32(dot_t(a, w)) * ws * da
    }
    C[0 :+ 128, 0 :+ 64] = acc
}
