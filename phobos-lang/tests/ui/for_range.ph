kernel k(Y: tensor<f32>[N]) {
    for i in steps(0, N) {
        Y[i] = 0.0
    }
    for<twice 2> i in range(0, N) {
        Y[i] = 1.0
    }
}
