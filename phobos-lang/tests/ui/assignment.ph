kernel k(A: tensor<f32>[N], B: tensor<f32>[N]) {
    1 = 2
    A[0][1] = 1.0
    A[0] = (B[0] + 1.0
}
