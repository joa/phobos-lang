// A literal the target cannot hold is an error rather than another value:
// an integer is an index, 32 bits wide here, and a float is an `f32`. A
// 32-bit word's bits, a mask up to 4294967295, fit.
kernel k(I: tensor<i64>[4, 4], X: tensor<f32>[4, 4], n: i32) {
    I[0, 0] = 5000000000 + n
    I[0, 1] = -2147483648 + n
    I[0, 2] = 4294967296 + n
    I[0, 3] = 2155905152 + n
    X[0, 0] = 1.0e39
    X[0, 1] = 1.0e-50
    X[0, 2] = 1.0e-40
}
