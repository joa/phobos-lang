kernel k(X: tensor<f32>[1, 256]) {
  var H: tensor<f32>[8, 32]
  H[0 :+ 1, 0 :+ 32] = X[0 :+ 1, 0 :+ 32]
}
