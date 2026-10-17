kernel k(X: tensor<f32>[N, 64], Y: tensor<f32>[N, 64]) {
  var x = X[0 :+ 16, :]
  rms_norm(x, 0.00001)
  Y[0 :+ 16, :] = x
}
