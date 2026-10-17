// A library function that writes its argument needs a `var`; the error
// points at the call, not into the library.
kernel k(X: tensor<f32>[N, 64], Y: tensor<f32>[N, 64]) {
  let x = X[0 :+ 16, :]
  softmax(x)
  Y[0 :+ 16, :] = x
}
