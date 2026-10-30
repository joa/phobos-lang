@tensorcore(wmmma)
@launch(128)
kernel k(A: tensor<f32>[64, 64], B: tensor<f32>[64, 64], C: tensor<f32>[64, 64]) {
  var acc: tile<f32>[64, 64] = 0.0
  acc += dot(A[0 :+ 64, 0 :+ 64], B[0 :+ 64, 0 :+ 64])
  C[0 :+ 64, 0 :+ 64] = acc
}
