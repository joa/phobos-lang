// Eight f16 rows of 256 bytes each, staged into shared memory. Under
// @padstage each row is padded by eight elements.
kernel stage(K: tensor<f16>[64, 128], O: tensor<f16>[8, 128]) {
  var k = K[0 :+ 8, 0 :+ 128]
  O[0 :+ 8, 0 :+ 128] = k
}
