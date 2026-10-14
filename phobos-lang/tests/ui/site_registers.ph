// From website/src/content/docs/docs/attributes/registers.mdx.
@launch(256)
@registers(acc)
kernel k(O: tensor<f32>[3, 5]) {
  var acc: tile<f32>[3, 5] = 0.0
  O[0 :+ 3, 0 :+ 5] = acc
}
