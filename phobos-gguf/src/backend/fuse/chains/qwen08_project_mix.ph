// The 0.8B's delta-rule input projection with its mix as one persistent
// kernel, written from the stage library: the projection's two runs, then
// per head the short convolution and its norms, and per value head the two
// gates. The first run writes the newest row of the history the convolution
// reads, which is what its grid barrier orders.
@launch(256)
@persistent(BLOCKS)
@autotune(BLOCKS in [192], NB0 in [32], IT1 in [1], UN1 in [192], DO1 in [18432], IT2 in [1], UN2 in [65], OF2 in [6144], DO2 in [6144], IT3 in [1], UN3 in [48], KS in [4], HD in [128], NH in [16], PS in [2048], ST in [128], GH4 in [16])
kernel fused(X0F: tensor<f32>[32, 32],
             X1F: tensor<f32>[32, 32],
             Wq9: tensor<i8>[8224, 1024],
             Ws9: tensor<f32>[8224, 32],
             X3: tensor<f32>[1, 8224],
             X2G: tensor<f32>[4, 6144],
             X4G: tensor<f32>[4, 6144],
             X5: tensor<f32>[1, 6176],
             X6: tensor<f32>[1, 16],
             X7: tensor<f32>[1, 16]) {
  let p = ctx.block.x
  let X2 = flat(X2G)
  var Aq8: tensor<i8>[32, 32]
  var As8: tensor<f32>[32, 1]
  let Aq8L = flat(Aq8)
  let As8L = flat(As8)

  rms_norm_q(X0F, X1F, Aq8, As8, NB0, 0.000001000000)

  for i1 in range(0, IT1) {
    let u1 = p + i1 * ctx.blocks.x
    if u1 < UN1 {
      let at1 = u1 * 32
      let to1 = DO1 + u1 * 32
      q8_proj_to(X2, Aq8L, As8L, Wq9, Ws9, at1, to1)
    }
  }

  for i2 in range(0, IT2) {
    let u2 = p + i2 * ctx.blocks.x
    if u2 < UN2 {
      let at2 = OF2 + u2 * 32
      let to2 = DO2 + u2 * 32
      q8_proj_to(X3, Aq8L, As8L, Wq9, Ws9, at2, to2)
    }
  }

  // the compiler places a grid barrier here

  for i3 in range(0, IT3) {
    let u3 = p + i3 * ctx.blocks.x
    if u3 < UN3 {
      delta_conv(X2G, X4G, X5, u3, 128, 0.088388346, 0.000000000001)
      if u3 < GH4 {
        delta_gates(X3, X7, X6, X5, u3, 8192, 8208, 6144, 6160)
      }
    }
  }
}
