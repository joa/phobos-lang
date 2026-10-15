// The 0.8B's delta-rule input projection as one persistent kernel, written
// from the stage library: the norm, then the channel and gate runs of one
// Q8_0 weight, 32 columns a unit. Every block writes the whole normed row,
// so it is held and no barrier is needed.
@launch(256)
@persistent(BLOCKS)
@autotune(BLOCKS in [192], NB0 in [32], IT1 in [1], UN1 in [192], DO1 in [18432], IT2 in [1], UN2 in [65], OF2 in [6144], DO2 in [6144])
kernel fused(X0F: tensor<f32>[32, 32],
             X1F: tensor<f32>[32, 32],
             Wq5: tensor<i8>[8224, 1024],
             Ws5: tensor<f32>[8224, 32],
             X2: tensor<f32>[1, 24576],
             X3: tensor<f32>[1, 8224]) {
  let p = ctx.block.x
  var Aq4: tensor<i8>[32, 32]
  var As4: tensor<f32>[32, 1]
  let Aq4L = flat(Aq4)
  let As4L = flat(As4)

  rms_norm_q(X0F, X1F, Aq4, As4, NB0, 0.000001000000)

  for i1 in range(0, IT1) {
    let u1 = p + i1 * ctx.blocks.x
    if u1 < UN1 {
      let at1 = u1 * 32
      let to1 = DO1 + u1 * 32
      q8_proj_to(X2, Aq4L, As4L, Wq5, Ws5, at1, to1)
    }
  }

  for i2 in range(0, IT2) {
    let u2 = p + i2 * ctx.blocks.x
    if u2 < UN2 {
      let at2 = OF2 + u2 * 32
      let to2 = DO2 + u2 * 32
      q8_proj_to(X3, Aq4L, As4L, Wq5, Ws5, at2, to2)
    }
  }
}
