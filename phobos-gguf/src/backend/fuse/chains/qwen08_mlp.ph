// The 0.8B's MLP as one persistent kernel, written from the stage library
// as the fuse pass plans it: the norm held per block, the gate and up
// projections a 32-column unit each with their SwiGLU quantized to scratch,
// a grid barrier, then the down projection accumulated into the residual.
@launch(256)
@persistent(BLOCKS)
@autotune(BLOCKS in [192], NB0 in [32], IT1 in [1], UN1 in [112], OF2 in [3584], IT2 in [1], UN2 in [128], TN in [8])
kernel fused(X0F: tensor<f32>[32, 32],
             X1F: tensor<f32>[32, 32],
             Wq3: tensor<i8>[7168, 1024],
             Ws3: tensor<f32>[7168, 32],
             Wq8: tensor<i8>[1024, 3584],
             Ws8: tensor<f32>[1024, 112],
             X0: tensor<f32>[1, 1024]) {
  let p = ctx.block.x
  var Aq2: tile<i8>[32, 32]
  var As2: tile<f32>[32, 1]
  let Aq2L = flat(Aq2)
  let As2L = flat(As2)
  var Vq7F: tensor<i8>[112, 32]
  var Vs7F: tensor<f32>[112, 1]
  let Vq7 = flat(Vq7F)
  let Vs7 = flat(Vs7F)

  rms_norm_q(X0F, X1F, Aq2, As2, NB0, 0.000001000000)

  for i1 in range(0, IT1) {
    let u1 = p + i1 * ctx.blocks.x
    if u1 < UN1 {
      let at1 = u1 * 32
      var v1: tile<f32>[1, 32] = q8_proj(Aq2L, As2L, Wq3, Ws3, at1)
      let at2 = OF2 + u1 * 32
      var v2: tile<f32>[1, 32] = q8_proj(Aq2L, As2L, Wq3, Ws3, at2)
      swiglu_q(v1, v2, Vq7F, Vs7F, u1)
    }
  }

  grid_barrier()

  for i2 in range(0, IT2) {
    let u2 = p + i2 * ctx.blocks.x
    if u2 < UN2 {
      let t5 = u2 * TN
      q8_proj_add(X0, Vq7[0 :+ 1, :], Vs7[0 :+ 1, :], Wq8, Ws8, t5, TN)
    }
  }
}
