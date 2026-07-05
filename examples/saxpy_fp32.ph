@autotune(BLOCK in [16, 8192])
kernel saxpy(x: tensor<f32>[N], y: tensor<f32>[N], out: tensor<f32>[N], alpha: f32) {
  let base = program_id(0) * BLOCK
  out[base :+ BLOCK] = alpha * x[base :+ BLOCK] + y[base :+ BLOCK]
}
