// Q8_0 dequantizing matvec: the decode-step projection with the weights left
// quantized in memory.
//
// GGUF Q8_0 stores weights as signed bytes with one f32 scale per block of 32
// values along the contraction axis. Keeping them as i8 all the way into the
// kernel is the point: a decode step is bound by weight bandwidth, and i8 reads
// a quarter of the bytes that the dequantized f32 copy would.
//
// The scale factors out of the block's dot product, so it applies once per
// block per output column rather than once per weight.
@autotune(TILE_N in [64], BLK in [32])
@launch(256)
kernel q8_matvec(A: tensor<f32>[1, K],
                 W: tensor<i8>[K, N],
                 S: tensor<f32>[KB, N],
                 C: tensor<f32>[1, N]) {
  let pn = program_id(0)
  var acc: tile<f32>[1, TILE_N] = 0.0

  for k in range(0, K, BLK) {
    let b = k / BLK
    let a = A[0 :+ 1, k :+ BLK]
    let w = W[k :+ BLK, pn * TILE_N :+ TILE_N]
    let s = S[b :+ 1, pn * TILE_N :+ TILE_N]

    var wf: tile<f32>[BLK, TILE_N] = f32(w)
    var part: tile<f32>[1, TILE_N] = dot(a, wf)
    acc += part * s
  }

  C[0 :+ 1, pn * TILE_N :+ TILE_N] = acc
}
