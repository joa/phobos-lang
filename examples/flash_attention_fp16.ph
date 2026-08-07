// FlashAttention-2 implementation
// see https://tridao.me/publications/flash2/flash2.pdf
//
// Half-precision FlashAttention-2: fp16 Q/K/V/O with an fp32 softmax state.
//
// This is fp16 tensor core with fp32 accumulation.
@cluster(BR in [1024, 4096])
@pipeline
@tensorcore
@launch(128)
@autotune(D in [64], BR in [4, 128], BC in [4, 128])
@aligned(Nq = BR, Nk = BC)
kernel flash_attention(Q: tensor<f16>[Nq, D],
                       K: tensor<f16>[Nk, D],
                       V: tensor<f16>[Nk, D],
                       O: tensor<f16>[Nq, D],
                       scale: f32) {
  let pid = program_id(0)
  
  let row = pid * BR
  let q = Q[row :+ BR, :]

  var acc: tile<f32>[BR, D] = 0.0       // unnormalized output sum p*v
  var m: tile<f32>[BR, 1] = -65504.0    // running row max (f16 min, ~ -inf)
  var l: tile<f32>[BR, 1] = 0.0         // running denominator sum p

  for kt in range(0, Nk, BC) {
    let k = K[kt :+ BC, :]
    let v = V[kt :+ BC, :]

    var s: tile<f32>[BR, BC] = dot_t(q, k)   // [BR, BC] = q @ k.T (f16 -> f32)
    s = s * scale

    // online softmax update (f32)
    var mnew: tile<f32>[BR, 1] = rowmax(s)
    mnew = tmax(m, mnew)                      // new running max
    s = exp(s - mnew)                         // scores -> probabilities, fused in place
    var corr: tile<f32>[BR, 1] = exp(m - mnew) // rescale factor for old state

    l = l * corr
    l += rowsum(s)

    acc = acc * corr                         // rescale carried output
    acc += dot(s, v)                         // add this block's contribution (f16 V)

    m = mnew
  }

  acc = acc / l                              // normalize
  O[row :+ BR, :] = acc                      // f32 -> f16 store
}
