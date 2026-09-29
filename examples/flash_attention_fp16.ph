// FlashAttention-2 in fp16: https://tridao.me/publications/flash2/flash2.pdf
//
// Q, K, V and O are fp16. The softmax state and the tensor core
// accumulation are fp32.
@cluster(BR in [1024, 4096])
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
  var m: tile<f32>[BR, 1] = -65504.0    // running row max, from the f16 minimum
  var l: tile<f32>[BR, 1] = 0.0         // running denominator sum p

  for kt in range(0, Nk, BC) {
    let k = K[kt :+ BC, :]
    let v = V[kt :+ BC, :]

    var s: tile<f32>[BR, BC] = dot_t(q, k)   // [BR, BC] = q @ k.T (f16 -> f32)
    s = s * scale

    // Online softmax update.
    var mnew: tile<f32>[BR, 1] = rowmax(s)
    mnew = tmax(m, mnew)
    s = exp(s - mnew)                         // unnormalized probabilities
    var corr: tile<f32>[BR, 1] = exp(m - mnew) // rescales the carried state

    l = l * corr
    l += rowsum(s)

    acc = acc * corr                         // rescale carried output
    acc += dot(s, v)                         // add this block's contribution (f16 V)

    m = mnew
  }

  acc = acc / l
  O[row :+ BR, :] = acc                      // narrows to f16 on store
}
