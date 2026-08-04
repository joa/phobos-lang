// FlashAttention-2 implementation (fp32)
// see https://tridao.me/publications/flash2/flash2.pdf
@cluster(BR in [1024, 4096])
@pipeline
@tensorcore
@launch(128)
@autotune(D in [64], BR in [4, 128], BC in [4, 128])
@aligned(Nq = BR, Nk = BC)
kernel flash_attention(Q: tensor<f32>[Nq, D],
                       K: tensor<f32>[Nk, D],
                       V: tensor<f32>[Nk, D],
                       O: tensor<f32>[Nq, D],
                       scale: f32) {
  let pid = program_id(0)
  let row = pid * BR
  let q = Q[row :+ BR, :]

  var acc: tile<f32>[BR, D] = 0.0       // unnormalized output sum p*v
  var m: tile<f32>[BR, 1] = -300000000.0  // running row max (~ -inf)
  var l: tile<f32>[BR, 1] = 0.0         // running denominator sum p

  for kt in range(0, Nk, BC) {
    let k = K[kt :+ BC, :]
    let v = V[kt :+ BC, :]

    var s: tile<f32>[BR, BC] = dot_t(q, k)   // [BR, BC] = q @ k.T
    s = s * scale

    // online softmax update
    var mnew: tile<f32>[BR, 1] = rowmax(s)
    mnew = tmax(m, mnew)                      // new running max
    s = exp(s - mnew)                         // scores -> probabilities, fused in place
    var corr: tile<f32>[BR, 1] = exp(m - mnew) // rescale factor for old state

    l = l * corr
    l += rowsum(s)

    acc = acc * corr                         // rescale carried output
    acc += dot(s, v)                         // add this block's contribution

    m = mnew
  }

  acc = acc / l                              // normalize
  O[row :+ BR, :] = acc
}
