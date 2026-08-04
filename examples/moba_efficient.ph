// MoBA (Mixture of Block Attention), the "moba branch" of MoonshotAI's
// moba_efficient.py (https://github.com/MoonshotAI/MoBA).
@tensorcore
@launch(128)
@autotune(D in [64], BR in [16, 128], CHUNK in [16, 128])
kernel moba_attention(Q: tensor<f16>[Nq, D],
                      K: tensor<f16>[Nk, D],
                      V: tensor<f16>[Nk, D],
                      O: tensor<f16>[Nq, D],
                      scale: f32,
                      inv_chunk: f32) {
  let pid = program_id(0)
  let row = pid * BR
  let q = Q[row :+ BR, :]                   // query block stays resident

  var acc: tile<f32>[BR, D] = 0.0           // unnormalized output sum p*v
  var m: tile<f32>[BR, 1] = -65504.0        // running row max (f16 min)
  var l: tile<f32>[BR, 1] = 0.0             // running denominator sum p

  for kt in range(0, Nk, CHUNK) {
    let k = K[kt :+ CHUNK, :]
    let v = V[kt :+ CHUNK, :]

    // Mean-pooled key of this chunk: ones[1, CHUNK] @ K[CHUNK, D] -> [1, D],
    // then scaled by 1/CHUNK. This is the chunk's gate representative.
    var ones: tile<f16>[1, CHUNK] = 1.0
    let meank = dot(ones, k)                 // [1, D] chunk key sum

    // Router gate: affinity of each query to the chunk representative.
    var gate: tile<f32>[BR, 1] = dot_t(q, meank)   // [BR, 1]
    gate = gate * inv_chunk                  // sum -> mean

    // Block scores, biased by the router gate (the soft top-k relaxation:
    // a higher gate lifts every score in the chunk, so more of the query's
    // softmax mass lands on the chunks it routes to).
    var s: tile<f32>[BR, CHUNK] = dot_t(q, k)      // [BR, CHUNK] = q @ k.T
    s = s * scale
    s = s + gate                             // broadcast [BR, 1] over columns

    // Online softmax update.
    var mnew: tile<f32>[BR, 1] = rowmax(s)
    mnew = tmax(m, mnew)                      // new running max
    s = exp(s - mnew)                         // scores -> probabilities
    var corr: tile<f32>[BR, 1] = exp(m - mnew) // rescale factor for old state

    l = l * corr
    l += rowsum(s)

    acc = acc * corr                          // rescale carried output
    acc += dot(s, v)                          // add this chunk's contribution

    m = mnew
  }

  acc = acc / l                               // normalize
  O[row :+ BR, :] = acc                       // f32 -> f16 store
}
