// The pieces of attention: the online softmax that Flash attention streams
// the keys through, a whole pass over them, and rotary position embeddings.

// One block of keys' step of the online softmax, Flash attention's inner
// update. `s` (`[BR, BC]`) holds this block's scaled scores, masked where
// needed, and is overwritten with their unnormalized probabilities. The
// running state is the row maximum `m` and the denominator `l`, both
// `[BR, 1]`, and the unnormalized output `acc` (`[BR, D]`), which gains
// the block's values `v` (`[BC, D]`) weighted by `s`. Start with `m` at
// `-1.0e30`, `l` and `acc` at 0, and divide `acc` by `l` after the last
// block.
fn online_softmax(acc, m, l, s, v) {
  var mnew = rowmax(s)
  mnew = tmax(m, mnew)
  s = exp(s - mnew)
  var corr = exp(m - mnew)
  l = l * corr
  l += rowsum(s)
  acc = acc * corr
  acc += dot(s, v)
  m = mnew
}

// Attention of the queries `q` (`[BR, D]`) over every key of `K` and value
// of `V` (both `[Nk, D]` tensors), `BC` keys a step: `acc` (`[BR, D]`, `f32`)
// becomes `softmax(q K^T * scale) V`. `acc` must start at 0, as a
// declaration `var acc: tile<f32>[BR, D] = 0.0` leaves it, which keeps it in
// the tensor cores' registers under `@tensorcore`. A last block of keys
// past `Nk` is masked.
fn attention(acc, q, K, V, scale, BR, BC) {
  var m: tile<f32>[BR, 1] = -1.0e30
  var l: tile<f32>[BR, 1] = 0.0
  for kt in range(0, extent(K, 0), BC) {
    let k = K[kt :+ BC, :]
    let v = V[kt :+ BC, :]
    var s: tile<f32>[BR, BC] = dot_t(q, k)
    s = s * scale
    mask_cols(s, extent(K, 0) - kt, -1.0e30)
    online_softmax(acc, m, l, s, v)
  }
  acc = acc / l
}

// Rotary position embedding, in place, in the rotate-half form of GPT-NeoX
// and Hugging Face's LLaMA: each row of `x` (`[R, D]`) is split into halves
// `a` and `b`, and becomes `[a cos - b sin, b cos + a sin]`. `cos` and
// `sin` are `[R, D / 2]`, the angles of each row's position. GPT-J's form,
// which rotates adjacent pairs, is `rope_pairs`.
fn rope(x, cos, sin) {
  var a = x[:, 0 :+ extent(x, 1) / 2]
  var b = x[:, extent(x, 1) / 2 :+ extent(x, 1) / 2]
  x[:, 0 :+ extent(x, 1) / 2] = a * cos - b * sin
  x[:, extent(x, 1) / 2 :+ extent(x, 1) / 2] = b * cos + a * sin
}

// Rotary position embedding of the pairs `(x0, x1)`, in place: `x0` becomes
// `x0 cos - x1 sin` and `x1` becomes `x1 cos + x0 sin`, all four the same
// shape. GPT-J's interleaved pairs are two names when the kernel declares
// the heads as `[T, D / 2, 2]`: `x0` is `X[t :+ R, :, 0]` and `x1` is
// `X[t :+ R, :, 1]`, with `cos` and `sin` declared `[T, D / 2, 1]` and
// sliced the same way.
fn rope_pairs(x0, x1, cos, sin) {
  var a = x0
  var b = x1
  x0 = a * cos - b * sin
  x1 = b * cos + a * sin
}
