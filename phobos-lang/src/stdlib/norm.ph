// Normalizations over the rows of a rank-2 tile `[R, C]`, in place, as a
// transformer normalizes each token's hidden state. `x` must be a name. A
// weight or a bias is a `[1, C]` row, a tile or a tensor slice such as
// `G[0 :+ 1, :]`, or a scalar: `rms_norm(x, 1.0, eps)` has no weight.
//
// The sums are taken in the tile's own type, so normalize an `f16` input
// in an `f32` tile.

// Root-mean-square norm: `x / sqrt(mean(x^2) + eps) * w`, LLaMA's.
fn rms_norm(x, w, eps) {
  var r = rsqrt(rowmean(x * x) + eps)
  x = x * r * w
}

// Layer norm: `(x - mean(x)) / sqrt(var(x) + eps) * w + b`, the biased
// variance, as `torch.nn.functional.layer_norm`.
fn layer_norm(x, w, b, eps) {
  var mu = rowmean(x)
  x = x - mu
  var r = rsqrt(rowmean(x * x) + eps)
  x = x * r * w + b
}

// Batch norm as inference applies it: `(x - mean) / sqrt(variance + eps)
// * w + b`, with `mean`, `variance`, `w` and `b` the running statistics and
// the affine parameters, each a `[1, C]` row or a scalar. An expression
// function, so it fuses into the statement that stores it.
fn batch_norm(x, mean, variance, w, b, eps) = (x - mean) * rsqrt(variance + eps) * w + b

// The batch statistics of the rows of `x`, a batch of `R` samples: each
// column's mean into `mean` and its biased variance into `variance`, both
// `[1, C]` names. The statistics of a batch spread over several blocks are
// the merge of each block's.
fn batch_norm_stats(mean, variance, x) {
  mean = colmean(x)
  variance = colmean(square(x - mean))
}

// Each row scaled to unit length: `x / max(||x||, eps)`, as
// `torch.nn.functional.normalize`.
fn l2_normalize(x, eps) {
  var r = 1.0 / max(sqrt(rowsum(x * x)), eps)
  x = x * r
}
