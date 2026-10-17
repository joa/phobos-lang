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

// Each row scaled to unit length: `x / max(||x||, eps)`, as
// `torch.nn.functional.normalize`.
fn l2_normalize(x, eps) {
  var r = 1.0 / max(sqrt(rowsum(x * x)), eps)
  x = x * r
}
