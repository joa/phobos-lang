// Loss functions, after `torch.nn.functional`, one value a row: a row of
// `[R, C]` is one sample, and each loss gives `[R, 1]`, that sample's loss.
// PyTorch's `reduction="sum"` or `"mean"` is the kernel's to take, often
// together with the other blocks' rows.

// Cross-entropy of the logits `t` (`[R, C]`) against the class of each row,
// `target` (`[R, 1]`, `i32`): `logsumexp(t) - t[target]`, into `out`
// (`[R, 1]`). `t` is left unchanged.
fn cross_entropy(out, t, target) {
  logsumexp(out, t)
  out = out - rowsum(where(iota(1) == target, t, 0.0))
}

// The negative log-likelihood of each row's class `target` (`[R, 1]`,
// `i32`) under the log-probabilities `logp`, as `log_softmax` leaves them:
// `-logp[target]`.
fn nll_loss(logp, target) = -rowsum(where(iota(1) == target, logp, 0.0))

// The mean squared error of each row: `mean((a - b)^2)`.
fn mse_loss(a, b) = rowmean(square(a - b))

// The mean absolute error of each row: `mean(|a - b|)`.
fn l1_loss(a, b) = rowmean(abs(a - b))

// Each row's mean of `0.5 d^2 / beta` where `|d| < beta` and `|d| - 0.5 beta`
// elsewhere, `d = a - b`: the L1 loss with a quadratic bowl around 0, Fast
// R-CNN's. `beta > 0`.
fn smooth_l1_loss(a, b, beta) = rowmean(_smooth_l1(a - b, beta))

// The pointwise term of `smooth_l1_loss`.
fn _smooth_l1(d, beta) = where(abs(d) < beta, 0.5 * d * d / beta, abs(d) - 0.5 * beta)

// The KL divergence of each row of `q` from `p`, `p` given by its
// log-probabilities `logp`: `sum q (log q - logp)`, a `q` of 0 adding 0. The
// mean over the rows is PyTorch's `reduction="batchmean"`.
fn kl_div(logp, q) = rowsum(where(q > 0.0, q * (log(q) - logp), 0.0))

// Binary cross-entropy of the logits `x` against the targets `y` in
// `[0, 1]`, each row's mean, in the form that neither overflows nor loses a
// small probability: `max(x, 0) - x y + log(1 + e^-|x|)`.
fn binary_cross_entropy_with_logits(x, y) = rowmean(max(x, 0.0) - x * y + log(1.0 + exp(-abs(x))))
