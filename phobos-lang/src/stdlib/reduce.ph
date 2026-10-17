// Row-wise operations on a rank-2 tile `[R, C]`, each row on its own, as a
// softmax over the keys of each query is. They build on the built-in
// reductions `rowmax`, `rowsum` and `rowargmax`.
//
// A function that keeps the shape works on its tile in place and must be
// passed a name: `softmax(s)` leaves the probabilities in `s`. One that
// makes a new value takes its destination first, as `logsumexp(out, t)`.

// The mean of each row, `[R, C] -> [R, 1]`. `t` may be an elementwise
// expression, as in `rowmean(x * x)`.
fn rowmean(t) = rowsum(t) / f32(extent(t, 1))

// The minimum of each row, `[R, C] -> [R, 1]`.
fn rowmin(t) = -rowmax(-t)

// The column of each row's minimum, `[R, C] -> [R, 1]` (`i32`), the
// smaller column on a tie.
fn rowargmin(t) = rowargmax(-t)

// The variance of each row, `[R, C] -> [R, 1]`: the mean squared distance
// from the row's mean, the biased variance a layer norm takes.
fn rowvar(t) = rowsum(square(t - rowmean(t))) / f32(extent(t, 1))

// Reductions down the columns, `[R, C] -> [1, C]`, as a batch norm folds
// a batch. Each transposes `t` through shared memory and reduces its rows.
fn colsum(t) = transpose(rowsum(transpose(t)))
fn colmax(t) = transpose(rowmax(transpose(t)))
fn colmin(t) = -colmax(-t)
fn colmean(t) = colsum(t) / f32(extent(t, 0))

// Each row's softmax, in place: `e^(t - max) / sum e^(t - max)`. The row
// maximum is taken out first, so no row overflows.
fn softmax(t) {
  var m = rowmax(t)
  t = exp(t - m)
  var r = 1.0 / rowsum(t)
  t = t * r
}

// Each row's log-softmax, in place: `t - logsumexp(t)`.
fn log_softmax(t) {
  var m = rowmax(t)
  t = t - m
  var lse = log(rowsum(exp(t)))
  t = t - lse
}

// `out` (`[R, 1]`) is `log sum_j e^t[i, j]` of each row, computed around
// the row maximum.
fn logsumexp(out, t) {
  var m = rowmax(t)
  out = m + log(rowsum(exp(t - m)))
}

// Sets every column at or past `n` to `fill`, in place: the mask a ragged
// last tile needs before a reduction that a zero would change. A slice past
// the end of a tensor reads as 0, so a softmax over one is masked with
// `mask_cols(s, n, -1.0e30)`.
fn mask_cols(t, n, fill) {
  t = where(iota(1) < n, t, fill)
}

// The causal mask of attention scores, in place: element `[i, j]` is query
// `q0 + i` against key `k0 + j`, and a key after its query is set to
// `-1.0e30`, so a softmax gives it no weight.
fn causal_mask(t, q0, k0) {
  t = where(k0 + iota(1) <= q0 + iota(0), t, -1.0e30)
}

// The `K` largest values of each row of `t` into `values` (`[R, K]`) and
// their columns into `indices` (`[R, K]`, `i32`), largest first, the
// smaller column on a tie. `t` is consumed: each pick is overwritten with
// `-1.0e30`. `values` and `indices` must be names.
fn topk(values, indices, t) {
  for j in range(0, extent(indices, 1)) {
    var i = rowargmax(t)
    values[:, j :+ 1] = rowmax(t)
    indices[:, j :+ 1] = i
    t = where(iota(1) == i, -1.0e30, t)
  }
}
