// Arithmetic the built-ins leave out. Every function here is an expression
// function: on a tile it fuses into the sweep of the statement that calls it,
// on a scalar it is plain arithmetic, so `clamp(x, 0.0, 1.0)` is two
// instructions an element either way.

// The number of `b`-sized pieces that cover `a`: `ceil(a / b)` for
// positive integers. The grid of a kernel tiling `N` by `T` is `cdiv(N, T)`.
fn cdiv(a, b) = (a + b - 1) / b

// `a` rounded up to a multiple of `b`, for positive integers.
fn round_up(a, b) = cdiv(a, b) * b

fn square(x) = x * x

// `1 / x`.
fn rcp(x) = 1.0 / x

// `x` limited to `[lo, hi]`, for `lo <= hi`.
fn clamp(x, lo, hi) = min(max(x, lo), hi)

// The point `w` of the way from `a` to `b`: `a` at 0, `b` at 1.
fn lerp(a, b, w) = a + w * (b - a)

// `2^x` and `log2(x)`, as approximate as `exp` and `log`.
fn exp2(x) = exp(x * 0.6931471805599453)
fn log2(x) = log(x) * 1.4426950408889634

// -1 for a negative `x`, 1 for a positive one. Within `1e-30` of zero the
// result runs linearly through 0, so it is 0 at 0.
fn sign(x) = clamp(x * 1.0e30, -1.0, 1.0)

// The error function, by Abramowitz and Stegun's 7.1.26: within 1.5e-7 of
// `erf` on every input, before the rounding of `exp`.
fn erf(x) = sign(x) * (1.0 - _erf_tail(1.0 / (1.0 + 0.3275911 * abs(x))) * exp(-x * x))

// The polynomial of `erf`, in `t = 1 / (1 + p |x|)`.
fn _erf_tail(t) = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
