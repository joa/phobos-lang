// Counter-based random numbers: a value is a hash of a seed and the
// element's own counter, so a tile of them costs a few integer instructions
// an element and keeps no state, and a backward pass that knows the seed
// draws the same numbers again. The counter is an `i32` tile, usually each
// element's flat index, as `(r0 + iota(0)) * C + iota(1)` is for row `r0`
// of a `[R, C]` block, and the seed a scalar `i32`.
//
// The hash is Chris Wellons' `lowbias32`, a 32-bit mixer that passes
// statistical tests run over a counter. It is neither cryptographic nor
// Philox, so a draw is not the number `torch.rand` gives for the seed.

// The `lowbias32` hash of each element of the `i32` value `x`.
fn hash32(x) = _mix(_mix(_mix(x, 16) * 2146121005, 15) * -2073254261, 16)

// `x ^ (x >> s)` with the shift logical: the sign bits an arithmetic shift
// copies in are masked off.
fn _mix(x, s) = x ^ ((x >> s) & ((1 << (32 - s)) - 1))

// A number in `[0, 1)` for each counter, uniform on a grid of `2^-23`: the
// hash's low 23 bits.
fn uniform(seed, counter) = f32(hash32(counter ^ hash32(seed)) & 8388607) * 1.1920928955078125e-7

// A standard normal number for each counter, the Box-Muller transform of
// the uniform pair at counters `2 counter` and `2 counter + 1`. The
// uniform's grid ends its tails near 5.6.
fn randn(seed, counter) = sqrt(-2.0 * log(1.0 - uniform(seed, 2 * counter))) * cos(6.283185307179586 * uniform(seed, 2 * counter + 1))

// Dropout: each element of `x` zeroed with probability `p` and the kept
// ones scaled by `1 / (1 - p)`, so the expectation is `x`. `p` in `[0, 1)`.
fn dropout(x, p, seed, counter) = where(uniform(seed, counter) >= p, x / (1.0 - p), 0.0)
