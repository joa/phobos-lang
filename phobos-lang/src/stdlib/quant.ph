// Symmetric int8 quantization by absolute maximum, one scale a row: the
// round trip a kernel takes to feed `qdot_t`, `qmma_t` or `dot_t` over
// `i8` from an `f32` activation.

// Each row of `x` (`[R, C]`) as `i8` in `q` (`[R, C]`) and its scale in `s`
// (`[R, 1]`, `f32`), so `x` is about `q * s`: the row's largest magnitude
// maps to 127. `q` and `s` may be tiles or tensor slices.
fn quantize_i8(q, s, x) {
  var m = rowmax(abs(x))
  var scaled = x * (127.0 / (m + 1.0e-30))
  q = i8(i32(round(scaled)))
  s = m / 127.0
}

// `x = q * s`, the inverse of `quantize_i8`.
fn dequantize_i8(x, q, s) {
  x = f32(q) * s
}
