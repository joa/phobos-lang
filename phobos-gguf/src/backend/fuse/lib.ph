// The fused decode chains' stages, as statement functions a chain kernel
// calls. Each body is the text the fuse pass writes for its stage, so a chain
// written from these compiles to the kernel the pass writes for it.
//
// A tile's dims are constants, not parameters, so a stage that declares one
// names the chain's: `HD`, a head's width.

// One row's RMS norm and its Q8_0 copy into `q` and `s`, `nb` blocks of 32.
// A thread owns four contiguous elements of every `4 * ctx.threads`: one
// pass sums their squares into a CTA total, a second normalizes them, and a
// block's 32 elements, eight lanes, fold their absolute maximum for its
// scale. A `barrier()` publishes the row. See `norm_q.rs`.
fn rms_norm_q(x, g, q, s, nb, eps) {
  var ss: f32 = 0.0
  for i in range(0, ((nb * 32) + 4 * ctx.threads - 1) / (4 * ctx.threads), 1) {
    let e = 4 * ctx.thread + i * 4 * ctx.threads
    if e < (nb * 32) {
      ss += sum(x[0 + e / 32, e % 32 :+ 4] * x[0 + e / 32, e % 32 :+ 4])
    }
  }
  let inv = 1.0 / sqrt(sum<over ctx.thread>(ss) / f32((nb * 32)) + eps)
  for i in range(0, ((nb * 32) + 4 * ctx.threads - 1) / (4 * ctx.threads), 1) {
    let e = 4 * ctx.thread + i * 4 * ctx.threads
    let row = 0 + e / 32
    let col = e % 32
    var n0: f32 = 0.0
    var n1: f32 = 0.0
    var n2: f32 = 0.0
    var n3: f32 = 0.0
    if e < (nb * 32) {
      n0 = elem(x[row, col :+ 4], 0) * inv * elem(g[e / 32, col :+ 4], 0)
      n1 = elem(x[row, col :+ 4], 1) * inv * elem(g[e / 32, col :+ 4], 1)
      n2 = elem(x[row, col :+ 4], 2) * inv * elem(g[e / 32, col :+ 4], 2)
      n3 = elem(x[row, col :+ 4], 3) * inv * elem(g[e / 32, col :+ 4], 3)
    }
    let mx = max<over ctx.lane, width 8>(max(max(abs(n0), abs(n1)), max(abs(n2), abs(n3))))
    let iq = 127.0 / (mx + 0.00000001)
    if e < (nb * 32) {
      store(q[row, col :+ 4], i8(i32(round(n0 * iq))), i8(i32(round(n1 * iq))), i8(i32(round(n2 * iq))), i8(i32(round(n3 * iq))))
      if ctx.lane % 8 == 0 {
        s[row, 0] = mx / 127.0
      }
    }
  }
  barrier()
}

// 32 columns of a Q8_0 projection from row `at` of the weight.
fn q8_proj(aq, sa, wq, ws, at) = qdot_t(aq, sa,
                                        wq[at :+ 32, :], ws[at :+ 32, :])

// `y += a W^T` over the `n` columns from `t`.
fn q8_proj_add(y, aq, sa, wq, ws, t, n) {
  y[0 :+ 1, t :+ n] += qdot_t(aq, sa,
                              wq[t :+ n, :], ws[t :+ n, :])
}

// SwiGLU of the gate `g` and the up projection `p`, quantized as row `u` of
// `q` and `s`.
fn swiglu_q(g, p, q, s, u) {
  var h: tile<f32>[1, 32] = (g / (1.0 + exp(-g))) * p
  var m: tile<f32>[1, 1] = rowmax(abs(h))
  var hq = h * (127.0 / (m + 0.00000001))
  q[u :+ 1, 0 :+ 32] = i8(i32(round(hq)))
  s[u :+ 1, 0 :+ 1] = m / 127.0
}

// 32 columns of a Q8_0 projection from row `at`, stored at column `to`.
fn q8_proj_to(y, aq, sa, wq, ws, at, to) {
  y[0 :+ 1, to :+ 32] = qdot_t(aq, sa,
                               wq[at :+ 32, :], ws[at :+ 32, :])
}

// Unit `u` of the delta rule's short convolution: head `u % NH` of plane
// `u / NH`, its `KS` taps of the history `hist` against the weights `w`,
// SiLU, then the query plane scaled by `qs` and the key plane to unit norm.
// Planes are `PS` apart and heads `ST`, `HD` wide; the result goes to `y` at
// `u * width`.
fn delta_conv(hist, w, y, u, width, qs, eps) {
  let pl = u / NH
  let hd = u % NH
  let cb = pl * PS + hd * ST
  var acc: tile<f32>[1, HD] = 0.0
  for k in range(0, KS) {
    acc = acc + hist[k :+ 1, cb :+ HD] * w[k :+ 1, cb :+ HD]
  }
  var yv: tile<f32>[1, HD] = acc / (1.0 + exp(-acc))
  var g: tile<f32>[1, 1] = 1.0
  if pl == 0 {
    g = qs / sqrt(rowsum(yv * yv) + eps)
  }
  if pl == 1 {
    g = 1.0 / sqrt(rowsum(yv * yv) + eps)
  }
  y[0 :+ 1, u * width :+ HD] = yv * g
}

// Head `h`'s two gates: the decay `exp(a_log * softplus(a + dt_bias))` and
// the write strength `sigmoid(b)`, from the projection `x` at `a_off` and
// `b_off`, to `y` at `g_off` and `beta_off`.
fn delta_gates(x, bias, a_log, y, h, a_off, b_off, g_off, beta_off) {
  var a: tile<f32>[1, 1] = x[0 :+ 1, a_off + h :+ 1] + bias[0 :+ 1, h :+ 1]
  var z: tile<f32>[1, 1] = 0.0
  var sp: tile<f32>[1, 1] = tmax(a, z) + log(1.0 + exp(-abs(a)))
  y[0 :+ 1, g_off + h :+ 1] = exp(a_log[0 :+ 1, h :+ 1] * sp)
  var b: tile<f32>[1, 1] = x[0 :+ 1, b_off + h :+ 1]
  y[0 :+ 1, beta_off + h :+ 1] = 1.0 / (1.0 + exp(-b))
}
