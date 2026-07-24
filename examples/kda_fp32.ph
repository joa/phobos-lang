// Kimi Delta Attention fp32
//
// One program streams a single head's sequence in chunks of C, carrying the
// [D, D] recurrent state S across chunks. With a scalar per-token gate
// a_t in (0, 1) the recurrence is
//
//   S_t = a_t * S_{t-1} + k_t^T v_t,   o_t = q_t @ S_t
//
// Writing b_i = sum_{r<=i} log a_r for the in-chunk cumulative log-gate, the
// chunk math folds the decay into q and k and splits into an intra-chunk
// (causal) and an inter-chunk (carried state) term:
//
//   qd_i = exp(b_i) * q_i           kd_j = exp(-b_j) * k_j
//   o_i  = (qd @ S_prev)_i  +  sum_{j<=i} (qd_i . kd_j) v_j
//   S_new = exp(b_last) * S_prev + sum_j exp(b_last - b_j) k_j^T v_j
//
// G holds the per-token log forget-gate (log a_t, <= 0). scale multiplies the
// queries (uniform output scaling), matching the softmax q-scaling slot.
@autotune(D in [64], C in [32, 128])
kernel kda(Q:     tensor<f32>[N, D],
           K:     tensor<f32>[N, D],
           V:     tensor<f32>[N, D],
           G:     tensor<f32>[N, 1],
           O:     tensor<f32>[N, D],
           scale: f32) {
  var S: tile<f32>[D, D] = 0.0               // recurrent state (keys x values)

  for c in range(0, N, C) {
    let q = Q[c :+ C, :]                     // [C, D]
    let k = K[c :+ C, :]                     // [C, D]
    let v = V[c :+ C, :]                     // [C, D]
    let g = G[c :+ C, :]                     // [C, 1] per-token log-gates

    // in-chunk cumulative decay (down the sequence axis)
    var b: tile<f32>[C, 1] = cumsum(g)       // b_i = sum_{r<=i} log a_r
    var db = exp(b)                          // decay from chunk start to i
    var negb = b * -1.0
    var dbi = exp(negb)                      // its inverse, exp(-b_i)

    // decay-folded queries and keys
    var qd: tile<f32>[C, D] = q * db         // broadcast [C,1] over [C,D]
    qd = qd * scale
    var kd: tile<f32>[C, D] = k * dbi

    // intra-chunk causal attention:  P = tril(qd @ kd^T),  o = P @ V
    var p: tile<f32>[C, C] = dot_t(qd, kd)
    p = tril(p)
    var o: tile<f32>[C, D] = dot(p, v)

    // inter-chunk: decay-folded queries read the carried state
    o += dot(qd, S)                          // [C, D] = [C, D] @ [D, D]
    O[c :+ C, :] = o

    // chunk-total decay = sum of the chunk's log-gates (b at the last row)
    var gt = transpose(g)                    // [1, C]
    var total: tile<f32>[1, 1] = rowsum(gt)

    // state update: decay S to the chunk end and add this chunk's K^T V,
    // each key scaled by its decay from j to the chunk end (total - b_j).
    var kfin = k * exp(total - b)            // [C, D]
    var kt = transpose(kfin)                 // [D, C]
    var kv: tile<f32>[D, D] = dot(kt, v)     // sum_j kfin_j^T v_j
    S = S * exp(total) + kv
  }
}
