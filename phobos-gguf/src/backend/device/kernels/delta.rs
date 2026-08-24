// Delta rule kernel sources: the WY transform, the scan, the
// causal convolution and the gates.

/// The gated delta rule, carrying the recurrent state in shared memory.
/// Positions are sequential (looped inside the kernel), but state columns
/// are independent, so the state tiles into `DELTA_TN`-wide slices to fit
/// alongside the operands in a head's 64K shared-memory footprint. Rows are
/// `[position, head]` with head fastest, so program `h` steps `H` at a time.
/// The rank-one write is a broadcast product rather than a `dot` of `[D, 1]`
/// by `[1, TN]`: same arithmetic, 12% of the kernel apart.
pub(crate) const DELTA_SRC: &str = "\
@launch(256)
@autotune(H in [{H}], D in [{D}], TN in [{TN}])
@aligned(R = H, D = TN, SD = D)
kernel delta_rule(Q:   tensor<f32>[R, D],
                  K:   tensor<f32>[R, D],
                  V:   tensor<f32>[R, D],
                  DEC: tensor<f32>[R, 1],
                  BET: tensor<f32>[R, 1],
                  S:   tensor<f32>[SD, D],
                  O:   tensor<f32>[R, D]) {
  let h = program_id(0)
  let jn = program_id(1)
  var st: tile<f32>[D, TN] = S[h * D :+ D, jn * TN :+ TN]

  for t in range(0, R, H) {
    let r = t + h
    var k = K[r :+ 1, 0 :+ D]
    var q = Q[r :+ 1, 0 :+ D]
    var v = V[r :+ 1, jn * TN :+ TN]
    var dec = DEC[r :+ 1, 0 :+ 1]
    var bet = BET[r :+ 1, 0 :+ 1]

    st = st * dec
    var e: tile<f32>[1, TN] = dot(k, st)
    e = (v - e) * bet
    var kt: tile<f32>[D, 1] = transpose(k)
    st = st + kt * e
    O[r :+ 1, jn * TN :+ TN] = dot(q, st)
  }

  S[h * D :+ D, jn * TN :+ TN] = st
}
";

/// The gated delta rule in chunks, as two passes ([`delta_wy_src`] here,
/// [`delta_scan_src`] below): unrolls the sequential dependency between
/// positions into matmuls over chunks of `C`. Split into two kernels
/// because `N`, its inverse, and the intra-chunk attention depend only on
/// the keys, so they're shared across every state column instead of being
/// recomputed per column (measured worse as one kernel).
///
/// Recurrence: `S_i = a_i (I - beta_i k_i^T k_i) S_{i-1} + beta_i k_i^T
/// v_i`. With `b_i` the cumulative log decay, the chunk's pseudo-values
/// solve `U = T diag(beta) (V - diag(exp b) K S_0)` for `T = (I + N)^-1`
/// (`N` strictly lower triangular), then
///
///   O   = diag(exp b) Q S_0 + tril(D * (Q K^T)) U
///   S_C = exp(b_C) (S_0 + (diag(exp(b_C - b)) K)^T U)
///
/// with `D[i, j] = exp(b_i - b_j)`.
///
/// Non-obvious choices: decay rides as the matrix `D` rather than folded
/// into `q_i exp(b_i)` / `k_j exp(-b_j)`, since the latter overflows f32 as
/// the chunk decays (D's used entries are all <= 1; `tril` selects rather
/// than multiplies, so the unused infinities never reach an operand).
/// `(I + N)^-1` uses `(I - M)^-1 = prod_j (I + M^(2^j))`, `log2(C) - 1`
/// matmul rounds since `N` is nilpotent, instead of `C` sequential
/// forward-substitution steps. `(T diag(exp b) K) S_0` associates right to
/// keep the intermediate `[C, TN]` rather than `[C, head_dim]`. `D`'s
/// diagonal is exactly 1, so `tril(D) - I` gives the strict lower mask.
///
/// `E` is the uploaded identity, serving as both `I` and that diagonal. The
/// three `[C, C]` matrices this pass leaves are laid out one row per
/// position with heads side by side, like the operands.
pub(crate) fn delta_wy_src(heads: usize, head_dim: usize, chunk: usize) -> String {
    // Squarings alternate between two names since a matmul writes its target
    // as it goes, and `p = dot(p, p)` would read what it overwrote.
    let rounds = chunk.trailing_zeros().max(2) - 1;
    let mut invert = String::from("  var p0: tile<f32>[C, C] = dot(nn, nn)\n");

    for round in 0..rounds {
        let (this, next) = if round % 2 == 0 {
            ("p0", "p1")
        } else {
            ("p1", "p0")
        };

        invert.push_str(&format!("  t = t + dot(t, {this})\n"));

        if round + 1 < rounds {
            let decl = if round == 0 {
                "var p1: tile<f32>[C, C] = "
            } else {
                ""
            };

            let assign = if round == 0 {
                String::new()
            } else {
                format!("{next} = ")
            };

            invert.push_str(&format!("  {decl}{assign}dot({this}, {this})\n"));
        }
    }

    format!(
        "@launch(256)
@autotune(H in [{heads}], D in [{head_dim}], C in [{chunk}])
@aligned(N = C, QW = D, WW = C)
kernel delta_wy(Q:   tensor<f32>[N, QW],
                K:   tensor<f32>[N, QW],
                DEC: tensor<f32>[N, GW],
                BET: tensor<f32>[N, GW],
                E:   tensor<f32>[EC, ED],
                W:   tensor<f32>[N, WW]) {{
  let h = program_id(0)
  let ci = program_id(1)
  let qc = h * D
  let wc = h * 4 * C
  let c = ci * C
  var eye: tile<f32>[C, C] = E[0 :+ C, 0 :+ C]

  var k = K[c :+ C, qc :+ D]
  var dec = DEC[c :+ C, h :+ 1]
  var bet = BET[c :+ C, h :+ 1]

  var g: tile<f32>[C, 1] = log(dec + 0.000000000000000000000000000001)
  var b: tile<f32>[C, 1] = cumsum(g)
  var eb = exp(b)
  var dl: tile<f32>[C, C] = tril(exp(b - transpose(b)))

  var nn: tile<f32>[C, C] = dot_t(k, k) * (dl - eye) * bet
  var t: tile<f32>[C, C] = eye - nn
{invert}
  var tb: tile<f32>[C, C] = t * transpose(bet)
  W[c :+ C, wc :+ C] = tb
  W[c :+ C, wc + C :+ C] = tb * transpose(eb)
  W[c :+ C, wc + 2 * C :+ C] = dot_t(Q[c :+ C, qc :+ D], k) * dl

  var total: tile<f32>[1, 1] = rowsum(transpose(g))
  var ef: tile<f32>[C, 1] = exp(total - b)
  W[c :+ C, wc + 3 * C :+ 1] = eb
  W[c :+ C, wc + 3 * C + 1 :+ 1] = ef
  W[c :+ C, wc + 3 * C + 2 :+ 1] = ef * eb
}}
"
    )
}

/// The state scan of the chunked delta rule; see [`delta_wy_src`]. The
/// transpose of `k` is taken at its last use so the query tile can reuse
/// its buffer: keeps shared memory at 42 KB against the 48 KB limit.
pub(crate) fn delta_scan_src(heads: usize, head_dim: usize, tile: usize, chunk: usize) -> String {
    format!(
        "@launch(256)
@autotune(H in [{heads}], D in [{head_dim}], TN in [{tile}], C in [{chunk}])
@aligned(N = C, QW = D, VW = TN, OW = TN, WW = C, SD = D)
@dynshared
kernel delta_scan(Q:   tensor<f32>[N, QW],
                  K:   tensor<f32>[N, QW],
                  V:   tensor<f32>[N, VW],
                  W:   tensor<f32>[N, WW],
                  S:   tensor<f32>[SD, D],
                  O:   tensor<f32>[N, OW]) {{
  let h = program_id(0)
  let jn = program_id(1)
  let qc = h * D
  let vc = h * D + jn * TN
  let wc = h * 4 * C
  var st: tile<f32>[D, TN] = S[h * D :+ D, jn * TN :+ TN]

  for c in range(0, N, C) {{
    var k = K[c :+ C, qc :+ D]
    var kst: tile<f32>[C, TN] = dot(k, st)
    var kt: tile<f32>[D, C] = transpose(k)
    var uk: tile<f32>[C, TN] = dot(W[c :+ C, wc + C :+ C], kst)
    var u: tile<f32>[C, TN] = dot(W[c :+ C, wc :+ C], V[c :+ C, vc :+ TN])
    u = u - uk
    var qs: tile<f32>[C, TN] = dot(Q[c :+ C, qc :+ D], st)
    var mu: tile<f32>[C, TN] = dot(W[c :+ C, wc + 2 * C :+ C], u)
    qs = qs * W[c :+ C, wc + 3 * C :+ 1]
    O[c :+ C, vc :+ TN] = qs + mu

    st = st * W[c :+ 1, wc + 3 * C + 2 :+ 1]
    st += dot(kt, u * W[c :+ C, wc + 3 * C + 1 :+ 1])
  }}

  S[h * D :+ D, jn * TN :+ TN] = st
}}
"
    )
}

/// Positions one chunk covers. Sized by shared memory: its tiles run 42 KB
/// of the 48 KB static limit.
pub(crate) const DELTA_CHUNK: usize = 16;

pub(crate) const DELTA_CHUNK_TN: usize = 64;

/// Columns of the delta-rule state one program owns.
pub(crate) const DELTA_TN: usize = 16;

/// The delta net's causal depthwise convolution, fused with the activation,
/// the split into per-head planes, and their normalization.
///
/// One program owns one position's one head of one plane (`D` channels).
/// `X` already carries the previous call's trailing positions ahead of this
/// call's, so tap `k` of position `t` is row `t + k` with no boundary case.
/// Taps arrive as `[KS, C]`, transposed relative to the file, so one tap
/// across a run of channels is one contiguous load. The plane rides the
/// grid's third axis rather than three launches, since one launch of 96
/// blocks beats three of 32.
///
/// `h`, the grid's head axis, ranges over the packed (value) head count.
/// Query and key exist at only `kv_heads` physical columns for a
/// grouped-query deltanet, so those two planes read column `h % kv_heads`
/// instead, matching upstream's `ggml_repeat_4d`; this collapses to `h` for
/// the non-grouped case.
#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_conv_src(
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    kernel: usize,
    stride: usize,
    plane_stride: usize,
    rows: usize,
    batch: usize,
    normalize: bool,
    query_scale: f32,
) -> String {
    let normalized = |scale: f32| {
        format!(
            "    g = {scale:.9} / sqrt(rowsum(s * s) + 0.000000000001)
"
        )
    };

    let query = if normalize {
        normalized(query_scale)
    } else {
        format!(
            "    g = {query_scale:.9}
"
        )
    };

    let key = if normalize {
        normalized(1.0)
    } else {
        String::new()
    };

    let key_case = if key.is_empty() {
        String::new()
    } else {
        format!(
            "  if p == 1 {{
{key}  }}
"
        )
    };

    format!(
        "@launch(256)
@autotune(H in [{heads}], D in [{head_dim}], KS in [{kernel}], ST in [{stride}], PS in [{plane_stride}], R in [{rows}], TB in [{batch}])
@aligned(C = D, HD = D)
kernel delta_conv(X: tensor<f32>[PR, C], W: tensor<f32>[KS, C], O: tensor<f32>[R3, HD]) {{
  let t = program_id(0)
  let h = program_id(1)
  let p = program_id(2)
  var hs: i32 = h % {kv_heads}
  if p == 2 {{
    hs = h
  }}
  let cb = p * PS + hs * ST
  var acc: tile<f32>[TB, D] = 0.0
  for k in range(0, KS, 1) {{
    var x = X[t * TB + k :+ TB, cb :+ D]
    var w = W[k :+ 1, cb :+ D]
    acc = acc + x * w
  }}
  var s = acc / (1.0 + exp(-acc))
  var g: tile<f32>[TB, 1] = 1.0
  if p == 0 {{
{query}  }}
{key_case}  O[p * R + t * TB :+ TB, h * D :+ D] = s * g
}}
"
    )
}

/// Positions one program of [`delta_conv_src`] carries. A position at a
/// time measured 512 microseconds a call at a 512-token prompt, almost all
/// dispatch overhead.
pub(crate) const DELTA_CONV_ROWS: usize = 8;

/// The positions a call batches. The tile has no remainder, so this is the
/// largest power of two up to [`DELTA_CONV_ROWS`] dividing the call.
pub(crate) fn delta_conv_batch(rows: usize) -> usize {
    let mut batch = DELTA_CONV_ROWS;
    while batch > 1 && !rows.is_multiple_of(batch) {
        batch /= 2;
    }
    batch
}

/// The delta rule's per-head gates. Softplus is `max(x, 0) + log(1 +
/// exp(-|x|))` rather than the direct `log(1 + exp(x))`, since the direct
/// form overflows well inside the decay projection's range.
pub(crate) fn delta_gates_src(heads: usize, tile_rows: usize) -> String {
    format!(
        "@launch(256)
@autotune(H in [{heads}], TR in [{tile_rows}])
kernel delta_gates(A: tensor<f32>[R, H], B: tensor<f32>[R, H],
                   RATE: tensor<f32>[M, H], BIAS: tensor<f32>[M, H],
                   DEC: tensor<f32>[R, H], BET: tensor<f32>[R, H]) {{
  let p = program_id(0)
  var a = A[p * TR :+ TR, 0 :+ H]
  var bias = BIAS[0 :+ 1, 0 :+ H]
  var rate = RATE[0 :+ 1, 0 :+ H]
  var x = a + bias
  var zero: tile<f32>[TR, H] = 0.0
  var mag = tmax(x, -x)
  var sp = tmax(x, zero) + log(1.0 + exp(-mag))
  DEC[p * TR :+ TR, 0 :+ H] = exp(rate * sp)
  var b = B[p * TR :+ TR, 0 :+ H]
  BET[p * TR :+ TR, 0 :+ H] = 1.0 / (1.0 + exp(-b))
}}
"
    )
}
