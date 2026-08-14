// Delta rule kernel sources: the WY transform, the scan, the
// causal convolution and the gates.

/// The gated delta rule, carrying the recurrent state in shared memory.
///
/// Positions are sequential, so their loop is inside the kernel, but every
/// column of the `[head_dim, head_dim]` state is independent, so the state
/// tiles. That matters because a whole head's state is 64K here, exactly
/// Turing's shared memory; `DELTA_TN` columns leave room for the operands and
/// give the grid a second axis.
///
/// `H` divides the row index rather than multiplying it: rows are
/// `[position, head]` with the head fastest, so program `h` walks its own head
/// by stepping `H` at a time from `h`.
///
/// The rank-one write is a broadcast product, not a `dot` of a `[D, 1]` by a
/// `[1, TN]`. Same arithmetic, 12% of the kernel apart.
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

/// The gated delta rule in chunks, as two passes. [`DELTA_SRC`] walks positions
/// one at a time, which for 512 of them is 512 rounds of staging five operand
/// tiles for two thousand elements of work; unrolling the dependency between
/// positions turns a chunk of `C` of them into matmuls.
///
/// The recurrence is `S_i = a_i (I - beta_i k_i^T k_i) S_{i-1} + beta_i k_i^T
/// v_i`, and the `- k_i^T k_i S` term is what makes position `i` depend on every
/// earlier write inside the chunk. Writing `b_i` for the cumulative log decay,
/// the chunk has one pseudo-value `u_i` per position with
///
///   u_i + beta_i sum_{j<i} exp(b_i - b_j) (k_i . k_j) u_j
///     = beta_i v_i - beta_i exp(b_i) k_i S_0
///
/// so `U = T diag(beta) (V - diag(exp b) K S_0)` for `T = (I + N)^-1`, `N` that
/// strictly lower triangular matrix, and then
///
///   O   = diag(exp b) Q S_0 + tril(D * (Q K^T)) U
///   S_C = exp(b_C) (S_0 + (diag(exp(b_C - b)) K)^T U)
///
/// with `D[i, j] = exp(b_i - b_j)`.
///
/// It is two kernels because `N`, its inverse and the intra-chunk attention
/// depend only on the keys and so are the same for every column of the state,
/// while the rest is walked chunk by chunk and splits over those columns to fill
/// the grid. As one kernel the key-only half is recomputed per column slice,
/// which measured worse than the sequential kernel it replaces.
///
/// Four details are not the obvious spelling:
///
/// - The decay rides as a matrix. Folding it into the operands as
///   `q_i exp(b_i)` and `k_j exp(-b_j)` is one multiply cheaper and overflows
///   f32 outright, since `exp(-b_j)` grows without bound as the chunk decays.
///   As `D[i, j] = exp(b_i - b_j)` every entry of the triangle that is used is
///   at most one, and `tril` selects rather than multiplies, so the upper
///   triangle's infinities never reach an arithmetic operand.
/// - `(I + N)^-1` is not solved. Forward substitution is `C` sequential steps,
///   the depth this exists to remove, while `N` is nilpotent, so
///   `(I - M)^-1 = prod_j (I + M^(2^j))` is `log2(C) - 1` rounds of two
///   `[C, C]` matmuls and no sequential depth.
/// - `(T diag(exp b) K) S_0` associates to the right. Left to right it builds a
///   `[C, head_dim]` intermediate; as `T diag(exp b) (K S_0)` the intermediate
///   is `[C, TN]`, smaller and less arithmetic.
/// - `D` has an exact 1 on its diagonal, so `tril(D) - I` is the strict lower
///   mask, which `tril` alone cannot give.
///
/// `E` is the identity, uploaded once, and serves as both the `I` of the
/// inversion and that diagonal. The three `[C, C]` matrices the first pass
/// leaves are laid out one row per position with the heads side by side, the
/// same reinterpretation the operands take.
pub(crate) fn delta_wy_src(heads: usize, head_dim: usize, chunk: usize) -> String {
    // `I - N` covers the first two powers and each round doubles the reach, so
    // `log2(C) - 1` rounds carry it to N^(C-1), where N vanishes. The squarings
    // alternate between two names because a matmul writes its target as it
    // goes, and `p = dot(p, p)` would read what it had overwritten.
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

/// The state scan of the chunked delta rule; see [`delta_wy_src`].
///
/// The order of the body is not free. Key and query tiles are both
/// `[C, head_dim]`, and the transpose is taken at the key's last use so the
/// query reuses its buffer: 42 KB against 50, and 48 is the limit.
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

/// Positions one chunk covers, and the state columns one scan program owns. The
/// scan does about twelve tile operations per (chunk, slice) where the
/// sequential kernel does six per position, so 16 over four slices is half the
/// tile operations of 512 positions over eight. Both are what shared memory
/// allows: those tiles are 42 KB of the 48 static shared memory gives.
pub(crate) const DELTA_CHUNK: usize = 16;

pub(crate) const DELTA_CHUNK_TN: usize = 64;

/// Columns of the delta-rule state one program owns.
pub(crate) const DELTA_TN: usize = 16;

/// The delta net's causal depthwise convolution, fused with the activation, the
/// split into per-head planes, and their normalization.
///
/// One program owns one position's one head of one plane: `D` channels of the
/// convolved stream. That span is both the row the delta rule reads and the span
/// the L2 normalization covers, so the convolution never writes an intermediate
/// and the split is a choice of destination rather than a pass.
///
/// `X` already carries the previous call's trailing positions ahead of this
/// call's, so tap `k` of position `t` is row `t + k` with no boundary case. The
/// taps arrive as `[KS, C]`, transposed relative to the file, so one tap across
/// a run of channels is one contiguous load.
///
/// The plane rides the grid's third axis rather than being three launches: one
/// launch of ninety-six blocks beats three of thirty-two. Planes are evenly
/// spaced in both layouts a file might use, so the offset is a multiple, not a
/// table.
///
/// Query and key are L2-normalized, the value is not, and the query carries the
/// `1/sqrt(d)` scale. The epilogue tests the block index so the choice stays
/// uniform across the CTA.
#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_conv_src(
    heads: usize,
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
  let cb = p * PS + h * ST
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

/// Positions one program of [`delta_conv_src`] carries. A position at a time is
/// 24576 blocks of 256 threads for two multiplies each at a 512-token prompt,
/// which measured 512 microseconds a call, almost all of it dispatch.
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

/// The delta rule's per-head gates.
///
/// The softplus is `max(x, 0) + log(1 + exp(-|x|))` rather than the direct
/// `log(1 + exp(x))`: the two agree everywhere, but the direct form overflows
/// well inside the range the decay projection reaches, and the host reference
/// guards that with a branch a tile has no room for.
///
/// The per-head parameters broadcast down the row tile, so a program covers `TR`
/// positions at once; a row on its own is only `H` wide.
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
