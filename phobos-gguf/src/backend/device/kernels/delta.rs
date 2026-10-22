// Delta rule kernel sources: the WY transform, the scan, the
// causal convolution and the gates.

/// The gated delta rule, with the recurrent state in shared memory.
///
/// Positions run sequentially inside the kernel. State columns are
/// independent, so the state is tiled into `DELTA_TN`-wide slices to fit in
/// shared memory. Rows are `[position, head]` with head fastest, so program
/// `h` steps by `H`.
///
/// The rank-one update is a broadcast product rather than a `dot`, which is
/// cheaper for the same arithmetic.
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
  let h = ctx.block.x
  let jn = ctx.block.y
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
    st += kt * e
    O[r :+ 1, jn * TN :+ TN] = dot(q, st)
  }

  S[h * D :+ D, jn * TN :+ TN] = st
}
";

/// First pass of the chunked gated delta rule; [`delta_scan_src`] is the
/// second.
///
/// The chunked form turns the sequential dependency between positions into
/// matmuls over chunks of `C`. This pass computes what depends only on the
/// keys (`N`, its inverse and the intra-chunk attention), since it is
/// shared across every state column.
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
/// Decay is kept as the matrix `D`, because per-vector exponents would
/// overflow f32. `tril` selects rather than multiplies, so masked
/// infinities never reach an operand.
///
/// `N` is nilpotent, so `(I + N)^-1` uses repeated squaring,
/// `(I - M)^-1 = prod_j (I + M^(2^j))`, in `log2(C) - 1` matmul rounds.
/// `(T diag(exp b) K) S_0` associates right to keep the intermediate at
/// `[C, TN]`. `tril(D) - I` is the strict lower mask since `D`'s diagonal
/// is exactly 1.
///
/// `E` is the uploaded identity. The three `[C, C]` outputs are laid out
/// one row per position with heads side by side, like the operands.
pub(crate) fn delta_wy_src(heads: usize, head_dim: usize, chunk: usize) -> String {
    // Squarings alternate between two names, since `p = dot(p, p)` would
    // read what it has already overwritten.
    let rounds = chunk.trailing_zeros().max(2) - 1;
    let mut invert = String::from("  var p0: tile<f32>[C, C] = dot(nn, nn)\n");

    for round in 0..rounds {
        let (this, next) = if round % 2 == 0 {
            ("p0", "p1")
        } else {
            ("p1", "p0")
        };

        invert.push_str(&format!("  t += dot(t, {this})\n"));

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
  let h = ctx.block.x
  let ci = ctx.block.y
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
/// The transpose of `k` is taken right after its last use, so the query
/// tile can reuse its buffer and shared memory stays under 48 KB.
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
  let h = ctx.block.x
  let jn = ctx.block.y
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

/// Positions a chunk of [`delta_scan_tc_src`] covers, and the state columns
/// one of its programs owns.
pub(crate) const DELTA_TC_CHUNK: usize = 32;
pub(crate) const DELTA_TC_TN: usize = 16;

/// Whether a prompt's delta rule takes [`delta_scan_tc_src`] by default on
/// `chip`: from sm_80, whose tensor cores multiply bf16, which keeps the
/// state's range.
pub(crate) fn delta_tc_default(chip: &str) -> bool {
    compute_capability(chip) >= 80
}

/// Whether [`delta_scan_tc_src`] at `head_dim` fits `chip`'s shared memory:
/// below sm_80 a block holds 64 KB, the split operands of a head of 64.
pub(crate) fn delta_tc_fits(chip: &str, head_dim: usize) -> bool {
    compute_capability(chip) >= 80 || head_dim <= 64
}

fn compute_capability(chip: &str) -> u32 {
    chip.strip_prefix("sm_").and_then(|cc| cc.parse().ok()).unwrap_or(0)
}

/// [`delta_scan_src`] on the tensor cores at about f32 accuracy
/// (`@tensorcore(f32)`), over [`DELTA_TC_CHUNK`] positions a chunk and
/// [`DELTA_TC_TN`] state columns a program of two warps. The state is
/// carried in the accumulator registers from chunk to chunk, seeded from
/// `S` by adding it to zero, and every product of a chunk tiles into the
/// two warps' fragments. `W` is [`delta_wy_src`]'s at the same chunk.
pub(crate) fn delta_scan_tc_src(heads: usize, head_dim: usize) -> String {
    format!(
        "@tensorcore(f32)
@launch(64)
@autotune(H in [{heads}], D in [{head_dim}], TN in [{DELTA_TC_TN}], C in [{DELTA_TC_CHUNK}])
@aligned(N = C, QW = D, VW = TN, OW = TN, WW = C, SD = D)
@dynshared
kernel delta_scan_tc(Q:   tensor<f32>[N, QW],
                     K:   tensor<f32>[N, QW],
                     V:   tensor<f32>[N, VW],
                     W:   tensor<f32>[N, WW],
                     S:   tensor<f32>[SD, D],
                     O:   tensor<f32>[N, OW]) {{
  let h = ctx.block.x
  let jn = ctx.block.y
  let qc = h * D
  let vc = h * D + jn * TN
  let wc = h * 4 * C
  var st: tile<f32>[D, TN] = 0.0
  st += S[h * D :+ D, jn * TN :+ TN]

  for c in range(0, N, C) {{
    let k = K[c :+ C, qc :+ D]
    let q = Q[c :+ C, qc :+ D]
    let w0 = W[c :+ C, wc :+ C]
    let w1 = W[c :+ C, wc + C :+ C]
    let w2 = W[c :+ C, wc + 2 * C :+ C]
    let vv = V[c :+ C, vc :+ TN]
    var kst: tile<f32>[C, TN] = dot(k, st)
    var uk: tile<f32>[C, TN] = dot(w1, kst)
    var u: tile<f32>[C, TN] = dot(w0, vv)
    u = u - uk
    var qs: tile<f32>[C, TN] = dot(q, st)
    var mu: tile<f32>[C, TN] = dot(w2, u)
    qs = qs * W[c :+ C, wc + 3 * C :+ 1]
    O[c :+ C, vc :+ TN] = qs + mu

    var kt: tile<f32>[D, C] = transpose(k)
    var ue: tile<f32>[C, TN] = u * W[c :+ C, wc + 3 * C + 1 :+ 1]
    st = st * W[c :+ 1, wc + 3 * C + 2 :+ 1]
    st += dot(kt, ue)
  }}

  S[h * D :+ D, jn * TN :+ TN] = st
}}
"
    )
}

/// State columns one program of [`delta_scan_reg_src`] owns: two warps of
/// four columns each.
pub(crate) const DELTA_REG_COLS: usize = 8;

/// The gated delta rule over a prompt with the state in registers: one
/// program per head and [`DELTA_REG_COLS`] state columns, walking the
/// positions in order. Per position `t`, with `S` the state:
///
///   S = dec_t * S
///   S = S + k_t^T (bet_t * (v_t - k_t S))
///   o_t = q_t S
///
/// Each warp owns four columns. Lane `l` holds column `l % 4` and the
/// `head_dim / 8` rows of row group `l / 4` in registers, so the two sums
/// over rows per position, `k . S` and `q . S`, are xor butterflies across
/// the row groups only. The shuffles, not the arithmetic, bound the scan on
/// a card with few SMs.
///
/// The operands are the packed planes, one row per position with the heads
/// side by side: `Q`, `K`, `V`, `O` as `[N, heads * D]`, the gates as
/// `[N, heads]`, the state as `[heads * D, D]`.
pub(crate) fn delta_scan_reg_src(head_dim: usize) -> String {
    let cols = DELTA_REG_COLS;
    let threads = cols / 4 * 32;
    let rows = head_dim / 8;
    let lines = |f: &dyn Fn(usize) -> String| (0..rows).map(|r| f(r) + "\n").collect::<String>();
    let terms = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<Vec<_>>().join(" + ");
    // A position's inputs, by name and by where they are read at row `{t}`:
    // the lane's rows of k and q, its column of v, and the two gates.
    let mut inputs = Vec::new();
    for r in 0..rows {
        for m in ["K", "Q"] {
            let at = format!("elem({m}[{{t}}, kc + {} :+ 4], {})", r / 4 * 4, r % 4);
            inputs.push((format!("{}{r}", m.to_lowercase()), at));
        }
    }
    inputs.push(("v".into(), "V[{t}, h * D + col]".into()));
    inputs.push(("dec".into(), "DEC[{t}, h]".into()));
    inputs.push(("bet".into(), "BET[{t}, h]".into()));
    let read = |line: &dyn Fn(&str, String) -> String, t: &str| {
        inputs.iter().map(|(name, at)| line(name, at.replace("{t}", t)) + "\n").collect::<String>()
    };
    let butterfly = |x: &str| [16, 8, 4].iter().map(|m| format!("    {x} += shfl<xor {m}>({x})\n")).collect::<String>();
    format!(
        "@launch({threads})
@autotune(D in [{head_dim}], C in [{cols}])
@aligned(SD = D, SW = C, W = D, GW = 1)
kernel delta_scan_reg(Q: tensor<f32>[N, W], K: tensor<f32>[N, W], V: tensor<f32>[N, W],
                  DEC: tensor<f32>[N, GW], BET: tensor<f32>[N, GW],
                  S: tensor<f32>[SD, SW], O: tensor<f32>[N, W]) {{
  let h = ctx.block.x
  let col = ctx.block.y * C + ctx.warp * 4 + ctx.lane % 4
  let row = h * D + ctx.lane / 4 * {rows}
  let kc = ctx.lane / 4 * {rows} + h * D
{init}{first}  for t in range(0, N, 1) {{
    let tn = min(t + 1, N - 1)
{next}{decay}    var kv: f32 = {kdot}
{kv_sum}    let u = (v - kv) * bet
{update}    var out: f32 = {qdot}
{out_sum}    if ctx.lane < 4 {{
      O[t, h * D + col] = out
    }}
{advance}  }}
{store}}}
",
        init = lines(&|r| format!("  var s{r}: f32 = S[row + {r}, col]")),
        first = read(&|name, at| format!("  var {name}: f32 = {at}"), "0"),
        next = read(&|name, at| format!("    let n{name} = {at}"), "tn"),
        decay = lines(&|r| format!("    s{r} = s{r} * dec")),
        kdot = terms(&|r| format!("s{r} * k{r}")),
        kv_sum = butterfly("kv"),
        update = lines(&|r| format!("    s{r} = s{r} + k{r} * u")),
        qdot = terms(&|r| format!("s{r} * q{r}")),
        out_sum = butterfly("out"),
        advance = read(&|name, _| format!("    {name} = n{name}"), "tn"),
        store = lines(&|r| format!("  S[row + {r}, col] = s{r}")),
    )
}

/// Positions one chunk covers, bounded by the 48 KB static shared memory
/// limit.
pub(crate) const DELTA_CHUNK: usize = 16;

/// Columns of the state one program of the chunked scan owns. Narrower than
/// a whole head so the grid holds two programs per head and SM at 32 heads
/// of 128 on a 128-SM card.
pub(crate) const DELTA_CHUNK_TN: usize = 32;

/// Columns of the delta-rule state one program owns.
pub(crate) const DELTA_TN: usize = 16;

/// The delta net's causal depthwise convolution, fused with the activation,
/// the split into per-head planes, and their normalization. A program owns
/// `TB` positions of one head of one plane (`D` channels).
///
/// `R3` is three planes of output positions. `X` already holds the previous
/// call's trailing positions ahead of this call's, so tap `k` of position
/// `t` is row `t + k` with no boundary case. Taps are `[KS, C]`, transposed
/// from the file, so one tap over a run of channels is a contiguous load.
/// One launch covers all three planes.
///
/// The grid's second axis runs over `kv_heads` query heads, `kv_heads` key
/// heads, then the value heads. In a grouped-query deltanet the scan reads a
/// query and key per value head, value head `h` taking head `h % kv_heads`
/// as upstream's `ggml_repeat_4d` does, so each query and key head is
/// computed once and stored to every value head that shares it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_conv_src(
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    kernel: usize,
    stride: usize,
    plane_stride: usize,
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

    let (qk, group) = (2 * kv_heads, heads / kv_heads);
    format!(
        "@launch(256)
@autotune(H in [{heads}], D in [{head_dim}], KS in [{kernel}], ST in [{stride}], PS in [{plane_stride}], TB in [{batch}])
@aligned(C = D, HD = D)
kernel delta_conv(X: tensor<f32>[PR, C], W: tensor<f32>[KS, C], O: tensor<f32>[R3, HD]) {{
  let t = ctx.block.x
  let j = ctx.block.y
  let r = R3 / 3
  var p: i32 = 2
  var hs: i32 = j - {qk}
  if j < {qk} {{
    p = 1
    hs = j - {kv_heads}
  }}
  if j < {kv_heads} {{
    p = 0
    hs = j
  }}
  let cb = p * PS + hs * ST
  var acc: tile<f32>[TB, D] = 0.0
  for k in range(0, KS, 1) {{
    var x = X[t * TB + k :+ TB, cb :+ D]
    var w = W[k :+ 1, cb :+ D]
    acc += x * w
  }}
  var s = acc / (1.0 + exp(-acc))
  var g: tile<f32>[TB, 1] = 1.0
  if p == 0 {{
{query}  }}
{key_case}  var y = s * g
  if p == 2 {{
    O[2 * r + t * TB :+ TB, hs * D :+ D] = y
  }} else {{
    for i in range(0, {group}, 1) {{
      O[p * r + t * TB :+ TB, (hs + i * {kv_heads}) * D :+ D] = y
    }}
  }}
}}
"
    )
}

/// Most positions one program of [`delta_conv_src`] carries.
pub(crate) const DELTA_CONV_ROWS: usize = 8;

/// Positions per program for a call: the largest power of two up to
/// [`DELTA_CONV_ROWS`] that divides `rows`, since the tile has no
/// remainder.
pub(crate) fn delta_conv_batch(rows: usize) -> usize {
    let mut batch = DELTA_CONV_ROWS;
    while batch > 1 && !rows.is_multiple_of(batch) {
        batch /= 2;
    }
    batch
}

/// The delta rule's per-head gates. Softplus is computed as
/// `max(x, 0) + log(1 + exp(-|x|))`, because `log(1 + exp(x))` overflows
/// within the decay projection's range.
pub(crate) fn delta_gates_src(heads: usize, tile_rows: usize) -> String {
    format!(
        "@launch(256)
@autotune(H in [{heads}], TR in [{tile_rows}])
kernel delta_gates(A: tensor<f32>[R, AW], B: tensor<f32>[R, AW],
                   RATE: tensor<f32>[M, H], BIAS: tensor<f32>[M, H],
                   DEC: tensor<f32>[R, H], BET: tensor<f32>[R, H]) {{
  let p = ctx.block.x
  var a = A[p * TR :+ TR, 0 :+ H]
  var bias = BIAS[0 :+ 1, 0 :+ H]
  var rate = RATE[0 :+ 1, 0 :+ H]
  var x = a + bias
  var zero: tile<f32>[TR, H] = 0.0
  var mag = abs(x)
  var sp = tmax(x, zero) + log(1.0 + exp(-mag))
  DEC[p * TR :+ TR, 0 :+ H] = exp(rate * sp)
  var b = B[p * TR :+ TR, 0 :+ H]
  BET[p * TR :+ TR, 0 :+ H] = 1.0 / (1.0 + exp(-b))
}}
"
    )
}
