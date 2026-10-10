// Attention kernel sources: prompt, blocked, split decode and persistent
// decode variants, plus rope.

use phobos_kernels::launch::WARP_THREADS;

use crate::backend::Attn;

/// Elements of one `[BC, head_dim]` tile in [`attention_src`].
///
/// Bounded by shared memory. Key and value tiles take one each, the codegen
/// adds a second pair for the masked loop replay, and all four (f16) must
/// fit in 48 KB.
pub(crate) const ATTN_TILE_ELEMS: usize = 4096;

/// Row cap on that tile, usually the tighter bound. The scan's remainder
/// takes up to `BC - 1` single-key steps, so rows are capped separately.
pub(crate) const ATTN_TILE_ROWS: usize = 16;

/// Cache rows one step of the scan covers. The single-key remainder after
/// the tiled loop is shorter than this.
pub(crate) fn attention_tile(head_dim: usize) -> usize {
    (ATTN_TILE_ELEMS / head_dim).clamp(1, ATTN_TILE_ROWS)
}

/// Cap on the query heads one decode split program covers.
///
/// The query, accumulator and rescale are `[QG, head_dim]` in shared
/// memory, so larger `QG` costs occupancy while its savings shrink.
pub(crate) const ATTN_QGROUP: usize = 2;

/// Query heads one decode split program carries: the largest value up to
/// [`ATTN_QGROUP`] that divides `group`.
///
/// A program's heads share one key head, so a program must not straddle two
/// groups.
pub(crate) fn attention_qgroup(group: usize) -> usize {
    (1..=ATTN_QGROUP)
        .rev()
        .find(|qgroup| group.is_multiple_of(*qgroup))
        .unwrap_or(1)
}

/// Elements of one `[BR, head_dim]` tile in [`attention_block_src`]. The
/// kernel stages about ten of these in static shared memory, which is
/// capped at 48 KB.
pub(crate) const ATTN_BLOCK_ELEMS: usize = 1024;

/// Queries one program of [`attention_block_src`] covers, for example four
/// at a head dimension of 256.
pub(crate) fn attention_block_tile(head_dim: usize) -> usize {
    (ATTN_BLOCK_ELEMS / head_dim).clamp(1, 16)
}

/// Query rows and keys of one [`attn_gemm_src`] score tile.
///
/// The tile is square, so on the tile straddling the diagonal key `j` and
/// query `i` share tile coordinates, and `tril` is the causal mask.
pub const ATTN_GEMM_TILE: usize = 64;

pub(crate) const ATTN_GEMM_STEP: usize = 32;

/// Row tile of the softmax and the mix, bounded by shared memory. The
/// softmax holds four square tiles at once, so 64 would not fit.
///
/// The mix must use the same row tile, since the softmax only zeroes a row
/// up to its own last query. Its step is shallower to fit the `[32, 64]`
/// accumulator.
pub const ATTN_SOFT_TILE: usize = 32;

pub(crate) const ATTN_MIX_STEP: usize = 16;

/// Rows one program of the key transpose carries.
pub(crate) const ATTN_KT_ROWS: usize = 8;

/// Whether a call tiles evenly for [`DeviceBackend::attention_gemm`]. The
/// fallback kernels need no alignment.
pub(crate) fn attn_gemm_fits(spec: Attn) -> bool {
    spec.rows >= ATTN_GEMM_TILE
        && spec.rows.is_multiple_of(ATTN_GEMM_TILE)
        && spec.start_pos.is_multiple_of(ATTN_GEMM_TILE)
        && spec.head_dim.is_multiple_of(ATTN_GEMM_TILE)
        && spec.total().is_multiple_of(ATTN_KT_ROWS)
}

/// Query rows and keys of one [`attn_tc_src`] score tile.
pub(crate) const ATTN_TC_TILE: usize = 64;

/// Bytes of scores [`attn_tc_src`] may hold at once. Past this a prompt takes
/// [`attention_block_src`], since the scores of every head are written out
/// in full.
pub(crate) const ATTN_TC_SCORE_BYTES: usize = 128 << 20;

/// Whether [`attn_tc_src`] takes this prompt: whole score tiles both ways,
/// head dimension a multiple of the tile, and the score planes within
/// [`ATTN_TC_SCORE_BYTES`].
pub(crate) fn attn_tc_fits(spec: Attn) -> bool {
    let score_bytes = spec.rows * spec.n_head * spec.total() * size_of::<f32>();
    spec.rows.is_multiple_of(ATTN_TC_TILE)
        && spec.start_pos.is_multiple_of(ATTN_TC_TILE)
        && spec.head_dim.is_multiple_of(ATTN_TC_TILE)
        && score_bytes <= ATTN_TC_SCORE_BYTES
}

/// Causal attention for a prompt on the tensor cores: every head's scores,
/// a softmax over them in place, and the mix, as three launches with heads
/// on the grid.
///
/// `S` is `[R, n_head * NK]`, head `h` in columns `h * NK`. The score kernel
/// skips tiles wholly past the diagonal. The softmax reads only up to each
/// row tile's diagonal tile and zeroes its masked half, so the mix must use
/// the softmax's row tile, [`ATTN_SOFT_TILE`]: a taller one would read
/// scores the softmax never normalized. Operands round to f16 on the way into
/// the tensor cores; the softmax and both accumulations stay f32.
pub(crate) fn attn_tc_src(head_dim: usize, group: usize) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    let (tile, step, soft) = (ATTN_TC_TILE, ATTN_GEMM_STEP, ATTN_SOFT_TILE);
    format!(
        "@tensorcore
@launch(256)
@autotune(TM in [{tile}], TN in [{tile}], TK in [{step}], D in [{head_dim}], G in [{group}])
@aligned(R = TM, NK = TN, QW = D, KW = D, SW = TN)
kernel attn_tc_scores(Q: tensor<f32>[R, QW], K: tensor<f16>[NK, KW], S: tensor<f32>[R, SW]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  let h = program_id(2)
  if pn * TN < NK - R + pm * TM + TM {{
    var acc: tile<f32>[TM, TN] = 0.0
    for kt in range(0, D, TK) {{
      var a = Q[pm * TM :+ TM, h * D + kt :+ TK]
      var b = K[pn * TN :+ TN, h / G * D + kt :+ TK]
      acc += dot_t(a, b)
    }}
    S[pm * TM :+ TM, h * NK + pn * TN :+ TN] = acc * {scale:.9}
  }}
}}

@launch(256)
@autotune(TM in [{soft}], TN in [{soft}])
@aligned(R = TM, NK = TN, SW = TN)
kernel attn_tc_softmax(K: tensor<f16>[NK, KW], S: tensor<f32>[R, SW], L: tensor<f32>[R, LW]) {{
  let p = program_id(0)
  let h = program_id(1)
  let c = h * NK
  let diag = NK - R + p * TM
  var m: tile<f32>[TM, 1] = -300000000.0
  for j in range(0, diag, TN) {{
    m = tmax(m, rowmax(S[p * TM :+ TM, c + j :+ TN]))
  }}
  var d = S[p * TM :+ TM, c + diag :+ TN]
  m = tmax(m, rowmax(d))
  var l: tile<f32>[TM, 1] = 0.0
  for j in range(0, diag, TN) {{
    var e = exp(S[p * TM :+ TM, c + j :+ TN] - m)
    l = l + rowsum(e)
    S[p * TM :+ TM, c + j :+ TN] = e
  }}
  var de = tril(exp(d - m))
  l = l + rowsum(de)
  S[p * TM :+ TM, c + diag :+ TN] = de
  L[p * TM :+ TM, h :+ 1] = l
}}

@tensorcore
@launch(256)
@autotune(TM in [{soft}], TN in [{tile}], TK in [{step}], D in [{head_dim}], G in [{group}])
@aligned(R = TM, NK = TK, SW = TK, KW = TN, QW = TN)
kernel attn_tc_mix(S: tensor<f32>[R, SW], V: tensor<f16>[NK, KW], L: tensor<f32>[R, LW],
                   O: tensor<f32>[R, QW]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  let h = program_id(2)
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, NK - R + pm * TM + TM, TK) {{
    var a = S[pm * TM :+ TM, h * NK + kt :+ TK]
    var b = V[kt :+ TK, h / G * D + pn * TN :+ TN]
    acc += dot(a, b)
  }}
  O[pm * TM :+ TM, h * D + pn * TN :+ TN] = acc / L[pm * TM :+ TM, h :+ 1]
}}
"
    )
}

/// Causal attention for a prompt as a score matmul, a softmax and a mix
/// matmul, with the full `[rows, keys]` score matrix in between.
///
/// [`crate::backend::Backend::attention`] uses it for shapes
/// [`attention_block_src`] declines: a head dimension whose block tile does
/// not divide 64, or a misaligned continuation.
///
/// Keys are transposed once per head so the score matmul is a plain `dot`.
/// Each head is gathered into its own buffer first, because a column window
/// at an unbounded offset would mask every operand and lose pipelining.
///
/// The kernels split where a score row must be complete: the softmax needs
/// the row max and the mix needs the row sum. All three skip blocks past
/// the diagonal.
pub fn attn_gemm_src(head_dim: usize, tile: usize) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    let (step, soft, mix_step) = (ATTN_GEMM_STEP, ATTN_SOFT_TILE, ATTN_MIX_STEP);
    format!(
        "@launch(256)
@autotune(TM in [{tile}], TN in [{tile}], TK in [{step}], TB in [{ATTN_KT_ROWS}], HD in [{head_dim}])
@aligned(NK = TN, KW = HD, DK = TB)
kernel attn_kt(K: tensor<f16>[NK, KW], T: tensor<f32>[DK, NK]) {{
  let p = program_id(0)
  var t = K[p * TB :+ TB, 0 :+ HD]
  T[0 :+ HD, p * TB :+ TB] = f32(transpose(t))
}}

@launch(256)
@autotune(TM in [{tile}], TN in [{tile}], TK in [{step}], HD in [{head_dim}])
@aligned(R = TM, NK = TN, DQ = TK, DK = TK)
kernel attn_scores(Q: tensor<f32>[R, DQ], KT: tensor<f32>[DK, NK], S: tensor<f32>[R, NK]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  if pn * TN < NK - R + pm * TM + TM {{
    var acc: tile<f32>[TM, TN] = 0.0
    for kt in range(0, DQ, TK) {{
      var a = Q[pm * TM :+ TM, kt :+ TK]
      var b = KT[kt :+ TK, pn * TN :+ TN]
      acc += dot(a, b)
    }}
    S[pm * TM :+ TM, pn * TN :+ TN] = acc * {scale:.9}
  }}
}}

@launch(256)
@autotune(TM in [{soft}], TN in [{soft}])
@aligned(R = TM, NK = TN)
kernel attn_softmax(S: tensor<f32>[R, NK], L: tensor<f32>[R, 1]) {{
  let p = program_id(0)
  let diag = NK - R + p * TM
  var m: tile<f32>[TM, 1] = -300000000.0
  for j in range(0, diag, TN) {{
    m = tmax(m, rowmax(S[p * TM :+ TM, j :+ TN]))
  }}
  var d = S[p * TM :+ TM, diag :+ TN]
  m = tmax(m, rowmax(d))

  var l: tile<f32>[TM, 1] = 0.0
  for j in range(0, diag, TN) {{
    var e = exp(S[p * TM :+ TM, j :+ TN] - m)
    l = l + rowsum(e)
    S[p * TM :+ TM, j :+ TN] = e
  }}
  var de = tril(exp(d - m))
  l = l + rowsum(de)
  S[p * TM :+ TM, diag :+ TN] = de
  L[p * TM :+ TM, 0 :+ 1] = l
}}

@launch(256)
@autotune(TM in [{soft}], TN in [{tile}], TK in [{mix_step}])
@aligned(R = TM, DV = TN, NK = TK)
kernel attn_mix(P: tensor<f32>[R, NK], V: tensor<f32>[NK, DV], L: tensor<f32>[R, 1],
                O: tensor<f32>[R, DV]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, NK - R + pm * TM + TM, TK) {{
    var a = P[pm * TM :+ TM, kt :+ TK]
    var b = V[kt :+ TK, pn * TN :+ TN]
    acc += dot(a, b)
  }}
  O[pm * TM :+ TM, pn * TN :+ TN] = acc / L[pm * TM :+ TM, 0 :+ 1]
}}
"
    )
}

/// Causal attention over a block of queries at once.
///
/// A program owns `BR` consecutive positions of one head, so one pass over
/// the cache serves all of them. `Q` is `[rows, n_head * head_dim]`, the
/// layout the norm and rotary use, so it needs no rearranging.
///
/// The key tile matches the query block, so on the diagonal tile column `j`
/// is key `base + j` and row `i` is query `base + i`, and `tril` is the
/// causal mask. A column past the last key only pairs with a row that is
/// also past the end and never stored.
///
/// The mask is applied to probabilities rather than scores, which is
/// equivalent. The running maximum includes masked entries, which only
/// shifts the exponentials down.
pub(crate) fn attention_block_src(
    n_head: usize,
    group: usize,
    head_dim: usize,
    tile: usize,
) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    format!(
        "@launch(256)
@autotune(NH in [{n_head}], G in [{group}], D in [{head_dim}], BR in [{tile}])
@aligned(KW = D)
@padstage
kernel attention_block(Q: tensor<f32>[R, QW], K: tensor<f16>[NK, KW],
                       V: tensor<f16>[NK, KW], O: tensor<f32>[R, QW]) {{
  let qt = program_id(0)
  let h = program_id(1)
  let qcol = h * D
  let kcol = h / G * D
  var q = Q[qt * BR :+ BR, qcol :+ D]

  var acc: tile<f32>[BR, D] = 0.0
  var m: tile<f32>[BR, 1] = -300000000.0
  var l: tile<f32>[BR, 1] = 0.0

  let base = NK - R + qt * BR
  for kt in range(0, base, BR) {{
    var k = K[kt :+ BR, kcol :+ D]
    var v = V[kt :+ BR, kcol :+ D]
    var s: tile<f32>[BR, BR] = dot_t(q, k)
    s = s * {scale:.9}
    var mn: tile<f32>[BR, 1] = rowmax(s)
    mn = tmax(m, mn)
    s = exp(s - mn)
    var corr: tile<f32>[BR, 1] = exp(m - mn)
    l = l * corr + rowsum(s)
    acc = acc * corr + dot(s, v)
    m = mn
  }}

  var dk = K[base :+ BR, kcol :+ D]
  var dv = V[base :+ BR, kcol :+ D]
  var ds: tile<f32>[BR, BR] = dot_t(q, dk)
  ds = ds * {scale:.9}
  var dm: tile<f32>[BR, 1] = rowmax(ds)
  dm = tmax(m, dm)
  ds = exp(ds - dm)
  ds = tril(ds)
  var dcorr: tile<f32>[BR, 1] = exp(m - dm)
  l = l * dcorr + rowsum(ds)
  acc = acc * dcorr + dot(ds, dv)

  O[qt * BR :+ BR, qcol :+ D] = acc / l
}}
"
    )
}

/// The rotary embedding, in place.
///
/// Angles come from a precomputed table, since the language has no sine
/// and the hardware one is too inexact at large positions. The caller
/// offsets `T` to the first row's position, so `r / H` indexes it.
///
/// Both halves are read before either store, which makes the in-place
/// update safe.
pub(crate) fn rope_src(heads: usize, half: usize) -> String {
    format!(
        "@launch(256)
@autotune(H in [{heads}])
kernel rope(X: tensor<f32>[R, D], T: tensor<f32>[P, RD]) {{
  let r = program_id(0)
  let p = r / H
  var a = X[r :+ 1, 0 :+ {half}]
  var b = X[r :+ 1, {half} :+ {half}]
  var c = T[p :+ 1, 0 :+ {half}]
  var s = T[p :+ 1, {half} :+ {half}]
  X[r :+ 1, 0 :+ {half}] = a * c - b * s
  X[r :+ 1, {half} :+ {half}] = a * s + b * c
}}
"
    )
}

/// A decode row's attention preparation, one program per head `h`: head
/// `h` of `S` RMS-normalized by the gains `G`, its first `2 * half`
/// channels rotated by the table row `T`, stored to `O`, and row `h` of `C`
/// copied to `E`. The query runs it with its output gate as the copy; the
/// key, with `cache`, with its value, both stored as f16 into the caches.
pub(crate) fn attn_prep_src(dim: usize, half: usize, eps: f32, cache: bool) -> String {
    let (name, ty) = match cache {
        true => ("attn_prep_kv", "f16"),
        false => ("attn_prep_q", "f32"),
    };
    let store = |v: String| if cache { format!("f16({v})") } else { v };
    let (rd, tail) = (2 * half, dim - 2 * half);
    let (rest_load, rest_sum, rest_store) = match tail {
        0 => (String::new(), String::new(), String::new()),
        _ => (
            format!("  var t = S[h :+ 1, {rd} :+ {tail}]
"),
            " + rowsum(t * t)".to_string(),
            format!("  O[h :+ 1, {rd} :+ {tail}] = {}
", store(format!("t * inv * G[0 :+ 1, {rd} :+ {tail}]"))),
        ),
    };
    format!(
        "@launch(256)
kernel {name}(S: tensor<f32>[H, SP], G: tensor<f32>[GR, GD], T: tensor<f32>[P, RD],
              O: tensor<{ty}>[H, OP], C: tensor<f32>[H, CP], E: tensor<{ty}>[H, EP]) {{
  let h = program_id(0)
  var a = S[h :+ 1, 0 :+ {half}]
  var b = S[h :+ 1, {half} :+ {half}]
{rest_load}  var ss: tile<f32>[1, 1] = rowsum(a * a) + rowsum(b * b){rest_sum}
  var inv: tile<f32>[1, 1] = 1.0 / sqrt(ss / {dim}.0 + {eps:.12})
  var na = a * inv * G[0 :+ 1, 0 :+ {half}]
  var nb = b * inv * G[0 :+ 1, {half} :+ {half}]
  var c = T[0 :+ 1, 0 :+ {half}]
  var s = T[0 :+ 1, {half} :+ {half}]
  O[h :+ 1, 0 :+ {half}] = {lo}
  O[h :+ 1, {half} :+ {half}] = {hi}
{rest_store}  E[h :+ 1, 0 :+ {dim}] = {copy}
}}
",
        lo = store("na * c - nb * s".into()),
        hi = store("na * s + nb * c".into()),
        copy = store(format!("C[h :+ 1, 0 :+ {dim}]")),
    )
}

/// [`rope_src`] that reads a strided window of a fused QKV projection and
/// writes a dense destination. A prompt's query and key both need this,
/// since past one row the projection's three parts interleave.
///
/// A fused row of `S` is `stride_heads` heads wide, and a program touches
/// only the `heads` in this call's window. The caller's pointer offset
/// selects the window; see [`Backend::rope_gather`].
///
/// When `half * 2 < head_dim`, the channels past `rope_dim` are copied
/// through unrotated.
pub(crate) fn rope_gather_src(
    heads: usize,
    half: usize,
    stride_heads: usize,
    head_dim: usize,
) -> String {
    let tail = head_dim - 2 * half;
    let passthrough = if tail > 0 {
        format!(
            "  D[r :+ 1, {rope_dim} :+ {tail}] = S[row :+ 1, {rope_dim} :+ {tail}]\n",
            rope_dim = 2 * half,
        )
    } else {
        String::new()
    };
    format!(
        "@launch(256)
@autotune(H in [{heads}])
kernel rope_gather(S: tensor<f32>[RS, {head_dim}], T: tensor<f32>[P, RD], D: tensor<f32>[RD, {head_dim}]) {{
  let r = program_id(0)
  let p = r / H
  let h = r - p * H
  let row = p * {stride_heads} + h
  var a = S[row :+ 1, 0 :+ {half}]
  var b = S[row :+ 1, {half} :+ {half}]
  var c = T[p :+ 1, 0 :+ {half}]
  var s = T[p :+ 1, {half} :+ {half}]
  D[r :+ 1, 0 :+ {half}] = a * c - b * s
  D[r :+ 1, {half} :+ {half}] = a * s + b * c
{passthrough}}}
"
    )
}

/// Decode attention over the whole cache, one group of query rows per
/// program.
///
/// The key axis is split across the grid, and each split is split again
/// across the block's warps by [`warp_partial`], a `phobos-lang` builtin in
/// `phobos-lang/src/codegen/tile/warp_attn.rs`.
///
/// Each warp takes its own `[lo, hi)` key range and keeps `(m, l, acc)` in
/// registers, with the head dimension spread over its lanes and reduced by
/// shuffle. Warps never wait on each other.
///
/// A block still publishes one `(m, l, acc)` per query row:
/// [`attention_combine`] merges the `WCT` warp partials first. Widening `S`
/// to `S * WCT` instead would exceed the merge's shared memory.
///
/// Keys are processed one at a time, so there is no remainder loop. `K`
/// and `V` stay f16 and widen on load; the query, max and accumulator are
/// f32.
pub(crate) fn attention_split_src(
    n_head: usize,
    group: usize,
    head_dim: usize,
    qgroup: usize,
    splits: usize,
    wct: usize,
) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    let qw = qgroup * wct;
    let combine = attention_combine(qgroup, "");
    // `warp_partial` requires a launch width of exactly `wct` warps.
    let cta = wct * WARP_THREADS;
    format!(
        "@launch({cta})
@autotune(NH in [{n_head}], G in [{group}], QG in [{qgroup}], D in [{head_dim}], S in [{splits}],
          WCT in [{wct}], QW in [{qw}])
@aligned(KW = D)
kernel attention_split(Q: tensor<f32>[R, D], K: tensor<f16>[NK, KW],
                       V: tensor<f16>[NK, KW],
                       P: tensor<f32>[SH, D], ML: tensor<f32>[NH, MW]) {{
  let g = program_id(0)
  let s = program_id(1)
  let h = g * QG
  let col = h / G * D
  var q = Q[h :+ QG, 0 :+ D]

  let per = (NK + S - 1) / S
  var lo = s * per
  if lo > NK {{
    lo = NK
  }}
  var hi = lo + per
  if hi > NK {{
    hi = NK
  }}

  var wm: tile<f32>[QG, WCT] = -300000000.0
  var wl: tile<f32>[QG, WCT] = 0.0
  var wacc: tile<f32>[QW, D] = 0.0
  warp_partial(q, K, V, lo, hi, col, wm, wl, wacc, {scale:.9})
{combine}}}

@launch(256)
@autotune(NH in [{n_head}], D in [{head_dim}], S in [{splits}])
kernel attention_merge(P: tensor<f32>[S, PW], ML: tensor<f32>[NH, MW],
                       O: tensor<f32>[R, D]) {{
  let h = program_id(0)
  var mv = ML[h :+ 1, 0 :+ S]
  var m: tile<f32>[1, 1] = rowmax(mv)
  var c: tile<f32>[1, S] = exp(mv - m)
  var l: tile<f32>[1, 1] = rowsum(c * ML[h :+ 1, S :+ S])
  var acc: tile<f32>[1, D] = dot(c, P[0 :+ S, h * D :+ D])
  O[h :+ 1, 0 :+ D] = acc / l
}}
"
    )
}

/// The online-softmax merge of the `WCT` warp partials in `wm`, `wl` and
/// `wacc`, used by [`attention_split_src`] and [`attention_persist_src`].
/// It is the same merge `attention_merge` runs across splits.
///
/// `indent` is extra leading whitespace for every line, so the text can be
/// nested inside a `for` or `if`.
fn attention_combine(qgroup: usize, indent: &str) -> String {
    let mut out = String::new();
    for i in 0..qgroup {
        out.push_str(&format!(
            "{indent}  var mv{i} = wm[{i} :+ 1, 0 :+ WCT]
{indent}  var mx{i}: tile<f32>[1, 1] = rowmax(mv{i})
{indent}  var c{i}: tile<f32>[1, WCT] = exp(mv{i} - mx{i})
{indent}  var ll{i}: tile<f32>[1, 1] = rowsum(c{i} * wl[{i} :+ 1, 0 :+ WCT])
{indent}  var ac{i}: tile<f32>[1, D] = dot(c{i}, wacc[{i} * WCT :+ WCT, 0 :+ D])
{indent}  P[s * NH + h + {i} :+ 1, 0 :+ D] = ac{i}
{indent}  ML[h + {i} :+ 1, s :+ 1] = mx{i}
{indent}  ML[h + {i} :+ 1, S + s :+ 1] = ll{i}
"
        ));
    }
    out
}

/// [`attention_split_src`]'s two kernels as one `@persistent` kernel, with
/// a `grid_barrier()` between the phases.
///
/// Phase one runs the split body over a grid-strided range of
/// `(group, split)` units. Phase two runs the merge body over a
/// grid-strided range of heads. The trip counts `IT1` and `IT2` are
/// compiled in and guarded with `if unit < total`, because a strided loop
/// needs a static shape.
///
/// `P` and `PM` are the same scratch pointer, bound under each phase's
/// shape, as [`DeviceBackend::attention_decode`] passes to the separate
/// kernels.
///
/// All `BLOCKS` blocks must be resident at once or `grid_barrier`
/// deadlocks, so the caller derives `BLOCKS` from the occupancy API.
/// `@dynshared` takes the max of the two phases' shared memory, which is
/// safe because phase two reads only the global `PM` and `ML`.
pub(crate) fn attention_persist_src(
    n_head: usize,
    group: usize,
    head_dim: usize,
    qgroup: usize,
    splits: usize,
    wct: usize,
    blocks: u32,
) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    let groups = n_head / qgroup;
    let units1 = groups * splits;
    let it1 = (units1 as u32).div_ceil(blocks);
    let it2 = (n_head as u32).div_ceil(blocks);
    let qw = qgroup * wct;
    let combine = attention_combine(qgroup, "    ");
    format!(
        "@launch(256)
@persistent
@dynshared
@autotune(NH in [{n_head}], G in [{group}], QG in [{qgroup}], D in [{head_dim}],
          S in [{splits}], WCT in [{wct}], QW in [{qw}], U1 in [{units1}], IT1 in [{it1}],
          IT2 in [{it2}], BLOCKS in [{blocks}])
@aligned(KW = D)
kernel attention_persist(Q: tensor<f32>[R, D], K: tensor<f16>[NK, KW],
                       V: tensor<f16>[NK, KW],
                       P: tensor<f32>[SH, D], PM: tensor<f32>[S, PW],
                       ML: tensor<f32>[NH, MW], O: tensor<f32>[R, D],
                       BAR: tensor<i32>[2, 1]) {{
  let pid = program_id(0)

  for i1 in range(0, IT1) {{
    let u = pid + i1 * BLOCKS
    if u < U1 {{
      let g = u / S
      let s = u % S
      let h = g * QG
      let col = h / G * D
      var q = Q[h :+ QG, 0 :+ D]

      let per = (NK + S - 1) / S
      var lo = s * per
      if lo > NK {{
        lo = NK
      }}
      var hi = lo + per
      if hi > NK {{
        hi = NK
      }}

      var wm: tile<f32>[QG, WCT] = -300000000.0
      var wl: tile<f32>[QG, WCT] = 0.0
      var wacc: tile<f32>[QW, D] = 0.0
      warp_partial(q, K, V, lo, hi, col, wm, wl, wacc, {scale:.9})
{combine}    }}
  }}

  grid_barrier(BAR)

  for i2 in range(0, IT2) {{
    let u2 = pid + i2 * BLOCKS
    if u2 < NH {{
      var mv = ML[u2 :+ 1, 0 :+ S]
      var mm: tile<f32>[1, 1] = rowmax(mv)
      var c: tile<f32>[1, S] = exp(mv - mm)
      var ll: tile<f32>[1, 1] = rowsum(c * ML[u2 :+ 1, S :+ S])
      var oacc: tile<f32>[1, D] = dot(c, PM[0 :+ S, u2 * D :+ D])
      O[u2 :+ 1, 0 :+ D] = oacc / ll
    }}
  }}
}}
"
    )
}

/// Pieces the key axis is split into while decoding.
///
/// Fixed rather than derived from the cache length, so the launch shape
/// never changes and the cached graph stays valid. A piece with no keys
/// exits at once.
///
/// The merge folds a head's pieces into one `[S, head_dim]` tile, so this
/// times [`ATTN_QGROUP`] is bounded by shared memory.
pub(crate) const ATTN_SPLITS: usize = 8;

/// Warps a block of [`attention_split_src`] or [`attention_persist_src`]
/// splits its key range across, one piece per warp. Equal to the block's
/// full warp count at `@launch(256)`, so no warp is idle.
pub(crate) const ATTN_WARP_SPLITS: usize = 8;

pub(crate) fn attention_src(n_head: usize, group: usize, head_dim: usize, tile: usize) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    format!(
        "@launch(256)
@autotune(NH in [{n_head}], G in [{group}], D in [{head_dim}], BC in [{tile}])
@aligned(KW = D)
kernel attention(Q: tensor<f32>[R, D], K: tensor<f16>[NK, KW],
                 V: tensor<f16>[NK, KW], O: tensor<f32>[R, D]) {{
  let t = program_id(0)
  let h = program_id(1)
  let col = h / G * D
  let row = t * NH + h
  var q = Q[row :+ 1, 0 :+ D]

  var acc: tile<f32>[1, D] = 0.0
  var m: tile<f32>[1, 1] = -300000000.0
  var l: tile<f32>[1, 1] = 0.0

  let visible = NK - R / NH + t + 1
  let full = visible / BC * BC
  for kt in range(0, full, BC) {{
    var k = K[kt :+ BC, col :+ D]
    var v = V[kt :+ BC, col :+ D]
    var s: tile<f32>[1, BC] = dot_t(q, k)
    s = s * {scale:.9}
    var mn: tile<f32>[1, 1] = rowmax(s)
    mn = tmax(m, mn)
    s = exp(s - mn)
    var corr: tile<f32>[1, 1] = exp(m - mn)
    l = l * corr + rowsum(s)
    acc = acc * corr + dot(s, v)
    m = mn
  }}
  for j in range(full, visible, 1) {{
    var k1 = K[j :+ 1, col :+ D]
    var v1 = V[j :+ 1, col :+ D]
    var s1: tile<f32>[1, 1] = dot_t(q, k1)
    s1 = s1 * {scale:.9}
    var mn: tile<f32>[1, 1] = tmax(m, s1)
    var p: tile<f32>[1, 1] = exp(s1 - mn)
    var corr: tile<f32>[1, 1] = exp(m - mn)
    l = l * corr + p
    acc = acc * corr + p * v1
    m = mn
  }}
  O[row :+ 1, 0 :+ D] = acc / l
}}
"
    )
}
