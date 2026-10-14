// Q8_0 kernel sources: the activation quantizer, the dp4a and imma
// contractions, the fused dot, and the split-k reduction.

use crate::backend::Q8_BLOCK;

/// Output tile along N for the quantized single-row matmul.
pub(crate) const Q8_TN: usize = 32;

/// Output tile for the quantized tensor-core matmul: the m8 fragment's
/// depth, and an 8x8 tile per warp so no two warps read the same weight
/// byte. Deeper tiles do not fit in shared memory.
pub(crate) const Q8_MMA_TM: usize = 8;

pub(crate) const Q8_MMA_TN: usize = 64;

/// Blocks quantized per CTA in the activation-quantization kernel.
pub(crate) const QUANT_TB: usize = 4;

/// Blocks per CTA for a prompt, which has thousands of blocks. Four leaves
/// half a 256-thread CTA idle, and 64 uses too much shared memory to fit
/// two CTAs.
pub(crate) const QUANT_TB_WIDE: usize = 16;

/// Quantizes an activation to int8 with one scale per block of 32. It is
/// viewed as `[blocks, 32]`, so each row is a block and `rowmax` gives its
/// magnitude.
///
/// Rounding is the hardware's ties-to-even, matching the host reference.
pub(crate) const QUANTIZE_SRC: &str = "\
@launch(256)
@autotune(TB in [8])
kernel quantize(X: tensor<f32>[R, 32], Q: tensor<i8>[R, 32], S: tensor<f32>[R, D]) {
  let p = ctx.block.x
  var x = X[p * TB :+ TB, 0 :+ 32]
  var mx: tile<f32>[TB, 1] = rowmax(abs(x))
  var inv = 127.0 / (mx + 0.00000001)
  var y = x * inv
  Q[p * TB :+ TB, 0 :+ 32] = i8(i32(round(y)))
  S[p * TB :+ TB, 0 :+ 1] = mx / 127.0
}
";

/// The Q8_0 projection with both operands in int8, contracted by `dp4a`.
///
/// `dot_t` contracts the last axis of both, so activation and weight row
/// both walk `k` contiguously and four bytes of each pack into one
/// instruction. Nothing is dequantized into shared memory.
pub(crate) const Q8_DP4A_SRC: &str = "\
@launch(256)
@autotune(TN in [32])
{ALIGNED}
kernel q8_dp4a(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
               C: tensor<f32>[M, N]) {
  let pn = ctx.block.x
  var acc: tile<f32>[1, TN] = 0.0
  for kt in range(0, K, 32) {
    let b = kt / 32
    let a = A[0 :+ 1, kt :+ 32]
    let w = W[pn * TN :+ TN, kt :+ 32]
    let ws = WS[b :+ 1, pn * TN :+ TN]
    let da = AS[0 :+ 1, b :+ 1]
    acc += f32(dot_t(a, w)) * ws * da
  }
  C[0 :+ 1, pn * TN :+ TN] = acc
}
";

/// The batched Q8_0 projection on the integer tensor cores.
///
/// [`Q8_DP4A_SRC`]'s contraction, tiled in both directions, so a prompt's
/// rows are covered by the grid and the contraction lowers to `mma.sync`,
/// whose smallest integer output tile is 8x8.
pub(crate) const Q8_MMA_SRC: &str = "\
@launch(256)
@autotune(TM in [8], TN in [64])
{ALIGNED}
kernel q8_mma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
              W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
              C: tensor<f32>[M, N]) {
  let pm = ctx.block.x
  let pn = ctx.block.y
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, K, 32) {
    let b = kt / 32
    let a = A[pm * TM :+ TM, kt :+ 32]
    let w = W[pn * TN :+ TN, kt :+ 32]
    let ws = WS[b :+ 1, pn * TN :+ TN]
    let da = AS[pm * TM :+ TM, b :+ 1]
    acc += f32(dot_t(a, w)) * ws * da
  }
  C[pm * TM :+ TM, pn * TN :+ TN] = acc
}
";

/// The Q8_0 contraction of `k` in `[from, to)` into `out`'s `[TM, TN]`
/// tile, block scales folded in: an int8 accumulator the integer tensor
/// cores keep in registers across the loop.
fn q8_qmma_loop(from: &str, to: &str, out: &str, indent: &str) -> String {
    let body = format!(
        "var acc: tile<f32>[TM, TN] = 0.0
for kt in range({from}, {to}, 32) {{
  let b = kt / 32
  let a = A[pm * TM :+ TM, kt :+ 32]
  let w = W[pn * TN :+ TN, kt :+ 32]
  let ws = WS[b :+ 1, pn * TN :+ TN]
  let da = AS[pm * TM :+ TM, b :+ 1]
  acc += f32(dot_t(a, w)) * ws * da
}}
{out}[pm * TM :+ TM, pn * TN :+ TN] = acc
"
    );
    body.lines().map(|l| format!("{indent}{l}
")).collect()
}

/// The Q8_0 projection prompt passes run on the integer tensor cores.
pub(crate) fn q8_qmma_src(block: usize) -> String {
    format!(
        "@launch({block})
@autotune(TM in [64], TN in [64])
@aligned(M = TM, N = TN, K = 32)
kernel q8_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
               C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
{}}}
",
        q8_qmma_loop("0", "K", "C", "  ")
    )
}

/// Output tiles of the `q8_qmma` projection: the depth, and the available
/// widths, widest first.
///
/// The output tile bounds a warp's patch of tensor-core tiles, which is
/// what amortizes operand loads and scales. The 64-deep kernel takes the
/// rows a prompt leaves over.
pub(crate) const Q8_QMMA_TM: usize = 128;

pub(crate) const Q8_QMMA_SHALLOW: usize = 64;

pub(crate) const Q8_QMMA_WIDTHS: [usize; 4] = [128, 64, 48, 32];

/// Whether one of the tiles above divides a projection this wide.
pub(crate) fn qmma_takes(n: usize) -> bool {
    Q8_QMMA_WIDTHS.iter().any(|&tn| n.is_multiple_of(tn))
}

pub(crate) const Q8_QMMA_TN: usize = 64;

/// Threads the projection's CTA carries, half the usual 256. A patch holds
/// 128 live accumulators regardless of CTA size, so registers bound the
/// warps per multiprocessor, and a narrower CTA gives the same warps over
/// twice the grid.
pub(crate) const Q8_QMMA_CTA: usize = 128;

/// The widest column tile that divides `n`, however few blocks that leaves.
/// A larger warp patch matters more than filling the grid.
pub(crate) fn qmma_width(n: usize) -> usize {
    Q8_QMMA_WIDTHS
        .iter()
        .copied()
        .find(|&tn| n.is_multiple_of(tn))
        .unwrap_or(Q8_QMMA_TN)
}

/// A projection as a single `qdot_t`, the source of Q5_0's through
/// [`q50_qdot_src`] (Q8_0 decodes with [`q8_qdot_lanes_src`]). The tile is
/// eight outputs, one per warp.
///
/// `qdot_t` folds the block scales in. A warp owns an output and its lanes
/// split `k`, so a warp reads contiguous bytes with nothing staged. No
/// k-split is needed, since 32 lanes per output already fill the machine.
pub(crate) const Q8_QDOT_SRC: &str = "\
@launch(256)
@autotune(TN in [8])
@aligned(N = TN)
kernel q8_qdot(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[N, KB],
               C: tensor<f32>[M, N]) {
  let pn = ctx.block.x
  C[0 :+ 1, pn * TN :+ TN] = qdot_t(A[0 :+ 1, :], AS[0 :+ 1, :],
                                    W[pn * TN :+ TN, :], WS[pn * TN :+ TN, :])
}
";

/// [`Q8_QDOT_SRC`] adding into its destination, for the residual add at the
/// end of every block.
pub(crate) const Q8_QDOT_ADD_SRC: &str = "@launch(256)
@autotune(TN in [8])
@aligned(N = TN)
kernel q8_qdot_add(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                   W: tensor<i8>[N, K], WS: tensor<f32>[N, KB],
                   C: tensor<f32>[M, N]) {
  let pn = ctx.block.x
  C[0 :+ 1, pn * TN :+ TN] += qdot_t(A[0 :+ 1, :], AS[0 :+ 1, :],
                                     W[pn * TN :+ TN, :], WS[pn * TN :+ TN, :])
}
";

/// [`Q8_QDOT_SRC`], or [`Q8_QDOT_ADD_SRC`] when it adds, against a Q5_0
/// weight held as its blocks less the scale: `q50_qdot_t`.
pub(crate) fn q50_qdot_src(accumulate: bool) -> String {
    q50_of(if accumulate { Q8_QDOT_ADD_SRC } else { Q8_QDOT_SRC })
}

/// The prompt projection against a Q5_0 weight, its tile fixed at `tm` by
/// `tn`: `q50_qmma_t`, or on a chip the `.ph` projections are measured on,
/// [`q50_qmma_ph_src`].
pub(crate) fn q50_qmma_src(block: usize, tm: usize, tn: usize) -> String {
    if super::qgemm_ph::measured_on(phobos_kernels::compile::chip()) {
        return q50_qmma_ph_src(block, tm, tn);
    }
    let src = format!(
        "@launch({block})
@autotune(TM in [64], TN in [64])
@aligned(M = TM, N = TN, K = 32)
kernel q8_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
               C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  C[pm * TM :+ TM, pn * TN :+ TN] = qmma_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                           W[pn * TN :+ TN, :], WS[:, pn * TN :+ TN])
}}
"
    );
    q50_of(&src).replace(
        "@autotune(TM in [64], TN in [64])",
        &format!("@autotune(TM in [{tm}], TN in [{tn}])"),
    )
}

/// [`q50_qmma_src`] as [`q8_qmma_src`]'s loop over a generated weight: each
/// lane decodes the words of its own fragments straight from the block, so
/// the weight never touches shared memory. A block is a word of fifth bits
/// and 16 bytes of nibbles: element `e` is nibble `e / 16` of byte `e % 16`
/// under bit `e` of the word, less 16.
fn q50_qmma_ph_src(block: usize, tm: usize, tn: usize) -> String {
    format!(
        "fn q50_word(j, q) {{
  let row = pn * TN + j
  let qh = word(W[row, b * 20 :+ 4], 0)
  let qs = word(W[row, b * 20 + 4 + 4 * (q % 4) :+ 4], 0)
  let hb = ((((qh >> (4 * q)) & 15) * {SPREAD}) & {LOW}) << 4
  let v = ((qs >> (4 * (q / 4))) & {NIB}) | hb
  ((v | {HIGH}) - {SIXTEEN}) ^ {HIGH}
}}

@launch({block})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 32, KW = 20)
kernel q50_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                W: tensor<i8>[N, KW], WS: tensor<f32>[KB, N],
                C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, K, 32) {{
    let b = kt / 32
    let a = A[pm * TM :+ TM, kt :+ 32]
    let w: tile<i8>[TN, 32] = generate(q50_word)
    let ws = WS[b :+ 1, pn * TN :+ TN]
    let da = AS[pm * TM :+ TM, b :+ 1]
    acc += f32(dot_t(a, w)) * ws * da
  }}
  C[pm * TM :+ TM, pn * TN :+ TN] = acc
}}
",
        SPREAD = super::iq_lanes::SPREAD,
        LOW = super::iq_lanes::LOW_BITS,
        NIB = 0x0F0F_0F0Fu32,
        HIGH = 0x8080_8080u32,
        SIXTEEN = 0x1010_1010u32,
    )
}

/// A Q8_0 kernel source turned onto a Q5_0 weight, `[N, K / 32 * 20]`.
fn q50_of(src: &str) -> String {
    src.replace("kernel q8_", "kernel q50_")
        .replace("W: tensor<i8>[N, K]", "W: tensor<i8>[N, KW]")
        .replace("qdot_t(", "q50_qdot_t(")
        .replace("qmma_t(", "q50_qmma_t(")
}

/// [`Q8_QDOT_SRC`] written with lanes instead of `qdot_t`: a warp per
/// output, each lane 16 bytes of `k` as four `dp4a` under one scale pair,
/// and a shuffle sum. `+=` when it adds into its destination.
pub(crate) fn q8_qdot_lanes_src(accumulate: bool) -> String {
    let (name, assign) = if accumulate { ("q8_qdot_add", "+=") } else { ("q8_qdot", "=") };
    format!(
        "@launch(256)
@aligned(K = 32)
kernel {name}(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
              W: tensor<i8>[N, K], WS: tensor<f32>[N, KB],
              C: tensor<f32>[M, N]) {{
  let col = ctx.block.x * ctx.warps + ctx.warp
  if col < N {{
    var acc: f32 = 0.0
    for k in range(ctx.lane * 16, K, ctx.lanes * 16) {{
      let b = k / 32
      let dots = sum(i32(A[0, k :+ 16]) * i32(W[col, k :+ 16]))
      acc += f32(dots) * AS[0, b] * WS[col, b]
    }}
    let total = sum<over ctx.lane>(acc)
    if ctx.lane == 0 {{
      C[0, col] {assign} total
    }}
  }}
}}
"
    )
}

/// Outputs per CTA in [`Q8_QDOT_SRC`], one per warp.
pub(crate) const Q8_QDOT_TN: usize = 8;

pub(crate) fn q8_qdot_persist_src(iters: usize, blocks: u32, accumulate: bool) -> String {
    let (name, assign) = if accumulate {
        ("q8_qdot_persist_add", "+=")
    } else {
        ("q8_qdot_persist", "=")
    };
    format!(
        "@launch(256)
@persistent
@autotune(TN in [{tn}], BLOCKS in [{blocks}], ITERS in [{iters}])
@aligned(N = TN)
kernel {name}(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
              W: tensor<i8>[N, K], WS: tensor<f32>[N, KB],
              C: tensor<f32>[M, N]) {{
  let pn = ctx.block.x
  for i in range(0, ITERS) {{
    let t = pn * TN + i * BLOCKS * TN
    if t < N {{
      C[0 :+ 1, t :+ TN] {assign} qdot_t(A[0 :+ 1, :], AS[0 :+ 1, :],
                                  W[t :+ TN, :], WS[t :+ TN, :])
    }}
  }}
}}
",
        tn = Q8_QDOT_TN
    )
}

/// The Q8_0 projection with the contraction split across the grid.
///
/// [`Q8_DP4A_SRC`] has only `n / TN` blocks, which starves a narrow decode
/// projection. Splitting `k` adds blocks.
///
/// Program `(pn, ps)` takes output tile `pn` and the `k` slice at `ps`, and
/// writes its partial sum to row `ps` of `P`; `q8_reduce` sums them. The
/// slice comes from the extents, so one module serves every shape and
/// split count.
pub(crate) const Q8_SPLIT_SRC: &str = "\
@launch(256)
@autotune(TN in [32])
{ALIGNED}
kernel q8_split(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
                P: tensor<f32>[S, N]) {
  let pn = ctx.block.x
  let ps = ctx.block.y
  let slice = K / S
  let from = ps * slice
  var acc: tile<f32>[1, TN] = 0.0
  for kt in range(from, from + slice, 32) {
    let b = kt / 32
    let a = A[0 :+ 1, kt :+ 32]
    let w = W[pn * TN :+ TN, kt :+ 32]
    let ws = WS[b :+ 1, pn * TN :+ TN]
    let da = AS[0 :+ 1, b :+ 1]
    acc += f32(dot_t(a, w)) * ws * da
  }
  P[ps :+ 1, pn * TN :+ TN] = acc
}

@launch(256)
@autotune(RT in [128])
kernel q8_reduce(P: tensor<f32>[S, N], C: tensor<f32>[M, N]) {
  let pn = ctx.block.x
  var acc: tile<f32>[1, RT] = 0.0
  for s in range(0, S, 1) {
    acc += P[s :+ 1, pn * RT :+ RT]
  }
  C[0 :+ 1, pn * RT :+ RT] = acc
}
";

/// Output tile for the split-K reduction.
pub(crate) const Q8_REDUCE_TN: usize = 128;

/// Block count the split aims for, and the most slices it cuts `k` into.
/// The cap keeps slices from getting so short that the reduction costs
/// more than the split saves.
pub(crate) const Q8_SPLIT_TARGET: usize = 256;

pub(crate) const Q8_SPLIT_MAX: usize = 16;

/// Slices of `k` for a `[1, k] x [k, n]` projection. One means no split:
/// [`Q8_DP4A_SRC`] with no reduction, which wide projections use.
pub(crate) fn q8_splits(n: usize, k: usize) -> usize {
    let grid = n.div_ceil(Q8_TN).max(1);
    let blocks = k / Q8_BLOCK;
    let mut splits = (Q8_SPLIT_TARGET / grid).min(Q8_SPLIT_MAX);

    // Scales change every Q8_0 block, so a slice must be whole blocks.
    while splits > 1 && !blocks.is_multiple_of(splits) {
        splits /= 2;
    }

    splits.max(1)
}

/// Block count below which `q8_qmma`'s deep tile is split along `k` with
/// [`q8_qmma_split_src`].
///
/// The unsplit grid is `(rows / TM) * (n / TN)`, so a short prompt or a
/// narrow projection leaves most of the card idle. Above this, splitting
/// only adds a reduction pass.
pub(crate) const Q8_QMMA_SPLIT_THRESHOLD: usize = 48;

/// Block count a split aims to reach: two per SM on a 48-SM card, the
/// register-bound occupancy of `q8_qmma`'s patch.
pub(crate) const Q8_QMMA_SPLIT_TARGET: usize = 96;

/// Most splits. Each split adds an `if ps == i` arm and an output operand
/// (see [`q8_qmma_split_src`]), so this bounds the kernel's parameters and
/// compiled variants.
pub(crate) const Q8_QMMA_SPLIT_MAX: usize = 8;

/// When `q8_qmma`'s deep tile splits on a card with `sms`: the block count
/// below which it splits, the block count it aims for, and the fewest
/// splits worth their reduction pass.
///
/// Up to 48 SMs these are the constants above, tuned on a 48-SM card and
/// measured even or better on a 24-SM one. Past that the grid starves at
/// longer prompts too, and two blocks per SM with at least four splits is
/// what measured best on a 128-SM card: pp512 up to 9 percent faster. Two
/// splits took the same gain from the projection but slowed the kernels
/// around it, through the partials' extra traffic.
pub(crate) fn q8_qmma_split_gate(sms: usize) -> (usize, usize, usize) {
    if sms <= Q8_QMMA_SPLIT_THRESHOLD {
        (Q8_QMMA_SPLIT_THRESHOLD, Q8_QMMA_SPLIT_TARGET, Q8_QMMA_SPLIT_MAX)
    } else {
        (2 * sms, 2 * sms, 4)
    }
}

/// Splits for `q8_qmma`'s deep tile at `rows x n x k` on a card with `sms`,
/// where 1 means no split. Only a starved grid splits; see
/// [`q8_qmma_split_gate`].
pub(crate) fn q8_qmma_splits(rows: usize, n: usize, k: usize, wide: usize, sms: usize) -> usize {
    let (below, target, fewest) = q8_qmma_split_gate(sms);
    let unsplit = (rows / Q8_QMMA_TM) * (n / wide);
    if unsplit == 0 || unsplit >= below {
        return 1;
    }
    let blocks = k / Q8_BLOCK;
    let mut splits = (target / unsplit).min(Q8_QMMA_SPLIT_MAX).min(blocks);
    while splits > 1 && !blocks.is_multiple_of(splits) {
        splits /= 2;
    }
    // The reduce pass costs about the same at any split count, so a split
    // too shallow to fill the card does not pay for it.
    if splits >= fewest { splits } else { 1 }
}

/// CTA threads and column tile of the narrow-CTA deep tile: half of
/// [`Q8_QMMA_CTA`] and half of [`Q8_QMMA_WIDTHS`]'s widest entry.
///
/// Both must be halved together. Then `qmma_patch` keeps the same
/// `(rm=8, rn=8)` per-warp patch as the 128-wide config, and the grid
/// doubles. Halving `TN` alone would shrink the patch to `(rm=4, rn=8)`.
pub(crate) const Q8_QMMA_NARROW_CTA: usize = 64;

pub(crate) const Q8_QMMA_NARROW_TN: usize = 64;

/// Whether `q8_qmma`'s deep tile at `rows x n` may take the narrow-CTA
/// path instead.
///
/// Requires `wide` to be the widest tile, since a narrower one has already
/// lost the patch size this path preserves. Gated on block count like
/// [`q8_qmma_splits`].
///
/// The path is off by default. The threshold is a flat block count that
/// ignores `rows`, and a shape just under it can regress. Make the gate
/// scale with how starved the grid is before enabling it by default.
pub(crate) fn q8_qmma_narrow_eligible(rows: usize, n: usize, wide: usize) -> bool {
    if wide != Q8_QMMA_WIDTHS[0] || !n.is_multiple_of(Q8_QMMA_NARROW_TN) {
        return false;
    }
    let unsplit = (rows / Q8_QMMA_TM) * (n / wide);
    unsplit > 0 && unsplit < Q8_QMMA_SPLIT_THRESHOLD
}

/// The split-K variant of [`q8_qmma_src`], for shapes where
/// [`q8_qmma_splits`] returns more than one.
///
/// Each split has its own output operand and `if ps == i` arm rather than
/// one indexed write. The accumulator is written straight to global memory
/// only when the slice is provably in bounds from its offset, which an
/// offset built from `ctx.block.z` can never be.
///
/// The `k` slice bounds are compiled in as literals, so `@aligned` covers
/// `A`, `AS`, `W` and `WS` too. The arms branch on a CTA-uniform grid
/// coordinate, so they do not diverge.
pub(crate) fn q8_qmma_split_src(block: usize, k: usize, s: usize) -> String {
    let slice = k / s;
    let sb = slice / Q8_BLOCK;
    let mut params = String::new();
    let mut body = String::new();
    for i in 0..s {
        let from = i * slice;
        params.push_str(&format!(", P{i}: tensor<f32>[M, N]"));
        let arm = q8_qmma_loop(&from.to_string(), &(from + slice).to_string(), &format!("P{i}"), "    ");
        body.push_str(&format!("  if ps == {i} {{
{arm}  }}
"));
    }
    format!(
        "@launch({block})
@autotune(TM in [64], TN in [64])
@aligned(M = TM, N = TN, K = {slice}, KB = {sb})
kernel q8_qmma_split(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[KB, N]{params}) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  let ps = ctx.block.z
{body}}}
"
    )
}

/// Sums [`q8_qmma_split_src`]'s `S` outputs, one row per program.
///
/// A `[TM, TN]` tile add would stage through shared memory, and at
/// `TM = 128` that exceeds the 48 KB static limit. A one-row span needs no
/// mask whatever its offset, so no `@aligned` promise is needed for the
/// row index.
pub(crate) fn q8_qmma_reduce_src(block: usize, s: usize) -> String {
    let mut params = String::new();
    let mut sum = String::new();
    for i in 0..s {
        params.push_str(&format!("P{i}: tensor<f32>[M, N], "));
        if i > 0 {
            sum.push_str(" + ");
        }
        sum.push_str(&format!("P{i}[pm :+ 1, pn * TN :+ TN]"));
    }
    format!(
        "@launch({block})
@autotune(TN in [64])
@aligned(N = TN)
kernel q8_qmma_reduce({params}C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  C[pm :+ 1, pn * TN :+ TN] = {sum}
}}
"
    )
}
