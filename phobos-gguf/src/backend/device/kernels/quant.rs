// Q8_0 kernel sources: the activation quantizer, the dp4a and imma
// contractions, the fused dot, and the split-k reduction.

use crate::backend::Q8_BLOCK;

/// Output tile along N for the quantized single-row matmul.
pub(crate) const Q8_TN: usize = 32;

/// Output tile for the quantized tensor-core matmul: the depth of the m8
/// fragment, and an 8x8 tile per warp so no two warps read the same weight
/// byte. Deeper row tiles measured the same and run out of shared memory.
pub(crate) const Q8_MMA_TM: usize = 8;

pub(crate) const Q8_MMA_TN: usize = 64;

/// Blocks quantized per CTA in the activation-quantization kernel.
pub(crate) const QUANT_TB: usize = 4;

/// The same for a prompt, thousands of blocks rather than thirty-two. Four rows
/// leaves half a 256-thread CTA idle; 64 is too deep in shared memory to fit two
/// CTAs.
pub(crate) const QUANT_TB_WIDE: usize = 16;

/// Quantize an activation to int8 with one scale per block of 32, viewed as
/// `[blocks, 32]` so a row is a block and `rowmax` gives its magnitude directly.
///
/// The rounding is the hardware's own, ties to even, matching the host
/// reference exactly. Biasing into `[1.5, 255.5]` and truncating would get
/// round-half-up without a rounding instruction, but costs seven bits of
/// mantissa, enough at the top of the range to cross a boundary.
pub(crate) const QUANTIZE_SRC: &str = "\
@launch(256)
@autotune(TB in [8])
kernel quantize(X: tensor<f32>[R, 32], Q: tensor<i8>[R, 32], S: tensor<f32>[R, D]) {
  let p = program_id(0)
  var x = X[p * TB :+ TB, 0 :+ 32]
  var mx: tile<f32>[TB, 1] = rowmax(tmax(x, -x))
  var inv = 127.0 / (mx + 0.00000001)
  var y = x * inv
  Q[p * TB :+ TB, 0 :+ 32] = i8(i32(round(y)))
  S[p * TB :+ TB, 0 :+ 1] = mx / 127.0
}
";

/// The Q8_0 projection with both operands in int8, contracted by `dp4a`.
/// `dot_t` contracts the last axis of both, so activation and weight row both
/// walk `k` contiguously and four bytes of each pack into one instruction.
/// Nothing is dequantized into shared memory.
pub(crate) const Q8_DP4A_SRC: &str = "\
@launch(256)
@autotune(TN in [32])
{ALIGNED}
kernel q8_dp4a(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
               C: tensor<f32>[M, N]) {
  let pn = program_id(0)
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

/// The batched Q8_0 projection, on the integer tensor cores: [`Q8_DP4A_SRC`]'s
/// contraction over a tile in both directions, which moves a prefill's rows from
/// a host loop of launches into the grid and reaches `mma.sync`, whose smallest
/// integer output tile is 8x8.
pub(crate) const Q8_MMA_SRC: &str = "\
@launch(256)
@autotune(TM in [8], TN in [64])
{ALIGNED}
kernel q8_mma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
              W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
              C: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)
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

/// Threads the `dp4a` decode matvecs carry. **256**, and the wider CTA that
/// looks better standalone is a whole-pass loss.
///
/// The CTA sets occupancy and, because the tile is derived from it, how many
/// columns a warp owns, so it is the one knob that moves these kernels. On
/// IQ1_S at the FFN shape `resident_probe` says wider is better, 256 leaving
/// only 24 of the multiprocessor's 32 warps resident:
///
/// | `@launch` | ms | GB/s | tg128 in the model |
/// | ---: | ---: | ---: | ---: |
/// | **256** | 0.136 | 128.4 | **12.90, 12.90** |
/// | 512 | 0.119 | 145.8 | 12.53, 12.54 |
/// | 1024 | 0.114 | 152.9 | 11.75 |
///
/// 1.19x standalone and 0.91x in the model, the same shape as the larger-CTA
/// result already on file for the float matvecs. A decode step interleaves a
/// thousand of these and a fatter block leaves fewer of them in flight.
/// `PHOBOS_QDOT_I8_CTA` re-measures it; the clamp is not optional, since a
/// warp owns `tn * WARP / threads` columns and the tile has to fill the CTA a
/// whole number of times, so the 16-column narrow tile will not compile at
/// 1024.
pub(crate) fn qdot_i8_cta(tn: usize) -> usize {
    let want = std::env::var("PHOBOS_QDOT_I8_CTA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    want.clamp(256, tn * 32).min(1024)
}

/// Columns a `dp4a` decode matvec's tile covers. **64**, or what
/// `PHOBOS_QDOT_I8_TN` overrides it to, so the tile can be swept against the
/// CTA rather than one at a time: a warp owns `tn * WARP / threads` columns and
/// that product, not either half, is what moves these kernels.
pub(crate) fn qdot_i8_tn(default: usize) -> usize {
    std::env::var("PHOBOS_QDOT_I8_TN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&tn| tn.is_power_of_two() && (8..=256).contains(&tn))
        .unwrap_or(default)
}

/// The Q8_0 projection as a single `qmma_t`, which is what a prompt pass runs.
///
/// [`Q8_MMA_SRC`] applies the block scales every 32 elements of `k`, which forces
/// its accumulator into shared memory and stages both operands there per block:
/// no tile shape moves it off 2.3 TOPS. `qmma_t` folds the scales in, leaving the
/// whole of `k` one operation with the accumulators in registers and no barrier
/// in the loop, 12.6 TOPS on the same shapes.
pub(crate) fn q8_qmma_src(block: usize) -> String {
    format!(
        "@launch({block})
@autotune(TM in [64], TN in [64])
@aligned(M = TM, N = TN, K = 32)
kernel q8_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
               C: tensor<f32>[M, N]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm * TM :+ TM, pn * TN :+ TN] = qmma_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                           W[pn * TN :+ TN, :], WS[:, pn * TN :+ TN])
}}
"
    )
}

/// Output tiles of the `qmma_t` projection: the depth, and the widths it comes
/// in. Operand loads and scale arithmetic are per output element however the
/// tiles are arranged, so what pays for them is the tensor-core tiles in a
/// warp's patch, and the output tile bounds the patch. Wider wins until the
/// register file runs out. The 64-deep kernel takes the rows a prompt leaves
/// over.
pub(crate) const Q8_QMMA_TM: usize = 128;

pub(crate) const Q8_QMMA_SHALLOW: usize = 64;

pub(crate) const Q8_QMMA_WIDTHS: [usize; 2] = [128, 64];

pub(crate) const Q8_QMMA_TN: usize = 64;

/// Threads the projection's CTA carries: half of every other kernel here. A
/// patch is 128 live accumulators whatever the CTA, so the warp is bounded by
/// the register file and a narrower block buys the same warps per
/// multiprocessor on twice the grid. 30.64 TOPS against 24.91 on the widest
/// projection.
pub(crate) const Q8_QMMA_CTA: usize = 128;

/// The widest column tile that divides `n`, regardless of how many blocks that
/// leaves. The warp's patch beats filling the grid even where the widest tile
/// leaves a third of the card idle; keeping the grid full was costing the two
/// 1024-wide projections about a quarter of their throughput. The depth stays
/// 128 for the same reason, 64 measures worse at every width.
pub(crate) fn qmma_width(n: usize) -> usize {
    Q8_QMMA_WIDTHS
        .iter()
        .copied()
        .find(|&tn| n.is_multiple_of(tn))
        .unwrap_or(Q8_QMMA_TN)
}

/// The Q8_0 projection as a single `qdot_t`, which is what decoding runs. The
/// tile is eight outputs, one per warp of the CTA.
///
/// [`Q8_DP4A_SRC`] and [`Q8_SPLIT_SRC`] both stop every 32 elements of `k` to
/// apply the block scales, and `dot_t` gives each output column to one thread:
/// five barriers and 32 scattered sectors per kilobyte of weight. `qdot_t` folds
/// the scales in and turns the mapping around, so a warp owns an output and its
/// lanes divide `k`, putting a warp's reads on 512 contiguous bytes with nothing
/// to stage. Over the seven projections a decode step runs, 2.4x to 7.4x. It
/// wants no k-split, since 32 lanes per output already fill the machine.
pub(crate) const Q8_QDOT_SRC: &str = "\
@launch(256)
@autotune(TN in [8])
@aligned(N = TN)
kernel q8_qdot(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[N, KB],
               C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = qdot_t(A[0 :+ 1, :], AS[0 :+ 1, :],
                                    W[pn * TN :+ TN, :], WS[pn * TN :+ TN, :])
}
";

/// [`Q8_QDOT_SRC`] adding into its destination, for the residual connection at
/// the end of every block.
pub(crate) const Q8_QDOT_ADD_SRC: &str = "@launch(256)
@autotune(TN in [8])
@aligned(N = TN)
kernel q8_qdot_add(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                   W: tensor<i8>[N, K], WS: tensor<f32>[N, KB],
                   C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] += qdot_t(A[0 :+ 1, :], AS[0 :+ 1, :],
                                     W[pn * TN :+ TN, :], WS[pn * TN :+ TN, :])
}
";

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
  let pn = program_id(0)
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
/// [`Q8_DP4A_SRC`] puts the whole of `k` in one program, leaving a grid of
/// `n / TN` blocks and nothing else, which binds at decode: achieved bandwidth
/// tracks the block count and little else, 38 GB/s at 32 blocks against roughly
/// 427 the card sustains. Splitting `k` is the only fix that adds blocks.
///
/// Program `(pn, ps)` takes output tile `pn` and the `k` slice at `ps` and
/// writes its partial sum to its own row of `P`, which `q8_reduce` then sums.
/// The slice comes from the extents rather than being compiled in, so one
/// module serves every shape and split count a model uses.
pub(crate) const Q8_SPLIT_SRC: &str = "\
@launch(256)
@autotune(TN in [32])
{ALIGNED}
kernel q8_split(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
                P: tensor<f32>[S, N]) {
  let pn = program_id(0)
  let ps = program_id(1)
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
  let pn = program_id(0)
  var acc: tile<f32>[1, RT] = 0.0
  for s in range(0, S, 1) {
    acc = acc + P[s :+ 1, pn * RT :+ RT]
  }
  C[0 :+ 1, pn * RT :+ RT] = acc
}
";

/// Output tile for the split-K reduction.
pub(crate) const Q8_REDUCE_TN: usize = 128;

/// Blocks the split aims for, and the most slices it will cut `k` into. The
/// target is a little over four times the card's SM count, where the measured
/// bandwidth curve flattens. The cap keeps a slice from shrinking to a block or
/// two, where the second pass costs more than the first saves.
pub(crate) const Q8_SPLIT_TARGET: usize = 256;

pub(crate) const Q8_SPLIT_MAX: usize = 16;

/// Slices to cut a `[1, k] x [k, n]` projection's contraction into. One means
/// the whole contraction in one program, [`Q8_DP4A_SRC`] with no second pass;
/// wide projections already fill the machine and stay there.
pub(crate) fn q8_splits(n: usize, k: usize) -> usize {
    let grid = n.div_ceil(Q8_TN).max(1);
    let blocks = k / Q8_BLOCK;
    let mut splits = (Q8_SPLIT_TARGET / grid).min(Q8_SPLIT_MAX);

    // The scales change at every Q8_0 block, so a slice has to be whole ones.
    while splits > 1 && !blocks.is_multiple_of(splits) {
        splits /= 2;
    }

    splits.max(1)
}

/// The `q8_qmma` deep tile's grid is `(rows / TM) * (n / TN)`, nothing else:
/// a prompt short enough, or a projection narrow enough, leaves most of the
/// card with no block at all, 12 or 20 blocks against 48 SMs measuring around
/// half the register-bound occupancy ceiling. Below this many blocks the
/// unsplit launch is declined in favor of [`q8_qmma_split_src`]. Above it,
/// splitting only adds a reduction pass to a shape already filling the grid
/// reasonably: 72 blocks reaches 79-80% of its own ceiling, with no occupancy
/// left for more blocks to buy.
pub(crate) const Q8_QMMA_SPLIT_THRESHOLD: usize = 48;

/// Blocks a split aims to reach: two per SM at the register-bound ceiling
/// this card's `q8_qmma` patch has (128 accumulators per patch, 48 SMs).
pub(crate) const Q8_QMMA_SPLIT_TARGET: usize = 96;

/// The widest a split kernel's branch chain gets. Each split is a whole
/// `if ps == i` arm plus its own output operand (see [`q8_qmma_split_src`]),
/// so this also bounds how many extra kernel parameters and compiled
/// variants a shape can cost.
pub(crate) const Q8_QMMA_SPLIT_MAX: usize = 8;

/// Splits for `q8_qmma`'s deep tile at rows x n x k, 1 meaning the unsplit
/// path stays. See [`Q8_QMMA_SPLIT_THRESHOLD`] for why only a starved grid
/// takes this at all.
pub(crate) fn q8_qmma_splits(rows: usize, n: usize, k: usize, wide: usize) -> usize {
    let unsplit = (rows / Q8_QMMA_TM) * (n / wide);
    if unsplit == 0 || unsplit >= Q8_QMMA_SPLIT_THRESHOLD {
        return 1;
    }
    let blocks = k / Q8_BLOCK;
    let mut splits = (Q8_QMMA_SPLIT_TARGET / unsplit)
        .min(Q8_QMMA_SPLIT_MAX)
        .min(blocks);
    while splits > 1 && !blocks.is_multiple_of(splits) {
        splits /= 2;
    }
    // The reduce pass costs roughly the same whatever the split count, so a
    // split short of the max halves the compute-time saving without shrinking
    // what pays for it, and can cost more than it buys. Below the max, the
    // unsplit path stays.
    if splits == Q8_QMMA_SPLIT_MAX {
        splits
    } else {
        1
    }
}

/// Threads the narrow-CTA variant of the deep tile carries, and its column
/// tile: half of [`Q8_QMMA_CTA`] and half of [`Q8_QMMA_WIDTHS`]'s widest
/// entry. The two must be halved *together* or the variant loses its whole
/// point: `qmma_patch` resolves the pair to the same register-bound `(rm=8,
/// rn=8)` per-warp patch as the shipped 128-wide config, so the grid doubles
/// on a starved shape at unchanged tensor-core intensity, while halving `TN`
/// alone at the full CTA collapses the patch to a worse-intensity `(rm=4,
/// rn=8)`. That narrower-tile-alone lever is what [`Q8_QMMA_WIDTHS`]'s 64-wide
/// fallback already is, and it measured a loss.
pub(crate) const Q8_QMMA_NARROW_CTA: usize = 64;

pub(crate) const Q8_QMMA_NARROW_TN: usize = 64;

/// Whether `q8_qmma`'s deep tile at `rows x n`, already resolved to
/// [`qmma_width`]'s widest option, should take the narrow-CTA path instead.
/// Gated on block count the same way [`q8_qmma_splits`] gates split-K: a grid
/// already at this card's SM count has no idle multiprocessor left for more
/// blocks to reach. Declines whenever `wide` is not the widest tile, since a
/// shape on the narrower entry already lost the intensity this path exists to
/// preserve.
///
/// Known limit, and why the caller ships default-off: the threshold is a flat
/// block count, so it does not scale with `rows`. A shape approaching it from
/// below rather than sitting far under it can regress, halving warps per CTA
/// costing more than the extra SM coverage buys back. Measured at `pp512`,
/// where `rows` is 4x `pp128`'s: `n = 1024` still passes the gate and loses.
/// Tighten the gate relative to how starved the grid actually is, or make it
/// row-count-aware, before loosening it or flipping the default.
pub(crate) fn q8_qmma_narrow_eligible(rows: usize, n: usize, wide: usize) -> bool {
    if wide != Q8_QMMA_WIDTHS[0] || !n.is_multiple_of(Q8_QMMA_NARROW_TN) {
        return false;
    }
    let unsplit = (rows / Q8_QMMA_TM) * (n / wide);
    unsplit > 0 && unsplit < Q8_QMMA_SPLIT_THRESHOLD
}

/// The split-K variant of [`q8_qmma_src`], for the shapes [`q8_qmma_splits`]
/// declines to leave at one program per output tile.
///
/// Two things here look like they could be simplified and cannot, both for
/// the same reason: `qmma_t`'s direct-to-global write only fires when a slice
/// is provably in bounds from its offset expression alone, and a dynamic
/// tensor dimension is only ever assumed a multiple of 4 elements whatever
/// `@aligned` claims. So an offset built from `program_id(2)` can never clear
/// that proof. Each split therefore gets its own output operand and its own
/// `if ps == i` arm, keeping the exact `pm * TM, pn * TN` write shape the
/// unsplit kernel already proves; and `k`'s slice bounds are baked in as
/// literals, since a literal's own divisor is itself and `@aligned` can then
/// cover `A`, `AS`, `W` and `WS` too. The arms are mutually exclusive on a
/// CTA-uniform grid coordinate, so this is predication, not divergence.
pub(crate) fn q8_qmma_split_src(block: usize, k: usize, s: usize) -> String {
    let slice = k / s;
    let sb = slice / Q8_BLOCK;
    let mut params = String::new();
    let mut body = String::new();
    for i in 0..s {
        let from = i * slice;
        let fb = from / Q8_BLOCK;
        params.push_str(&format!(", P{i}: tensor<f32>[M, N]"));
        body.push_str(&format!(
            "  if ps == {i} {{
    P{i}[pm * TM :+ TM, pn * TN :+ TN] = qmma_t(A[pm * TM :+ TM, {from} :+ {slice}], AS[pm * TM :+ TM, {fb} :+ {sb}],
                                             W[pn * TN :+ TN, {from} :+ {slice}], WS[{fb} :+ {sb}, pn * TN :+ TN])
  }}
"
        ));
    }
    format!(
        "@launch({block})
@autotune(TM in [64], TN in [64])
@aligned(M = TM, N = TN, K = {slice}, KB = {sb})
kernel q8_qmma_split(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[KB, N]{params}) {{
  let pm = program_id(0)
  let pn = program_id(1)
  let ps = program_id(2)
{body}}}
"
    )
}

/// Sums [`q8_qmma_split_src`]'s `S` output tiles into one, a row at a time
/// rather than a `[TM, TN]` tile at a time: a plain tile add is not `qmma_t`,
/// so it always stages through shared memory, and at `TM = 128` that tile is
/// a 64 KB round trip past the 48 KB static ceiling, which `Module::from_ptx`
/// rejects outright. A one-element span is unmasked whatever produced its
/// offset, so the row index needs no `@aligned` promise, matching
/// [`Q8_SPLIT_SRC`]'s own row-at-a-time reduction.
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
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm :+ 1, pn * TN :+ TN] = {sum}
}}
"
    )
}
