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

/// Threads the `dp4a` decode matvecs launch with: 256 unless
/// `PHOBOS_QDOT_I8_CTA` overrides it. Clamped to between 256 and 1024, and
/// to at most `tn * 32` so each warp owns at least one column.
pub(crate) fn qdot_i8_cta(tn: usize) -> usize {
    static CTA: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let want = CTA
        .get_or_init(|| std::env::var("PHOBOS_QDOT_I8_CTA").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(256);
    want.clamp(256, tn * 32).min(1024)
}

/// Columns a `dp4a` decode matvec's tile covers: `default` unless
/// `PHOBOS_QDOT_I8_TN` overrides it with a power of two in `8..=256`.
///
/// A warp owns `tn * 32 / threads` columns, so sweep this together with
/// [`qdot_i8_cta`].
pub(crate) fn qdot_i8_tn(default: usize) -> usize {
    // Read once: every decode matvec asks, and reading the environment
    // takes a process-wide lock.
    static TN: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    TN.get_or_init(|| {
        std::env::var("PHOBOS_QDOT_I8_TN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&tn| tn.is_power_of_two() && (8..=256).contains(&tn))
    })
    .unwrap_or(default)
}

/// The Q8_0 projection as a single `qmma_t`, used by prompt passes.
///
/// Unlike [`Q8_MMA_SRC`], which applies block scales every 32 elements and
/// so stages through shared memory, `qmma_t` folds the scales in. The whole
/// of `k` is one operation, with accumulators in registers and no barrier
/// in the loop.
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

/// Output tiles of the `qmma_t` projection: the depth, and the available
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

/// The Q8_0 projection as a single `qdot_t`, used by decoding. The tile is
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
  let pn = program_id(0)
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

/// Splits for `q8_qmma`'s deep tile at `rows x n x k`, where 1 means no
/// split. Only a starved grid splits; see [`Q8_QMMA_SPLIT_THRESHOLD`].
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
    // The reduce pass costs about the same at any split count, so only a
    // full split pays for it.
    if splits == Q8_QMMA_SPLIT_MAX {
        splits
    } else {
        1
    }
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
/// one indexed write. `qmma_t` writes straight to global memory only when
/// the slice is provably in bounds from its offset, which an offset built
/// from `program_id(2)` can never be.
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
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm :+ 1, pn * TN :+ TN] = {sum}
}}
"
    )
}
