// The staged prompt projections, one kernel per raw format; see
// `phobos-lang`'s `codegen/tile/qgemm.rs`.

use crate::quant::Quant;

/// Rows, columns and threads of a `<fmt>_qgemm` kernel, fixed by the
/// intrinsic.
pub(crate) const QGEMM_TM: usize = 128;
pub(crate) const QGEMM_TN: usize = 64;
pub(crate) const QGEMM_CTA: usize = 256;

/// Length of the ternary grid at two bits per lane, shared by IQ1_S and
/// IQ1_M.
pub(crate) const IQ1_GRID2_LEN: usize = 2048 * 2;

/// Length of the same grid at a nibble per lane, which the decode matvecs
/// read.
pub(crate) const IQ1_GRID4_LEN: usize = 2048 * 4;

/// Lengths of the table operands a format's staged kernel takes after
/// `(qb, d)`: the magnitude grid, then the 0/-1 sign masks if the format
/// stores signs separately. Mirrors `QgFormat::tables` in the compiler.
pub(crate) fn qgemm_tables(quant: Quant) -> Option<&'static [usize]> {
    Some(match quant {
        Quant::IQ1_S | Quant::IQ1_M => &[IQ1_GRID2_LEN],
        Quant::IQ2_XXS => &[256 * 8, 128 * 8],
        Quant::IQ2_XS => &[512 * 8, 128 * 8],
        Quant::IQ2_S => &[1024 * 8, 256 * 8],
        Quant::IQ3_XXS => &[256 * 4, 128 * 8],
        Quant::IQ3_S => &[512 * 4, 256 * 8],
        // The K-quants and PTQ1_0 decode from the block bytes alone.
        Quant::Q4_K | Quant::Q5_K | Quant::Q6_K | Quant::PTQ1_0 => &[],
        _ => return None,
    })
}

/// The kernel and intrinsic a format's staged projection is named by.
pub(crate) fn qgemm_name(quant: Quant) -> Option<&'static str> {
    Some(match quant {
        Quant::IQ1_S => "iq1s",
        Quant::IQ1_M => "iq1m",
        Quant::IQ2_XXS => "iq2xxs",
        Quant::IQ2_XS => "iq2xs",
        Quant::IQ2_S => "iq2s",
        Quant::IQ3_XXS => "iq3xxs",
        Quant::IQ3_S => "iq3s",
        Quant::Q4_K => "q4k",
        Quant::Q5_K => "q5k",
        Quant::Q6_K => "q6k",
        Quant::PTQ1_0 => "ptq1",
        _ => return None,
    })
}

/// The kernel a format's staged source declares.
pub(crate) fn qgemm_kernel(quant: Quant) -> Option<&'static str> {
    Some(match quant {
        Quant::IQ1_S => "iq1s_qgemm",
        Quant::IQ1_M => "iq1m_qgemm",
        Quant::IQ2_XXS => "iq2xxs_qgemm",
        Quant::IQ2_XS => "iq2xs_qgemm",
        Quant::IQ2_S => "iq2s_qgemm",
        Quant::IQ3_XXS => "iq3xxs_qgemm",
        Quant::IQ3_S => "iq3s_qgemm",
        Quant::Q4_K => "q4k_qgemm",
        Quant::Q5_K => "q5k_qgemm",
        Quant::Q6_K => "q6k_qgemm",
        Quant::PTQ1_0 => "ptq1_qgemm",
        _ => return None,
    })
}

/// The kernel a format's split-K source declares; see [`qgemm_split_src`].
pub(crate) fn qgemm_split_kernel(quant: Quant) -> Option<&'static str> {
    Some(match quant {
        Quant::IQ1_S => "iq1s_qgemm_split",
        Quant::IQ1_M => "iq1m_qgemm_split",
        Quant::IQ2_XXS => "iq2xxs_qgemm_split",
        Quant::IQ2_XS => "iq2xs_qgemm_split",
        Quant::IQ2_S => "iq2s_qgemm_split",
        Quant::IQ3_XXS => "iq3xxs_qgemm_split",
        Quant::IQ3_S => "iq3s_qgemm_split",
        Quant::Q4_K => "q4k_qgemm_split",
        Quant::Q5_K => "q5k_qgemm_split",
        Quant::Q6_K => "q6k_qgemm_split",
        Quant::PTQ1_0 => "ptq1_qgemm_split",
        _ => return None,
    })
}

/// The source of a format's staged projection, at two CTAs per
/// multiprocessor (128 registers).
pub(crate) fn qgemm_src(quant: Quant) -> Option<String> {
    let name = qgemm_name(quant)?;
    let tables = qgemm_tables(quant)?;
    // Each piece carries its own comma, so no tables leaves no stray comma.
    let params: String = tables
        .iter()
        .enumerate()
        .map(|(i, len)| format!("T{i}: tensor<i8>[1, {len}], "))
        .collect();
    let args: String = (0..tables.len())
        .map(|i| format!(",\n                                                T{i}[0 :+ 1, :]"))
        .collect();
    Some(format!(
        "@launch({QGEMM_CTA}, 2)
@autotune(TM in [{QGEMM_TM}], TN in [{QGEMM_TN}])
@aligned(M = TM, N = TN, K = 256)
kernel {name}_qgemm(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                  QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                  {params}C: tensor<f32>[M, N]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm * TM :+ TM, pn * TN :+ TN] = {name}_qgemm_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :]{args})
}}
"
    ))
}

/// Most ways a staged projection splits along `k`.
const QGEMM_SPLIT_MAX: usize = 8;

/// Shortest `k` slice a split may leave each program on a grid that still
/// fills up to two waves with it. Below this a program's fixed cost, its
/// first stage and its output tile, outweighs the slots it fills.
const QGEMM_SPLIT_DEEP_K: usize = 3072;

/// Shortest `k` slice when the split grid still fills at most half a wave,
/// where the card is idle enough that shallower programs pay.
const QGEMM_SPLIT_SHALLOW_K: usize = 1280;

/// Splits for a staged projection at `rows x n x k` with blocks of `block`
/// elements on a card with `sms`, where 1 means no split.
///
/// Only a grid under three quarters of a wave splits, at two CTAs per SM:
/// fuller than that, the reduction pass costs what the fill returns, as on a
/// 24-SM card at 40 tiles of 48 slots. The count divides
/// the block count, so every slice is a whole number of blocks, and is the
/// largest whose slices stay deep enough for the fill it reaches; see
/// [`QGEMM_SPLIT_DEEP_K`] and [`QGEMM_SPLIT_SHALLOW_K`].
pub(crate) fn qgemm_splits(rows: usize, n: usize, k: usize, block: usize, sms: usize) -> usize {
    let slots = 2 * sms;
    let tiles = (rows / QGEMM_TM) * (n / QGEMM_TN);
    let nb = k / block;
    if tiles == 0 || 4 * tiles >= 3 * slots {
        return 1;
    }
    (2..=QGEMM_SPLIT_MAX)
        .rev()
        .find(|&s| {
            let (slice, fill) = (k / s, tiles * s);
            nb.is_multiple_of(s)
                && slice.is_multiple_of(256)
                && ((slice >= QGEMM_SPLIT_DEEP_K && fill <= 2 * slots)
                    || (slice >= QGEMM_SPLIT_SHALLOW_K && 2 * fill <= slots))
        })
        .unwrap_or(1)
}

/// The split-K form of [`qgemm_src`]: program `(pm, ps * gn + pn)` contracts
/// slice `ps` of `k` into column tile `ps * gn + pn` of one `[M, S * N]`
/// partial, which [`qgemm_reduce_src`] sums.
///
/// One call rather than one arm per split, since each arm would take its own
/// shared-memory stage. The weight's payload and scale planes are grouped by
/// eight columns with each group's blocks interleaved along its first row,
/// so a slice of blocks starts at eight times its block offset there. Every
/// slice bound is a multiple of its length, which keeps the views unmasked.
pub(crate) fn qgemm_split_src(quant: Quant, n: usize, k: usize, block: usize, s: usize) -> Option<String> {
    let name = qgemm_name(quant)?;
    let tables = qgemm_tables(quant)?;
    let bb = quant.device_block_bytes();
    let gn = n / QGEMM_TN;
    let sb = k / block / s;
    let (sk, skb, srb) = (sb * block, sb * block / 32, sb * bb);
    let params: String = tables
        .iter()
        .enumerate()
        .map(|(i, len)| format!("T{i}: tensor<i8>[1, {len}], "))
        .collect();
    let args: String = (0..tables.len())
        .map(|i| format!(", T{i}[0 :+ 1, :]"))
        .collect();
    let (gk, gb) = (8 * srb, 8 * sb);
    Some(format!(
        "@launch({QGEMM_CTA}, 2)
@autotune(TM in [{QGEMM_TM}], TN in [{QGEMM_TN}])
@aligned(M = TM, N = TN, K = {sk}, KB = {skb}, RB = {srb}, NB = {sb}, NP = TN)
kernel {name}_qgemm_split(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                  QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                  {params}P: tensor<f32>[M, NP]) {{
  let pm = program_id(0)
  let pid = program_id(1)
  let ps = pid / {gn}
  let pn = pid - ps * {gn}
  P[pm * TM :+ TM, pid * TN :+ TN] = {name}_qgemm_t(A[pm * TM :+ TM, ps * {sk} :+ {sk}],
      AS[pm * TM :+ TM, ps * {skb} :+ {skb}], QB[pn * TN :+ TN, ps * {gk} :+ {srb}],
      D[pn * TN :+ TN, ps * {gb} :+ {sb}]{args})
}}
"
    ))
}

/// Sums [`qgemm_split_src`]'s `s` partial column tiles into `C`, one row per
/// program.
pub(crate) fn qgemm_reduce_src(n: usize, s: usize, add: bool) -> String {
    let gn = n / QGEMM_TN;
    // With `add`, the sum lands on what `C` holds, a residual it updates.
    let base = add.then(|| "C[pm :+ 1, pn * TN :+ TN]".to_string());
    let sum: Vec<String> = base
        .into_iter()
        .chain((0..s).map(|i| format!("P[pm :+ 1, (pn + {}) * TN :+ TN]", i * gn)))
        .collect();
    format!(
        "@launch(128)
@autotune(TN in [{QGEMM_TN}])
@aligned(N = TN, NP = TN)
kernel qgemm_reduce(P: tensor<f32>[M, NP], C: tensor<f32>[M, N]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm :+ 1, pn * TN :+ TN] = {}
}}
",
        sum.join(" + ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // The splits that measured fastest on a 128-SM card for the 4B's
    // projections, and unsplit wherever a split measured slower.
    #[test]
    fn splits_only_a_starved_deep_grid() {
        let q4k = |rows, n, k| qgemm_splits(rows, n, k, 256, 128);
        assert_eq!(q4k(128, 2560, 9216), 3);
        assert_eq!(q4k(512, 2560, 9216), 3);
        assert_eq!(q4k(128, 2560, 2560), 2);
        assert_eq!(q4k(512, 2560, 2560), 1);
        assert_eq!(q4k(128, 9216, 2560), 1);
        assert_eq!(q4k(512, 9216, 2560), 1);
        assert_eq!(q4k(1024, 2560, 9216), 1);
    }

    #[test]
    fn a_nearly_full_wave_never_splits() {
        // 20 SMs hold 40 CTAs, which 128 rows by 2560 columns fill; 24 SMs
        // hold 48, which they fill to 83 percent.
        assert_eq!(qgemm_splits(128, 2560, 9216, 256, 20), 1);
        assert_eq!(qgemm_splits(512, 2560, 9216, 256, 20), 1);
        assert_eq!(qgemm_splits(128, 2560, 9216, 256, 24), 1);
    }

    #[test]
    fn a_split_divides_the_blocks() {
        for (rows, n, k) in [(128, 2560, 9216), (128, 2560, 2560), (128, 1024, 4864)] {
            let s = qgemm_splits(rows, n, k, 256, 128);
            assert!((k / 256).is_multiple_of(s), "{rows} x {n} x {k} split {s}");
        }
    }
}
