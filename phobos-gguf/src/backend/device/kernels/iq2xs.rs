// IQ2_XS's kernels: the f32 decode matvec and weight expansion, written with
// lanes in `iq_f32.rs`, and the fused prompt projection.
//
// Each lane's 16-bit `qs` halfword holds a 9-bit grid index in the low bits
// and 7 bits of sign parity in the high bits. Each half of a 32-element
// group has its own scale (see quant/iq2_xs.rs). The grid is IQ2_XS's own,
// but the signs work as in IQ2_XXS, so this reuses its sign table.

use super::IQ2XXS_SIGNS_LEN;

/// Output columns per CTA.
pub(crate) const IQ2XS_TN: usize = 8;

/// [`crate::quant::iq2xs_packed_grid`]'s length: 512 grid entries of eight
/// lanes each.
pub(crate) const IQ2XS_GRID_LEN: usize = 512 * 8;

/// [`super::iq1s_qdot_matvec_src`] for IQ2_XS.
pub(crate) fn iq2xs_qdot_matvec_src(tn: usize) -> String {
    super::qdot_matvec_src(super::IqFmt::Iq2xs, tn)
}

/// [`super::iq1s_qdecode_src`] for IQ2_XS.
pub(crate) fn iq2xs_qdecode_src(tn: usize) -> String {
    super::qdecode_src(super::IqFmt::Iq2xs, tn)
}

/// IQ2_XS's prompt projection, decode and contraction in one kernel.
///
/// There is one scale per sixteen elements, so the tile language keeps one
/// accumulator per half block; see `qmma_signed.rs`.
pub(crate) fn iq2xs_qmma_src(block: usize, tm: usize, tn: usize) -> String {
    format!(
        "@launch({block})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 256)
kernel iq2xs_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                  QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                  GRID: tensor<i8>[1, {IQ2XS_GRID_LEN}],
                  SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}],
                  C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  C[pm * TM :+ TM, pn * TN :+ TN] = iq2xs_qmma_staged_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                        QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                        GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}
