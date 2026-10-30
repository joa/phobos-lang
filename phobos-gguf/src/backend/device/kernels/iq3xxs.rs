// IQ3_XXS's kernels: the f32 decode matvec and weight expansion, written
// with lanes in `iq_f32.rs`, and the fused prompt projection.
//
// The scale-and-sign field has IQ2_XXS's `aux32` layout (see
// quant/iq3_xxs.rs), so this reuses IQ2_XXS's sign table. Grid entries are
// four bytes rather than eight, so a lane joins two of them.

use super::IQ2XXS_SIGNS_LEN;

/// Output columns per CTA.
pub(crate) const IQ3XXS_TN: usize = 8;

/// IQ3_XXS's packed grid's length: 256 entries of four lanes each.
pub(crate) const IQ3XXS_GRID_LEN: usize = 256 * 4;

/// [`super::iq1s_qdot_matvec_src`] for IQ3_XXS.
pub(crate) fn iq3xxs_qdot_matvec_src(tn: usize) -> String {
    super::qdot_matvec_src(super::IqFmt::Iq3xxs, tn)
}

/// [`super::iq1s_qdecode_src`] for IQ3_XXS.
pub(crate) fn iq3xxs_qdecode_src(tn: usize) -> String {
    super::qdecode_src(super::IqFmt::Iq3xxs, tn)
}

/// IQ3_XXS's prompt projection, decode and contraction in one kernel.
///
/// A grid entry is four bytes here, so a lane joins two of them. The scale
/// and sign layout is the same as IQ2_XXS's. See `qmma_signed.rs`.
pub(crate) fn iq3xxs_qmma_src(block: usize, tm: usize, tn: usize) -> String {
    format!(
        "@launch({block})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 256)
kernel iq3xxs_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                  QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                  GRID: tensor<i8>[1, {IQ3XXS_GRID_LEN}],
                  SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}],
                  C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  C[pm * TM :+ TM, pn * TN :+ TN] = iq3xxs_qmma_staged_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                        QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                        GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}
