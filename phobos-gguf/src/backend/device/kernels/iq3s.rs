// IQ3_S's kernels: the f32 decode matvec and weight expansion, written with
// lanes in `iq_f32.rs`, and the fused prompt projection.
//
// Like IQ3_XXS, each lane decodes two four-wide grid entries. Like IQ2_S,
// the sign is a direct byte rather than a parity index (see
// quant/iq3_s.rs), so this reuses IQ2_S's sign table.

use super::IQ2S_SIGNS_LEN;

/// Output columns per CTA.
pub(crate) const IQ3S_TN: usize = 8;

/// [`crate::quant::iq3s_packed_grid`]'s length: 512 grid entries of four
/// lanes each.
pub(crate) const IQ3S_GRID_LEN: usize = 512 * 4;

/// [`super::iq1s_qdot_matvec_src`] for IQ3_S.
pub(crate) fn iq3s_qdot_matvec_src(tn: usize) -> String {
    super::qdot_matvec_src(super::IqFmt::Iq3s, tn)
}

/// [`super::iq1s_qdecode_src`] for IQ3_S.
pub(crate) fn iq3s_qdecode_src(tn: usize) -> String {
    super::qdecode_src(super::IqFmt::Iq3s, tn)
}

/// IQ3_S's prompt projection, decode and contraction in one kernel.
///
/// Like IQ3_XXS, with a wider grid and one scale per sixty-four elements,
/// which is still constant across a k step. See `qmma_signed.rs`.
pub(crate) fn iq3s_qmma_src(block: usize, tm: usize, tn: usize) -> String {
    format!(
        "@launch({block})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 256)
kernel iq3s_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                  QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                  GRID: tensor<i8>[1, {IQ3S_GRID_LEN}],
                  SIGNS: tensor<i8>[1, {IQ2S_SIGNS_LEN}],
                  C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  C[pm * TM :+ TM, pn * TN :+ TN] = iq3s_qmma_staged_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                        QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                        GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}
