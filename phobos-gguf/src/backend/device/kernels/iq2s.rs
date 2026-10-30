// IQ2_S's kernels: the f32 decode matvec and weight expansion, written with
// lanes in `iq_f32.rs`, and the fused prompt projection.
//
// The magnitude index is a qs byte widened by two qh bits, and the sign byte
// is looked up directly rather than through IQ2_XXS's parity table (see
// quant/iq2_s.rs).

/// Output columns per CTA.
pub(crate) const IQ2S_TN: usize = 8;

/// IQ2_S's packed grid's length: 1024 entries of eight lanes each.
pub(crate) const IQ2S_GRID_LEN: usize = 1024 * 8;
/// The +-1 half of its packed sign table's length: eight multipliers for
/// every byte value.
pub(crate) const IQ2S_SIGNS_LEN: usize = 256 * 8;

/// [`super::iq1s_qdot_matvec_src`] for IQ2_S.
pub(crate) fn iq2s_qdot_matvec_src(tn: usize) -> String {
    super::qdot_matvec_src(super::IqFmt::Iq2s, tn)
}

/// [`super::iq1s_qdecode_src`] for IQ2_S.
pub(crate) fn iq2s_qdecode_src(tn: usize) -> String {
    super::qdecode_src(super::IqFmt::Iq2s, tn)
}

/// IQ2_S's prompt projection, decode and contraction in one kernel.
///
/// There is one scale per sixteen elements, so the tile language keeps one
/// accumulator per half block; see `qmma_signed.rs`.
pub(crate) fn iq2s_qmma_src(block: usize, tm: usize, tn: usize) -> String {
    format!(
        "@launch({block})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 256)
kernel iq2s_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                  QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                  GRID: tensor<i8>[1, {IQ2S_GRID_LEN}],
                  SIGNS: tensor<i8>[1, {IQ2S_SIGNS_LEN}],
                  C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  C[pm * TM :+ TM, pn * TN :+ TN] = iq2s_qmma_staged_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                        QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                        GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}
