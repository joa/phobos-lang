// IQ2_XXS's kernels: the f32 decode matvec and weight expansion, written
// with lanes in `iq_f32.rs`, and the fused prompt projection.
//
// The magnitude is a byte index into the grid. The sign is a 7-bit index
// into the sign table, taken from the group's four-byte `aux` field (see
// quant/iq2_xxs.rs). The scale is aux's top four bits.

/// Output columns per CTA.
pub(crate) const IQ2XXS_TN: usize = 8;

/// [`crate::quant::iq2xxs_packed_grid`]'s length: 256 grid entries of eight
/// lanes each.
pub(crate) const IQ2XXS_GRID_LEN: usize = 256 * 8;
/// [`crate::quant::iq2xxs_packed_signs`]'s length: eight sign multipliers for
/// each of 128 sign indices.
pub(crate) const IQ2XXS_SIGNS_LEN: usize = 128 * 8;

/// [`super::iq1s_qdot_matvec_src`] for IQ2_XXS.
pub(crate) fn iq2xxs_qdot_matvec_src(tn: usize) -> String {
    super::qdot_matvec_src(super::IqFmt::Iq2xxs, tn)
}

/// [`super::iq1s_qdecode_src`] for IQ2_XXS.
pub(crate) fn iq2xxs_qdecode_src(tn: usize) -> String {
    super::qdecode_src(super::IqFmt::Iq2xxs, tn)
}

/// IQ2_XXS's prompt projection, decode and contraction in one kernel, like
/// [`super::iq1s::iq1s_qmma_src`].
pub(crate) fn iq2xxs_qmma_src(block: usize, tm: usize, tn: usize) -> String {
    format!(
        "@launch({block})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 256)
kernel iq2xxs_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                   QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                   GRID: tensor<i8>[1, {IQ2XXS_GRID_LEN}],
                   SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}],
                   C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  C[pm * TM :+ TM, pn * TN :+ TN] = iq2xxs_qmma_staged_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                         QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                         GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}
