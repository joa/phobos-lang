// IQ1_M's kernels: the f32 decode matvec and weight expansion, written with
// lanes in `iq_f32.rs`. IQ1_M shares IQ1_S's grid, but each group of 32
// carries its own pair of 3-bit scales, and the block has no `f16` of its
// own. `quant/iq1_m.rs::raw_scales` rebuilds the equivalent word on the
// host, so `d` is still a plain per-block read.

/// Output columns per CTA.
pub(crate) const IQ1M_TN: usize = 8;

/// [`super::iq1s_qdot_matvec_src`] for IQ1_M.
pub(crate) fn iq1m_qdot_matvec_src(tn: usize) -> String {
    super::qdot_matvec_src(super::IqFmt::Iq1m, tn)
}

/// [`super::iq1s_qdecode_src`] for IQ1_M.
pub(crate) fn iq1m_qdecode_src(tn: usize) -> String {
    super::qdecode_src(super::IqFmt::Iq1m, tn)
}
