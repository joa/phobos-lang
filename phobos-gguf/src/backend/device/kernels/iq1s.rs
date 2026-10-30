// IQ1_S's kernels: the f32 decode matvec and weight expansion, written with
// lanes in `iq_f32.rs`, and the fused prompt projection.

/// Output columns per CTA.
pub(crate) const IQ1S_TN: usize = 8;

/// [`crate::quant::iq1s_packed_grid`]'s length: 2048 grid entries of eight
/// lanes each.
pub(crate) const IQ1S_GRID_LEN: usize = 2048 * 8;

/// The f32 decode matvec for `m == 1`; see `iq_f32.rs`. It needs `N` to be a
/// multiple of `tn`, and [`crate::backend::device::DeviceBackend::project_raw`]
/// expands a ragged width densely instead.
pub(crate) fn iq1s_qdot_matvec_src(tn: usize) -> String {
    super::qdot_matvec_src(super::IqFmt::Iq1s, tn)
}

/// The decode stored into a `[K, N]` scratch instead of reduced, so a matmul
/// over many rows decodes only once; see `iq_f32.rs`.
/// [`crate::backend::device::DeviceBackend::project_raw_dense`] rounds a strip
/// up to whole tiles, which the grouped upload pads.
pub(crate) fn iq1s_qdecode_src(tn: usize) -> String {
    super::qdecode_src(super::IqFmt::Iq1s, tn)
}

/// The fused projection's tile and CTA width, as `(TM, TN, CTA)`.
///
/// `PHOBOS_QMMA_TILE=TMxTNxCTA` overrides all three, to sweep the shape
/// without a rebuild. The tile must divide the batch, and the staged form
/// needs exactly one patch per warp.
pub(crate) fn qmma_tile() -> (usize, usize, usize) {
    let Ok(spec) = std::env::var("PHOBOS_QMMA_TILE") else {
        return (IQ1S_QMMA_TM, IQ1S_QMMA_TN, IQ1S_QMMA_CTA);
    };
    let mut parts = spec.split('x').map(str::parse::<usize>);
    match (parts.next(), parts.next(), parts.next()) {
        (Some(Ok(tm)), Some(Ok(tn)), Some(Ok(cta))) => (tm, tn, cta),
        _ => (IQ1S_QMMA_TM, IQ1S_QMMA_TN, IQ1S_QMMA_CTA),
    }
}

/// Rows and columns of the fused prompt projection's output tile, the same
/// as the Q8_0 projection's. The output tile bounds a warp's patch of
/// tensor-core tiles, which is what amortizes operand loads and scales.
pub(crate) const IQ1S_QMMA_TM: usize = 128;
pub(crate) const IQ1S_QMMA_TN: usize = 64;

/// Threads the fused projection's CTA carries.
///
/// Narrower than `q8_qmma`'s, so each warp gets a larger patch. With `rm`
/// row-tiles in a patch, one decoded weight fragment feeds `rm` tensor
/// instructions. `qmma_patch` never grows a patch so far that warps sit
/// idle, so two warps allow `rm = 8` where four allow only 4.
pub(crate) const IQ1S_QMMA_CTA: usize = 64;

/// Twice [`IQ1S_GRID_LEN`]. The signed table holds both sign foldings of
/// every entry, so the group's sign bit is an index bit rather than
/// arithmetic. See `crate::quant::iq1s_signed_grid`.
pub(crate) const IQ1S_SIGNED_GRID_LEN: usize = 2 * IQ1S_GRID_LEN;

/// IQ1_S's prompt projection, decode and contraction in one kernel. It
/// avoids writing an expanded weight to scratch for `matmul_tc` to read.
///
/// `@aligned` is required, as for `iq1s_qdot_matvec`, because
/// `iq1s_qmma_t` assumes in-bounds slices. `K = 256` because a lane
/// indexes the block bytes itself.
pub(crate) fn iq1s_qmma_src(block: usize, tm: usize, tn: usize) -> String {
    // The staged form decodes each column once per CTA instead of once per
    // warp. The choice is made here so the two forms differ in source text,
    // which the kernel cache keys on.
    //
    // The staged form needs exactly one patch per warp, and the intrinsic
    // rejects a tile that does not give it. `PHOBOS_QMMA_STAGE=0` selects
    // the register form.
    let intrinsic = match phobos_base::env::flag_on("PHOBOS_QMMA_STAGE") {
        true => "iq1s_qmma_staged_t",
        false => "iq1s_qmma_t",
    };
    format!(
        "@launch({block})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 256)
kernel iq1s_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                 QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                 GRID: tensor<i8>[1, {IQ1S_SIGNED_GRID_LEN}],
                 C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  C[pm * TM :+ TM, pn * TN :+ TN] = {intrinsic}(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                GRID[0 :+ 1, :])
}}
"
    )
}

