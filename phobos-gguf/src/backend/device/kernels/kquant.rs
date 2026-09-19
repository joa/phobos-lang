// The table-free decode matvecs: Q4_K, Q5_K, Q6_K and PTQ1_0, each one
// `<fmt>_qdot_i8_t` call, no table operands. The decode is the compiler's
// (`phobos-lang`'s `codegen/tile/kquant_qdot.rs`); the source only names the
// kernel and the tile, as the IQ formats' do. Q4_K and Q5_K read their `d`
// and `dmin` out of the block header, so the `D` plane is an operand the
// intrinsic takes and never reads; Q6_K's `d` is trailing on disk and lives in
// the plane alone (see `Quant::device_block`). PTQ1_0 keeps both of a block
// pair's scales in the re-laid block itself (`quant/ptq1_0.rs`).

use super::quant::qdot_i8_cta;
use phobos_kernels::launch::WARP_THREADS;

/// Output tile for the dp4a matvecs: a warp owns eight columns, so 64 gives
/// a 256-thread CTA one warp-turn a block. PTQ1_0 halves both, the same
/// eight columns a warp in twice the programs: its 27B's narrow projections
/// otherwise launch fewer CTAs than the card holds.
pub(crate) const Q4K_I8_TN: usize = 64;
pub(crate) const Q5K_I8_TN: usize = 64;
pub(crate) const Q6K_I8_TN: usize = 64;
pub(crate) const PTQ1_I8_TN: usize = 32;

/// The tile for an `n` that 64 does not divide; a ragged `n` past that is
/// padded into a scratch by `project_raw`.
pub(crate) const Q4K_I8_NARROW_TN: usize = 16;
pub(crate) const Q5K_I8_NARROW_TN: usize = 16;
pub(crate) const Q6K_I8_NARROW_TN: usize = 16;
pub(crate) const PTQ1_I8_NARROW_TN: usize = 16;

/// `resident` CTAs of the tile's threads a multiprocessor sets the register
/// budget: Q4_K fits four (64 registers); Q5_K and Q6_K, whose pipeline
/// holds the whole `qh` plane or three planes and the scales, spill there
/// and take three (80), as IQ3_S does.
fn kquant_qdot_i8_matvec_src(name: &str, tn: usize, resident: usize) -> String {
    kquant_qdot_i8_matvec_cta_src(name, tn, qdot_i8_cta(tn), resident)
}

/// [`kquant_qdot_i8_matvec_src`] at a CTA of `cta` threads.
fn kquant_qdot_i8_matvec_cta_src(name: &str, tn: usize, cta: usize, resident: usize) -> String {
    let min_blocks = (1024 / cta * resident / 4).max(1);
    format!(
        "@launch({cta}, {min_blocks})
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel {name}_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                          QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                          C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = {name}_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                           QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :])
}}
"
    )
}

pub(crate) fn q4k_qdot_i8_matvec_src(tn: usize) -> String {
    kquant_qdot_i8_matvec_src("q4k", tn, 4)
}

pub(crate) fn q5k_qdot_i8_matvec_src(tn: usize) -> String {
    kquant_qdot_i8_matvec_src("q5k", tn, 3)
}

pub(crate) fn q6k_qdot_i8_matvec_src(tn: usize) -> String {
    kquant_qdot_i8_matvec_src("q6k", tn, 3)
}

/// Eight columns a warp at any tile, as [`PTQ1_I8_TN`] explains.
pub(crate) fn ptq1_qdot_i8_matvec_src(tn: usize) -> String {
    kquant_qdot_i8_matvec_cta_src("ptq1", tn, (tn * 4).max(WARP_THREADS), 4)
}

