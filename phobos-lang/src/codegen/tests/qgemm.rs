// The shared-memory-staged prompt projections.

use super::*;

/// The prompt projection kernel for one format, with the table operands its
/// decode reads. The K-quants have none.
pub(super) fn qgemm_src(fmt: &str, tables: &[usize], launch: &str) -> String {
    let params: String = tables
        .iter()
        .enumerate()
        .map(|(i, len)| format!("T{i}: tensor<i8>[1, {len}], "))
        .collect();
    let args: String = (0..tables.len()).map(|i| format!(", T{i}[0 :+ 1, :]")).collect();
    format!(
        "@launch({launch})
        @autotune(TM in [128], TN in [64])
        @aligned(M = TM, N = TN, K = 256)
        kernel {fmt}_qgemm(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                          QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                          {params}C: tensor<f32>[M, N]) {{
            let pm = program_id(0)
            let pn = program_id(1)
            C[pm * TM :+ TM, pn * TN :+ TN] = {fmt}_qgemm_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                         QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :]{args})
        }}"
    )
}

/// The decode matvec of a format with no tables.
pub(super) fn qdot_i8_src(fmt: &str) -> String {
    format!(
        "@launch(256, 4)
        @autotune(TN in [64])
        @aligned(N = TN)
        kernel {fmt}_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                                    QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                                    C: tensor<f32>[M, N]) {{
            let pn = program_id(0)
            C[0 :+ 1, pn * TN :+ TN] = {fmt}_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                                     QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :])
        }}"
    )
}

const FORMATS: [(&str, &[usize]); 11] = [
    ("iq1s", &[4096]),
    ("iq1m", &[4096]),
    ("iq2xxs", &[2048, 1024]),
    ("iq2xs", &[4096, 1024]),
    ("iq2s", &[8192, 2048]),
    ("iq3xxs", &[1024, 1024]),
    ("iq3s", &[2048, 2048]),
    ("q4k", &[]),
    ("q5k", &[]),
    ("q6k", &[]),
    ("ptq1", &[]),
];

/// Whether the format's runs subtract a minimum, which stages two more
/// planes.
fn has_min(fmt: &str) -> bool {
    matches!(fmt, "q4k" | "q5k")
}

#[test]
fn every_format_stages_both_operands_and_reads_them_with_ldmatrix() {
    for (fmt, tables) in FORMATS {
        let mlir = emit_mlir(&qgemm_src(fmt, tables, "256, 2"));
        assert_contains(
            &mlir,
            &["nvgpu.ldmatrix", "vector<4x4xi8>", "nvgpu.mma.sync", "gpu.barrier", "iter_args"],
        );
        // Both operand tiles, the two scale planes, one tile per table, and
        // the minimums and row sums for a format with a minimum.
        let buffers = 4 + tables.len() + if has_min(fmt) { 2 } else { 0 };
        assert_eq!(mlir.matches("memref.view").count(), buffers, "{fmt}:
{mlir}");
    }
}

#[test]
fn a_k_quant_with_a_minimum_sums_the_activation_at_stage_time() {
    // The row sums: a byte dot against ones, one xor shuffle to join the
    // two halves of a group, and an i32 plane with one row per stage.
    let with_min = emit_mlir(&qgemm_src("q4k", &[], "256, 2"));
    assert_contains(&with_min, &["nvvm.dot.accumulate.4way", "gpu.shuffle", "memref<128x4xi32, 3>", "nvvm.prmt"]);
    // Q6_K has no minimum, so no sums and no shuffle. Its two scales per
    // group take the split epilogue.
    let without = emit_mlir(&qgemm_src("q6k", &[], "256, 2"));
    assert!(!without.contains("gpu.shuffle"), "{without}");
    assert!(without.contains("memref<8x64xf32, 3>"), "{without}");
    assert_contains(&without, &["arith.shrsi"]);
}

#[test]
fn the_k_quant_decode_matvecs_sum_each_run_in_its_own_lane() {
    let dots = |fmt: &str| {
        let mlir = emit_mlir(&qdot_i8_src(fmt));
        assert_contains(&mlir, &["nvvm.dot.accumulate.4way", "nvvm.prmt", "gpu.shuffle", "scf.for"]);
        // No prologue, so no shared i32 run-sum tile.
        assert!(!mlir.contains("xi32, 3>"), "{fmt}:
{mlir}");
        mlir.matches("nvvm.dot.accumulate.4way").count()
    };
    // A format with a minimum also dots the activation against ones, so it
    // has more dots than Q6_K.
    let (q4k, q5k, q6k) = (dots("q4k"), dots("q5k"), dots("q6k"));
    assert!(q4k > q6k && q5k > q6k, "q4k {q4k}, q5k {q5k}, q6k {q6k}");
}

#[test]
fn the_ternary_formats_decode_with_a_byte_permute_and_the_rest_with_masks() {
    let ternary = emit_mlir(&qgemm_src("iq1m", &[4096], "256, 2"));
    assert_contains(&ternary, &["nvvm.prmt"]);
    let masked = emit_mlir(&qgemm_src("iq2xxs", &[2048, 1024], "256, 2"));
    assert!(!masked.contains("nvvm.prmt"), "{masked}");
    // The negate: xor with the mask, then add the mask's low bits back.
    assert_contains(&masked, &["arith.xori", "arith.addi"]);
}

#[test]
fn a_split_scale_format_contracts_each_half_on_its_own() {
    // Two scales per group means each group contracts in two halves, one
    // per scale. The mma.sync count stays the same as a one-scale format.
    let split = emit_mlir(&qgemm_src("iq2s", &[8192, 2048], "256, 2"));
    let whole = emit_mlir(&qgemm_src("iq2xxs", &[2048, 1024], "256, 2"));
    assert_eq!(split.matches("nvgpu.mma.sync").count(), whole.matches("nvgpu.mma.sync").count());
    // It reads two scale rows per group where the other reads one.
    assert!(split.contains("memref<8x64xf32, 3>"), "{split}");
    assert!(whole.contains("memref<4x64xf32, 3>"), "{whole}");
}

#[test]
fn qgemm_refuses_a_cta_that_is_not_eight_warps() {
    let err = emit_err(&qgemm_src("iq1s", &[4096], "128"));
    assert!(err.contains("256 threads"), "{err}");
}

#[test]
fn qgemm_refuses_the_wrong_table() {
    let err = emit_err(&qgemm_src("iq2xxs", &[2048, 512], "256, 2"));
    assert!(err.contains("tables of"), "{err}");
}

#[test]
fn blackwell_contracts_with_m16n8k32_at_one_cta_an_sm() {
    // sm_120 has no native m8n8k16 and runs it as a half-used m16n8k16, so
    // the int8 contraction takes the wide shape there, and the fragments it
    // holds ask for one CTA an SM. Ada keeps the narrow shape and two.
    let src = qgemm_src("q4k", &[], "256, 2");
    let ada = emit_mlir_sync(&src, "sm_89");
    let blackwell = emit_mlir_sync(&src, "sm_120");
    assert!(ada.contains("mmaShape = [8, 8, 16]") && !ada.contains("mmaShape = [16, 8, 32]"), "{ada}");
    assert!(blackwell.contains("mmaShape = [16, 8, 32]") && !blackwell.contains("mmaShape = [8, 8, 16]"), "{blackwell}");
    assert!(ada.contains("nvvm.minctasm = 2"), "{ada}");
    assert!(blackwell.contains("nvvm.minctasm = 1"), "{blackwell}");
}
