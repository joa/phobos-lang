// The shared-memory-staged prompt projections.

use super::*;

/// One kernel per format, with the table operands its decode reads (none
/// for the K-quants).
fn qgemm_src(fmt: &str, tables: &[usize], launch: &str) -> String {
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
fn qdot_i8_src(fmt: &str) -> String {
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

const FORMATS: [(&str, &[usize]); 10] = [
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
];

/// The formats whose runs subtract a minimum, and so stage two planes more.
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
        // Both operand tiles, the two scale planes, one tile a table, and
        // the minimums and row sums where the format subtracts a minimum.
        let globals = 4 + tables.len() + if has_min(fmt) { 2 } else { 0 };
        assert_eq!(mlir.matches("memref.global").count(), globals, "{fmt}:
{mlir}");
    }
}

#[test]
fn a_k_quant_with_a_minimum_sums_the_activation_at_stage_time() {
    // The row sums: a byte dot against ones, one xor shuffle to join the
    // two halves of a group, an i32 plane of a row a stage.
    let with_min = emit_mlir(&qgemm_src("q4k", &[], "256, 2"));
    assert_contains(&with_min, &["nvvm.dot.accumulate.4way", "gpu.shuffle", "memref<128x4xi32, 3>", "nvvm.prmt"]);
    // Q6_K has no minimum: no sums, no shuffle, and its two scales a group
    // take the split epilogue.
    let without = emit_mlir(&qgemm_src("q6k", &[], "256, 2"));
    assert!(!without.contains("gpu.shuffle"), "{without}");
    assert!(without.contains("memref<8x64xf32, 3>"), "{without}");
    assert_contains(&without, &["arith.shrsi"]);
}

#[test]
fn the_k_quant_decode_matvecs_sum_the_row_once_where_there_is_a_minimum() {
    for fmt in ["q4k", "q5k", "q6k"] {
        let mlir = emit_mlir(&qdot_i8_src(fmt));
        assert_contains(&mlir, &["nvvm.dot.accumulate.4way", "nvvm.prmt", "gpu.shuffle", "scf.for"]);
        // The run-sum prologue is a shared i32 tile of KQ_MAX_GROUPS.
        let sums = mlir.contains("memref<1x1024xi32, 3>");
        assert_eq!(sums, fmt != "q6k", "{fmt}:
{mlir}");
    }
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
    // Two scales a group means two mma chains a tile per group: sixteen
    // tiles, four groups, two halves, against one chain for the rest.
    let split = emit_mlir(&qgemm_src("iq2s", &[8192, 2048], "256, 2"));
    let whole = emit_mlir(&qgemm_src("iq2xxs", &[2048, 1024], "256, 2"));
    assert_eq!(split.matches("nvgpu.mma.sync").count(), whole.matches("nvgpu.mma.sync").count());
    // ..and reads two scale rows a group where the other reads one.
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
