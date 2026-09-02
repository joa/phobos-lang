// The shared-memory-staged prompt projections.

use super::*;

/// One kernel per format, with the table operands its decode reads.
fn qgemm_src(fmt: &str, tables: &[usize], launch: &str) -> String {
    let params: Vec<String> = tables
        .iter()
        .enumerate()
        .map(|(i, len)| format!("T{i}: tensor<i8>[1, {len}]"))
        .collect();
    let args: Vec<String> = (0..tables.len()).map(|i| format!("T{i}[0 :+ 1, :]")).collect();
    format!(
        "@launch({launch})
        @autotune(TM in [128], TN in [64])
        @aligned(M = TM, N = TN, K = 256)
        kernel {fmt}_qgemm(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                          QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                          {}, C: tensor<f32>[M, N]) {{
            let pm = program_id(0)
            let pn = program_id(1)
            C[pm * TM :+ TM, pn * TN :+ TN] = {fmt}_qgemm_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                         QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                         {})
        }}",
        params.join(", "),
        args.join(", "),
    )
}

const FORMATS: [(&str, &[usize]); 7] = [
    ("iq1s", &[4096]),
    ("iq1m", &[4096]),
    ("iq2xxs", &[2048, 1024]),
    ("iq2xs", &[4096, 1024]),
    ("iq2s", &[8192, 2048]),
    ("iq3xxs", &[1024, 1024]),
    ("iq3s", &[2048, 2048]),
];

#[test]
fn every_format_stages_both_operands_and_reads_them_with_ldmatrix() {
    for (fmt, tables) in FORMATS {
        let mlir = emit_mlir(&qgemm_src(fmt, tables, "256, 2"));
        assert_contains(
            &mlir,
            &["nvgpu.ldmatrix", "vector<4x4xi8>", "nvgpu.mma.sync", "gpu.barrier", "iter_args"],
        );
        // Both operand tiles, the two scale planes, and one tile a table.
        let globals = 4 + tables.len();
        assert_eq!(mlir.matches("memref.global").count(), globals, "{fmt}:
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
