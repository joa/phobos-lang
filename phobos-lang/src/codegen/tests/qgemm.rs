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
            let pm = ctx.block.x
            let pn = ctx.block.y
            C[pm * TM :+ TM, pn * TN :+ TN] = {fmt}_qgemm_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                         QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :]{args})
        }}"
    )
}

/// The projection of a format with a minimum, given the activation's
/// (scale, sum) pairs as its last operand.
fn qgemm_sums_src(fmt: &str) -> String {
    format!(
        "@launch(256, 2)
        @autotune(TM in [128], TN in [64])
        @aligned(M = TM, N = TN, K = 256)
        kernel {fmt}_qgemm(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB], AP: tensor<f32>[M, KP],
                          QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB], C: tensor<f32>[M, N]) {{
            let pm = ctx.block.x
            let pn = ctx.block.y
            C[pm * TM :+ TM, pn * TN :+ TN] = {fmt}_qgemm_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                         QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :], AP[pm * TM :+ TM, :])
        }}"
    )
}

const FORMATS: [(&str, &[usize]); 12] = [
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
    ("q80", &[]),
    ("ptq1", &[]),
];

/// Whether the format's runs subtract a minimum, which stages two more
/// planes.
fn has_min(fmt: &str) -> bool {
    matches!(fmt, "q4k" | "q5k")
}

#[test]
fn every_format_stages_both_operands_and_reads_them_with_ldmatrix() {
    // At 64-bit indices, which `Kernel::wants_ldmatrix` asks for.
    for (fmt, tables) in FORMATS {
        let mlir = emit_mlir_sync(&qgemm_src(fmt, tables, "256, 2"), "sm_75");
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
fn a_k_quant_given_its_group_sums_neither_sums_nor_contracts_them() {
    // With the pairs an operand, the stage holds them as they come, eight
    // per row plus four of padding, and no plane of sums: Turing stops
    // summing bytes and shuffling, and from sm_80 the row-sum mma against
    // ones goes, leaving the patch's eight tiles per group.
    for fmt in ["q4k", "q5k"] {
        let turing = emit_mlir_sync(&qgemm_sums_src(fmt), "sm_75");
        assert!(turing.contains("memref<128x12xf32, 3>"), "{turing}");
        for absent in ["nvvm.dot.accumulate.4way", "gpu.shuffle", "memref<128x4xi32, 3>"] {
            assert!(!turing.contains(absent), "{fmt} {absent}:
{turing}");
        }
        let ampere = emit_mlir_sync(&qgemm_sums_src(fmt), "sm_86");
        let summing = emit_mlir_sync(&qgemm_src(fmt, &[], "256, 2"), "sm_86");
        let (given, own) = (ampere.matches("nvgpu.mma.sync").count(), summing.matches("nvgpu.mma.sync").count());
        assert_eq!(given % 8, 0, "{ampere}");
        assert_eq!(own, given / 8 * 10, "{fmt}: {own} mma with its own sums, {given} given them");
        assert!(ampere.contains("memref<128x12xf32, 3>"), "{ampere}");
    }
    // A format without a minimum has no use for them.
    let err = emit_err(&qgemm_sums_src("q6k"));
    assert!(err.contains("4 operands"), "{err}");
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
fn sm_80_on_contracts_with_m16n8k32_and_blackwell_at_one_cta_an_sm() {
    // From sm_80 the int8 contraction takes the wide shape, its fragments
    // loaded whole, for a format with a scale per 32 elements. sm_120 has no native m8n8k16 and holds more of them, so
    // it asks for one CTA an SM; Ada keeps two, and Turing the narrow shape.
    let src = qgemm_src("q4k", &[], "256, 2");
    let turing = emit_mlir_sync(&src, "sm_75");
    let ada = emit_mlir_sync(&src, "sm_89");
    let blackwell = emit_mlir_sync(&src, "sm_120");
    assert!(turing.contains("mmaShape = [8, 8, 16]") && !turing.contains("mmaShape = [16, 8, 32]"), "{turing}");
    for wide in [&ada, &blackwell] {
        assert!(wide.contains("mmaShape = [16, 8, 32]") && !wide.contains("mmaShape = [8, 8, 16]"), "{wide}");
    }
    assert!(ada.contains("nvvm.minctasm = 2"), "{ada}");
    assert!(blackwell.contains("nvvm.minctasm = 1"), "{blackwell}");
    // A scale per 16 elements keeps the native narrow shape until Blackwell.
    let split = qgemm_src("iq2s", &[8192, 2048], "256, 2");
    let ada = emit_mlir_sync(&split, "sm_89");
    let blackwell = emit_mlir_sync(&split, "sm_120");
    assert!(ada.contains("mmaShape = [8, 8, 16]") && !ada.contains("mmaShape = [16, 8, 16]"), "{ada}");
    assert!(blackwell.contains("mmaShape = [16, 8, 16]") && !blackwell.contains("mmaShape = [8, 8, 16]"), "{blackwell}");
}

#[test]
fn m16n8k32_sums_the_activation_groups_itself() {
    // A format with a minimum needs each activation group's sum. Turing
    // stages a plane of them; the wide shape takes them from one more mma
    // per row pair against ones, as many ldmatrix as before.
    let src = qgemm_src("q4k", &[], "256, 2");
    let turing = emit_mlir_sync(&src, "sm_75");
    let ampere = emit_mlir_sync(&src, "sm_86");
    assert!(turing.contains("memref<128x4xi32, 3>"), "{turing}");
    assert!(!ampere.contains("memref<128x4xi32, 3>"), "{ampere}");
    assert_eq!(turing.matches("nvvm.ldmatrix").count(), ampere.matches("nvvm.ldmatrix").count());
    // Per group, eight tiles of the patch and two row sums.
    assert_eq!(ampere.matches("nvgpu.mma.sync").count() % 10, 0, "{ampere}");
}

#[test]
fn sm_80_copies_the_activation_tile_with_cp_async_where_two_fit() {
    for (fmt, tables) in FORMATS {
        let src = qgemm_src(fmt, tables, "256, 2");
        let turing = emit_mlir_sync(&src, "sm_75");
        let ampere = emit_mlir_sync(&src, "sm_80");
        assert!(!turing.contains("nvgpu.device_async_copy"), "{fmt}");
        assert!(shared_bytes(&ampere) <= 48 * 1024, "{fmt}");
        // The two IQ2 formats with the largest tables keep one stage.
        let doubled = !matches!(fmt, "iq2xs" | "iq2s");
        assert_eq!(ampere.matches("nvgpu.device_async_copy").count(), if doubled { 8 } else { 0 }, "{fmt}");
    }
}
