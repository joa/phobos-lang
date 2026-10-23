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
    // With the pairs an operand, the stage holds their scales and the four
    // product words in a twelve-word row, and no plane of sums: Turing
    // stops summing bytes and shuffling, and from sm_80 the row-sum mma
    // against ones goes, leaving the patch's eight wide tiles per group.
    for fmt in ["q4k", "q5k"] {
        let turing = emit_mlir_sync(&qgemm_sums_src(fmt), "sm_75");
        assert!(turing.contains("memref<128x12xf32, 3>"), "{turing}");
        for absent in ["nvvm.dot.accumulate.4way", "gpu.shuffle", "memref<128x4xi32, 3>"] {
            assert!(!turing.contains(absent), "{fmt} {absent}:
{turing}");
        }
        let ampere = emit_mlir_sync(&qgemm_sums_src(fmt), "sm_86");
        let summing = emit_mlir_sync(&qgemm_src(fmt, &[], "256, 2"), "sm_86");
        let wide = |mlir: &str| mlir.matches("mmaShape = [16, 8, 32]").count();
        assert_eq!(wide(&ampere), 32, "{fmt}: {ampere}");
        assert_eq!(wide(&summing), 40, "{fmt}: {summing}");
        assert!(ampere.contains("memref<128x12xf32, 3>"), "{ampere}");
    }
    // A format without a minimum has no use for them.
    let err = emit_err(&qgemm_sums_src("q6k"));
    assert!(err.contains("4 operands"), "{err}");
}

#[test]
fn a_k_quant_given_its_pairs_takes_the_minimum_term_through_an_f16_mma() {
    // Once per 256-element block, on its second stage: eight f16 m16n8k8
    // over the patch's 16 x 8 tiles, A joined from the product words by a
    // byte permute, B from the f16 minimum plane of both halves, and the
    // epilogue down to three ops a group. Without the pairs the minimum
    // plane is one stage of f32 and the epilogue subtracts the term itself.
    for chip in ["sm_75", "sm_86"] {
        let given = emit_mlir_sync(&qgemm_sums_src("q4k"), chip);
        assert_eq!(given.matches("mmaShape = [16, 8, 8]").count(), 8, "{chip}: {given}");
        assert_contains(
            &given,
            &["vector<2x2xf16>", "vector<1x2xf16>", "memref<64x8xf16, 3>", "scf.if", "nvvm.prmt", "arith.truncf"],
        );
        let own = emit_mlir_sync(&qgemm_src("q4k", &[], "256, 2"), chip);
        assert!(!own.contains("mmaShape = [16, 8, 8]"), "{chip}: {own}");
        // The scale plane keeps that shape; the minimum plane no longer has it.
        let f32_planes = |mlir: &str| mlir.matches("memref<4x64xf32, 3>").count();
        assert!(f32_planes(&given) < f32_planes(&own), "{chip}: {given}");
        let fmas = |mlir: &str| mlir.matches("math.fma").count();
        assert!(fmas(&given) < fmas(&own), "{chip}: {} fma given the pairs, {} without", fmas(&given), fmas(&own));
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

/// The Q8_0 projection over the weight's planes, `q8_qgemm_t`.
fn q8_qgemm_src(launch: &str) -> String {
    format!(
        "@launch({launch})
        @autotune(TM in [128], TN in [64])
        @aligned(M = TM, N = TN, K = 256)
        kernel q8_qgemm(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                        W: tensor<i8>[N, K], WS: tensor<f32>[KB, N], C: tensor<f32>[M, N]) {{
            let pm = ctx.block.x
            let pn = ctx.block.y
            C[pm * TM :+ TM, pn * TN :+ TN] = q8_qgemm_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                       W[pn * TN :+ TN, :], WS[:, pn * TN :+ TN])
        }}"
    )
}

#[test]
fn a_q8_weight_held_as_planes_stages_like_a_grouped_one() {
    // The quants are copied as they are and the scale read from its f32
    // plane, so nothing widens an f16. The rest is the family's: ldmatrix
    // on both operands, the wide shape and the double-buffered activation
    // from sm_80.
    let src = q8_qgemm_src("256, 2");
    let turing = emit_mlir_sync(&src, "sm_75");
    assert_contains(&turing, &["nvgpu.ldmatrix", "nvgpu.mma.sync", "mmaShape = [8, 8, 16]", "iter_args"]);
    assert!(!turing.contains("arith.extf"), "{turing}");
    // Both operand tiles and the two scale planes, nothing more.
    assert_eq!(turing.matches("memref.view").count(), 4, "{turing}");
    let ampere = emit_mlir_sync(&src, "sm_80");
    assert!(ampere.contains("mmaShape = [16, 8, 32]"), "{ampere}");
    assert_eq!(ampere.matches("nvgpu.device_async_copy").count(), 8, "{ampere}");
    assert!(shared_bytes(&ampere) <= 48 * 1024);
}

#[test]
fn q8_qgemm_refuses_a_grouped_weight() {
    // The grouped form's f16 block scales are not the planes it reads.
    let err = emit_err(&qgemm_src("q8", &[], "256, 2"));
    assert!(err.contains("weight scales must be f32"), "{err}");
}
