// Integer contractions: dp4a, imma and the Q8 fused dot.

use super::*;

const I8_DOT_T: &str = "@launch(256)
        @autotune(TN in [32])
        @aligned(K = 32, N = TN)
        kernel i8dot(A: tensor<i8>[M, K], W: tensor<i8>[N, K], C: tensor<i32>[M, N]) {
            let pn = program_id(0)
            let a = A[0 :+ 1, 0 :+ 32]
            let w = W[pn * TN :+ TN, 0 :+ 32]
            C[0 :+ 1, pn * TN :+ TN] = dot_t(a, w)
        }";

#[test]
fn i8_dot_t_uses_the_hardware_four_way_dot() {
    // dot_t contracts the last axis of both operands, which matches dp4a's
    // four-byte load.
    let mlir = emit_mlir(I8_DOT_T);
    assert_contains(
        &mlir,
        &[
            "nvvm.dot.accumulate.4way",
            "vector<4xi8>",
            "<signed>",
            "memref<?x?xi8, 1>",
        ],
    );
}

#[test]
fn i8_dot_t_falls_back_below_pascal() {
    // dp4a needs Pascal. Older targets take the generic integer path.
    let mlir = emit_mlir_on(I8_DOT_T, "sm_50");
    assert!(
        !mlir.contains("nvvm.dot.accumulate.4way"),
        "sm_50 has no dp4a:\n{mlir}"
    );
    assert_contains(&mlir, &["arith.muli", "arith.addi"]);
}

const I8_DOT_T_TILED: &str = "@launch(256)
        @autotune(TM in [16], TN in [32])
        @aligned(M = TM, K = 32, N = TN)
        kernel i8mma(A: tensor<i8>[M, K], W: tensor<i8>[N, K], C: tensor<i32>[M, N]) {
            let pm = program_id(0)
            let pn = program_id(1)
            let a = A[pm * TM :+ TM, 0 :+ 32]
            let w = W[pn * TN :+ TN, 0 :+ 32]
            C[pm * TM :+ TM, pn * TN :+ TN] = dot_t(a, w)
        }";

#[test]
fn i8_dot_t_uses_the_integer_tensor_cores() {
    // Whole 8x8 output tiles use mma.sync instead of dp4a. m8n8k16 takes
    // vector<1x4xi8> operands into a vector<1x2xi32> accumulator.
    let mlir = emit_mlir(I8_DOT_T_TILED);
    assert_contains(
        &mlir,
        &[
            "nvgpu.mma.sync",
            "mmaShape = [8, 8, 16]",
            "vector<1x4xi8>",
            "vector<1x2xi32>",
        ],
    );
    assert!(
        !mlir.contains("nvvm.dot.accumulate.4way"),
        "dp4a left on the tensor-core path:\n{mlir}"
    );
}

#[test]
fn a_single_row_i8_dot_t_stays_on_dp4a() {
    // A single row cannot fill the tensor core's 8-row minimum tile.
    let mlir = emit_mlir(I8_DOT_T);
    assert!(
        !mlir.contains("nvgpu.mma.sync"),
        "one row does not fill an m8 tile:\n{mlir}"
    );
}

#[test]
fn i8_dot_t_falls_back_below_turing() {
    // Integer tensor cores need Turing. Pascal and Volta use dp4a.
    let mlir = emit_mlir_on(I8_DOT_T_TILED, "sm_70");
    assert!(
        !mlir.contains("nvgpu.mma.sync"),
        "sm_70 has no integer tensor cores:\n{mlir}"
    );
    assert_contains(&mlir, &["nvvm.dot.accumulate.4way"]);
}

const Q8_QMMA: &str = "            @launch(256)
        @autotune(TM in [64], TN in [64])
        @aligned(M = TM, N = TN, K = 32)
        kernel qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                    W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
                    C: tensor<f32>[M, N]) {
            let pm = program_id(0)
            let pn = program_id(1)
            C[pm * TM :+ TM, pn * TN :+ TN] = qmma_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                     W[pn * TN :+ TN, :], WS[:, pn * TN :+ TN])
        }";

#[test]
fn qmma_t_keeps_its_accumulators_in_registers() {
    // With the block scales folded into qmma_t, the accumulators stay
    // loop-carried f32 values instead of living in shared memory.
    let mlir = emit_mlir(Q8_QMMA);
    assert_contains(
        &mlir,
        &[
            "nvgpu.mma.sync",
            "mmaShape = [8, 8, 16]",
            "vector<1x4xi8>",
            "vector<1x2xi32>",
            // i32 to f32 via the 1.5*2^23 mantissa trick, not a conversion.
            "arith.bitcast",
            "arith.constant 0x4B400000 : f32",
        ],
    );
    assert!(
        !mlir.contains("arith.sitofp"),
        "the block accumulator still converts the slow way in:
{mlir}"
    );
    // Eight tiles per warp, two f32 accumulators each, carried across k.
    assert_contains(&mlir, &["iter_args"]);
    let carried = mlir
        .matches("%cst = arith.constant 0.000000e+00 : f32")
        .count();
    assert!(
        carried > 0,
        "no f32 accumulator initializer in:
{mlir}"
    );
}

#[test]
fn qmma_t_needs_whole_tensor_core_tiles() {
    // 8x8 tensor-core outputs need tile dims that are multiples of 8.
    let src = Q8_QMMA.replace("TN in [64]", "TN in [12]");
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    let module = Module::new(Location::unknown(&context));
    let kernels = crate::parse(&src).unwrap();
    let base = phobos_base::context::Context::default();
    let err = crate::codegen::emit(&base, &kernels, &context, &module).unwrap_err();
    assert!(
        err.to_string().contains("multiple of 8"),
        "unexpected error: {err}"
    );
}

#[test]
fn i8_contraction_accumulates_in_i32() {
    // i8*i8 overflows i8 at once, so the accumulator is always i32.
    let mlir = emit_mlir(
        "@launch(256)
        @aligned(K = 32, N = 32)
        kernel acc(A: tensor<i8>[M, K], W: tensor<i8>[N, K], C: tensor<f32>[M, N]) {
            let a = A[0 :+ 1, 0 :+ 32]
            let w = W[0 :+ 32, 0 :+ 32]
            C[0 :+ 1, 0 :+ 32] = f32(dot_t(a, w))
        }",
    );
    assert_contains(&mlir, &["i32", "arith.sitofp"]);
}

#[test]
fn a_ragged_contraction_stays_off_the_dp4a_path() {
    // 30 is not a multiple of 4, so this stays off dp4a.
    let mlir = emit_mlir(
        "@launch(256)
        @aligned(K = 30, N = 32)
        kernel ragged(A: tensor<i8>[M, K], W: tensor<i8>[N, K], C: tensor<i32>[M, N]) {
            let a = A[0 :+ 1, 0 :+ 30]
            let w = W[0 :+ 32, 0 :+ 30]
            C[0 :+ 1, 0 :+ 32] = dot_t(a, w)
        }",
    );
    assert!(
        !mlir.contains("nvvm.dot.accumulate.4way"),
        "30 is not a multiple of 4:\n{mlir}"
    );
}

#[test]
fn gather_indexes_a_table_per_element() {
    // gather does one lookup per element inside the distributed loop, not a
    // single CTA-uniform load.
    let mlir = emit_mlir(
        "@launch(256)
        kernel gather_test(IDX: tensor<i32>[N, 4], TABLE: tensor<i32>[256], OUT: tensor<i32>[N, 4]) {
            let p = program_id(0)
            var idx = IDX[p * 32 :+ 32, 0 :+ 4]
            var val = gather(TABLE[:], idx)
            OUT[p * 32 :+ 32, 0 :+ 4] = val
        }",
    );
    assert_contains(&mlir, &["arith.index_cast", "scf.for"]);
    let index_casts = mlir.matches("arith.index_cast").count();
    assert_eq!(
        index_casts, 1,
        "expected exactly one index cast, the gather's own:\n{mlir}"
    );
}

#[test]
fn gather_accepts_a_rank_two_table_with_a_leading_one() {
    // Kernel params are always rank-2, so a [1, n] table must work like a
    // rank-1 one. `A[0, :]` is not an option, since point and slice
    // subscripts do not mix.
    let mlir = emit_mlir(
        "@launch(256)
        kernel gather_test(IDX: tensor<i32>[N, 4], TABLE: tensor<i32>[1, 256], OUT: tensor<i32>[N, 4]) {
            let p = program_id(0)
            var idx = IDX[p * 32 :+ 32, 0 :+ 4]
            var val = gather(TABLE[0 :+ 1, :], idx)
            OUT[p * 32 :+ 32, 0 :+ 4] = val
        }",
    );
    assert_contains(&mlir, &["scf.for"]);
}

#[test]
fn gather_rejects_a_non_integer_index() {
    // A float index has no meaning as a table offset.
    let src = "@launch(256)
        kernel gather_test(IDX: tensor<f32>[N, 4], TABLE: tensor<i32>[256], OUT: tensor<i32>[N, 4]) {
            let p = program_id(0)
            var idx = IDX[p * 32 :+ 32, 0 :+ 4]
            var val = gather(TABLE[:], idx)
            OUT[p * 32 :+ 32, 0 :+ 4] = val
        }";
    let err = emit_err(src);
    assert!(err.contains("integer"), "unexpected error: {err}");
}

const IQ1S_QMMA: &str = "            @launch(128)
        @autotune(TM in [128], TN in [64])
        @aligned(M = TM, N = TN, K = 256)
        kernel iq1s_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                         QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                         GRID: tensor<i8>[1, 32768],
                         C: tensor<f32>[M, N]) {
            let pm = program_id(0)
            let pn = program_id(1)
            C[pm * TM :+ TM, pn * TN :+ TN] = iq1s_qmma_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                          QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                          GRID[0 :+ 1, :])
        }";

#[test]
fn iq1s_qmma_t_decodes_into_the_tensor_core_fragments() {
    // The grid entry goes into the same `vector<1x4xi8>` operand a Q8_0
    // weight would load into, so no expanded weight is ever written.
    let mlir = emit_mlir(IQ1S_QMMA);
    assert_contains(
        &mlir,
        &[
            "nvgpu.mma.sync",
            "mmaShape = [8, 8, 16]",
            "vector<1x4xi8>",
            "vector<1x2xi32>",
        ],
    );
    // Accumulators are carried across k, not staged.
    assert_contains(&mlir, &["iter_args"]);
}

#[test]
fn iq1s_qmma_t_writes_no_expanded_weight() {
    // Unlike `_qdecode`, this path never stores the expanded weight.
    let mlir = emit_mlir(IQ1S_QMMA);
    let stores = mlir.matches("memref.store").count();
    // Two accumulator halves per tile of the warp's patch, written once
    // after the loop. Anything more would be an expansion.
    assert!(
        stores <= 64,
        "the fused projection is storing more than its accumulators ({stores}) in:
{mlir}"
    );
}

#[test]
fn iq1s_qmma_t_needs_whole_tensor_core_tiles() {
    // 8x8 tensor-core outputs need tile dims that are multiples of 8. `k` is
    // dynamic, so the host's `raw_qmma_eligible` promises whole 256-blocks.
    let src = IQ1S_QMMA.replace("TN in [64]", "TN in [12]");
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    let module = Module::new(Location::unknown(&context));
    let kernels = crate::parse(&src).unwrap();
    let base = phobos_base::context::Context::default();
    let err = crate::codegen::emit(&base, &kernels, &context, &module).unwrap_err();
    assert!(
        err.to_string().contains("multiple of 8"),
        "unexpected error: {err}"
    );
}

const Q50_QDOT: &str = "@launch(256)
        @autotune(TN in [8])
        @aligned(N = TN)
        kernel q50dot(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                      W: tensor<i8>[N, KW], WS: tensor<f32>[N, KB],
                      C: tensor<f32>[M, N]) {
            let pn = program_id(0)
            C[0 :+ 1, pn * TN :+ TN] = q50_qdot_t(A[0 :+ 1, :], AS[0 :+ 1, :],
                                                  W[pn * TN :+ TN, :], WS[pn * TN :+ TN, :])
        }";

/// The multiplier that spreads four `qh` bits one to a byte, 0x00204081.
const Q50_SPREAD: &str = "arith.constant 2113665 : i32";

#[test]
fn q50_qdot_t_widens_its_weight_before_the_four_way_dot() {
    let mlir = emit_mlir(Q50_QDOT);
    assert_contains(&mlir, &["nvvm.dot.accumulate.4way", Q50_SPREAD, "arith.xori"]);
}

#[test]
fn q50_qmma_t_widens_into_the_tensor_core_fragments() {
    let src = Q8_QMMA
        .replace("qmma_t(", "q50_qmma_t(")
        .replace("W: tensor<i8>[N, K]", "W: tensor<i8>[N, KW]");
    let mlir = emit_mlir(&src);
    assert_contains(&mlir, &["nvgpu.mma.sync", "vector<1x4xi8>", Q50_SPREAD]);
}

#[test]
fn qmma_t_reads_its_weight_as_is() {
    assert!(!emit_mlir(Q8_QMMA).contains(Q50_SPREAD));
}
