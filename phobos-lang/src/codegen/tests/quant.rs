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
    // dp4a needs four contiguous bytes from each operand, which is why this
    // lands on dot_t: it contracts the last axis of both, so both walk
    // memory contiguously. One vector<4xi8> load per operand per step.
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
    // dp4a arrived with Pascal. Older targets still compile, on the generic
    // integer path.
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
    // With whole 8x8 output tiles the contraction goes to mma.sync instead
    // of dp4a: same four bytes per operand per lane, sixteen times the
    // products per issue. The m8n8k16 fragments are vector<1x4xi8> operands
    // into a vector<1x2xi32> accumulator.
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
    // The tensor core's smallest output tile is 8 rows, so decoding one
    // token would do eight times the arithmetic to keep one row of it.
    let mlir = emit_mlir(I8_DOT_T);
    assert!(
        !mlir.contains("nvgpu.mma.sync"),
        "one row does not fill an m8 tile:\n{mlir}"
    );
}

#[test]
fn i8_dot_t_falls_back_below_turing() {
    // The integer tensor cores arrived with Turing; Pascal and Volta still
    // compile, on dp4a.
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
    // Written as a tile-language loop the block scales have to be applied
    // every 32 elements of k, which puts the accumulator in shared memory
    // and stages both operands there. Folding them into the operation
    // leaves the accumulators as loop-carried f32 values and the operands
    // as plain loads.
    let mlir = emit_mlir(Q8_QMMA);
    assert_contains(
        &mlir,
        &[
            "nvgpu.mma.sync",
            "mmaShape = [8, 8, 16]",
            "vector<1x4xi8>",
            "vector<1x2xi32>",
            // The accumulator becomes an f32 by landing in the mantissa of
            // 1.5 * 2^23, not by a quarter-rate conversion instruction.
            "arith.bitcast",
            "arith.constant 0x4B400000 : f32",
        ],
    );
    assert!(
        !mlir.contains("arith.sitofp"),
        "the block accumulator still converts the slow way in:
{mlir}"
    );
    // Eight tiles a warp, two f32 accumulators each, carried across k.
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
    // The integer tensor core issues 8x8 outputs, so a tile that is not a
    // multiple of eight both ways has no fragment layout to sit in.
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
    // A dot product of bytes overflows i8 almost immediately, so the
    // accumulator widens even when the result type is not written down.
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
    // 30 bytes is not a whole number of four-byte groups; the generic path
    // handles the remainder correctly and dp4a would not.
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
