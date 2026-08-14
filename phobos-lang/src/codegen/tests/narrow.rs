// f16 and bf16 operands, and where they widen.

use super::*;

#[test]
fn narrow_types_convert_on_load_and_store() {
    // An i8 tensor sign-extends into f32 and a bf16 result rounds back
    // down: the two halves of a dequantizing weight load.
    let mlir = emit_mlir(
        "@launch(256)
        kernel narrow(W: tensor<i8>[M, N], S: tensor<f32>[M, N], C: tensor<bf16>[M, N]) {
            let p = program_id(0)
            let w = W[0 :+ 32, p * 32 :+ 32]
            let s = S[0 :+ 32, p * 32 :+ 32]
            var out: tile<f32>[32, 32] = f32(w) * s
            C[0 :+ 32, p * 32 :+ 32] = bf16(out)
        }",
    );
    assert_contains(
        &mlir,
        &["memref<?x?xi8, 1>", "arith.sitofp", "arith.truncf", "bf16"],
    );
}

#[test]
fn f16_and_bf16_operands_meet_at_f32() {
    // Neither 16-bit float contains the other, so mixing them widens to
    // f32 rather than picking a side. Two extf, no truncf.
    let mlir = emit_mlir(
        "@launch(256)
        kernel meet(A: tensor<f16>[M, N], B: tensor<bf16>[M, N], C: tensor<f32>[M, N]) {
            let p = program_id(0)
            let a = A[0 :+ 32, p * 32 :+ 32]
            let b = B[0 :+ 32, p * 32 :+ 32]
            C[0 :+ 32, p * 32 :+ 32] = a + b
        }",
    );
    assert_contains(&mlir, &["arith.extf", "f16 to f32", "bf16 to f32"]);
    assert!(
        !mlir.contains("arith.truncf"),
        "the join is f32, so nothing should round down:\n{mlir}"
    );
}

#[test]
fn bf16_is_emulated_below_ampere_and_native_from_ampere() {
    // The type is available on every target; only the instruction count
    // changes. sm_75 has no bf16 unit, so NVPTX emits the shift and
    // round-to-nearest-even sequence, while sm_80 has cvt.rn.bf16.f32.
    let src = "@launch(256)
        kernel round(A: tensor<f32>[M, N], C: tensor<bf16>[M, N]) {
            let p = program_id(0)
            var a = A[0 :+ 32, p * 32 :+ 32]
            C[0 :+ 32, p * 32 :+ 32] = a * 2.0
        }";
    for chip in ["sm_75", "sm_80"] {
        assert_contains(&emit_mlir_on(src, chip), &["arith.truncf", "bf16"]);
    }
    // Capability, not availability: sm_75 has no native bf16 arithmetic and
    // still compiles this, because the conversion is all the source asks for.
}

#[test]
fn converting_to_a_non_numeric_type_is_an_error() {
    let err = emit_err(
        "kernel bad(A: tensor<f32>[M, N], C: tensor<f32>[M, N]) {
            let a = A[0 :+ 32, 0 :+ 32]
            C[0 :+ 32, 0 :+ 32] = bool(a)
        }",
    );
    assert!(
        err.contains("unknown function 'bool'"),
        "unexpected error: {err}"
    );
}

#[test]
fn f16_tensors_lower_to_f16_memrefs() {
    // f16 tensor params and tile buffers carry the f16 element type, and
    // the 0.0 f32 literal seed is rounded down on store.
    let mlir = emit_mlir(
        "@autotune(T in [8])
        kernel k(A: tensor<f16>[N]) {
            var acc: tile<f16>[T] = 0.0
            let a = A[0 :+ T]
            acc += a
        }",
    );
    assert_contains(
        &mlir,
        &[
            "memref<?xf16, 1>", // the f16 tensor param
            "memref<8xf16, 3>", // the f16 shared tile buffer
            "arith.truncf",     // the f32 literal rounded to f16
            "arith.addf",       // the elementwise f16 accumulate
        ],
    );
    // f16 rows aren't 16B-aligned under the multiple-of-4 ABI, so the
    // elementwise op stays scalar (no 128-bit f32 vectors).
    assert!(
        !mlir.contains("vector<4xf32>"),
        "unexpected f32 vectorization of an f16 tile:\n{mlir}"
    );
}

#[test]
fn f16_scalar_arithmetic_widens_to_f32() {
    // Mixing an f16 operand with an f32 one widens to f32 (arith.extf).
    let mlir = emit_mlir(
        "kernel k(out: tensor<f32>[N], a: f16, b: f32) {
            out[0] = a + b
        }",
    );
    assert_contains(&mlir, &["arith.extf", "arith.addf"]);
}

#[test]
fn f16_matmul_runs_on_tensor_cores() {
    // f16 inputs and output, f32 accumulation: the operands stage into
    // the WMMA fragments verbatim (no rounding), and the f32 result is
    // rounded back to f16 in the epilogue.
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @tensorcore
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
            let pm = program_id(0)
            let pn = program_id(1)
            var acc: tile<f32>[TILE_M, TILE_N] = 0.0
            for kt in range(0, K, TILE_K) {
                let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
        }",
    );
    assert_contains(
        &mlir,
        &[
            // f16 operand tensors and f16 shared staging (a m-major),
            // inner dims bank-conflict padded by 8 (16 -> 24, 64 -> 72)
            "memref<?x?xf16, 1>",
            "memref<64x24xf16, 3>",
            "memref<16x72xf16, 3>",
            "gpu.subgroup_mma_compute",
            "!gpu.mma_matrix<16x16xf32, \"COp\">",
            // the f32 accumulator is rounded to the f16 output
            "arith.truncf",
        ],
    );
    assert!(
        !mlir.contains("vector.contract"),
        "vector MACs left on the f16 tensor-core path:\n{mlir}"
    );
}

#[test]
fn f16_matmul_accumulates_in_f16_on_tensor_cores() {
    // An f16 accumulator runs the WMMA in the m16n16k16 f16.f16 mode:
    // f16 COp fragments and an f16 drain slab, no f32 anywhere in the MAC.
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @tensorcore
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
            let pm = program_id(0)
            let pn = program_id(1)
            var acc: tile<f16>[TILE_M, TILE_N] = 0.0
            for kt in range(0, K, TILE_K) {
                let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
        }",
    );
    assert_contains(
        &mlir,
        &[
            "gpu.subgroup_mma_compute",
            "!gpu.mma_matrix<16x16xf16, \"COp\">", // f16 accumulator fragments
            "memref<128x16xf16, 3>",               // the f16 drain slab
        ],
    );
    // No f32 accumulator fragments or slab on the f16-accumulate path.
    assert!(
        !mlir.contains("\"COp\">") || !mlir.contains("mma_matrix<16x16xf32, \"COp\">"),
        "unexpected f32 accumulator fragment in f16-accumulate matmul:\n{mlir}"
    );
    assert!(
        !mlir.contains("memref<128x16xf32, 3>"),
        "unexpected f32 drain slab in f16-accumulate matmul:\n{mlir}"
    );
}

#[test]
fn f16_matmul_without_tensorcore_uses_f16_vector_contract() {
    // No @tensorcore and an f16 accumulator: the generic register matmul
    // contracts in f16 (no fusion, no WMMA).
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
            let pm = program_id(0)
            let pn = program_id(1)
            var acc: tile<f16>[TILE_M, TILE_N] = 0.0
            for kt in range(0, K, TILE_K) {
                var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
        }",
    );
    assert_contains(&mlir, &["vector<4x4xf16>", "vector.contract"]);
    assert!(
        !mlir.contains("subgroup_mma"),
        "WMMA emitted for an f16 matmul without @tensorcore:\n{mlir}"
    );
}

#[test]
fn f16_flash_attention_runs_on_tensor_cores() {
    // f16 Q/K/V/O with an f32 online-softmax state: both matmuls run on
    // the tensor cores (f16 operands, f32 accumulate), the softmax math
    // stays f32, and the result rounds back to f16 on the store.
    let mlir = emit_mlir(
        "@autotune(D in [64], BR in [64], BC in [64])
        @tensorcore
        @pipeline
        @launch(256)
        @aligned(Nq = BR, Nk = BC)
        kernel flash_attention(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D],
                               V: tensor<f16>[Nk, D], O: tensor<f16>[Nq, D], scale: f32) {
            let pid = program_id(0)
            let row = pid * BR
            let q = Q[row :+ BR, :]
            var acc: tile<f32>[BR, D] = 0.0
            var m: tile<f32>[BR, 1] = -65504.0
            var l: tile<f32>[BR, 1] = 0.0
            for kt in range(0, Nk, BC) {
                let k = K[kt :+ BC, :]
                let v = V[kt :+ BC, :]
                var s: tile<f32>[BR, BC] = dot_t(q, k)
                s = s * scale
                var mnew: tile<f32>[BR, 1] = rowmax(s)
                mnew = tmax(m, mnew)
                var p: tile<f32>[BR, BC] = exp(s - mnew)
                var corr: tile<f32>[BR, 1] = exp(m - mnew)
                l = l * corr
                l += rowsum(p)
                acc = acc * corr
                acc += dot(p, v)
                m = mnew
            }
            acc = acc / l
            O[row :+ BR, :] = acc
        }",
    );
    assert_contains(
        &mlir,
        &[
            "gpu.func @flash_attention",
            "memref<?x64xf16, 1>",      // f16 Q/K/V/O tensors
            "gpu.subgroup_mma_compute", // dot_t and dot on the cores
            "ex2.approx.ftz.f32",       // f32 softmax exp
            "arith.truncf",             // f32 acc rounded to the f16 O
        ],
    );
    assert!(
        !mlir.contains("vector.contract"),
        "vector MACs left on the f16 tensor-core attention path:\n{mlir}"
    );
}

#[test]
fn f16_flash_attention_without_tensorcore_widens_to_f32() {
    // Same kernel, no @tensorcore and a non-fragmenting tile: the mixed
    // f16-input/f32-accumulate dots fall back to the vector path, widening
    // each f16 operand to f32 (arith.extf) on load.
    let mlir = emit_mlir(
        "@autotune(D in [8], BR in [8], BC in [8])
        @aligned(Nq = BR, Nk = BC)
        kernel flash_attention(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D],
                               V: tensor<f16>[Nk, D], O: tensor<f16>[Nq, D], scale: f32) {
            let pid = program_id(0)
            let row = pid * BR
            let q = Q[row :+ BR, :]
            var acc: tile<f32>[BR, D] = 0.0
            var m: tile<f32>[BR, 1] = -65504.0
            var l: tile<f32>[BR, 1] = 0.0
            for kt in range(0, Nk, BC) {
                let k = K[kt :+ BC, :]
                let v = V[kt :+ BC, :]
                var s: tile<f32>[BR, BC] = dot_t(q, k)
                s = s * scale
                var mnew: tile<f32>[BR, 1] = rowmax(s)
                mnew = tmax(m, mnew)
                var p: tile<f32>[BR, BC] = exp(s - mnew)
                var corr: tile<f32>[BR, 1] = exp(m - mnew)
                l = l * corr
                l += rowsum(p)
                acc = acc * corr
                acc += dot(p, v)
                m = mnew
            }
            acc = acc / l
            O[row :+ BR, :] = acc
        }",
    );
    assert_contains(
        &mlir,
        &[
            "vector.contract", // generic mixed-precision dots
            "arith.extf",      // f16 operands widened to the f32 accumulator
            "arith.truncf",    // f32 result rounded to the f16 output tensor
        ],
    );
    assert!(
        !mlir.contains("subgroup_mma"),
        "WMMA emitted without @tensorcore:\n{mlir}"
    );
}
