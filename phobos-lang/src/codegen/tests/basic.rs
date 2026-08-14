// Kernels, statements and scalar expressions: the parts of the language
// that do not involve tiles.

use super::*;

#[test]
fn add_kernel_lowers_to_gpu_func() {
    let mlir = emit_mlir(
        "kernel add(X: tensor<f32>[N], Y: tensor<f32>[N], Z: tensor<f32>[N], n: i32) {
            let i = program_id(0)
            if i < n {
                Z[i] = X[i] + Y[i]
            }
        }",
    );
    assert_contains(
        &mlir,
        &[
            "gpu.module",
            "gpu.func @add",
            "gpu.block_id",
            "arith.cmpi slt",
            "scf.if",
            "memref.load",
            "memref.store",
            "memref<?xf32, 1>",
        ],
    );
}

#[test]
fn constant_range_uses_affine_dynamic_uses_scf() {
    let mlir = emit_mlir(
        "@autotune(TILE in [16, 32])
        kernel k(A: tensor<f32>[N], B: tensor<f32>[N], n: i32) {
            for j in range(0, TILE) {
                B[j] += A[j]
            }
            for j in range(0, n) {
                B[j] = A[j]
            }
        }",
    );
    assert_contains(&mlir, &["affine.for", "scf.for"]);
    // @autotune's first choice seeds the bound.
    assert!(mlir.contains("to 16"), "expected `to 16` bound in:\n{mlir}");
}

#[test]
fn vars_and_while_lower_to_alloca_and_scf_while() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[N]) {
            var s = 0.0
            var i = 0
            while i < 10 {
                s += A[i]
                i = i + 1
            }
        }",
    );
    assert_contains(&mlir, &["memref.alloca() : memref<f32>", "scf.while"]);
}

#[test]
fn static_dims_from_literals_and_autotune() {
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64, 128])
        kernel k(A: tensor<f32>[TILE_M, 8], B: tensor<f32>[M, N]) {
            A[0, 0] = 1.0
        }",
    );
    assert_contains(&mlir, &["memref<64x8xf32, 1>", "memref<?x?xf32, 1>"]);
}

#[test]
fn scalar_ops_cover_arith_and_comparisons() {
    // every scalar operator across float, integer, and bool operands,
    // plus unary neg/not and an f32/f64 mixed-width promotion.
    let mlir = emit_mlir(
        "kernel k(out: tensor<f32>[N], a: f32, b: f64) {
            var f = a * a + a - a / a
            f = -f
            let wide = a + b
            let lt = a < a
            let cmp = (a == a) != (a <= a)
            var i = 0
            i = i * 2 + 1 - i / 1
            let ic = i < 4
            let bn = !lt
            out[0] = f
        }",
    );
    assert_contains(
        &mlir,
        &[
            "arith.mulf",
            "arith.addf",
            "arith.subf",
            "arith.divf",
            "arith.negf", // unary neg on a float
            "arith.cmpf", // float comparisons
            "arith.extf", // f32 -> f64 widening in a + b
            "arith.muli", // integer arithmetic on index
            "arith.cmpi", // integer comparison
            "arith.xori", // !lt on a bool
        ],
    );
}

#[test]
fn unknown_builtin_is_an_error() {
    let err = emit_err("kernel k(A: tensor<f32>[N]) { let x = wobble(A) }");
    assert!(err.contains("unknown function"), "got: {err}");
}

#[test]
fn program_id_dimension_must_be_literal_0_to_2() {
    let err = emit_err("kernel k(A: tensor<f32>[N]) { let x = program_id(3) }");
    assert!(err.contains("program_id"), "got: {err}");
}

#[test]
fn unknown_identifier_is_an_error() {
    let err = emit_err("kernel k(A: tensor<f32>[N]) { A[0] = nope }");
    assert!(err.contains("unknown identifier"), "got: {err}");
}

#[test]
fn tensor_used_as_value_is_an_error() {
    let err = emit_err("kernel k(A: tensor<f32>[N]) { let x = A }");
    assert!(err.contains("index or slice it"), "got: {err}");
}

#[test]
fn cumsum_tril_transpose_lower_and_verify() {
    // The linear-attention primitives: cumsum scans the sequence axis,
    // tril masks the strict upper triangle (a compare plus select), and
    // transpose mirrors a rank-2 tile so a contraction can run over the
    // leading axis.
    let mlir = emit_mlir(
        "@autotune(D in [32], C in [32])
        kernel prim(G: tensor<f32>[N, 1], X: tensor<f32>[N, D], O: tensor<f32>[N, D]) {
            let c = program_id(0)
            let g = G[c :+ C, :]
            var b: tile<f32>[C, 1] = cumsum(g)
            let x = X[c :+ C, :]
            var xb: tile<f32>[C, D] = x * b
            var p: tile<f32>[C, C] = dot_t(xb, xb)
            p = tril(p)
            var xt = transpose(xb)               // [D, C]
            var kv: tile<f32>[C, C] = dot(xb, xt) // [C,D] @ [D,C] -> [C,C]
            var o: tile<f32>[C, D] = dot(p, x)
            O[c :+ C, :] = o
        }",
    );
    assert_contains(
        &mlir,
        &[
            "gpu.func @prim",
            "arith.cmpi sle",  // tril's j <= i predicate
            "arith.select",    // tril keeps or zeroes each element
            "vector.contract", // the dot / dot_t matmuls
        ],
    );
}

#[test]
fn kda_chunkwise_gated_linear_attention_lowers() {
    // The KDA backbone (examples/kda_fp32.ph): chunkwise gated linear
    // attention carrying an [D, D] recurrent state, exercising cumsum
    // (the gate), tril (causal mask), transpose (K^T V), exp, and the
    // intra/inter dot products.
    let mlir = emit_mlir(
        "@autotune(D in [64], C in [32, 128])
        kernel kda(Q: tensor<f32>[N, D], K: tensor<f32>[N, D], V: tensor<f32>[N, D],
                   G: tensor<f32>[N, 1], O: tensor<f32>[N, D], scale: f32) {
            var S: tile<f32>[D, D] = 0.0
            for c in range(0, N, C) {
                let q = Q[c :+ C, :]
                let k = K[c :+ C, :]
                let v = V[c :+ C, :]
                let g = G[c :+ C, :]
                var b: tile<f32>[C, 1] = cumsum(g)
                var db = exp(b)
                var negb = b * -1.0
                var dbi = exp(negb)
                var qd: tile<f32>[C, D] = q * db
                qd = qd * scale
                var kd: tile<f32>[C, D] = k * dbi
                var p: tile<f32>[C, C] = dot_t(qd, kd)
                p = tril(p)
                var o: tile<f32>[C, D] = dot(p, v)
                o += dot(qd, S)
                O[c :+ C, :] = o
                var gt = transpose(g)
                var total: tile<f32>[1, 1] = rowsum(gt)
                var kfin = k * exp(total - b)
                var kt = transpose(kfin)
                var kv: tile<f32>[D, D] = dot(kt, v)
                S = S * exp(total) + kv
            }
        }",
    );
    assert_contains(
        &mlir,
        &[
            "gpu.func @kda",
            "ex2.approx.ftz.f32", // exp on the gates
            "arith.select",       // tril causal mask
            "vector.contract",    // the chunk matmuls
        ],
    );
}
