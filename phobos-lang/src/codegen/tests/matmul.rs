// The vector and register-blocked matmul path, without tensor cores.

use super::*;

#[test]
fn matmul_kernel_lowers_to_subviews_and_distributed_loops() {
    // The SPEC example, verbatim.
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64, 128], TILE_N in [64, 128], TILE_K in [16, 32])
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
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
            // staging buffers in shared memory: a is staged k-major
            // (transposed: 16x64, not 64x16), b is 16x64 anyway
            "memref.global \"private\" @__matmul_tile0 : memref<16x64xf32, 3>",
            "memref.global \"private\" @__matmul_tile1 : memref<16x64xf32, 3>",
            "memref.get_global @__matmul_tile0",
            // K is dynamic, so the kt loop is scf
            "memref.dim",
            // tile loads are subviews with static sizes and dynamic offsets
            "memref.subview",
            "memref<64x16xf32, strided<[?, 1], offset: ?>, 1>",
            // staging is distributed across the CTA's threads, with
            // barriers publishing it
            "gpu.thread_id",
            "gpu.block_dim",
            "gpu.barrier",
            // the lane's accumulator vector rides the kt loop as an
            // iter_arg, fed by k-chunk contractions (whose outer-product
            // lowering ends in single-rounding vector FMAs -> PTX fma.rn)
            "iter_args",
            "vector.contract",
            // fragment loads and the epilogue store are 128-bit vectors
            "vector.load",
            "vector.store",
        ],
    );
    // The acc tile itself never exists in shared memory.
    assert!(
        !mlir.contains("memref<64x64xf32, 3>"),
        "unexpected shared acc buffer in:\n{mlir}"
    );
}

#[test]
fn matmul_accumulates_in_registers() {
    // Same kernel as above: the canonical pattern fuses, so the lane's
    // 4x4 accumulator vector rides the kt loop, surplus warps clamp
    // onto the last warp tile, and the epilogue writes registers
    // straight to the C subview.
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
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
        &["arith.minsi", "vector<4x4xf32>", "vector.contract"],
    );
    assert!(
        !mlir.contains("memref<64x64xf32, 3>"),
        "unexpected shared acc buffer in:\n{mlir}"
    );
}

#[test]
fn register_fusion_bails_when_acc_outlives_store() {
    // acc is read again after the epilogue store -> no fusion; the
    // shared-accumulator path runs instead (also contraction-based,
    // but acc lives in shared memory and round-trips per kt).
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
            let pm = program_id(0)
            let pn = program_id(1)
            var acc: tile<f32>[TILE_M, TILE_N] = 0.0
            for kt in range(0, K, TILE_K) {
                let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
        }",
    );
    assert_contains(&mlir, &["memref<64x64xf32, 3>", "vector.contract"]);
    assert!(
        !mlir.contains("math.fma"),
        "scalar FMAs left on the unfused path:\n{mlir}"
    );
}

#[test]
fn matmul_is_warp_tiled() {
    // 64x64 output, 4x4 sub-tiles -> 16x16 sub-tile grid; the 4x8 lane
    // grid wins (16x32 warp tiles, most square -> minimal shared traffic),
    // giving (16/4)*(16/8) = 8 warp tiles.
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
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
            // tid decomposes against the warp size into warp id and
            // lane (unsigned: non-negative, pow2 -> shift/mask)
            "arith.constant 32 : index",
            "arith.divui",
            "arith.remui",
            // warps stride over the 8 warp tiles
            "arith.constant 8 : index",
            // lane offsets scale by the 16x32 warp-tile extents
            "arith.constant 16 : index",
        ],
    );
}

#[test]
fn large_tiles_widen_register_blocking() {
    let src = |tile_m: &str| {
        format!(
            "@autotune(TILE_M in [{tile_m}], TILE_N in [64], TILE_K in [16])
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {{
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {{
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }}
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }}"
        )
    };
    // 128x64: a 16x16 grid of 8x4 sub-tiles (256, one per thread of a
    // 256-thread CTA) -> the lane accumulator is an 8x4 vector.
    let mlir = emit_mlir(&src("128"));
    assert_contains(&mlir, &["vector<8x4xf32>"]);
    // 64x64 can't afford TM=8 (would leave only 128 sub-tiles): 4x4.
    let mlir = emit_mlir(&src("64"));
    assert_contains(&mlir, &["vector<4x4xf32>"]);
    assert!(
        !mlir.contains("vector<8x4xf32>"),
        "unexpected TM=8 upgrade in:\n{mlir}"
    );
}

#[test]
fn shape_overrides_pin_autotune_choices() {
    let src = "@autotune(TILE in [16, 32])
        kernel k(A: tensor<f32>[N], B: tensor<f32>[N]) {
            for j in range(0, TILE) {
                B[j] = A[j]
            }
        }";
    // Default: the first choice seeds the shape env.
    assert!(emit_mlir(src).contains("to 16"));
    // The autotuner pins a choice through the base context.
    let base = phobos_base::context::Context {
        shape_overrides: [("TILE".to_string(), 32)].into(),
        ..Default::default()
    };
    assert!(emit_mlir_base(src, &base).contains("to 32"));
}

#[test]
fn a_matmul_into_one_of_its_own_operands_uses_a_temp() {
    // Every matmul path writes the target as it goes, so an operand that
    // is the target would be read after it had been partly overwritten.
    // Squaring a matrix in place is the shape this takes in practice, and
    // it produced a plausible wrong answer rather than a failure: the
    // first four rows of a 16-row triangular inverse were right and the
    // rest were not.
    let mlir = emit_mlir(
        "@launch(256)
        kernel square(X: tensor<f32>[R, N], O: tensor<f32>[R, N]) {
            var p: tile<f32>[16, 16] = X[0 :+ 16, 0 :+ 16]
            p = dot(p, p)
            O[0 :+ 16, 0 :+ 16] = p
        }",
    );
    // The temp is the tell: two 16x16 buffers rather than one.
    let buffers = mlir.matches("memref<16x16xf32, 3>").count();
    assert!(
        buffers >= 2,
        "in-place square wants a temp, got {buffers} buffers:
{mlir}"
    );
}

#[test]
fn dot_falls_back_to_vector_without_tensorcore() {
    // Same shapes, but no @tensorcore: the generic tile-dot (vector)
    // path must run, never WMMA.
    let mlir = emit_mlir(
        "@autotune(D in [64], BR in [64], BC in [64])
        @launch(256)
        @aligned(Nq = BR, Nk = BC)
        kernel qk(Q: tensor<f32>[Nq, D], K: tensor<f32>[Nk, D], S: tensor<f32>[Nq, Nk]) {
            let pid = program_id(0)
            let row = pid * BR
            let q = Q[row :+ BR, :]
            let k = K[0 :+ BC, :]
            var s: tile<f32>[BR, BC] = dot_t(q, k)
            S[row :+ BR, 0 :+ BC] = s
        }",
    );
    assert!(
        !mlir.contains("subgroup_mma"),
        "WMMA emitted for dot_t without @tensorcore:\n{mlir}"
    );
}
