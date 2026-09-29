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
            // a stages k-major as 16x64, not 64x16. b is 16x64 anyway.
            "memref.view",
            "to memref<16x64xf32, 3>",
            "memref.dim",
            "memref.subview",
            "memref<64x16xf32, strided<[?, 1], offset: ?>, 1>",
            "gpu.thread_id",
            "gpu.block_dim",
            "gpu.barrier",
            // The accumulator rides the kt loop through vector.contract.
            "iter_args",
            "vector.contract",
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
    // The canonical pattern fuses. The lane's 4x4 accumulator rides the kt
    // loop, surplus warps clamp onto the last warp tile, and the epilogue
    // writes registers straight to the C subview.
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
    // acc is read again after the epilogue store, so it does not fuse. The
    // shared-accumulator path runs instead, still with vector contractions.
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
    // A 64x64 output in 4x4 sub-tiles is a 16x16 sub-tile grid. The 4x8
    // lane grid gives the squarest warp tiles (16x32), so there are
    // (16/4)*(16/8) = 8 warp tiles.
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
            "arith.constant 32 : index",
            "arith.divui",
            "arith.remui",
            "arith.constant 8 : index",
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
    // 128x64 is a 16x16 grid of 8x4 sub-tiles, one per thread of a
    // 256-thread CTA, so the lane accumulator is 8x4.
    let mlir = emit_mlir(&src("128"));
    assert_contains(&mlir, &["vector<8x4xf32>"]);
    // 64x64 with TM=8 would leave only 128 sub-tiles, so it stays 4x4.
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
    // Every matmul path writes the target as it goes, so an operand that is
    // also the target would be read after being partly overwritten, as when
    // squaring a matrix in place.
    let mlir = emit_mlir(
        "@launch(256)
        kernel square(X: tensor<f32>[R, N], O: tensor<f32>[R, N]) {
            var p: tile<f32>[16, 16] = X[0 :+ 16, 0 :+ 16]
            p = dot(p, p)
            O[0 :+ 16, 0 :+ 16] = p
        }",
    );
    // The temp shows as a second 16x16 buffer.
    let buffers = mlir.matches("memref<16x16xf32, 3>").count();
    assert!(
        buffers >= 2,
        "in-place square wants a temp, got {buffers} buffers:
{mlir}"
    );
}

#[test]
fn dot_falls_back_to_vector_without_tensorcore() {
    // Without @tensorcore the generic vector tile-dot runs, never WMMA.
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

#[test]
fn a_gemm_epilogue_with_bare_terms_scales_by_one() {
    // acc + c_old: both coefficients are emitted as f32 ones, and the prior
    // C is read back before the store.
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel gemm(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
            let pm = program_id(0)
            let pn = program_id(1)
            var acc: tile<f32>[TILE_M, TILE_N] = 0.0
            for kt in range(0, K, TILE_K) {
                let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }
            let c_old = C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N]
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc + c_old
        }",
    );
    let (_, epilogue) = split_at_kt_loop(&mlir);
    assert_eq!(epilogue.matches("arith.constant 1.000000e+00 : f32").count(), 2, "{epilogue}");
    assert_contains(epilogue, &["vector.load", "arith.mulf", "arith.addf", "vector.store"]);
}

#[test]
fn a_gemm_epilogue_scales_acc_by_its_coefficient_and_c_by_one() {
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel gemm(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
            let pm = program_id(0)
            let pn = program_id(1)
            var acc: tile<f32>[TILE_M, TILE_N] = 0.0
            for kt in range(0, K, TILE_K) {
                let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }
            let c_old = C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N]
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = 2.0 * acc + c_old
        }",
    );
    let (_, epilogue) = split_at_kt_loop(&mlir);
    assert_contains(
        epilogue,
        &["arith.constant 2.000000e+00 : f32", "arith.constant 1.000000e+00 : f32", "vector.load"],
    );
    assert_eq!(epilogue.matches("arith.constant 1.000000e+00 : f32").count(), 1, "{epilogue}");
}
