// `@tensorcore` matmul: wmma on sm_80 and later, mma.sync on sm_75.

use super::*;

#[test]
fn tensorcore_matmul_uses_wmma() {
    // At the 32-bit-index default emit_mlir uses, @tensorcore falls back
    // to the legacy WMMA path (mma.sync needs 64-bit index).
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @tensorcore
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
            // f16 staging pads the inner dim by 8 for bank conflicts
            // (16 -> 24, 64 -> 72).
            "memref<64x24xf16, 3>",
            "memref<16x72xf16, 3>",
            "arith.truncf",
            "gpu.subgroup_mma_load_matrix",
            "!gpu.mma_matrix<16x16xf16, \"AOp\">",
            "!gpu.mma_matrix<16x16xf16, \"BOp\">",
            "gpu.subgroup_mma_compute",
            "!gpu.mma_matrix<16x16xf32, \"COp\">",
            "iter_args",
            "leadDimension = 24 : index",
            "leadDimension = 72 : index",
            // the epilogue drains through a per-warp f32 slab (8 warps x 16 rows).
            "gpu.subgroup_mma_store_matrix",
            "memref<128x16xf32, 3>",
        ],
    );
    // The tensor cores replace the vector-FMA MAC grid entirely.
    assert!(
        !mlir.contains("vector.contract"),
        "vector MACs left on the tensor-core path:\n{mlir}"
    );
}

#[test]
fn tensorcore_uses_mma_sync_on_sm75() {
    // Bare @tensorcore defaults to the mma.sync path (at 64-bit index).
    let mlir = emit_mlir_sync(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @tensorcore
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
        "sm_75",
    );
    assert_contains(
        &mlir,
        &[
            // f16 staging, but UNPADDED (no +8): the legacy pad fought bank
            // conflicts ldmatrix instead defeats with a swizzle.
            "memref<64x16xf16, 3>",
            "memref<16x64xf16, 3>",
            "arith.truncf",
            // ldmatrix loads the per-lane fragments: A as two 8x8 tiles
            // (m16k8), B as one transposed 8x8 (k8n8).
            "nvgpu.ldmatrix",
            "numTiles = 2 : i32",
            "numTiles = 1 : i32",
            "transpose = true",
            "transpose = false",
            "nvgpu.mma.sync",
            "mmaShape = [16, 8, 8]",
            "vector<2x2xf16>",
            "vector<1x2xf16>",
            "-> vector<2x2xf32>",
            // arith.xori is the bank-conflict swizzle on the staging store and load.
            "arith.xori",
            // the epilogue still drains through the per-warp f32 slab
            "memref<128x16xf32, 3>",
        ],
    );
    // The legacy WMMA ops are gone, and so is the padded staging.
    assert!(
        !mlir.contains("subgroup_mma") && !mlir.contains("mma_matrix"),
        "legacy WMMA ops left on the mma.sync path:\n{mlir}"
    );
    assert!(
        !mlir.contains("memref<64x24xf16"),
        "padded staging left on the mma.sync path:\n{mlir}"
    );
}

#[test]
fn tensorcore_sync_widens_k_to_16_on_sm80() {
    // Ampere+ has the m16n8k16 shape, halving the k-steps vs Turing's k8.
    let mlir = emit_mlir_sync(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @tensorcore(sync)
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
        "sm_80",
    );
    assert_contains(
        &mlir,
        &[
            "mmaShape = [16, 8, 16]",
            // wider k: A spans four 8x8 tiles, B spans two.
            "numTiles = 4 : i32",
            "numTiles = 2 : i32",
            "vector<4x2xf16>",
            "vector<2x2xf16>",
        ],
    );
}

#[test]
fn tensorcore_sync_f16_accumulator() {
    // The gemm_fp16.ph shape: f16 inputs and an f16 accumulator, which the
    // mma.sync path carries as a vector<2x2xf16> C/D fragment.
    let mlir = emit_mlir_sync(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @tensorcore(sync)
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
        "sm_75",
    );
    assert_contains(
        &mlir,
        &[
            "nvgpu.mma.sync",
            "mmaShape = [16, 8, 8]",
            "-> vector<2x2xf16>",
            "memref<128x16xf16, 3>",
            // extf/truncf here are the epilogue's f16 drain, not input staging.
            "vector<4xf16>",
            "arith.extf",
            "arith.truncf",
        ],
    );
}

#[test]
fn tensorcore_wmma_optout_forces_legacy() {
    // @tensorcore defaults to mma.sync, but @tensorcore(wmma) forces the
    // legacy warp-collective WMMA back on even at 64-bit index, where
    // mma.sync would otherwise be selected.
    let mlir = emit_mlir_sync(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @tensorcore(wmma)
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
        "sm_75",
    );
    assert!(
        mlir.contains("subgroup_mma") && !mlir.contains("nvgpu.mma.sync"),
        "@tensorcore(wmma) should force the legacy WMMA path:\n{mlir}"
    );
}

#[test]
fn tensorcore_falls_back_to_wmma_without_wide_index() {
    // mma.sync needs 64-bit index lowering (nvgpu memref descriptors are
    // pointer-width). At the 32-bit default (what emit_mlir uses), codegen
    // stays on WMMA rather than emit casts that won't reconcile.
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
    assert!(
        mlir.contains("subgroup_mma") && !mlir.contains("nvgpu."),
        "sync at 32-bit index should fall back to WMMA:\n{mlir}"
    );
}

#[test]
fn tensorcore_pipelines_f16_staging() {
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [32])
        @pipeline
        @tensorcore
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
            let pm = program_id(0)
            let pn = program_id(1)
            var acc: tile<f32>[TILE_M, TILE_N] = 0.0
            for kt in range(0, K, TILE_K) {
                var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
        }",
    );
    // f16 staging pads the inner dim by 8 for bank conflicts (32 -> 40,
    // 64 -> 72), one pair per operand.
    assert_eq!(tile_views(&mlir, "64x40xf16").len(), 2, "{mlir}");
    assert_eq!(tile_views(&mlir, "32x72xf16").len(), 2, "{mlir}");
    assert_contains(&mlir, &["scf.if", "gpu.subgroup_mma_compute"]);
    // No cp.async here: these inputs are f32, and the f32 -> f16 staging
    // round-down can't be a raw cp.async byte transfer (f16 inputs can;
    // see tensorcore_f16_inputs_pipeline_with_cp_async).
    assert!(
        !mlir.contains("nvgpu."),
        "cp.async leaked into the f32 -> f16 staging:\n{mlir}"
    );
}

#[test]
fn tensorcore_drops_padding_only_when_maxregs_admits_a_cta() {
    // A large pipelined f16 tile (256x128x16): padded staging exceeds half
    // of sm_75's 64 KB carveout, so the pad blocks a second resident CTA.
    let body = "@autotune(TILE_M in [256], TILE_N in [128], TILE_K in [16])
        @LAUNCH
        @pipeline
        @tensorcore
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
        }";

    // Without a register cap, ptxas would keep one CTA on the register
    // limit anyway, so the pad stays (dropping it would only cost the
    // bank-conflict mitigation): buffers keep their padded width.
    let padded = emit_mlir(&body.replace("@LAUNCH", "@launch(256, 2)"));
    assert_contains(&padded, &["memref<256x24xf16, 3>", "memref<16x136xf16, 3>"]);

    // Hard-capping registers at 128 lets two CTAs fit by registers, so the
    // shared pad is now what blocks the second CTA -- it is dropped and the
    // staging stays at its logical width (lead = inner dim).
    let unpadded = emit_mlir(&body.replace("@LAUNCH", "@launch(256, 2, 128)"));
    assert_contains(
        &unpadded,
        &[
            "memref<256x16xf16, 3>",
            "memref<16x128xf16, 3>",
            "leadDimension = 16 : index",
            "leadDimension = 128 : index",
        ],
    );
    assert!(
        !unpadded.contains("memref<256x24xf16, 3>"),
        "padding survived despite the register cap admitting a second CTA:\n{unpadded}"
    );
}

#[test]
fn tensorcore_bails_to_vector_path_when_shape_does_not_fragment() {
    // TILE_K = 8 is not a multiple of 16 -> no WMMA; the regular
    // register-accumulator vector path must run instead.
    let src = "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [8])
        @tensorcore
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
        }";
    let mlir = emit_mlir(src);
    assert_contains(&mlir, &["vector.contract"]);
    assert!(
        !mlir.contains("subgroup_mma"),
        "WMMA emitted for a non-fragmenting shape:\n{mlir}"
    );
    // Pre-Volta chips have no tensor cores: @tensorcore is ignored.
    let old = emit_mlir_on(&src.replace("TILE_K in [8]", "TILE_K in [16]"), "sm_60");
    assert_contains(&old, &["vector.contract"]);
    assert!(
        !old.contains("subgroup_mma"),
        "WMMA emitted for a pre-sm_70 chip:\n{old}"
    );
}
