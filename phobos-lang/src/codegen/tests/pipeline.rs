// `@pipeline`: double buffering and cp.async.

use super::*;

#[test]
fn pipeline_double_buffers_staged_slices() {
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @pipeline
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
    assert_contains(
        &mlir,
        &[
            // two shared buffers per staged tile; the fused matmul has
            // no acc buffer and stages a k-major, so all four are 16x64
            // (allocation order: a0, b0, a1, b1)
            "@__matmul_tile0 : memref<16x64xf32, 3>",
            "@__matmul_tile1 : memref<16x64xf32, 3>",
            "@__matmul_tile2 : memref<16x64xf32, 3>",
            "@__matmul_tile3 : memref<16x64xf32, 3>",
            // guarded prefetches and the unrolled half-B guard
            "scf.if",
        ],
    );
    // Buffers are referenced statically (unroll-by-2), never selected.
    assert!(
        !mlir.contains("arith.select"),
        "unexpected dynamic buffer select in:\n{mlir}"
    );
}

#[test]
fn pipeline_uses_cp_async_on_supporting_targets() {
    let src = "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @pipeline
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
        }";
    // cp.async additionally requires 64-bit index lowering (upstream's
    // convert-nvgpu-to-nvvm hardcodes a 64-bit converter).
    use phobos_base::context::{Context as BaseContext, GpuConfig, NvidiaGpuConfig};
    let base = BaseContext {
        gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_80")),
        index_bitwidth: 64,
        ..Default::default()
    };
    let mlir = emit_mlir_base(src, &base);
    assert_contains(
        &mlir,
        &[
            // prefetches are 16B cp.async transfers, L1-bypassed
            "nvgpu.device_async_copy",
            "bypassL1",
            // one group per stage, waited on before the closing barrier
            "nvgpu.device_async_create_group",
            "nvgpu.device_async_wait",
        ],
    );
    // Without the capability the same kernel uses plain vector copies.
    let plain = emit_mlir(src);
    assert!(
        !plain.contains("nvgpu."),
        "cp.async leaked into a non-sm_80 target:\n{plain}"
    );
    // Under the default 32-bit index ABI, sm_80 must also stay plain.
    let narrow = emit_mlir_on(src, "sm_80");
    assert!(
        !narrow.contains("nvgpu."),
        "cp.async leaked into a 32-bit-index module:\n{narrow}"
    );
}

#[test]
fn tensorcore_f16_inputs_pipeline_with_cp_async() {
    // f16 operands stage into the WMMA fragments as a straight byte copy,
    // so the pipelined prefetch can lower to cp.async. The wmma opt-out
    // keeps this on the legacy path; at sm_80 + 64-bit index bare
    // @tensorcore would select mma.sync instead.
    let src = "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [32])
        @pipeline
        @tensorcore(wmma)
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
    // cp.async needs sm_80+ and 64-bit index lowering, as for the f32 path.
    use phobos_base::context::{Context as BaseContext, GpuConfig, NvidiaGpuConfig};
    let base = BaseContext {
        gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_80")),
        index_bitwidth: 64,
        ..Default::default()
    };
    let mlir = emit_mlir_base(src, &base);
    assert_contains(
        &mlir,
        &[
            "nvgpu.device_async_copy",
            "nvgpu.device_async_create_group",
            "nvgpu.device_async_wait",
            "gpu.subgroup_mma_compute",
        ],
    );
    // The promise on K makes the f16 transfers 8xf16, so they reach
    // cp.async.cg's 16 bytes and skip L1 as the f32 path does: a staged
    // tile is consumed from shared memory and never re-read through L1.
    assert!(
        mlir.contains("bypassL1"),
        "16-byte f16 cp.async should bypass L1:\n{mlir}"
    );
    // Capability-gated: sm_75 and the 32-bit-index ABI stay on plain copies.
    assert!(
        !emit_mlir(src).contains("nvgpu."),
        "cp.async leaked into a non-sm_80 target"
    );
    assert!(
        !emit_mlir_on(src, "sm_80").contains("nvgpu."),
        "cp.async leaked into a 32-bit-index module"
    );
}

#[test]
fn tensorcore_f16_pipeline_register_stages_on_sm75() {
    // Without cp.async (sm_75) the f16-input WMMA pipeline register-stages:
    // the next tile's global loads are hoisted into registers and held
    // across the WMMA compute, with the shared store deferred past it, so
    // the global latency overlaps the math. The load is unconditional with a
    // clamped index, the store guarded. This locks in that the path is taken
    // and stays cp.async-free; the ordering is checked in the emitted PTX.
    let src = "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
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
    let mlir = emit_mlir(src); // sm_75 (no cp.async)
    assert_contains(
        &mlir,
        &[
            // Unconditional global loads held in registers; the in-bounds
            // clamp is what the synchronous path lacks.
            "vector.load",
            "arith.subi",
            "arith.minsi",
            // the deferred shared store sits under the prefetch guard
            "scf.if",
            "vector.store",
            "gpu.subgroup_mma_compute",
        ],
    );
    assert!(
        !mlir.contains("nvgpu."),
        "cp.async leaked into the sm_75 register-staged path:\n{mlir}"
    );
    // The synchronous fallback applies when the staging tile does not divide
    // evenly across the CTA: a 16x16 A-tile is 256 elements, under one
    // 4-wide vector per thread, so no clamp is emitted.
    let small = src.replace("TILE_M in [64]", "TILE_M in [16]");
    let small = small.replace("TILE_N in [64]", "TILE_N in [128]");
    let plain = emit_mlir(&small);
    assert!(
        !plain.contains("arith.subi"),
        "register-staging fired on an indivisible staging tile:\n{plain}"
    );
}

#[test]
fn generic_pipeline_double_buffers_a_non_matmul_loop() {
    // a @pipeline loop that stages a slice but is not the fused matmul
    // template, so it exercises the generic software-pipelining path
    // (double buffers + guarded prefetch) rather than the GEMM backend.
    let mlir = emit_mlir(
        "@pipeline
        @autotune(T in [16])
        kernel stage(A: tensor<f32>[M, K], C: tensor<f32>[M, K]) {
            let pm = program_id(0)
            var acc: tile<f32>[T, T] = 0.0
            for kt in range(0, K, T) {
                var a = A[pm * T :+ T, kt :+ T]
                acc += a
            }
            C[pm * T :+ T, 0 :+ T] = acc
        }",
    );
    assert_contains(
        &mlir,
        &[
            "gpu.func @stage",
            // two shared staging buffers for the single staged slice
            "@__stage_tile0 : memref<16x16xf32, 3>",
            "@__stage_tile1 : memref<16x16xf32, 3>",
            "scf.if",      // the guarded prefetch of the next tile
            "gpu.barrier", // publish/consume barriers around staging
        ],
    );
}
