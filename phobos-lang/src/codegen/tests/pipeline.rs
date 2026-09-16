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
            "@__matmul_tile0 : memref<16x64xf32, 3>",
            "@__matmul_tile1 : memref<16x64xf32, 3>",
            "@__matmul_tile2 : memref<16x64xf32, 3>",
            "@__matmul_tile3 : memref<16x64xf32, 3>",
            "scf.if",
        ],
    );
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
            "nvgpu.device_async_copy",
            "bypassL1",
            "nvgpu.device_async_create_group",
            "nvgpu.device_async_wait",
        ],
    );
    let plain = emit_mlir(src);
    assert!(
        !plain.contains("nvgpu."),
        "cp.async leaked into a non-sm_80 target:\n{plain}"
    );
    let narrow = emit_mlir_on(src, "sm_80");
    assert!(
        !narrow.contains("nvgpu."),
        "cp.async leaked into a 32-bit-index module:\n{narrow}"
    );
}

#[test]
fn tensorcore_f16_inputs_pipeline_with_cp_async() {
    // f16 operands byte-copy into the WMMA fragments, so the prefetch can
    // lower to cp.async. wmma keeps this on the legacy path; bare @tensorcore
    // at sm_80 + 64-bit index would select mma.sync instead.
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
    // K's alignment makes the f16 transfer 8xf16 (16 bytes), reaching
    // cp.async.cg's L1-bypass threshold.
    assert!(
        mlir.contains("bypassL1"),
        "16-byte f16 cp.async should bypass L1:\n{mlir}"
    );
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
    // Without cp.async (sm_75) the f16 WMMA pipeline register-stages: the
    // next tile's global load is hoisted into registers and held across the
    // WMMA compute, with the shared store deferred past it so global latency
    // overlaps the math.
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
            "vector.load",
            "arith.subi",
            "arith.minsi",
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
    // Misnamed: this loop's slice is partial, so `pipeline_candidate`
    // declines it. What is actually under test is `emit_split_for`'s
    // main-loop-plus-masked-remainder buffers, not double buffering;
    // `bare_kernel_auto_pipelines_without_the_attribute` is the real
    // generic-pipeline test.
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
            "@__stage_tile0 : memref<16x16xf32, 3>",
            "@__stage_tile1 : memref<16x16xf32, 3>",
            "scf.if",
            "gpu.barrier",
        ],
    );
}

#[test]
fn bare_kernel_auto_pipelines_without_the_attribute() {
    // A row-at-a-time loop: dimension 0's size-1 span is always in bounds,
    // and `@aligned(K = T)` covers dimension 1, so the slice is not partial
    // and auto-pipelines with no `@pipeline` attribute written.
    let mlir = emit_mlir(
        "@autotune(T in [16])
        @aligned(K = T)
        kernel stage(A: tensor<f32>[M, K], C: tensor<f32>[M, K]) {
            let pm = program_id(0)
            var acc: tile<f32>[1, T] = 0.0
            for kt in range(0, M, 1) {
                var a = A[kt :+ 1, 0 :+ T]
                acc += a
            }
            C[0 :+ 1, 0 :+ T] = acc
        }",
    );
    // tile0 is `acc`; tile1 and tile2 are `a`'s ping-pong buffers. Three
    // globals distinguishes this from the single-buffered path, which mints
    // only tile0 and tile1.
    assert_contains(
        &mlir,
        &[
            "@__stage_tile0 : memref<1x16xf32, 3>",
            "@__stage_tile1 : memref<1x16xf32, 3>",
            "@__stage_tile2 : memref<1x16xf32, 3>",
            "scf.if",
        ],
    );
}

#[test]
fn pipeline_assertion_fails_with_the_decline_reason() {
    // `@pipeline` on a kernel with no loop shaped for it (here: no loop at
    // all) is an assertion, so it must fail to compile and name why.
    // `codegen::emit` reports rather than errors; `compile_shared` enforces.
    let err = crate::compile_shared(
        &phobos_base::context::Context::default(),
        "@pipeline
        kernel flat(A: tensor<f32>[N], C: tensor<f32>[N]) {
            let i = program_id(0)
            C[i :+ 1] = A[i :+ 1] * 2.0
        }",
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("flat"), "should name the kernel: {err}");
    assert!(
        err.contains("@pipeline"),
        "should say this was an assertion: {err}"
    );
}

#[test]
fn shared_memory_budget_declines_silently_without_the_attribute() {
    // At T = 8192 a staged buffer is 32768 bytes, 65536 doubled: over the 48
    // KiB budget, so pipelining declines silently and falls back to the
    // plain loop. The literal row count keeps the bound affine, so the
    // fallback skips `emit_split_for`'s split, whose remainder buffer would
    // defeat the assertion below.
    let mlir = emit_mlir(
        "@autotune(T in [8192])
        @aligned(K = T)
        kernel stage(A: tensor<f32>[64, K], C: tensor<f32>[64, K]) {
            let pm = program_id(0)
            var acc: tile<f32>[1, T] = 0.0
            for kt in range(0, 64, 1) {
                var a = A[kt :+ 1, 0 :+ T]
                acc += a
            }
            C[0 :+ 1, 0 :+ T] = acc
        }",
    );
    // tile0 is `acc`, tile1 is `a`'s single (non-doubled) buffer; a
    // pipelined form would additionally mint tile2 for `a`'s second buffer.
    assert!(
        !mlir.contains("__stage_tile2"),
        "a loop over the shared-memory budget should not get a second buffer:\n{mlir}"
    );
}

#[test]
fn atomic_add_cannot_reach_a_loop_bound() {
    // Not a pipelining test: guards the CTA-uniformity argument that no
    // `.ph` expression can put a data-dependent value into a loop bound. An
    // int-to-index conversion added later would need to pass this too.
    let err = emit_err(
        "kernel stage(A: tensor<f32>[M, K], C: tensor<f32>[M, K], BAR: tensor<i32>[2]) {
            let n = atomic_add(BAR, 0, 1)
            for kt in range(0, n, 1) {
                C[kt :+ 1, 0 :+ 1] = A[kt :+ 1, 0 :+ 1]
            }
        }",
    );
    assert!(
        err.contains("loop end must be an integer, got i32"),
        "atomic_add's i32 result should not typecheck as a loop bound: {err}"
    );

    // The same holds combined with an index-typed value through arithmetic:
    // `unify` bails on the mismatch rather than promoting the i32 side.
    let err = emit_err(
        "kernel stage(A: tensor<f32>[M, K], C: tensor<f32>[M, K], BAR: tensor<i32>[2]) {
            let n = atomic_add(BAR, 0, 1)
            for kt in range(0, n * 1, 1) {
                C[kt :+ 1, 0 :+ 1] = A[kt :+ 1, 0 :+ 1]
            }
        }",
    );
    assert!(
        err.contains("mismatched operand types"),
        "an i32/index binary op should not silently promote to index: {err}"
    );
}
