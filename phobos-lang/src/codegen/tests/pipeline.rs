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
    // Despite its name and the `@pipeline` attribute, this loop's slice is
    // partial (K is dynamic, unaligned, and offset by the loop's own
    // induction variable -- see the block comment at the end of pipeline.rs
    // on why an unbound loop var's divisor defaults to 1, which no `@aligned`
    // promise on the *tensor* side can satisfy): `pipeline_candidate`
    // declines it, and what this test actually locks in is
    // `emit_split_for`'s pre-existing main-loop-plus-masked-remainder
    // structure, which happens to also mint two same-shaped buffers for the
    // same staged name. `bare_kernel_auto_pipelines_without_the_attribute`
    // below is the real generic-pipeline-path test (verified via a distinct
    // third buffer, which this shape never reaches). Kept as-is: it still
    // exercises real codegen and still passes, just not for the reason its
    // name implies -- retitling it belongs with whoever next touches
    // `emit_split_for`, not this change.
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

#[test]
fn bare_kernel_auto_pipelines_without_the_attribute() {
    // A row-at-a-time loop: dimension 0's size-1 span is provably in bounds
    // regardless of alignment (any dynamic extent has room for one more
    // element wherever the loop var points), and `@aligned(K = T)` covers
    // dimension 1, so the slice is not partial -- eligible, and pipelined
    // with no `@pipeline` written at all.
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
    // tile0 is `acc` (declared once, outside the loop); tile1 and tile2 are
    // `a`'s two ping-pong buffers -- three globals total is what tells this
    // apart from the ordinary single-buffered path, which would only ever
    // mint tile0 and tile1.
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
    // `@pipeline` on a kernel with no loop shaped for pipelining (here: no
    // for loop at all) is now an assertion, not an opt-in, so it has to fail
    // to compile -- and name why, not just that it failed. `codegen::emit`
    // itself does not error (see `EmitOutput::pipeline_failures`, which
    // `Variants::compile` needs raw); the assertion is enforced by
    // `phobos_lang::compile_shared`, the entry point an ordinary single-
    // kernel caller uses.
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
    // Same eligible row-at-a-time shape as
    // `bare_kernel_auto_pipelines_without_the_attribute`, but T = 8192: one
    // staged buffer is 32768 bytes, doubled 65536 -- over the 48 KiB
    // ceiling. No `@pipeline` here, so this is a bare auto-attempt: declining
    // must fall back to the plain loop quietly, not error. A literal row
    // count (64, not the symbolic M the other tests use) keeps the bound
    // affine, so a decline here falls to the plain unmasked loop instead of
    // `emit_split_for`'s ragged-remainder split, whose own main+remainder
    // structure would otherwise mint a second buffer for an unrelated
    // reason and defeat this assertion.
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
    // Not a pipelining test: a tripwire for the CTA-uniformity proof at the
    // end of pipeline.rs, which argues no `.ph` expression can put a
    // data-dependent value (an atomic's return, a tensor load, ...) into a
    // loop bound, because `Codegen::coerce` has no int-to-index arm and
    // `Codegen::unify` bails on an int/index mismatch rather than promoting
    // it. That argument is why `pipeline_candidate`'s callers run no runtime
    // divergence check: every bound is CTA-uniform by construction, so a
    // lane-divergent one (which would turn `emit_pipelined_for`'s
    // barrier-in-`scf.if` guard into a hang) is not a case pipelining has to
    // defend against. If a future change adds an int-to-index conversion,
    // this stops failing and the assertions below catch it -- which is
    // exactly when that proof, and this loop, need a second look.
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
