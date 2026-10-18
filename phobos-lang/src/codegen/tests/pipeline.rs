// `@pipeline`: double buffering and cp.async.

use super::*;

#[test]
fn pipeline_double_buffers_staged_slices() {
    let mlir = emit_mlir(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @pipeline
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
            let pm = ctx.block.x
            let pn = ctx.block.y
            var acc: tile<f32>[TILE_M, TILE_N] = 0.0
            for kt in range(0, K, TILE_K) {
                var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
        }",
    );
    assert_eq!(
        tile_views(&mlir, "16x64xf32").len(),
        4,
        "two staging pairs, doubled:\n{mlir}"
    );
    assert_contains(&mlir, &["scf.if"]);
    assert!(
        !mlir.contains("arith.select"),
        "unexpected dynamic buffer select in:\n{mlir}"
    );
}

#[test]
fn pipeline_uses_cp_async_on_supporting_targets() {
    // Not a gemm: the f32 gemm stages a k-major and keeps register staging
    // on every chip.
    let src = "@autotune(TILE_M in [64], TILE_K in [16])
        @pipeline
        @aligned(M = TILE_M, K = TILE_K)
        kernel rowsum(A: tensor<f32>[M, K], C: tensor<f32>[M, TILE_K]) {
            let pm = ctx.block.x
            var acc: tile<f32>[TILE_M, TILE_K] = 0.0
            for kt in range(0, K, TILE_K) {
                var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                acc += a
            }
            C[pm * TILE_M :+ TILE_M, 0 :+ TILE_K] = acc
        }";
    // cp.async also needs 64-bit index lowering, since upstream's
    // convert-nvgpu-to-nvvm hardcodes a 64-bit converter.
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
    // f16 operands are copied byte for byte, so the prefetch can use
    // cp.async. `@tensorcore(wmma)` forces the WMMA path, since bare
    // @tensorcore at sm_80 with 64-bit indices would pick mma.sync.
    let src = "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [32])
        @pipeline
        @tensorcore(wmma)
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
            let pm = ctx.block.x
            let pn = ctx.block.y
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
    // K's alignment makes the f16 copy 8xf16 (16 bytes), which is what
    // cp.async.cg needs to bypass L1.
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
    // Without cp.async (sm_75), the f16 WMMA pipeline stages in registers.
    // The next tile's global load goes into registers before the compute,
    // and the shared store comes after it.
    let src = "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        @pipeline
        @tensorcore
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
            let pm = ctx.block.x
            let pn = ctx.block.y
            var acc: tile<f16>[TILE_M, TILE_N] = 0.0
            for kt in range(0, K, TILE_K) {
                var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
        }";
    let mlir = emit_mlir(src); // sm_75, no cp.async
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
    // A staging tile that does not divide evenly across the CTA takes the
    // synchronous path. A 16x16 A-tile is 256 elements, less than one
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
    // declines it. The test really covers the buffers of `emit_split_for`'s
    // main loop and masked remainder. The generic pipeline test is
    // `bare_kernel_auto_pipelines_without_the_attribute`.
    let mlir = emit_mlir(
        "@pipeline
        @autotune(T in [16])
        kernel stage(A: tensor<f32>[M, K], C: tensor<f32>[M, K]) {
            let pm = ctx.block.x
            var acc: tile<f32>[T, T] = 0.0
            for kt in range(0, K, T) {
                var a = A[pm * T :+ T, kt :+ T]
                acc += a
            }
            C[pm * T :+ T, 0 :+ T] = acc
        }",
    );
    assert_contains(&mlir, &["gpu.func @stage", "scf.if", "gpu.barrier"]);
    // `acc` and `a`. The remainder's copy of `a` reuses the main loop's
    // bytes.
    let mut offsets = view_offsets(&mlir, "16x16xf32");
    offsets.sort_unstable();
    offsets.dedup();
    assert_eq!(offsets.len(), 2, "{mlir}");
}

#[test]
fn bare_kernel_auto_pipelines_without_the_attribute() {
    // A row-at-a-time loop. Dimension 0's size-1 span is always in bounds,
    // and `@aligned(K = T)` covers dimension 1. So the slice is whole and
    // pipelines without a `@pipeline` attribute.
    let mlir = emit_mlir(
        "@autotune(T in [16])
        @aligned(K = T)
        kernel stage(A: tensor<f32>[M, K], C: tensor<f32>[M, K]) {
            let pm = ctx.block.x
            var acc: tile<f32>[1, T] = 0.0
            for kt in range(0, M, 1) {
                var a = A[kt :+ 1, 0 :+ T]
                acc += a
            }
            C[0 :+ 1, 0 :+ T] = acc
        }",
    );
    // One buffer for `acc` and two for `a`'s ping-pong pair. The
    // single-buffered path would have two in total.
    assert_eq!(tile_views(&mlir, "1x16xf32").len(), 3, "{mlir}");
    assert_contains(&mlir, &["scf.if"]);
}

#[test]
fn pipeline_assertion_fails_with_the_decline_reason() {
    // `@pipeline` is an assertion. On a kernel with no suitable loop, here
    // no loop at all, compiling must fail and say why. `codegen::emit` only
    // reports, and `compile_shared` turns that into an error.
    let err = crate::compile_shared(
        &phobos_base::context::Context::default(),
        "@pipeline
        kernel flat(A: tensor<f32>[N], C: tensor<f32>[N]) {
            let i = ctx.block.x
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
    // At T = 8192 a staged buffer is 32768 bytes, 65536 doubled, which is
    // over the 48 KiB budget. So pipelining declines silently and the plain
    // loop runs. The literal row count keeps the bound affine, which avoids
    // `emit_split_for` and its remainder buffer.
    let stage = |t: i64| {
        emit_mlir(&format!(
            "@autotune(T in [{t}])
            @aligned(K = T)
            kernel stage(A: tensor<f32>[64, K], C: tensor<f32>[64, K]) {{
                for kt in range(0, 64, 1) {{
                    var a = A[kt :+ 1, 0 :+ T]
                    C[kt :+ 1, 0 :+ T] = a * 2.0
                }}
            }}"
        ))
    };
    // One buffer for `a`; pipelining adds a second, as it does at T = 2048.
    let mlir = stage(8192);
    assert_eq!(
        tile_views(&mlir, "1x8192xf32").len(),
        1,
        "a loop over the shared-memory budget should not get a second buffer:\n{mlir}"
    );
    assert_eq!(tile_views(&stage(2048), "1x2048xf32").len(), 2);
}

#[test]
fn atomic_add_cannot_reach_a_loop_bound() {
    // Not a pipelining test. It guards the CTA-uniformity argument: no
    // `.ph` expression can put a data-dependent value into a loop bound. Any
    // future int-to-index conversion must keep this passing.
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

    // The same holds through arithmetic with an index value. `unify`
    // rejects the mismatch instead of promoting the i32 side.
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

#[test]
fn a_dynshared_kernel_pipelines_past_the_static_ceiling() {
    // Two 64 x 64 f32 stages are 32 KB, doubled 64 KB: over the 48 KB a
    // static kernel may hold, within what a dynamic launch opts into.
    let src = |attrs: &str| {
        format!(
            "@pipeline @aligned(K = 64) {attrs}
            kernel k(A: tensor<f32>[64, K], B: tensor<f32>[64, K], C: tensor<f32>[64, 64]) {{
                var acc: tile<f32>[64, 64] = 0.0
                for kt in range(0, K, 64) {{
                    var a = A[0 :+ 64, kt :+ 64]
                    var b = B[0 :+ 64, kt :+ 64]
                    acc += a * b
                }}
                C[0 :+ 64, 0 :+ 64] = acc
            }}"
        )
    };
    use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};
    let ctx = Context {
        gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_80")),
        ..Context::default()
    };
    let fails = |attrs: &str| crate::compile_raw(&ctx, &src(attrs)).unwrap().pipeline_failures;
    let static_fails = fails("");
    assert!(
        static_fails.iter().any(|f| f.reasons.iter().any(|w| w.contains("49152-byte ceiling"))),
        "{static_fails:?}"
    );
    assert!(fails("@dynshared").is_empty());
}

#[test]
fn a_dynamic_extent_pipelines_its_whole_tiles() {
    // K is only known at launch: the whole tiles pipeline, and the ragged
    // rest runs once, masked.
    let src = "@pipeline
        kernel k(A: tensor<f32>[64, K], B: tensor<f32>[64, K], C: tensor<f32>[64, 32]) {
            var acc: tile<f32>[64, 32] = 0.0
            for kt in range(0, K, 32) {
                var a = A[0 :+ 64, kt :+ 32]
                var b = B[0 :+ 64, kt :+ 32]
                acc += a * b
            }
            C[0 :+ 64, 0 :+ 32] = acc
        }";
    use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};
    let ctx = Context {
        gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_80")),
        ..Context::default()
    };
    let out = crate::compile_raw(&ctx, src).unwrap();
    assert!(out.pipeline_failures.is_empty(), "{:?}", out.pipeline_failures);
    // The remainder stages into the loop's buffers: the accumulator and
    // four 8 KiB stages, not two more beside them.
    assert_eq!(shared_bytes(&emit_mlir_on(src, "sm_80")), 40960);
}

/// A gemm k-loop over `elem` tiles, with `attrs` above the kernel.
fn gemm(attrs: &str, elem: &str, tile_k: i64) -> String {
    format!(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [{tile_k}])
        {attrs}
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<{elem}>[M, K], B: tensor<{elem}>[K, N], C: tensor<f32>[M, N]) {{
            let pm = ctx.block.x
            let pn = ctx.block.y
            var acc: tile<f32>[TILE_M, TILE_N] = 0.0
            for kt in range(0, K, TILE_K) {{
                var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }}
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
        }}"
    )
}

#[test]
fn a_gemm_double_buffers_without_the_attribute() {
    assert_eq!(emit_mlir(&gemm("", "f32", 16)), emit_mlir(&gemm("@pipeline", "f32", 16)));
    let mlir = emit_mlir(&gemm("", "f32", 16));
    assert_eq!(tile_views(&mlir, "16x64xf32").len(), 4, "two staging pairs, doubled:\n{mlir}");
    // On the tensor cores at sm_80 the doubled stages stream through cp.async.
    let mlir = emit_mlir_sync(&gemm("@tensorcore", "f16", 16), "sm_80");
    assert_contains(&mlir, &["nvgpu.device_async_copy", "nvgpu.device_async_wait"]);
}

#[test]
fn pipeline_off_keeps_one_staging_pair() {
    let mlir = emit_mlir(&gemm("@pipeline(off)", "f32", 16));
    assert_eq!(tile_views(&mlir, "16x64xf32").len(), 2, "one staging pair:\n{mlir}");
    let mlir = emit_mlir_sync(&gemm("@pipeline(off)\n        @tensorcore", "f16", 16), "sm_80");
    assert!(!mlir.contains("nvgpu.device_async"), "cp.async under @pipeline(off):\n{mlir}");
}

#[test]
fn a_gemm_whose_doubled_stages_do_not_fit_stays_single_unless_asserted() {
    // 64-deep f32 stages are 32 KB a pair: one fits a block, two do not.
    let mlir = emit_mlir(&gemm("", "f32", 64));
    assert_eq!(tile_views(&mlir, "64x64xf32").len(), 2, "one staging pair:\n{mlir}");
    let err = crate::compile_shared(&phobos_base::context::Context::default(), &gemm("@pipeline", "f32", 64))
        .unwrap_err()
        .to_string();
    assert!(err.contains("doubled staging would need"), "got: {err}");
}

#[test]
fn pipeline_takes_on_or_off() {
    let err = crate::compile_shared(&phobos_base::context::Context::default(), &gemm("@pipeline(sometimes)", "f32", 16))
        .unwrap_err()
        .to_string();
    assert!(err.contains("`@pipeline` takes `on` or `off`"), "got: {err}");
}

#[test]
fn lanes_read_a_staged_tile_the_loop_pipelines() {
    // The loop carries a register accumulator, so it pipelines by default and
    // binds `xs` to its stage, which lane runs read as the whole buffer.
    let src = "@launch(256)
        @aligned(K = 128)
        kernel mv(W: tensor<f32>[N, K], X: tensor<f32>[1, K], Y: tensor<f32>[1, N]) {
            let row = ctx.block.x * ctx.warps + ctx.warp
            var acc: f32 = 0.0
            for kt in range(0, K, 128) {
                var xs = X[0 :+ 1, kt :+ 128]
                let at = ctx.lane * 4
                if row < N {
                    acc += sum(W[row, kt + at :+ 4] * xs[0, at :+ 4])
                }
            }
            let total = sum<over ctx.lane>(acc)
            if ctx.lane == 0 && row < N {
                Y[0, row] = total
            }
        }";
    let mlir = emit_mlir_sync(src, "sm_80");
    assert_contains(&mlir, &["nvgpu.device_async_copy"]);
}
