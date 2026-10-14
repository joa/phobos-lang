// Staging into shared tiles: padding, word stores, the barriers around them,
// and a pipelined loop that carries register accumulators.

use super::*;

#[test]
fn an_explicit_barrier_orders_the_reads_after_it() {
    // The per-thread stores into `t` are ordered by `barrier()`, so the
    // second sweep over `t` needs no barrier of its own.
    let mlir = emit_mlir(
        "@launch(128)
        kernel f(A: tensor<f32>[64, 64], B: tensor<f32>[64, 64], C: tensor<f32>[64, 64]) {
            var t: tile<f32>[64, 64]
            for i in range(ctx.thread, 4096, ctx.threads) {
                t[i / 64, i % 64] = A[i / 64, i % 64]
            }
            barrier()
            B[0 :+ 64, 0 :+ 64] = t * 2.0
            C[0 :+ 64, 0 :+ 64] = t * 3.0
        }",
    );
    assert_eq!(mlir.matches("gpu.barrier").count(), 1, "{mlir}");
}

/// A kernel staging an int8 tile and a one-row table under `@padstage`,
/// reading the tile through a slice.
const PADDED: &str = "@launch(128)
    @padstage
    kernel p(A: tensor<i8>[64, 128], G: tensor<i8>[1, 2048], C: tensor<i8>[64, 32], D: tensor<i8>[1, 32]) {
        var a = A[0 :+ 64, 0 :+ 128]
        var g = G[0 :+ 1, 0 :+ 2048]
        C[0 :+ 64, 0 :+ 32] = a[0 :+ 64, 96 :+ 32]
        D[0 :+ 1, 0 :+ 32] = g[0 :+ 1, 0 :+ 32]
    }";

#[test]
fn a_padded_byte_tile_keeps_its_rows_16_byte_aligned_and_slices() {
    // Sixteen bytes of pad, not eight elements, so a row starts on a
    // 16-byte boundary; one row has no rows to spread and is not padded.
    let mlir = emit_mlir(PADDED);
    assert_contains(&mlir, &["memref<64x144xi8, 3>", "memref<1x2048xi8, 3>"]);
    assert!(!mlir.contains("memref<1x2064xi8"), "{mlir}");
    // The slice steps rows at the padded pitch.
    assert!(mlir.contains("strided<[144, 1]"), "{mlir}");
}

#[test]
fn store_writes_words_into_a_byte_run() {
    let src = |values: &str| {
        format!(
            "@launch(64)
            kernel w(A: tensor<i32>[64, 2], C: tensor<i8>[64, 8]) {{
                var t: tile<i8>[64, 8]
                let r = ctx.thread
                let lo = A[r, 0]
                let hi = A[r, 1]
                store(t[r, 0 :+ 8], {values})
                barrier()
                C[0 :+ 64, 0 :+ 8] = t
            }}"
        )
    };
    let mlir = emit_mlir(&src("lo, hi"));
    assert_contains(&mlir, &["vector<2xi32>", "vector.bitcast", "vector<8xi8>"]);
    let err = emit_err(&src("lo, hi, lo"));
    assert!(err.contains("a run of 8 takes 8 values, got 3"), "{err}");
}

/// An int8 projection whose activation is the loop's staged prefix, read
/// through slices, beside a byte tile the loop writes, under `@pipeline`.
const PIPELINED: &str = "@launch(128)
    @pipeline
    @aligned(M = 64, N = 64, K = 64)
    kernel pl(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
              W: tensor<i8>[N, K], WS: tensor<f32>[KB, N], C: tensor<f32>[M, N]) {
        let pm = ctx.block.x
        let pn = ctx.block.y
        var acc: tile<f32>[64, 64] = 0.0
        for kt in range(0, K, 64) {
            var at = A[pm * 64 :+ 64, kt :+ 64]
            var w: tile<i8>[64, 64]
            w = W[pn * 64 :+ 64, kt :+ 64]
            let a0 = at[0 :+ 64, 0 :+ 32]
            let w0 = w[0 :+ 64, 0 :+ 32]
            let da = AS[pm * 64 :+ 64, kt / 32 :+ 1]
            let ws = WS[kt / 32 :+ 1, pn * 64 :+ 64]
            acc += f32(dot_t(a0, w0)) * ws * da
        }
        C[pm * 64 :+ 64, pn * 64 :+ 64] = acc
    }";

#[test]
fn a_loop_carrying_registers_pipelines_on_request() {
    // On sm_80 at 64-bit indices the staged prefix is copied with cp.async
    // into a pair of buffers, and the accumulators ride the loop.
    let mlir = emit_mlir_sync(PIPELINED, "sm_80");
    assert_contains(&mlir, &["nvgpu.device_async_copy", "iter_args", "math.fma"]);
}

#[test]
fn without_cp_async_the_next_stage_rides_in_registers() {
    // On sm_75 the next stage is loaded inside the guarded branch before
    // the contraction and stored after it, over the one buffer, where
    // sm_80 copies into a pair.
    let mlir = emit_mlir_sync(PIPELINED, "sm_75");
    assert!(!mlir.contains("device_async_copy"), "{mlir}");
    let pair = tile_views(&emit_mlir_sync(PIPELINED, "sm_80"), "64x64xi8").len();
    assert!(tile_views(&mlir, "64x64xi8").len() < pair, "{mlir}");
    let fma = mlir.find("math.fma").expect("the contraction");
    let fetch = mlir[..fma].rfind("-> (vector<16xi8>").expect("the guarded loads yield their vectors");
    let store = mlir[fma..].find("vector.store").expect("the stores after it");
    assert!(fetch < fma && store > 0, "{mlir}");
}

#[test]
fn a_barrier_before_the_stage_reads_orders_the_stores() {
    // The refill of `at` lands before the next turn's `barrier()`, which
    // the reads of `at` follow, so the turn needs no barrier of its own
    // after the stores: the explicit one replaces it.
    let fenced = PIPELINED.replace("let a0 = at", "barrier()\n            let a0 = at");
    let count = |src: &str| emit_mlir_sync(src, "sm_75").matches("gpu.barrier").count();
    assert!(count(&fenced) <= count(PIPELINED));
}
