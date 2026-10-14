// Staging into shared tiles and the barriers around them.

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
