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
