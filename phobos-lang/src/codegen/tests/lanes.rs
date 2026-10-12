// Per-thread values: `ctx.thread`, `ctx.warp`, `ctx.lane`, and the rules
// that keep code reading them safe.

use super::*;

/// A matvec with a warp per column and its lanes splitting k.
const MATVEC: &str = "@launch(256)
kernel mv(A: tensor<f32>[1, K], W: tensor<f32>[N, K], C: tensor<f32>[1, N]) {
    let col = ctx.block.x * ctx.warps + ctx.warp
    var acc: f32 = 0.0
    for k in range(ctx.lane, K, ctx.lanes) {
        acc += W[col, k] * A[0, k]
    }
    if ctx.lane == 0 {
        C[0, col] = acc
    }
}";

#[test]
fn a_lane_strided_loop_keeps_its_sum_per_thread() {
    let mlir = emit_mlir(MATVEC);
    assert_contains(&mlir, &["gpu.lane_id", "gpu.thread_id", "scf.if"]);
    // Nothing in it is a CTA-wide op, so nothing waits on a barrier.
    assert!(!mlir.contains("gpu.barrier"), "\n{mlir}");
}

/// `body` inside a kernel with a shared tile `t`, a tensor `O` and a
/// row-sized tensor `A`.
fn kernel(body: &str) -> String {
    format!(
        "@launch(256)
        kernel k(A: tensor<f32>[1, 32], O: tensor<f32>[1, 32]) {{
            var t: tile<f32>[1, 32] = A[0 :+ 1, 0 :+ 32]
            {body}
            O[0 :+ 1, 0 :+ 32] = t
        }}"
    )
}

#[test]
fn a_cta_wide_op_needs_every_thread_and_the_same_operands() {
    for (body, want) in [
        // Only lane 0 of each warp would reach the sweep's barrier.
        ("if ctx.lane == 0 { t = t * 2.0 }", "differs between lanes"),
        ("if ctx.warp == 0 { t = t * 2.0 }", "differs between warps"),
        // A fill whose value differs per thread.
        ("t = f32(ctx.lane)", "differs between lanes"),
        // A slice at a per-thread offset is not a tile.
        ("var u = A[0, ctx.lane :+ 1]", "differs between lanes"),
        // A `var` raised by a later iteration still counts.
        (
            "var x = 0
            for i in range(0, 4) {
                if x > 0 { t = t + 1.0 }
                x = ctx.lane
            }",
            "differs between lanes",
        ),
        // A lane-divergent break leaves the rest of the loop divergent.
        (
            "for i in range(0, 4) {
                if ctx.lane == i { break }
                t = t * 2.0
            }",
            "differs between lanes",
        ),
    ] {
        let err = emit_err(&kernel(body));
        assert!(err.contains(want), "{body}\ngot: {err}");
    }
}

#[test]
fn a_per_thread_store_never_meets_a_cta_read() {
    let err = emit_err(&kernel("t[0, ctx.lane] = 1.0"));
    assert!(err.contains("shared tile"), "got: {err}");
    let err = emit_err(&kernel("O[0, ctx.lane] = 1.0\n t = O[0 :+ 1, 0 :+ 32]"));
    assert!(err.contains("stored per thread"), "got: {err}");
}

#[test]
fn uniform_code_under_uniform_control_is_unchanged() {
    // A per-thread value that never reaches a CTA-wide op is fine, and so
    // is a CTA-wide op after a per-thread branch has closed.
    let mlir = emit_mlir(&kernel(
        "var x = 0
        if ctx.lane == 3 { x = 1 }
        t = t * 2.0",
    ));
    assert!(module_verifies(&mlir));
}
