// A row reduction whose argument is a per-element tree evaluates it inside
// the reduction's loop, so the argument never takes a tile of its own.

use super::*;

/// `rowmax` or `rowsum` of `arg` over a `[16, 64]` tile `x` of `elem`.
fn reduce_kernel(elem: &str, reduce: &str, arg: &str) -> String {
    format!(
        "kernel k(X: tensor<{elem}>[16, 64], O: tensor<{elem}>[16, 1], n: i32) {{
            var x = X[0 :+ 16, 0 :+ 64]
            var m: tile<{elem}>[16, 1] = {reduce}({arg})
            O[0 :+ 16, 0 :+ 1] = m
        }}"
    )
}

/// Asserts `arg` reduces at the cost of reducing `x` itself, and returns
/// the module.
fn costs_like_a_tile(elem: &str, reduce: &str, arg: &str) -> String {
    let fused = emit_mlir(&reduce_kernel(elem, reduce, arg));
    let plain = emit_mlir(&reduce_kernel(elem, reduce, "x"));
    assert_eq!(shared_bytes(&fused), shared_bytes(&plain), "\n{fused}");
    let barriers = |m: &str| m.matches("gpu.barrier").count();
    assert_eq!(barriers(&fused), barriers(&plain), "\n{fused}");
    fused
}

#[test]
fn an_absmax_reads_no_tile_of_its_own() {
    let mlir = costs_like_a_tile("f32", "rowmax", "abs(x)");
    assert_contains(&mlir, &["abs.f32", "gpu.shuffle"]);
}

#[test]
fn a_masked_rowmax_fuses() {
    let mlir = costs_like_a_tile("f32", "rowmax", "where(iota(1) < n, x, -1.0e30)");
    assert_contains(&mlir, &["arith.cmpi slt", "arith.select"]);
}

#[test]
fn a_sum_of_squares_fuses_on_the_serial_path() {
    // f16 never takes the shuffle path.
    let mlir = costs_like_a_tile("f16", "rowsum", "x * x");
    assert_contains(&mlir, &["arith.mulf"]);
    assert!(!mlir.contains("gpu.shuffle"), "f16 took the warp path:\n{mlir}");
}

#[test]
fn a_reduction_of_a_slice_still_materializes() {
    // A tree over an unnamed slice is not typed up front, so it keeps the
    // tile path and still reduces correctly shaped.
    let mlir = emit_mlir(
        "kernel k(X: tensor<f32>[16, 64], O: tensor<f32>[16, 1]) {
            var m: tile<f32>[16, 1] = rowsum(X[0 :+ 16, 0 :+ 64] * 2.0)
            O[0 :+ 16, 0 :+ 1] = m
        }",
    );
    assert_contains(&mlir, &["arith.mulf", "arith.addf"]);
}
