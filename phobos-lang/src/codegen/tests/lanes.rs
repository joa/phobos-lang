// Per-thread values: `ctx.thread`, `ctx.warp`, `ctx.lane`, and the rules
// that keep code reading them safe.

use super::*;

/// A matvec with a warp per column and its lanes splitting k, under control
/// that differs between warps.
const MATVEC: &str = "@launch(256)
kernel mv(A: tensor<f32>[1, K], W: tensor<f32>[N, K], C: tensor<f32>[1, N]) {
    let col = ctx.block.x * ctx.warps + ctx.warp
    if col < N {
        var acc: f32 = 0.0
        for k in range(ctx.lane, K, ctx.lanes) {
            acc += W[col, k] * A[0, k]
        }
        if ctx.lane == 0 {
            C[0, col] = acc
        }
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

/// `fold` of one `f32` per thread, written by thread 0.
fn folded(fold: &str) -> String {
    format!(
        "fn mx(a, b) = max(a, b)

        @launch(128)
        kernel k(A: tensor<f32>[1, 128], O: tensor<f32>[1, 1]) {{
            let x = A[0, ctx.thread]
            let r = {fold}
            if ctx.thread == 0 {{
                O[0, 0] = f32(r)
            }}
        }}"
    )
}

#[test]
fn a_lane_collective_is_a_butterfly_with_no_barrier() {
    for (fold, shuffles) in [
        ("sum<over ctx.lane>(x)", 5),
        ("max<over ctx.lane>(x)", 5),
        ("min<over ctx.lane, width 8>(x)", 3),
        ("reduce<over ctx.lane>(x, mx)", 5),
        // The value and its lane, one shuffle each per step.
        ("argmax<over ctx.lane>(x)", 10),
        ("where(any<over ctx.lane>(x > 0.0), 1.0, 0.0)", 5),
        ("shfl<from 3>(x)", 1),
        ("shfl<down 1, width 16>(x)", 1),
    ] {
        let mlir = emit_mlir(&folded(fold));
        assert_eq!(mlir.matches("gpu.shuffle").count(), shuffles, "{fold}\n{mlir}");
        assert!(!mlir.contains("gpu.barrier"), "{fold}\n{mlir}");
    }
    let mlir = emit_mlir(&folded("shfl<from 3>(x)"));
    assert_contains(&mlir, &["gpu.shuffle  idx"]);
}

#[test]
fn a_cta_collective_exchanges_each_warps_result() {
    let mlir = emit_mlir(&folded("sum<over ctx.thread>(x)"));
    assert_eq!(mlir.matches("gpu.shuffle").count(), 5, "\n{mlir}");
    // Before the scratch is written, after it is, and after it is read.
    assert_eq!(mlir.matches("gpu.barrier").count(), 3, "\n{mlir}");
    assert!(module_verifies(&mlir));
}

#[test]
fn a_full_warp_result_is_the_same_in_every_lane() {
    // A shuffle under control that depends on a warp's total is allowed,
    // since every lane of the warp agrees on it.
    let mlir = emit_mlir(&folded(
        "shfl<xor 1>(where(sum<over ctx.lane>(x) > 0.0, x, 0.0))",
    ));
    assert!(module_verifies(&mlir));
    let src = folded("0.0")
        .replace("let r = 0.0", "var r = 0.0\n if sum<over ctx.lane>(x) > 0.0 { r = shfl<xor 1>(x) }");
    assert!(module_verifies(&emit_mlir(&src)));
}

#[test]
fn collective_errors_name_the_problem() {
    for (src, want) in [
        (
            folded("0.0").replace("let r = 0.0", "var r = 0.0\n if ctx.lane == 0 { r = sum<over ctx.lane>(x) }"),
            "every lane of the warp",
        ),
        (
            folded("0.0").replace("let r = 0.0", "var r = 0.0\n if ctx.warp == 0 { r = sum<over ctx.thread>(x) }"),
            "every thread of the CTA",
        ),
        (folded("sum<over ctx.lane, width 3>(x)"), "power of two"),
        (folded("reduce<over ctx.lane>(x, nothing)"), "none is defined"),
        (folded("any<over ctx.lane>(x)"), "folds conditions"),
    ] {
        let err = emit_err(&src);
        assert!(err.contains(want), "{src}\ngot: {err}");
    }
    for (fold, want) in [
        ("sum(x)", "one thread's own slice"),
        ("sum<over ctx.warp>(x)", "takes `ctx.lane` or `ctx.thread`"),
        ("sum<width 8>(x)", "needs `over"),
        ("sum<over ctx.thread, width 8>(x)", "takes no `width`"),
        ("shfl<xor 1, down 2>(x)", "one of `from`"),
    ] {
        let src = folded(fold);
        match crate::parse(&src) {
            Err(e) => assert!(e.to_string().contains(want), "{fold}: {e}"),
            Ok(_) => {
                let err = emit_err(&src);
                assert!(err.contains(want), "{fold}: {err}");
            }
        }
    }
}

#[test]
fn the_settings_are_keywords_only_inside_the_brackets() {
    let mlir = emit_mlir(&folded("shfl<from 1>(x)").replace("let x =", "let from = 2\n let x = f32(from) * 0.0 +"));
    assert!(module_verifies(&mlir));
}

/// One thread's own `i8` slices of `a` and `w`, contracted by `body`.
fn lane_dot(body: &str) -> String {
    format!(
        "@launch(256)
        @aligned(K = 32)
        kernel k(A: tensor<i8>[1, K], W: tensor<i8>[N, K], C: tensor<f32>[1, N]) {{
            let col = ctx.block.x * ctx.warps + ctx.warp
            var acc: f32 = 0.0
            for k in range(ctx.lane * 16, K, ctx.lanes * 16) {{
                acc += f32({body})
            }}
            let total = sum<over ctx.lane>(acc)
            if ctx.lane == 0 {{
                C[0, col] = total
            }}
        }}"
    )
}

#[test]
fn a_thread_reads_its_own_slice_in_one_vector_load() {
    let mlir = emit_mlir(&lane_dot("sum(i32(A[0, k :+ 16]) * i32(W[col, k :+ 16]))"));
    // 16 bytes each, aligned since `k` steps by 16 and K by 32. The loop and
    // its split-off remainder each hold two such loads and four dp4a, and
    // no load of a single byte.
    let loads = mlir.matches("vector.load").count();
    assert!(loads > 0 && loads.is_multiple_of(2), "\n{mlir}");
    assert_contains(&mlir, &["{alignment = 16 : i64}"]);
    assert_eq!(mlir.matches("nvvm.dot.accumulate.4way").count(), 2 * loads, "\n{mlir}");
    assert!(!mlir.lines().any(|l| l.contains("memref.load") && l.contains("xi8, 1>")), "\n{mlir}");
}

#[test]
fn a_slice_with_no_alignment_proof_loads_per_element() {
    // `k + 1` is odd, so the run may straddle a 16-byte line.
    let mlir = emit_mlir(&lane_dot("sum(f32(A[0, k + 1 :+ 4]) * f32(W[col, k + 1 :+ 4]))"));
    assert!(!mlir.contains("vector.load"), "\n{mlir}");
    assert!(!mlir.contains("nvvm.dot.accumulate"), "\n{mlir}");
    let global_i8 = mlir.lines().filter(|l| l.contains("memref.load") && l.contains("xi8, 1>")).count();
    assert!(global_i8 > 0 && global_i8.is_multiple_of(8), "\n{mlir}");
}

#[test]
fn a_lane_sum_needs_a_slice() {
    let err = emit_err(&lane_dot("sum(acc)"));
    assert!(err.contains("one thread's own slice"), "got: {err}");
}

#[test]
fn byte_masks_and_run_sums_stay_on_dp4a() {
    // Low and high nibbles, and the sum of a run, each one dp4a per word.
    for body in [
        "sum(i32(W[col, k :+ 16] & 15) * i32(A[0, k :+ 16]))",
        "sum(i32((W[col, k :+ 16] >> 4) & 15) * i32(A[0, k :+ 16]))",
        "sum(i32(A[0, k :+ 16]))",
    ] {
        let mlir = emit_mlir(&lane_dot(body));
        let loads = mlir.matches("vector.load").count();
        assert!(loads > 0, "{body}\n{mlir}");
        assert!(mlir.contains("nvvm.dot.accumulate.4way"), "{body}\n{mlir}");
        assert!(!mlir.lines().any(|l| l.contains("memref.load") && l.contains("xi8, 1>")), "{body}\n{mlir}");
    }
    // A mask past 127 would read a byte as negative; that one is summed
    // per element instead.
    let mlir = emit_mlir(&lane_dot("sum((W[col, k :+ 16] & 255) * A[0, k :+ 16])"));
    assert!(!mlir.contains("nvvm.dot.accumulate"), "\n{mlir}");
}

#[test]
fn f16bits_reads_a_half_from_its_bits() {
    let mlir = emit_mlir(
        "@launch(32)
        kernel k(B: tensor<i8>[1, 2], O: tensor<f32>[1, 1]) {
            if ctx.lane == 0 {
                O[0, 0] = f32(f16bits((B[0, 0] & 255) | ((B[0, 1] & 255) << 8)))
            }
        }",
    );
    assert_contains(&mlir, &["arith.bitcast", "i16 to f16", "arith.extf"]);
}
