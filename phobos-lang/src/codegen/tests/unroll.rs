// `for<unroll n>`: the body built `n` times a turn, plus a remainder loop.

use super::*;

/// A kernel summing `X` per thread with an unrolled loop of `n` and step `st`.
fn summed(n: &str, st: &str) -> String {
    format!(
        "@launch(32)
        kernel k(X: tensor<f32>[1, N], O: tensor<f32>[1, 32]) {{
            var acc: f32 = 0.0
            for<unroll {n}> i in range(ctx.thread, N, {st}) {{
                acc += X[0, i]
            }}
            O[0, ctx.thread] = acc
        }}"
    )
}

#[test]
fn an_unrolled_loop_is_a_main_loop_and_a_remainder() {
    let mlir = emit_mlir(&summed("4", "32"));
    assert!(module_verifies(&mlir), "{mlir}");
    // The main loop's body reads X four times, the remainder's once.
    assert_eq!(mlir.matches("scf.for").count(), 2, "{mlir}");
    assert_eq!(mlir.matches("memref.load").count() - mlir.matches("memref.load %alloca").count(), 5, "{mlir}");
}

#[test]
fn an_unroll_of_one_is_the_plain_loop() {
    assert_eq!(emit_mlir(&summed("1", "32")), emit_mlir(&summed("1", "32").replace("for<unroll 1>", "for")));
}

#[test]
fn unroll_errors_name_the_problem() {
    let err = emit_err(&summed("ctx.lane", "32"));
    assert!(err.contains("constant n"), "got: {err}");
    let err = emit_err(&summed("2", "ctx.lane + 1"));
    assert!(err.contains("constant positive step"), "got: {err}");
    let err = crate::parse(&summed("2", "32").replace("for<unroll 2>", "for<twice 2>")).unwrap_err();
    assert!(err.to_string().contains("takes `unroll`"), "got: {err}");
}

#[test]
fn the_unroll_factor_can_be_tuned() {
    let tuned = summed("U", "32").replace("@launch(32)", "@launch(32)
        @autotune(U in [4])");
    // The tuned constant builds what the literal does.
    assert_eq!(emit_mlir(&tuned), emit_mlir(&summed("4", "32")));
}

/// A register matmul whose k-loop is `for<unroll n>`, f32 or on the tensor
/// cores in f16.
fn matmul(n: &str, attrs: &str, elem: &str) -> String {
    format!(
        "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
        {attrs}
        @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
        kernel matmul(A: tensor<{elem}>[M, K], B: tensor<{elem}>[K, N], C: tensor<{elem}>[M, N]) {{
            let pm = ctx.block.x
            let pn = ctx.block.y
            var acc: tile<{elem}>[TILE_M, TILE_N] = 0.0
            for<unroll {n}> kt in range(0, K, TILE_K) {{
                var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                acc += dot(a, b)
            }}
            C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
        }}"
    )
}

#[test]
fn a_register_matmul_unrolled_once_is_the_plain_loop() {
    for (attrs, elem) in [("@pipeline", "f32"), ("", "f32"), ("@pipeline\n@tensorcore", "f16")] {
        let src = matmul("1", attrs, elem);
        assert_eq!(emit_mlir(&src), emit_mlir(&src.replace("for<unroll 1>", "for")), "{attrs} {elem}");
    }
}

#[test]
fn a_pipelined_register_matmul_runs_an_even_number_of_halves_a_turn() {
    let ifs = |n: &str| emit_mlir(&matmul(n, "@pipeline", "f32")).matches("scf.if").count();
    // Each half adds the guard that skips it. Without cp.async its
    // prefetch is clamped rather than guarded.
    assert_eq!(ifs("4"), ifs("1") + 2);
    // Two halves a turn already, and an odd count rounds up.
    assert_eq!(ifs("2"), ifs("1"));
    assert_eq!(emit_mlir(&matmul("3", "@pipeline", "f32")), emit_mlir(&matmul("4", "@pipeline", "f32")));
    let mlir = emit_mlir_sync(&matmul("4", "@pipeline\n@tensorcore", "f16"), "sm_80");
    assert!(module_verifies(&mlir), "{mlir}");
}

#[test]
fn an_unpipelined_register_matmul_builds_its_mac_n_times_a_turn() {
    let macs = |n: &str| emit_mlir(&matmul(n, "", "f32")).matches("vector.contract").count();
    assert_eq!(macs("3"), 3 * macs("1"));
}

#[test]
fn an_unrolled_loop_says_why_it_did_not_pipeline() {
    let src = summed("2", "32").replace("@launch(32)", "@launch(32)\n        @pipeline");
    let err = crate::compile_shared(&phobos_base::context::Context::default(), &src).unwrap_err().to_string();
    assert!(err.contains("pipelines only as a register matmul"), "got: {err}");
}
