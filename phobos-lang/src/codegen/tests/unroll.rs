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
