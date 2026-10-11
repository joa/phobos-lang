// `@registers`: a tile kept in each thread's registers across a loop.

use super::*;

/// A sum of squares over k, the accumulator named by `attrs`.
fn squares(attrs: &str) -> String {
    format!(
        "@launch(256)
        {attrs}
        kernel k(X: tensor<f32>[16, K], O: tensor<f32>[16, 64], S: tensor<f32>[16, 1]) {{
            var acc: tile<f32>[16, 64] = 0.0
            for kt in range(0, K, 64) {{
                var x = X[0 :+ 16, kt :+ 64]
                acc += x * x
            }}
            O[0 :+ 16, 0 :+ 64] = acc
            var s: tile<f32>[16, 1] = rowsum(acc)
            S[0 :+ 16, 0 :+ 1] = s
        }}"
    )
}

#[test]
fn a_register_accumulator_rides_the_loop() {
    let regs = emit_mlir(&squares("@registers(acc)"));
    let shared = emit_mlir(&squares(""));
    // 16 x 64 over 256 threads: four values per thread through the loop.
    assert!(regs.contains("iter_args"), "\n{regs}");
    assert!(
        regs.matches("gpu.barrier").count() < shared.matches("gpu.barrier").count(),
        "\n{regs}"
    );
}

#[test]
fn register_tile_errors_name_the_problem() {
    for (src, want) in [
        (
            "@launch(256) @registers(acc) kernel k(O: tensor<f32>[16, 64]) { var t: tile<f32>[16, 64] = 0.0 }",
            "names no `var acc",
        ),
        (
            "@launch(256) @registers(acc) kernel k(O: tensor<f32>[3, 5]) { var acc: tile<f32>[3, 5] = 0.0 }",
            "do not split evenly",
        ),
        (
            "@launch(256) @registers(acc) kernel k(O: tensor<f32>[16, 64], n: i32) {
                var acc: tile<f32>[16, 64] = 0.0
                if n > 0 {
                    acc += 1.0
                }
            }",
            "updated inside an `if`",
        ),
    ] {
        let e = emit_err(src);
        assert!(e.contains(want), "{src}\ngot: {e}");
    }
}
