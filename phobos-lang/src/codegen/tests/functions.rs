// Device functions: inlined by the parser, expression bodies substituted and
// statement bodies spliced.

use super::*;

#[test]
fn an_expression_function_fuses_like_its_body() {
    let body = "kernel k(G: tensor<f32>[4, 32], U: tensor<f32>[4, 32], O: tensor<f32>[4, 32]) {
            let g = G[0 :+ 4, 0 :+ 32]
            let u = U[0 :+ 4, 0 :+ 32]
            var y: tile<f32>[4, 32] = VALUE
            O[0 :+ 4, 0 :+ 32] = y
        }";
    let called = emit_mlir(&format!(
        "fn silu(x) = x / (1.0 + exp(-x))\n{}",
        body.replace("VALUE", "u * silu(g)")
    ));
    let written = emit_mlir(&body.replace("VALUE", "u * (g / (1.0 + exp(-g)))"));
    assert_eq!(called, written);
}

#[test]
fn a_statement_function_writes_the_tile_it_is_handed() {
    let mlir = emit_mlir(
        "fn clear_head(t, n) {
            var tmp = 0.0
            t[0 :+ n, 0 :+ 16] = tmp
        }
        kernel k(A: tensor<f32>[16, 16], O: tensor<f32>[16, 16]) {
            var tmp: tile<f32>[16, 16] = A[0 :+ 16, 0 :+ 16]
            clear_head(tmp, 4)
            O[0 :+ 16, 0 :+ 16] = tmp
        }",
    );
    assert_contains(&mlir, &["memref.subview"]);
}

#[test]
fn functions_nest_and_keep_their_own_loops() {
    let mlir = emit_mlir(
        "fn sq(x) = x * x
        fn first_big(A, n, out) {
            var found = -1
            for i in range(0, n) {
                if sq(A[i]) > 4.0 {
                    found = i
                    break
                }
            }
            out[0] = found
        }
        kernel k(A: tensor<f32>[N], O: tensor<i32>[1], n: i32) {
            first_big(A, n, O)
        }",
    );
    assert_contains(&mlir, &["scf.for", "arith.mulf"]);
}

#[test]
fn function_errors_name_the_problem() {
    let parse_err = |src: &str| crate::parse(src).unwrap_err().to_string();
    for (src, want) in [
        ("fn f(x) = x\nkernel k(A: tensor<f32>[N]) { let i = f(1, 2) }", "takes 1 argument(s), got 2"),
        ("fn f(x) = f(x)\nkernel k(A: tensor<f32>[N]) { let i = f(1) }", "is it recursive?"),
        ("fn f(x) { return }\nkernel k(A: tensor<f32>[N]) { f(1) }", "`return` would leave the function body"),
        ("fn f(x) { break }\nkernel k(A: tensor<f32>[N]) { f(1) }", "`break` would leave the function body"),
        ("fn f(x) { let y = x }\nkernel k(A: tensor<f32>[N]) { let i = f(1) }", "cannot be used as a value"),
        ("fn f(x) = x\nfn f(y) = y\nkernel k(A: tensor<f32>[N]) { let i = f(1) }", "defined twice"),
        ("fn f(t) { t[0] = 1.0 }\nkernel k(A: tensor<f32>[N]) { f(A[0 :+ 4]) }", "must be passed a name"),
    ] {
        let e = parse_err(src);
        assert!(e.contains(want), "{src}\ngot: {e}");
    }
}
