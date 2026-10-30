// `break`, `continue` and `return`, which the parser lowers to uniform flags.

use super::*;

/// A search loop: the first `i` with `A[i] > t`, written to `O[0]`.
const FIND: &str = "
kernel find(A: tensor<f32>[N], O: tensor<i32>[1], t: f32, n: i32) {
    var found = -1
    var i = 0
    while i < n {
        if A[i] > t {
            found = i
            break
        }
        i = i + 1
    }
    O[0] = found
}
";

#[test]
fn break_ends_a_while_through_its_condition() {
    let mlir = emit_mlir(FIND);
    assert_contains(&mlir, &["scf.while", "arith.andi", "arith.xori"]);
}

#[test]
fn continue_skips_the_rest_of_an_iteration() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[N], n: i32) {
            for i in range(0, n) {
                if A[i] < 0.0 {
                    continue
                }
                A[i] = A[i] * 2.0
            }
        }",
    );
    // The store sits under the flag's guard, inside the loop.
    assert_contains(&mlir, &["scf.for", "scf.if", "arith.mulf"]);
}

#[test]
fn return_skips_the_rest_of_the_kernel_and_its_loops() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[N], n: i32) {
            let p = ctx.block.x
            if p >= n {
                return
            }
            for i in range(0, n) {
                if i == 2 {
                    return
                }
                A[p] = A[p] + 1.0
            }
            A[p] = 0.0
        }",
    );
    assert_contains(&mlir, &["scf.for", "scf.if", "arith.addf"]);
}

#[test]
fn a_jump_outside_a_loop_is_an_error() {
    let e = crate::parse("kernel k(A: tensor<f32>[N]) { break }").unwrap_err().to_string();
    assert!(e.contains("`break` outside a loop"), "got: {e}");
    let e = crate::parse("kernel k(A: tensor<f32>[N]) { if true { continue } }").unwrap_err().to_string();
    assert!(e.contains("`continue` outside a loop"), "got: {e}");
}

#[test]
fn a_kernel_without_jumps_is_untouched() {
    let src = "kernel k(A: tensor<f32>[N], n: i32) {
        for i in range(0, n) {
            A[i] = 1.0
        }
    }";
    assert!(!emit_mlir(src).contains("memref<i1>"));
}

#[test]
fn flags_reach_ptx() {
    for (chip, index_bits) in [("sm_75", 32), ("sm_120", 64)] {
        let code = crate::compile_raw(&base(chip, index_bits), FIND).unwrap().code;
        assert!(code.contains(".entry find"), "{chip}:\n{code}");
    }
}
