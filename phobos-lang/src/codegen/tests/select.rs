// `where`, `iota` and the per-element math: what a tile can compute without
// a host-built table.

use super::*;

#[test]
fn where_fuses_into_the_sweep() {
    let mlir = emit_mlir(
        "kernel k(X: tensor<f32>[16, 64], Y: tensor<f32>[16, 64]) {
            let x = X[0 :+ 16, 0 :+ 64]
            var y: tile<f32>[16, 64] = where(x > 0.0, x, 0.1 * x)
            Y[0 :+ 16, 0 :+ 64] = y
        }",
    );
    assert_contains(&mlir, &["arith.cmpf ogt", "arith.select"]);
}

#[test]
fn where_masks_padded_columns_with_iota() {
    let mlir = emit_mlir(
        "kernel k(S: tensor<f32>[16, 64], O: tensor<f32>[16, 1], n: i32) {
            var s: tile<f32>[16, 64] = S[0 :+ 16, 0 :+ 64]
            s = where(iota(1) < n && s > -1.0e30, s, -1.0e30)
            var m: tile<f32>[16, 1] = rowmax(s)
            O[0 :+ 16, 0 :+ 1] = m
        }",
    );
    assert_contains(
        &mlir,
        &[
            "arith.index_cast",
            "arith.cmpi slt",
            "arith.cmpf ogt",
            "arith.andi",
            "arith.select",
        ],
    );
}

#[test]
fn iota_alone_fills_positions() {
    let mlir = emit_mlir(
        "kernel k(P: tensor<i32>[16, 64]) {
            var p: tile<i32>[16, 64] = iota(0) * 64 + iota(1)
            P[0 :+ 16, 0 :+ 64] = p
        }",
    );
    assert_contains(&mlir, &["arith.index_cast", "arith.muli", "arith.addi"]);
}

#[test]
fn iota_past_the_rank_is_rejected() {
    let e = emit_err(
        "kernel k(P: tensor<i32>[64]) {
            var p: tile<i32>[64] = iota(1)
            P[0 :+ 64] = p
        }",
    );
    assert!(e.contains("`iota` axis 1 is past"), "got: {e}");
}

#[test]
fn iota_outside_a_tile_assignment_is_explained() {
    let e = emit_err("kernel k(A: tensor<i32>[N]) { let i = iota(0) }");
    assert!(e.contains("belongs in"), "got: {e}");
}

#[test]
fn scalar_where_selects() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[N]) {
            let i = ctx.block.x
            A[i] = where(i < 4, 1.0, A[i])
        }",
    );
    assert_contains(&mlir, &["arith.select"]);
}

#[test]
fn tile_comparison_outside_where_is_explained() {
    let e = emit_err(
        "kernel k(A: tensor<f32>[64]) {
            let a = A[0 :+ 64]
            var m: tile<f32>[64] = a
            m = a > a
        }",
    );
    assert!(e.contains("where(a > b, x, y)"), "got: {e}");
}

#[test]
fn math_on_tiles_and_scalars() {
    let mlir = emit_mlir(
        "kernel k(X: tensor<f32>[64], Y: tensor<f32>[64], S: tensor<f32>[N]) {
            let i = ctx.block.x
            let a = S[i]
            S[i] = abs(a) + floor(a) + ceil(a) + rsqrt(a) + sin(a) * cos(a) + exp(a)
            let x = X[0 :+ 64]
            var y: tile<f32>[64] = abs(x) + sin(x)
            y += floor(x)
            Y[0 :+ 64] = y
        }",
    );
    assert_contains(
        &mlir,
        &[
            "abs.f32",
            "cvt.rmi.f32.f32",
            "cvt.rpi.f32.f32",
            "rsqrt.approx.f32",
            "sin.approx.f32",
            "cos.approx.f32",
        ],
    );
}

#[test]
fn abs_of_an_integer_has_no_float_round_trip() {
    let mlir = emit_mlir(
        "kernel k(C: tensor<i32>[N]) {
            let i = ctx.block.x
            C[i] = abs(i - 7)
        }",
    );
    assert_contains(&mlir, &["arith.maxsi"]);
    assert!(
        !mlir.contains("abs.f32"),
        "integer abs went through a float:\n{mlir}"
    );
}

#[test]
fn math_needs_a_float() {
    let e = emit_err("kernel k(A: tensor<f32>[N]) { let i = ctx.block.x\n let x = sin(i) }");
    assert!(e.contains("`sin` needs a float operand"), "got: {e}");
}

#[test]
fn a_where_bound_keeps_its_width_on_a_narrow_tile() {
    // On an f16 tile the bound must not round to f16, which is exact only
    // to 2048: the comparison runs in f32 against the i32 position.
    let mlir = emit_mlir(
        "kernel k(S: tensor<f16>[16, 64], n: i32) {
            var s: tile<f16>[16, 64] = S[0 :+ 16, 0 :+ 64]
            s = where(iota(1) < n, s, 0.0)
            S[0 :+ 16, 0 :+ 64] = s
        }",
    );
    assert_contains(&mlir, &["arith.cmpi slt"]);
    assert!(
        !mlir.contains("arith.cmpf"),
        "the bound went through a float:\n{mlir}"
    );
}

#[test]
fn where_messages_name_the_real_problem() {
    let e = emit_err(
        "kernel k(A: tensor<f32>[64]) {
            let a = A[0 :+ 64]
            let m = where(a > 0.0, 1.0, 0.0)
        }",
    );
    assert!(e.contains("value of a tile assignment"), "got: {e}");
    let e = emit_err("kernel k(A: tensor<f32>[N]) { let x = where(missing < 1, 1.0, 0.0) }");
    assert!(e.contains("unknown identifier `missing`"), "got: {e}");
}

#[test]
fn an_untyped_var_takes_a_where() {
    let mlir = emit_mlir(
        "kernel k(X: tensor<f32>[16, 64], Y: tensor<f32>[16, 64]) {
            var x = X[0 :+ 16, 0 :+ 64]
            var y = where(x > 0.0, x, 0.1 * x)
            Y[0 :+ 16, 0 :+ 64] = y
        }",
    );
    assert_contains(&mlir, &["arith.cmpf ogt", "arith.select"]);
}

#[test]
fn pow_is_exp_of_a_scaled_log_on_tiles_and_scalars() {
    let mlir = emit_mlir(
        "kernel k(X: tensor<f32>[16, 64], Y: tensor<f32>[16, 64], S: tensor<f32>[N]) {
            let i = ctx.block.x
            S[i] = pow(S[i], 0.5)
            var x = X[0 :+ 16, 0 :+ 64]
            var y = pow(x, 2.2)
            Y[0 :+ 16, 0 :+ 64] = y
        }",
    );
    assert_contains(&mlir, &["ex2.approx", "lg2.approx"]);
}
