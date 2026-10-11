// Bitwise, logical and compound operators, and the accumulates and slices
// that had no spelling before.

use super::*;

#[test]
fn bitwise_and_shift_on_index_scalars() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<i32>[N], B: tensor<i32>[N]) {
            let i = program_id(0)
            let q = A[i]
            B[i] = ((q >> 4) & 15) | ((q << 2) ^ ~q)
        }",
    );
    assert_contains(
        &mlir,
        &[
            "arith.shrsi",
            "arith.andi",
            "arith.ori",
            "arith.shli",
            "arith.xori",
        ],
    );
}

#[test]
fn bitwise_on_integer_tiles_sweeps_once() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<i32>[64], B: tensor<i32>[64]) {
            var u: tile<i32>[64] = A[0 :+ 64]
            u = (u >> 4) & 15 | ~u
            B[0 :+ 64] = u
        }",
    );
    assert_contains(
        &mlir,
        &["arith.shrsi", "arith.andi", "arith.ori", "arith.xori"],
    );
}

#[test]
fn bitwise_on_a_float_is_rejected() {
    let e = emit_err("kernel k(A: tensor<f32>[N]) { let x = A[0] & 1.0 }");
    assert!(e.contains("`&` does not take f32"), "got: {e}");
    let e = emit_err(
        "kernel k(A: tensor<f32>[64]) {
            let t = A[0 :+ 64]
            var u: tile<f32>[64] = t << 1.0
        }",
    );
    assert!(e.contains("does not take float tile elements"), "got: {e}");
}

#[test]
fn logical_operators_join_conditions() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[N]) {
            let i = program_id(0)
            if i > 2 && i < 9 || i == 0 {
                A[i] = 1.0
            }
        }",
    );
    assert_contains(&mlir, &["arith.andi", "arith.ori", "scf.if"]);
}

#[test]
fn logical_operators_need_bools() {
    let e = emit_err("kernel k(A: tensor<f32>[N]) { let i = program_id(0)\n let x = i && i }");
    assert!(e.contains("`&&` does not take index"), "got: {e}");
}

#[test]
fn compound_assignment_on_scalars_and_tiles() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[64], B: tensor<f32>[64]) {
            var s = 1.0
            s -= 0.5
            s *= 3.0
            var t: tile<f32>[64] = A[0 :+ 64]
            t -= A[0 :+ 64]
            t *= s
            B[0 :+ 64] = t
        }",
    );
    assert_contains(&mlir, &["arith.subf", "arith.mulf"]);
}

#[test]
fn min_and_max_on_scalars_and_tiles() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[64], B: tensor<f32>[64], C: tensor<i32>[N]) {
            let i = program_id(0)
            C[i] = max(i, 3) - min(i, 7)
            var t: tile<f32>[64] = A[0 :+ 64]
            t = min(max(t, 0.0), 6.0)
            B[0 :+ 64] = t
        }",
    );
    assert_contains(
        &mlir,
        &[
            "arith.maxsi",
            "arith.minsi",
            "llvm.intr.maxnum",
            "llvm.intr.minnum",
        ],
    );
}

#[test]
fn tile_plus_eq_scalar() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[64], B: tensor<f32>[64]) {
            var t: tile<f32>[64] = A[0 :+ 64]
            let s = A[0]
            t += s
            t += 2
            B[0 :+ 64] = t
        }",
    );
    assert_contains(&mlir, &["arith.addf"]);
}

#[test]
fn integer_row_reductions() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<i32>[16, 64], R: tensor<i32>[16, 1]) {
            let q = A[0 :+ 16, 0 :+ 64]
            var r: tile<i32>[16, 1] = rowsum(q)
            r += rowmax(q)
            R[0 :+ 16, 0 :+ 1] = r
        }",
    );
    assert_contains(
        &mlir,
        &[
            "arith.addi",
            "arith.maxsi",
            "arith.constant -2147483648 : i32",
        ],
    );
}

/// An int8 `dot_t` accumulated in a loop, by the given statement.
fn dot_t_kernel(accumulate: &str) -> String {
    format!(
        "kernel k(A: tensor<i8>[16, 64], B: tensor<i8>[16, 64], C: tensor<i32>[16, 16]) {{
            var acc: tile<i32>[16, 16] = 0
            for r in range(0, 2) {{
                let a = A[0 :+ 16, 0 :+ 64]
                let b = B[0 :+ 16, 0 :+ 64]
                {accumulate}
            }}
            C[0 :+ 16, 0 :+ 16] = acc
        }}"
    )
}

#[test]
fn dot_t_accumulates_without_a_temp_tile() {
    let plus_eq = shared_bytes(&emit_mlir(&dot_t_kernel("acc += dot_t(a, b)")));
    let spelled = shared_bytes(&emit_mlir(&dot_t_kernel("acc = acc + dot_t(a, b)")));
    assert!(
        plus_eq < spelled,
        "+= took {plus_eq} bytes, the spelled-out sum {spelled}"
    );
}

#[test]
fn float_dot_t_accumulates() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[16, 32], B: tensor<f32>[16, 32], C: tensor<f32>[16, 16]) {
            var acc: tile<f32>[16, 16] = 0.0
            for r in range(0, 2) {
                let a = A[0 :+ 16, 0 :+ 32]
                let b = B[0 :+ 16, 0 :+ 32]
                acc += dot_t(a, b)
            }
            C[0 :+ 16, 0 :+ 16] = acc
        }",
    );
    assert_contains(&mlir, &["math.fma", "arith.addf"]);
}

#[test]
fn a_point_among_slices_keeps_its_axis() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[M, 64], B: tensor<f32>[M, 64]) {
            let i = program_id(0)
            var row: tile<f32>[1, 64] = A[i, 0 :+ 64]
            B[i, 0 :+ 64] = row
        }",
    );
    assert_contains(&mlir, &["memref<1x64xf32"]);
}

#[test]
fn open_ended_slice_runs_to_the_end() {
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[64], B: tensor<f32>[48]) {
            var t: tile<f32>[48] = A[16:]
            B[0:] = t
        }",
    );
    assert_contains(&mlir, &["memref<48xf32"]);
}

#[test]
fn open_ended_slice_past_the_extent_is_rejected() {
    let e = emit_err("kernel k(A: tensor<f32>[64]) { var t: tile<f32>[16] = A[80:] }");
    assert!(e.contains("slice start 80 is outside the extent 64"), "got: {e}");
}

#[test]
fn atomic_add_on_floats() {
    let mlir = emit_mlir(
        "kernel k(S: tensor<f32>[N], X: tensor<f32>[N]) {
            let i = program_id(0)
            let old = atomic_add(S, 0, X[i])
        }",
    );
    assert_contains(&mlir, &["memref.atomic_rmw addf"]);
}

#[test]
fn unknown_attribute_is_an_error() {
    let e = emit_err("@fast_math\nkernel k(A: tensor<f32>[N]) { let i = program_id(0) }");
    assert!(e.contains("unknown attribute @fast_math"), "got: {e}");
}

#[test]
fn tile_plus_eq_scalar_keeps_its_target_alive() {
    // `t` is still live while `u` is: the in-place add must not end its life.
    let mlir = emit_mlir(
        "kernel k(A: tensor<f32>[64], B: tensor<f32>[64]) {
            var t: tile<f32>[64] = A[0 :+ 64]
            t += 2.0
            var u: tile<f32>[64] = A[0 :+ 64]
            u = u * 3.0
            B[0 :+ 64] = t + u
        }",
    );
    let offsets = view_offsets(&mlir, "64xf32");
    let mut distinct = offsets.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() >= 2,
        "t and u share bytes: offsets {offsets:?}\n{mlir}"
    );
}

/// A SiLU-gated product over `[4, 32]` tiles, with `gate` as the exponent.
fn silu_kernel(gate: &str) -> String {
    format!(
        "kernel k(G: tensor<f32>[4, 32], U: tensor<f32>[4, 32], O: tensor<f32>[4, 32]) {{
            let g = G[0 :+ 4, 0 :+ 32]
            let u = U[0 :+ 4, 0 :+ 32]
            var y: tile<f32>[4, 32] = u / (1.0 + exp({gate}))
            O[0 :+ 4, 0 :+ 32] = y
        }}"
    )
}

#[test]
fn a_negated_tile_fuses_into_the_sweep() {
    // `-g` costs no tile and no barrier of its own.
    let negated = emit_mlir(&silu_kernel("-g"));
    let plain = emit_mlir(&silu_kernel("g"));
    assert_eq!(shared_bytes(&negated), shared_bytes(&plain), "
{negated}");
    let barriers = |m: &str| m.matches("gpu.barrier").count();
    assert_eq!(barriers(&negated), barriers(&plain), "
{negated}");
}
