// u8, u16 and u32: storage types that every read zero-extends.

use super::*;

#[test]
fn a_scalar_load_zero_extends_to_an_index() {
    let mlir = emit_mlir(
        "kernel k(B: tensor<u8>[N], O: tensor<i32>[N]) {
            let i = program_id(0)
            O[i] = B[i] + 1
        }",
    );
    assert_contains(&mlir, &["arith.index_castui"]);
}

#[test]
fn a_cast_of_an_unsigned_tile_fuses_and_zero_extends() {
    let mlir = emit_mlir(
        "kernel k(B: tensor<u8>[16, 64], O: tensor<i32>[16, 64]) {
            var b = B[0 :+ 16, 0 :+ 64]
            var q: tile<i32>[16, 64] = (i32(b) >> 4) & 15
            O[0 :+ 16, 0 :+ 64] = q
        }",
    );
    assert_contains(&mlir, &["arith.extui", "arith.shrsi", "arith.andi"]);
    assert!(!mlir.contains("arith.extsi"), "a u8 sign-extended:\n{mlir}");
}

#[test]
fn every_tile_conversion_zero_extends() {
    for (decl, op) in [
        ("var q: tile<i32>[16, 64] = B[0 :+ 16, 0 :+ 64]", "arith.extui"),
        ("var q = f32(b)", "arith.uitofp"),
        ("var q: tile<f32>[16, 64] = f32(b) * 0.5", "arith.uitofp"),
    ] {
        let mlir = emit_mlir(&format!(
            "kernel k(B: tensor<u16>[16, 64], O: tensor<f32>[16, 64]) {{
                var b = B[0 :+ 16, 0 :+ 64]
                {decl}
            }}"
        ));
        assert_contains(&mlir, &[op]);
        assert!(!mlir.contains("arith.extsi") && !mlir.contains("arith.sitofp"), "{decl}:\n{mlir}");
    }
}

#[test]
fn arithmetic_on_an_unsigned_tile_points_at_the_conversion() {
    let e = emit_err(
        "kernel k(B: tensor<u8>[16, 64], O: tensor<u8>[16, 64]) {
            var b = B[0 :+ 16, 0 :+ 64]
            var c = b & 15
        }",
    );
    assert!(e.contains("a u8 tile is storage: convert it first"), "got: {e}");
}

#[test]
fn an_unsigned_tile_stores_and_copies() {
    let mlir = emit_mlir(
        "kernel k(B: tensor<u32>[16, 64], O: tensor<u32>[16, 64]) {
            var b = B[0 :+ 16, 0 :+ 64]
            O[0 :+ 16, 0 :+ 64] = b
        }",
    );
    assert_contains(&mlir, &["memref<16x64xi32"]);
}
