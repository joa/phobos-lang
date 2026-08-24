// `iq1s_qdot_t`: see phobos-lang/src/codegen/tile/iq1s_qdot.rs.

use super::*;

const SRC: &str = "\
@launch(256)
kernel iq1s_decode(A: tensor<f32>[1, 256], QB: tensor<i8>[8, 50],
                   D: tensor<f16>[8, 1], GRID: tensor<i32>[1, 16384],
                   C: tensor<f32>[1, 8]) {
  C[0 :+ 1, 0 :+ 8] = iq1s_qdot_t(A[0 :+ 1, :], QB[0 :+ 8, :], D[0 :+ 8, :], GRID[0 :+ 1, :])
}";

#[test]
fn iq1s_qdot_t_has_no_per_lane_staging() {
    let mlir = emit_mlir(SRC);
    assert_eq!(
        mlir.matches("memref.global").count(),
        1,
        "a per-lane decode should stage nothing beyond the output tile:\n{mlir}"
    );
    assert_eq!(
        mlir.matches("gpu.shuffle").count(),
        5,
        "expected a 5-step warp-shuffle reduction:\n{mlir}"
    );
    assert!(
        mlir.contains("scf.for"),
        "expected the per-block contraction loop:\n{mlir}"
    );
}

#[test]
fn iq1s_qdot_t_rejects_a_ragged_contraction() {
    let src = SRC.replace("A: tensor<f32>[1, 256]", "A: tensor<f32>[1, 128]")
        .replace("QB: tensor<i8>[8, 50]", "QB: tensor<i8>[8, 25]");
    let err = std::panic::catch_unwind(|| emit_mlir(&src));
    assert!(err.is_err(), "a 128-wide activation should be rejected");
}

#[test]
fn iq1s_qdot_t_rejects_a_non_i32_grid() {
    let src = SRC.replace("GRID: tensor<i32>[1, 16384]", "GRID: tensor<i8>[1, 16384]");
    let err = std::panic::catch_unwind(|| emit_mlir(&src));
    assert!(err.is_err(), "a non-i32 grid table should be rejected");
}
