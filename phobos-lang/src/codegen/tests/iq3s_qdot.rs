// `iq3s_qdot_t`: see phobos-lang/src/codegen/tile/iq3s_qdot.rs.

use super::*;

const SRC: &str = "\
@launch(256)
@autotune(TN in [8])
@aligned(N = TN)
kernel iq3s_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                        D: tensor<f16>[N, NB], GRID: tensor<i32>[1, 2048],
                        SIGNS: tensor<i32>[1, 2048], C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq3s_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                         D[pn * TN :+ TN, :], GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}";

#[test]
fn iq3s_qdot_t_has_no_per_lane_staging() {
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
fn iq3s_qdot_t_reads_a_dynamic_contraction_length_at_runtime() {
    let mlir = emit_mlir(SRC);
    assert!(
        mlir.contains("memref.dim"),
        "a dynamic K should read its extent at runtime:\n{mlir}"
    );
}

#[test]
fn iq3s_qdot_t_rejects_a_ragged_contraction() {
    let src = "@launch(256)
        kernel iq3s_decode(A: tensor<f32>[1, 128], QB: tensor<i8>[8, 55],
                           D: tensor<f16>[8, 1], GRID: tensor<i32>[1, 2048],
                           SIGNS: tensor<i32>[1, 2048], C: tensor<f32>[1, 8]) {
            C[0 :+ 1, 0 :+ 8] = iq3s_qdot_t(A[0 :+ 1, :], QB[0 :+ 8, :], D[0 :+ 8, :], GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
        }";
    let err = std::panic::catch_unwind(|| emit_mlir(src));
    assert!(err.is_err(), "a 128-wide activation should be rejected");
}
