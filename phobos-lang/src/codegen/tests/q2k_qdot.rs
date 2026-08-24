// `q2k_qdot_t`: see phobos-lang/src/codegen/tile/q2k_qdot.rs.

use super::*;

const SRC: &str = "\
@launch(256)
@autotune(TN in [32])
@aligned(N = TN)
kernel q2k_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                       D: tensor<f16>[N, NB], DMIN: tensor<f16>[N, NB],
                       C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = q2k_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                        D[pn * TN :+ TN, :], DMIN[pn * TN :+ TN, :])
}";

#[test]
fn q2k_qdot_t_has_no_per_lane_staging() {
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
fn q2k_qdot_t_reads_a_dynamic_contraction_length_at_runtime() {
    let mlir = emit_mlir(SRC);
    assert!(
        mlir.contains("memref.dim"),
        "a dynamic K should read its extent at runtime:\n{mlir}"
    );
}

#[test]
fn q2k_qdot_t_rejects_a_ragged_contraction() {
    let src = "@launch(256)
        kernel q2k_decode(A: tensor<f32>[1, 128], QB: tensor<i8>[32, 42],
                          D: tensor<f16>[32, 1], DMIN: tensor<f16>[32, 1],
                          C: tensor<f32>[1, 32]) {
            C[0 :+ 1, 0 :+ 32] = q2k_qdot_t(A[0 :+ 1, :], QB[0 :+ 32, :], D[0 :+ 32, :], DMIN[0 :+ 32, :])
        }";
    let err = std::panic::catch_unwind(|| emit_mlir(src));
    assert!(err.is_err(), "a 128-wide activation should be rejected");
}
