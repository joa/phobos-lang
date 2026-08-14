// `@launch` bounds and the nvvm attributes they lower to.

use super::*;

#[test]
fn launch_emits_nvvm_bounds() {
    let mlir = emit_mlir(
        "@launch(128, 2)
        kernel k(A: tensor<f32>[N]) { let i = program_id(0) }",
    );
    assert_contains(
        &mlir,
        &["nvvm.maxntid = array<i32: 128>", "nvvm.minctasm = 2"],
    );
}

#[test]
fn launch_min_blocks_optional() {
    let mlir = emit_mlir(
        "@launch(64)
        kernel k(A: tensor<f32>[N]) { let i = program_id(0) }",
    );
    assert_contains(&mlir, &["nvvm.maxntid = array<i32: 64>"]);
    assert!(
        !mlir.contains("nvvm.minctasm"),
        "unexpected minctasm without minBlocks:\n{mlir}"
    );
}

#[test]
fn launch_max_regs_emits_nvvm_maxnreg() {
    // The third @launch arg hard-caps registers per thread (PTX .maxnreg),
    // the lever for forcing occupancy when .minnctapersm is only advisory.
    let mlir = emit_mlir(
        "@launch(256, 2, 128)
        kernel k(A: tensor<f32>[N]) { let i = program_id(0) }",
    );
    assert_contains(
        &mlir,
        &[
            "nvvm.maxntid = array<i32: 256>",
            "nvvm.minctasm = 2",
            "nvvm.maxnreg = 128",
        ],
    );
}

#[test]
fn launch_max_regs_must_be_in_range() {
    let err = emit_err(
        "@launch(256, 2, 8)
        kernel k(A: tensor<f32>[N]) { let i = program_id(0) }",
    );
    assert!(
        err.contains("between 16 and 255"),
        "unexpected error: {err}"
    );
}

#[test]
fn no_launch_emits_no_bounds() {
    let mlir = emit_mlir("kernel k(A: tensor<f32>[N]) { let i = program_id(0) }");
    assert!(
        !mlir.contains("nvvm.maxntid")
            && !mlir.contains("nvvm.minctasm")
            && !mlir.contains("nvvm.maxnreg"),
        "unexpected launch bounds on a kernel without @launch:\n{mlir}"
    );
}

#[test]
fn launch_max_threads_must_be_warp_multiple() {
    let err = emit_err(
        "@launch(100)
        kernel k(A: tensor<f32>[N]) { let i = program_id(0) }",
    );
    assert!(err.contains("multiple of 32"), "unexpected error: {err}");
}
