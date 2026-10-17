// `@launch` bounds and the nvvm attributes they lower to.

use super::*;

#[test]
fn launch_emits_nvvm_bounds() {
    let mlir = emit_mlir(
        "@launch(128, 2)
        kernel k(A: tensor<f32>[N]) { let i = ctx.block.x }",
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
        kernel k(A: tensor<f32>[N]) { let i = ctx.block.x }",
    );
    assert_contains(&mlir, &["nvvm.maxntid = array<i32: 64>"]);
    assert!(
        !mlir.contains("nvvm.minctasm"),
        "unexpected minctasm without minBlocks:\n{mlir}"
    );
}

#[test]
fn launch_max_regs_emits_nvvm_maxnreg() {
    // The third @launch arg hard-caps registers per thread (PTX .maxnreg).
    // Unlike .minnctapersm, which is only advisory, it forces occupancy.
    let mlir = emit_mlir(
        "@launch(256, 2, 128)
        kernel k(A: tensor<f32>[N]) { let i = ctx.block.x }",
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
        kernel k(A: tensor<f32>[N]) { let i = ctx.block.x }",
    );
    assert!(
        err.contains("between 16 and 255"),
        "unexpected error: {err}"
    );
}

#[test]
fn no_launch_emits_no_bounds() {
    let mlir = emit_mlir("kernel k(A: tensor<f32>[N]) { let i = ctx.block.x }");
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
        kernel k(A: tensor<f32>[N]) { let i = ctx.block.x }",
    );
    assert!(err.contains("multiple of 32"), "unexpected error: {err}");
}

/// A kernel whose CTA size is an `@autotune` constant.
const TUNED_LAUNCH: &str = "@autotune(T in [64, 128])
    @launch(T, 2)
    kernel k(O: tensor<i32>[1, 1]) { O[0, 0] = i32(ctx.threads) }";

#[test]
fn launch_takes_autotune_constants() {
    assert_contains(&emit_mlir(TUNED_LAUNCH), &["nvvm.maxntid = array<i32: 64>", "nvvm.minctasm = 2"]);
    let base = phobos_base::context::Context {
        shape_overrides: [("T".to_string(), 128)].into(),
        ..Default::default()
    };
    let mlir = emit_mlir_base(TUNED_LAUNCH, &base);
    assert_contains(&mlir, &["nvvm.maxntid = array<i32: 128>", "arith.constant 128"]);
    let kernel = &crate::parse(TUNED_LAUNCH).unwrap()[0];
    assert_eq!(kernel.cta_threads(), Ok(64));
    assert_eq!(kernel.cta_threads_for(&base.shape_overrides), Ok(128));
}

#[test]
fn launch_names_only_autotune_constants() {
    let err = emit_err(
        "@launch(T)
        kernel k(A: tensor<f32>[N]) { let i = ctx.block.x }",
    );
    assert!(err.contains("`T` is not an `@autotune` constant"), "unexpected error: {err}");
}
