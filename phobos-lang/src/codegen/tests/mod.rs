// Codegen tests.
//
// The shared harness lives here; each submodule covers one area of the
// emitter. See `emit_mlir` for the default target, which is sm_75.

mod basic;
mod dot;
mod iq1m_qdot;
mod iq1s_qdot;
mod iq2s_qdot;
mod iq2xs_qdot;
mod iq2xxs_qdot;
mod iq3s_qdot;
mod iq3xxs_qdot;
mod iq4xs_qdot;
mod launch;
mod mask;
mod math;
mod matmul;
mod narrow;
mod pipeline;
mod q2k_qdot;
mod q3k_qdot;
mod qdecode;
mod qgemm;
mod quant;
mod sync;
mod tensorcore;
mod tile;

use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{Location, Module, operation::OperationLike},
    utility::register_all_dialects,
};

fn emit_mlir(src: &str) -> String {
    emit_mlir_on(src, "sm_75")
}

fn emit_mlir_on(src: &str, chip: &str) -> String {
    use phobos_base::context::{Context as BaseContext, GpuConfig, NvidiaGpuConfig};
    emit_mlir_base(
        src,
        &BaseContext {
            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)),
            ..Default::default()
        },
    )
}

/// Like [`emit_mlir_on`] but with 64-bit index lowering, which the
/// @tensorcore mma.sync path requires (see Codegen::mma_sync).
fn emit_mlir_sync(src: &str, chip: &str) -> String {
    use phobos_base::context::{Context as BaseContext, GpuConfig, NvidiaGpuConfig};
    emit_mlir_base(
        src,
        &BaseContext {
            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)),
            index_bitwidth: 64,
            ..Default::default()
        },
    )
}

fn emit_mlir_base(src: &str, base: &phobos_base::context::Context) -> String {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let module = Module::new(Location::unknown(&context));
    let kernels = crate::parse(src).unwrap();
    super::emit(base, &kernels, &context, &module).unwrap();

    let text = module.as_operation().to_string();
    assert!(module.as_operation().verify(), "invalid module:\n{text}");
    text
}

/// emit_mlir already verifies; this just makes the intent of a test that
/// only cares about validity readable.
fn module_verifies(mlir: &str) -> bool {
    !mlir.is_empty()
}

fn assert_contains(mlir: &str, needles: &[&str]) {
    for needle in needles {
        assert!(mlir.contains(needle), "missing `{needle}` in:\n{mlir}");
    }
}

fn emit_err(src: &str) -> String {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    let module = Module::new(Location::unknown(&context));
    let kernels = crate::parse(src).unwrap();
    super::emit(
        &phobos_base::context::Context::default(),
        &kernels,
        &context,
        &module,
    )
    .expect_err("expected codegen to fail")
    .to_string()
}

/// Splits emitted IR at the flash kt loop (the only loop bounded by a
/// dynamic %dim) into (preheader, body) for staging-placement asserts. A
/// ragged-split loop is trimmed first and bounded by that arithmetic rather
/// than %dim directly, so either pattern is accepted, whichever comes first.
fn split_at_kt_loop(mlir: &str) -> (&str, &str) {
    let pos = [" = arith.subi %dim", " to %dim"]
        .iter()
        .filter_map(|pat| mlir.find(pat))
        .min()
        .expect("no dynamically-bounded loop in module");
    mlir.split_at(pos)
}
