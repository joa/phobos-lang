pub mod ast;
pub mod codegen;
pub mod contract;
pub mod diag;
pub mod ir;
pub mod lexer;
pub mod parser;
pub mod race;
mod report;
pub(crate) mod shape;
pub mod stdlib;
pub mod stubs;
pub mod token;
#[cfg(feature = "trace")]
pub mod trace;

pub use contract::Contract;
pub use report::{LaunchReport, ScratchParam, launch_report, launch_report_source};

/// Parses `src`. A failure is a [`diag::Diagnostics`] whose locations are
/// bare `line:col`; [`parse_source`] names the file.
pub fn parse(src: &str) -> anyhow::Result<Vec<ast::Kernel>> {
    parse_source(&diag::SourceFile::new("", src))
}

/// Parses `file` against the shipped standard library, failing with every
/// error found as [`diag::Diagnostics`].
pub fn parse_source(file: &std::sync::Arc<diag::SourceFile>) -> anyhow::Result<Vec<ast::Kernel>> {
    parse_source_with(file, &phobos_base::context::Stdlib::Builtin)
}

/// [`parse_source`] against the standard library `stdlib` selects.
pub fn parse_source_with(
    file: &std::sync::Arc<diag::SourceFile>,
    stdlib: &phobos_base::context::Stdlib,
) -> anyhow::Result<Vec<ast::Kernel>> {
    let prelude = stdlib::prelude(stdlib)?;
    let fail = |list| anyhow::Error::new(diag::Diagnostics::new(file.clone(), list));
    let toks = lexer::Lexer::new(file.text()).tokenize().map_err(fail)?;
    parser::parse_on_big_stack(|| {
        parser::Parser::new(toks, file.clone()).parse_program_with(&prelude)
    })
    .map_err(fail)
}

/// `file`'s kernels and functions as written, before anything is inlined
/// or lowered: what a printer of the source in another language reads.
pub fn parse_items(file: &std::sync::Arc<diag::SourceFile>) -> anyhow::Result<(Vec<ast::Kernel>, Vec<parser::Function>)> {
    let fail = |list| anyhow::Error::new(diag::Diagnostics::new(file.clone(), list));
    let toks = lexer::Lexer::new(file.text()).tokenize().map_err(fail)?;
    parser::parse_on_big_stack(|| parser::Parser::new(toks, file.clone()).parse_items()).map_err(fail)
}

/// Each `/` the kernels of `file` build under `context`, at the source
/// between its operands, and whether it divides integers there. A kernel
/// that fails to build gives the ones built before it failed.
pub fn divisions(context: &phobos_base::context::Context, file: &std::sync::Arc<diag::SourceFile>) -> Vec<(diag::Span, bool)> {
    let Ok(kernels) = parse_source_with(file, &context.stdlib) else {
        return Vec::new();
    };
    let target = codegen::target::build_target(context);
    let mut out = Vec::new();
    for kernel in &kernels {
        if let Ok((_, report)) = ir::build::build(context, target, kernel) {
            out.extend(report.divisions);
        }
    }
    out
}

/// Compiles the given code.
///
/// Uses the context's shape_overrides (autotuner).
pub fn compile(context: &phobos_base::context::Context, code: &str) -> anyhow::Result<String> {
    Ok(compile_shared(context, code)?.0)
}

/// Compiles the given code and reports the dynamic shared memory each
/// kernel needs at launch.
///
/// Only kernels marked `@dynshared` report a nonzero amount. The others use
/// static globals capped at 48 KB, and the launch ABI must respect this.
///
/// Fails if a kernel carries `@pipeline` and nothing in it pipelined. To
/// accept the assertion when any of several variants pipelines, use
/// [`compile_raw`] and aggregate `pipeline_failures` yourself.
pub fn compile_shared(
    context: &phobos_base::context::Context,
    code: &str,
) -> anyhow::Result<(String, Vec<(String, usize)>)> {
    let out = compile_raw(context, code)?;
    assert_pipelined(&out.pipeline_failures)?;
    Ok((out.code, out.shared))
}

/// Fails on the first kernel whose `@pipeline` assertion went unsatisfied,
/// as [`codegen::EmitOutput`] reports them.
pub fn assert_pipelined(failures: &[codegen::PipelineFailure]) -> anyhow::Result<()> {
    let Some(failure) = failures.first() else {
        return Ok(());
    };
    let why = if failure.reasons.is_empty() {
        "no loop in the kernel is shaped for pipelining".to_string()
    } else {
        failure.reasons.join("; ")
    };
    let message = format!("`@pipeline` found nothing in kernel `{}` to pipeline: {why}", failure.kernel);
    let d = diag::Diagnostic::error(message).primary(failure.span, "asserted here");
    Err(diag::Diagnostics::new(failure.source.clone(), vec![d]).into())
}

/// With `PHOBOS_DUMP_DIR` set, writes every compiled source there as a `.ph`
/// named by a hash of its text. `examples/snapshot.rs` can sweep the result.
fn dump_source(code: &str) {
    use std::hash::{Hash, Hasher};
    let Ok(dir) = std::env::var("PHOBOS_DUMP_DIR") else {
        return;
    };
    let mut h = std::collections::hash_map::DefaultHasher::new();
    code.hash(&mut h);
    let name = ast_kernel_name(code).unwrap_or("kernel");
    let dir = std::path::PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_ok() {
        let _ = std::fs::write(dir.join(format!("{name}_{:016x}.ph", h.finish())), code);
    }
}

/// The first kernel's name in a source, for the dump's file name. None when
/// no name is found; the compile itself reports the parse error.
fn ast_kernel_name(code: &str) -> Option<&str> {
    let at = code.find("kernel ")?;
    let rest = &code[at + "kernel ".len()..];
    let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    Some(&rest[..end])
}

/// [`compile_shared`] without its `@pipeline`-assertion enforcement: reports
/// every kernel's outcome instead of failing on the first unsatisfied one.
pub fn compile_raw(
    context: &phobos_base::context::Context,
    code: &str,
) -> anyhow::Result<CompileOutput> {
    compile_source(context, &diag::SourceFile::new("", code))
}

/// [`compile_raw`] of `file`, so an error names it.
pub fn compile_source(
    context: &phobos_base::context::Context,
    file: &std::sync::Arc<diag::SourceFile>,
) -> anyhow::Result<CompileOutput> {
    let code = file.text();
    dump_source(code);
    if context.print_phases {
        println!("=== SOURCE ========================");
        println!("{code}");
        println!("===================================");
    }

    let kernels = parse_source_with(file, &context.stdlib)?;

    let mut wide;
    let context = match index_bits(context, &kernels)? {
        bits if bits == context.index_bitwidth => context,
        bits => {
            wide = context.clone();
            wide.index_bitwidth = bits;
            &wide
        }
    };

    if context.print_phases {
        println!("=== AST ===========================");
        for kernel in &kernels {
            println!("{kernel:?}");
        }
        println!("===================================");
    }

    let out = std::cell::RefCell::new(None);
    let code = phobos_mlir::gen_code(context, |base, context, module| {
        *out.borrow_mut() = Some(codegen::emit(base, &kernels, context, module)?);
        Ok(())
    })?;

    let out = out.into_inner().expect("gen_code's closure always runs");
    let backend = context.gpu_config.backend();
    let code = backend.post_process(code, !out.shared.is_empty());

    if context.print_phases {
        let name = backend.code_name();
        println!("=== {name} ===========================");
        println!("{code}");
        println!("===================================");
    }

    Ok(CompileOutput {
        code,
        shared: out.shared,
        pipeline_failures: out.pipeline_failures,
        contracts: out.contracts,
    })
}

/// The launch contract of every kernel `file` compiles to under `context`,
/// widened as [`compile_source`] widens it, from the emitted module alone:
/// what a cached PTX was compiled beside, without lowering it again.
pub fn contracts(
    context: &phobos_base::context::Context,
    file: &std::sync::Arc<diag::SourceFile>,
) -> anyhow::Result<Vec<Contract>> {
    let kernels = parse_source_with(file, &context.stdlib)?;
    let mut wide = context.clone();
    wide.index_bitwidth = index_bits(context, &kernels)?;
    Ok(emit_unlowered(&wide, &kernels)?.contracts)
}

/// [`codegen::emit`] of `kernels` into a module of its own, never lowered.
fn emit_unlowered(
    context: &phobos_base::context::Context,
    kernels: &[ast::Kernel],
) -> anyhow::Result<codegen::EmitOutput> {
    use melior::dialect::DialectRegistry;
    use melior::ir::{Location, Module};
    use melior::utility::register_all_dialects;

    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let mlir = melior::Context::new();
    mlir.append_dialect_registry(&registry);
    mlir.load_all_available_dialects();
    let module = Module::new(Location::unknown(&mlir));
    codegen::emit(context, kernels, &mlir, &module)
}

/// Result of [`compile_raw`]: the generated code plus the sidebands
/// `codegen::EmitOutput` carries.
pub struct CompileOutput {
    pub code: String,
    pub shared: Vec<(String, usize)>,
    pub pipeline_failures: Vec<codegen::PipelineFailure>,
    pub contracts: Vec<Contract>,
}

/// The index width a module of `kernels` compiles at under `context`: what
/// a kernel's `@index_bits` asks for, else 64 where [`needs_wide_index`]
/// holds, and never narrower than the context asks. The kernels of a module
/// share one width, and asking for 32 fails where a kernel needs 64.
pub fn index_bits(context: &phobos_base::context::Context, kernels: &[ast::Kernel]) -> anyhow::Result<u32> {
    use diag::{Diagnostic, Diagnostics};
    let fail = |kernel: &ast::Kernel, d: Diagnostic| Diagnostics::new(kernel.source.clone(), vec![d]);
    let mut pinned: Option<(u32, &ast::Kernel, diag::Span)> = None;
    for kernel in kernels {
        let Some(attr) = kernel.attr("index_bits") else {
            continue;
        };
        let Some(bits) = kernel
            .index_bits_in(&kernel.shape_env(&context.shape_overrides))
            .map_err(|message| fail(kernel, Diagnostic::error(message).primary(attr.span, "")))?
        else {
            continue;
        };
        if let Some((first, other, at)) = pinned
            && first != bits
        {
            let d = Diagnostic::error("the kernels of one module share an index width")
                .primary(attr.span, format!("`{}` asks for {bits} bits", kernel.name))
                .secondary(at, format!("`{}` asks for {first} bits", other.name));
            return Err(fail(kernel, d).into());
        }
        pinned = Some((bits, kernel, attr.span));
    }
    let Some((bits, asking, at)) = pinned else {
        let auto = if needs_wide_index(context, kernels) { 64 } else { 32 };
        return Ok(auto.max(context.index_bitwidth));
    };
    if bits == 32
        && let Some((kernel, why)) = kernels.iter().find_map(|k| Some((k, k.wide_index_reason()?)))
    {
        let d = Diagnostic::error("`@index_bits(32)` on a module that needs 64-bit indices")
            .primary(at, "asks for 32 bits")
            .note(format!("in `{}`, {why}", kernel.name));
        return Err(fail(asking, d).into());
    }
    Ok(bits.max(context.index_bitwidth))
}

/// Whether a module of `kernels` is indexed at 64 bits on the context's
/// chip, whatever width was asked for, when no kernel says with
/// `@index_bits`. The `@tensorcore` path's `mma.sync`,
/// `ldmatrix` and a system-scope load or store need it on any chip. So does
/// `cp.async` from sm_80 on, which a module is widened for when, emitted at
/// 64 bits, it copies asynchronously; one that never does keeps the narrower
/// index, which is cheaper.
pub fn needs_wide_index(context: &phobos_base::context::Context, kernels: &[ast::Kernel]) -> bool {
    requires_wide_index(kernels)
        || kernels.iter().any(|k| k.wants_ldmatrix() || k.wants_system_scope())
        || (context.gpu_config.compute_capability() >= 80 && copies_async_when_wide(context, kernels))
}

fn copies_async_when_wide(context: &phobos_base::context::Context, kernels: &[ast::Kernel]) -> bool {
    let mut wide = context.clone();
    wide.index_bitwidth = 64;
    // A module that fails to emit fails again in the compile that follows,
    // which reports it.
    emit_unlowered(&wide, kernels).is_ok_and(|out| out.copies_async)
}

pub fn requires_wide_index(kernels: &[ast::Kernel]) -> bool {
    kernels.iter().any(|k| k.wants_mma_sync())
}

/// The @autotune search dims (name, choices) of a kernel.
pub fn search_space(kernel: &ast::Kernel) -> Vec<(String, Vec<i64>)> {
    kernel
        .attrs
        .iter()
        .filter(|a| a.name == "autotune")
        .flat_map(|a| a.args.iter())
        .filter_map(|arg| match arg {
            ast::AttrArg::Search { name, choices } => {
                Some((name.clone(), ast::search_choices(choices)))
            }
            _ => None,
        })
        .collect()
}

/// The @autotune search dims of a module, merged by name across its kernels.
/// A choice reaches every kernel declaring the name, so two kernels may share
/// one only with the same choices.
pub fn module_search_space(kernels: &[ast::Kernel]) -> anyhow::Result<Vec<(String, Vec<i64>)>> {
    let mut space: Vec<(String, Vec<i64>)> = Vec::new();
    for kernel in kernels {
        for (name, choices) in search_space(kernel) {
            match space.iter().find(|(n, _)| *n == name) {
                Some((_, seen)) => anyhow::ensure!(
                    *seen == choices,
                    "`@autotune` parameter `{name}` has choices {seen:?} in one kernel and {choices:?} in `{}`",
                    kernel.name
                ),
                None => space.push((name, choices)),
            }
        }
    }
    Ok(space)
}

/// `space` with each pinned dim narrowed to its pinned value, which need not
/// be one of its choices. A pin naming no dim of `space` is an error, since
/// codegen would ignore it.
pub fn pin(
    space: Vec<(String, Vec<i64>)>,
    pins: &std::collections::HashMap<String, i64>,
) -> anyhow::Result<Vec<(String, Vec<i64>)>> {
    for name in pins.keys() {
        anyhow::ensure!(
            space.iter().any(|(n, _)| n == name),
            "autotune pin `{name}` is not an `@autotune` parameter here (parameters: {})",
            space
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(space
        .into_iter()
        .map(|(name, choices)| match pins.get(&name) {
            Some(&v) => (name, vec![v]),
            None => (name, choices),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Scalar, Type},
        parse,
    };

    #[test]
    fn parses_tensor_kernel() {
        let p = parse(
            "kernel add(X: tensor<f32>[N], Y: tensor<f32>[N], Z: tensor<f32>[N]) {
                let i = ctx.block.x
                Z[i] = X[i] + Y[i]
             }",
        )
        .unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].name, "add");
        assert_eq!(p[0].params.len(), 3);
        assert!(matches!(p[0].params[0].ty, Type::Tensor(Scalar::F32, _)));
    }

    #[test]
    fn module_search_space_merges_kernels_and_pins() {
        let ks = parse(
            "@autotune(T in [16, 64], U in [1, 2, 4])
             kernel a(X: tensor<f32>[N]) { X[0] = 1.0 }
             @autotune(T in [16, 64], V in [8, 9, 10])
             kernel b(X: tensor<f32>[N]) { X[0] = 1.0 }",
        )
        .unwrap();
        let space = crate::module_search_space(&ks).unwrap();
        let names: Vec<_> = space.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["T", "U", "V"]);
        assert_eq!(space[0].1, [16, 32, 64]);

        let pinned = crate::pin(space.clone(), &[("U".to_string(), 3)].into()).unwrap();
        assert_eq!(pinned[1].1, [3]);
        assert!(crate::pin(space, &[("N".to_string(), 8)].into()).is_err());
    }

    #[test]
    fn module_search_space_rejects_a_name_with_two_meanings() {
        let ks = parse(
            "@autotune(T in [16, 64])
             kernel a(X: tensor<f32>[N]) { X[0] = 1.0 }
             @autotune(T in [8, 9, 10])
             kernel b(X: tensor<f32>[N]) { X[0] = 1.0 }",
        )
        .unwrap();
        assert!(crate::module_search_space(&ks).is_err());
    }
}
