pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod token;

pub fn parse(src: &str) -> anyhow::Result<Vec<ast::Kernel>> {
    let toks = lexer::Lexer::new(src)
        .tokenize()
        .map_err(|e| anyhow::anyhow!(e))?;
    parser::Parser::new(toks)
        .parse_program()
        .map_err(|e| anyhow::anyhow!(e))
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
/// Note: Must use `@dynshared` for this to have any effect. Kernels not annotated
///       with dynamic shared memory will use static globals with a cap at 48 KB.
///       Shared memory must be respected in the launch ABI.
///
/// Enforces every kernel's `@pipeline` assertion: fails if a kernel wrote the
/// attribute and nothing in it pipelined. A caller that compiles several
/// variants of one source and accepts the assertion if any of them pipelines
/// should use [`compile_raw`] and aggregate `pipeline_failures` itself.
pub fn compile_shared(
    context: &phobos_base::context::Context,
    code: &str,
) -> anyhow::Result<(String, Vec<(String, usize)>)> {
    let out = compile_raw(context, code)?;
    if let Some((name, reasons)) = out.pipeline_failures.first() {
        let why = if reasons.is_empty() {
            "no loop in the kernel is shaped for pipelining".to_string()
        } else {
            reasons.join("; ")
        };
        anyhow::bail!(
            "kernel `{name}`: @pipeline asserts this kernel can be pipelined, but nothing in it \
             did: {why}"
        );
    }
    Ok((out.code, out.shared))
}

/// [`compile_shared`] without its `@pipeline`-assertion enforcement: reports
/// every kernel's outcome instead of failing on the first unsatisfied one.
pub fn compile_raw(
    context: &phobos_base::context::Context,
    code: &str,
) -> anyhow::Result<CompileOutput> {
    if context.print_phases {
        println!("=== SOURCE ========================");
        println!("{code}");
        println!("===================================");
    }

    let kernels = parse(code)?;

    // `Kernel::wants_ldmatrix`: 64-bit indices for the whole module.
    let mut wide;
    let context = if context.index_bitwidth < 64 && kernels.iter().any(ast::Kernel::wants_ldmatrix)
    {
        wide = context.clone();
        wide.index_bitwidth = 64;
        &wide
    } else {
        context
    };

    if context.print_phases {
        println!("=== AST ===========================");
        println!("{:?}", kernels.first().unwrap());
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
    })
}

/// Result of [`compile_raw`]: the generated code plus the sidebands
/// `codegen::EmitOutput` carries.
pub struct CompileOutput {
    pub code: String,
    pub shared: Vec<(String, usize)>,
    pub pipeline_failures: Vec<(String, Vec<String>)>,
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
                let i = program_id(0)
                Z[i] = X[i] + Y[i]
             }",
        )
        .unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].name, "add");
        assert_eq!(p[0].params.len(), 3);
        assert!(matches!(p[0].params[0].ty, Type::Tensor(Scalar::F32, _)));
    }
}
