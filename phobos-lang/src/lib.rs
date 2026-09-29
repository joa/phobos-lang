pub mod ast;
pub mod codegen;
pub mod ir;
pub mod lexer;
pub mod parser;
pub(crate) mod shape;
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
    dump_source(code);
    if context.print_phases {
        println!("=== SOURCE ========================");
        println!("{code}");
        println!("===================================");
    }

    let kernels = parse(code)?;

    // A kernel that wants `ldmatrix` needs 64-bit indices for the whole module.
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
