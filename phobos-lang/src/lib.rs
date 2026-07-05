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
    if context.print_phases {
        println!("=== SOURCE ========================");
        println!("{code}");
        println!("===================================");
    }

    let kernels = parse(code)?;

    if context.print_phases {
        println!("=== AST ===========================");
        println!("{:?}", kernels.first().unwrap());
        println!("===================================");
    }

    let ptx = phobos_mlir::gen_ptx(context, |base, context, module| {
        codegen::emit(base, &kernels, context, module)
    })?;

    if context.print_phases {
        println!("=== PTX ===========================");
        println!("{ptx}");
        println!("===================================");
    }

    Ok(ptx)
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
