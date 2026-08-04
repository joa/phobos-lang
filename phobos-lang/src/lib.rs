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
pub fn compile_shared(
    context: &phobos_base::context::Context,
    code: &str,
) -> anyhow::Result<(String, Vec<(String, usize)>)> {
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

    let shared = std::cell::RefCell::new(Vec::new());
    let ptx = phobos_mlir::gen_ptx(context, |base, context, module| {
        *shared.borrow_mut() = codegen::emit(base, &kernels, context, module)?;
        Ok(())
    })?;

    let shared = shared.into_inner();
    let ptx = if shared.is_empty() {
        ptx
    } else {
        extern_dynamic_shared(&ptx)
    };

    if context.print_phases {
        println!("=== PTX ===========================");
        println!("{ptx}");
        println!("===================================");
    }

    Ok((ptx, shared))
}

/// Patch the demoted dynamic-shared declaration into an external one.
///
/// This is an issue with MLIR and the NVPTX backend we don't really care
/// about.
fn extern_dynamic_shared(ptx: &str) -> String {
    const NEWLINE: char = '\n';

    let mut declarations = Vec::new();
    let mut body = String::with_capacity(ptx.len());
    
    for line in ptx.lines() {
        let trimmed = line.trim();
        let demoted = trimmed
            .strip_prefix(".shared .align ")
            .filter(|rest| rest.contains("__dynamic_shmem__"));

        match demoted {
            Some(rest) => {
                let (align, name) = rest.split_once(" .b8 ").unwrap_or(("16", rest));
                let name = name.trim_end_matches(';');
                let declaration = format!(".extern .shared .align {align} .b8 {name}[];");

                if !declarations.contains(&declaration) {
                    declarations.push(declaration);
                }
            }
            None => {
                body.push_str(line);
                body.push(NEWLINE);
            }
        }
    }

    if declarations.is_empty() {
        return body;
    }
    
    let at = body
        .find(".address_size")
        .and_then(|i| body[i..].find(NEWLINE).map(|j| i + j + 1))
        .unwrap_or(0);

    let mut out = String::with_capacity(body.len() + 64);

    out.push_str(&body[..at]);
    
    for declaration in &declarations {
        out.push_str(declaration);
        out.push(NEWLINE);
    }
    
    out.push_str(&body[at..]);
    out
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
    #[test]
    fn dynamic_shared_is_declared_external_at_module_scope() {
        // The back end demotes the allocation into the kernel as a one-byte
        // static object; a launch-time size attaches only to the module-scope
        // external form.
        let ptx = concat!(
            ".version 9.0
.target sm_75
.address_size 64
",
            "// __dynamic_shmem__0 has been demoted
",
            ".visible .entry k(
)
{
",
            "	// demoted variable
",
            "	.shared .align 16 .b8 __dynamic_shmem__0;
",
            "	ret;
}
"
        );
        let out = super::extern_dynamic_shared(ptx);
        assert!(
            out.contains(".extern .shared .align 16 .b8 __dynamic_shmem__0[];"),
            "no external declaration in:
{out}"
        );
        assert!(
            !out.contains("	.shared .align 16 .b8 __dynamic_shmem__0;"),
            "the demoted definition survived:
{out}"
        );
        // Module scope: before the kernel that uses it.
        let declared = out.find(".extern .shared").expect("declaration");
        let entry = out.find(".visible .entry").expect("kernel");
        assert!(
            declared < entry,
            "declared after the kernel:
{out}"
        );
    }

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
