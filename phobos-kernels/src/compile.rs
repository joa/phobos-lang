use anyhow::{Context as _, Result};
use cust::module::Module;
use phobos_base::context::Context;

/// Compile-time `@autotune` override.
pub type Override<'a> = (&'a str, usize);

pub fn compile(source: &str, shapes: &[Override<'_>], what: &str) -> Result<Module> {
    Ok(compile_shared(source, shapes, what)?.0)
}

pub fn compile_shared(
    source: &str,
    shapes: &[Override<'_>],
    what: &str,
) -> Result<(Module, Vec<(String, usize)>)> {
    let mut ctx = Context::default();
    for (name, value) in shapes {
        ctx.shape_overrides
            .insert((*name).to_string(), *value as i64);
    }
    compile_in(&ctx, source, what)
}

pub fn compile_in(
    ctx: &Context,
    source: &str,
    what: &str,
) -> Result<(Module, Vec<(String, usize)>)> {
    let (ptx, shared) =
        phobos_lang::compile_shared(ctx, source).with_context(|| format!("compiling {what}"))?;
    let module = Module::from_ptx(&ptx, &[]).with_context(|| format!("loading {what} PTX"))?;
    Ok((module, shared))
}

pub struct Variants {
    pub aligned: Module,
    pub general: Module,
}

impl Variants {
    pub fn compile(
        source: &str,
        shapes: &[Override<'_>],
        what: &str,
        claims: (&str, &str),
    ) -> Result<Variants> {
        Ok(Variants {
            aligned: compile(&source.replace("{ALIGNED}", claims.0), shapes, what)?,
            general: compile(&source.replace("{ALIGNED}", claims.1), shapes, what)?,
        })
    }

    pub fn pick(&self, aligned: bool) -> &Module {
        if aligned {
            &self.aligned
        } else {
            &self.general
        }
    }
}
