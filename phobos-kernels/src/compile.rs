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
    compile_in(&context_for(shapes), source, what)
}

/// A compilation context with `shapes` bound as `@autotune` overrides.
fn context_for(shapes: &[Override<'_>]) -> Context {
    let mut ctx = Context::default();
    for &(name, value) in shapes {
        ctx.shape_overrides.insert(name.to_string(), value as i64);
    }
    ctx
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
    /// Compiles both the aligned and the masked-fallback text of one kernel
    /// source, substituting `{ALIGNED}` with `claims.0` / `claims.1`.
    ///
    /// A `@pipeline` assertion is checked across the pair rather than per
    /// compile, which is why this uses `compile_raw` instead of
    /// [`compile_shared`]. The fallback variant's slices are partial by
    /// construction, so it is never expected to pipeline; asserting per
    /// variant would fail on exactly the one that was never supposed to.
    /// Either variant pipelining satisfies the assertion.
    pub fn compile(
        source: &str,
        shapes: &[Override<'_>],
        what: &str,
        claims: (&str, &str),
    ) -> Result<Variants> {
        let ctx = context_for(shapes);
        let build = |claim: &str, which: &str| {
            phobos_lang::compile_raw(&ctx, &source.replace("{ALIGNED}", claim))
                .with_context(|| format!("compiling {what} ({which})"))
        };
        let aligned_out = build(claims.0, "aligned")?;
        let general_out = build(claims.1, "general")?;

        for (name, aligned_reasons) in &aligned_out.pipeline_failures {
            // A kernel missing from the general variant's failures pipelined
            // there, which satisfies the assertion for the pair.
            let Some((_, general_reasons)) = general_out
                .pipeline_failures
                .iter()
                .find(|(n, _)| n == name)
            else {
                continue;
            };
            let why = |reasons: &[String]| match reasons {
                [] => "no loop shaped for pipelining".to_string(),
                _ => reasons.join("; "),
            };
            anyhow::bail!(
                "kernel `{name}` in {what}: @pipeline asserts this kernel can be pipelined, but \
                 neither compiled variant did (aligned: {}; general: {})",
                why(aligned_reasons),
                why(general_reasons),
            );
        }

        let load = |out: &phobos_lang::CompileOutput, which: &str| {
            Module::from_ptx(&out.code, &[])
                .with_context(|| format!("loading {what} ({which}) PTX"))
        };
        Ok(Variants {
            aligned: load(&aligned_out, "aligned")?,
            general: load(&general_out, "general")?,
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
