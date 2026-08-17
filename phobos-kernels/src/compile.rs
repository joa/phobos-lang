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
    /// Compiles both the aligned and the masked-fallback text of one kernel
    /// source, substituting `{ALIGNED}` with `claims.0` / `claims.1`
    /// respectively.
    ///
    /// The masked-fallback variant's slices are partial by construction --
    /// that is what makes it the fallback -- so `pipeline_candidate`
    /// (phobos-lang) correctly declines to pipeline it. If `@pipeline` sits
    /// in the shared part of `source` (outside the `{ALIGNED}` slot, so both
    /// variants carry it textually), a naive per-compile assertion would
    /// then fail to compile on exactly the variant that was never supposed
    /// to pipeline. Instead of enforcing the assertion inside each
    /// individual compile (`phobos_lang::compile_shared`, which is what
    /// every other, non-`Variants` caller here still uses), this goes
    /// through `compile_raw` for both variants and treats the assertion as
    /// satisfied kernel-source-wide if *either* variant pipelined --
    /// checked once, after both compiles, not once per variant.
    pub fn compile(
        source: &str,
        shapes: &[Override<'_>],
        what: &str,
        claims: (&str, &str),
    ) -> Result<Variants> {
        let mut ctx = Context::default();
        for (name, value) in shapes {
            ctx.shape_overrides
                .insert((*name).to_string(), *value as i64);
        }

        let aligned_src = source.replace("{ALIGNED}", claims.0);
        let general_src = source.replace("{ALIGNED}", claims.1);

        let aligned_out = phobos_lang::compile_raw(&ctx, &aligned_src)
            .with_context(|| format!("compiling {what} (aligned)"))?;
        let general_out = phobos_lang::compile_raw(&ctx, &general_src)
            .with_context(|| format!("compiling {what} (general)"))?;

        for (name, aligned_reasons) in &aligned_out.pipeline_failures {
            let Some((_, general_reasons)) = general_out
                .pipeline_failures
                .iter()
                .find(|(n, _)| n == name)
            else {
                continue; // the general variant pipelined; assertion satisfied.
            };
            anyhow::bail!(
                "kernel `{name}` in {what}: @pipeline asserts this kernel can be pipelined, but \
                 neither compiled variant did (aligned: {}; general: {})",
                if aligned_reasons.is_empty() {
                    "no loop shaped for pipelining".to_string()
                } else {
                    aligned_reasons.join("; ")
                },
                if general_reasons.is_empty() {
                    "no loop shaped for pipelining".to_string()
                } else {
                    general_reasons.join("; ")
                },
            );
        }

        let aligned = Module::from_ptx(&aligned_out.code, &[])
            .with_context(|| format!("loading {what} (aligned) PTX"))?;
        let general = Module::from_ptx(&general_out.code, &[])
            .with_context(|| format!("loading {what} (general) PTX"))?;

        Ok(Variants { aligned, general })
    }

    pub fn pick(&self, aligned: bool) -> &Module {
        if aligned {
            &self.aligned
        } else {
            &self.general
        }
    }
}
