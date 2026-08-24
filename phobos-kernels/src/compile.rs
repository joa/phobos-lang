use anyhow::{Context as _, Result};
use cust::module::Module;
use phobos_base::context::Context;

/// Compile-time `@autotune` override.
pub type Override<'a> = (&'a str, usize);

/// One kernel's PTX plus the dynamic-shared-memory byte count each of its
/// `@dynshared` functions needs.
type CompiledSource = (String, Vec<(String, usize)>);

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

/// Stack size for the compile thread: MLIR-to-PTX lowering recurses with the
/// emitted IR's size, and a wide kernel like `q2k_matvec` can exceed a
/// thread's default (1 MiB on Windows, fixed at link time). `Module::from_ptx`
/// stays on the calling thread, where the CUDA context lives.
const COMPILE_STACK_BYTES: usize = 256 << 20;

pub fn compile_in(
    ctx: &Context,
    source: &str,
    what: &str,
) -> Result<(Module, Vec<(String, usize)>)> {
    let (ptx, shared) = match crate::cache::load(ctx, source) {
        Some(hit) => hit,
        None => {
            let handle = spawn_compile(ctx, source)
                .with_context(|| format!("spawning the compile thread for {what}"))?;
            let out = join_compile(handle, what)?;
            crate::cache::store(ctx, source, &out.0, &out.1);
            out
        }
    };
    let module = Module::from_ptx(&ptx, &[]).with_context(|| format!("loading {what} PTX"))?;
    Ok((module, shared))
}

/// Spawns `phobos_lang::compile_shared` on its own stack (see
/// [`COMPILE_STACK_BYTES`]); does not join.
fn spawn_compile(
    ctx: &Context,
    source: &str,
) -> std::io::Result<std::thread::JoinHandle<Result<CompiledSource>>> {
    let ctx = ctx.clone();
    let source = source.to_string();
    std::thread::Builder::new()
        .stack_size(COMPILE_STACK_BYTES)
        .spawn(move || phobos_lang::compile_shared(&ctx, &source))
}

fn join_compile(
    handle: std::thread::JoinHandle<Result<CompiledSource>>,
    what: &str,
) -> Result<CompiledSource> {
    handle
        .join()
        .map_err(|_| anyhow::anyhow!("the compile thread for {what} panicked"))?
        .with_context(|| format!("compiling {what}"))
}

/// [`compile`] for several independent sources at once: each source's
/// MLIR-to-PTX lowering runs on its own stack in parallel. Only the
/// PTX-to-`Module` load, which touches the CUDA driver, stays sequential on
/// the calling thread, after every lowering has finished.
pub fn compile_parallel(jobs: &[(&str, &[Override<'_>], &str)]) -> Result<Vec<Module>> {
    enum Slot {
        Hit(String),
        Miss(std::thread::JoinHandle<Result<CompiledSource>>, Context, String),
    }

    let slots = jobs
        .iter()
        .map(|&(source, shapes, what)| {
            let ctx = context_for(shapes);
            match crate::cache::load(&ctx, source) {
                Some((ptx, _)) => Ok(Slot::Hit(ptx)),
                None => {
                    let handle = spawn_compile(&ctx, source)
                        .with_context(|| format!("spawning the compile thread for {what}"))?;
                    Ok(Slot::Miss(handle, ctx, source.to_string()))
                }
            }
        })
        .collect::<Result<Vec<_>>>()?;

    jobs.iter()
        .zip(slots)
        .map(|(&(_, _, what), slot)| match slot {
            Slot::Hit(ptx) => {
                Module::from_ptx(&ptx, &[]).with_context(|| format!("loading {what} PTX"))
            }
            Slot::Miss(handle, ctx, source) => {
                let (ptx, shared) = join_compile(handle, what)?;
                crate::cache::store(&ctx, &source, &ptx, &shared);
                Module::from_ptx(&ptx, &[]).with_context(|| format!("loading {what} PTX"))
            }
        })
        .collect()
}

pub struct Variants {
    pub aligned: Module,
    pub general: Module,
}

impl Variants {
    /// Compiles both the aligned and the masked-fallback text of one kernel
    /// source, substituting `{ALIGNED}` with `claims.0` / `claims.1`.
    ///
    /// Uses `compile_raw` rather than [`compile_shared`] because `@pipeline`
    /// is checked across the pair: the fallback variant's slices are partial
    /// by construction and never expected to pipeline, so either variant
    /// pipelining satisfies the assertion.
    pub fn compile(
        source: &str,
        shapes: &[Override<'_>],
        what: &str,
        claims: (&str, &str),
    ) -> Result<Variants> {
        let ctx = context_for(shapes);
        let (aligned_src, general_src) = (
            source.replace("{ALIGNED}", claims.0),
            source.replace("{ALIGNED}", claims.1),
        );

        let (aligned_ptx, general_ptx) = match crate::cache::load_pair(&ctx, &aligned_src, &general_src)
        {
            Some(hit) => hit,
            None => {
                let ctx_owned = ctx.clone();
                let (aligned_owned, general_owned) = (aligned_src.clone(), general_src.clone());
                // See COMPILE_STACK_BYTES: same lowering, off the caller's stack.
                let (aligned_out, general_out) = std::thread::Builder::new()
                    .stack_size(COMPILE_STACK_BYTES)
                    .spawn(move || {
                        let aligned = phobos_lang::compile_raw(&ctx_owned, &aligned_owned);
                        let general = phobos_lang::compile_raw(&ctx_owned, &general_owned);
                        (aligned, general)
                    })
                    .with_context(|| format!("spawning the compile thread for {what}"))?
                    .join()
                    .map_err(|_| anyhow::anyhow!("the compile thread for {what} panicked"))?;
                let aligned_out = aligned_out.with_context(|| format!("compiling {what} (aligned)"))?;
                let general_out = general_out.with_context(|| format!("compiling {what} (general)"))?;

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

                crate::cache::store_pair(
                    &ctx,
                    &aligned_src,
                    &general_src,
                    &aligned_out.code,
                    &general_out.code,
                );
                (aligned_out.code, general_out.code)
            }
        };

        let load = |code: &str, which: &str| {
            Module::from_ptx(code, &[]).with_context(|| format!("loading {what} ({which}) PTX"))
        };
        Ok(Variants {
            aligned: load(&aligned_ptx, "aligned")?,
            general: load(&general_ptx, "general")?,
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
