use anyhow::{Context as _, Result};
use cust::module::Module;
use phobos_base::context::Context;
use phobos_base::progress::{self, Step};
use std::time::{Duration, Instant};

/// What compilation calls itself when it reports progress.
const STAGE: &str = "kernels";

/// Compile-time `@autotune` override.
pub type Override<'a> = (&'a str, usize);

/// One kernel's PTX plus the dynamic-shared-memory byte count each of its
/// `@dynshared` functions needs.
type CompiledSource = (String, Vec<(String, usize)>);

/// A lowering's result and what it cost. Timed inside the thread that ran it:
/// a batch lowers everything at once, so the wall time around a join says how
/// long the batch has been going rather than what this kernel took.
type Timed = (CompiledSource, Duration);

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
    let hit = crate::cache::load(ctx, source);
    let cached = hit.is_some();
    let (took, (ptx, shared)) = match hit {
        Some(hit) => (Duration::ZERO, hit),
        None => {
            progress::started(STAGE, what);
            let handle = spawn_compile(ctx, source)
                .with_context(|| format!("spawning the compile thread for {what}"))?;
            let (out, took) = join_compile(handle, what)?;
            crate::cache::store(ctx, source, &out.0, &out.1);
            (took, out)
        }
    };
    let module = Module::from_ptx(&ptx, &[]).with_context(|| format!("loading {what} PTX"))?;
    // A kernel asked for on its own is a batch of one: a caller watching gets
    // the same shape of report either way and never has to special-case it.
    progress::report(Step {
        stage: STAGE,
        item: what,
        done: 1,
        total: 1,
        cached,
        took,
        source,
        ptx: &ptx,
    });
    Ok((module, shared))
}

/// Spawns `phobos_lang::compile_shared` on its own stack (see
/// [`COMPILE_STACK_BYTES`]); does not join.
fn spawn_compile(
    ctx: &Context,
    source: &str,
) -> std::io::Result<std::thread::JoinHandle<Result<Timed>>> {
    let ctx = ctx.clone();
    let source = source.to_string();
    std::thread::Builder::new()
        .stack_size(COMPILE_STACK_BYTES)
        .spawn(move || {
            let at = Instant::now();
            let out = phobos_lang::compile_shared(&ctx, &source)?;
            Ok((out, at.elapsed()))
        })
}

fn join_compile(handle: std::thread::JoinHandle<Result<Timed>>, what: &str) -> Result<Timed> {
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
        Miss(std::thread::JoinHandle<Result<Timed>>, Context, String),
    }

    let slots = jobs
        .iter()
        .map(|&(source, shapes, what)| {
            let ctx = context_for(shapes);
            match crate::cache::load(&ctx, source) {
                Some((ptx, _)) => Ok(Slot::Hit(ptx)),
                None => {
                    // Said before the spawn, so the whole batch is named the
                    // moment it starts rather than as each one is joined.
                    progress::started(STAGE, what);
                    let handle = spawn_compile(&ctx, source)
                        .with_context(|| format!("spawning the compile thread for {what}"))?;
                    Ok(Slot::Miss(handle, ctx, source.to_string()))
                }
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let total = jobs.len();
    jobs.iter()
        .zip(slots)
        .enumerate()
        .map(|(i, (&(_, _, what), slot))| {
            let cached = matches!(slot, Slot::Hit(_));
            let (ptx, took) = match slot {
                Slot::Hit(ptx) => (ptx, Duration::ZERO),
                Slot::Miss(handle, ctx, source) => {
                    let ((ptx, shared), took) = join_compile(handle, what)?;
                    crate::cache::store(&ctx, &source, &ptx, &shared);
                    (ptx, took)
                }
            };
            let module =
                Module::from_ptx(&ptx, &[]).with_context(|| format!("loading {what} PTX"))?;
            // Every lowering started at once, and they are joined in the
            // order the jobs were given, so this counts how far along that
            // order it has got rather than how many threads have finished.
            // It only ever moves forward, and a job that finishes early is
            // not reported until the ones before it have.
            progress::report(Step {
                stage: STAGE,
                item: what,
                done: i + 1,
                total,
                cached,
                took,
                source: jobs[i].0,
                ptx: &ptx,
            });
            Ok(module)
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
    /// Uses `compile_raw` rather than [`compile_shared`] because `@pipeline` is
    /// checked across the pair, and the fallback variant's partial slices never
    /// pipeline on their own, so either variant satisfies the assertion.
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

        let hit = crate::cache::load_pair(&ctx, &aligned_src, &general_src);
        // The pair is one cache entry and one unit of work, so it is timed and
        // reported as one kernel; `what` names the kernel rather than either
        // of its two alignment variants.
        let cached = hit.is_some();
        let at = Instant::now();
        let (aligned_ptx, general_ptx) = match hit {
            Some(hit) => hit,
            None => {
                progress::started(STAGE, what);
                // A thread each, not one thread doing both: the two texts are
                // the same kernel under different alignment claims and neither
                // reads the other, so lowering them at once costs nothing but
                // a second stack. See COMPILE_STACK_BYTES for why that stack
                // is not the caller's.
                let lower = |ctx: Context, src: String, half: &'static str| {
                    std::thread::Builder::new()
                        .stack_size(COMPILE_STACK_BYTES)
                        .spawn(move || phobos_lang::compile_raw(&ctx, &src))
                        .with_context(|| format!("spawning the compile thread for {what} ({half})"))
                };
                let aligned_thread = lower(ctx.clone(), aligned_src.clone(), "aligned")?;
                let general_thread = lower(ctx.clone(), general_src.clone(), "general")?;
                let join = |handle: std::thread::JoinHandle<_>| {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("the compile thread for {what} panicked"))
                };
                let aligned_out = join(aligned_thread)?;
                let general_out = join(general_thread)?;
                let aligned_out =
                    aligned_out.with_context(|| format!("compiling {what} (aligned)"))?;
                let general_out =
                    general_out.with_context(|| format!("compiling {what} (general)"))?;

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

        progress::report(Step {
            stage: STAGE,
            item: what,
            done: 1,
            total: 1,
            cached,
            took: if cached { Duration::ZERO } else { at.elapsed() },
            source: &aligned_src,
            ptx: &aligned_ptx,
        });

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
