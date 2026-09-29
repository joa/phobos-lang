use anyhow::{Context as _, Result};
use phobos_base::context::Context;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// One kernel's PTX plus the dynamic-shared-memory byte count each of its
/// `@dynshared` functions needs.
pub type CompiledSource = (String, Vec<(String, usize)>);

/// A lowering's result and the time it took, measured inside its own thread.
/// A batch lowers everything at once, so wall time around a join would
/// measure the batch, not this kernel.
pub type Timed = (CompiledSource, Duration);

/// Stack size for the compile thread. MLIR-to-PTX lowering recurses deeper
/// the larger the emitted IR, and a wide kernel like `q2k_matvec` can exceed
/// the default stack (1 MiB on Windows).
const COMPILE_STACK_BYTES: usize = 256 << 20;

/// Spawns `phobos_lang::compile_shared` on its own stack (see
/// [`COMPILE_STACK_BYTES`]); does not join.
pub fn spawn(ctx: &Context, source: &str) -> std::io::Result<JoinHandle<Result<Timed>>> {
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

pub fn join(handle: JoinHandle<Result<Timed>>, what: &str) -> Result<Timed> {
    handle
        .join()
        .map_err(|_| anyhow::anyhow!("the compile thread for {what} panicked"))?
        .with_context(|| format!("compiling {what}"))
}

/// Lowers one source to PTX and waits for it.
pub fn single(ctx: &Context, source: &str, what: &str) -> Result<Timed> {
    let handle =
        spawn(ctx, source).with_context(|| format!("spawning the compile thread for {what}"))?;
    join(handle, what)
}

/// Lowers the aligned and general texts of one kernel to a PTX pair.
///
/// Uses `compile_raw` rather than `compile_shared` because `@pipeline` is
/// checked across the pair: the assertion holds if either variant pipelines.
/// The general variant's partial slices never pipeline on their own.
pub fn pair(ctx: &Context, aligned_src: &str, general_src: &str, what: &str) -> Result<(String, String)> {
    // One thread per text. The two are independent, so lowering them at once
    // costs only a second stack.
    let lower = |src: &str, half: &'static str| {
        let (ctx, src) = (ctx.clone(), src.to_string());
        std::thread::Builder::new()
            .stack_size(COMPILE_STACK_BYTES)
            .spawn(move || phobos_lang::compile_raw(&ctx, &src))
            .with_context(|| format!("spawning the compile thread for {what} ({half})"))
    };
    let aligned_thread = lower(aligned_src, "aligned")?;
    let general_thread = lower(general_src, "general")?;
    let join = |handle: JoinHandle<_>| {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("the compile thread for {what} panicked"))
    };
    let aligned_out = join(aligned_thread)?;
    let general_out = join(general_thread)?;
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
    Ok((aligned_out.code, general_out.code))
}
