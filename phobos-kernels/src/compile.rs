use anyhow::{Context as _, Result};
use cust::device::DeviceAttribute;
use cust::module::Module;
use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig, SUPPORTED_CHIPS};
use phobos_base::progress::{self, Step};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::lower::{self, Timed};
use crate::{cache, manifest};

/// What compilation calls itself when it reports progress.
const STAGE: &str = "kernels";

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

/// The chip kernels compile for: `PHOBOS_CHIP`, or the newest supported chip
/// the card can run, since PTX for an older target JITs forward. A card below
/// every supported chip gets the oldest, which the driver then refuses.
fn chip() -> &'static str {
    static CHIP: OnceLock<String> = OnceLock::new();
    CHIP.get_or_init(|| {
        if let Ok(chip) = std::env::var("PHOBOS_CHIP") {
            return chip;
        }
        let device = cust::device::Device::get_device(0).ok();
        let attribute = |which| {
            device
                .and_then(|d| d.get_attribute(which).ok())
                .map_or(0, |v| v.max(0) as u32)
        };
        let capability = attribute(DeviceAttribute::ComputeCapabilityMajor) * 10
            + attribute(DeviceAttribute::ComputeCapabilityMinor);
        best_chip(capability).to_string()
    })
}

/// The newest of [`SUPPORTED_CHIPS`] at or below `capability`.
fn best_chip(capability: u32) -> &'static str {
    SUPPORTED_CHIPS
        .iter()
        .copied()
        .rev()
        .find(|chip| GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(*chip)).compute_capability() <= capability)
        .unwrap_or(SUPPORTED_CHIPS[0])
}

/// A compilation context for the card, with `shapes` bound as `@autotune`
/// overrides.
fn context_for(shapes: &[Override<'_>]) -> Context {
    let mut ctx = Context {
        gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip())),
        ..Context::default()
    };
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
    manifest::record(ctx, &[source]);
    let hit = cache::load(ctx, source);
    let cached = hit.is_some();
    let (took, (ptx, shared)) = match hit {
        Some(hit) => (Duration::ZERO, hit),
        None => {
            progress::started(STAGE, what);
            let (out, took) = lower::single(ctx, source, what)?;
            cache::store(ctx, source, &out.0, &out.1);
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
            manifest::record(&ctx, &[source]);
            match cache::load(&ctx, source) {
                Some((ptx, _)) => Ok(Slot::Hit(ptx)),
                None => {
                    // Said before the spawn, so the whole batch is named the
                    // moment it starts rather than as each one is joined.
                    progress::started(STAGE, what);
                    let handle = lower::spawn(&ctx, source)
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
                    let ((ptx, shared), took) = lower::join(handle, what)?;
                    cache::store(&ctx, &source, &ptx, &shared);
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
    /// source, substituting `{ALIGNED}` with `claims.0` / `claims.1`; see
    /// [`lower::pair`].
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

        manifest::record(&ctx, &[&aligned_src, &general_src]);
        let hit = cache::load_pair(&ctx, &aligned_src, &general_src);
        // The pair is one cache entry and one unit of work, so it is timed and
        // reported as one kernel; `what` names the kernel rather than either
        // of its two alignment variants.
        let cached = hit.is_some();
        let at = Instant::now();
        let (aligned_ptx, general_ptx) = match hit {
            Some(hit) => hit,
            None => {
                progress::started(STAGE, what);
                let (aligned_ptx, general_ptx) = lower::pair(&ctx, &aligned_src, &general_src, what)?;
                cache::store_pair(&ctx, &aligned_src, &general_src, &aligned_ptx, &general_ptx);
                (aligned_ptx, general_ptx)
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

#[cfg(test)]
mod tests {
    use super::best_chip;

    #[test]
    fn a_card_gets_the_newest_chip_it_can_run() {
        assert_eq!(best_chip(75), "sm_75");
        assert_eq!(best_chip(87), "sm_86");
        assert_eq!(best_chip(89), "sm_89");
        assert_eq!(best_chip(100), "sm_90");
        assert_eq!(best_chip(121), "sm_120");
        assert_eq!(best_chip(61), "sm_75");
    }
}
