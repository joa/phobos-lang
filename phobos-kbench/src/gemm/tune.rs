// The GEMM autotuner: every variant compiled and loaded once a run, then
// ranked per problem by event-timed medians, winners remembered on disk.

use super::contract::Kind;
use super::device::Problem;
use crate::autotune::{Compiled, Loaded, Setting, compile, fmt_config};
use crate::timing::{self, Stats, Timed};
use cust::function::{BlockSize, FunctionAttribute};
use cust::stream::Stream;
use phobos_base::{phdebug, phinfo};
use std::collections::BTreeMap;
use std::rc::Rc;
use std::path::{Path, PathBuf};

/// Batches a stage-1 probe times. Their median ranks the variants.
const SHORT_PROBES: u32 = 5;
/// GPU time a stage-1 batch covers at least, so a short kernel's launch
/// overhead amortizes.
const SHORT_BATCH_MS: f64 = 4.0;
/// Wall clock a stage-1 probe warms for after its correctness check. The
/// card drops to idle clocks during host work, and a short problem's probes
/// ran in the ramp: 0.3 ms kernels measured 2-4 ms, and a 0.44 ms variant
/// ranked above a 0.29 ms one.
const SHORT_WARMUP_MS: f64 = 30.0;
/// Interleaved rounds of the stage-2 finalists.
const LONG_ROUNDS: u32 = 30;
/// Stage 2 takes one finalist per this many runnable variants, and at least
/// [`MIN_FINALISTS`].
const VARIANTS_PER_FINALIST: usize = 16;
const MIN_FINALISTS: usize = 4;

/// Static resources of a loaded variant, as the driver reports them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Resources {
    pub regs: i32,
    pub smem_bytes: i32,
    /// Per-thread local memory: spills or a stack.
    pub local_bytes: i32,
    pub blocks_per_sm: u32,
}

impl Resources {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "registers": self.regs,
            "static_smem_bytes": self.smem_bytes,
            "local_bytes_per_thread": self.local_bytes,
            "blocks_per_sm": self.blocks_per_sm,
        })
    }
}

pub(crate) struct Variant {
    /// The compiled settings, plus `S` for a split-K kind.
    pub config: Vec<Setting>,
    /// Threads a CTA, which `@launch` may take from the config.
    pub block: u32,
    /// Shared by the splits of one compiled config.
    pub module: Rc<Loaded>,
    pub res: Resources,
}

/// Every variant of one kind's search space that compiled and can launch.
pub(crate) struct Variants {
    pub kind: Kind,
    /// The kernel source the variants were compiled from.
    pub code: String,
    pub all: Vec<Variant>,
}

impl Variants {
    /// Compiles the space on every core, then loads each module and drops
    /// the ones the card cannot launch at all.
    pub fn build(kind: Kind, code: &str, pins: &std::collections::HashMap<String, i64>) -> anyhow::Result<Self> {
        let kernels = phobos_lang::parse(code)?;
        let space = phobos_lang::pin(phobos_lang::search_space(&kernels[0]), pins)?;
        let configs = phobos_base::combo::cartesian_product(&space);
        phinfo!("{}: compiling {} variants", kind.name(), configs.len());

        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut compiled: Vec<Option<anyhow::Result<Compiled>>> = (0..configs.len()).map(|_| None).collect();
        let slots = std::sync::Mutex::new(&mut compiled);
        std::thread::scope(|s| {
            for _ in 0..threads {
                std::thread::Builder::new()
                    .stack_size(256 << 20)
                    .spawn_scoped(s, || {
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(cfg) = configs.get(i) else { break };
                            let out = compile(code, cfg);
                            slots.lock().expect("no compile thread panics holding it")[i] = Some(out);
                        }
                    })
                    .expect("spawning a compile thread");
            }
        });

        let mut all = Vec::new();
        for (cfg, out) in configs.into_iter().zip(compiled) {
            let loaded = out
                .expect("every slot is filled")
                .and_then(|compiled| Loaded::new(&compiled))
                .and_then(|module| {
                    let block = kernels[0]
                        .cta_threads_for(&cfg.iter().cloned().collect())
                        .map_err(anyhow::Error::msg)? as u32;
                    let res = resources(&module, kind.kernel(), block)?;
                    anyhow::ensure!(res.blocks_per_sm > 0, "does not fit an SM ({} regs, {} B smem)", res.regs, res.smem_bytes);
                    Ok((block, Rc::new(module), res))
                });
            match loaded {
                Ok((block, module, res)) => {
                    for &split in kind.splits() {
                        let mut config = cfg.clone();
                        if kind.splits().len() > 1 {
                            config.push(("S".into(), i64::from(split)));
                        }
                        all.push(Variant {
                            config,
                            block,
                            module: module.clone(),
                            res,
                        });
                    }
                }
                Err(e) => phinfo!("  {}: dropped ({e})", fmt_config(&cfg)),
            }
        }
        anyhow::ensure!(!all.is_empty(), "{}: no variant compiles and fits", kind.name());
        phinfo!("{}: {} variants loaded", kind.name(), all.len());
        Ok(Variants {
            kind,
            code: code.to_string(),
            all,
        })
    }

    /// The variant with exactly these settings, in any order.
    pub fn find(&self, config: &[Setting]) -> Option<usize> {
        self.all
            .iter()
            .position(|v| v.config.len() == config.len() && config.iter().all(|s| v.config.contains(s)))
    }
}

/// What `kernel` of `loaded` takes from an SM, its dynamic shared memory
/// counted with its static.
fn resources(loaded: &Loaded, kernel: &str, block: u32) -> anyhow::Result<Resources> {
    let f = loaded.module.get_function(kernel)?;
    let dynamic_bytes = loaded.shared_bytes(kernel);
    Ok(Resources {
        regs: f.get_attribute(FunctionAttribute::NumRegisters)?,
        smem_bytes: f.get_attribute(FunctionAttribute::SharedMemorySizeBytes)? + dynamic_bytes as i32,
        local_bytes: f.get_attribute(FunctionAttribute::LocalSizeBytes)?,
        blocks_per_sm: f.max_active_blocks_per_multiprocessor(BlockSize::x(block), dynamic_bytes as usize)?,
    })
}

/// The launch grid of `config` over an m x n output, the k splits its
/// third axis, or why it has none.
pub(crate) fn grid_for(kind: Kind, config: &[Setting], (m, n, k): (usize, usize, usize)) -> anyhow::Result<(u32, u32, u32)> {
    let tm = crate::autotune::cfg_value(config, "TILE_M")? as usize;
    let tn = crate::autotune::cfg_value(config, "TILE_N")? as usize;
    let split = crate::autotune::cfg_value(config, "S").unwrap_or(1) as usize;
    if kind.aligned() {
        let tk = crate::autotune::cfg_value(config, "TILE_K")? as usize;
        anyhow::ensure!(
            m.is_multiple_of(tm) && n.is_multiple_of(tn) && k.is_multiple_of(tk * split),
            "tile does not divide the problem"
        );
    }
    if kind.splits().len() > 1 {
        // The reduce works on 32x32 tiles.
        anyhow::ensure!(m.is_multiple_of(32) && n.is_multiple_of(32), "the reduce's tile does not divide the problem");
    }
    Ok((m.div_ceil(tm) as u32, n.div_ceil(tn) as u32, split as u32))
}

/// One timed launch of variant `v` on `p`, C restored outside the events.
pub(crate) fn timed<'a>(stream: &'a Stream, p: &'a Problem, vs: &'a Variants, v: usize, scalars: (f32, f32)) -> anyhow::Result<Timed<'a>> {
    let (gx, gy, split) = grid_for(vs.kind, &vs.all[v].config, p.dims)?;
    let module = &vs.all[v].module;
    let (block, wide) = (vs.all[v].block, module.wide);
    let launch: Box<dyn FnMut() -> anyhow::Result<()> + 'a> = if vs.kind.splits().len() > 1 {
        anyhow::ensure!(split <= p.max_split, "{split} splits exceed the workspace");
        let (sk, red) = (module.module.get_function("gemm_split")?, module.module.get_function("gemm_reduce")?);
        let shared = (module.shared_bytes("gemm_split"), module.shared_bytes("gemm_reduce"));
        Box::new(move || p.launch_split(stream, (&sk, &red), (gx, gy, split), block, shared, scalars, wide))
    } else {
        let func = module.module.get_function("gemm")?;
        let shared = module.shared_bytes("gemm");
        Box::new(move || p.launch_phobos(stream, &func, (gx, gy), (block, shared), wide, scalars))
    };
    Ok(Timed {
        prepare: Box::new(move || p.restore(stream, &p.c_phobos)),
        launch,
    })
}

/// Tunes `vs` on `p`, returning the winner's index and its stage-2 timing.
/// `check` runs the winner-to-be's output past a quick verification; a
/// variant that fails it is out.
pub(crate) fn tune(
    stream: &Stream,
    p: &Problem,
    vs: &Variants,
    scalars: (f32, f32),
    mut check: impl FnMut() -> anyhow::Result<()>,
) -> anyhow::Result<(usize, Stats)> {
    let mut ranked: Vec<(usize, f64)> = Vec::new();
    for (i, v) in vs.all.iter().enumerate() {
        let probe = (|| {
            let mut t = timed(stream, p, vs, i, scalars)?;
            timing::interleaved(stream, std::slice::from_mut(&mut t), (0, 0.0), 1)?;
            check()?;
            let warm = timing::interleaved(stream, std::slice::from_mut(&mut t), (1, SHORT_WARMUP_MS), 3)?;
            let reps = (SHORT_BATCH_MS / warm[0].median().max(1e-3)).ceil().min(1000.0) as u32;
            Ok::<_, anyhow::Error>(timing::batched(stream, &mut t, reps, SHORT_PROBES)?.median())
        })();
        match probe {
            Ok(ms) => {
                phdebug!("  stage 1 {}: {ms:.4} ms", fmt_config(&v.config));
                ranked.push((i, ms))
            }
            Err(e) => phinfo!("  {}: skipped ({e})", fmt_config(&v.config)),
        }
    }
    anyhow::ensure!(!ranked.is_empty(), "no variant runs on this problem");
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    ranked.truncate(MIN_FINALISTS.max(ranked.len().div_ceil(VARIANTS_PER_FINALIST)));

    let mut items: Vec<Timed<'_>> = ranked
        .iter()
        .map(|&(i, _)| timed(stream, p, vs, i, scalars))
        .collect::<anyhow::Result<_>>()?;
    let stats = timing::interleaved(stream, &mut items, (3, 200.0), LONG_ROUNDS)?;
    let mut best = 0;
    for (r, s) in stats.iter().enumerate() {
        phinfo!("  {}: {}", fmt_config(&vs.all[ranked[r].0].config), s.summary());
        if s.median() < stats[best].median() {
            best = r;
        }
    }
    Ok((ranked[best].0, stats[best].clone()))
}

/// Winners remembered across runs, keyed on everything that could change
/// which variant wins: card, driver, compiler, kernel source, problem.
pub(crate) struct WinnerCache {
    path: PathBuf,
    entries: BTreeMap<String, serde_json::Value>,
}

impl WinnerCache {
    pub fn open(path: &Path) -> Self {
        let entries = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        WinnerCache {
            path: path.to_path_buf(),
            entries,
        }
    }

    pub fn key(env_key: &str, vs: &Variants, (m, n, k): (usize, usize, usize), pins: &std::collections::HashMap<String, i64>) -> String {
        let mut pins: Vec<_> = pins.iter().collect();
        pins.sort();
        format!(
            "{env_key}|src {:016x}|{}|{m}x{n}x{k}|pins {pins:?}",
            fnv1a(&vs.code),
            vs.kind.name()
        )
    }

    pub fn get(&self, key: &str) -> Option<Vec<Setting>> {
        let cfg = self.entries.get(key)?.get("config")?.as_object()?;
        Some(cfg.iter().filter_map(|(n, v)| Some((n.clone(), v.as_i64()?))).collect())
    }

    pub fn put(&mut self, key: String, config: &[Setting], median_ms: f64) -> anyhow::Result<()> {
        let cfg: serde_json::Map<String, serde_json::Value> = config.iter().map(|(n, v)| (n.clone(), (*v).into())).collect();
        self.entries.insert(key, serde_json::json!({"config": cfg, "median_ms": median_ms}));
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&self.path, serde_json::to_string_pretty(&self.entries)?)?;
        Ok(())
    }
}

/// FNV-1a, a hash that stays the same across Rust releases.
pub(crate) fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}
