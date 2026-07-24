use cust::module::Module;
use phobos_base::combo::cartesian_product;
use phobos_base::{phdebug, phinfo};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct Grid(pub u32, pub u32);

pub type Setting = (String, i64);

pub struct Winner {
    pub config: Vec<Setting>,
    pub ptx: String,
}

/// Compile Phobos source with the given autotune choices pinned.
pub fn compile(code: &str, config: &[Setting]) -> anyhow::Result<String> {
    // mma.sync requires 64bit index width.
    // we widen implicitly when required.
    let requires_wide_index = phobos_lang::parse(code)
        .map(|ks| phobos_lang::requires_wide_index(&ks))
        .unwrap_or(false);
    let mut ctx = phobos_base::context::Context {
        //print_phases: true,
        shape_overrides: config.iter().cloned().collect(),
        ..Default::default()
    };
    if requires_wide_index && ctx.index_bitwidth < 64 {
        phdebug!("widening index_bitwidth to 64");
        ctx.index_bitwidth = 64;
    }
    phobos_lang::compile(&ctx, code)
}

pub fn pin(
    space: Vec<(String, Vec<i64>)>,
    pins: &std::collections::HashMap<String, i64>,
) -> anyhow::Result<Vec<(String, Vec<i64>)>> {
    for name in pins.keys() {
        anyhow::ensure!(
            space.iter().any(|(n, _)| n == name),
            "autotune pin '{name}' is not a search dim of this kernel (dims: {})",
            space
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(space
        .into_iter()
        .map(|(name, choices)| match pins.get(&name) {
            Some(&v) => (name, vec![v]),
            None => (name, choices),
        })
        .collect())
}

pub fn cfg_value(cfg: &[Setting], name: &str) -> anyhow::Result<i64> {
    cfg.iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| *v)
        .ok_or_else(|| anyhow::anyhow!("config has no '{name}'"))
}

pub fn fmt_config(cfg: &[Setting]) -> String {
    cfg.iter()
        .map(|(n, v)| format!("{n}={v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

const WARMUP: u32 = 10;

/// Two-stage kernel autotuner.
pub struct Autotuner<'a, G, L, V>
where
    G: Fn(&[Setting]) -> anyhow::Result<Grid>,
    L: FnMut(&Module, Grid) -> anyhow::Result<()>,
    V: FnMut() -> anyhow::Result<()>,
{
    pub code: &'a str,
    pub grid_for: G,
    pub launch: L,
    pub verify: V,
    pub short_probes: u32,
    pub long_probes: u32,
    pub finalists: usize,
}

impl<'a, G, L, V> Autotuner<'a, G, L, V>
where
    G: Fn(&[Setting]) -> anyhow::Result<Grid>,
    L: FnMut(&Module, Grid) -> anyhow::Result<()>,
    V: FnMut() -> anyhow::Result<()>,
{
    pub fn run(&mut self, space: &[(String, Vec<i64>)]) -> anyhow::Result<Winner> {
        let all = cartesian_product(space);
        phinfo!("autotune: {} configs", all.len());

        // Stage 1: compile and short-probe every config. We rank by the fastest observed launch
        let mut candidates: Vec<(Vec<Setting>, String, Duration)> = Vec::new();
        for cfg in &all {
            match self.probe_short(cfg) {
                Ok((ptx, best)) => {
                    phinfo!("  {}: {best:.2?}", fmt_config(cfg));
                    candidates.push((cfg.clone(), ptx, best));
                }
                Err(e) => phinfo!("  {}: skipped ({e})", fmt_config(cfg)),
            }

            thread::sleep(Duration::from_millis(100));
        }

        // Stage 2: long-probe the top finalists, interleaved round-robin.
        // Timing each finalist in its own contiguous block hands whichever
        // runs first (already the stage 1 leader, after the sort) the coolest
        // silicon and the highest boost clocks, and that bias is often larger
        // than the margin between finalists, so stage 1 noise gets confirmed
        // instead of corrected. One launch per finalist per round keeps every
        // config under the same clock and thermal state, so min-over-rounds
        // compares kernels rather than GPU moods.
        candidates.sort_by_key(|(_, _, best)| *best);
        candidates.truncate(self.finalists);
        anyhow::ensure!(!candidates.is_empty(), "autotune: no config works");
        phinfo!("autotune: long probe of the top {}", candidates.len());

        struct Finalist {
            idx: usize,
            module: Module,
            grid: Grid,
            min: f64,
            max: f64,
        }

        let mut finalists: Vec<Finalist> = Vec::new();
        for (idx, (cfg, ptx, _)) in candidates.iter().enumerate() {
            match Module::from_ptx(ptx.as_str(), &[])
                .map_err(anyhow::Error::from)
                .and_then(|module| Ok((module, (self.grid_for)(cfg)?)))
            {
                Ok((module, grid)) => finalists.push(Finalist {
                    idx,
                    module,
                    grid,
                    min: f64::INFINITY,
                    max: 0f64,
                }),
                Err(e) => phinfo!("  {}: failed ({e})", fmt_config(cfg)),
            }
        }

        // One joint warmup ramps the clocks before any timed probe.
        for f in &finalists {
            for _ in 0..WARMUP {
                (self.launch)(&f.module, f.grid)?;
            }
        }

        for _ in 0..self.long_probes.max(1) {
            for f in &mut finalists {
                let now = Instant::now();
                (self.launch)(&f.module, f.grid)?;
                let dt = now.elapsed().as_secs_f64();
                f.min = f.min.min(dt);
                f.max = f.max.max(dt);
            }
        }

        let mut best: Option<(usize, Duration)> = None;
        for f in &finalists {
            let fastest = Duration::from_secs_f64(f.min);
            let spread = Duration::from_secs_f64(f.max - f.min);
            phinfo!(
                "  {}: {fastest:.2?} (spread {spread:.2?})",
                fmt_config(&candidates[f.idx].0)
            );
            if best.is_none_or(|(_, t)| fastest < t) {
                best = Some((f.idx, fastest));
            }
        }

        let (best_idx, _) = best.ok_or_else(|| anyhow::anyhow!("autotune: no finalist works"))?;
        let (best_cfg, best_ptx, _) = candidates.swap_remove(best_idx);
        phinfo!("autotune winner: {}", fmt_config(&best_cfg));

        Ok(Winner {
            config: best_cfg,
            ptx: best_ptx,
        })
    }

    fn probe_short(&mut self, cfg: &[Setting]) -> anyhow::Result<(String, Duration)> {
        let ptx = compile(self.code, cfg)?;
        let module = Module::from_ptx(ptx.as_str(), &[])?;
        let grid = (self.grid_for)(cfg)?;

        // verify correctness on the first launch (also a warmup).
        (self.launch)(&module, grid)?;
        (self.verify)()?;
        let (fastest, _) = self.time(&module, grid, self.short_probes)?;
        Ok((ptx, fastest))
    }

    fn time(
        &mut self,
        module: &Module,
        grid: Grid,
        probes: u32,
    ) -> anyhow::Result<(Duration, Duration)> {
        for _ in 0..WARMUP {
            (self.launch)(module, grid)?;
        }

        let (mut min, mut max) = (f64::INFINITY, 0f64);
        for _ in 0..probes.max(1) {
            let now = Instant::now();
            (self.launch)(module, grid)?;
            let dt = now.elapsed().as_secs_f64();
            min = min.min(dt);
            max = max.max(dt);
        }

        Ok((
            Duration::from_secs_f64(min),
            Duration::from_secs_f64(max - min), // spread
        ))
    }
}
