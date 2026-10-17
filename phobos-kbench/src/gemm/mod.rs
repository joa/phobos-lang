// The GEMM benchmarks: autotune each kind per problem, time the winner
// against the cuBLAS call with the same numerical contract, and check both.
//
// Timings are kernel-only, between CUDA events, C restored outside them,
// Phobos and cuBLAS interleaved launch by launch. A host wall-clock figure
// over restore, launch and synchronize is kept beside them, labelled
// end-to-end.

pub(crate) mod contract;
mod device;
pub(crate) mod env;
mod tune;
mod verify;

use crate::cublas::CuBlas;
use crate::report::Results;
use crate::timing::{self, Stats, Timed};
use contract::Kind;
use device::{Converter, Problem};
use phobos_base::phinfo;
use std::collections::HashMap;
use std::path::PathBuf;
use verify::{Errors, Pattern};

pub(crate) type Dims = (usize, usize, usize);

/// The scalars every correctness check runs, beta = 0 included.
const SCALARS: &[(f32, f32)] = &[(1.0, 0.0), (1.5, 2.5), (-1.0, 1.0), (0.5, -0.75)];

/// Host wall-clock repetitions of the end-to-end figure.
const E2E_REPS: u32 = 20;

pub(crate) struct Options {
    pub shapes: Vec<Dims>,
    pub suite: bool,
    /// Interleaved final rounds; None sizes them to about two seconds.
    pub rounds: Option<u32>,
    pub cache: PathBuf,
    pub retune: bool,
    /// A kernel source to run in place of the kind's example.
    pub source: Option<PathBuf>,
}

/// The compact suite: squares, GPT-2 small's prefill projections at 512
/// tokens, small-M problems with a long k, and for the unaligned kind,
/// sizes no tile divides.
pub(crate) fn suite(kind: Kind) -> Vec<Dims> {
    match kind {
        Kind::F32 => vec![
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (512, 2304, 768),
            (512, 768, 768),
            (512, 3072, 768),
            (512, 768, 3072),
            (32, 4096, 4096),
            (64, 1024, 16384),
        ],
        Kind::F32Unaligned => vec![(1000, 1000, 1000), (509, 2304, 771), (4000, 4000, 4000)],
        Kind::F32SplitK => suite(Kind::F32),
        _ => vec![(1024, 1024, 1024), (4096, 4096, 4096), (512, 3072, 768)],
    }
}

fn default_shapes(kind: Kind) -> Vec<Dims> {
    match kind {
        Kind::F32Unaligned => vec![(4000, 4000, 4000)],
        Kind::F32SplitK => vec![SPLIT_K_SHAPE],
        _ => vec![(4096, 4096, 4096)],
    }
}

/// Split-K's problem: a small M and a long k, which leave a plain tiling a
/// fraction of a wave.
const SPLIT_K_SHAPE: Dims = (64, 1024, 16384);

/// The problem a kind's CSV row reports, if any.
fn csv_shape(kind: Kind) -> Option<Dims> {
    match kind {
        Kind::F32Unaligned => None,
        Kind::F32SplitK => Some(SPLIT_K_SHAPE),
        _ => Some((4096, 4096, 4096)),
    }
}

fn tflops((m, n, k): Dims, ms: f64) -> f64 {
    2.0 * (m * n * k) as f64 / (ms * 1e-3) / 1e12
}

/// Runs one kind over its problems, appending a JSON row per problem.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    stream: &cust::stream::Stream,
    kind: Kind,
    opts: &Options,
    pins: &HashMap<String, i64>,
    env: &env::Env,
    blas: &CuBlas,
    results: &mut Results,
    rows: &mut Vec<serde_json::Value>,
) -> anyhow::Result<()> {
    let code = match &opts.source {
        Some(path) => std::fs::read_to_string(path)?,
        None => kind.code().to_string(),
    };
    let variants = tune::Variants::build(kind, &code, pins)?;
    let conv = if kind.baseline_converts() {
        Some(Converter::new()?)
    } else {
        None
    };
    let mut cache = tune::WinnerCache::open(&opts.cache);
    let shapes = if !opts.shapes.is_empty() {
        opts.shapes.clone()
    } else if opts.suite {
        suite(kind)
    } else {
        default_shapes(kind)
    };

    for dims in shapes {
        match problem(stream, kind, dims, opts, pins, env, blas, conv.as_ref(), &variants, &mut cache) {
            Ok(row) => {
                if Some(dims) == csv_shape(kind) {
                    let t = &row["timing_kernel_only"];
                    let phobos = t["phobos"]["median_ms"].as_f64().unwrap_or(f64::NAN);
                    let blas_ms = t[if kind.baseline_converts() { "cublas_with_convert" } else { "cublas" }]["median_ms"]
                        .as_f64()
                        .unwrap_or(f64::NAN);
                    results.push(kind.name(), "phobos", kind.precision(), tflops(dims, phobos) * 1e3);
                    results.push(kind.name(), "cuBLAS", kind.precision(), tflops(dims, blas_ms) * 1e3);
                }
                rows.push(row);
            }
            Err(e) => {
                phinfo!("{} {dims:?}: FAILED: {e:#}", kind.name());
                rows.push(serde_json::json!({
                    "benchmark": kind.name(),
                    "shape": {"m": dims.0, "n": dims.1, "k": dims.2},
                    "error": format!("{e:#}"),
                }));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn problem(
    stream: &cust::stream::Stream,
    kind: Kind,
    dims: Dims,
    opts: &Options,
    pins: &HashMap<String, i64>,
    env: &env::Env,
    blas: &CuBlas,
    conv: Option<&Converter>,
    vs: &tune::Variants,
    cache: &mut tune::WinnerCache,
) -> anyhow::Result<serde_json::Value> {
    let (m, n, k) = dims;
    phinfo!("{} {m}x{n}x{k}: {}", kind.name(), kind.contract());
    let mut p = Problem::new(kind, dims)?;
    let inp = verify::make_inputs(kind, Pattern::Uniform, dims, 42);
    p.upload(&inp)?;
    let scalars = kind.timed_scalars();

    // Stage-1 screening checks corners and a few random elements.
    let quick: Vec<(usize, usize)> = {
        let (at, _) = verify::positions(m, n, k);
        let stride = (at.len() / 64).max(1);
        at.into_iter().step_by(stride).collect()
    };
    let tol = verify::tolerance(kind, k);
    let quick_check = || -> anyhow::Result<()> {
        let got = p.read(stream, &p.c_phobos)?;
        let e = verify::check(&got, &inp, kind.mul(), scalars, dims, &quick, false, tol);
        anyhow::ensure!(e.pass(), "wrong result: {}", e.summary());
        Ok(())
    };

    let key = tune::WinnerCache::key(&env.key, vs, dims, pins);
    let cached = if opts.retune { None } else { cache.get(&key).and_then(|c| vs.find(&c)) };
    let (win, tuned) = match cached {
        Some(i) => {
            phinfo!("  winner from cache: {}", crate::autotune::fmt_config(&vs.all[i].config));
            (i, false)
        }
        None => {
            let (i, s) = tune::tune(stream, &p, vs, scalars, quick_check)?;
            cache.put(key, &vs.all[i].config, s.median())?;
            (i, true)
        }
    };
    let winner = &vs.all[win];
    let grid = tune::grid_for(kind, &winner.config, dims)?;
    phinfo!(
        "  winner {}: grid {grid:?} x {} threads, {} regs, {} B smem, {} B local, {} blocks/SM",
        crate::autotune::fmt_config(&winner.config),
        winner.block,
        winner.res.regs,
        winner.res.smem_bytes,
        winner.res.local_bytes,
        winner.res.blocks_per_sm
    );

    // The final: Phobos, cuBLAS and, when cuBLAS needs f16 operands the
    // kernel rounds itself, the conversion alone and conversion plus GEMM.
    let blas_item = || Timed {
        prepare: Box::new(|| p.restore(stream, &p.c_blas)),
        launch: Box::new(|| p.launch_blas(blas, scalars)),
    };
    let mut items = vec![tune::timed(stream, &p, vs, win, scalars)?, blas_item()];
    let mut names = vec!["phobos", "cublas"];
    if let Some(conv) = conv {
        let pr = &p;
        pr.convert(stream, conv)?;
        items.push(Timed {
            prepare: Box::new(|| Ok(())),
            launch: Box::new(move || pr.convert(stream, conv)),
        });
        items.push(Timed {
            prepare: Box::new(move || pr.restore(stream, &pr.c_blas)),
            launch: Box::new(move || {
                pr.convert(stream, conv)?;
                pr.launch_blas(blas, scalars)
            }),
        });
        names.extend(["convert_ab", "cublas_with_convert"]);
    }
    let rounds = opts.rounds.unwrap_or_else(|| {
        let est_ms = tflops(dims, 1.0) / 8.0; // a guess at 8 TFLOP/s
        (2000.0 / (est_ms * items.len() as f64)).clamp(30.0, 400.0) as u32
    });
    let state_before = env::gpu_state();
    let stats = timing::interleaved(stream, &mut items, (5, 300.0), rounds)?;
    let state_after = env::gpu_state();
    for (name, s) in names.iter().zip(&stats) {
        phinfo!("  {name:>20}: {} = {:.2} TFLOP/s", s.summary(), tflops(dims, s.median()));
    }
    let ratio = stats[1].median() / stats[0].median();
    phinfo!("  phobos / {}: {:.1}% of its throughput (medians)", kind.baseline().label(), 100.0 * ratio);

    let e2e: Vec<Stats> = items
        .iter_mut()
        .take(2)
        .map(|t| timing::end_to_end(stream, t, E2E_REPS))
        .collect::<anyhow::Result<_>>()?;
    drop(items);

    let checks = correctness(stream, &mut p, &inp, vs, win, blas, conv)?;
    let small = small_checks(stream, kind, vs, win, blas, conv)?;
    let nan = nan_c_check(stream, &mut p, &inp, vs, win, blas, conv)?;
    let all_pass = checks
        .iter()
        .chain(&small)
        .all(|c| c["phobos"]["pass"].as_bool() == Some(true))
        && nan["phobos_nonfinite"].as_u64() == Some(0);
    phinfo!(
        "  correctness: {} ({} checks on this problem, {} on small ones, beta = 0 over a NaN C)",
        if all_pass { "pass" } else { "FAIL" },
        checks.len(),
        small.len()
    );

    let mut timing_json = serde_json::Map::new();
    for (name, s) in names.iter().zip(&stats) {
        let mut j = s.json();
        j["tflops_median"] = tflops(dims, s.median()).into();
        j["tflops_best"] = tflops(dims, s.min()).into();
        timing_json.insert(name.to_string(), j);
    }
    Ok(serde_json::json!({
        "benchmark": kind.name(),
        "contract": kind.contract_json(),
        "baseline": kind.baseline().label(),
        "baseline_converts_ab": kind.baseline_converts(),
        "source": opts.source.as_ref().map_or("example".to_string(), |p| p.display().to_string()),
        "source_fnv1a": format!("{:016x}", tune::fnv1a(&vs.code)),
        "shape": {"m": m, "n": n, "k": k},
        "flops": 2.0 * (m * n * k) as f64,
        "timed_scalars": {"alpha": scalars.0, "beta": scalars.1},
        "winner": {
            "config": winner.config.iter().map(|(n, v)| (n.clone(), (*v).into())).collect::<serde_json::Map<_, _>>(),
            "from_cache": !tuned,
            "grid": [grid.0, grid.1, grid.2],
            "block": winner.block,
            "index_bits": if vs.wide { 64 } else { 32 },
            "resources": winner.res.json(),
        },
        "timing_kernel_only": timing_json,
        "phobos_over_cublas_median": ratio,
        "timing_end_to_end": {"phobos": e2e[0].json(), "cublas": e2e[1].json(), "scope": "host wall clock: restore C + launch + stream synchronize"},
        "gpu_state": {"before_final": state_before, "after_final": state_after},
        "correctness": {"pass": all_pass, "problem": checks, "small": small, "beta0_nan_c": nan},
    }))
}

/// Runs Phobos and cuBLAS once each on `p`'s current inputs and checks both.
#[allow(clippy::too_many_arguments)]
fn run_and_check(
    stream: &cust::stream::Stream,
    p: &Problem,
    inp: &verify::Inputs,
    vs: &tune::Variants,
    win: usize,
    blas: &CuBlas,
    conv: Option<&Converter>,
    scalars: (f32, f32),
) -> anyhow::Result<(Errors, Errors)> {
    let kind = p.kind;
    let (at, full) = verify::positions(p.dims.0, p.dims.1, p.dims.2);
    let tol = verify::tolerance(kind, p.dims.2);
    let mut t = tune::timed(stream, p, vs, win, scalars)?;
    (t.prepare)()?;
    (t.launch)()?;
    let got = p.read(stream, &p.c_phobos)?;
    let phobos = verify::check(&got, inp, kind.mul(), scalars, p.dims, &at, full, tol);

    p.restore(stream, &p.c_blas)?;
    if let Some(conv) = conv {
        p.convert(stream, conv)?;
    }
    p.launch_blas(blas, scalars)?;
    let got = p.read(stream, &p.c_blas)?;
    let cublas = verify::check(&got, inp, kind.mul(), scalars, p.dims, &at, full, tol);
    Ok((phobos, cublas))
}

/// Every pattern and scalar pair on the benchmarked problem.
fn correctness(
    stream: &cust::stream::Stream,
    p: &mut Problem,
    timed_inputs: &verify::Inputs,
    vs: &tune::Variants,
    win: usize,
    blas: &CuBlas,
    conv: Option<&Converter>,
) -> anyhow::Result<Vec<serde_json::Value>> {
    let mut out = Vec::new();
    for &pattern in verify::PATTERNS {
        let inp = verify::make_inputs(p.kind, pattern, p.dims, 7 + pattern as u64);
        p.upload(&inp)?;
        for &scalars in SCALARS {
            let (phobos, cublas) = run_and_check(stream, p, &inp, vs, win, blas, conv, scalars)?;
            log_check(pattern, scalars, &phobos, &cublas);
            out.push(check_json(pattern, scalars, p.dims, &phobos, &cublas));
        }
    }
    p.upload(timed_inputs)?;
    Ok(out)
}

/// Full checks of the winner on small problems: for an aligned kind, two by
/// three tiles over an odd number of k-tiles a split, so an unrolled loop
/// runs its guarded tail; for the unaligned one, sizes no tile divides.
fn small_checks(
    stream: &cust::stream::Stream,
    kind: Kind,
    vs: &tune::Variants,
    win: usize,
    blas: &CuBlas,
    conv: Option<&Converter>,
) -> anyhow::Result<Vec<serde_json::Value>> {
    let cfg = &vs.all[win].config;
    let get = |n: &str| crate::autotune::cfg_value(cfg, n).map(|v| v as usize);
    let shapes: Vec<Dims> = if kind.aligned() {
        let unroll = get("U").unwrap_or(1);
        let split = get("S").unwrap_or(1);
        vec![(2 * get("TILE_M")?, 3 * get("TILE_N")?, split * (2 * unroll + 1) * get("TILE_K")?)]
    } else {
        vec![(67, 131, 37), (get("TILE_M")? + 1, get("TILE_N")? - 1, 3 * get("TILE_K")? + 1)]
    };
    let mut out = Vec::new();
    for dims in shapes {
        let mut p = Problem::new(kind, dims)?;
        for &pattern in &[Pattern::Uniform, Pattern::Cancel] {
            let inp = verify::make_inputs(kind, pattern, dims, 99 + pattern as u64);
            p.upload(&inp)?;
            for &scalars in SCALARS {
                let (phobos, cublas) = run_and_check(stream, &p, &inp, vs, win, blas, conv, scalars)?;
                if !phobos.pass() {
                    log_check(pattern, scalars, &phobos, &cublas);
                }
                out.push(check_json(pattern, scalars, dims, &phobos, &cublas));
            }
        }
    }
    Ok(out)
}

/// beta = 0 over a C of NaN. BLAS reads no C at beta = 0, and neither do
/// the example kernels; one that computes beta * C anyway returns NaN.
fn nan_c_check(
    stream: &cust::stream::Stream,
    p: &mut Problem,
    inp: &verify::Inputs,
    vs: &tune::Variants,
    win: usize,
    blas: &CuBlas,
    conv: Option<&Converter>,
) -> anyhow::Result<serde_json::Value> {
    let nan = vec![f32::NAN; p.dims.0 * p.dims.1];
    p.set_c_init(&nan)?;
    let (phobos, cublas) = run_and_check(stream, p, inp, vs, win, blas, conv, (1.0, 0.0))?;
    p.upload(inp)?;
    phinfo!(
        "  beta = 0 over a NaN C: phobos {} nonfinite, cuBLAS {} nonfinite",
        phobos.nonfinite,
        cublas.nonfinite
    );
    Ok(serde_json::json!({"phobos_nonfinite": phobos.nonfinite, "cublas_nonfinite": cublas.nonfinite, "checked": phobos.checked}))
}

fn log_check(pattern: Pattern, (alpha, beta): (f32, f32), phobos: &Errors, cublas: &Errors) {
    phinfo!(
        "  {:<16} alpha {alpha:>5} beta {beta:>5}: phobos {} {}; cuBLAS max norm err {:.2e}",
        pattern.label(),
        if phobos.pass() { "ok" } else { "FAIL" },
        phobos.summary(),
        cublas.max_norm
    );
}

fn check_json(pattern: Pattern, (alpha, beta): (f32, f32), (m, n, k): Dims, phobos: &Errors, cublas: &Errors) -> serde_json::Value {
    serde_json::json!({
        "pattern": pattern.label(),
        "alpha": alpha,
        "beta": beta,
        "shape": {"m": m, "n": n, "k": k},
        "phobos": phobos.json(),
        "cublas": cublas.json(),
    })
}
