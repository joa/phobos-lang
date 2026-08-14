use cust::prelude::*;
use phobos_base::phinfo;
use rand::prelude::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

mod autotune;
mod flash;
mod gemm;
mod harness;
mod saxpy;

use flash::*;
use gemm::*;
use saxpy::*;
mod cublas;
mod report;

use report::{Precision, Results};

pub(crate) const CODE_SAXPY: &str = include_str!("../../examples/saxpy_fp32.ph");
pub(crate) const CODE_MATMUL: &str = include_str!("../../examples/gemm_fp32.ph");
pub(crate) const CODE_MATMUL_TC: &str = include_str!("../../examples/gemm_fp16tc_fp32acc.ph");
pub(crate) const CODE_MATMUL_FP16: &str = include_str!("../../examples/gemm_fp16.ph");
pub(crate) const CODE_FLASH: &str = include_str!("../../examples/flash_attention_fp32.ph");
pub(crate) const CODE_FLASH_F16: &str = include_str!("../../examples/flash_attention_fp16.ph");

pub(crate) const PROBES_SHORT: u32 = 5u32;
// Stage 2 rounds: one interleaved launch per finalist per round (see
// autotune::Autotuner::run). More rounds tighten the min at trivial cost.
pub(crate) const PROBES_LONG: u32 = 30u32;

/// All benchmark names, for --bench and usage messages.
pub(crate) const BENCHES: &[&str] = &[
    "saxpy_fp32",
    "gemm_fp32",
    "gemm_fp16tc_fp32acc",
    "gemm_fp16",
    "flash_fp32",
    "flash_fp16",
];

/// Parsed command line. bench selects a single benchmark (all of them when
/// None); pins fixes autotune dims to skip the search (for ncu profiling).
struct Options {
    bench: Option<String>,
    pins: HashMap<String, i64>,
    /// Where to write the results CSV, if --csv was given.
    csv: Option<std::path::PathBuf>,
    /// Theoretical-peak overrides (TFLOP/s) for the CSV's reference columns.
    peak_fp32: Option<f64>,
    peak_fp16tc: Option<f64>,
    peak_fp16tcf32acc: Option<f64>,
}

/// Default path used when --csv is passed without an explicit value.
pub(crate) const DEFAULT_CSV: &str = "phobos-bench.csv";

impl Options {
    fn parse(args: impl Iterator<Item = String>) -> anyhow::Result<Options> {
        let mut bench = None;
        let mut pins = HashMap::new();
        let mut csv = None;
        let mut peak_fp32 = None;
        let mut peak_fp16tc = None;
        let mut peak_fp16tcf32acc = None;
        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--bench" | "-bench" => {
                    bench = Some(
                        args.next()
                            .ok_or_else(|| anyhow::anyhow!("{arg} needs a benchmark name"))?,
                    );
                }
                "--autotune" | "-autotune" => {
                    let spec = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("{arg} needs a \"NAME=VALUE ...\" spec"))?;
                    parse_pins(&spec, &mut pins)?;
                }
                "--csv" | "-csv" => {
                    // Optional value: a following token that is not another flag.
                    let path = match args.peek() {
                        Some(next) if !next.starts_with('-') => args.next().unwrap(),
                        _ => DEFAULT_CSV.to_string(),
                    };
                    csv = Some(std::path::PathBuf::from(path));
                }
                "--peak-fp32" | "-peak-fp32" => {
                    peak_fp32 = Some(parse_peak(&arg, args.next())?);
                }
                "--peak-fp16tc" | "-peak-fp16tc" => {
                    peak_fp16tc = Some(parse_peak(&arg, args.next())?);
                }
                "--peak-fp16tcf32acc" | "-peak-fp16tcf32acc" => {
                    peak_fp16tcf32acc = Some(parse_peak(&arg, args.next())?);
                }
                "--help" | "-h" => {
                    println!(
                        "usage: phobos-bench [--bench NAME] [--autotune \"DIM=VAL ...\"] \
                         [--csv [PATH]] [--peak-fp32 TFLOPS] [--peak-fp16tc TFLOPS] [--peak-fp16tcf32acc TFLOPS]\n\
                         \n  --bench NAME                run only one benchmark: {}\
                         \n  --autotune SPEC             pin autotune dims (skips the search), e.g.\
                         \n                              --bench gemm_fp16 --autotune \"TILE_M=256 TILE_N=128 TILE_K=16\"\
                         \n  --csv [PATH]                write a results CSV (default {DEFAULT_CSV}) of achieved\
                         \n                              GFLOP/s vs theoretical peak\
                         \n  --peak-fp32         TFLOPS  override the detected fp32 CUDA-core peak\
                         \n  --peak-fp16tc       TFLOPS  override the detected fp16 tensor-core peak\
                         \n  --peak-fp16tcf32acc TFLOPS  override the detected fp16 tensor-core f32 acc peak",
                        BENCHES.join(", ")
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument '{other}' (try --help)"),
            }
        }
        if let Some(b) = &bench {
            anyhow::ensure!(
                BENCHES.contains(&b.as_str()),
                "unknown --bench '{b}'; available: {}",
                BENCHES.join(", ")
            );
        }
        anyhow::ensure!(
            pins.is_empty() || bench.is_some(),
            "--autotune pins are kernel-specific; pass --bench to pick one"
        );
        Ok(Options {
            bench,
            pins,
            csv,
            peak_fp32,
            peak_fp16tc,
            peak_fp16tcf32acc,
        })
    }

    /// Whether name should run under the current --bench selection.
    fn wants(&self, name: &str) -> bool {
        self.bench.as_deref().is_none_or(|b| b == name)
    }
}

/// Parses a positive TFLOP/s value for a --peak-* override.
fn parse_peak(flag: &str, value: Option<String>) -> anyhow::Result<f64> {
    let value = value.ok_or_else(|| anyhow::anyhow!("{flag} needs a value in TFLOP/s"))?;
    let tflops: f64 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("{flag} value '{value}' is not a number"))?;
    anyhow::ensure!(tflops > 0.0, "{flag} value must be positive");
    Ok(tflops)
}

/// Parses a "TILE_M=256 TILE_N=128 ..." spec (whitespace- or comma-separated)
/// into the pin map.
fn parse_pins(spec: &str, pins: &mut HashMap<String, i64>) -> anyhow::Result<()> {
    for tok in spec.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        let (name, val) = tok
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("bad autotune pin '{tok}', expected NAME=VALUE"))?;
        let val: i64 = val.parse().map_err(|_| {
            anyhow::anyhow!("autotune pin '{name}' has a non-integer value '{val}'")
        })?;
        pins.insert(name.to_string(), val);
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let opts = Options::parse(std::env::args().skip(1))?;
    let pins = &opts.pins;

    let _ctx = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut results = Results::default();

    if opts.wants("saxpy_fp32") {
        bench_saxpy(&stream, pins, &mut results)?;
    }
    if opts.wants("gemm_fp32") {
        bench_gemm_fp32(
            &stream,
            CODE_MATMUL,
            "gemm_fp32",
            "phobos gemm_fp32",
            false,
            1.5f32,
            2.5f32,
            pins,
            &mut results,
        )?;
    }
    if opts.wants("gemm_fp16tc_fp32acc") {
        bench_gemm_fp32(
            &stream,
            CODE_MATMUL_TC,
            "gemm_fp16tc_fp32acc",
            "phobos gemm_fp16tc_fp32acc",
            true,
            1.0f32,
            1.0f32,
            pins,
            &mut results,
        )?;
    }
    if opts.wants("gemm_fp16") {
        bench_gemm_fp16(
            &stream,
            CODE_MATMUL_FP16,
            "gemm_fp16",
            "phobos gemm_fp16",
            1.0f32,
            1.0f32,
            pins,
            &mut results,
        )?;
    }
    if opts.wants("flash_fp32") {
        bench_flash_attention_fp32(&stream, pins, &mut results)?;
    }
    if opts.wants("flash_fp16") {
        bench_flash_attention_fp16(&stream, pins, &mut results)?;
    }

    if let Some(path) = &opts.csv {
        let peaks =
            report::Peaks::detect(opts.peak_fp32, opts.peak_fp16tc, opts.peak_fp16tcf32acc)?;
        results.write_csv(path, &peaks)?;
    }

    Ok(())
}
