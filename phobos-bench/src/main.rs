use cust::prelude::*;
use phobos_base::phinfo;
use rand::prelude::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

mod autotune;
mod cublas;
mod report;

use report::{Precision, Results};

const CODE_SAXPY: &str = include_str!("../../examples/saxpy_fp32.ph");
const CODE_MATMUL: &str = include_str!("../../examples/gemm_fp32.ph");
const CODE_MATMUL_TC: &str = include_str!("../../examples/gemm_fp16tc_fp32acc.ph");
const CODE_MATMUL_FP16: &str = include_str!("../../examples/gemm_fp16.ph");
const CODE_FLASH: &str = include_str!("../../examples/flash_attention_fp32.ph");
const CODE_FLASH_F16: &str = include_str!("../../examples/flash_attention_fp16.ph");

const PROBES_SHORT: u32 = 5u32;
// Stage 2 rounds: one interleaved launch per finalist per round (see
// autotune::Autotuner::run). More rounds tighten the min at trivial cost.
const PROBES_LONG: u32 = 30u32;

/// All benchmark names, for --bench and usage messages.
const BENCHES: &[&str] = &[
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
const DEFAULT_CSV: &str = "phobos-bench.csv";

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
                         [--csv [PATH]] [--peak-fp32 TFLOPS] [--peak-fp16 TFLOPS]\n\
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
        let peaks = report::Peaks::detect(opts.peak_fp32, opts.peak_fp16tc, opts.peak_fp16tcf32acc)?;
        results.write_csv(path, &peaks)?;
    }

    Ok(())
}

/// bench returns (rms, stddev).
/// Times launch and reports the fastest of N runs (with the spread as a
/// noise indicator). A kernel's compute time is fixed; run-to-run variance is
/// external noise (clock ramp, scheduling, background driver work) that only
/// slows a sample, so the minimum is the truest measure and the right basis for
/// the cuBLAS ratio. This matches the autotuner's ranking metric, so the winner
/// it picks is the one reported here (see autotune::Autotuner::run).
fn bench(
    name: &str,
    mut launch: impl FnMut() -> anyhow::Result<()>,
) -> anyhow::Result<(Duration, Duration)> {
    const N: u32 = 100;

    for _ in 0..10 {
        launch()?;
    }

    let (mut min, mut max) = (f64::INFINITY, 0f64);
    for _ in 0..N {
        let now = Instant::now();
        launch()?;
        let dt = now.elapsed().as_secs_f64();
        min = min.min(dt);
        max = max.max(dt);
    }
    let (d, s) = (
        Duration::from_secs_f64(min),
        Duration::from_secs_f64(max - min),
    );
    phinfo!("{name}: {d:.2?} (spread {s:.2?})");
    Ok((d, s))
}

/// Zeroes bytes of device memory at ptr (a byte memset to 0, which is +0.0
/// for both f32 and fp16). Centralizes the cuMemset* error check shared by the
/// GEMM launch and cuBLAS comparison loops.
fn zero_device_async(
    ptr: cust::sys::CUdeviceptr,
    bytes: usize,
    stream: &cust::stream::Stream,
) -> anyhow::Result<()> {
    let r = unsafe { cust::sys::cuMemsetD8Async(ptr, 0, bytes, stream.as_inner()) };
    anyhow::ensure!(
        r == cust::sys::CUresult::CUDA_SUCCESS,
        "cuMemsetD8Async failed: {:?}",
        r
    );
    Ok(())
}

/// Launches a matmul kernel over C[M,N] = A[M,K] @ B[K,N] (plus alpha/beta),
/// dispatching on wide so the memref descriptor's offset/size/stride fields
/// match the kernel's index width (i64 for the default @tensorcore mma.sync
/// path, i32 otherwise). The host metadata must match the kernel's ABI or every
/// field after the first pointer shifts; keeping the two layouts here is what
/// guarantees the fp32 and fp16 benches stay in sync. Pointers are raw device
/// addresses (8 bytes either way), so this is element-type agnostic.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn launch_gemm(
    func: &cust::function::Function,
    grid: (u32, u32),
    block: u32,
    stream: &cust::stream::Stream,
    a: cust::sys::CUdeviceptr,
    b: cust::sys::CUdeviceptr,
    c: cust::sys::CUdeviceptr,
    M: i32,
    N: i32,
    K: i32,
    alpha: f32,
    beta: f32,
    wide: bool,
) -> anyhow::Result<()> {
    let (m64, n64, k64) = (M as i64, N as i64, K as i64);
    unsafe {
        if wide {
            launch!(func<<<grid, block, 0, stream>>>(
                a, a, 0i64, m64, k64, k64, 1i64,
                b, b, 0i64, k64, n64, n64, 1i64,
                c, c, 0i64, m64, n64, n64, 1i64,
                alpha,
                beta
            ))?;
        } else {
            launch!(func<<<grid, block, 0, stream>>>(
                a, a, 0i32, M, K, K, 1i32,
                b, b, 0i32, K, N, N, 1i32,
                c, c, 0i32, M, N, N, 1i32,
                alpha,
                beta
            ))?;
        }
    }
    stream.synchronize()?;
    Ok(())
}

/// Launches a flash_attention kernel over Q/K/V/O (each [rows, D]),
/// dispatching on wide exactly as [`launch_gemm`] does. Pointers are raw
/// device addresses, so this serves both the f32 and fp16 benches.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn launch_flash(
    func: &cust::function::Function,
    grid_x: u32,
    block: u32,
    stream: &cust::stream::Stream,
    q: cust::sys::CUdeviceptr,
    k: cust::sys::CUdeviceptr,
    v: cust::sys::CUdeviceptr,
    o: cust::sys::CUdeviceptr,
    Nq: i32,
    Nk: i32,
    D: i32,
    scale: f32,
    wide: bool,
) -> anyhow::Result<()> {
    let (nq64, nk64, d64) = (Nq as i64, Nk as i64, D as i64);
    unsafe {
        if wide {
            launch!(func<<<grid_x, block, 0, stream>>>(
                q, q, 0i64, nq64, d64, d64, 1i64,
                k, k, 0i64, nk64, d64, d64, 1i64,
                v, v, 0i64, nk64, d64, d64, 1i64,
                o, o, 0i64, nq64, d64, d64, 1i64,
                scale
            ))?;
        } else {
            launch!(func<<<grid_x, block, 0, stream>>>(
                q, q, 0i32, Nq, D, D, 1i32,
                k, k, 0i32, Nk, D, D, 1i32,
                v, v, 0i32, Nk, D, D, 1i32,
                o, o, 0i32, Nq, D, D, 1i32,
                scale
            ))?;
        }
    }
    stream.synchronize()?;
    Ok(())
}

/// A full CPU reference is 2*M*N*K flops (137 GFLOP at 4096^3), so spot-check
/// a sample of output elements against an f64 reference instead.
///
/// Expected value: alpha * (A*B)[i,j] + beta * c_in[i,j].
///
/// The f32 kernel accumulates in f32 over K=4096 terms (worst-case relative
/// error ~ K*eps ~ 2.4e-4): 1e-3 relative tolerance. The tensor-core kernel
/// additionally rounds each input to fp16 (eps ~ 4.9e-4), an error relative
/// to the dot's RMS magnitude sqrt(K)/3 (uniform [-1,1] inputs), not to want;
/// near-cancelling outputs would blow up a plain relative test, so its
/// errors are normalized by max(|want|, sqrt(K)/3) with a 1e-2 tolerance.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)] // mirrors the BLAS gemm signature
fn verify_matmul(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    c_in: &[f32],
    alpha: f64,
    beta: f64,
    M: usize,
    N: usize,
    K: usize,
    fp16: bool,
) -> anyhow::Result<()> {
    let (tol, floor) = if fp16 {
        (1e-2, (K as f64).sqrt() / 3.0)
    } else {
        (1e-3, f64::MIN_POSITIVE)
    };
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut sample = |limit: usize| {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as usize) % limit
    };
    for s in 0..64 {
        let (i, j) = match s {
            0 => (0, 0),
            1 => (M - 1, N - 1),
            _ => (sample(M), sample(N)),
        };
        let (want, got) = if fp16 {
            // The reference must see the same fp16-rounded inputs the
            // tensor cores do.
            let mut dot = 0.0f64;
            for k in 0..K {
                dot += fp16_round(a[i * K + k]) as f64 * fp16_round(b[k * N + j]) as f64;
            }
            (
                alpha * dot + beta * c_in[i * N + j] as f64,
                c[i * N + j] as f64,
            )
        } else {
            let mut dot = 0.0f64;
            for k in 0..K {
                dot += a[i * K + k] as f64 * b[k * N + j] as f64;
            }
            (
                alpha * dot + beta * c_in[i * N + j] as f64,
                c[i * N + j] as f64,
            )
        };
        let rel = (got - want).abs() / want.abs().max(floor);
        anyhow::ensure!(
            rel < tol,
            "c[{i},{j}] = {got:e}, want ~ {want:e} (rel err {rel:.2e})"
        );
    }
    Ok(())
}

fn fp16_round(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)] // gemm dims plus the result sink
fn bench_gemm_fp32(
    stream: &cust::stream::Stream,
    code: &'static str,
    name: &str,
    label: &str,
    fp16: bool,
    alpha: f32,
    beta: f32,
    pins: &HashMap<String, i64>,
    results: &mut Results,
) -> anyhow::Result<()> {
    let kernels = phobos_lang::parse(code)?;
    let space = autotune::pin(phobos_lang::search_space(&kernels[0]), pins)?;

    let M: i32 = 4096;
    let N: i32 = 4096;
    let K: i32 = 4096;
    let (m64, n64, k64) = (M as i64, N as i64, K as i64);

    let mut rng = SmallRng::seed_from_u64(42);
    let a: Vec<f32> = (0..(M * K)).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();
    let b: Vec<f32> = (0..(K * N)).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();

    let c_in = vec![0.0f32; (M * N) as usize];
    let mut c = c_in.clone();

    let a_dev = a.as_slice().as_dbuf()?;
    let b_dev = b.as_slice().as_dbuf()?;
    let c_dev = c.as_slice().as_dbuf()?;

    let (a_ptr, b_ptr, c_ptr) = (
        a_dev.as_device_ptr(),
        b_dev.as_device_ptr(),
        c_dev.as_device_ptr(),
    );

    // Match the kernel's @launch thread count (a PTX .maxntid); launching
    // more threads than that is a hard error.
    let block: u32 = kernels[0].cta_threads().map_err(anyhow::Error::msg)? as u32;

    // The default @tensorcore matmul path (mma.sync) compiles at 64-bit index,
    // widening the memref descriptor's offset/size/stride params from i32 to i64;
    // the host metadata must match (see bench_gemm_fp16).
    let wide = phobos_lang::requires_wide_index(&kernels);

    let mut tuner = autotune::Autotuner {
        code,
        grid_for: |cfg: &[autotune::Setting]| {
            let tile_m = autotune::cfg_value(cfg, "TILE_M")? as u32;
            let tile_n = autotune::cfg_value(cfg, "TILE_N")? as u32;
            anyhow::ensure!(
                (M as u32).is_multiple_of(tile_m) && (N as u32).is_multiple_of(tile_n),
                "tile does not divide the problem size"
            );
            Ok(autotune::Grid(M as u32 / tile_m, N as u32 / tile_n))
        },

        launch: |module: &cust::module::Module, grid: autotune::Grid| {
            zero_device_async(c_ptr.as_raw(), (M * N) as usize * 4, stream)?;
            let func = module.get_function("gemm")?;
            launch_gemm(
                &func,
                (grid.0, grid.1),
                block,
                stream,
                a_ptr.as_raw(),
                b_ptr.as_raw(),
                c_ptr.as_raw(),
                M,
                N,
                K,
                alpha,
                beta,
                wide,
            )
        },
        verify: || {
            c_dev.copy_to(&mut c)?;
            verify_matmul(
                &c,
                &a,
                &b,
                &c_in,
                alpha as f64,
                beta as f64,
                M as usize,
                N as usize,
                K as usize,
                fp16,
            )
        },
        short_probes: PROBES_SHORT,
        long_probes: PROBES_LONG,
        finalists: 4,
    };

    let winner = tuner.run(&space)?;
    let module = cust::module::Module::from_ptx(winner.ptx.as_str(), &[])?;
    let grid = (tuner.grid_for)(&winner.config)?;

    let (phobos_avg, _) = bench(label, || {
        (tuner.launch)(&module, grid)?;
        Ok(())
    })?;
    (tuner.verify)()?;

    let blas = cublas::CuBlas::new(stream)?;
    let (cublas_avg, _) = bench("cuBLAS sgemm ", || {
        zero_device_async(c_ptr.as_raw(), (M * N) as usize * 4, stream)?;
        blas.matmul(
            M,
            N,
            K,
            a_dev.as_device_ptr().as_raw(),
            b_dev.as_device_ptr().as_raw(),
            c_ptr.as_raw(),
            alpha,
            beta,
        )?;
        stream.synchronize()?;
        Ok(())
    })?;
    c_dev.copy_to(&mut c)?;

    verify_matmul(
        &c,
        &a,
        &b,
        &c_in,
        alpha as f64,
        beta as f64,
        M as usize,
        N as usize,
        K as usize,
        false,
    )?;

    phinfo!("check: {} elements, both correct", M * N);
    let gflop = (2.0 * m64 as f64 * n64 as f64 * k64 as f64 + m64 as f64 * n64 as f64) / 1e9;
    phinfo!(
        "phobos: {:.1} GFLOP/s, cuBLAS: {:.1} GFLOP/s",
        gflop / phobos_avg.as_secs_f64(),
        gflop / cublas_avg.as_secs_f64()
    );
    phinfo!(
        "phobos / cuBLAS: {:.2}x / {:.2}%",
        phobos_avg.as_secs_f64() / cublas_avg.as_secs_f64(),
        100.0f64 * cublas_avg.as_secs_f64() / phobos_avg.as_secs_f64()
    );

    // Inputs are rounded to fp16 for the tensor-core path (fp16), so phobos
    // runs against the fp16f32acc tensor peak; the cuBLAS baseline here is always
    // f32 sgemm.
    let phobos_prec = if fp16 {
        Precision::F16TcF32
    } else {
        Precision::F32
    };
    results.push(
        name,
        "phobos",
        phobos_prec,
        gflop / phobos_avg.as_secs_f64(),
    );
    results.push(
        name,
        "cuBLAS",
        Precision::F32,
        gflop / cublas_avg.as_secs_f64(),
    );

    Ok(())
}

/// Reference for the fp16-accumulate GEMM (examples/gemm_fp16.ph): the
/// kernel rounds every input to fp16, accumulates the dot in an fp16 WMMA
/// fragment, then scales (alpha/beta in f32) and rounds the result back
/// to fp16. The reference mirrors that, rounding each accumulation step to
/// fp16. fp16 accumulation over K=4096 terms is intentionally low precision
/// (~fp16 ulp at magnitude sqrt(K/3)); errors are normalized by
/// max(|want|, sqrt(K)/3) (the dot's RMS magnitude for uniform [-1,1]
/// inputs) with a generous 1.5e-1 tolerance, and the kernel's pairwise
/// fragment reduction is typically more accurate than this sequential model.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn verify_matmul_fp16acc(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    c_in: &[f32],
    alpha: f32,
    beta: f32,
    M: usize,
    N: usize,
    K: usize,
) -> anyhow::Result<()> {
    let (tol, floor) = (1.5e-1f64, (K as f64).sqrt() / 3.0);
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut sample = |limit: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as usize) % limit
    };
    for s in 0..64 {
        let (i, j) = match s {
            0 => (0, 0),
            1 => (M - 1, N - 1),
            _ => (sample(M), sample(N)),
        };
        // fp16 inputs, fp16 accumulation (round each step), mirroring the kernel.
        let mut acc = 0.0f32;
        for k in 0..K {
            let af = fp16_round(a[i * K + k]);
            let bf = fp16_round(b[k * N + j]);
            acc = fp16_round(acc + af * bf);
        }
        let want = fp16_round(alpha * acc + beta * c_in[i * N + j]) as f64;
        let got = c[i * N + j] as f64;
        let rel = (got - want).abs() / want.abs().max(floor);
        anyhow::ensure!(
            rel < tol,
            "c[{i},{j}] = {got:e}, want ~ {want:e} (rel err {rel:.2e})"
        );
    }
    Ok(())
}

/// fp32 -> fp16 bit pattern, for uploading host data to an fp16 device tensor.
fn to_fp16_bits(xs: &[f32]) -> Vec<u16> {
    xs.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

/// Reads back an fp16 device tensor (u16 bit patterns) as fp32.
fn from_fp16_bits(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

/// Half-precision GEMM benchmark (examples/gemm_fp16.ph): fp16 A/B/C and an
/// fp16 accumulator on the tensor cores. Data lives on the device as fp16
/// (uploaded as u16 bit patterns; the kernel reads it as fp16). Compared
/// against cublasHgemm, the matching fp16-operand, fp16-accumulate cuBLAS gemm.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)] // gemm dims plus the result sink
fn bench_gemm_fp16(
    stream: &cust::stream::Stream,
    code: &'static str,
    name: &str,
    label: &str,
    alpha: f32,
    beta: f32,
    pins: &HashMap<String, i64>,
    results: &mut Results,
) -> anyhow::Result<()> {
    let kernels = phobos_lang::parse(code)?;
    let space = autotune::pin(phobos_lang::search_space(&kernels[0]), pins)?;

    let M: i32 = 4096;
    let N: i32 = 4096;
    let K: i32 = 4096;
    let (m64, n64, k64) = (M as i64, N as i64, K as i64);

    let mut rng = SmallRng::seed_from_u64(42);
    let a: Vec<f32> = (0..(M * K)).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();
    let b: Vec<f32> = (0..(K * N)).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();
    let c_in = vec![0.0f32; (M * N) as usize];

    // fp16 device tensors, stored as u16 bit patterns (bit-compatible with the
    // kernel's fp16 reads; 0 bits = +0.0).
    let a_dev = to_fp16_bits(&a).as_slice().as_dbuf()?;
    let b_dev = to_fp16_bits(&b).as_slice().as_dbuf()?;
    let c_dev = vec![0u16; (M * N) as usize].as_slice().as_dbuf()?;
    let mut c_bits = vec![0u16; (M * N) as usize];

    let (a_ptr, b_ptr, c_ptr) = (
        a_dev.as_device_ptr(),
        b_dev.as_device_ptr(),
        c_dev.as_device_ptr(),
    );

    let block: u32 = kernels[0].cta_threads().map_err(anyhow::Error::msg)? as u32;

    // @tensorcore (the default mma.sync path) compiles at 64-bit index (nvgpu
    // ABI), which widens the flattened memref descriptor's offset/size/stride
    // params from i32 to i64; the host metadata must match or every field after
    // the first pointer shifts and the kernel reads garbage. The legacy WMMA
    // opt-out (@tensorcore(wmma)) stays 32-bit.
    let wide = phobos_lang::requires_wide_index(&kernels);

    let mut tuner = autotune::Autotuner {
        code,
        grid_for: |cfg: &[autotune::Setting]| {
            let tile_m = autotune::cfg_value(cfg, "TILE_M")? as u32;
            let tile_n = autotune::cfg_value(cfg, "TILE_N")? as u32;
            anyhow::ensure!(
                (M as u32).is_multiple_of(tile_m) && (N as u32).is_multiple_of(tile_n),
                "tile does not divide the problem size"
            );
            Ok(autotune::Grid(M as u32 / tile_m, N as u32 / tile_n))
        },
        launch: |module: &cust::module::Module, grid: autotune::Grid| {
            // Zero C (fp16 +0.0 is all-zero bits): a byte memset over 2 bytes
            // per element.
            zero_device_async(c_ptr.as_raw(), (M * N) as usize * 2, stream)?;
            let func = module.get_function("gemm")?;
            launch_gemm(
                &func,
                (grid.0, grid.1),
                block,
                stream,
                a_ptr.as_raw(),
                b_ptr.as_raw(),
                c_ptr.as_raw(),
                M,
                N,
                K,
                alpha,
                beta,
                wide,
            )
        },
        verify: || {
            c_dev.copy_to(&mut c_bits)?;
            let c = from_fp16_bits(&c_bits);
            verify_matmul_fp16acc(
                &c, &a, &b, &c_in, alpha, beta, M as usize, N as usize, K as usize,
            )
        },
        short_probes: PROBES_SHORT,
        long_probes: PROBES_LONG,
        finalists: 4,
    };

    let winner = tuner.run(&space)?;
    let module = cust::module::Module::from_ptx(winner.ptx.as_str(), &[])?;
    let grid = (tuner.grid_for)(&winner.config)?;

    let (phobos_avg, _) = bench(label, || {
        (tuner.launch)(&module, grid)?;
        Ok(())
    })?;
    (tuner.verify)()?;

    let blas = cublas::CuBlas::new(stream)?;
    let (cublas_avg, _) = bench("cuBLAS hgemm ", || {
        // Zero C (fp16 +0.0 is all-zero bits): 2 bytes per element.
        zero_device_async(c_ptr.as_raw(), (M * N) as usize * 2, stream)?;
        blas.matmul_fp16(
            M,
            N,
            K,
            a_dev.as_device_ptr().as_raw(),
            b_dev.as_device_ptr().as_raw(),
            c_ptr.as_raw(),
            alpha,
            beta,
        )?;
        stream.synchronize()?;
        Ok(())
    })?;
    c_dev.copy_to(&mut c_bits)?;
    let c = from_fp16_bits(&c_bits);
    verify_matmul_fp16acc(
        &c, &a, &b, &c_in, alpha, beta, M as usize, N as usize, K as usize,
    )?;

    phinfo!("check: {} elements, both correct", M * N);
    let gflop = (2.0 * m64 as f64 * n64 as f64 * k64 as f64 + m64 as f64 * n64 as f64) / 1e9;
    phinfo!(
        "phobos gemm_fp16: {:.1} GFLOP/s, cuBLAS hgemm: {:.1} GFLOP/s",
        gflop / phobos_avg.as_secs_f64(),
        gflop / cublas_avg.as_secs_f64()
    );
    phinfo!(
        "phobos / cuBLAS: {:.2}x / {:.2}%",
        phobos_avg.as_secs_f64() / cublas_avg.as_secs_f64(),
        100.0f64 * cublas_avg.as_secs_f64() / phobos_avg.as_secs_f64()
    );

    // fp16 operands on the tensor cores; the cuBLAS baseline is hgemm (also fp16).
    results.push(
        name,
        "phobos",
        Precision::F16Tc,
        gflop / phobos_avg.as_secs_f64(),
    );
    results.push(
        name,
        "cuBLAS",
        Precision::F16Tc,
        gflop / cublas_avg.as_secs_f64(),
    );

    Ok(())
}

/// Flash attention: O = softmax(scale * Q @ K.T) @ V, with the softmax taken
/// row-wise over the Nk keys. A full reference is Nq*Nk*D work, so (as with
/// the large matmul) spot-check a sample of output elements against an f64
/// reference instead.
///
/// The reference replays the kernel's own online-softmax recurrence in f64, so
/// the two agree on algorithm and differ only in precision. Each output is a
/// convex combination of the V rows, hence bounded by max|V| ~ 1; the f32
/// kernel's absolute error is dominated by the Nk-term accumulation
/// (~Nk*eps ~ 2.4e-4) plus the f32 exp. Errors are normalized by
/// max(|want|, 0.1) (an absolute floor, since near-zero averages should not
/// blow up a plain relative test). The @tensorcore path rounds both the
/// scores (Q @ K.T) and P @ V through fp16 fragments, so the tolerance is the
/// looser fp16-grade 2e-2 (the f32 fallback configs clear it comfortably).
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn verify_flash_attention(
    o: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    scale: f64,
    Nq: usize,
    Nk: usize,
    D: usize,
) -> anyhow::Result<()> {
    let (tol, floor) = (2e-2, 1e-1);
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut sample = |limit: usize| {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as usize) % limit
    };
    for s in 0..64 {
        let (i, d) = match s {
            0 => (0, 0),
            1 => (Nq - 1, D - 1),
            _ => (sample(Nq), sample(D)),
        };
        // Online softmax over the keys, mirroring the kernel's recurrence.
        let mut m = f64::NEG_INFINITY;
        let mut l = 0.0f64;
        let mut acc = 0.0f64;
        for j in 0..Nk {
            let mut score = 0.0f64;
            for e in 0..D {
                score += q[i * D + e] as f64 * k[j * D + e] as f64;
            }
            score *= scale;
            let mnew = m.max(score);
            let corr = (m - mnew).exp();
            let p = (score - mnew).exp();
            l = l * corr + p;
            acc = acc * corr + p * v[j * D + d] as f64;
            m = mnew;
        }
        let want = acc / l;
        let got = o[i * D + d] as f64;
        let rel = (got - want).abs() / want.abs().max(floor);
        anyhow::ensure!(
            rel < tol,
            "O[{i},{d}] = {got:e}, want ~ {want:e} (rel err {rel:.2e})"
        );
    }
    Ok(())
}

#[allow(non_snake_case)]
fn bench_flash_attention_fp32(
    stream: &cust::stream::Stream,
    pins: &HashMap<String, i64>,
    results: &mut Results,
) -> anyhow::Result<()> {
    let kernels = phobos_lang::parse(CODE_FLASH)?;
    let space = autotune::pin(phobos_lang::search_space(&kernels[0]), pins)?;

    // D is pinned to 64 by the kernel's @autotune(D in [64]) and must match
    // the static head-dim of the tensors.
    let Nq: i32 = 4096;
    let Nk: i32 = 4096;
    let D: i32 = 64;
    let scale: f32 = 1.0 / (D as f32).sqrt();

    let mut rng = SmallRng::seed_from_u64(42);
    let q: Vec<f32> = (0..(Nq * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();
    let k: Vec<f32> = (0..(Nk * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();
    let v: Vec<f32> = (0..(Nk * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();
    let mut o = vec![0.0f32; (Nq * D) as usize];

    let q_dev = q.as_slice().as_dbuf()?;
    let k_dev = k.as_slice().as_dbuf()?;
    let v_dev = v.as_slice().as_dbuf()?;
    let o_dev = o.as_slice().as_dbuf()?;

    let (q_ptr, k_ptr, v_ptr, o_ptr) = (
        q_dev.as_device_ptr(),
        k_dev.as_device_ptr(),
        v_dev.as_device_ptr(),
        o_dev.as_device_ptr(),
    );

    // Launch the block size the kernel was compiled for: @launch sets a
    // PTX .maxntid, and launching more threads than that is a hard error.
    let block: u32 = kernels[0].cta_threads().map_err(anyhow::Error::msg)? as u32;

    // @tensorcore compiles at 64-bit index (the default mma.sync path), widening
    // the memref descriptor's offset/size/stride params from i32 to i64; the
    // host metadata must match. (Flash attention's dots still run on legacy WMMA
    // via wmma_dot, but the wide index is forced by the kernel attr regardless.)
    let wide = phobos_lang::requires_wide_index(&kernels);

    let mut tuner = autotune::Autotuner {
        code: CODE_FLASH,
        grid_for: |cfg: &[autotune::Setting]| {
            let br = autotune::cfg_value(cfg, "BR")? as u32;
            anyhow::ensure!(
                (Nq as u32).is_multiple_of(br),
                "BR does not divide the query length"
            );
            Ok(autotune::Grid(Nq as u32 / br, 1))
        },
        launch: |module: &cust::module::Module, grid: autotune::Grid| {
            // Each tensor is a memref<?x64>; the kernel writes all of O, so no
            // memset.
            let func = module.get_function("flash_attention")?;
            launch_flash(
                &func,
                grid.0,
                block,
                stream,
                q_ptr.as_raw(),
                k_ptr.as_raw(),
                v_ptr.as_raw(),
                o_ptr.as_raw(),
                Nq,
                Nk,
                D,
                scale,
                wide,
            )
        },
        verify: || {
            o_dev.copy_to(&mut o)?;
            verify_flash_attention(
                &o,
                &q,
                &k,
                &v,
                scale as f64,
                Nq as usize,
                Nk as usize,
                D as usize,
            )
        },
        short_probes: PROBES_SHORT,
        long_probes: PROBES_LONG,
        finalists: 4,
    };

    let winner = tuner.run(&space)?;
    let module = cust::module::Module::from_ptx(winner.ptx.as_str(), &[])?;
    let grid = (tuner.grid_for)(&winner.config)?;

    let (phobos_avg, _) = bench("phobos flash", || {
        (tuner.launch)(&module, grid)?;
        Ok(())
    })?;
    (tuner.verify)()?;

    phinfo!("check: 64 probes, correct");
    // Two Nq*Nk*D matmuls (Q@K.T and P@V), 2 flops each.
    let gflop = 4.0 * Nq as f64 * Nk as f64 * D as f64 / 1e9;
    phinfo!(
        "phobos flash_fp32: {:.1} GFLOP/s",
        gflop / phobos_avg.as_secs_f64()
    );

    results.push(
        "flash_fp32",
        "phobos",
        Precision::F32,
        gflop / phobos_avg.as_secs_f64(),
    );

    Ok(())
}

/// Half-precision flash attention benchmark (examples/flash_attention_fp16.ph):
/// fp16 Q/K/V/O with an f32 online-softmax state, both matmuls on the tensor
/// cores. Inputs/outputs live on the device as fp16 (u16 bit patterns). The
/// reference replays the recurrence in f64 over the fp16-rounded inputs, so the
/// fp16-grade 2e-2 tolerance from the f32 tensor-core path applies.
#[allow(non_snake_case)]
fn bench_flash_attention_fp16(
    stream: &cust::stream::Stream,
    pins: &HashMap<String, i64>,
    results: &mut Results,
) -> anyhow::Result<()> {
    let kernels = phobos_lang::parse(CODE_FLASH_F16)?;
    let space = autotune::pin(phobos_lang::search_space(&kernels[0]), pins)?;

    let Nq: i32 = 4096;
    let Nk: i32 = 4096;
    let D: i32 = 64;
    let scale: f32 = 1.0 / (D as f32).sqrt();

    let mut rng = SmallRng::seed_from_u64(42);
    let q: Vec<f32> = (0..(Nq * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();
    let k: Vec<f32> = (0..(Nk * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();
    let v: Vec<f32> = (0..(Nk * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();

    // The reference must see the same fp16-rounded inputs the kernel does.
    let qf: Vec<f32> = q.iter().map(|&x| fp16_round(x)).collect();
    let kf: Vec<f32> = k.iter().map(|&x| fp16_round(x)).collect();
    let vf: Vec<f32> = v.iter().map(|&x| fp16_round(x)).collect();

    let q_dev = to_fp16_bits(&q).as_slice().as_dbuf()?;
    let k_dev = to_fp16_bits(&k).as_slice().as_dbuf()?;
    let v_dev = to_fp16_bits(&v).as_slice().as_dbuf()?;
    let o_dev = vec![0u16; (Nq * D) as usize].as_slice().as_dbuf()?;
    let mut o_bits = vec![0u16; (Nq * D) as usize];

    let (q_ptr, k_ptr, v_ptr, o_ptr) = (
        q_dev.as_device_ptr(),
        k_dev.as_device_ptr(),
        v_dev.as_device_ptr(),
        o_dev.as_device_ptr(),
    );

    let block: u32 = kernels[0].cta_threads().map_err(anyhow::Error::msg)? as u32;

    // @tensorcore compiles at 64-bit index (the default mma.sync path); the host
    // descriptor metadata must match (see bench_flash_attention_fp32).
    let wide = phobos_lang::requires_wide_index(&kernels);

    let mut tuner = autotune::Autotuner {
        code: CODE_FLASH_F16,
        grid_for: |cfg: &[autotune::Setting]| {
            let br = autotune::cfg_value(cfg, "BR")? as u32;
            anyhow::ensure!(
                (Nq as u32).is_multiple_of(br),
                "BR does not divide the query length"
            );
            Ok(autotune::Grid(Nq as u32 / br, 1))
        },
        launch: |module: &cust::module::Module, grid: autotune::Grid| {
            let func = module.get_function("flash_attention")?;
            launch_flash(
                &func,
                grid.0,
                block,
                stream,
                q_ptr.as_raw(),
                k_ptr.as_raw(),
                v_ptr.as_raw(),
                o_ptr.as_raw(),
                Nq,
                Nk,
                D,
                scale,
                wide,
            )
        },
        verify: || {
            o_dev.copy_to(&mut o_bits)?;
            let o = from_fp16_bits(&o_bits);
            verify_flash_attention(
                &o,
                &qf,
                &kf,
                &vf,
                scale as f64,
                Nq as usize,
                Nk as usize,
                D as usize,
            )
        },
        short_probes: PROBES_SHORT,
        long_probes: PROBES_LONG,
        finalists: 4,
    };

    let winner = tuner.run(&space)?;
    let module = cust::module::Module::from_ptx(winner.ptx.as_str(), &[])?;
    let grid = (tuner.grid_for)(&winner.config)?;

    let (phobos_avg, _) = bench("phobos flash_fp16", || {
        (tuner.launch)(&module, grid)?;
        Ok(())
    })?;
    (tuner.verify)()?;

    phinfo!("check: 64 probes, correct");
    let gflop = 4.0 * Nq as f64 * Nk as f64 * D as f64 / 1e9;
    phinfo!(
        "phobos flash_fp16: {:.1} GFLOP/s",
        gflop / phobos_avg.as_secs_f64()
    );

    results.push(
        "flash_fp16",
        "phobos",
        Precision::F16TcF32,
        gflop / phobos_avg.as_secs_f64(),
    );

    Ok(())
}

/// SAXPY reference: out = alpha * x + y. The kernel lowers to a separate
/// mulf then addf in f32 (no fused multiply-add), so an exact f32 match is
/// expected.
fn verify_saxpy(out: &[f32], x: &[f32], y: &[f32], alpha: f32) {
    for i in 0..out.len() {
        let want = alpha * x[i] + y[i];
        assert_eq!(
            out[i], want,
            "expected {}, got {} at index {}",
            want, out[i], i
        );
    }
}

fn bench_saxpy(
    stream: &cust::stream::Stream,
    pins: &HashMap<String, i64>,
    results: &mut Results,
) -> anyhow::Result<()> {
    let kernels = phobos_lang::parse(CODE_SAXPY)?;
    let space = autotune::pin(phobos_lang::search_space(&kernels[0]), pins)?;

    let n: i32 = 1 << 25;
    let alpha: f32 = 2.0;

    let mut rng = SmallRng::seed_from_u64(42);
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();
    let mut out = vec![0.0f32; n as usize];

    let x_dev = x.as_slice().as_dbuf()?;
    let y_dev = y.as_slice().as_dbuf()?;
    let out_dev = out.as_slice().as_dbuf()?;

    let (x_ptr, y_ptr, out_ptr) = (
        x_dev.as_device_ptr(),
        y_dev.as_device_ptr(),
        out_dev.as_device_ptr(),
    );

    let block: u32 = 1024;

    let mut tuner = autotune::Autotuner {
        code: CODE_SAXPY,
        grid_for: |cfg: &[autotune::Setting]| {
            let block_size = autotune::cfg_value(cfg, "BLOCK")? as u32;
            Ok(autotune::Grid((n as u32).div_ceil(block_size), 1))
        },
        launch: |module: &cust::module::Module, grid: autotune::Grid| {
            let func = module.get_function("saxpy")?;
            let grid_x = grid.0;
            // Kernel params: x, y, out (each a tensor<f32>[N] => 5-scalar
            // memref descriptor), then the alpha scalar.
            unsafe {
                launch!(func<<<grid_x, block, 0, stream>>>(
                    x_ptr, x_ptr, 0i32, n, 1i32,
                    y_ptr, y_ptr, 0i32, n, 1i32,
                    out_ptr, out_ptr, 0i32, n, 1i32,
                    alpha
                ))?;
            }
            stream.synchronize()?;
            Ok(())
        },
        verify: || {
            out_dev.copy_to(&mut out)?;
            verify_saxpy(&out, &x, &y, alpha);
            Ok(())
        },
        short_probes: PROBES_SHORT,
        long_probes: PROBES_LONG,
        finalists: 4,
    };

    let winner = tuner.run(&space)?;

    let module = cust::module::Module::from_ptx(winner.ptx.as_str(), &[])?;
    let grid = (tuner.grid_for)(&winner.config)?;

    let (phobos_avg, _) = bench("phobos saxpy", || {
        (tuner.launch)(&module, grid)?;
        Ok(())
    })?;
    (tuner.verify)()?;

    // cuBLAS saxpy is in-place (y := alpha*x + y), so it runs on a dedicated
    // buffer. The timed loop lets it accumulate (irrelevant to the runtime);
    // correctness is checked separately on a freshly reset buffer.
    let mut yb = y.clone();
    let mut yb_dev = y.as_slice().as_dbuf()?;
    let yb_ptr = yb_dev.as_device_ptr();

    let blas = cublas::CuBlas::new(stream)?;
    let (cublas_avg, _) = bench("cuBLAS saxpy", || {
        blas.saxpy(n, alpha, x_dev.as_device_ptr().as_raw(), yb_ptr.as_raw())?;
        stream.synchronize()?;
        Ok(())
    })?;
    yb_dev.copy_from(y.as_slice())?;
    blas.saxpy(n, alpha, x_dev.as_device_ptr().as_raw(), yb_ptr.as_raw())?;
    stream.synchronize()?;
    yb_dev.copy_to(&mut yb)?;
    verify_saxpy(&yb, &x, &y, alpha);

    phinfo!("check: {} elements, both correct", n);
    // SAXPY is one fused multiply-add per element: 2 flops. It is memory-bound,
    // so this sits far below the f32 FLOP peak, but it is reported like any
    // other bench.
    let gflop = 2.0 * n as f64 / 1e9;
    phinfo!(
        "phobos saxpy: {:.1} GFLOP/s, cuBLAS saxpy: {:.1} GFLOP/s",
        gflop / phobos_avg.as_secs_f64(),
        gflop / cublas_avg.as_secs_f64()
    );
    phinfo!(
        "phobos / cuBLAS: {:.2}x",
        phobos_avg.as_secs_f64() / cublas_avg.as_secs_f64()
    );

    results.push(
        "saxpy_fp32",
        "phobos",
        Precision::F32,
        gflop / phobos_avg.as_secs_f64(),
    );
    results.push(
        "saxpy_fp32",
        "cuBLAS",
        Precision::F32,
        gflop / cublas_avg.as_secs_f64(),
    );

    Ok(())
}
