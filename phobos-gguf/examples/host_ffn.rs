//! Times the host expert kernels on a mixture-of-experts file, no GPU, in
//! the two shapes the runtime runs them: a batch of experts each over its
//! own rows, as a prompt pass's host share, from the file's layout and
//! from the device's grouped one; and one row's few experts spread over
//! the pool by rows, as a decode step's misses. The reference decoder's
//! time for one expert stands beside them.
//!
//! `host_ffn MODEL [--rows 1,4,16,64] [--threads N] [--reps R]`

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use phobos_gguf::Gguf;
use phobos_gguf::experts::{ExpertSet, Stack};
use phobos_gguf::qwen35::Config;
use phobos_gguf::simd::{self, Job, Shape, Source, Weight};

struct Args {
    model: String,
    rows: Vec<usize>,
    threads: Option<usize>,
    reps: usize,
}

fn parse() -> Result<Args> {
    let mut args = Args { model: String::new(), rows: vec![1, 4, 16, 64], threads: None, reps: 3 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().with_context(|| format!("{arg} takes a value"));
        match arg.as_str() {
            "--rows" => args.rows = value()?.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>().context("--rows")?,
            "--threads" => args.threads = Some(value()?.parse().context("--threads")?),
            "--reps" => args.reps = value()?.parse().context("--reps")?,
            other if other.starts_with("--") => bail!("unknown flag {other}"),
            _ => args.model = arg,
        }
    }
    if args.model.is_empty() {
        bail!("usage: host_ffn MODEL [--rows 1,4,16,64] [--threads N] [--reps R]");
    }
    Ok(args)
}

/// Floats in [-0.5, 0.5) from a seed.
fn random(mut seed: u64) -> impl FnMut() -> f32 {
    move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

/// The first block of each down-stack format, so every kernel gets timed.
fn representative_blocks(gguf: &Gguf, blocks: usize) -> Vec<usize> {
    let mut seen = Vec::new();
    let mut picks = Vec::new();
    for b in 0..blocks {
        let Some(info) = gguf.tensor(&format!("blk.{b}.ffn_down_exps.weight")) else { continue };
        if !seen.contains(&info.ggml_type) {
            seen.push(info.ggml_type);
            picks.push(b);
        }
    }
    picks
}

/// A block's experts copied into the device's grouped layout in ordinary
/// memory, the layout the pinned mirror holds.
struct GroupedCopy<'a> {
    set: &'a ExpertSet,
    bytes: [Vec<u8>; 3],
    scales: [Vec<u16>; 3],
}

impl<'a> GroupedCopy<'a> {
    fn build(set: &'a ExpertSet) -> GroupedCopy<'a> {
        let mut bytes: [Vec<u8>; 3] = Default::default();
        let mut scales: [Vec<u16>; 3] = Default::default();
        for (i, &stack) in Stack::ALL.iter().enumerate() {
            let s = set.stack(stack);
            let (per, plane) = (s.grouped_bytes(), s.grouped_scales());
            bytes[i] = vec![0; s.count() * per];
            scales[i] = vec![0; s.count() * plane];
            for e in 0..s.count() {
                s.grouped_into(e, &mut bytes[i][e * per..(e + 1) * per]);
                s.grouped_scales_into(e, &mut scales[i][e * plane..(e + 1) * plane]);
            }
        }
        GroupedCopy { set, bytes, scales }
    }
}

impl Source for GroupedCopy<'_> {
    fn shape(&self, stack: Stack) -> Shape {
        self.set.shape(stack)
    }

    fn weight(&self, stack: Stack, e: usize) -> Weight<'_> {
        let (i, s) = (stack as usize, self.set.stack(stack));
        let (per, plane) = (s.grouped_bytes(), s.grouped_scales());
        Weight::Grouped {
            bytes: &self.bytes[i][e * per..(e + 1) * per],
            block_bytes: s.quant().device_block_bytes(),
            nb: s.blocks_per_row(),
            scales: &self.scales[i][e * plane..(e + 1) * plane],
        }
    }
}

/// A batch of `jobs_n` experts, each over `rows` rows, `reps` times.
fn batch_shape(source: &impl Source, label: &str, rows_list: &[usize], n_expert: usize, reps: usize) -> Result<()> {
    let (d, d_ff) = (source.shape(Stack::Gate).k, source.shape(Stack::Gate).n);
    let mut next = random(0x2545_f491_4f6c_dd1d);
    let jobs_n = 48;
    println!("{jobs_n} experts as jobs over the pool, {label}:");
    println!("{:>6} {:>12} {:>14} {:>10}", "rows", "us/expert", "us/expert-row", "GFLOP/s");
    for &rows in rows_list {
        let x: Vec<f32> = (0..rows * d).map(|_| next()).collect();
        let jobs: Vec<Job> = (0..jobs_n).map(|e| Job { expert: e % n_expert, rows: (0..rows).collect(), weights: vec![1.0; rows] }).collect();
        let mut out = vec![0.0f32; rows * d];
        simd::experts_ffn(source, &jobs, &x, rows, &mut out)?;
        let started = Instant::now();
        for _ in 0..reps {
            simd::experts_ffn(source, &jobs, &x, rows, &mut out)?;
        }
        let micros = started.elapsed().as_secs_f64() * 1e6 / (reps * jobs_n) as f64;
        println!("{rows:>6} {micros:>12.1} {:>14.2} {:>10.1}", micros / rows as f64, (rows * 6 * d * d_ff) as f64 / micros * 1e-3);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = parse()?;
    if let Some(threads) = args.threads {
        simd::init_pool(threads)?;
    }
    let gguf = Gguf::open(Path::new(&args.model))?;
    let config = Config::from_gguf(&gguf)?;
    let moe = config.moe.context("the model has no experts")?;
    let (d, d_ff) = (config.d_model, moe.d_expert);
    println!("{} experts of [{d_ff}, {d}] a block, {} host threads, avx2 {}", moe.n_expert, simd::threads(), simd::avx2());
    let mut next = random(0x9e37_79b9_7f4a_7c15);

    for block in representative_blocks(&gguf, config.n_block) {
        let set = ExpertSet::load(&gguf, &format!("blk.{block}"), moe.n_expert, d, d_ff)?;
        println!(
            "block {block}: gate/up {}, down {}, {:.2} MiB an expert",
            set.gate.quant().name(),
            set.down.quant().name(),
            set.expert_bytes() as f64 / (1 << 20) as f64
        );
        batch_shape(&*set, "file layout", &args.rows, moe.n_expert, args.reps)?;
        let grouped = GroupedCopy::build(&set);
        batch_shape(&grouped, "grouped layout", &args.rows, moe.n_expert, args.reps)?;

        // A decode step's misses: two experts over one row.
        let x: Vec<f32> = (0..d).map(|_| next()).collect();
        let mut out = vec![0.0f32; d];
        let calls = 40;
        let started = Instant::now();
        for i in 0..calls {
            let misses = [((7 * i) % moe.n_expert, 1.0), ((7 * i + 3) % moe.n_expert, 1.0)];
            simd::experts_row(&grouped, &misses, &x, &mut out)?;
        }
        println!("two misses of one row, spread by rows: {:.0} us a call", started.elapsed().as_secs_f64() * 1e6 / calls as f64);

        // The reference decoder: one expert, one row, one thread.
        let started = Instant::now();
        let mut dense = vec![0.0f32; d_ff * d];
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f32>();
        set.gate.dequantize(0, &mut dense)?;
        let g: Vec<f32> = dense.chunks_exact(d).map(|w| dot(w, &x)).collect();
        set.up.dequantize(0, &mut dense)?;
        let h: Vec<f32> = dense.chunks_exact(d).zip(&g).map(|(w, &g)| simd::swiglu(g, dot(w, &x))).collect();
        set.down.dequantize(0, &mut dense)?;
        let y: f32 = dense.chunks_exact(d_ff).map(|w| dot(w, &h)).sum();
        println!("reference decoder, one row, one thread: {:.0} us (checksum {y:.3})", started.elapsed().as_secs_f64() * 1e6);
    }
    Ok(())
}
