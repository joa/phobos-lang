//! Times the host expert kernels on a mixture-of-experts file, no GPU: one
//! expert's gate, up and down over a batch of rows, in microseconds an
//! expert-row and GFLOP/s, against the reference decoder for one row.
//!
//! `host_ffn MODEL [--rows 1,4,16,64] [--threads N] [--experts E]`

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use phobos_gguf::experts::ExpertSet;
use phobos_gguf::qwen35::Config;
use phobos_gguf::simd::{self, Job, Q8Act, Scratch, Source, Stack, Weight};
use phobos_gguf::Gguf;

struct Args {
    model: String,
    rows: Vec<usize>,
    threads: Option<usize>,
    experts: usize,
}

fn parse() -> Result<Args> {
    let mut args = Args { model: String::new(), rows: vec![1, 4, 16, 64], threads: None, experts: 8 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().with_context(|| format!("{arg} takes a value"));
        match arg.as_str() {
            "--rows" => args.rows = value()?.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>().context("--rows")?,
            "--threads" => args.threads = Some(value()?.parse().context("--threads")?),
            "--experts" => args.experts = value()?.parse().context("--experts")?,
            other if other.starts_with("--") => bail!("unknown flag {other}"),
            _ => args.model = arg,
        }
    }
    if args.model.is_empty() {
        bail!("usage: host_ffn MODEL [--rows 1,4,16,64] [--threads N] [--experts E]");
    }
    Ok(args)
}

/// The blocks whose down stack differs in format from the first block's:
/// the first of each format, so every kernel gets timed.
fn representative_blocks(gguf: &Gguf, blocks: usize) -> Vec<usize> {
    let mut seen = Vec::new();
    let mut picks = Vec::new();
    for b in 0..blocks {
        let Some(info) = gguf.tensor(&format!("blk.{b}.ffn_down_exps.weight")) else { continue };
        let kind = info.ggml_type;
        if !seen.contains(&kind) {
            seen.push(kind);
            picks.push(b);
        }
    }
    picks
}

/// A block's experts copied into the device's grouped layout in ordinary
/// memory, the layout the pinned mirror holds, to time that path alone.
struct GroupedCopy<'a> {
    set: &'a ExpertSet,
    bytes: [Vec<u8>; 3],
    scales: [Vec<u16>; 3],
}

impl<'a> GroupedCopy<'a> {
    fn build(set: &'a ExpertSet) -> GroupedCopy<'a> {
        let stacks = [Stack::Gate, Stack::Up, Stack::Down];
        let mut bytes: [Vec<u8>; 3] = Default::default();
        let mut scales: [Vec<u16>; 3] = Default::default();
        for (i, &stack) in stacks.iter().enumerate() {
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
    fn shape(&self, stack: Stack) -> (phobos_gguf::quant::Quant, usize, usize) {
        self.set.shape(stack)
    }

    fn weight(&self, stack: Stack, e: usize) -> Weight<'_> {
        let i = match stack {
            Stack::Gate => 0,
            Stack::Up => 1,
            Stack::Down => 2,
        };
        let s = self.set.stack(stack);
        let (per, plane) = (s.grouped_bytes(), s.grouped_scales());
        Weight::Grouped {
            bytes: &self.bytes[i][e * per..(e + 1) * per],
            unit: s.quant().device_block_bytes(),
            nb: s.blocks_per_row(),
            scales: &self.scales[i][e * plane..(e + 1) * plane],
        }
    }
}

/// The same bytes in memory the driver pinned, as the mirror's are.
#[cfg(feature = "cuda")]
impl<'a> GroupedCopy<'a> {
    fn pinned(&self) -> Result<GroupedCopy<'a>> {
        // The context owns the pinned memory; it lives for the run.
        std::mem::forget(cust::quick_init()?);
        let mut bytes: [Vec<u8>; 3] = Default::default();
        let mut scales: [Vec<u16>; 3] = Default::default();
        for i in 0..3 {
            let mut host: *mut std::ffi::c_void = std::ptr::null_mut();
            let len = self.bytes[i].len() + self.scales[i].len() * 2;
            phobos_kernels::cuda_ok(unsafe { cust::sys::cuMemHostAlloc(&mut host, len, 0) }, "pinning")?;
            // SAFETY: `len` bytes just allocated, leaked for the run.
            let region = unsafe { std::slice::from_raw_parts_mut(host.cast::<u8>(), len) };
            let (b, sc) = region.split_at_mut(self.bytes[i].len());
            b.copy_from_slice(&self.bytes[i]);
            let sc: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(sc.as_mut_ptr().cast(), self.scales[i].len()) };
            sc.copy_from_slice(&self.scales[i]);
            bytes[i] = unsafe { Vec::from_raw_parts(b.as_mut_ptr(), b.len(), b.len()) };
            scales[i] = unsafe { Vec::from_raw_parts(sc.as_mut_ptr(), sc.len(), sc.len()) };
        }
        Ok(GroupedCopy { set: self.set, bytes, scales })
    }
}

fn jobs_shape(source: &impl Source, label: &str, rows_list: &[usize], d: usize, flops: impl Fn(usize) -> f64) -> Result<()> {
    let mut next = 0x2545_f491_4f6c_dd1du64;
    let mut random = || {
        next ^= next << 13;
        next ^= next >> 7;
        next ^= next << 17;
        (next >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let jobs_n = 48;
    println!("{jobs_n} experts as jobs over the pool, {label}:");
    println!("{:>6} {:>12} {:>14} {:>10}", "rows", "us/expert", "us/expert-row", "GFLOP/s");
    for &rows in rows_list {
        let x: Vec<f32> = (0..rows * d).map(|_| random()).collect();
        let jobs: Vec<Job> = (0..jobs_n).map(|e| Job { expert: e, rows: (0..rows).collect(), weights: vec![1.0; rows] }).collect();
        let mut out = vec![0.0f32; rows * d];
        simd::experts_ffn(source, &jobs, &x, rows, &mut out)?;
        let reps = 3;
        let started = Instant::now();
        for _ in 0..reps {
            simd::experts_ffn(source, &jobs, &x, rows, &mut out)?;
        }
        let micros = started.elapsed().as_secs_f64() * 1e6 / (reps * jobs_n) as f64;
        println!("{rows:>6} {micros:>12.1} {:>14.2} {:>10.1}", micros / rows as f64, flops(rows) / micros * 1e-3);
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

    let mut next = 0x9e37_79b9_7f4a_7c15u64;
    let mut random = || {
        next ^= next << 13;
        next ^= next >> 7;
        next ^= next << 17;
        (next >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let flops = |rows: usize| (rows * 6 * d * d_ff) as f64;

    for block in representative_blocks(&gguf, config.n_block) {
        let set = ExpertSet::load(&gguf, &format!("blk.{block}"), moe.n_expert, d, d_ff)?;
        println!(
            "block {block}: gate/up {}, down {}, {:.2} MiB an expert",
            set.gate.quant().name(),
            set.down.quant().name(),
            set.expert_bytes() as f64 / (1 << 20) as f64
        );
        println!("{:>6} {:>12} {:>14} {:>10}", "rows", "us/expert", "us/expert-row", "GFLOP/s");
        for &rows in &args.rows {
            let x: Vec<f32> = (0..rows * d).map(|_| random()).collect();
            let weights = vec![1.0f32; rows];
            let act = Q8Act::quantize(&x, rows, d);
            let mut scratch = Scratch::default();
            let mut out = vec![0.0f32; rows * d];
            // Once first: the mapping's pages come in, from the page cache
            // or the disk, and that pass is reported apart.
            let cold = Instant::now();
            for e in 0..args.experts {
                simd::expert_ffn(&*set, e % moe.n_expert, &act, &weights, &mut scratch, &mut out)?;
            }
            let cold_micros = cold.elapsed().as_secs_f64() * 1e6 / args.experts as f64;
            let started = Instant::now();
            for e in 0..args.experts {
                simd::expert_ffn(&*set, e % moe.n_expert, &act, &weights, &mut scratch, &mut out)?;
            }
            let micros = started.elapsed().as_secs_f64() * 1e6 / args.experts as f64;
            println!("{rows:>6} {micros:>12.1} {:>14.2} {:>10.1}   first touch {cold_micros:.0} us/expert", micros / rows as f64, flops(rows) / micros * 1e-3);
        }

        // The shape a split pass runs: many experts at once, each over its
        // own rows, the GEMMs serial inside a task; from the file's layout
        // and from the device's.
        jobs_shape(&*set, "file layout", &args.rows, d, flops)?;
        let grouped = GroupedCopy::build(&set);
        jobs_shape(&grouped, "grouped layout", &args.rows, d, flops)?;
        #[cfg(feature = "cuda")]
        {
            let pinned = grouped.pinned()?;
            jobs_shape(&pinned, "grouped layout, pinned", &args.rows, d, flops)?;
            // Its vectors are the driver's pinned memory, not the
            // allocator's, so they must not be freed as vectors.
            std::mem::forget(pinned);
        }

        // A decode step's misses: one row, one or two experts, spread over
        // the pool by rows, with a millisecond of idleness between calls as
        // a step's blocks leave, against calls back to back.
        for (label, idle) in [("back to back", 0u64), ("1 ms idle between", 1000u64)] {
            let x: Vec<f32> = (0..d).map(|_| random()).collect();
            let mut out = vec![0.0f32; d];
            let mut total = 0.0;
            let calls = 40;
            for i in 0..calls {
                let jobs = [Job { expert: (7 * i) % moe.n_expert, rows: vec![0], weights: vec![1.0] }, Job { expert: (7 * i + 3) % moe.n_expert, rows: vec![0], weights: vec![1.0] }];
                if idle > 0 {
                    let until = Instant::now() + std::time::Duration::from_micros(idle);
                    while Instant::now() < until {}
                }
                let started = Instant::now();
                simd::experts_row(&*set, &jobs, &x, &mut out)?;
                total += started.elapsed().as_secs_f64() * 1e6;
            }
            println!("two misses a call, {label}: {:.0} us a call", total / calls as f64);
        }

        // The reference decoder, one row, one expert: what the CPU miss
        // path cost before this.
        let x: Vec<f32> = (0..d).map(|_| random()).collect();
        let started = Instant::now();
        let mut dense = vec![0.0f32; d_ff * d];
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f32>();
        set.gate.dequantize(0, &mut dense)?;
        let g: Vec<f32> = dense.chunks_exact(d).map(|w| dot(w, &x)).collect();
        set.up.dequantize(0, &mut dense)?;
        let h: Vec<f32> = dense.chunks_exact(d).zip(&g).map(|(w, &g)| g / (1.0 + (-g).exp()) * dot(w, &x)).collect();
        set.down.dequantize(0, &mut dense)?;
        let y: f32 = dense.chunks_exact(d_ff).map(|w| dot(w, &h)).sum();
        println!("reference decoder, one row, one thread: {:.0} us (checksum {y:.3})", started.elapsed().as_secs_f64() * 1e6);
    }
    Ok(())
}
