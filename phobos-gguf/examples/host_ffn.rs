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
use phobos_gguf::simd::{self, Q8Act, Scratch};
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
            // Once untimed: the mapping's pages come in from disk.
            for e in 0..args.experts {
                simd::expert_ffn(&set, e % moe.n_expert, &act, &weights, &mut scratch, &mut out)?;
            }
            let started = Instant::now();
            for e in 0..args.experts {
                simd::expert_ffn(&set, e % moe.n_expert, &act, &weights, &mut scratch, &mut out)?;
            }
            let micros = started.elapsed().as_secs_f64() * 1e6 / args.experts as f64;
            println!("{rows:>6} {micros:>12.1} {:>14.2} {:>10.1}", micros / rows as f64, flops(rows) / micros * 1e-3);
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
