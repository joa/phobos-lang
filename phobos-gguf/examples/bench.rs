// Throughput benchmark for the GGUF path, shaped like `llama-bench`:
//
//   cargo run --release -p phobos-gguf --features cuda --example bench -- \
//       -m MODEL.gguf -p 128,512 -n 32,128,512 -r 3
//
// Reports the same numbers llama-bench does, in the same units: pp<N> is
// prompt processing (N tokens fed into a fresh state) and tg<N> is text
// generation (N tokens produced one at a time), one row per size given.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use phobos_gguf::Decoder;
use phobos_gguf::Gguf;
use phobos_gguf::backend::Backend;

use phobos_gguf::backend::device;

const DEFAULT_MODEL: &str = "models/Qwen3.5-0.8B-Q8_0.gguf";

struct Args {
    model: PathBuf,
    prompt_tokens: Vec<usize>,
    gen_tokens: Vec<usize>,
    repetitions: usize,
    warmup: bool,
    /// A cap on a streamed model's expert cache, as `phobos-cli`'s
    /// `--expert-cache`.
    expert_cache_bytes: Option<u64>,
}

fn print_usage() {
    eprintln!(
        "\
usage: bench [OPTIONS]

OPTIONS:
  -m, --model FILE      the GGUF file to run (default: {DEFAULT_MODEL})
  -p, --n-prompt N,...  prompt-processing tokens, one pp<N> row each (default: 128)
  -n, --n-gen N,...     generated tokens, one tg<N> row each (default: 32)
  -r, --repetitions N   timed repetitions per row (default: 3)
      --no-warmup       skip the warmup pass, which leaves each kernel's first
                        compile inside the repetition it lands in
      --expert-cache SIZE
                        device memory for a streamed model's expert cache
                        (2g, 1500m, bytes); default, what the weights leave
  -h, --help            print this message

  -p and -n take a comma-separated list, and repeat, so -p 128,512 and
  -p 128 -p 512 both ask for the same two rows."
    );
}

/// One size, or a comma-separated list of them. A zero drops the row, which is
/// how a caller asks for prompt passes alone or decode steps alone.
fn parse_sizes(value: &str) -> Result<Vec<usize>> {
    value
        .split(',')
        .filter(|piece| !piece.trim().is_empty())
        .map(|piece| {
            piece
                .trim()
                .parse::<usize>()
                .with_context(|| format!("{piece:?} is not a token count"))
        })
        .collect()
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        model: PathBuf::from(DEFAULT_MODEL),
        prompt_tokens: Vec::new(),
        gen_tokens: Vec::new(),
        repetitions: 3,
        warmup: true,
        expert_cache_bytes: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut next = |flag: &str| it.next().with_context(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "-m" | "--model" => args.model = next("--model")?.into(),
            "-p" | "--n-prompt" => args.prompt_tokens.extend(parse_sizes(&next("-p")?)?),
            "-n" | "--n-gen" => args.gen_tokens.extend(parse_sizes(&next("-n")?)?),
            "-r" | "--repetitions" => args.repetitions = next("-r")?.parse().context("-r")?,
            "--no-warmup" => args.warmup = false,
            "--expert-cache" => {
                args.expert_cache_bytes = Some(phobos_base::cli::parse_size(&next("--expert-cache")?)?)
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other:?} (try --help)"),
        }
    }
    // Only when the flag never appeared: -p 0 is a request for no prompt row,
    // not a request for the default one.
    if !std::env::args().any(|a| a == "-p" || a == "--n-prompt") {
        args.prompt_tokens.push(128);
    }
    if !std::env::args().any(|a| a == "-n" || a == "--n-gen") {
        args.gen_tokens.push(32);
    }
    args.prompt_tokens.retain(|&n| n > 0);
    args.gen_tokens.retain(|&n| n > 0);
    Ok(args)
}

fn make_backend() -> Result<Box<dyn Backend>> {
    #[cfg(feature = "cuda")]
    {
        Ok(Box::new(device::DeviceBackend::new()?))
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(Box::new(phobos_gguf::backend::HostBackend::new()))
    }
}

fn backend_name() -> &'static str {
    if cfg!(feature = "cuda") {
        "Phobos GPU"
    } else {
        "Phobos host"
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let load_start = Instant::now();
    let gguf = Gguf::open(&args.model)?;
    // The label llama-bench prints, read off the file since architectures
    // differ in size and width. The quantization is whichever type carries
    // the most elements: the norms are always f32.
    let quantization = gguf
        .tensors()
        .iter()
        .max_by_key(|t| t.numel())
        .map_or("?", |t| t.ggml_type.name());
    let architecture = format!(
        "{} {:.1}B {quantization}",
        gguf.architecture()?,
        gguf.parameter_count() as f64 / 1e9,
    );
    let model = Decoder::load(&gguf)?;
    let backend = make_backend()?;
    if let Some(bytes) = args.expert_cache_bytes {
        backend.limit_expert_cache(bytes as usize)?;
    }
    let load_millis = load_start.elapsed().as_secs_f64() * 1e3;

    let vocab = model.vocab();
    eprintln!(
        "model {} ({architecture}, {vocab} vocab), backend {}, load {load_millis:.0} ms",
        args.model.display(),
        backend_name()
    );

    // llama-bench feeds pseudorandom token ids rather than real text: the cost
    // of a position does not depend on which token sits there.
    let longest = args
        .prompt_tokens
        .iter()
        .chain(args.gen_tokens.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let tokens = synthetic_tokens(longest.max(128) + 1, vocab);

    if args.warmup {
        eprint!("warmup... ");
        // Batch and single-step warmups both run: they hit different kernels,
        // and a backend may compile each on first use. The batch needs at
        // least 128 rows to reach the widest tile a batched kernel has (the
        // quantized projection's, and past the attention matmul path's own
        // 64-row floor). One warmup pass runs per distinct prompt size, not
        // just the largest, since the first pass at a new shape also builds
        // its graph.
        let mut shapes: Vec<usize> = args.prompt_tokens.iter().map(|&n| n.max(128)).collect();
        shapes.push(128);
        shapes.sort_unstable();
        shapes.dedup();
        for batch in shapes {
            let mut state = model.new_state();
            model.forward(&mut state, &tokens[..tokens.len().min(batch)], backend.as_ref())?;
            state.release(backend.as_ref());
        }
        let mut state = model.new_state();
        for &token in tokens.iter().take(4) {
            // Warms the same greedy fast path the timed tg loop takes, so its
            // kernels compile here rather than inside a timed repetition.
            model.forward_greedy(&mut state, &[token], backend.as_ref())?;
        }
        state.release(backend.as_ref());
        eprintln!("done");
    }

    let mut rows = Vec::new();
    for &prompt_tokens in &args.prompt_tokens {
        let mut rates = Vec::new();
        for rep in 0..args.repetitions {
            let mut state = model.new_state();
            let start = Instant::now();
            // The whole prompt in one pass: pp's projections are real matmuls,
            // not a matvec per position like tg's.
            model.forward(&mut state, &tokens[..prompt_tokens], backend.as_ref())?;
            let secs = start.elapsed().as_secs_f64();
            state.release(backend.as_ref());
            rates.push(prompt_tokens as f64 / secs);
            eprintln!(
                "  pp{prompt_tokens} rep {}/{}: {secs:.3} s",
                rep + 1,
                args.repetitions,
            );
            if let (Some(stats), Some((free, _))) = (backend.cache_stats(), backend.device_memory()) {
                eprintln!("    pool: {} reused, {} allocated; device free {} MiB", stats.buffers_reused, stats.buffers_allocated, free >> 20);
            }
        }
        rows.push((format!("pp{prompt_tokens}"), rates));
    }
    for &gen_tokens in &args.gen_tokens {
        let mut rates = Vec::new();
        for rep in 0..args.repetitions {
            let mut state = model.new_state();
            // Prime with one position so the timed loop is pure decoding, the
            // same split llama-bench uses.
            model.forward(&mut state, &tokens[..1], backend.as_ref())?;
            let start = Instant::now();
            // The greedy fast path: a temperature-0 deployment takes this
            // route, re-feeding a fixed token regardless of what comes back,
            // so timing it instead of `forward` measures what actually ships.
            for &token in tokens.iter().skip(1).take(gen_tokens) {
                model.forward_greedy(&mut state, &[token], backend.as_ref())?;
            }
            let secs = start.elapsed().as_secs_f64();
            state.release(backend.as_ref());
            rates.push(gen_tokens as f64 / secs);
            eprintln!(
                "  tg{gen_tokens} rep {}/{}: {secs:.3} s",
                rep + 1,
                args.repetitions,
            );
        }
        rows.push((format!("tg{gen_tokens}"), rates));
    }

    println!("\n| model | backend | test | t/s |");
    println!("| ----- | ------- | ---- | --- |");
    for (test, rates) in &rows {
        let (mean, stddev) = mean_stddev(rates);
        println!(
            "| {architecture} | {} | {test} | {mean:.2} +/- {stddev:.2} |",
            backend_name()
        );
    }
    // A model whose experts stream has one more number that decides its
    // decode rate: what share of them the device cache had.
    if let Some(stats) = backend.cache_stats()
        && let Some(rate) = stats.expert_hit_rate()
    {
        println!(
            "
expert cache: {:.1}% hits ({} of {}), {:.2} GB copied over the bus",
            rate * 100.0,
            stats.expert_hits,
            stats.expert_hits + stats.expert_misses,
            stats.expert_bytes as f64 / 1e9
        );
        if stats.expert_cpu_misses > 0 {
            println!(
                "cpu misses: {} computed on the host, {:.0} us each",
                stats.expert_cpu_misses,
                stats.expert_cpu_nanos as f64 / 1e3 / stats.expert_cpu_misses as f64
            );
        }
        if stats.expert_prefetches > 0 {
            println!(
                "lookahead: {} prefetched, {} of them wanted ({:.1}%)",
                stats.expert_prefetches,
                stats.expert_prefetch_hits,
                stats.expert_prefetch_hits as f64 / stats.expert_prefetches as f64 * 100.0
            );
        }
    }
    Ok(())
}

/// A deterministic spread of valid token ids (xorshift64*, same generator the
/// other checks use).
fn synthetic_tokens(count: usize, vocab: usize) -> Vec<u32> {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    (0..count)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % vocab as u64) as u32
        })
        .collect()
}

fn mean_stddev(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    (mean, variance.sqrt())
}
