// Throughput benchmark for the GGUF path, shaped like `llama-bench`:
//
//   cargo run --release -p phobos-gguf --features cuda --example bench -- \
//       -m MODEL.gguf -p 128 -n 32 -r 3
//
// Reports the same two numbers llama-bench does, in the same units:
//
//   pp<N>  prompt processing, N tokens fed into a fresh state
//   tg<N>  text generation, N tokens produced one at a time
//
// pp feeds the whole prompt in one pass, so its projections are real matmuls
// rather than a matvec per position. The delta-rule recurrence and the softmax
// attention are still sequential over positions on the host, so the pp-to-tg
// ratio stays well under llama.cpp's.

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
    prompt_tokens: usize,
    gen_tokens: usize,
    repetitions: usize,
    warmup: bool,
}

fn print_usage() {
    eprintln!(
        "\
usage: bench [OPTIONS]

OPTIONS:
  -m, --model FILE      the GGUF file to run (default: {DEFAULT_MODEL})
  -p, --n-prompt N      prompt-processing tokens, the pp<N> row (default: 128)
  -n, --n-gen N         generated tokens, the tg<N> row (default: 32)
  -r, --repetitions N   timed repetitions per row (default: 3)
      --no-warmup       skip the warmup pass, which leaves each kernel's first
                        compile inside the repetition it lands in
  -h, --help            print this message"
    );
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        model: PathBuf::from(DEFAULT_MODEL),
        prompt_tokens: 128,
        gen_tokens: 32,
        repetitions: 3,
        warmup: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut next = |flag: &str| it.next().with_context(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "-m" | "--model" => args.model = next("--model")?.into(),
            "-p" | "--n-prompt" => args.prompt_tokens = next("-p")?.parse().context("-p")?,
            "-n" | "--n-gen" => args.gen_tokens = next("-n")?.parse().context("-n")?,
            "-r" | "--repetitions" => args.repetitions = next("-r")?.parse().context("-r")?,
            "--no-warmup" => args.warmup = false,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other:?} (try --help)"),
        }
    }
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
    // The label llama-bench prints, read off the file rather than assumed: two
    // architectures of different sizes and widths now run through here. The
    // quantization is whichever type carries the most elements, since the norms
    // are f32 in every file.
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
    let load_millis = load_start.elapsed().as_secs_f64() * 1e3;

    let vocab = model.vocab();
    eprintln!(
        "model {} ({architecture}, {vocab} vocab), backend {}, load {load_millis:.0} ms",
        args.model.display(),
        backend_name()
    );

    // llama-bench feeds pseudorandom token ids rather than real text: the cost
    // of a position does not depend on which token sits there.
    let tokens = synthetic_tokens(args.prompt_tokens.max(args.gen_tokens) + 1, vocab);

    if args.warmup {
        eprint!("warmup... ");
        let mut state = model.new_state();
        // A batch and then single steps, because the two take different kernels
        // and a backend may compile each on first use. Warming only one leaves
        // the other's compile inside the first timed repetition, where it is
        // worth more than the repetition it lands in. The batch has to be deep
        // enough to reach the widest tile a batched kernel has: 128 rows is the
        // quantized projection's, and it also carries the attention block past
        // the 64 rows its matmul path needs. Past that it is the prompt itself,
        // which is what llama-bench warms with: it runs each test once before
        // timing it, and the first pass at a new shape also builds the graph.
        let batch = tokens.len().min(args.prompt_tokens.max(128));
        model.forward(&mut state, &tokens[..batch], backend.as_ref())?;
        for &token in tokens.iter().take(4) {
            model.forward(&mut state, &[token], backend.as_ref())?;
        }
        state.release(backend.as_ref());
        eprintln!("done");
    }

    let mut rows = Vec::new();
    if args.prompt_tokens > 0 {
        let mut rates = Vec::new();
        for rep in 0..args.repetitions {
            let mut state = model.new_state();
            let start = Instant::now();
            // The whole prompt in one pass, which is what makes pp a different
            // measurement from tg: the projections become real matmuls instead
            // of a matvec per position.
            model.forward(&mut state, &tokens[..args.prompt_tokens], backend.as_ref())?;
            let secs = start.elapsed().as_secs_f64();
            state.release(backend.as_ref());
            rates.push(args.prompt_tokens as f64 / secs);
            eprintln!(
                "  pp{} rep {}/{}: {:.3} s",
                args.prompt_tokens,
                rep + 1,
                args.repetitions,
                secs
            );
        }
        rows.push((format!("pp{}", args.prompt_tokens), rates));
    }
    if args.gen_tokens > 0 {
        let mut rates = Vec::new();
        for rep in 0..args.repetitions {
            let mut state = model.new_state();
            // Prime with one position so the timed loop is pure decoding, the
            // same split llama-bench uses.
            model.forward(&mut state, &tokens[..1], backend.as_ref())?;
            let start = Instant::now();
            for &token in tokens.iter().skip(1).take(args.gen_tokens) {
                model.forward(&mut state, &[token], backend.as_ref())?;
            }
            let secs = start.elapsed().as_secs_f64();
            state.release(backend.as_ref());
            rates.push(args.gen_tokens as f64 / secs);
            eprintln!(
                "  tg{} rep {}/{}: {:.3} s",
                args.gen_tokens,
                rep + 1,
                args.repetitions,
                secs
            );
        }
        rows.push((format!("tg{}", args.gen_tokens), rates));
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
