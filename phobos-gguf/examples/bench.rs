// Throughput benchmark for the GGUF path, shaped like `llama-bench`:
//
//   cargo run --release -p phobos-gguf --features cuda --example bench -- \
//       -m MODEL.gguf -p 128,512 -n 32,128,512 -r 3
//
// Reports the same numbers llama-bench does, in the same units, one row per
// size given:
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
    prompt_tokens: Vec<usize>,
    gen_tokens: Vec<usize>,
    repetitions: usize,
    warmup: bool,
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
        // A batch and then single steps, because the two take different kernels
        // and a backend may compile each on first use. Warming only one leaves
        // the other's compile inside the first timed repetition, where it is
        // worth more than the repetition it lands in. The batch has to be deep
        // enough to reach the widest tile a batched kernel has: 128 rows is the
        // quantized projection's, and it also carries the attention block past
        // the 64 rows its matmul path needs. Past that it is the prompt itself,
        // which is what llama-bench warms with: it runs each test once before
        // timing it, and the first pass at a new shape also builds the graph.
        // Hence one pass per prompt size rather than one at the largest: a
        // shape that was never run is a graph that gets built inside the
        // repetition that first asks for it.
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
            model.forward(&mut state, &[token], backend.as_ref())?;
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
            // The whole prompt in one pass, which is what makes pp a different
            // measurement from tg: the projections become real matmuls instead
            // of a matvec per position.
            model.forward(&mut state, &tokens[..prompt_tokens], backend.as_ref())?;
            let secs = start.elapsed().as_secs_f64();
            state.release(backend.as_ref());
            rates.push(prompt_tokens as f64 / secs);
            eprintln!(
                "  pp{prompt_tokens} rep {}/{}: {secs:.3} s",
                rep + 1,
                args.repetitions,
            );
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
            for &token in tokens.iter().skip(1).take(gen_tokens) {
                model.forward(&mut state, &[token], backend.as_ref())?;
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
