// Throughput benchmark for the GGUF path, shaped like `llama-bench`:
//
//   cargo run --release -p phobos-bench --features cuda -- \
//       -m MODEL.gguf -p 128,512 -n 32,128,512 -r 3
//
// Reports llama-bench's numbers in its units, one row per size given.
// pp<N> is prompt processing: N tokens fed into a fresh state.
// tg<N> is text generation: N tokens produced one at a time.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use phobos_gguf::Decoder;
use phobos_gguf::Gguf;
use phobos_gguf::Quant;
use phobos_gguf::backend::Backend;
use phobos_inference::CacheStats;

#[cfg(feature = "cuda")]
use phobos_gguf::backend::device;

/// llama-bench's defaults for `-b` and `-ub`. The ubatch is also the
/// runtime's untuned prompt batch; on the device the runtime tunes its pass
/// length per depth, which this benchmark does not.
const DEFAULT_BATCH: usize = 2048;
const DEFAULT_UBATCH: usize = 512;

const DEFAULT_MODEL: &str = "models/Qwen3.5-0.8B-Q8_0.gguf";

struct Args {
    model: PathBuf,
    prompt_tokens: Vec<usize>,
    gen_tokens: Vec<usize>,
    /// Positions already in the context when a tg row starts, as
    /// llama-bench's `-d`. Each depth gets its own set of tg rows.
    depths: Vec<usize>,
    repetitions: usize,
    /// Tokens handed over at a time, as llama-bench's `-b`. The logits are
    /// read back at the end of each such batch.
    batch: usize,
    /// Tokens in one forward pass, as llama-bench's `-ub`. Never more than
    /// `batch`.
    ubatch: usize,
    warmup: bool,
    /// A cap on a streamed model's expert cache, as `phobos-cli`'s
    /// `--expert-cache`.
    expert_cache_bytes: Option<u64>,
    /// A cap on the host memory a streamed model's experts are held in, as
    /// `phobos-cli`'s `--max-ram`.
    host_memory_bytes: Option<u64>,
}

fn print_usage() {
    eprintln!(
        "\
usage: phobos-bench [OPTIONS]

OPTIONS:
  -m, --model FILE      the GGUF file to run (default: {DEFAULT_MODEL})
  -p, --n-prompt N,...  prompt-processing tokens, one pp<N> row each (default: 128)
  -n, --n-gen N,...     generated tokens, one tg<N> row each (default: 32)
  -d, --n-depth N,...   positions in the context before each tg row starts,
                        one set of tg rows per depth (default: 0)
  -r, --repetitions N   timed repetitions per row (default: 3)
  -b, --batch-size N    tokens a prompt is handed over in at a time, its logits
                        read back after each batch (default: {DEFAULT_BATCH})
  -ub, --ubatch-size N  tokens in one forward pass, at most the batch size
                        (default: {DEFAULT_UBATCH}, the runtime's untuned one)
      --no-warmup       skip the warmup pass, which leaves each kernel's first
                        compile inside the repetition it lands in
      --expert-cache SIZE
                        device memory for a streamed model's expert cache
                        (2g, 1500m, bytes); default, what the weights leave
      --max-ram SIZE    host memory for a streamed model's experts, refused
                        when the host has less available; default all of
                        them when they fit beside a quarter of the host's
                        memory, else what does
  -h, --help            print this message

  -p and -n take a comma-separated list, and repeat, so -p 128,512 and
  -p 128 -p 512 both ask for the same two rows. -b and -ub take one value;
  llama-bench's lists of them, a row per combination, are not supported.
  A pp<N> row and the positions a -d row primes are fed the same way."
    );
}

/// A batch or ubatch size: one positive token count.
fn parse_batch(value: &str) -> Result<usize> {
    let size: usize = value
        .trim()
        .parse()
        .with_context(|| format!("{value:?} is not one token count"))?;
    anyhow::ensure!(size > 0, "a batch holds at least one token");
    Ok(size)
}

/// How a prompt of `len` tokens is fed: batches of `batch` tokens, each run
/// as forward passes of at most `ubatch`. Each pass's length, and whether it
/// ends a batch, where the logits are read back.
fn passes(len: usize, batch: usize, ubatch: usize) -> Vec<(usize, bool)> {
    let mut out = Vec::new();
    let mut left = len;
    while left > 0 {
        let mut in_batch = left.min(batch);
        left -= in_batch;
        while in_batch > 0 {
            let pass = in_batch.min(ubatch);
            in_batch -= pass;
            out.push((pass, in_batch == 0));
        }
    }
    out
}

/// Runs `tokens` into `state` as [`passes`] lays them out. A pass that ends a
/// batch reads the logits back, as llama.cpp returns them per decode call;
/// the others take the greedy path, which reads back only a token id.
fn feed(model: &Decoder, state: &mut phobos_gguf::model::State, tokens: &[u32], args: &Args, backend: &dyn Backend) -> Result<()> {
    let mut at = 0;
    for (len, ends_batch) in passes(tokens.len(), args.batch, args.ubatch) {
        let chunk = &tokens[at..at + len];
        if ends_batch {
            model.forward(state, chunk, backend)?;
        } else {
            model.forward_greedy(state, chunk, backend)?;
        }
        at += len;
    }
    Ok(())
}

/// One size, or a comma-separated list of them. A zero drops the row, so
/// `-p 0` or `-n 0` runs only the other kind.
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
        depths: Vec::new(),
        repetitions: 3,
        batch: DEFAULT_BATCH,
        ubatch: DEFAULT_UBATCH,
        warmup: true,
        expert_cache_bytes: None,
        host_memory_bytes: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut next = |flag: &str| it.next().with_context(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "-m" | "--model" => args.model = next("--model")?.into(),
            "-p" | "--n-prompt" => args.prompt_tokens.extend(parse_sizes(&next("-p")?)?),
            "-n" | "--n-gen" => args.gen_tokens.extend(parse_sizes(&next("-n")?)?),
            "-d" | "--n-depth" | "--depth" => args.depths.extend(parse_sizes(&next("-d")?)?),
            "-r" | "--repetitions" => args.repetitions = next("-r")?.parse().context("-r")?,
            "-b" | "--batch-size" => args.batch = parse_batch(&next("-b")?).context("-b")?,
            "-ub" | "--ubatch-size" => args.ubatch = parse_batch(&next("-ub")?).context("-ub")?,
            "--no-warmup" => args.warmup = false,
            "--expert-cache" => {
                args.expert_cache_bytes = Some(phobos_base::cli::parse_size(&next("--expert-cache")?)?)
            }
            "--max-ram" => {
                let bytes = phobos_base::cli::parse_size(&next("--max-ram")?)?;
                phobos_base::host::ensure_available(bytes).context("--max-ram")?;
                args.host_memory_bytes = Some(bytes);
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "-V" | "--version" => {
                let fingerprint = phobos_kernels::COMPILER_FINGERPRINT;
                println!("phobos-bench {} (compiler {fingerprint})", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other:?} (try --help)"),
        }
    }
    // Default only when the flag never appeared, so -p 0 means no prompt row.
    if !std::env::args().any(|a| a == "-p" || a == "--n-prompt") {
        args.prompt_tokens.push(128);
    }
    if !std::env::args().any(|a| a == "-n" || a == "--n-gen") {
        args.gen_tokens.push(32);
    }
    if args.depths.is_empty() {
        args.depths.push(0);
    }
    // llama.cpp clamps the ubatch to the batch the same way.
    args.ubatch = args.ubatch.min(args.batch);
    args.prompt_tokens.retain(|&n| n > 0);
    args.gen_tokens.retain(|&n| n > 0);
    Ok(args)
}

/// The backend for a model whose weights are stored in `quants`.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn make_backend(quants: &[Quant]) -> Result<Box<dyn Backend>> {
    #[cfg(feature = "cuda")]
    {
        Ok(Box::new(device::DeviceBackend::new(quants)?))
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
    // The model label llama-bench prints. The quantization is the type that
    // holds the most weights: a mixed file's largest single tensor is often
    // its embedding, kept wider than the rest.
    let mut by_type: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
    for t in gguf.tensors() {
        *by_type.entry(t.ggml_type.name()).or_default() += t.numel();
    }
    let quantization = by_type.into_iter().max_by_key(|&(_, n)| n).map_or("?", |(name, _)| name);
    let architecture = format!(
        "{} {:.1}B {quantization}",
        gguf.architecture()?,
        gguf.parameter_count() as f64 / 1e9,
    );
    let model = Decoder::load(&gguf)?;
    let backend = make_backend(&gguf.quants())?;
    if let Some(bytes) = args.expert_cache_bytes {
        backend.limit_expert_cache(bytes as usize)?;
    }
    if let Some(bytes) = args.host_memory_bytes {
        backend.limit_host_memory(bytes as usize)?;
    }
    let load_millis = load_start.elapsed().as_secs_f64() * 1e3;

    let vocab = model.vocab();
    eprintln!("batch {}, ubatch {}", args.batch, args.ubatch);
    eprintln!(
        "model {} ({architecture}, {vocab} vocab), backend {}, load {load_millis:.0} ms",
        args.model.display(),
        backend_name()
    );

    // Pseudorandom token ids, as llama-bench uses. A position's cost does not
    // depend on the token, but a mixture-of-experts model's does: routing is
    // deterministic, so repeating a sequence over a warm expert cache measures
    // a cache that holds exactly what the pass needs. Each repetition gets a
    // sequence of its own, and starts with the cache empty.
    let longest = args
        .prompt_tokens
        .iter()
        .copied()
        .chain(args.gen_tokens.iter().map(|&n| n + args.depths.iter().max().unwrap_or(&0)))
        .max()
        .unwrap_or(0);
    let count = longest.max(128) + 1;
    let tokens = synthetic_tokens(count, vocab, 0);
    let rep_tokens: Vec<Vec<u32>> = (1..=args.repetitions as u64).map(|rep| synthetic_tokens(count, vocab, rep)).collect();

    if args.warmup {
        eprint!("warmup... ");
        // Batch and single-step passes hit different kernels, and each may
        // compile on first use, so both are warmed. A batch needs at least 128
        // rows to reach the widest batched tile. Each distinct pass length the
        // timed runs feed gets its own pass, because the first pass at a shape
        // builds its graph.
        let fed = args.prompt_tokens.iter().copied().chain(args.depths.iter().map(|&d| d + 1));
        let mut shapes: Vec<usize> =
            fed.flat_map(|n| passes(n, args.batch, args.ubatch)).map(|(len, _)| len).collect();
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
            // The same greedy path the timed tg loop takes.
            model.forward_greedy(&mut state, &[token], backend.as_ref())?;
        }
        state.release(backend.as_ref());
        eprintln!("done");
    }

    let mut rows = Vec::new();
    for &prompt_tokens in &args.prompt_tokens {
        let mut rates = Vec::new();
        for (rep, tokens) in rep_tokens.iter().enumerate() {
            backend.cool_expert_cache()?;
            let mut state = model.new_state();
            let before = backend.cache_stats();
            let start = Instant::now();
            feed(&model, &mut state, &tokens[..prompt_tokens], &args, backend.as_ref())?;
            let secs = start.elapsed().as_secs_f64();
            state.release(backend.as_ref());
            rates.push(prompt_tokens as f64 / secs);
            let hits = hit_rate(before, backend.cache_stats(), |s| (s.expert_prompt_hits, s.expert_prompt_misses));
            eprintln!(
                "  pp{prompt_tokens} rep {}/{}: {secs:.3} s{hits}",
                rep + 1,
                args.repetitions,
            );
        }
        rows.push((format!("pp{prompt_tokens}"), rates));
    }
    for (&depth, &gen_tokens) in args.depths.iter().flat_map(|d| args.gen_tokens.iter().map(move |n| (d, n))) {
        let test = if depth == 0 { format!("tg{gen_tokens}") } else { format!("tg{gen_tokens} @ d{depth}") };
        let mut rates = Vec::new();
        for (rep, tokens) in rep_tokens.iter().enumerate() {
            backend.cool_expert_cache()?;
            let mut state = model.new_state();
            // Prime `depth + 1` positions as a prompt is fed, so the timed
            // loop is pure decoding, as in llama-bench.
            feed(&model, &mut state, &tokens[..depth + 1], &args, backend.as_ref())?;
            let before = backend.cache_stats();
            let faults = phobos_base::host::page_faults();
            let start = Instant::now();
            // The greedy fast path, which temperature-0 decoding takes. It is
            // fed fixed tokens regardless of what comes back.
            for &token in tokens.iter().skip(depth + 1).take(gen_tokens) {
                model.forward_greedy(&mut state, &[token], backend.as_ref())?;
            }
            let secs = start.elapsed().as_secs_f64();
            state.release(backend.as_ref());
            rates.push(gen_tokens as f64 / secs);
            // A streamed model's decode rate follows its expert hit rate.
            let after = backend.cache_stats();
            let hits = hit_rate(before, after, |s| (s.expert_hits, s.expert_misses));
            let host = host_misses(before, after, gen_tokens);
            let faults = match (faults, phobos_base::host::page_faults()) {
                (Some(a), Some(b)) => format!(", {:.0} page faults a token", (b - a) as f64 / gen_tokens as f64),
                _ => String::new(),
            };
            eprintln!("  {test} rep {}/{}: {secs:.3} s{hits}{host}{faults}", rep + 1, args.repetitions);
        }
        rows.push((test, rates));
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
    // For a model that streams its experts, the device cache's hit rate.
    if let Some(stats) = backend.cache_stats()
        && let Some(rate) = stats.expert_hit_rate()
    {
        println!(
            "\nexpert cache: {:.1}% hits ({} of {}), {:.2} GB copied over the bus",
            rate * 100.0,
            stats.expert_hits,
            stats.expert_hits + stats.expert_misses,
            stats.expert_bytes as f64 / 1e9
        );
        if stats.expert_cpu_misses > 0 {
            println!(
                "host misses: {}, {:.0} us each, {} read from the file",
                stats.expert_cpu_misses,
                stats.expert_cpu_nanos as f64 / 1e3 / stats.expert_cpu_misses as f64,
                stats.expert_cpu_mapped
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

/// The expert hit rate between two readings, as a suffix for a progress
/// line, or nothing when the model streams no experts or none were looked up.
/// `lookups` picks the counters, the decode or the prompt ones.
fn hit_rate(
    before: Option<CacheStats>,
    after: Option<CacheStats>,
    lookups: impl Fn(&CacheStats) -> (u64, u64),
) -> String {
    let (Some(before), Some(after)) = (before, after) else {
        return String::new();
    };
    let ((bh, bm), (ah, am)) = (lookups(&before), lookups(&after));
    let (hits, misses) = (ah - bh, am - bm);
    if hits + misses == 0 {
        return String::new();
    }
    format!(", {:.1}% expert hits", 100.0 * hits as f64 / (hits + misses) as f64)
}

/// The host misses between two readings of `tokens` decode steps, as a
/// suffix for a progress line: how many a token, their time each and how
/// many were read from the file. Nothing when there were none.
fn host_misses(before: Option<CacheStats>, after: Option<CacheStats>, tokens: usize) -> String {
    let (Some(before), Some(after)) = (before, after) else {
        return String::new();
    };
    let misses = after.expert_cpu_misses - before.expert_cpu_misses;
    if misses == 0 {
        return String::new();
    }
    format!(
        ", {:.1} host misses a token at {:.0} us, {:.0}% from the file",
        misses as f64 / tokens as f64,
        (after.expert_cpu_nanos - before.expert_cpu_nanos) as f64 / 1e3 / misses as f64,
        100.0 * (after.expert_cpu_mapped - before.expert_cpu_mapped) as f64 / misses as f64
    )
}

/// A deterministic spread of valid token ids from an xorshift generator.
/// Each `stream` is a different sequence.
fn synthetic_tokens(count: usize, vocab: usize, stream: u64) -> Vec<u32> {
    let mut seed = 0x2545_f491_4f6c_dd1du64 ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15);
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

#[cfg(test)]
mod tests {
    use super::passes;

    #[test]
    fn a_prompt_is_cut_into_batches_then_passes() {
        // llama-bench's defaults over a prompt of 2048: one batch, four passes.
        assert_eq!(passes(2048, 2048, 512), [(512, false), (512, false), (512, false), (512, true)]);
        // A short prompt is one pass.
        assert_eq!(passes(128, 2048, 512), [(128, true)]);
        // A batch boundary reads the logits back even mid-ubatch.
        assert_eq!(passes(700, 600, 512), [(512, false), (88, true), (100, true)]);
        assert_eq!(passes(0, 2048, 512), []);
    }
}
