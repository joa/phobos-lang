// Generate text from a GGUF model on the host backend:
//
//   cargo run --release -p phobos-gguf --example generate -- \
//       MODEL.gguf -n 40 "The capital of France is"
//
// Greedy, so runs are reproducible. The GPU backend is in phobos-inference.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use phobos_gguf::compute::HostBackend;
use phobos_gguf::{Bpe, Decoder, Gguf};

fn main() -> Result<()> {
    let mut path: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut num_tokens = 32usize;
    let mut show_logits = false;
    let mut probs = 0usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-n" => num_tokens = args.next().unwrap_or_default().parse()?,
            "--logits" => show_logits = true,
            "--probs" => probs = args.next().unwrap_or_default().parse()?,
            other if path.is_none() => path = Some(other.into()),
            other => prompt = Some(other.to_string()),
        }
    }
    let (Some(path), Some(prompt)) = (path, prompt) else {
        bail!("usage: generate [-n TOKENS] [--logits] [--probs K] MODEL.gguf PROMPT");
    };

    let started = Instant::now();
    let gguf = Gguf::open(&path)?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    eprint!("loading {} weights... ", gguf.tensors().len());
    std::io::stderr().flush().ok();
    let model = Decoder::load(&gguf)?;
    eprintln!("ready in {:.1}s", started.elapsed().as_secs_f32());
    eprintln!("{}\n", model.summary());

    let ids = bpe.encode(&prompt)?;
    if ids.is_empty() {
        bail!("prompt encoded to zero tokens");
    }
    eprintln!(
        "prompt: {prompt:?} -> {} tokens {:?}",
        ids.len(),
        &ids[..ids.len().min(16)]
    );

    let backend = HostBackend::new();
    let mut state = model.new_state();

    let prefill = Instant::now();
    let mut logits = model.forward(&mut state, &ids, &backend)?;
    eprintln!(
        "prefill: {} tokens in {:.2}s\n",
        ids.len(),
        prefill.elapsed().as_secs_f32()
    );

    if show_logits {
        report_top(&bpe, &logits, 10);
    }
    if probs > 0 {
        // No sampling has happened yet, so any difference against another
        // implementation's distribution is the forward pass.
        println!("{}", top_probs(&logits, probs));
        return Ok(());
    }

    print!("{prompt}");
    std::io::stdout().flush().ok();

    let eos = bpe.eos();
    let decode = Instant::now();
    let mut produced = 0usize;
    let mut pending: Vec<u8> = Vec::new();
    for _ in 0..num_tokens {
        let next = argmax(&logits);
        if Some(next) == eos {
            break;
        }
        pending.extend(bpe.decode_bytes(&[next]));
        let valid = match std::str::from_utf8(&pending) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid > 0 {
            print!("{}", std::str::from_utf8(&pending[..valid]).unwrap());
            std::io::stdout().flush().ok();
            pending.drain(..valid);
        }
        produced += 1;
        logits = model.forward(&mut state, &[next], &backend)?;
    }
    println!();

    let seconds = decode.elapsed().as_secs_f32();
    eprintln!(
        "\ndecode: {produced} tokens in {seconds:.2}s ({:.2} tok/s)",
        produced as f32 / seconds
    );
    Ok(())
}

/// The `k` most likely next tokens as a JSON array of `[id, probability]`.
fn top_probs(logits: &[f32], k: usize) -> String {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let pairs: Vec<String> = order
        .iter()
        .take(k)
        .map(|&i| format!("[{i}, {}]", exps[i] / sum))
        .collect();
    format!("[{}]", pairs.join(", "))
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// The top-k next-token candidates: the quickest read on whether the forward
/// pass produces a sane distribution.
fn report_top(bpe: &Bpe, logits: &[f32], k: usize) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

    eprintln!("top {k} candidates:");
    for &i in order.iter().take(k) {
        eprintln!(
            "  {:>7.3}%  {:>10.4}  {:?}",
            100.0 * exps[i] / sum,
            logits[i],
            bpe.decode(&[i as u32])
        );
    }
    let entropy: f32 = exps
        .iter()
        .map(|&e| e / sum)
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    eprintln!("  entropy {entropy:.3} nats over {} logits\n", logits.len());
}
