// Localize a broken qwen35 forward pass:
//
//   cargo run --release -p phobos-gguf --example diagnose -- MODEL.gguf
//
// Reports per-position negative log-likelihood on a repeated phrase. Any
// working context mechanism makes the second repetition far cheaper than the
// first, so a flat profile means neither mixer is propagating information.

use std::path::PathBuf;

use anyhow::{Result, bail};
use phobos_gguf::compute::HostBackend;
use phobos_gguf::qwen35::{Model, Variants};
use phobos_gguf::{Bpe, Gguf};

const REPEATED: &str = " apple banana cherry apple banana cherry apple banana cherry";

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1).map(PathBuf::from) else {
        bail!("usage: diagnose MODEL.gguf");
    };

    let gguf = Gguf::open(&path)?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    let model = Model::load(&gguf)?;
    let backend = HostBackend::new();

    let tokens = bpe.encode(REPEATED)?;
    println!("{REPEATED:?} -> {} tokens\n", tokens.len());

    let mut state = model.new_state();
    let mut logits = model.forward_with(&mut state, &tokens[..1], &backend, Variants::REFERENCE)?;
    println!("{:>4}  {:>10}  {:>9}  text", "pos", "token", "nll");
    for (i, &actual) in tokens[1..].iter().enumerate() {
        let nll = -log_softmax_at(&logits, actual as usize);
        println!(
            "{:>4}  {:>10}  {nll:>9.4}  {:?}",
            i + 1,
            actual,
            bpe.decode(&[actual])
        );
        logits = model.forward_with(&mut state, &[actual], &backend, Variants::REFERENCE)?;
    }
    Ok(())
}

fn log_softmax_at(logits: &[f32], index: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&l| (l as f64 - max).exp()).sum();
    (logits[index] as f64 - max) - sum.ln()
}
