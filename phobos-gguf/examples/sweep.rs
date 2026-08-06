// Resolve the ambiguous points of the qwen35 forward pass empirically:
//
//   cargo run --release -p phobos-gguf --example sweep -- MODEL.gguf
//
// The file fixes every tensor extent but not, say, whether the fused q/k/v
// projection groups per head or in three blocks. Each combination in
// Variants::all is scored by teacher-forced negative log-likelihood: only the
// one the weights were trained under predicts real text well.

use std::path::PathBuf;

use anyhow::{Result, bail};
use phobos_gguf::compute::HostBackend;
use phobos_gguf::qwen35::{Model, Variants};
use phobos_gguf::{Bpe, Gguf};

/// Scoring only the repetitions measures one thing: whether context reaches the
/// current position. Any working mixer drives these near zero and a broken one
/// sits near the vocabulary entropy of about 12 nats, so fluency effects cannot
/// confuse the ranking.
const PROBE: &str = " apple banana cherry apple banana cherry apple banana cherry";

/// Predictions before this index cover the first, genuinely unpredictable pass.
const SCORE_FROM: usize = 3;

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1).map(PathBuf::from) else {
        bail!("usage: sweep MODEL.gguf");
    };

    let gguf = Gguf::open(&path)?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    let model = Model::load(&gguf)?;
    let backend = HostBackend::new();

    let tokens = bpe.encode(PROBE)?;
    let scored = tokens.len() - 1 - SCORE_FROM;
    eprintln!(
        "scoring {} combinations over {scored} repeated positions",
        Variants::all().len()
    );

    let mut results: Vec<(f64, Variants)> = Vec::new();
    for variants in Variants::all() {
        let mean = score(&model, &backend, &tokens, variants)? / scored as f64;
        results.push((mean, variants));
    }

    results.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!("\n=== ranked ===");
    for (nll, variants) in results.iter().take(10) {
        println!("  {nll:>8.4} nats/token  {}", describe(*variants));
    }
    println!("\nbest: {:#?}", results[0].1);
    Ok(())
}

/// Teacher-forced negative log-likelihood over the repeated tail, one token at
/// a time so every position is scored.
fn score(model: &Model, backend: &HostBackend, tokens: &[u32], variants: Variants) -> Result<f64> {
    let mut state = model.new_state();
    let mut nll = 0.0;
    let mut logits = model.forward_with(&mut state, &tokens[..1], backend, variants)?;
    for (i, &actual) in tokens[1..].iter().enumerate() {
        if i >= SCORE_FROM {
            nll += -log_softmax_at(&logits, actual as usize);
        }
        logits = model.forward_with(&mut state, &[actual], backend, variants)?;
    }
    Ok(nll)
}

fn log_softmax_at(logits: &[f32], index: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&l| (l as f64 - max).exp()).sum();
    (logits[index] as f64 - max) - sum.ln()
}

fn describe(v: Variants) -> String {
    let flag = |on: bool, name: &str| {
        if on {
            name.to_string()
        } else {
            format!("!{name}")
        }
    };
    [
        flag(v.attn_gate_contiguous, "gate_contig"),
        flag(v.attn_gate_first, "gate_first"),
        flag(v.norm_before_gate, "norm_first"),
        flag(v.decay_from_log, "a_is_log"),
        flag(v.l2_normalize_qk, "l2"),
        flag(v.swap_alpha_beta, "swap_ab"),
        flag(v.conv_reversed, "conv_rev"),
        flag(v.query_contracts_value, "q_dot_v"),
    ]
    .join(" ")
}
