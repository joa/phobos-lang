// What rounding the K-quant projections' activations to int8 costs, on the
// host reference, at one scale per 32 (the device today) and per 256:
//
//   cargo run --release -p phobos-gguf --example act_numerics -- \
//       MODEL.gguf [-p PROMPT] [-n TOKENS] [-r RUN,RUN..]
//
// The prompt is fed a token at a time, so every step's next-token
// distribution is compared with the exact pass's: the KL divergence from
// it (mean and worst step), how often the top token agrees, and per
// projection shape the relative error of the product itself. Needs no GPU.
use std::collections::BTreeMap;

use anyhow::{Result, bail};
use phobos_gguf::backend::{ActError, HostBackend};
use phobos_gguf::{Bpe, Decoder, Gguf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        bail!("usage: act_numerics MODEL.gguf [-p PROMPT] [-n TOKENS] [-r RUN,RUN..]");
    };
    let after = |flag: &str| args.iter().position(|a| a == flag).and_then(|at| args.get(at + 1)).cloned();
    let prompt = after("-p").unwrap_or_else(|| {
        "The quick brown fox jumps over the lazy dog. In 1969, astronauts landed on the Moon, and the world watched. def fib(n): return n if n < 2 else fib(n - 1) + fib(n - 2)".to_string()
    });
    let count: usize = after("-n").map(|n| n.parse()).transpose()?.unwrap_or(24);
    let runs: Vec<usize> = match after("-r") {
        Some(list) => list.split(',').map(|r| r.trim().parse()).collect::<Result<_, _>>()?,
        None => vec![32, 256],
    };

    let gguf = Gguf::open(path.as_ref())?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    let model = Decoder::load(&gguf)?;
    let mut tokens = bpe.encode(&prompt)?;
    tokens.truncate(count);
    println!("{} tokens, schemes: exact, {}", tokens.len(), runs.iter().map(|r| format!("per {r}")).collect::<Vec<_>>().join(", "));

    let exact = pass(&model, &tokens, 0)?.0;
    for &run in &runs {
        let (probs, errors) = pass(&model, &tokens, run)?;
        let kls: Vec<f64> = exact.iter().zip(&probs).map(|(p, q)| kl(p, q)).collect();
        let agree = exact.iter().zip(&probs).filter(|(p, q)| argmax(p) == argmax(q)).count();
        let mean = kls.iter().sum::<f64>() / kls.len() as f64;
        let worst = kls.iter().cloned().fold(0.0f64, f64::max);
        println!("\nper {run}: KL from exact, mean {mean:.3e}  worst {worst:.3e}  top-1 agrees {agree}/{}", kls.len());
        // Per projection shape: the K-quant projections tell apart by (k, n).
        let mut by_shape: BTreeMap<(usize, usize), (usize, f64, f64)> = BTreeMap::new();
        for e in &errors {
            let entry = by_shape.entry((e.k, e.n)).or_insert((0, 0.0, 0.0));
            entry.0 += 1;
            entry.1 += e.rel;
            entry.2 = entry.2.max(e.rel);
        }
        println!("  product error by shape (k x n): calls, mean, worst");
        for ((k, n), (calls, sum, worst)) in &by_shape {
            println!("    {k:>6} x {n:<6} {calls:>5}  {:.3e}  {worst:.3e}", sum / *calls as f64);
        }
    }
    Ok(())
}

/// The prompt a token at a time: each step's next-token probabilities, and
/// the projection errors the rounding gathered.
fn pass(model: &Decoder, tokens: &[u32], run: usize) -> Result<(Vec<Vec<f64>>, Vec<ActError>)> {
    let host = HostBackend::new();
    host.round_activations(run);
    let mut state = model.new_state();
    let mut probs = Vec::with_capacity(tokens.len());
    for &t in tokens {
        let logits = model.forward(&mut state, &[t], &host)?;
        probs.push(softmax(&logits));
    }
    Ok((probs, host.act_errors()))
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&v| (v as f64 - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// KL(p || q), with the probabilities floored so a q of zero stays finite.
fn kl(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).filter(|(pi, _)| **pi > 0.0).map(|(&pi, &qi)| pi * (pi / qi.max(1e-300)).ln()).sum()
}

fn argmax(p: &[f64]) -> usize {
    p.iter().enumerate().fold(0, |best, (i, &v)| if v > p[best] { i } else { best })
}
