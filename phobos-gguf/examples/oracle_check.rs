// The device forward pass in the shape a llama.cpp server reports it, for a
// token-level diff against a reference build:
//
//   cargo run --release -p phobos-gguf --features cuda --example oracle_check -- \
//       MODEL.gguf -k 8 -n 24 "The capital of France is" "def fib(n):"
//
// Per prompt, one JSON line: the prompt's token ids, the top `k` next tokens
// with their log-probabilities after the whole prompt in one pass, and `n`
// greedy tokens after that: what `llama-server`'s `/completion` returns for
// the same ids with `n_probs` set and greedy sampling. `--ids 1,2,3` takes a
// prompt as token ids, to probe a position a diff has already found.

use std::path::PathBuf;

use anyhow::{Result, bail};
use phobos_gguf::backend::Backend;
use phobos_gguf::{Bpe, Decoder, Gguf};

fn make_backend() -> Result<Box<dyn Backend>> {
    #[cfg(feature = "cuda")]
    {
        Ok(Box::new(phobos_gguf::backend::device::DeviceBackend::new()?))
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(Box::new(phobos_gguf::backend::HostBackend::new()))
    }
}

fn main() -> Result<()> {
    let (mut path, mut prompts) = (None::<PathBuf>, Vec::new());
    let (mut top_k, mut greedy) = (8usize, 24usize);
    let mut id_prompts: Vec<Vec<u32>> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-k" => top_k = args.next().unwrap_or_default().parse()?,
            "-n" => greedy = args.next().unwrap_or_default().parse()?,
            "--ids" => id_prompts.push(
                args.next()
                    .unwrap_or_default()
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?,
            ),
            other if path.is_none() => path = Some(other.into()),
            other => prompts.push(other.to_string()),
        }
    }
    let Some(path) = path.filter(|_| !prompts.is_empty() || !id_prompts.is_empty()) else {
        bail!("usage: oracle_check [-k TOP] [-n GREEDY] [--ids ID,...] MODEL.gguf PROMPT...");
    };

    let gguf = Gguf::open(&path)?;
    let bpe = Bpe::from_vocab(&gguf.vocab()?)?;
    let model = Decoder::load(&gguf)?;
    let backend = make_backend()?;
    let backend = backend.as_ref();

    let mut inputs = Vec::with_capacity(prompts.len() + id_prompts.len());
    for prompt in prompts {
        inputs.push((bpe.encode(&prompt)?, prompt));
    }
    for ids in id_prompts {
        let text = bpe.decode(&ids);
        inputs.push((ids, text));
    }
    for (ids, prompt) in inputs {
        let mut state = model.new_state();
        let logits = model.forward(&mut state, &ids, backend)?;
        let top = log_softmax_top(&logits, top_k);
        let mut tokens = Vec::with_capacity(greedy);
        let mut next = top[0].0;
        for _ in 0..greedy {
            tokens.push(next);
            next = model.forward_greedy(&mut state, &[next], backend)? as u32;
        }
        state.release(backend);

        let top: Vec<String> = top
            .iter()
            .map(|(id, lp)| format!("[{id}, {lp:.5}]"))
            .collect();
        println!(
            "{{\"prompt\": {prompt:?}, \"ids\": {ids:?}, \"top\": [{}], \"greedy\": {tokens:?}, \"text\": {:?}}}",
            top.join(", "),
            bpe.decode(&tokens)
        );
    }
    Ok(())
}

/// The `k` likeliest ids and their log-probabilities, likeliest first.
fn log_softmax_top(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits.iter().map(|&v| (v - max).exp()).sum::<f32>().ln() + max;
    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v - log_sum))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(k);
    ranked
}
