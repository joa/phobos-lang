// The KV-cache invariant for the exported with-past GPT-2: a single-token step
// fed the cached K/V has to reproduce the last-position logits of a full
// recompute over the whole sequence.
//
//   cargo run -p phobos-onnx --example kv_check -- models/gpt2-kv

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use phobos_onnx::backend::{Tensor, host};
use phobos_onnx::eval::fold_graph;
use phobos_onnx::load_model;

const N_LAYER: usize = 12;

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .context("usage: kv_check <gpt2-kv-dir>")?;
    let dir = Path::new(&dir);

    // "Here is some text to encode Hello World"
    let ids: Vec<i64> = vec![4342, 318, 617, 2420, 284, 37773, 18435, 2159];
    let n = ids.len();

    let decoder = load_model(&std::fs::read(dir.join("decoder.onnx"))?)?.graph;
    let with_past = load_model(&std::fs::read(dir.join("decoder_with_past.onnx"))?)?.graph;

    // --- prompt pass: full sequence through the no-past decoder ---
    let folded = fold_graph(
        &decoder,
        &HashMap::from([("input_ids".to_string(), vec![1, n as i64])]),
    )?;
    let inputs = HashMap::from([(
        "input_ids".to_string(),
        Tensor::i64(vec![1, n as i64], ids.clone()),
    )]);
    let out = host::run(&folded, &inputs)?;

    let logits = out.get("logits").context("no logits")?.to_f32();
    let next = argmax(&logits[(n - 1) * 50257..n * 50257]) as i64;
    println!("prompt {n} tokens -> next id {next}");

    // Cache the present K/V as the step's past inputs.
    let mut step_inputs: HashMap<String, Tensor> = HashMap::new();
    step_inputs.insert("input_ids".to_string(), Tensor::i64(vec![1, 1], vec![next]));
    let mut past_dims = HashMap::new();
    for l in 0..N_LAYER {
        for kind in ["key", "value"] {
            let name = format!("past.{l}.{kind}");
            let src = out
                .get(&format!("present.{l}.{kind}"))
                .context("no present")?;
            past_dims.insert(name.clone(), src.dims.clone());
            step_inputs.insert(name, src.clone());
        }
    }

    // --- step: one token with past through the with-past decoder ---
    let mut shapes = HashMap::from([("input_ids".to_string(), vec![1, 1])]);
    shapes.extend(past_dims);
    let folded_step = fold_graph(&with_past, &shapes)?;
    let step_out = host::run(&folded_step, &step_inputs)?;
    let step_logits = step_out.get("logits").context("no step logits")?.to_f32();

    // --- oracle: full recompute over n+1 tokens ---
    let mut ids2 = ids.clone();
    ids2.push(next);
    let folded2 = fold_graph(
        &decoder,
        &HashMap::from([("input_ids".to_string(), vec![1, (n + 1) as i64])]),
    )?;
    let inputs2 = HashMap::from([(
        "input_ids".to_string(),
        Tensor::i64(vec![1, (n + 1) as i64], ids2),
    )]);
    let out2 = host::run(&folded2, &inputs2)?;
    let full_logits = out2.get("logits").context("no logits")?.to_f32();
    let full_last = &full_logits[n * 50257..(n + 1) * 50257];

    let err = max_rel_err(&step_logits, full_last);
    println!("KV-step vs full-recompute last-row logits: max rel err {err:e}");
    println!(
        "  argmax: step {}  full {}",
        argmax(&step_logits),
        argmax(full_last)
    );
    if err > 1e-3 {
        bail!("KV cache does not match full recompute (err {err:e})");
    }
    println!("OK: with-past step reproduces the full-recompute logits");
    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g - w).abs() / w.abs().max(1e-2))
        .fold(0.0f32, f32::max)
}
