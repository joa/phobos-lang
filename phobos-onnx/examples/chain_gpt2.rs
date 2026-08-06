// The real GPT-2 `decoder_with_past` step graph through `ChainExec`, verified
// against the interp oracle. Prints the on-device against host-fallback op
// histogram.
//
//   cargo run -p phobos-onnx --example chain_gpt2 --features cuda -- models/gpt2-kv

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Result, bail};
use phobos_onnx::chain::ChainExec;
use phobos_onnx::eval::fold_graph;
use phobos_onnx::interp::{self, Tensor};
use phobos_onnx::load_model;
use phobos_onnx::transform;

const N_LAYER: usize = 12;

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/gpt2-kv".to_string());
    let dir = Path::new(&dir);

    let ids: Vec<i64> = vec![4342, 318, 617, 2420, 284, 37773, 18435, 2159];
    let n = ids.len();

    let decoder = load_model(&std::fs::read(dir.join("decoder.onnx"))?)?.graph;
    let with_past = load_model(&std::fs::read(dir.join("decoder_with_past.onnx"))?)?.graph;

    // Prompt pass on the host to build a cache to feed the step.
    let folded = fold_graph(
        &decoder,
        &HashMap::from([("input_ids".to_string(), vec![1, n as i64])]),
    )?;
    let out = interp::run(
        &folded,
        &HashMap::from([("input_ids".to_string(), Tensor::i64(vec![1, n as i64], ids))]),
    )?;
    let next = argmax(&out["logits"].to_f32()[(n - 1) * 50257..n * 50257]) as i64;

    // Build the single-token step inputs (token + cached K/V).
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("input_ids".to_string(), Tensor::i64(vec![1, 1], vec![next]));
    let mut shapes = HashMap::from([("input_ids".to_string(), vec![1, 1])]);
    for l in 0..N_LAYER {
        for kind in ["key", "value"] {
            let src = out[&format!("present.{l}.{kind}")].clone();
            shapes.insert(format!("past.{l}.{kind}"), src.dims.clone());
            inputs.insert(format!("past.{l}.{kind}"), src);
        }
    }
    let step = transform::fuse_layernorm(&fold_graph(&with_past, &shapes)?);
    println!("step graph: {} nodes", step.nodes.len());

    // Oracle (host interp) vs ChainExec (device-resident).
    let want = interp::run(&step, &inputs)?;
    let mut exec = ChainExec::new()?;
    let got = exec.run(&step, &inputs)?;

    let err = max_rel_err(&got["logits"].to_f32(), &want["logits"].to_f32());
    let stats = exec.stats();
    println!("\nChainExec vs interp oracle: max rel err {err:e}");
    println!(
        "device ops {}  host (fallback) ops {}  syncs {}",
        stats.device_ops, stats.host_ops, stats.syncs
    );
    println!("\non-device by op:");
    print_hist(exec.device_hist());
    println!("\nfallback (coverage gap) by op:");
    print_hist(exec.fallback_hist());

    if err > 2e-3 {
        bail!("ChainExec disagrees with the oracle on the real step graph (err {err:e})");
    }
    println!("\nOK: ChainExec runs the real GPT-2 step graph and matches the oracle");
    Ok(())
}

fn print_hist(hist: &HashMap<String, usize>) {
    let sorted: BTreeMap<(usize, &str), ()> = hist
        .iter()
        .map(|(k, v)| ((usize::MAX - v, k.as_str()), ()))
        .collect();
    for ((inv, op), _) in sorted {
        println!("  {:>4}  {op}", usize::MAX - inv);
    }
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
