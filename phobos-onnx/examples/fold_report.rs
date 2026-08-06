// The constant-folding and shape evaluator over a model at a fixed input
// shape, reporting how many edges resolved, which ops still block folding, and
// the residual compute-op histogram the runner would need:
//
//   cargo run -p phobos-onnx --example fold_report -- MODEL.onnx [input_0.pb]
//   cargo run -p phobos-onnx --example fold_report -- MODEL.onnx 1 1 8

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context, Result};
use phobos_onnx::eval::{evaluate, fold_graph};
use phobos_onnx::load_model;
use phobos_onnx::proto::TensorProto;
use prost::Message;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .context("usage: fold_report <model.onnx> [input.pb | dims...]")?;
    let rest: Vec<String> = args.collect();

    let model = load_model(&std::fs::read(&path)?)?;
    let g = &model.graph;
    let input_name = g
        .inputs
        .first()
        .context("model has no inputs")?
        .name
        .clone();

    // Input dims: from a .pb TensorProto, or explicit integers, else a default.
    let dims: Vec<i64> = if rest.len() == 1 && rest[0].ends_with(".pb") {
        let t = TensorProto::decode(std::fs::read(&rest[0])?.as_slice())?;
        t.dims
    } else if !rest.is_empty() {
        rest.iter().map(|s| s.parse().unwrap()).collect()
    } else {
        vec![1, 1, 8]
    };
    println!("input '{input_name}' dims {dims:?}\n");

    let inputs = HashMap::from([(input_name, dims)]);
    let ev = evaluate(g, &inputs);

    let total = g.nodes.len();
    let with_shape = g
        .nodes
        .iter()
        .filter(|n| n.outputs.iter().any(|o| ev.vals.contains_key(o)))
        .count();
    let folded: usize = ev.vals.values().filter(|v| v.data.is_some()).count();
    println!("nodes: {total}");
    println!("nodes with a resolved output shape: {with_shape}");
    println!("edges folded to constants: {folded}");

    if !ev.unsupported.is_empty() {
        println!("\nunsupported ops (block folding):");
        for (op, c) in sorted(&ev.unsupported) {
            println!("  {c:>5}  {op}");
        }
    }

    // Residual compute ops: nodes whose output shape resolved but that are not
    // pure constants (the actual work the runner must lower).
    let mut compute: BTreeMap<String, usize> = BTreeMap::new();
    for n in &g.nodes {
        let out_const = n
            .outputs
            .iter()
            .all(|o| o.is_empty() || ev.vals.get(o).map(|v| v.data.is_some()).unwrap_or(false));
        let has_shape = n.outputs.iter().any(|o| ev.vals.contains_key(o));
        if has_shape && !out_const {
            *compute.entry(n.op_type.clone()).or_default() += 1;
        }
    }
    println!("\nresidual compute ops (shape known, not constant):");
    for (op, c) in sorted(&compute) {
        println!("  {c:>5}  {op}");
    }

    // Build the residual static graph and report its size.
    match fold_graph(g, &inputs) {
        Ok(residual) => {
            let mut op_hist: BTreeMap<String, usize> = BTreeMap::new();
            for n in &residual.nodes {
                *op_hist.entry(n.op_type.clone()).or_default() += 1;
            }
            println!(
                "\nfold_graph -> residual: {} nodes, {} initializers, {} op types",
                residual.nodes.len(),
                residual.initializers.len(),
                op_hist.len()
            );
            for (op, c) in sorted(&op_hist) {
                println!("  {c:>5}  {op}");
            }
        }
        Err(e) => println!("\nfold_graph failed: {e:#}"),
    }

    // Frontier: first nodes whose output shape is still unknown, with the
    // resolution state of their inputs (to see what blocked them).
    println!("\nfrontier (first unresolved nodes):");
    let mut shown = 0;
    for n in &g.nodes {
        let unresolved = n
            .outputs
            .iter()
            .any(|o| !o.is_empty() && !ev.vals.contains_key(o));
        if !unresolved {
            continue;
        }
        let in_state: Vec<String> = n
            .inputs
            .iter()
            .map(|e| {
                if e.is_empty() {
                    "-".into()
                } else if let Some(v) = ev.vals.get(e) {
                    format!("{:?}{}", v.dims, if v.data.is_some() { "c" } else { "" })
                } else {
                    format!("?{e}")
                }
            })
            .collect();
        println!("  {:<16} in {:?}", n.op_type, in_state);
        shown += 1;
        if shown >= 15 {
            break;
        }
    }
    Ok(())
}

fn sorted(m: &BTreeMap<String, usize>) -> Vec<(&String, &usize)> {
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    v
}
