// An ONNX model's opset, inputs, outputs and op-type histogram. Front end
// only, so no GPU is needed:
//
//   cargo run -p phobos-onnx --example inspect -- MODEL.onnx

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use phobos_onnx::ir::{Dim, Shape};
use phobos_onnx::load_model;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: inspect <model.onnx> [node-start node-count]")?;
    let bytes = std::fs::read(&path).with_context(|| format!("reading {path}"))?;
    let model = load_model(&bytes)?;
    let g = &model.graph;

    println!("== {path} ==");
    println!(
        "ir_version {}  producer {:?}",
        model.ir_version, model.producer_name
    );
    let mut opset: Vec<_> = model.opset.iter().collect();
    opset.sort();
    println!("opset {opset:?}");
    println!(
        "nodes {}  initializers {}",
        g.nodes.len(),
        g.initializers.len()
    );

    println!("\ninputs:");
    for vi in &g.inputs {
        println!("  {:<24} {:?} {}", vi.name, vi.data_type, fmt_shape(&vi.shape));
    }
    println!("outputs:");
    for vi in &g.outputs {
        println!("  {:<24} {:?} {}", vi.name, vi.data_type, fmt_shape(&vi.shape));
    }

    let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
    for n in &g.nodes {
        *hist.entry(n.op_type.as_str()).or_default() += 1;
    }
    let mut by_count: Vec<_> = hist.iter().collect();
    by_count.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    println!("\nop types ({}):", hist.len());
    for (op, count) in by_count {
        println!("  {count:>5}  {op}");
    }

    // Optional: dump a window of nodes with `-- <model> <start> <count>`.
    if let (Some(start), Some(count)) = (arg_usize(2), arg_usize(3)) {
        println!("\nnodes [{start}..{}]:", start + count);
        for n in g.nodes.iter().skip(start).take(count) {
            let attrs: Vec<&str> = n.attrs.keys().map(|s| s.as_str()).collect();
            println!(
                "  {:<20} {:?} -> {:?}  attrs {:?}",
                n.op_type, n.inputs, n.outputs, attrs
            );
        }
    }
    Ok(())
}

fn arg_usize(i: usize) -> Option<usize> {
    std::env::args().nth(i)?.parse().ok()
}

fn fmt_shape(shape: &Shape) -> String {
    match &shape.0 {
        None => "<unranked>".to_string(),
        Some(dims) => {
            let parts: Vec<String> = dims
                .iter()
                .map(|d| match d {
                    Dim::Fixed(n) => n.to_string(),
                    Dim::Symbol(s) => s.clone(),
                    Dim::Unknown => "?".to_string(),
                })
                .collect();
            format!("[{}]", parts.join(", "))
        }
    }
}
