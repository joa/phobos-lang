// A real exported GPT-2 end to end, load then fold_graph then host interp,
// with every output compared against the model's bundled `test_data_set_0`:
//
//   cargo run -p phobos-onnx --example run_gpt2 -- models/GPT2

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use phobos_onnx::backend::{Tensor, host};
use phobos_onnx::eval::fold_graph;
use phobos_onnx::load_model;
use phobos_onnx::proto::{TensorProto, tensor_proto::DataType};
use phobos_onnx::transform;
use prost::Message;

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .context("usage: run_gpt2 <model-dir>")?;
    let dir = Path::new(&dir);

    let model = load_model(&std::fs::read(dir.join("model.onnx"))?)?;
    let graph = &model.graph;
    let input_name = graph.inputs[0].name.clone();

    // Read the reference input (token ids) and its shape.
    let input_pb = read_pb(&dir.join("test_data_set_0/input_0.pb"))?;
    let dims = input_pb.dims.clone();
    let ids = pb_i64(&input_pb);
    println!("input '{input_name}' dims {dims:?}");

    // Fold to a static graph for this input shape, then fuse the decomposed
    // LayerNorm chains into single LayerNormalization nodes.
    let folded = fold_graph(graph, &HashMap::from([(input_name.clone(), dims.clone())]))?;
    let fused = transform::fuse_layernorm(&folded);
    let lns = fused
        .nodes
        .iter()
        .filter(|n| n.op_type == "LayerNormalization")
        .count();
    println!(
        "folded to {} nodes; fused {lns} LayerNorms -> {} nodes",
        folded.nodes.len(),
        fused.nodes.len()
    );
    let folded = fused;

    let inputs = HashMap::from([(input_name, Tensor::i64(dims, ids))]);
    let start = std::time::Instant::now();
    let outputs = host::run(&folded, &inputs)?;
    println!("interp ran in {:.2?}\n", start.elapsed());

    // Compare each graph output (in order) to output_{i}.pb.
    let mut worst = 0.0f32;
    for (i, vi) in graph.outputs.iter().enumerate() {
        let got = outputs.get(&vi.name).context("missing output")?;
        let want = pb_f32(&read_pb(
            &dir.join(format!("test_data_set_0/output_{i}.pb")),
        )?);
        if got.to_f32().len() != want.len() {
            bail!(
                "output {} length {} != reference {}",
                vi.name,
                got.to_f32().len(),
                want.len()
            );
        }
        let err = max_rel_err(&got.to_f32(), &want);
        worst = worst.max(err);
        println!(
            "  output {:<9} {:>14?}  max rel err {err:e}",
            vi.name, got.dims
        );
    }
    println!("\nworst relative error across all outputs: {worst:e}");
    if worst > 1e-3 {
        bail!("GPT-2 outputs do not match the reference (worst {worst:e})");
    }
    println!("OK: GPT-2 matches the ONNX reference outputs");
    Ok(())
}

fn read_pb(path: &Path) -> Result<TensorProto> {
    Ok(TensorProto::decode(std::fs::read(path)?.as_slice())?)
}

fn pb_i64(t: &TensorProto) -> Vec<i64> {
    if !t.int64_data.is_empty() {
        t.int64_data.clone()
    } else if let Some(raw) = &t.raw_data {
        raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    } else {
        Vec::new()
    }
}

fn pb_f32(t: &TensorProto) -> Vec<f32> {
    if !t.float_data.is_empty() {
        t.float_data.clone()
    } else if let Some(raw) = &t.raw_data {
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    } else if t.data_type == Some(DataType::Int64 as i32) {
        pb_i64(t).iter().map(|&x| x as f32).collect()
    } else {
        Vec::new()
    }
}

/// Max error normalized by the reference magnitude (with a small floor).
fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g - w).abs() / w.abs().max(1e-2))
        .fold(0.0f32, f32::max)
}
