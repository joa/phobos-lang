// A real exported GPT-2 with the FLOP-heavy Gemm projections on Phobos GPU
// kernels, padded to tile-aligned dims, verified against the model's bundled
// reference. Also spot-checks the padded matmul against the host on GPT-2's
// own non-tile-aligned shapes.
//
//   cargo run -p phobos-onnx --example run_gpt2_gpu --features cuda -- models/GPT2

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use phobos_onnx::backend::device::GpuBackend;
use phobos_onnx::backend::{HostBackend, MatmulBackend, Tensor, host};
use phobos_onnx::eval::fold_graph;
use phobos_onnx::load_model;
use phobos_onnx::proto::{TensorProto, tensor_proto::DataType};
use phobos_onnx::transform;
use prost::Message;

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .context("usage: run_gpt2_gpu <model-dir>")?;
    let dir = Path::new(&dir);

    let gpu = GpuBackend::new()?;

    // Spot-check the padded GPU matmul against the host on real GPT-2 shapes
    // (M = seq = 8; N = 768/2304/3072 projections and 50257 lm-head; none are
    // tile-aligned).
    println!("padded matmul vs host on GPT-2 shapes:");
    for &(m, k, n) in &[
        (8, 768, 768),
        (8, 768, 2304),
        (8, 768, 3072),
        (8, 768, 50257),
    ] {
        let a = pseudo(m * k, 1);
        let b = pseudo(k * n, 2);
        let g = gpu.matmul(&a, m, k, &b, n)?;
        let h = HostBackend.matmul(&a, m, k, &b, n)?;
        let err = max_rel_err(&g, &h);
        println!("  [{m}x{k}] @ [{k}x{n}] -> max rel err {err:e}");
        if err > 1e-3 {
            bail!("GPU matmul disagrees with host for {m}x{k}x{n}");
        }
    }

    // Full model with Gemms on the GPU.
    let model = load_model(&std::fs::read(dir.join("model.onnx"))?)?;
    let graph = &model.graph;
    let input_name = graph.inputs[0].name.clone();
    let input_pb = read_pb(&dir.join("test_data_set_0/input_0.pb"))?;
    let dims = input_pb.dims.clone();
    let ids = pb_i64(&input_pb);

    let folded = fold_graph(graph, &HashMap::from([(input_name.clone(), dims.clone())]))?;
    let folded = transform::fuse_layernorm(&folded);
    let lns = folded
        .nodes
        .iter()
        .filter(|n| n.op_type == "LayerNormalization")
        .count();
    let inputs = HashMap::from([(input_name, Tensor::i64(dims, ids))]);

    println!(
        "\nrunning {} with Gemms + {lns} LayerNorms on Phobos GPU...",
        dir.display()
    );
    let start = std::time::Instant::now();
    let outputs = host::run_with(&folded, &inputs, &gpu)?;
    println!("ran in {:.2?}\n", start.elapsed());

    let mut worst = 0.0f32;
    for (i, vi) in graph.outputs.iter().enumerate() {
        let got = outputs.get(&vi.name).context("missing output")?.to_f32();
        let want = pb_f32(&read_pb(
            &dir.join(format!("test_data_set_0/output_{i}.pb")),
        )?);
        let err = max_rel_err(&got, &want);
        worst = worst.max(err);
        println!("  output {:<9} max rel err {err:e}", vi.name);
    }
    println!("\nworst relative error vs reference: {worst:e}");
    if worst > 2e-3 {
        bail!("GPU GPT-2 does not match the reference (worst {worst:e})");
    }
    println!("OK: GPT-2 with Phobos-GPU Gemms matches the ONNX reference");
    Ok(())
}

fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    // Small deterministic values so the naive host matmul stays well-conditioned.
    (0..n)
        .map(|i| (((i as u64 * 2654435761 + seed) % 97) as f32 - 48.0) * 0.02)
        .collect()
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

fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g - w).abs() / w.abs().max(1e-2))
        .fold(0.0f32, f32::max)
}
