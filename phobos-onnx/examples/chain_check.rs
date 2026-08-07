// A small graph mixing device-supported ops with a host fallback (Tanh),
// run through `ChainExec` and checked against the interp oracle:
//
//   cargo run -p phobos-onnx --example chain_check --features cuda

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use phobos_onnx::backend::chain::ChainExec;
use phobos_onnx::backend::{Tensor, host};
use phobos_onnx::ir::{Attribute, DataType, Graph, Node, TensorData, ValueInfo};

fn f32_init(dims: &[i64], data: Vec<f32>) -> Arc<phobos_onnx::ir::Tensor> {
    Arc::new(phobos_onnx::ir::Tensor {
        data_type: DataType::F32,
        dims: dims.to_vec(),
        data: TensorData::F32(data),
    })
}

fn node(op: &str, ins: &[&str], outs: &[&str], attrs: &[(&str, Attribute)]) -> Node {
    Node {
        name: format!("{op}_n"),
        op_type: op.to_string(),
        inputs: ins.iter().map(|s| s.to_string()).collect(),
        outputs: outs.iter().map(|s| s.to_string()).collect(),
        attrs: attrs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    }
}

fn main() -> Result<()> {
    // A mixed chain: MatMul -> Add (device), Tanh (host fallback),
    // Mul -> LayerNorm -> Softmax (device). The fallback in the middle forces
    // one download/sync and a re-upload; the rest chains on the device.
    // Dims are tile-aligned: the Phase-A kernels have no tail masking yet, so a
    // non-aligned extent would (correctly) fall back to the host.
    let (rows, w) = (64usize, 64usize);
    let seed = |k: u64| {
        (0..(rows * w))
            .map(|i| (((i as u64 * 2654435761 + k) % 13) as f32 - 6.0) * 0.02)
            .collect::<Vec<_>>()
    };

    let graph = Graph {
        nodes: vec![
            node("MatMul", &["X", "W"], &["H"], &[]),
            node("Add", &["H", "B"], &["A1"], &[]),
            node("Tanh", &["A1"], &["T"], &[]),
            node("Mul", &["T", "A1"], &["M"], &[]),
            node(
                "LayerNormalization",
                &["M", "scale", "bias"],
                &["L"],
                &[("epsilon", Attribute::Float(1e-5))],
            ),
            node("Softmax", &["L"], &["Y"], &[("axis", Attribute::Int(-1))]),
        ],
        initializers: HashMap::from([
            (
                "W".to_string(),
                f32_init(
                    &[w as i64, w as i64],
                    (0..(w * w))
                        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
                        .collect(),
                ),
            ),
            ("B".to_string(), f32_init(&[rows as i64, w as i64], seed(3))),
            ("scale".to_string(), f32_init(&[w as i64], vec![1.0; w])),
            ("bias".to_string(), f32_init(&[w as i64], vec![0.0; w])),
        ]),
        outputs: vec![ValueInfo {
            name: "Y".to_string(),
            data_type: Some(DataType::F32),
            shape: Default::default(),
        }],
        ..Default::default()
    };

    let inputs = HashMap::from([(
        "X".to_string(),
        Tensor::f32(vec![rows as i64, w as i64], seed(1)),
    )]);

    let want = host::run(&graph, &inputs)?;
    let mut exec = ChainExec::new()?;
    let got = exec.run(&graph, &inputs)?;
    let stats = exec.stats();

    let err = max_rel_err(&got["Y"].to_f32(), &want["Y"].to_f32());
    println!("chain vs interp oracle: max rel err {err:e}");
    println!(
        "device ops {}  host (fallback) ops {}  syncs {}",
        stats.device_ops, stats.host_ops, stats.syncs
    );
    if err > 1e-3 {
        bail!("ChainExec disagrees with the interp oracle (err {err:e})");
    }
    // Expect 5 device ops (MatMul, Add, Mul, LayerNorm, Softmax) and 1 fallback
    // (Tanh). Syncs: one for the fallback's input download, one for the final
    // output download (consecutive device ops chain without a sync).
    if stats.device_ops != 5 || stats.host_ops != 1 {
        bail!("unexpected device/host split: {stats:?}");
    }
    println!("OK: ChainExec matches the oracle; supported ops ran on-device and chained");
    Ok(())
}

fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g - w).abs() / w.abs().max(1e-2))
        .fold(0.0f32, f32::max)
}
