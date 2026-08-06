// The two fusion passes on the GPU:
//
// 1. Linear epilogue: `MatMul -> Add(bias) -> Gelu` becomes one kernel, and
//    fused, unfused and the CPU reference all have to agree.
// 2. Attention: `Transpose -> MatMul -> Mul(scale) -> Softmax -> MatMul`
//    becomes one flash-attention kernel, checked against the CPU only, since
//    the unfused path needs a scalar-broadcast Mul that is not emitted yet.
//
// Both also assert the fused graph actually collapsed the group.
//
//   cargo run -p phobos-onnx --example run_fusion --features cuda

use std::collections::HashMap;

use anyhow::{Result, bail};
use phobos_onnx::backend::device as runner;
use phobos_onnx::load_model;
use phobos_onnx::proto::{self, tensor_proto::DataType};
use phobos_onnx::transform::{self, FLASH_ATTENTION, FUSED_LINEAR};
use prost::Message;

fn main() -> Result<()> {
    check_linear_fusion()?;
    check_attention_fusion()?;
    println!("OK: both fusions produce correct GPU results");
    Ok(())
}

/// MatMul + bias + Gelu, verified fused == unfused == CPU.
fn check_linear_fusion() -> Result<()> {
    const M: usize = 128;
    const K: usize = 64;
    const N: usize = 128;
    let x = fill(M * K, |i| ((i % 11) as f32 - 5.0) * 0.05);
    let w = fill(K * N, |i| ((i % 7) as f32 - 3.0) * 0.02);
    let b = fill(N, |i| ((i % 5) as f32 - 2.0) * 0.1);

    let graph = load_model(&linear_graph(M, K, N, &w, &b)?)?.graph;
    let fused = transform::fuse(&graph);
    // The three nodes (MatMul, Add, Gelu) collapse to one PhobosFusedLinear.
    if fused.nodes.len() != 1 || fused.nodes[0].op_type != FUSED_LINEAR {
        bail!(
            "linear fusion did not collapse the block: {:?}",
            node_types(&fused)
        );
    }

    let inputs = HashMap::from([("X".to_string(), x.clone())]);
    let unfused_out = runner::run(&graph, &inputs)?;
    let fused_out = runner::run(&fused, &inputs)?;

    let want = linear_reference(&x, &w, &b, M, K, N);
    let e_unfused = max_err(&unfused_out["Y"], &want);
    let e_fused = max_err(&fused_out["Y"], &want);
    let e_fu = max_err(&fused_out["Y"], &unfused_out["Y"]);
    println!(
        "linear fusion: unfused err {e_unfused:e}, fused err {e_fused:e}, fused-vs-unfused {e_fu:e}"
    );
    if e_unfused.max(e_fused).max(e_fu) > 2e-3 {
        bail!("linear fusion mismatch");
    }
    Ok(())
}

/// Scaled single-head attention, verified fused == CPU.
fn check_attention_fusion() -> Result<()> {
    const S: usize = 64;
    const D: usize = 32;
    let scale = 1.0 / (D as f32).sqrt();
    let q = fill(S * D, |i| ((i % 13) as f32 - 6.0) * 0.05);
    let k = fill(S * D, |i| ((i % 9) as f32 - 4.0) * 0.04);
    let v = fill(S * D, |i| ((i % 7) as f32 - 3.0) * 0.06);

    let graph = load_model(&attention_graph(S, D, scale)?)?.graph;
    let fused = transform::fuse(&graph);
    if fused.nodes.len() != 1 || fused.nodes[0].op_type != FLASH_ATTENTION {
        bail!(
            "attention fusion did not collapse the block: {:?}",
            node_types(&fused)
        );
    }

    let inputs = HashMap::from([
        ("Q".to_string(), q.clone()),
        ("K".to_string(), k.clone()),
        ("V".to_string(), v.clone()),
    ]);
    let fused_out = runner::run(&fused, &inputs)?;
    let want = attention_reference(&q, &k, &v, scale, S, D);
    let err = max_err(&fused_out["O"], &want);
    println!("attention fusion: flash err {err:e}");
    if err > 2e-3 {
        bail!("attention fusion mismatch");
    }
    Ok(())
}

// ---- references ----------------------------------------------------------

fn linear_reference(x: &[f32], w: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += x[i * k + p] * w[p * n + j];
            }
            let z = acc + b[j];
            y[i * n + j] = z / (1.0 + (-1.702 * z).exp()); // gelu (logistic)
        }
    }
    y
}

fn attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    scale: f32,
    s: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; s * d];
    for i in 0..s {
        // scores row, scaled.
        let mut scores = vec![0.0f32; s];
        for j in 0..s {
            let mut dot = 0.0f32;
            for e in 0..d {
                dot += q[i * d + e] * k[j * d + e];
            }
            scores[j] = dot * scale;
        }
        // softmax.
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        // weighted sum of V.
        for e in 0..d {
            let mut acc = 0.0f32;
            for j in 0..s {
                acc += (exps[j] / sum) * v[j * d + e];
            }
            out[i * d + e] = acc;
        }
    }
    out
}

// ---- graph builders ------------------------------------------------------

fn linear_graph(m: usize, k: usize, n: usize, w: &[f32], b: &[f32]) -> Result<Vec<u8>> {
    let (mi, ki, ni) = (m as i64, k as i64, n as i64);
    let graph = proto::GraphProto {
        name: Some("linear".into()),
        node: vec![
            node("MatMul", &["X", "W"], &["m"]),
            node("Add", &["m", "b"], &["hb"]),
            node("Gelu", &["hb"], &["Y"]),
        ],
        initializer: vec![init_f32("W", &[ki, ni], w), init_f32("b", &[ni], b)],
        input: vec![vinfo("X", &[mi, ki])],
        output: vec![vinfo("Y", &[mi, ni])],
        ..Default::default()
    };
    Ok(wrap(graph))
}

fn attention_graph(s: usize, d: usize, scale: f32) -> Result<Vec<u8>> {
    let (si, di) = (s as i64, d as i64);
    let graph = proto::GraphProto {
        name: Some("attention".into()),
        node: vec![
            transpose_node("K", "Kt", &[1, 0]),
            node("MatMul", &["Q", "Kt"], &["scores"]),
            node("Mul", &["scores", "scale"], &["scaled"]),
            node("Softmax", &["scaled"], &["p"]),
            node("MatMul", &["p", "V"], &["O"]),
        ],
        initializer: vec![init_f32("scale", &[1], &[scale])],
        input: vec![
            vinfo("Q", &[si, di]),
            vinfo("K", &[si, di]),
            vinfo("V", &[si, di]),
        ],
        output: vec![vinfo("O", &[si, di])],
        ..Default::default()
    };
    Ok(wrap(graph))
}

// ---- helpers -------------------------------------------------------------

fn fill(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn node_types(g: &phobos_onnx::ir::Graph) -> Vec<String> {
    g.nodes.iter().map(|n| n.op_type.clone()).collect()
}

fn wrap(graph: proto::GraphProto) -> Vec<u8> {
    proto::ModelProto {
        ir_version: Some(9),
        opset_import: vec![proto::OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(21),
        }],
        graph: Some(graph),
        ..Default::default()
    }
    .encode_to_vec()
}

fn vinfo(name: &str, dims: &[i64]) -> proto::ValueInfoProto {
    proto::ValueInfoProto {
        name: Some(name.to_string()),
        r#type: Some(proto::TypeProto {
            value: Some(proto::type_proto::Value::TensorType(
                proto::type_proto::Tensor {
                    elem_type: Some(DataType::Float as i32),
                    shape: Some(proto::TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|&n| proto::tensor_shape_proto::Dimension {
                                value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(
                                    n,
                                )),
                                denotation: None,
                            })
                            .collect(),
                    }),
                },
            )),
            denotation: None,
        }),
        ..Default::default()
    }
}

fn init_f32(name: &str, dims: &[i64], data: &[f32]) -> proto::TensorProto {
    proto::TensorProto {
        name: Some(name.to_string()),
        data_type: Some(DataType::Float as i32),
        dims: dims.to_vec(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}

fn node(op: &str, ins: &[&str], outs: &[&str]) -> proto::NodeProto {
    proto::NodeProto {
        input: ins.iter().map(|s| s.to_string()).collect(),
        output: outs.iter().map(|s| s.to_string()).collect(),
        op_type: Some(op.to_string()),
        name: Some(format!("{op}_0")),
        ..Default::default()
    }
}

fn transpose_node(input: &str, output: &str, perm: &[i64]) -> proto::NodeProto {
    let mut n = node("Transpose", &[input], &[output]);
    n.attribute = vec![proto::AttributeProto {
        name: Some("perm".into()),
        r#type: Some(proto::attribute_proto::AttributeType::Ints as i32),
        ints: perm.to_vec(),
        ..Default::default()
    }];
    n
}
