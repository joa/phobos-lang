// A GPT-2-style transformer sublayer on the GPU through Phobos, checked
// against a CPU reference:
//
//   ln = LayerNorm(X, g, b)
//   h  = Gelu(ln @ W1 + b1)
//   o  = h @ W2 + b2
//   Y  = Softmax(o)
//
// LayerNormalization, MatMul, a broadcast bias-row Add, Gelu and Softmax in one
// graph.
//
//   cargo run -p phobos-onnx --example run_transformer_ffn --features cuda

use std::collections::HashMap;

use anyhow::{Result, bail};
use phobos_onnx::proto::{self, tensor_proto::DataType};
use phobos_onnx::{load_model, runner};
use prost::Message;

const S: usize = 64; // sequence length (rows)
const W: usize = 64; // model / hidden width
const H: usize = 128; // feed-forward inner width
const EPS: f32 = 1e-5;
const GELU_C: f32 = 1.702;

fn main() -> Result<()> {
    let x = fill(S * W, |i| ((i % 13) as f32 - 6.0) * 0.1);
    let g = fill(W, |i| 1.0 + ((i % 4) as f32) * 0.05);
    let bn = fill(W, |i| ((i % 5) as f32 - 2.0) * 0.02);
    let w1 = fill(W * H, |i| ((i % 7) as f32 - 3.0) * 0.03);
    let b1 = fill(H, |i| ((i % 3) as f32 - 1.0) * 0.1);
    let w2 = fill(H * W, |i| ((i % 5) as f32 - 2.0) * 0.02);
    let b2 = fill(W, |i| ((i % 6) as f32 - 3.0) * 0.05);

    let model_bytes = build_graph(&g, &bn, &w1, &b1, &w2, &b2)?;
    let model = load_model(&model_bytes)?;

    let inputs = HashMap::from([("X".to_string(), x.clone())]);
    let outputs = runner::run(&model.graph, &inputs)?;
    let got = outputs.get("Y").expect("output Y");

    let want = reference(&x, &g, &bn, &w1, &b1, &w2, &b2);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("ran LayerNorm -> MatMul -> bias -> Gelu -> MatMul -> bias -> Softmax");
    println!(
        "{} outputs, max abs error vs CPU reference: {max_err:e}",
        got.len()
    );
    if max_err > 2e-3 {
        bail!("result mismatch: max abs error {max_err:e} exceeds tolerance");
    }
    println!("OK: GPU result matches the CPU reference");
    Ok(())
}

fn fill(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}

// ---- CPU reference -------------------------------------------------------

fn reference(
    x: &[f32],
    g: &[f32],
    bn: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> Vec<f32> {
    let ln = layernorm(x, g, bn, S, W);
    let h1 = matmul(&ln, w1, S, W, H);
    let h1 = bias_add(&h1, b1, S, H);
    let a = h1.iter().map(|&z| gelu(z)).collect::<Vec<_>>();
    let o = matmul(&a, w2, S, H, W);
    let o = bias_add(&o, b2, S, W);
    softmax(&o, S, W)
}

fn layernorm(x: &[f32], g: &[f32], b: &[f32], rows: usize, w: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * w];
    for r in 0..rows {
        let row = &x[r * w..r * w + w];
        let mean = row.iter().sum::<f32>() / w as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / w as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for c in 0..w {
            y[r * w + c] = (row[c] - mean) * inv * g[c] + b[c];
        }
    }
    y
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn bias_add(a: &[f32], bias: &[f32], rows: usize, w: usize) -> Vec<f32> {
    let mut y = a.to_vec();
    for r in 0..rows {
        for c in 0..w {
            y[r * w + c] += bias[c];
        }
    }
    y
}

fn gelu(z: f32) -> f32 {
    z / (1.0 + (-GELU_C * z).exp())
}

fn softmax(x: &[f32], rows: usize, w: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * w];
    for r in 0..rows {
        let row = &x[r * w..r * w + w];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for c in 0..w {
            y[r * w + c] = exps[c] / sum;
        }
    }
    y
}

// ---- ONNX graph construction ---------------------------------------------

fn build_graph(
    g: &[f32],
    bn: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> Result<Vec<u8>> {
    let (s, w, h) = (S as i64, W as i64, H as i64);
    let graph = proto::GraphProto {
        name: Some("ffn".into()),
        node: vec![
            node("LayerNormalization", &["X", "g", "bn"], &["ln"]),
            node("MatMul", &["ln", "W1"], &["m1"]),
            node("Add", &["m1", "b1"], &["hb"]),
            node("Gelu", &["hb"], &["a"]),
            node("MatMul", &["a", "W2"], &["m2"]),
            node("Add", &["m2", "b2"], &["o"]),
            node("Softmax", &["o"], &["Y"]),
        ],
        initializer: vec![
            initializer("g", &[w], g),
            initializer("bn", &[w], bn),
            initializer("W1", &[w, h], w1),
            initializer("b1", &[h], b1),
            initializer("W2", &[h, w], w2),
            initializer("b2", &[w], b2),
        ],
        input: vec![value_info("X", &[s, w])],
        output: vec![value_info("Y", &[s, w])],
        ..Default::default()
    };
    let model = proto::ModelProto {
        ir_version: Some(9),
        producer_name: Some("phobos-onnx-example".into()),
        opset_import: vec![proto::OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(21),
        }],
        graph: Some(graph),
        ..Default::default()
    };
    Ok(model.encode_to_vec())
}

fn value_info(name: &str, dims: &[i64]) -> proto::ValueInfoProto {
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

fn initializer(name: &str, dims: &[i64], data: &[f32]) -> proto::TensorProto {
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
