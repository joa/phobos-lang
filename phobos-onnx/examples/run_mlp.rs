// `Y = Relu(X @ W + B)` built as ONNX protobuf, loaded into the IR, run on the
// GPU through Phobos and checked against a CPU reference. W and B are
// initializers and X is the graph input.
//
//   cargo run -p phobos-onnx --example run_mlp --features cuda

use std::collections::HashMap;

use anyhow::{Result, bail};
use phobos_onnx::proto::{self, tensor_proto::DataType};
use phobos_onnx::{load_model, runner};
use prost::Message;

const M: usize = 128;
const K: usize = 64;
const N: usize = 128;

fn main() -> Result<()> {
    // Deterministic, small inputs.
    let x: Vec<f32> = (0..M * K).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let w: Vec<f32> = (0..K * N).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
    let b: Vec<f32> = (0..M * N).map(|i| ((i % 3) as f32 - 1.0) * 0.2).collect();

    let model_bytes = build_mlp_onnx(&w, &b)?;
    let model = load_model(&model_bytes)?;

    let mut inputs = HashMap::new();
    inputs.insert("X".to_string(), x.clone());

    let outputs = runner::run(&model.graph, &inputs)?;
    let got = outputs.get("Y").expect("output Y");

    let want = reference(&x, &w, &b);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);

    println!("ran MatMul+Add+Relu graph: {} outputs", got.len());
    println!("max abs error vs CPU reference: {max_err:e}");
    if max_err > 1e-4 {
        bail!("result mismatch: max abs error {max_err:e} exceeds tolerance");
    }
    println!("OK: GPU result matches the CPU reference");
    Ok(())
}

/// CPU reference for Y = Relu(X @ W + B).
fn reference(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for k in 0..K {
                acc += x[m * K + k] * w[k * N + n];
            }
            y[m * N + n] = (acc + b[m * N + n]).max(0.0);
        }
    }
    y
}

/// Build the ONNX ModelProto bytes for the MLP graph.
fn build_mlp_onnx(w: &[f32], b: &[f32]) -> Result<Vec<u8>> {
    let tensor_type = |dims: &[i64]| proto::TypeProto {
        value: Some(proto::type_proto::Value::TensorType(
            proto::type_proto::Tensor {
                elem_type: Some(DataType::Float as i32),
                shape: Some(proto::TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&n| proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(n)),
                            denotation: None,
                        })
                        .collect(),
                }),
            },
        )),
        denotation: None,
    };
    let value_info = |name: &str, dims: &[i64]| proto::ValueInfoProto {
        name: Some(name.to_string()),
        r#type: Some(tensor_type(dims)),
        ..Default::default()
    };
    let initializer = |name: &str, dims: &[i64], data: &[f32]| proto::TensorProto {
        name: Some(name.to_string()),
        data_type: Some(DataType::Float as i32),
        dims: dims.to_vec(),
        float_data: data.to_vec(),
        ..Default::default()
    };
    let node = |op: &str, ins: &[&str], outs: &[&str]| proto::NodeProto {
        input: ins.iter().map(|s| s.to_string()).collect(),
        output: outs.iter().map(|s| s.to_string()).collect(),
        op_type: Some(op.to_string()),
        name: Some(format!("{op}_0")),
        ..Default::default()
    };

    let (m, k, n) = (M as i64, K as i64, N as i64);
    let graph = proto::GraphProto {
        name: Some("mlp".into()),
        node: vec![
            node("MatMul", &["X", "W"], &["T0"]),
            node("Add", &["T0", "B"], &["T1"]),
            node("Relu", &["T1"], &["Y"]),
        ],
        initializer: vec![initializer("W", &[k, n], w), initializer("B", &[m, n], b)],
        input: vec![value_info("X", &[m, k])],
        output: vec![value_info("Y", &[m, n])],
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
