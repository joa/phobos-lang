// The layout and index ops interleaved with compute ops on the GPU, checked
// against a CPU reference:
//
//   emb   = Gather(table, ids)        # embedding lookup, int ids
//   ln    = LayerNorm(emb, g, b)
//   h0,h1 = Split(ln, axis=1)
//   sw    = Concat(h1, h0, axis=1)    # swap the halves
//   t     = Transpose(sw)             # [S,W] to [W,S]
//   r     = Reshape(t, [2, 8])
//   Y     = Softmax(r)
//
//   cargo run -p phobos-onnx --example run_layout --features cuda

#![allow(clippy::useless_vec)] // const-size reference buffers read as slices

use std::collections::HashMap;

use anyhow::{Result, bail};
use phobos_onnx::proto::{self, tensor_proto::DataType};
use phobos_onnx::{load_model, runner};
use prost::Message;

const VOCAB: usize = 8;
const S: usize = 4; // sequence length
const W: usize = 4; // width
const EPS: f32 = 1e-5;

fn main() -> Result<()> {
    let table: Vec<f32> = (0..VOCAB * W).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let g: Vec<f32> = (0..W).map(|i| 1.0 + i as f32 * 0.1).collect();
    let bn: Vec<f32> = (0..W).map(|i| i as f32 * 0.05).collect();
    let ids: Vec<i64> = vec![1, 3, 5, 7];

    let model = load_model(&build_graph(&table, &g, &bn)?)?;

    let int_inputs = HashMap::from([("ids".to_string(), ids.clone())]);
    let outputs = runner::run_typed(&model.graph, &HashMap::new(), &int_inputs)?;
    let got = outputs.get("Y").expect("output Y");

    let want = reference(&table, &g, &bn, &ids);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("ran Gather -> LayerNorm -> Split -> Concat -> Transpose -> Reshape -> Softmax");
    println!(
        "{} outputs, max abs error vs CPU reference: {max_err:e}",
        got.len()
    );
    if max_err > 2e-3 {
        bail!("result mismatch: max abs error {max_err:e}");
    }
    println!("OK: GPU result matches the CPU reference");
    Ok(())
}

// ---- CPU reference -------------------------------------------------------

fn reference(table: &[f32], g: &[f32], bn: &[f32], ids: &[i64]) -> Vec<f32> {
    // Gather.
    let mut emb = vec![0.0f32; S * W];
    for (r, &id) in ids.iter().enumerate() {
        emb[r * W..r * W + W].copy_from_slice(&table[id as usize * W..id as usize * W + W]);
    }
    // LayerNorm.
    let mut ln = vec![0.0f32; S * W];
    for r in 0..S {
        let row = &emb[r * W..r * W + W];
        let mean = row.iter().sum::<f32>() / W as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / W as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for c in 0..W {
            ln[r * W + c] = (row[c] - mean) * inv * g[c] + bn[c];
        }
    }
    // Split into halves and concat swapped: [h0 | h1] -> [h1 | h0].
    let half = W / 2;
    let mut sw = vec![0.0f32; S * W];
    for r in 0..S {
        for c in 0..half {
            sw[r * W + c] = ln[r * W + half + c];
            sw[r * W + half + c] = ln[r * W + c];
        }
    }
    // Transpose [S,W] -> [W,S].
    let mut t = vec![0.0f32; W * S];
    for r in 0..S {
        for c in 0..W {
            t[c * S + r] = sw[r * W + c];
        }
    }
    // Reshape [W,S]=[4,4] -> [2,8] is a no-op on data; Softmax over rows of [2,8].
    let mut y = vec![0.0f32; t.len()];
    let cols = 8;
    for r in 0..2 {
        let row = &t[r * cols..r * cols + cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for c in 0..cols {
            y[r * cols + c] = exps[c] / sum;
        }
    }
    y
}

// ---- ONNX graph construction ---------------------------------------------

fn build_graph(table: &[f32], g: &[f32], bn: &[f32]) -> Result<Vec<u8>> {
    let graph = proto::GraphProto {
        name: Some("layout".into()),
        node: vec![
            node_attr("Gather", &["table", "ids"], &["emb"], &[("axis", 0)]),
            node("LayerNormalization", &["emb", "g", "bn"], &["ln"]),
            split_node("ln", &["h0", "h1"], 1, &[2, 2]),
            node_attr("Concat", &["h1", "h0"], &["sw"], &[("axis", 1)]),
            transpose_node("sw", "t", &[1, 0]),
            node("Reshape", &["t", "newshape"], &["r"]),
            node("Softmax", &["r"], &["Y"]),
        ],
        initializer: vec![
            initializer_f32("table", &[VOCAB as i64, W as i64], table),
            initializer_f32("g", &[W as i64], g),
            initializer_f32("bn", &[W as i64], bn),
            initializer_i64("newshape", &[2], &[2, 8]),
        ],
        input: vec![int_value_info("ids", &[S as i64])],
        output: vec![value_info("Y", &[2, 8])],
        ..Default::default()
    };
    let model = proto::ModelProto {
        ir_version: Some(9),
        opset_import: vec![proto::OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(21),
        }],
        graph: Some(graph),
        ..Default::default()
    };
    Ok(model.encode_to_vec())
}

fn tensor_type(elem: i32, dims: &[i64]) -> proto::TypeProto {
    proto::TypeProto {
        value: Some(proto::type_proto::Value::TensorType(
            proto::type_proto::Tensor {
                elem_type: Some(elem),
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
    }
}

fn value_info(name: &str, dims: &[i64]) -> proto::ValueInfoProto {
    proto::ValueInfoProto {
        name: Some(name.to_string()),
        r#type: Some(tensor_type(DataType::Float as i32, dims)),
        ..Default::default()
    }
}

fn int_value_info(name: &str, dims: &[i64]) -> proto::ValueInfoProto {
    proto::ValueInfoProto {
        name: Some(name.to_string()),
        r#type: Some(tensor_type(DataType::Int64 as i32, dims)),
        ..Default::default()
    }
}

fn initializer_f32(name: &str, dims: &[i64], data: &[f32]) -> proto::TensorProto {
    proto::TensorProto {
        name: Some(name.to_string()),
        data_type: Some(DataType::Float as i32),
        dims: dims.to_vec(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}

fn initializer_i64(name: &str, dims: &[i64], data: &[i64]) -> proto::TensorProto {
    proto::TensorProto {
        name: Some(name.to_string()),
        data_type: Some(DataType::Int64 as i32),
        dims: dims.to_vec(),
        int64_data: data.to_vec(),
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

fn node_attr(op: &str, ins: &[&str], outs: &[&str], attrs: &[(&str, i64)]) -> proto::NodeProto {
    let mut n = node(op, ins, outs);
    n.attribute = attrs
        .iter()
        .map(|(name, val)| int_attribute(name, *val))
        .collect();
    n
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

fn split_node(input: &str, outs: &[&str], axis: i64, sizes: &[i64]) -> proto::NodeProto {
    let mut n = node("Split", &[input], outs);
    n.attribute = vec![
        int_attribute("axis", axis),
        proto::AttributeProto {
            name: Some("split".into()),
            r#type: Some(proto::attribute_proto::AttributeType::Ints as i32),
            ints: sizes.to_vec(),
            ..Default::default()
        },
    ];
    n
}

fn int_attribute(name: &str, val: i64) -> proto::AttributeProto {
    proto::AttributeProto {
        name: Some(name.to_string()),
        r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
        i: Some(val),
        ..Default::default()
    }
}
