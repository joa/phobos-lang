use super::*;
use crate::ir::{Node, Shape, TensorData, ValueInfo};
use std::collections::HashMap;

fn input(name: &str, dims: &[i64]) -> ValueInfo {
    ValueInfo {
        name: name.into(),
        data_type: None,
        shape: Shape(Some(dims.iter().map(|&n| Dim::Fixed(n)).collect())),
    }
}

fn f32_init(dims: &[i64]) -> std::sync::Arc<crate::ir::Tensor> {
    let n = dims.iter().product::<i64>() as usize;
    std::sync::Arc::new(crate::ir::Tensor {
        data_type: crate::ir::DataType::F32,
        dims: dims.to_vec(),
        data: TensorData::F32(vec![0.0; n]),
    })
}

fn node(op: &str, ins: &[&str], outs: &[&str]) -> Node {
    Node {
        name: format!("{op}_0"),
        op_type: op.into(),
        inputs: ins.iter().map(|s| s.to_string()).collect(),
        outputs: outs.iter().map(|s| s.to_string()).collect(),
        attrs: HashMap::new(),
    }
}

/// Shapes flow input, MatMul, Add, Relu, output.
#[test]
fn infers_mlp_graph() {
    let x = input("X", &[128, 64]);
    let mut graph = Graph {
        name: "mlp".into(),
        inputs: vec![x.clone()],
        outputs: vec![],
        nodes: vec![
            node("MatMul", &["X", "W"], &["T0"]),
            node("Add", &["T0", "B"], &["T1"]),
            node("Relu", &["T1"], &["Y"]),
        ],
        initializers: HashMap::from([
            ("W".to_string(), f32_init(&[64, 128])),
            ("B".to_string(), f32_init(&[128, 128])),
        ]),
        values: HashMap::from([("X".to_string(), x)]),
    };
    let dims = infer(&graph).unwrap();

    assert_eq!(dims["X"], vec![128, 64]);
    assert_eq!(dims["W"], vec![64, 128]);
    assert_eq!(dims["T0"], vec![128, 128]); // MatMul [M,K]x[K,N]
    assert_eq!(dims["T1"], vec![128, 128]); // Add same-shape
    assert_eq!(dims["Y"], vec![128, 128]); // Relu same-shape

    // A dangling reference fails cleanly.
    graph.nodes.push(node("Relu", &["missing"], &["Z"]));
    assert!(infer(&graph).is_err());
}

#[test]
fn matmul_requires_matching_inner_dim() {
    let x = input("X", &[8, 4]);
    let graph = Graph {
        inputs: vec![x.clone()],
        nodes: vec![node("MatMul", &["X", "W"], &["Y"])],
        initializers: HashMap::from([("W".to_string(), f32_init(&[5, 3]))]),
        values: HashMap::from([("X".to_string(), x)]),
        ..Default::default()
    };
    assert!(infer(&graph).is_err());
}

#[test]
fn symbolic_input_dim_is_rejected() {
    let x = ValueInfo {
        name: "X".into(),
        data_type: None,
        shape: Shape(Some(vec![Dim::Symbol("batch".into()), Dim::Fixed(4)])),
    };
    let graph = Graph {
        inputs: vec![x.clone()],
        values: HashMap::from([("X".to_string(), x)]),
        ..Default::default()
    };
    assert!(infer(&graph).is_err());
}

#[test]
fn elementwise_requires_matching_shapes() {
    let a = input("A", &[4, 4]);
    let b = input("B", &[4, 8]);
    let graph = Graph {
        inputs: vec![a.clone(), b.clone()],
        nodes: vec![node("Add", &["A", "B"], &["Y"])],
        values: HashMap::from([("A".to_string(), a), ("B".to_string(), b)]),
        ..Default::default()
    };
    assert!(infer(&graph).is_err());
}

#[test]
fn broadcasts_right_aligned() {
    assert_eq!(broadcast(&[128, 64], &[64]), Some(vec![128, 64])); // bias row
    assert_eq!(broadcast(&[128, 64], &[1, 64]), Some(vec![128, 64]));
    assert_eq!(broadcast(&[2, 1, 4], &[3, 4]), Some(vec![2, 3, 4]));
    assert_eq!(broadcast(&[4, 4], &[4]), Some(vec![4, 4]));
    assert_eq!(broadcast(&[128, 64], &[32]), None); // incompatible
}

#[test]
fn infers_bias_add_and_norm_ops() {
    let x = input("X", &[128, 64]);
    let graph = Graph {
        inputs: vec![x.clone()],
        nodes: vec![
            node("LayerNormalization", &["X", "G", "Bn"], &["L"]),
            node("Add", &["L", "bias"], &["A"]), // bias-row broadcast
            node("Gelu", &["A"], &["Gout"]),
            node("Softmax", &["Gout"], &["Y"]),
        ],
        initializers: HashMap::from([
            ("G".to_string(), f32_init(&[64])),
            ("Bn".to_string(), f32_init(&[64])),
            ("bias".to_string(), f32_init(&[64])),
        ]),
        values: HashMap::from([("X".to_string(), x)]),
        ..Default::default()
    };
    let dims = infer(&graph).unwrap();
    assert_eq!(dims["L"], vec![128, 64]);
    assert_eq!(dims["A"], vec![128, 64]);
    assert_eq!(dims["Gout"], vec![128, 64]);
    assert_eq!(dims["Y"], vec![128, 64]);
}

fn i64_init(dims: &[i64], data: &[i64]) -> std::sync::Arc<crate::ir::Tensor> {
    std::sync::Arc::new(crate::ir::Tensor {
        data_type: crate::ir::DataType::I64,
        dims: dims.to_vec(),
        data: TensorData::I64(data.to_vec()),
    })
}

fn node_attr(op: &str, ins: &[&str], outs: &[&str], attrs: &[(&str, Attribute)]) -> Node {
    let mut n = node(op, ins, outs);
    for (k, v) in attrs {
        n.attrs.insert(k.to_string(), v.clone());
    }
    n
}

#[test]
fn infers_layout_ops() {
    let x = input("X", &[2, 3, 4]);
    let ids = input("ids", &[3]);
    let graph = Graph {
        inputs: vec![x.clone(), ids.clone()],
        nodes: vec![
            node_attr(
                "Transpose",
                &["X"],
                &["Xt"],
                &[("perm", Attribute::Ints(vec![0, 2, 1]))],
            ),
            node("Reshape", &["Xt", "shape"], &["Xr"]),
            node_attr(
                "Gather",
                &["table", "ids"],
                &["emb"],
                &[("axis", Attribute::Int(0))],
            ),
            node_attr(
                "Split",
                &["row"],
                &["a", "b", "c"],
                &[
                    ("axis", Attribute::Int(1)),
                    ("split", Attribute::Ints(vec![2, 2, 2])),
                ],
            ),
            node_attr(
                "Concat",
                &["a", "b"],
                &["ab"],
                &[("axis", Attribute::Int(1))],
            ),
        ],
        initializers: HashMap::from([
            ("shape".to_string(), i64_init(&[2], &[-1, 3])),
            ("table".to_string(), f32_init(&[10, 5])),
            ("row".to_string(), f32_init(&[4, 6])),
        ]),
        values: HashMap::from([("X".to_string(), x), ("ids".to_string(), ids)]),
        ..Default::default()
    };
    let dims = infer(&graph).unwrap();
    assert_eq!(dims["Xt"], vec![2, 4, 3]); // perm [0,2,1]
    assert_eq!(dims["Xr"], vec![8, 3]); // reshape [-1,3] over 24 elems
    assert_eq!(dims["emb"], vec![3, 5]); // gather rows of [10,5]
    assert_eq!(dims["a"], vec![4, 2]); // split [4,6] -> three [4,2]
    assert_eq!(dims["ab"], vec![4, 4]); // concat two [4,2] on axis 1
}

#[test]
fn unmodeled_op_is_rejected() {
    let x = input("X", &[4, 4]);
    let graph = Graph {
        inputs: vec![x.clone()],
        nodes: vec![node("Conv", &["X"], &["Y"])],
        values: HashMap::from([("X".to_string(), x)]),
        ..Default::default()
    };
    assert!(infer(&graph).is_err());
}
