use super::*;
use crate::ir::{DataType, Tensor, ValueInfo};

fn node(op: &str, ins: &[&str], outs: &[&str], attrs: &[(&str, Attribute)]) -> Node {
    Node {
        name: format!("{op}_0"),
        op_type: op.into(),
        inputs: ins.iter().map(|s| s.to_string()).collect(),
        outputs: outs.iter().map(|s| s.to_string()).collect(),
        attrs: attrs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    }
}

fn i64_init(name: &str, dims: &[i64], data: &[i64]) -> (String, Arc<Tensor>) {
    (
        name.to_string(),
        Arc::new(Tensor {
            data_type: DataType::I64,
            dims: dims.to_vec(),
            data: TensorData::I64(data.to_vec()),
        }),
    )
}

#[test]
fn folds_shape_gather_concat_reshape_chain() {
    // The GPT-2 preamble: derive dims from the input shape, build a new
    // shape vector, and reshape a shape-only activation.
    let graph = Graph {
        inputs: vec![ValueInfo {
            name: "x".into(),
            data_type: None,
            shape: Default::default(),
        }],
        nodes: vec![
            node("Shape", &["x"], &["s"], &[]),
            node(
                "Gather",
                &["s", "zero"],
                &["d0"],
                &[("axis", Attribute::Int(0))],
            ),
            node(
                "Unsqueeze",
                &["d0"],
                &["d0u"],
                &[("axes", Attribute::Ints(vec![0]))],
            ),
            node(
                "Concat",
                &["d0u", "neg1"],
                &["newshape"],
                &[("axis", Attribute::Int(0))],
            ),
            node("Reshape", &["x", "newshape"], &["y"], &[]),
        ],
        initializers: [i64_init("zero", &[1], &[0]), i64_init("neg1", &[1], &[-1])]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    let inputs = HashMap::from([("x".to_string(), vec![4, 6])]);
    let ev = evaluate(&graph, &inputs);
    assert!(
        ev.unsupported.is_empty(),
        "unsupported: {:?}",
        ev.unsupported
    );

    // Shape(x) folds to [4, 6], Gather axis 0 to 4, Concat to [4, -1].
    assert_eq!(ev.vals["s"].data, Some(Const::I64(vec![4, 6])));
    assert_eq!(ev.vals["d0"].data, Some(Const::I64(vec![4])));
    assert_eq!(ev.vals["newshape"].data, Some(Const::I64(vec![4, -1])));
    // Reshape [4, 6] by [4, -1] is [4, 6] again, 24 elements.
    assert_eq!(ev.vals["y"].dims, vec![4, 6]);
}

#[test]
fn folds_range_and_arithmetic() {
    let graph = Graph {
        nodes: vec![
            node("Range", &["start", "limit", "delta"], &["r"], &[]),
            node("Add", &["r", "r"], &["r2"], &[]),
        ],
        initializers: [
            i64_init("start", &[], &[0]),
            i64_init("limit", &[], &[5]),
            i64_init("delta", &[], &[1]),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    let ev = evaluate(&graph, &HashMap::new());
    assert_eq!(ev.vals["r"].data, Some(Const::I64(vec![0, 1, 2, 3, 4])));
    assert_eq!(ev.vals["r2"].data, Some(Const::I64(vec![0, 2, 4, 6, 8])));
}

#[test]
fn folds_nonzero_arange_trick() {
    // ConstantOfShape([4]) of ones, then NonZero, is [1, 4] of 0..4.
    let graph = Graph {
        nodes: vec![
            node(
                "ConstantOfShape",
                &["shape"],
                &["ones"],
                &[(
                    "value",
                    Attribute::Tensor(Tensor {
                        data_type: DataType::I64,
                        dims: vec![1],
                        data: TensorData::I64(vec![1]),
                    }),
                )],
            ),
            node("NonZero", &["ones"], &["nz"], &[]),
        ],
        initializers: [i64_init("shape", &[1], &[4])].into_iter().collect(),
        ..Default::default()
    };
    let ev = evaluate(&graph, &HashMap::new());
    assert_eq!(ev.vals["ones"].data, Some(Const::I64(vec![1, 1, 1, 1])));
    assert_eq!(ev.vals["nz"].dims, vec![1, 4]);
    assert_eq!(ev.vals["nz"].data, Some(Const::I64(vec![0, 1, 2, 3])));
}

#[test]
fn folds_slice_of_constant_mask() {
    // Slice a [1, 4] constant to its first 2 columns.
    let graph = Graph {
        nodes: vec![node(
            "Slice",
            &["m", "starts", "ends", "axes", "steps"],
            &["s"],
            &[],
        )],
        initializers: [
            i64_init("m", &[1, 4], &[10, 11, 12, 13]),
            i64_init("starts", &[1], &[0]),
            i64_init("ends", &[1], &[2]),
            i64_init("axes", &[1], &[1]),
            i64_init("steps", &[1], &[1]),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    let ev = evaluate(&graph, &HashMap::new());
    assert_eq!(ev.vals["s"].dims, vec![1, 2]);
    assert_eq!(ev.vals["s"].data, Some(Const::I64(vec![10, 11])));
}

#[test]
fn fold_graph_drops_plumbing_and_promotes_constants() {
    // Shape, Gather and Concat build a reshape target for a runtime input,
    // so the residual graph keeps only the Reshape and a constant shape.
    let graph = Graph {
        inputs: vec![ValueInfo {
            name: "x".into(),
            data_type: None,
            shape: Default::default(),
        }],
        nodes: vec![
            node("Shape", &["x"], &["s"], &[]),
            node(
                "Gather",
                &["s", "zero"],
                &["d0"],
                &[("axis", Attribute::Int(0))],
            ),
            node(
                "Unsqueeze",
                &["d0"],
                &["d0u"],
                &[("axes", Attribute::Ints(vec![0]))],
            ),
            node(
                "Concat",
                &["d0u", "neg1"],
                &["newshape"],
                &[("axis", Attribute::Int(0))],
            ),
            node("Reshape", &["x", "newshape"], &["y"], &[]),
        ],
        initializers: [i64_init("zero", &[1], &[0]), i64_init("neg1", &[1], &[-1])]
            .into_iter()
            .collect(),
        outputs: vec![ValueInfo {
            name: "y".into(),
            data_type: None,
            shape: Default::default(),
        }],
        ..Default::default()
    };
    let inputs = HashMap::from([("x".to_string(), vec![4, 6])]);
    let residual = fold_graph(&graph, &inputs).unwrap();

    // Only the Reshape survives, its shape input now an initializer.
    assert_eq!(residual.nodes.len(), 1);
    assert_eq!(residual.nodes[0].op_type, "Reshape");
    assert_eq!(
        residual.initializers["newshape"].data,
        TensorData::I64(vec![4, -1])
    );
    // x is still a runtime input, at its concrete shape.
    assert_eq!(residual.inputs[0].name, "x");
    assert_eq!(residual.values["y"].shape, static_shape(&[4, 6]));
}

#[test]
fn propagates_compute_shapes_without_data() {
    let graph = Graph {
        inputs: vec![ValueInfo {
            name: "h".into(),
            data_type: None,
            shape: Default::default(),
        }],
        nodes: vec![
            node(
                "ReduceMean",
                &["h"],
                &["m"],
                &[
                    ("axes", Attribute::Ints(vec![-1])),
                    ("keepdims", Attribute::Int(1)),
                ],
            ),
            node(
                "Gemm",
                &["h", "w"],
                &["g"],
                &[("transB", Attribute::Int(1))],
            ),
        ],
        initializers: [(
            "w".to_string(),
            Arc::new(Tensor {
                data_type: DataType::F32,
                dims: vec![16, 8],
                data: TensorData::F32(vec![0.0; 128]),
            }),
        )]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    let inputs = HashMap::from([("h".to_string(), vec![4, 8])]);
    let ev = evaluate(&graph, &inputs);
    assert_eq!(ev.vals["m"].dims, vec![4, 1]); // reduced last axis, keepdims
    assert_eq!(ev.vals["g"].dims, vec![4, 16]); // Gemm with transB
    assert!(ev.vals["m"].data.is_none()); // f32 activation, not folded
}
