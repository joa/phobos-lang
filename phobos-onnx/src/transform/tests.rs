use super::*;
use crate::ir::{DataType, Tensor, TensorData};

fn node(op: &str, ins: &[&str], outs: &[&str]) -> Node {
    Node {
        name: format!("{op}_0"),
        op_type: op.into(),
        inputs: ins.iter().map(|s| s.to_string()).collect(),
        outputs: outs.iter().map(|s| s.to_string()).collect(),
        attrs: HashMap::new(),
    }
}

fn f32_const(dims: &[i64]) -> Tensor {
    Tensor {
        data_type: DataType::F32,
        dims: dims.to_vec(),
        data: TensorData::F32(vec![0.5; dims.iter().product::<i64>() as usize]),
    }
}

fn graph_of(nodes: Vec<Node>, inits: &[(&str, Tensor)], outputs: &[&str]) -> Graph {
    Graph {
        nodes,
        initializers: inits
            .iter()
            .map(|(n, t)| (n.to_string(), std::sync::Arc::new(t.clone())))
            .collect(),
        outputs: outputs
            .iter()
            .map(|n| crate::ir::ValueInfo {
                name: n.to_string(),
                data_type: None,
                shape: Default::default(),
            })
            .collect(),
        ..Default::default()
    }
}

#[test]
fn fuses_matmul_bias_gelu() {
    let g = graph_of(
        vec![
            node("MatMul", &["X", "W"], &["m"]),
            node("Add", &["m", "bias"], &["ab"]),
            node("Gelu", &["ab"], &["Y"]),
        ],
        &[("bias", f32_const(&[8]))],
        &["Y"],
    );
    let fused = fuse(&g);
    assert_eq!(fused.nodes.len(), 1);
    let n = &fused.nodes[0];
    assert_eq!(n.op_type, FUSED_LINEAR);
    assert_eq!(n.inputs, vec!["X", "W", "bias"]);
    assert_eq!(n.outputs, vec!["Y"]);
    assert!(matches!(n.attrs.get("activation"), Some(Attribute::String(s)) if s == "gelu"));
}

#[test]
fn fuses_matmul_bias_without_activation() {
    let g = graph_of(
        vec![
            node("MatMul", &["X", "W"], &["m"]),
            node("Add", &["m", "bias"], &["Y"]),
        ],
        &[("bias", f32_const(&[1, 8]))],
        &["Y"],
    );
    let fused = fuse(&g);
    assert_eq!(fused.nodes.len(), 1);
    assert_eq!(fused.nodes[0].op_type, FUSED_LINEAR);
    assert!(
        matches!(fused.nodes[0].attrs.get("activation"), Some(Attribute::String(s)) if s == "none")
    );
}

#[test]
fn does_not_fuse_when_matmul_output_is_reused() {
    // `m` feeds both the Add and a separate consumer, so it must stay.
    let g = graph_of(
        vec![
            node("MatMul", &["X", "W"], &["m"]),
            node("Add", &["m", "bias"], &["Y"]),
            node("Relu", &["m"], &["Z"]),
        ],
        &[("bias", f32_const(&[8]))],
        &["Y", "Z"],
    );
    let fused = fuse(&g);
    assert_eq!(fused.nodes.len(), 3);
    assert!(fused.nodes.iter().all(|n| n.op_type != FUSED_LINEAR));
}

#[test]
fn does_not_fuse_non_constant_bias() {
    let g = graph_of(
        vec![
            node("MatMul", &["X", "W"], &["m"]),
            node("Add", &["m", "other"], &["Y"]),
        ],
        &[],
        &["Y"],
    );
    assert_eq!(fuse(&g).nodes.len(), 2);
}

#[test]
fn fuses_attention_block_into_flash() {
    // scores = Q @ transpose(K); scaled = scores * c; p = softmax(scaled);
    // out = p @ V
    let g = graph_of(
        vec![
            node("Transpose", &["K"], &["Kt"]),
            node("MatMul", &["Q", "Kt"], &["scores"]),
            node("Mul", &["scores", "scale"], &["scaled"]),
            node("Softmax", &["scaled"], &["p"]),
            node("MatMul", &["p", "V"], &["O"]),
        ],
        &[("scale", f32_const(&[1]))],
        &["O"],
    );
    let fused = fuse(&g);
    assert_eq!(fused.nodes.len(), 1);
    let n = &fused.nodes[0];
    assert_eq!(n.op_type, FLASH_ATTENTION);
    assert_eq!(n.inputs, vec!["Q", "K", "V"]);
    assert_eq!(n.outputs, vec!["O"]);
    assert!(matches!(n.attrs.get("scale"), Some(Attribute::Float(_))));
}

#[test]
fn fuses_decomposed_layernorm() {
    // The GPT-2 decomposed-LayerNorm chain over `x`.
    let g = graph_of(
        vec![
            node("ReduceMean", &["x"], &["mean"]),
            node("Sub", &["x", "mean"], &["xc"]),
            node("Pow", &["xc", "two"], &["sq"]),
            node("ReduceMean", &["sq"], &["var"]),
            node("Add", &["var", "eps"], &["vare"]),
            node("Sqrt", &["vare"], &["std"]),
            node("Div", &["xc", "std"], &["norm"]),
            node("Mul", &["norm", "scale"], &["scaled"]),
            node("Add", &["scaled", "bias"], &["y"]),
        ],
        &[("eps", f32_const(&[1]))],
        &["y"],
    );
    let fused = fuse_layernorm(&g);
    assert_eq!(fused.nodes.len(), 1);
    let n = &fused.nodes[0];
    assert_eq!(n.op_type, LAYER_NORM);
    assert_eq!(n.inputs, vec!["x", "scale", "bias"]);
    assert_eq!(n.outputs, vec!["y"]);
    assert!(matches!(n.attrs.get("epsilon"), Some(Attribute::Float(_))));
}

#[test]
fn does_not_fuse_masked_attention() {
    // A mask Add between scores and softmax blocks the plain-flash fusion.
    let g = graph_of(
        vec![
            node("Transpose", &["K"], &["Kt"]),
            node("MatMul", &["Q", "Kt"], &["scores"]),
            node("Add", &["scores", "mask"], &["masked"]),
            node("Softmax", &["masked"], &["p"]),
            node("MatMul", &["p", "V"], &["O"]),
        ],
        &[],
        &["O"],
    );
    // The mask is not a scalar constant, so no scale matches, and `masked`
    // is produced by an Add rather than a MatMul.
    let fused = fuse(&g);
    assert!(fused.nodes.iter().all(|n| n.op_type != FLASH_ATTENTION));
}
