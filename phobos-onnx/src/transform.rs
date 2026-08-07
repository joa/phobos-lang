use std::collections::{HashMap, HashSet};

use crate::ir::{Attribute, Graph, Node};

/// The fused matmul-with-epilogue node.
pub const FUSED_LINEAR: &str = "PhobosFusedLinear";
/// The fused flash-attention node.
pub const FLASH_ATTENTION: &str = "PhobosFlashAttention";
/// Emitted when the decomposed LayerNorm chain is recognized.
pub const LAYER_NORM: &str = "LayerNormalization";

/// Rewrite groups of nodes into single fused nodes the runner lowers to one
/// kernel each, cutting launches and intermediate buffers. The matches are
/// structural, on op types, single-use edges and initializer ranks, so this
/// runs on the raw graph before shape inference. Patterns already fused are
/// left alone.
pub fn fuse(graph: &Graph) -> Graph {
    let mut g = graph.clone();
    g.nodes = fuse_linear(&g);
    g.nodes = fuse_attention(&g);
    g
}

/// How many nodes and graph outputs consume each edge. An edge with a single
/// consumer and no external use is safe to fuse across.
fn consumer_count(graph: &Graph) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for node in &graph.nodes {
        for input in &node.inputs {
            if !input.is_empty() {
                *counts.entry(input.clone()).or_default() += 1;
            }
        }
    }
    for vi in &graph.outputs {
        *counts.entry(vi.name.clone()).or_default() += 1;
    }
    counts
}

/// The single node index that consumes `edge`, if it has exactly one consumer
/// and is not a graph output.
fn sole_consumer(edge: &str, nodes: &[Node], counts: &HashMap<String, usize>) -> Option<usize> {
    if counts.get(edge).copied().unwrap_or(0) != 1 {
        return None;
    }
    nodes
        .iter()
        .position(|n| n.inputs.iter().any(|i| i == edge))
}

/// The node indices a fusion absorbs and the node replacing them.
type Fusion = (Vec<usize>, Node);

/// Rebuild the node list, each fusion's members replaced by its fused node at
/// the group's earliest index. The fused inputs all come from outside the
/// group, so they still precede it.
fn rebuild(nodes: &[Node], fusions: Vec<Fusion>) -> Vec<Node> {
    let mut place: HashMap<usize, Node> = HashMap::new();
    let mut skip: HashSet<usize> = HashSet::new();
    for (members, node) in fusions {
        let anchor = *members.iter().min().expect("non-empty group");
        for m in members {
            if m != anchor {
                skip.insert(m);
            }
        }
        place.insert(anchor, node);
    }
    let mut out = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        if let Some(fused) = place.remove(&i) {
            out.push(fused);
        } else if !skip.contains(&i) {
            out.push(node.clone());
        }
    }
    out
}

/// Greedily collect non-overlapping fusions produced by `matcher` at each node
/// of `op`, then rebuild the graph.
fn apply(
    graph: &Graph,
    op: &str,
    matcher: impl Fn(usize, &Graph, &HashMap<String, usize>) -> Option<Fusion>,
) -> Vec<Node> {
    let nodes = &graph.nodes;
    let counts = consumer_count(graph);
    let mut used: HashSet<usize> = HashSet::new();
    let mut fusions: Vec<Fusion> = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        if node.op_type != op {
            continue;
        }
        if let Some((members, fused)) = matcher(i, graph, &counts)
            && !members.iter().any(|m| used.contains(m))
        {
            used.extend(members.iter().copied());
            fusions.push((members, fused));
        }
    }
    rebuild(nodes, fusions)
}

fn fuse_linear(graph: &Graph) -> Vec<Node> {
    apply(graph, "MatMul", try_linear)
}

/// Fuse the `MatMul` at `mi` with a following bias `Add` and an optional `Relu`
/// or `Gelu`.
fn try_linear(mi: usize, graph: &Graph, counts: &HashMap<String, usize>) -> Option<Fusion> {
    let nodes = &graph.nodes;
    let mm = &nodes[mi];
    let mm_out = mm.outputs.first()?;

    // MatMul output must feed exactly one Add.
    let ai = sole_consumer(mm_out, nodes, counts)?;
    let add = &nodes[ai];
    if add.op_type != "Add" {
        return None;
    }
    // The other Add operand must be a constant bias row, [N] or [1, N], against
    // the matmul's [M, N].
    let bias = add.inputs.iter().find(|e| *e != mm_out)?;
    if !is_row_constant(graph, bias) {
        return None;
    }

    let mut members = vec![mi, ai];
    let mut activation = "none";
    let mut final_out = add.outputs.first()?.clone();

    if let Some(gi) = sole_consumer(&final_out, nodes, counts) {
        let act = &nodes[gi];
        if let Some(kind) = activation_kind(&act.op_type) {
            activation = kind;
            final_out = act.outputs.first()?.clone();
            members.push(gi);
        }
    }

    let fused = Node {
        name: format!("{}_fused", mm.name),
        op_type: FUSED_LINEAR.into(),
        inputs: vec![mm.inputs[0].clone(), mm.inputs[1].clone(), bias.clone()],
        outputs: vec![final_out],
        attrs: HashMap::from([(
            "activation".to_string(),
            Attribute::String(activation.into()),
        )]),
    };
    Some((members, fused))
}

/// Whether `op` is a supported epilogue activation.
fn activation_kind(op: &str) -> Option<&'static str> {
    match op {
        "Relu" => Some("relu"),
        "Gelu" => Some("gelu"),
        _ => None,
    }
}

/// Whether `edge` is an initializer shaped `[N]` or `[1, N]`.
fn is_row_constant(graph: &Graph, edge: &str) -> bool {
    match graph.initializers.get(edge) {
        Some(t) => matches!(t.dims.as_slice(), [_] | [1, _]),
        None => false,
    }
}

fn fuse_attention(graph: &Graph) -> Vec<Node> {
    apply(graph, "Softmax", try_attention)
}

/// Collapse the decomposed-LayerNorm chain into one `LayerNormalization` node.
/// Runs on the folded, static residual graph.
pub fn fuse_layernorm(graph: &Graph) -> Graph {
    let mut g = graph.clone();
    g.nodes = apply(&g, "ReduceMean", try_layernorm);
    g
}

/// Anchored at the mean `ReduceMean`:
///
///   mean = ReduceMean(x); xc = Sub(x, mean); sq = Pow(xc, 2);
///   var = ReduceMean(sq); std = Sqrt(Add(var, eps)); norm = Div(xc, std);
///   out = Add(Mul(norm, scale), bias)
///
/// becomes `LayerNormalization(x, scale, bias)` with the recovered epsilon.
fn try_layernorm(mi: usize, graph: &Graph, counts: &HashMap<String, usize>) -> Option<Fusion> {
    let nodes = &graph.nodes;
    let mean = &nodes[mi];
    let x = mean.inputs.first()?.clone();
    let mean_out = mean.outputs.first()?;

    // mean feeds only Sub(x, mean).
    let sub_i = sole_consumer(mean_out, nodes, counts)?;
    let sub = &nodes[sub_i];
    if sub.op_type != "Sub" || sub.inputs.first()? != &x || sub.inputs.get(1)? != mean_out {
        return None;
    }
    let xc = sub.outputs.first()?.clone();
    // xc is used exactly twice, by Pow and by Div.
    if counts.get(&xc).copied().unwrap_or(0) != 2 {
        return None;
    }
    let pow_i = consumer_with_op(&xc, nodes, "Pow")?;
    let sq = nodes[pow_i].outputs.first()?.clone();

    let rm2_i = sole_consumer(&sq, nodes, counts)?;
    if nodes[rm2_i].op_type != "ReduceMean" {
        return None;
    }
    let var = nodes[rm2_i].outputs.first()?.clone();

    let add_i = sole_consumer(&var, nodes, counts)?;
    let add = &nodes[add_i];
    if add.op_type != "Add" {
        return None;
    }
    let eps_edge = add.inputs.iter().find(|e| **e != var)?;
    let eps = scalar_constant(graph, eps_edge).unwrap_or(1e-5);
    let stdpre = add.outputs.first()?.clone();

    let sqrt_i = sole_consumer(&stdpre, nodes, counts)?;
    if nodes[sqrt_i].op_type != "Sqrt" {
        return None;
    }
    let std = nodes[sqrt_i].outputs.first()?.clone();

    let div_i = consumer_with_op(&std, nodes, "Div")?;
    let div = &nodes[div_i];
    if div.inputs.first()? != &xc || div.inputs.get(1)? != &std {
        return None;
    }
    let norm = div.outputs.first()?.clone();

    let mul_i = sole_consumer(&norm, nodes, counts)?;
    let mul = &nodes[mul_i];
    if mul.op_type != "Mul" {
        return None;
    }
    let scale = mul.inputs.iter().find(|e| **e != norm)?.clone();
    let scaled = mul.outputs.first()?.clone();

    let addb_i = sole_consumer(&scaled, nodes, counts)?;
    let addb = &nodes[addb_i];
    if addb.op_type != "Add" {
        return None;
    }
    let bias = addb.inputs.iter().find(|e| **e != scaled)?.clone();
    let out = addb.outputs.first()?.clone();

    let members = vec![mi, sub_i, pow_i, rm2_i, add_i, sqrt_i, div_i, mul_i, addb_i];
    let fused = Node {
        name: format!("{}_ln", mean.name),
        op_type: LAYER_NORM.into(),
        inputs: vec![x, scale, bias],
        outputs: vec![out],
        attrs: HashMap::from([("epsilon".to_string(), Attribute::Float(eps))]),
    };
    Some((members, fused))
}

/// A consumer of `edge` whose op type is `op`; the edge may have several.
fn consumer_with_op(edge: &str, nodes: &[Node], op: &str) -> Option<usize> {
    nodes
        .iter()
        .position(|n| n.op_type == op && n.inputs.iter().any(|i| i == edge))
}

/// The attention block anchored at the `Softmax` at `si`:
///
///   K^T = Transpose(K); scores = MatMul(Q, K^T); [scaled = Mul/Div(scores, c);]
///   probs = Softmax(scaled); out = MatMul(probs, V)
///
/// becomes one flash-attention node. Fires only when the softmax input is
/// unmasked, no intervening Add, so the plain flash kernel is exact.
fn try_attention(si: usize, graph: &Graph, counts: &HashMap<String, usize>) -> Option<Fusion> {
    let nodes = &graph.nodes;
    let softmax = &nodes[si];
    let sm_in = softmax.inputs.first()?;
    let sm_out = softmax.outputs.first()?;

    // The softmax input's producer: an optional scale wrapping the Q@K^T.
    let (scores_edge, scale) = unwrap_scale(graph, sm_in)?;

    // K^T must itself be a Transpose of K swapping the last two axes.
    let mm1 = producer(nodes, &scores_edge)?;
    if nodes[mm1].op_type != "MatMul" {
        return None;
    }
    let q = nodes[mm1].inputs.first()?.clone();
    let kt = nodes[mm1].inputs.get(1)?.clone();
    let (ti, k) = transpose_of(nodes, &kt)?;

    // Softmax output must feed exactly one MatMul(probs, V).
    let mv = sole_consumer(sm_out, nodes, counts)?;
    if nodes[mv].op_type != "MatMul" || nodes[mv].inputs.first()? != sm_out {
        return None;
    }
    let v = nodes[mv].inputs.get(1)?.clone();
    let out = nodes[mv].outputs.first()?.clone();

    // The intermediate scores and probs must not be used elsewhere.
    if counts.get(&scores_edge).copied().unwrap_or(0) != 1
        || counts.get(sm_out).copied().unwrap_or(0) != 1
    {
        return None;
    }

    // The transpose, both matmuls, the optional scale, and the softmax.
    let mut members = vec![ti, mm1, si, mv];
    if let Some(scale_idx) = scale.node {
        members.push(scale_idx);
    }

    let fused = Node {
        name: format!("{}_flash", softmax.name),
        op_type: FLASH_ATTENTION.into(),
        inputs: vec![q, k, v],
        outputs: vec![out],
        attrs: HashMap::from([("scale".to_string(), Attribute::Float(scale.value))]),
    };
    Some((members, fused))
}

/// A multiplicative factor and the node that applied it, if any.
struct Scale {
    value: f32,
    node: Option<usize>,
}

/// The pre-scale edge and factor when `edge` comes from `Mul(x, c)` or
/// `Div(x, c)` with `c` a scalar constant. Otherwise the edge is its own
/// producer at unit scale.
fn unwrap_scale(graph: &Graph, edge: &str) -> Option<(String, Scale)> {
    let nodes = &graph.nodes;
    let Some(pi) = producer(nodes, edge) else {
        return Some((
            edge.to_string(),
            Scale {
                value: 1.0,
                node: None,
            },
        ));
    };
    let n = &nodes[pi];
    let (op, x, c) = match n.op_type.as_str() {
        "Mul" | "Div" => (n.op_type.as_str(), &n.inputs[0], &n.inputs[1]),
        _ => {
            return Some((
                edge.to_string(),
                Scale {
                    value: 1.0,
                    node: None,
                },
            ));
        }
    };
    let Some(scalar) = scalar_constant(graph, c) else {
        return Some((
            edge.to_string(),
            Scale {
                value: 1.0,
                node: None,
            },
        ));
    };
    let value = if op == "Div" { 1.0 / scalar } else { scalar };
    Some((
        x.clone(),
        Scale {
            value,
            node: Some(pi),
        },
    ))
}

/// The node index producing `edge`, if any.
fn producer(nodes: &[Node], edge: &str) -> Option<usize> {
    nodes
        .iter()
        .position(|n| n.outputs.iter().any(|o| o == edge))
}

/// The transpose node index and `x`, when `edge` is `Transpose(x)`.
fn transpose_of(nodes: &[Node], edge: &str) -> Option<(usize, String)> {
    let ti = producer(nodes, edge)?;
    let t = &nodes[ti];
    if t.op_type != "Transpose" {
        return None;
    }
    Some((ti, t.inputs.first()?.clone()))
}

/// The value of `edge` when it is a single-element f32 initializer.
fn scalar_constant(graph: &Graph, edge: &str) -> Option<f32> {
    let t = graph.initializers.get(edge)?;
    match &t.data {
        crate::ir::TensorData::F32(v) if v.len() == 1 => Some(v[0]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
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
}
