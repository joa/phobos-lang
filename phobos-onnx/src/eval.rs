use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, bail};

use crate::ir::{Attribute, DataType, Dim, Graph, Node, Shape, Tensor, TensorData, ValueInfo};
use crate::shape::{self, Dims};

#[derive(Clone, Debug, PartialEq)]
pub enum Const {
    I64(Vec<i64>),
    F32(Vec<f32>),
}

impl Const {
    fn len(&self) -> usize {
        match self {
            Const::I64(v) => v.len(),
            Const::F32(v) => v.len(),
        }
    }

    /// As i64 for shape and index use, casting f32.
    fn as_i64(&self) -> Vec<i64> {
        match self {
            Const::I64(v) => v.clone(),
            Const::F32(v) => v.iter().map(|&x| x as i64).collect(),
        }
    }
}

/// What is known about an edge: its static shape, plus its data if it folded to
/// a constant.
#[derive(Clone, Debug)]
pub struct Val {
    pub dims: Dims,
    pub data: Option<Const>,
}

impl Val {
    fn shape_only(dims: Dims) -> Val {
        Val { dims, data: None }
    }
}

pub struct Eval {
    /// Every edge that resolved.
    pub vals: HashMap<String, Val>,
    /// Nodes that could not be evaluated, by op type.
    pub unsupported: BTreeMap<String, usize>,
    /// Nodes whose output shape stayed unknown, by op type.
    pub shape_gaps: BTreeMap<String, usize>,
}

/// Combined static-shape inference and constant folding.
///
/// The exported GPT-2 graphs are fully dynamic: most nodes compute shapes at
/// runtime, position ids by the ConstantOfShape and NonZero arange trick. Fix a
/// concrete input shape and all of that folds to constants, leaving the compute
/// graph static. This propagates a shape and an optional constant value per
/// edge, evaluating the shape and index ops on the host.
///
/// Deliberately tolerant: an unhandled op or a missing input leaves that edge
/// unknown and is tallied in [`Eval::unsupported`], so one run over a real
/// model reports the whole coverage gap rather than stopping at the first hole.
pub fn evaluate(graph: &Graph, input_dims: &HashMap<String, Dims>) -> Eval {
    let mut vals: HashMap<String, Val> = HashMap::new();

    // Shape always; data only for the integer tensors that can fold.
    for (name, t) in &graph.initializers {
        let data = match &t.data {
            TensorData::I64(v) => Some(Const::I64(v.clone())),
            TensorData::I32(v) => Some(Const::I64(v.iter().map(|&x| x as i64).collect())),
            TensorData::F32(v) if v.len() <= FOLD_F32_LIMIT => Some(Const::F32(v.clone())),
            _ => None,
        };
        vals.insert(
            name.clone(),
            Val {
                dims: t.dims.clone(),
                data,
            },
        );
    }
    // Graph inputs are shape only: their data is a runtime value.
    for vi in &graph.inputs {
        if let Some(dims) = input_dims.get(&vi.name) {
            vals.entry(vi.name.clone())
                .or_insert_with(|| Val::shape_only(dims.clone()));
        }
    }

    let mut unsupported = BTreeMap::new();
    let mut shape_gaps = BTreeMap::new();

    for node in &graph.nodes {
        match eval_node(node, &vals) {
            Some(outs) => {
                for (name, val) in node.outputs.iter().zip(outs) {
                    if !name.is_empty() {
                        vals.insert(name.clone(), val);
                    }
                }
            }
            None => {
                *unsupported.entry(node.op_type.clone()).or_default() += 1;
                // Note whether at least the shape was derivable elsewhere.
                if node
                    .outputs
                    .iter()
                    .any(|o| !o.is_empty() && !vals.contains_key(o))
                {
                    *shape_gaps.entry(node.op_type.clone()).or_default() += 1;
                }
            }
        }
    }

    Eval {
        vals,
        unsupported,
        shape_gaps,
    }
}

/// Rewrite `graph` into the residual static graph for a fixed input shape:
/// drop every node whose outputs are all constant, promote each folded constant
/// a surviving node consumes into an initializer, keep the real weights, and
/// record static shapes in `values`. What is left is plain compute with no
/// runtime shape plumbing.
pub fn fold_graph(graph: &Graph, input_dims: &HashMap<String, Dims>) -> Result<Graph> {
    let ev = evaluate(graph, input_dims);

    let is_const = |edge: &str| ev.vals.get(edge).map(|v| v.data.is_some()).unwrap_or(false);

    // A node survives if any output is a runtime value.
    let residual: Vec<Node> = graph
        .nodes
        .iter()
        .filter(|n| !n.outputs.iter().all(|o| o.is_empty() || is_const(o)))
        .cloned()
        .collect();

    let graph_inputs: HashSet<&str> = graph.inputs.iter().map(|v| v.name.as_str()).collect();
    let residual_outputs: HashSet<&str> = residual
        .iter()
        .flat_map(|n| n.outputs.iter())
        .map(String::as_str)
        .collect();

    // The weights residual nodes use, plus any folded constant a residual node
    // reads, promoted out of the dropped node that produced it.
    let mut initializers: HashMap<String, Arc<Tensor>> = HashMap::new();
    for node in &residual {
        for edge in node.inputs.iter().filter(|e| !e.is_empty()) {
            if initializers.contains_key(edge)
                || residual_outputs.contains(edge.as_str())
                || graph_inputs.contains(edge.as_str())
            {
                continue;
            }
            if let Some(t) = graph.initializers.get(edge) {
                initializers.insert(edge.clone(), t.clone());
            } else if let Some(v) = ev.vals.get(edge).filter(|v| v.data.is_some()) {
                initializers.insert(edge.clone(), Arc::new(const_tensor(v)));
            } else {
                bail!(
                    "residual input '{edge}' is neither a graph input, initializer, constant, nor produced by a surviving node"
                );
            }
        }
    }

    // Static shapes for every residual edge.
    let mut values = HashMap::new();
    let mut record = |name: &str| {
        if let Some(v) = ev.vals.get(name) {
            values.insert(
                name.to_string(),
                ValueInfo {
                    name: name.to_string(),
                    data_type: None,
                    shape: static_shape(&v.dims),
                },
            );
        }
    };
    for node in &residual {
        node.inputs.iter().for_each(|e| record(e));
        node.outputs.iter().for_each(|e| record(e));
    }

    // Graph inputs take their concrete shape; outputs are carried over.
    let inputs = graph
        .inputs
        .iter()
        .map(|vi| ValueInfo {
            name: vi.name.clone(),
            data_type: vi.data_type,
            shape: input_dims
                .get(&vi.name)
                .map(|d| static_shape(d))
                .unwrap_or_else(|| vi.shape.clone()),
        })
        .collect();

    Ok(Graph {
        name: graph.name.clone(),
        inputs,
        outputs: graph.outputs.clone(),
        nodes: residual,
        initializers,
        values,
    })
}

fn static_shape(dims: &[i64]) -> Shape {
    Shape(Some(dims.iter().map(|&d| Dim::Fixed(d)).collect()))
}

fn const_tensor(v: &Val) -> Tensor {
    match &v.data {
        Some(Const::I64(d)) => Tensor {
            data_type: DataType::I64,
            dims: v.dims.clone(),
            data: TensorData::I64(d.clone()),
        },
        Some(Const::F32(d)) => Tensor {
            data_type: DataType::F32,
            dims: v.dims.clone(),
            data: TensorData::F32(d.clone()),
        },
        None => unreachable!("const_tensor called on a non-constant value"),
    }
}

/// Larger constant f32 tensors are treated as runtime data. Sized to take in
/// the 1024x1024 causal-mask bias, so attention-mask slices fold, but to leave
/// out the weight and embedding tables.
const FOLD_F32_LIMIT: usize = 1 << 21;

fn eval_node(node: &Node, vals: &HashMap<String, Val>) -> Option<Vec<Val>> {
    let ins: Vec<&Val> = node
        .inputs
        .iter()
        .filter(|e| !e.is_empty())
        .map(|e| vals.get(e))
        .collect::<Option<_>>()?;
    let op = node.op_type.as_str();

    let out = match op {
        "Constant" => vec![constant_value(node)?],
        "Shape" => {
            let dims = &ins[0].dims;
            vec![Val {
                dims: vec![dims.len() as i64],
                data: Some(Const::I64(dims.clone())),
            }]
        }
        "Gather" => vec![gather(node, ins)?],
        "Unsqueeze" => vec![unsqueeze(node, ins[0])?],
        "Squeeze" => vec![squeeze(node, ins[0])?],
        "Concat" => vec![concat(node, &ins)?],
        "Cast" => vec![cast(ins[0])],
        "Reshape" => vec![reshape(ins[0], ins[1])?],
        "Transpose" => vec![transpose(node, ins[0])?],
        "NonZero" => vec![nonzero(ins[0])?],
        "Slice" => vec![slice(&ins)?],
        "ConstantOfShape" => vec![constant_of_shape(node, ins[0])?],
        "Range" => vec![range(ins)?],
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => vec![elementwise(op, ins[0], ins[1])?],
        "Equal" => vec![equal(ins[0], ins[1])?],
        "Where" => {
            let d = shape::broadcast(&ins[0].dims, &ins[1].dims)?;
            vec![Val::shape_only(shape::broadcast(&d, &ins[2].dims)?)]
        }
        "MatMul" => vec![matmul_shape(ins[0], ins[1])?],
        "Gemm" => vec![gemm_shape(node, ins)?],
        "ReduceMean" => vec![reduce_shape(node, ins[0])?],
        "Softmax" | "Sqrt" | "Tanh" | "Relu" | "Gelu" | "Erf" | "Neg" => {
            vec![Val::shape_only(ins[0].dims.clone())]
        }
        "Split" => split_shapes(node, ins[0])?,
        // Where and anything else is not handled yet.
        _ => return None,
    };
    Some(out)
}

// ---- op evaluators ----

fn constant_value(node: &Node) -> Option<Val> {
    match node.attrs.get("value") {
        Some(Attribute::Tensor(t)) => {
            let data = match &t.data {
                TensorData::I64(v) => Const::I64(v.clone()),
                TensorData::I32(v) => Const::I64(v.iter().map(|&x| x as i64).collect()),
                TensorData::F32(v) => Const::F32(v.clone()),
                _ => return None,
            };
            Some(Val {
                dims: t.dims.clone(),
                data: Some(data),
            })
        }
        _ => None,
    }
}

fn attr_ints<'a>(node: &'a Node, name: &str) -> Option<&'a [i64]> {
    match node.attrs.get(name) {
        Some(Attribute::Ints(v)) => Some(v),
        _ => None,
    }
}

fn attr_int(node: &Node, name: &str) -> Option<i64> {
    match node.attrs.get(name) {
        Some(Attribute::Int(i)) => Some(*i),
        _ => None,
    }
}

fn norm_axis(axis: i64, rank: usize) -> usize {
    if axis < 0 {
        (axis + rank as i64) as usize
    } else {
        axis as usize
    }
}

fn gather(node: &Node, ins: Vec<&Val>) -> Option<Val> {
    let (data, idx) = (ins[0], ins[1]);
    let axis = norm_axis(attr_int(node, "axis").unwrap_or(0), data.dims.len().max(1));
    // data[..axis] ++ idx.dims ++ data[axis+1..].
    let mut dims = data.dims[..axis].to_vec();
    dims.extend_from_slice(&idx.dims);
    dims.extend_from_slice(&data.dims[axis + 1..]);

    // Only integer gathers fold; the big f32 ones stay runtime.
    let data_c = data.data.as_ref();
    let idx_c = idx.data.as_ref();
    let folded = match (data_c, idx_c) {
        (Some(d), Some(i)) if d.len() <= FOLD_F32_LIMIT => {
            let dv = d.as_i64();
            let iv = i.as_i64();
            // Only the common rank-1 data with a scalar or 1-D index.
            if data.dims.len() == 1 {
                let axis_len = data.dims[0];
                let picked: Option<Vec<i64>> = iv
                    .iter()
                    .map(|&j| {
                        let j = if j < 0 { j + axis_len } else { j };
                        dv.get(j as usize).copied()
                    })
                    .collect();
                picked.map(Const::I64)
            } else {
                None
            }
        }
        _ => None,
    };
    Some(Val { dims, data: folded })
}

fn unsqueeze(node: &Node, v: &Val) -> Option<Val> {
    let axes = attr_ints(node, "axes")?;
    let mut dims = v.dims.clone();
    let rank = dims.len() + axes.len();
    let mut norm: Vec<usize> = axes.iter().map(|&a| norm_axis(a, rank)).collect();
    norm.sort_unstable();
    for &a in &norm {
        dims.insert(a.min(dims.len()), 1);
    }
    // The data is unchanged; only the shape gains size-1 axes.
    Some(Val {
        dims,
        data: v.data.clone(),
    })
}

fn squeeze(node: &Node, v: &Val) -> Option<Val> {
    let dims: Dims = match attr_ints(node, "axes") {
        Some(axes) => {
            let drop: Vec<usize> = axes.iter().map(|&a| norm_axis(a, v.dims.len())).collect();
            v.dims
                .iter()
                .enumerate()
                .filter(|(i, _)| !drop.contains(i))
                .map(|(_, &d)| d)
                .collect()
        }
        None => v.dims.iter().copied().filter(|&d| d != 1).collect(),
    };
    Some(Val {
        dims,
        data: v.data.clone(),
    })
}

fn concat(node: &Node, ins: &[&Val]) -> Option<Val> {
    let axis = norm_axis(attr_int(node, "axis")?, ins[0].dims.len());
    let mut dims = ins[0].dims.clone();
    dims[axis] = ins.iter().map(|v| v.dims[axis]).sum();

    // Folds when every input is constant, as when a shape vector is built up.
    let folded: Option<Vec<i64>> = ins
        .iter()
        .map(|v| v.data.as_ref().map(Const::as_i64))
        .collect::<Option<Vec<_>>>()
        .map(|parts| parts.concat());
    Some(Val {
        dims,
        data: folded.map(Const::I64),
    })
}

fn cast(v: &Val) -> Val {
    // The shape is preserved and integer data passes through as it stands.
    v.clone()
}

fn reshape(data: &Val, shape: &Val) -> Option<Val> {
    let target = shape.data.as_ref()?.as_i64();
    let dims = crate::layout::reshape_dims(&data.dims, &target).ok()?;
    // A reshape leaves constant data alone.
    Some(Val {
        dims,
        data: data.data.clone(),
    })
}

fn transpose(node: &Node, v: &Val) -> Option<Val> {
    let rank = v.dims.len();
    let perm: Vec<usize> = match attr_ints(node, "perm") {
        Some(p) => p.iter().map(|&a| a as usize).collect(),
        None => (0..rank).rev().collect(),
    };
    let dims: Dims = perm.iter().map(|&p| v.dims[p]).collect();
    // Integer data folds: shape vectors and position coordinates transpose.
    let data = match &v.data {
        Some(Const::I64(d)) => Some(Const::I64(permute_i64(d, &v.dims, &perm))),
        _ => None,
    };
    Some(Val { dims, data })
}

/// Row-major permutation of an i64 tensor's data.
fn permute_i64(data: &[i64], dims: &[i64], perm: &[usize]) -> Vec<i64> {
    let out_dims: Dims = perm.iter().map(|&p| dims[p]).collect();
    let in_strides = crate::abi::row_major_strides(dims);
    let out_strides = crate::abi::row_major_strides(&out_dims);
    let mut out = vec![0i64; data.len()];
    for (lin, slot) in out.iter_mut().enumerate() {
        let mut rem = lin;
        let mut src = 0usize;
        for k in 0..perm.len() {
            let c = rem / out_strides[k] as usize;
            rem %= out_strides[k] as usize;
            src += c * in_strides[perm[k]] as usize;
        }
        *slot = data[src];
    }
    out
}

/// Opset-10 Slice. Every input but the data must be constant. Computes the
/// per-axis index selection and, when the data is constant too, the values.
fn slice(ins: &[&Val]) -> Option<Val> {
    let data = ins[0];
    let starts = ins.get(1)?.data.as_ref()?.as_i64();
    let ends = ins.get(2)?.data.as_ref()?.as_i64();
    let rank = data.dims.len();
    let axes: Vec<usize> = match ins.get(3) {
        Some(v) => v
            .data
            .as_ref()?
            .as_i64()
            .iter()
            .map(|&a| norm_axis(a, rank))
            .collect(),
        None => (0..starts.len()).collect(),
    };
    let steps: Vec<i64> = match ins.get(4) {
        Some(v) => v.data.as_ref()?.as_i64(),
        None => vec![1; starts.len()],
    };

    // Unsliced axes keep their full range.
    let mut sel: Vec<Vec<i64>> = data.dims.iter().map(|&d| (0..d).collect()).collect();
    for (i, &ax) in axes.iter().enumerate() {
        let dim = data.dims[ax];
        let step = steps[i];
        if step == 0 {
            return None;
        }
        let norm = |x: i64| if x < 0 { x + dim } else { x };
        let mut idxs = Vec::new();
        if step > 0 {
            let s = norm(starts[i]).clamp(0, dim);
            let e = norm(ends[i]).clamp(0, dim);
            let mut x = s;
            while x < e {
                idxs.push(x);
                x += step;
            }
        } else {
            let s = norm(starts[i]).clamp(0, dim - 1);
            let e = norm(ends[i]).clamp(-1, dim - 1);
            let mut x = s;
            while x > e {
                idxs.push(x);
                x += step;
            }
        }
        sel[ax] = idxs;
    }

    let dims: Dims = sel.iter().map(|s| s.len() as i64).collect();
    let out = data.data.as_ref().map(|c| select(c, &data.dims, &sel));
    Some(Val { dims, data: out })
}

/// Gather a subtensor by per-axis index lists, preserving element type.
fn select(data: &Const, dims: &[i64], sel: &[Vec<i64>]) -> Const {
    let in_strides = crate::abi::row_major_strides(dims);
    let out_dims: Dims = sel.iter().map(|s| s.len() as i64).collect();
    let out_n = out_dims.iter().product::<i64>() as usize;
    let out_strides = crate::abi::row_major_strides(&out_dims);
    let rank = dims.len();
    let src_of = |lin: usize| -> usize {
        let mut rem = lin;
        let mut src = 0usize;
        for k in 0..rank {
            let c = rem / out_strides[k] as usize;
            rem %= out_strides[k] as usize;
            src += sel[k][c] as usize * in_strides[k] as usize;
        }
        src
    };
    match data {
        Const::I64(v) => Const::I64((0..out_n).map(|l| v[src_of(l)]).collect()),
        Const::F32(v) => Const::F32((0..out_n).map(|l| v[src_of(l)]).collect()),
    }
}

/// The coordinates of a constant tensor's non-zero elements, as an
/// `[rank, nnz]` i64 tensor. GPT-2's position-id arange trick runs this over an
/// all-ones vector.
fn nonzero(v: &Val) -> Option<Val> {
    let flat = v.data.as_ref()?.as_i64();
    let rank = v.dims.len();
    let strides = crate::abi::row_major_strides(&v.dims);
    let mut coords: Vec<Vec<i64>> = vec![Vec::new(); rank];
    for (lin, &val) in flat.iter().enumerate() {
        if val != 0 {
            let mut rem = lin as i64;
            for k in 0..rank {
                coords[k].push(rem / strides[k]);
                rem %= strides[k];
            }
        }
    }
    let nnz = coords.first().map(Vec::len).unwrap_or(0) as i64;
    let out: Vec<i64> = coords.into_iter().flatten().collect();
    Some(Val {
        dims: vec![rank as i64, nnz],
        data: Some(Const::I64(out)),
    })
}

fn constant_of_shape(node: &Node, shape: &Val) -> Option<Val> {
    let dims = shape.data.as_ref()?.as_i64();
    let n: i64 = dims.iter().product();
    let fill = match node.attrs.get("value") {
        Some(Attribute::Tensor(t)) => match &t.data {
            TensorData::I64(v) => v.first().copied().unwrap_or(0),
            TensorData::F32(v) => v.first().copied().unwrap_or(0.0) as i64,
            _ => 0,
        },
        _ => 0,
    };
    Some(Val {
        dims: dims.clone(),
        data: Some(Const::I64(vec![fill; n as usize])),
    })
}

fn range(ins: Vec<&Val>) -> Option<Val> {
    let start = ins[0].data.as_ref()?.as_i64().first().copied()?;
    let limit = ins[1].data.as_ref()?.as_i64().first().copied()?;
    let delta = ins[2].data.as_ref()?.as_i64().first().copied()?;
    if delta == 0 {
        return None;
    }
    let mut v = Vec::new();
    let mut x = start;
    while (delta > 0 && x < limit) || (delta < 0 && x > limit) {
        v.push(x);
        x += delta;
    }
    Some(Val {
        dims: vec![v.len() as i64],
        data: Some(Const::I64(v)),
    })
}

fn elementwise(op: &str, a: &Val, b: &Val) -> Option<Val> {
    let dims = shape::broadcast(&a.dims, &b.dims)?;
    // Only elementwise-equal integer vectors fold, the shape-math case.
    let folded = match (a.data.as_ref(), b.data.as_ref()) {
        (Some(x), Some(y)) if x.len() == y.len() && dims == a.dims && dims == b.dims => {
            let (xv, yv) = (x.as_i64(), y.as_i64());
            let out: Vec<i64> = xv
                .iter()
                .zip(&yv)
                .map(|(&p, &q)| match op {
                    "Add" => p + q,
                    "Sub" => p - q,
                    "Mul" => p * q,
                    "Div" => {
                        if q != 0 {
                            p / q
                        } else {
                            0
                        }
                    }
                    "Pow" => p.pow(q.max(0) as u32),
                    _ => 0,
                })
                .collect();
            Some(Const::I64(out))
        }
        _ => None,
    };
    Some(Val { dims, data: folded })
}

fn equal(a: &Val, b: &Val) -> Option<Val> {
    let dims = shape::broadcast(&a.dims, &b.dims)?;
    Some(Val::shape_only(dims))
}

fn matmul_shape(a: &Val, b: &Val) -> Option<Val> {
    let (ad, bd) = (&a.dims, &b.dims);
    // 2-D and stacked batched matmul: batch dims broadcast, the last two
    // contract.
    let (am, ak) = (ad[ad.len() - 2], ad[ad.len() - 1]);
    let (bk, bn) = (bd[bd.len() - 2], bd[bd.len() - 1]);
    if ak != bk {
        return None;
    }
    let batch = shape::broadcast(&ad[..ad.len() - 2], &bd[..bd.len() - 2])?;
    let mut dims = batch;
    dims.push(am);
    dims.push(bn);
    Some(Val::shape_only(dims))
}

fn gemm_shape(node: &Node, ins: Vec<&Val>) -> Option<Val> {
    let (a, b) = (&ins[0].dims, &ins[1].dims);
    let ta = attr_int(node, "transA").unwrap_or(0) != 0;
    let tb = attr_int(node, "transB").unwrap_or(0) != 0;
    let m = if ta { a[1] } else { a[0] };
    let n = if tb { b[0] } else { b[1] };
    Some(Val::shape_only(vec![m, n]))
}

fn reduce_shape(node: &Node, v: &Val) -> Option<Val> {
    let keepdims = attr_int(node, "keepdims").unwrap_or(1) != 0;
    let axes: Vec<usize> = match attr_ints(node, "axes") {
        Some(a) => a.iter().map(|&x| norm_axis(x, v.dims.len())).collect(),
        None => (0..v.dims.len()).collect(),
    };
    let mut dims = Vec::new();
    for (i, &d) in v.dims.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                dims.push(1);
            }
        } else {
            dims.push(d);
        }
    }
    Some(Val::shape_only(dims))
}

fn split_shapes(node: &Node, v: &Val) -> Option<Vec<Val>> {
    let axis = norm_axis(attr_int(node, "axis").unwrap_or(0), v.dims.len());
    let n = node.outputs.iter().filter(|o| !o.is_empty()).count() as i64;
    let sizes: Vec<i64> = match attr_ints(node, "split") {
        Some(s) => s.to_vec(),
        None if v.dims[axis] % n == 0 => vec![v.dims[axis] / n; n as usize],
        None => return None,
    };
    Some(
        sizes
            .iter()
            .map(|&s| {
                let mut d = v.dims.clone();
                d[axis] = s;
                Val::shape_only(d)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests;
