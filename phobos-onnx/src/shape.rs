use std::collections::HashMap;

use anyhow::{Context, Result, bail};

use crate::ir::{Attribute, Dim, Graph, Node, TensorData};

/// Fully static dimensions of a tensor edge.
pub type Dims = Vec<i64>;

/// Infer static dims for every edge in `graph`, walking it in node order from
/// the initializers and the declared input shapes. Fails on a symbolic or
/// unknown shape, or on an op this pass does not model.
pub fn infer(graph: &Graph) -> Result<HashMap<String, Dims>> {
    let mut dims: HashMap<String, Dims> = HashMap::new();

    for (name, t) in &graph.initializers {
        dims.insert(name.clone(), t.dims.clone());
    }

    // An input already fed by an initializer is an optional one with a default,
    // which ONNX allows.
    for vi in &graph.inputs {
        if dims.contains_key(&vi.name) {
            continue;
        }
        dims.insert(vi.name.clone(), static_dims(graph, &vi.name)?);
    }

    for node in &graph.nodes {
        let ins: Vec<&Dims> = node
            .inputs
            .iter()
            .filter(|e| !e.is_empty())
            .map(|e| {
                dims.get(e.as_str())
                    .ok_or_else(|| anyhow::anyhow!("edge '{e}' has no known shape"))
            })
            .collect::<Result<_>>()?;

        let outs = infer_node(node, &ins, graph)
            .with_context(|| format!("inferring shape of '{}' ({})", node.name, node.op_type))?;

        // A node may declare more outputs than this pass models, such as
        // LayerNorm's Mean and InvStdDev; only the resolved ones are set.
        for (name, out) in node.outputs.iter().zip(outs) {
            if !name.is_empty() {
                dims.insert(name.clone(), out);
            }
        }
    }

    Ok(dims)
}

/// One entry per output the op produces, in output order.
fn infer_node(node: &Node, ins: &[&Dims], graph: &Graph) -> Result<Vec<Dims>> {
    let op = node.op_type.as_str();
    let one = |d: Dims| vec![d];
    Ok(match op {
        "MatMul" => {
            let [a, b] = expect_n::<2>(op, ins)?;
            if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
                bail!("MatMul expects 2-D [M,K] x [K,N] with matching K, got {a:?} x {b:?}");
            }
            one(vec![a[0], b[1]])
        }
        "Add" | "Sub" | "Mul" | "Div" => {
            let [a, b] = expect_n::<2>(op, ins)?;
            one(broadcast(&a, &b)
                .ok_or_else(|| anyhow::anyhow!("{op}: shapes {a:?} and {b:?} do not broadcast"))?)
        }
        "Relu" | "Gelu" => one(expect_n::<1>(op, ins)?[0].clone()),
        // Normalization and softmax preserve the first input's shape.
        "LayerNormalization" | "Softmax" => one(first_in(op, ins)?.clone()),
        "Reshape" => {
            let data = first_in(op, ins)?;
            let target = const_i64(graph, &node.inputs[1])
                .ok_or_else(|| anyhow::anyhow!("Reshape needs a constant shape input"))?;
            one(crate::layout::reshape_dims(data, &target)?)
        }
        "Transpose" => {
            let data = first_in(op, ins)?;
            let perm = transpose_perm(node, data.len());
            one(perm.iter().map(|&p| data[p]).collect())
        }
        "Gather" => {
            let [data, idx] = expect_n::<2>(op, ins)?;
            let axis = norm_axis(attr_int(node, "axis").unwrap_or(0), data.len())?;
            let mut out = data[..axis].to_vec();
            out.extend_from_slice(&idx);
            out.extend_from_slice(&data[axis + 1..]);
            one(out)
        }
        "Concat" => {
            let axis = norm_axis(
                attr_int(node, "axis").ok_or_else(|| anyhow::anyhow!("Concat needs axis"))?,
                first_in(op, ins)?.len(),
            )?;
            let mut out = first_in(op, ins)?.clone();
            out[axis] = ins.iter().map(|d| d[axis]).sum();
            one(out)
        }
        "Split" => {
            let data = first_in(op, ins)?;
            let axis = norm_axis(attr_int(node, "axis").unwrap_or(0), data.len())?;
            let sizes = split_sizes(node, graph, data[axis])?;
            sizes
                .iter()
                .map(|&s| {
                    let mut d = data.clone();
                    d[axis] = s;
                    d
                })
                .collect()
        }
        // Fused ops, see crate::transform.
        "PhobosFusedLinear" => {
            let a = ins
                .first()
                .ok_or_else(|| anyhow::anyhow!("FusedLinear needs A"))?;
            let b = ins
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("FusedLinear needs B"))?;
            if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
                bail!("FusedLinear expects 2-D [M,K] x [K,N], got {a:?} x {b:?}");
            }
            one(vec![a[0], b[1]])
        }
        "PhobosFlashAttention" => one(first_in(op, ins)?.clone()),
        other => bail!("shape inference does not model op '{other}' yet"),
    })
}

fn first_in<'a>(op: &str, ins: &'a [&Dims]) -> Result<&'a Dims> {
    ins.first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("{op} needs a data input"))
}

/// The Transpose `perm` attribute, defaulting to reversing all axes.
fn transpose_perm(node: &Node, rank: usize) -> Vec<usize> {
    match attr_ints(node, "perm") {
        Some(perm) => perm.iter().map(|&p| p as usize).collect(),
        None => (0..rank).rev().collect(),
    }
}

/// Split sizes from the `split` attribute, the opset-13 split input, or an
/// equal division across the declared outputs.
fn split_sizes(node: &Node, graph: &Graph, axis_len: i64) -> Result<Dims> {
    if let Some(s) = attr_ints(node, "split") {
        return Ok(s.to_vec());
    }
    if let Some(edge) = node.inputs.get(1)
        && let Some(s) = const_i64(graph, edge)
    {
        return Ok(s);
    }
    let n = node.outputs.iter().filter(|o| !o.is_empty()).count() as i64;
    if n > 0 && axis_len % n == 0 {
        return Ok(vec![axis_len / n; n as usize]);
    }
    bail!("cannot determine Split sizes for axis extent {axis_len} across {n} outputs");
}

/// Normalize a possibly-negative axis into `0..rank`.
fn norm_axis(axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a as usize >= rank {
        bail!("axis {axis} out of range for rank {rank}");
    }
    Ok(a as usize)
}

/// Read an i64 or i32 initializer as i64 values.
pub fn const_i64(graph: &Graph, edge: &str) -> Option<Dims> {
    match &graph.initializers.get(edge)?.data {
        TensorData::I64(v) => Some(v.clone()),
        TensorData::I32(v) => Some(v.iter().map(|&x| x as i64).collect()),
        _ => None,
    }
}

fn attr_int(node: &Node, name: &str) -> Option<i64> {
    match node.attrs.get(name) {
        Some(Attribute::Int(i)) => Some(*i),
        _ => None,
    }
}

fn attr_ints<'a>(node: &'a Node, name: &str) -> Option<&'a [i64]> {
    match node.attrs.get(name) {
        Some(Attribute::Ints(v)) => Some(v),
        _ => None,
    }
}

/// NumPy-style broadcasting of two shapes, right-aligned. `None` when a pair of
/// aligned axes is neither equal nor has one side of extent 1.
pub fn broadcast(a: &[i64], b: &[i64]) -> Option<Dims> {
    let (mut i, mut j) = (a.len(), b.len());
    let mut out = Vec::with_capacity(i.max(j));
    while i > 0 || j > 0 {
        let da = if i > 0 { a[i - 1] } else { 1 };
        let db = if j > 0 { b[j - 1] } else { 1 };
        let d = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            return None;
        };
        out.push(d);
        i = i.saturating_sub(1);
        j = j.saturating_sub(1);
    }
    out.reverse();
    Some(out)
}

fn expect_n<const N: usize>(op_type: &str, ins: &[&Dims]) -> Result<[Dims; N]> {
    if ins.len() != N {
        bail!("{op_type} expects {N} inputs, got {}", ins.len());
    }
    Ok(std::array::from_fn(|i| ins[i].clone()))
}

/// Read a fully static shape for a named edge out of the graph's value info.
fn static_dims(graph: &Graph, name: &str) -> Result<Dims> {
    let Some(vi) = graph.values.get(name) else {
        bail!("input '{name}' has no declared type");
    };
    let Some(axes) = &vi.shape.0 else {
        bail!("input '{name}' has an unknown-rank shape");
    };
    axes.iter()
        .map(|d| match d {
            Dim::Fixed(n) => Ok(*n),
            Dim::Symbol(s) => {
                bail!("input '{name}' has a symbolic dim '{s}'; step 2 needs static shapes")
            }
            Dim::Unknown => bail!("input '{name}' has an unknown dim; step 2 needs static shapes"),
        })
        .collect()
}

#[cfg(test)]
mod tests;
