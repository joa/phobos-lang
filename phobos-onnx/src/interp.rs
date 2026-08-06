use std::collections::HashMap;

use anyhow::{Context, Result, bail};

use crate::abi::row_major_strides;
use crate::ir::{Attribute, DataType, Graph, Node, TensorData};
use crate::shape::Dims;

/// Row-major data plus its shape.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub dims: Dims,
    pub data: Data,
}

#[derive(Clone, Debug)]
pub enum Data {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl Tensor {
    pub fn f32(dims: Dims, data: Vec<f32>) -> Tensor {
        Tensor {
            dims,
            data: Data::F32(data),
        }
    }
    pub fn i64(dims: Dims, data: Vec<i64>) -> Tensor {
        Tensor {
            dims,
            data: Data::I64(data),
        }
    }
    /// Values as f32, casting i64, for callers reading results.
    pub fn to_f32(&self) -> Vec<f32> {
        self.as_f32()
    }
    /// Values as f32, casting i64, for arithmetic.
    fn as_f32(&self) -> Vec<f32> {
        match &self.data {
            Data::F32(v) => v.clone(),
            Data::I64(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }
    fn as_i64(&self) -> Vec<i64> {
        match &self.data {
            Data::I64(v) => v.clone(),
            Data::F32(v) => v.iter().map(|&x| x as i64).collect(),
        }
    }
}

/// The ops worth offloading to Phobos GPU kernels: the FLOP-heavy `Gemm`
/// projections and `LayerNormalization`. Everything else stays on the host, and
/// `layer_norm` defaults to a host implementation so a matmul-only backend
/// still works.
pub trait MatmulBackend {
    /// `C[m,n] = A[m,k] @ B[k,n]`, both row-major.
    fn matmul(&self, a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Result<Vec<f32>>;

    /// [`MatmulBackend::matmul`] where `b_key`, when set, names a constant right
    /// operand so the backend can keep it device-resident across calls. The
    /// default ignores it; only weight-caching backends override this.
    fn matmul_cached(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        _b_key: Option<&str>,
    ) -> Result<Vec<f32>> {
        self.matmul(a, m, k, b, n)
    }

    /// LayerNorm over the last axis of an `[rows, w]` view:
    /// `y = (x - mean) / sqrt(var + eps) * scale + bias`, both of length w.
    fn layer_norm(
        &self,
        x: &[f32],
        rows: usize,
        w: usize,
        scale: &[f32],
        bias: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>> {
        host_layer_norm(x, rows, w, scale, bias, eps)
    }
}

pub struct HostBackend;

impl MatmulBackend for HostBackend {
    fn matmul(&self, a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Result<Vec<f32>> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += aip * b[p * n + j];
                }
            }
        }
        Ok(c)
    }
}

pub fn host_layer_norm(
    x: &[f32],
    rows: usize,
    w: usize,
    scale: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; rows * w];
    for r in 0..rows {
        let row = &x[r * w..r * w + w];
        let mean = row.iter().sum::<f32>() / w as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / w as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..w {
            out[r * w + c] = (row[c] - mean) * inv * scale[c] + bias[c];
        }
    }
    Ok(out)
}

/// Execute `graph` with the given inputs on the host, returning its outputs.
pub fn run(graph: &Graph, inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    run_with(graph, inputs, &HostBackend)
}

/// [`run`] with `Gemm` matmuls dispatched to `backend`. Everything else stays
/// on the host.
pub fn run_with(
    graph: &Graph,
    inputs: &HashMap<String, Tensor>,
    backend: &dyn MatmulBackend,
) -> Result<HashMap<String, Tensor>> {
    run_impl(graph, inputs, &HashMap::new(), backend)
}

/// One node on the host; `inputs` must hold every non-empty input edge it
/// reads. The device executor falls back to this for ops with no kernel.
pub fn run_node_host(node: &Node, inputs: &HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
    eval_node(node, inputs, &HashMap::new(), &HostBackend)
}

/// Decode a graph's initializers once, for reuse across runs. KV-cache steps
/// re-fold the same weights every token, and decoding per step would
/// re-materialize hundreds of megabytes. Feeds [`run_with_env`].
pub fn decode_initializers(graph: &Graph) -> Result<HashMap<String, Tensor>> {
    graph
        .initializers
        .iter()
        .map(|(n, t)| Ok((n.clone(), from_initializer(t)?)))
        .collect()
}

/// [`run_with`] taking its weights from `weights`, decoded once by
/// [`decode_initializers`], rather than from the graph's initializers on every
/// call. Folded plumbing constants unique to this graph are still decoded.
pub fn run_with_env(
    graph: &Graph,
    inputs: &HashMap<String, Tensor>,
    weights: &HashMap<String, Tensor>,
    backend: &dyn MatmulBackend,
) -> Result<HashMap<String, Tensor>> {
    run_impl(graph, inputs, weights, backend)
}

fn run_impl(
    graph: &Graph,
    inputs: &HashMap<String, Tensor>,
    weights: &HashMap<String, Tensor>,
    backend: &dyn MatmulBackend,
) -> Result<HashMap<String, Tensor>> {
    let mut env: HashMap<String, Tensor> = HashMap::new();

    // Only the initializers `weights` does not already provide are decoded.
    for (name, t) in &graph.initializers {
        if !weights.contains_key(name) {
            env.insert(name.clone(), from_initializer(t)?);
        }
    }
    for (name, t) in inputs {
        env.insert(name.clone(), t.clone());
    }

    for node in &graph.nodes {
        let outs = eval_node(node, &env, weights, backend)
            .with_context(|| format!("interp node '{}' ({})", node.name, node.op_type))?;
        for (name, t) in node.outputs.iter().zip(outs) {
            if !name.is_empty() {
                env.insert(name.clone(), t);
            }
        }
    }

    let mut out = HashMap::new();
    for vi in &graph.outputs {
        let t = env
            .get(&vi.name)
            .or_else(|| weights.get(&vi.name))
            .with_context(|| format!("output '{}' was not produced", vi.name))?;
        out.insert(vi.name.clone(), t.clone());
    }
    Ok(out)
}

fn from_initializer(t: &crate::ir::Tensor) -> Result<Tensor> {
    let data = match &t.data {
        TensorData::F32(v) => Data::F32(v.clone()),
        TensorData::I64(v) => Data::I64(v.clone()),
        TensorData::I32(v) => Data::I64(v.iter().map(|&x| x as i64).collect()),
        TensorData::Bool(v) => Data::F32(v.iter().map(|&b| b as i32 as f32).collect()),
        // Undecoded payloads, such as the int8 causal mask.
        TensorData::Raw(bytes) => match t.data_type {
            DataType::I8 => Data::F32(bytes.iter().map(|&b| b as i8 as f32).collect()),
            DataType::U8 | DataType::Bool => Data::F32(bytes.iter().map(|&b| b as f32).collect()),
            other => bail!("raw initializer data type {other:?} not supported by interp"),
        },
        other => bail!("initializer data type {other:?} not supported by interp"),
    };
    Ok(Tensor {
        dims: t.dims.clone(),
        data,
    })
}

fn eval_node(
    node: &Node,
    env: &HashMap<String, Tensor>,
    weights: &HashMap<String, Tensor>,
    backend: &dyn MatmulBackend,
) -> Result<Vec<Tensor>> {
    let ins: Vec<&Tensor> = node
        .inputs
        .iter()
        .filter(|e| !e.is_empty())
        .map(|e| {
            env.get(e)
                .or_else(|| weights.get(e))
                .with_context(|| format!("missing input '{e}'"))
        })
        .collect::<Result<_>>()?;
    let op = node.op_type.as_str();

    let out = match op {
        // A constant the folder could not resolve, a bool causal-mask buffer
        // among them, survives into the residual graph.
        "Constant" => match node.attrs.get("value") {
            Some(Attribute::Tensor(t)) => vec![from_initializer(t)?],
            _ => bail!("Constant '{}' has no tensor value attribute", node.name),
        },
        "Add" => vec![broadcast_bin(ins[0], ins[1], |a, b| a + b)?],
        "Sub" => vec![broadcast_bin(ins[0], ins[1], |a, b| a - b)?],
        "Mul" => vec![broadcast_bin(ins[0], ins[1], |a, b| a * b)?],
        "Div" => vec![broadcast_bin(ins[0], ins[1], |a, b| a / b)?],
        "Pow" => vec![broadcast_bin(ins[0], ins[1], |a, b| a.powf(b))?],
        "Sqrt" => vec![unary(ins[0], |x| x.sqrt())],
        "Tanh" => vec![unary(ins[0], |x| x.tanh())],
        "Relu" => vec![unary(ins[0], |x| x.max(0.0))],
        "Neg" => vec![unary(ins[0], |x| -x)],
        "Gelu" => vec![unary(ins[0], gelu)],
        "Erf" => vec![unary(ins[0], erf)],
        "LayerNormalization" => vec![layer_norm(node, ins[0], ins[1], ins[2], backend)?],
        "ReduceMean" => vec![reduce_mean(node, ins[0])?],
        "Softmax" => vec![softmax(node, ins[0])?],
        "MatMul" => vec![matmul(node, ins[0], ins[1], weights, backend)?],
        "Gemm" => vec![gemm(node, &ins, weights, backend)?],
        "Cast" => vec![cast(node, ins[0])?],
        "Where" => vec![where_op(ins[0], ins[1], ins[2])?],
        "Reshape" => vec![reshape(ins[0], ins[1])?],
        "Transpose" => vec![transpose(node, ins[0])?],
        "Unsqueeze" => vec![unsqueeze(node, ins[0])?],
        "Squeeze" => vec![squeeze(node, ins[0])?],
        "Concat" => vec![concat(node, &ins)?],
        "Split" => split(node, ins[0])?,
        "Gather" => vec![gather(node, ins[0], ins[1])?],
        "Slice" => vec![slice(&ins)?],
        // Shape plumbing the folder could not resolve, the with-past graph's
        // dynamic past length among it.
        "Shape" => vec![Tensor::i64(
            vec![ins[0].dims.len() as i64],
            ins[0].dims.clone(),
        )],
        "Range" => vec![range_op(ins[0], ins[1], ins[2])?],
        other => bail!("interp does not implement op '{other}'"),
    };
    Ok(out)
}

// ---- shape plumbing ----

/// `Range(start, limit, delta)`: values from `start` stepping by `delta` while
/// below `limit`. Stays i64 for integer inputs, f32 otherwise.
fn range_op(start: &Tensor, limit: &Tensor, delta: &Tensor) -> Result<Tensor> {
    let (s, l, d) = (start.as_f32()[0], limit.as_f32()[0], delta.as_f32()[0]);
    if d == 0.0 {
        bail!("Range with zero delta");
    }
    let n = (((l - s) / d).ceil()).max(0.0) as usize;
    let integral = matches!(
        (&start.data, &limit.data, &delta.data),
        (Data::I64(_), Data::I64(_), Data::I64(_))
    );
    Ok(if integral {
        Tensor::i64(
            vec![n as i64],
            (0..n).map(|i| (s + i as f32 * d) as i64).collect(),
        )
    } else {
        Tensor::f32(vec![n as i64], (0..n).map(|i| s + i as f32 * d).collect())
    })
}

// ---- elementwise ----

fn unary(t: &Tensor, f: impl Fn(f32) -> f32) -> Tensor {
    Tensor::f32(t.dims.clone(), t.as_f32().iter().map(|&x| f(x)).collect())
}

/// NumPy-broadcasting binary op over f32.
fn broadcast_bin(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor> {
    let dims = crate::shape::broadcast(&a.dims, &b.dims)
        .with_context(|| format!("shapes {:?} and {:?} do not broadcast", a.dims, b.dims))?;
    let (av, bv) = (a.as_f32(), b.as_f32());
    let (as_, bs_) = (bcast_strides(&a.dims, &dims), bcast_strides(&b.dims, &dims));
    let out_strides = row_major_strides(&dims);
    let n: usize = dims.iter().product::<i64>() as usize;
    let mut out = vec![0.0f32; n];
    for (lin, slot) in out.iter_mut().enumerate() {
        let mut rem = lin;
        let (mut ai, mut bi) = (0usize, 0usize);
        for k in 0..dims.len() {
            let c = rem / out_strides[k] as usize;
            rem %= out_strides[k] as usize;
            ai += c * as_[k];
            bi += c * bs_[k];
        }
        *slot = f(av[ai], bv[bi]);
    }
    Ok(Tensor::f32(dims, out))
}

/// Strides of `src` broadcast against `out`, zero where src has extent 1.
fn bcast_strides(src: &[i64], out: &[i64]) -> Vec<usize> {
    let src_strides = row_major_strides(src);
    let off = out.len() - src.len();
    (0..out.len())
        .map(|k| {
            if k < off {
                0
            } else {
                let s = k - off;
                if src[s] == 1 {
                    0
                } else {
                    src_strides[s] as usize
                }
            }
        })
        .collect()
}

fn where_op(cond: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor> {
    // All three broadcast; a non-zero cond selects x, otherwise y.
    let d1 = crate::shape::broadcast(&cond.dims, &x.dims).context("Where broadcast")?;
    let dims = crate::shape::broadcast(&d1, &y.dims).context("Where broadcast")?;
    let cond_b = broadcast_to(cond, &dims);
    let x_b = broadcast_to(x, &dims);
    let y_b = broadcast_to(y, &dims);
    let out = cond_b
        .iter()
        .zip(&x_b)
        .zip(&y_b)
        .map(|((&c, &xv), &yv)| if c != 0.0 { xv } else { yv })
        .collect();
    Ok(Tensor::f32(dims, out))
}

fn broadcast_to(t: &Tensor, dims: &[i64]) -> Vec<f32> {
    let v = t.as_f32();
    let strides = bcast_strides(&t.dims, dims);
    let out_strides = row_major_strides(dims);
    let n: usize = dims.iter().product::<i64>() as usize;
    (0..n)
        .map(|lin| {
            let mut rem = lin;
            let mut idx = 0usize;
            for k in 0..dims.len() {
                let c = rem / out_strides[k] as usize;
                rem %= out_strides[k] as usize;
                idx += c * strides[k];
            }
            v[idx]
        })
        .collect()
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// The Abramowitz-Stegun erf approximation, max absolute error 1.5e-7.
#[allow(clippy::excessive_precision)]
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    y.copysign(x)
}

// ---- reductions and softmax ----

fn axes_attr(node: &Node, rank: usize, default_all: bool) -> Vec<usize> {
    match node.attrs.get("axes") {
        Some(Attribute::Ints(a)) => a.iter().map(|&x| norm_axis(x, rank)).collect(),
        _ if default_all => (0..rank).collect(),
        _ => vec![rank - 1],
    }
}

fn reduce_mean(node: &Node, t: &Tensor) -> Result<Tensor> {
    let rank = t.dims.len();
    let axes = axes_attr(node, rank, true);
    let keepdims = int_attr(node, "keepdims").unwrap_or(1) != 0;
    let v = t.as_f32();
    let strides = row_major_strides(&t.dims);

    let mut out_dims = t.dims.clone();
    for &a in &axes {
        out_dims[a] = 1;
    }
    let out_n: usize = out_dims.iter().product::<i64>() as usize;
    let out_strides = row_major_strides(&out_dims);
    let mut sums = vec![0.0f32; out_n];
    let mut count = 1i64;
    for &a in &axes {
        count *= t.dims[a];
    }
    // Each input element maps to its reduced-output slot.
    for (lin, &val) in v.iter().enumerate() {
        let mut rem = lin;
        let mut oi = 0usize;
        for k in 0..rank {
            let c = (rem / strides[k] as usize) as i64;
            rem %= strides[k] as usize;
            let oc = if out_dims[k] == 1 { 0 } else { c };
            oi += oc as usize * out_strides[k] as usize;
        }
        sums[oi] += val;
    }
    for s in &mut sums {
        *s /= count as f32;
    }
    if !keepdims {
        out_dims = out_dims
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, d)| d)
            .collect();
    }
    Ok(Tensor::f32(out_dims, sums))
}

/// LayerNormalization over the last axis, dispatched to `backend`.
fn layer_norm(
    node: &Node,
    x: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    backend: &dyn MatmulBackend,
) -> Result<Tensor> {
    let eps = float_attr(node, "epsilon").unwrap_or(1e-5);
    let w = *x.dims.last().context("LayerNorm input is a scalar")? as usize;
    let rows = x.dims.iter().product::<i64>() as usize / w;
    let out = backend.layer_norm(&x.as_f32(), rows, w, &scale.as_f32(), &bias.as_f32(), eps)?;
    Ok(Tensor::f32(x.dims.clone(), out))
}

fn softmax(node: &Node, t: &Tensor) -> Result<Tensor> {
    let rank = t.dims.len();
    let axis = norm_axis(int_attr(node, "axis").unwrap_or(-1), rank);
    // Collapsed to [outer, axis_len, inner], softmax over the middle.
    let axis_len = t.dims[axis] as usize;
    let inner: usize = t.dims[axis + 1..].iter().product::<i64>() as usize;
    let outer: usize = t.dims[..axis].iter().product::<i64>() as usize;
    let v = t.as_f32();
    let mut out = vec![0.0f32; v.len()];
    for o in 0..outer {
        for i in 0..inner {
            let at = |k: usize| (o * axis_len + k) * inner + i;
            let mut m = f32::NEG_INFINITY;
            for k in 0..axis_len {
                m = m.max(v[at(k)]);
            }
            let mut sum = 0.0f32;
            for k in 0..axis_len {
                let e = (v[at(k)] - m).exp();
                out[at(k)] = e;
                sum += e;
            }
            for k in 0..axis_len {
                out[at(k)] /= sum;
            }
        }
    }
    Ok(Tensor::f32(t.dims.clone(), out))
}

// ---- matmul ----

/// Contract the last dim of `a` with the second-to-last of `b`, the leading
/// batch dims broadcasting.
fn matmul(
    node: &Node,
    a: &Tensor,
    b: &Tensor,
    weights: &HashMap<String, Tensor>,
    backend: &dyn MatmulBackend,
) -> Result<Tensor> {
    let (ad, bd) = (&a.dims, &b.dims);
    let (m, k) = (ad[ad.len() - 2], ad[ad.len() - 1]);
    let (bk, n) = (bd[bd.len() - 2], bd[bd.len() - 1]);
    if k != bk {
        bail!("MatMul inner dims disagree: {ad:?} x {bd:?}");
    }

    // A plain 2-D matmul, the lm-head among them, goes to the backend. A
    // constant B stays device-resident like the Gemm weights.
    if ad.len() == 2 && bd.len() == 2 {
        let b_name = node.inputs.get(1).map(String::as_str).unwrap_or("");
        let b_key = weights.contains_key(b_name).then(|| format!("{b_name}:mm"));
        let c = backend.matmul_cached(
            &a.as_f32(),
            m as usize,
            k as usize,
            &b.as_f32(),
            n as usize,
            b_key.as_deref(),
        )?;
        return Ok(Tensor::f32(vec![m, n], c));
    }
    let a_batch = &ad[..ad.len() - 2];
    let b_batch = &bd[..bd.len() - 2];
    let batch = crate::shape::broadcast(a_batch, b_batch).context("MatMul batch broadcast")?;
    let nb: usize = batch.iter().product::<i64>() as usize;

    let (av, bv) = (a.as_f32(), b.as_f32());
    let a_bs = bcast_strides(a_batch, &batch);
    let b_bs = bcast_strides(b_batch, &batch);
    let (a_mat, b_mat) = ((m * k) as usize, (k * n) as usize);
    let batch_strides = row_major_strides(&batch);

    let mut out = vec![0.0f32; nb * (m * n) as usize];
    for bi in 0..nb {
        // This batch index as per-operand offsets.
        let (mut rem, mut ao, mut bo) = (bi, 0usize, 0usize);
        for k2 in 0..batch.len() {
            let c = rem / batch_strides[k2] as usize;
            rem %= batch_strides[k2] as usize;
            ao += c * a_bs[k2];
            bo += c * b_bs[k2];
        }
        let a_off = ao * a_mat;
        let b_off = bo * b_mat;
        let o_off = bi * (m * n) as usize;
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut acc = 0.0f32;
                for p in 0..k as usize {
                    acc += av[a_off + i * k as usize + p] * bv[b_off + p * n as usize + j];
                }
                out[o_off + i * n as usize + j] = acc;
            }
        }
    }
    let mut dims = batch;
    dims.push(m);
    dims.push(n);
    Ok(Tensor::f32(dims, out))
}

fn gemm(
    node: &Node,
    ins: &[&Tensor],
    weights: &HashMap<String, Tensor>,
    backend: &dyn MatmulBackend,
) -> Result<Tensor> {
    let ta = int_attr(node, "transA").unwrap_or(0) != 0;
    let tb = int_attr(node, "transB").unwrap_or(0) != 0;
    let alpha = float_attr(node, "alpha").unwrap_or(1.0);
    let beta = float_attr(node, "beta").unwrap_or(1.0);
    let a = if ta {
        transpose2(ins[0])
    } else {
        ins[0].clone()
    };
    let b = if tb {
        transpose2(ins[1])
    } else {
        ins[1].clone()
    };
    // A[m,k] @ B[k,n] on the backend, alpha, bias and beta on the host.
    let (m, k, n) = (a.dims[0] as usize, a.dims[1] as usize, b.dims[1] as usize);
    // A constant B, one present in the persistent env, gets a stable key of
    // name and transpose so the backend can keep it device-resident.
    let b_name = node.inputs.get(1).map(String::as_str).unwrap_or("");
    let b_key = weights
        .contains_key(b_name)
        .then(|| format!("{b_name}:{tb}"));
    let c = backend.matmul_cached(&a.as_f32(), m, k, &b.as_f32(), n, b_key.as_deref())?;
    let mut out = Tensor::f32(vec![m as i64, n as i64], c);
    if alpha != 1.0 {
        for x in out.f32_mut() {
            *x *= alpha;
        }
    }
    if let Some(c) = ins.get(2) {
        let cb = broadcast_to(c, &out.dims);
        for (x, cv) in out.f32_mut().iter_mut().zip(cb) {
            *x += beta * cv;
        }
    }
    Ok(out)
}

fn transpose2(t: &Tensor) -> Tensor {
    let (r, c) = (t.dims[0] as usize, t.dims[1] as usize);
    let v = t.as_f32();
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = v[i * c + j];
        }
    }
    Tensor::f32(vec![c as i64, r as i64], out)
}

impl Tensor {
    fn f32_mut(&mut self) -> &mut Vec<f32> {
        if let Data::I64(_) = self.data {
            self.data = Data::F32(self.as_f32());
        }
        match &mut self.data {
            Data::F32(v) => v,
            Data::I64(_) => unreachable!(),
        }
    }
}

// ---- layout and index ----

fn cast(node: &Node, t: &Tensor) -> Result<Tensor> {
    // TensorProto.DataType: 1 FLOAT, 6 INT32, 7 INT64, 9 BOOL.
    match int_attr(node, "to") {
        Some(7) | Some(6) => Ok(Tensor::i64(t.dims.clone(), t.as_i64())),
        _ => Ok(Tensor::f32(t.dims.clone(), t.as_f32())),
    }
}

fn reshape(data: &Tensor, shape: &Tensor) -> Result<Tensor> {
    let dims = crate::layout::reshape_dims(&data.dims, &shape.as_i64())?;
    Ok(Tensor {
        dims,
        data: data.data.clone(),
    })
}

fn transpose(node: &Node, t: &Tensor) -> Result<Tensor> {
    let rank = t.dims.len();
    let perm: Vec<usize> = match node.attrs.get("perm") {
        Some(Attribute::Ints(p)) => p.iter().map(|&a| a as usize).collect(),
        _ => (0..rank).rev().collect(),
    };
    let (out, dims) = crate::layout::transpose(&t.as_f32(), &t.dims, &perm)?;
    Ok(Tensor::f32(dims, out))
}

fn unsqueeze(node: &Node, t: &Tensor) -> Result<Tensor> {
    let axes = match node.attrs.get("axes") {
        Some(Attribute::Ints(a)) => a.clone(),
        _ => bail!("Unsqueeze needs axes"),
    };
    let mut dims = t.dims.clone();
    let rank = dims.len() + axes.len();
    let mut norm: Vec<usize> = axes.iter().map(|&a| norm_axis(a, rank)).collect();
    norm.sort_unstable();
    for a in norm {
        dims.insert(a.min(dims.len()), 1);
    }
    Ok(Tensor {
        dims,
        data: t.data.clone(),
    })
}

fn squeeze(node: &Node, t: &Tensor) -> Result<Tensor> {
    let dims: Dims = match node.attrs.get("axes") {
        Some(Attribute::Ints(a)) => {
            let drop: Vec<usize> = a.iter().map(|&x| norm_axis(x, t.dims.len())).collect();
            t.dims
                .iter()
                .enumerate()
                .filter(|(i, _)| !drop.contains(i))
                .map(|(_, &d)| d)
                .collect()
        }
        _ => t.dims.iter().copied().filter(|&d| d != 1).collect(),
    };
    Ok(Tensor {
        dims,
        data: t.data.clone(),
    })
}

fn concat(node: &Node, ins: &[&Tensor]) -> Result<Tensor> {
    let axis = norm_axis(
        int_attr(node, "axis").context("Concat axis")?,
        ins[0].dims.len(),
    );
    let hosts: Vec<Vec<f32>> = ins.iter().map(|t| t.as_f32()).collect();
    let pairs: Vec<(&[f32], &[i64])> = hosts
        .iter()
        .zip(ins)
        .map(|(h, t)| (h.as_slice(), t.dims.as_slice()))
        .collect();
    let (out, dims) = crate::layout::concat(&pairs, axis)?;
    Ok(Tensor::f32(dims, out))
}

fn split(node: &Node, t: &Tensor) -> Result<Vec<Tensor>> {
    let axis = norm_axis(int_attr(node, "axis").unwrap_or(0), t.dims.len());
    let n = node.outputs.iter().filter(|o| !o.is_empty()).count() as i64;
    let sizes: Dims = match node.attrs.get("split") {
        Some(Attribute::Ints(s)) => s.clone(),
        _ => vec![t.dims[axis] / n; n as usize],
    };
    let parts = crate::layout::split(&t.as_f32(), &t.dims, axis, &sizes)?;
    Ok(parts
        .into_iter()
        .map(|(d, dims)| Tensor::f32(dims, d))
        .collect())
}

fn gather(node: &Node, data: &Tensor, idx: &Tensor) -> Result<Tensor> {
    let axis = norm_axis(int_attr(node, "axis").unwrap_or(0), data.dims.len());
    let (out, dims) =
        crate::layout::gather(&data.as_f32(), &data.dims, &idx.as_i64(), &idx.dims, axis)?;
    Ok(Tensor::f32(dims, out))
}

fn slice(ins: &[&Tensor]) -> Result<Tensor> {
    let data = ins[0];
    let starts = ins[1].as_i64();
    let ends = ins[2].as_i64();
    let rank = data.dims.len();
    let axes: Vec<usize> = match ins.get(3) {
        Some(v) => v.as_i64().iter().map(|&a| norm_axis(a, rank)).collect(),
        None => (0..starts.len()).collect(),
    };
    let steps: Vec<i64> = ins
        .get(4)
        .map(|v| v.as_i64())
        .unwrap_or_else(|| vec![1; starts.len()]);

    let mut sel: Vec<Vec<i64>> = data.dims.iter().map(|&d| (0..d).collect()).collect();
    for (i, &ax) in axes.iter().enumerate() {
        let dim = data.dims[ax];
        let step = steps[i];
        let norm = |x: i64| if x < 0 { x + dim } else { x };
        let mut idxs = Vec::new();
        if step > 0 {
            let (s, e) = (norm(starts[i]).clamp(0, dim), norm(ends[i]).clamp(0, dim));
            let mut x = s;
            while x < e {
                idxs.push(x);
                x += step;
            }
        } else {
            let (s, e) = (
                norm(starts[i]).clamp(0, dim - 1),
                norm(ends[i]).clamp(-1, dim - 1),
            );
            let mut x = s;
            while x > e {
                idxs.push(x);
                x += step;
            }
        }
        sel[ax] = idxs;
    }

    let dims: Dims = sel.iter().map(|s| s.len() as i64).collect();
    let v = data.as_f32();
    let in_strides = row_major_strides(&data.dims);
    let out_strides = row_major_strides(&dims);
    let n: usize = dims.iter().product::<i64>() as usize;
    let mut out = vec![0.0f32; n];
    for (lin, slot) in out.iter_mut().enumerate() {
        let mut rem = lin;
        let mut src = 0usize;
        for k in 0..rank {
            let c = rem / out_strides[k] as usize;
            rem %= out_strides[k] as usize;
            src += sel[k][c] as usize * in_strides[k] as usize;
        }
        *slot = v[src];
    }
    Ok(Tensor::f32(dims, out))
}

// ---- attribute helpers ----

fn norm_axis(axis: i64, rank: usize) -> usize {
    if axis < 0 {
        (axis + rank as i64) as usize
    } else {
        axis as usize
    }
}

fn int_attr(node: &Node, name: &str) -> Option<i64> {
    match node.attrs.get(name) {
        Some(Attribute::Int(i)) => Some(*i),
        _ => None,
    }
}

fn float_attr(node: &Node, name: &str) -> Option<f32> {
    match node.attrs.get(name) {
        Some(Attribute::Float(f)) => Some(*f),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_sub_over_last_axis() {
        // [2,3] minus [2,1], LayerNorm centering.
        let x = Tensor::f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m = Tensor::f32(vec![2, 1], vec![2.0, 5.0]);
        let r = broadcast_bin(&x, &m, |a, b| a - b).unwrap();
        assert_eq!(r.as_f32(), vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn reduce_mean_last_axis_keepdims() {
        let x = Tensor::f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let node = Node {
            name: "m".into(),
            op_type: "ReduceMean".into(),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            attrs: [("axes".to_string(), Attribute::Ints(vec![-1]))]
                .into_iter()
                .collect(),
        };
        let r = reduce_mean(&node, &x).unwrap();
        assert_eq!(r.dims, vec![2, 1]);
        assert_eq!(r.as_f32(), vec![2.0, 5.0]);
    }

    #[test]
    fn matmul_2d_and_batched() {
        let mm = Node {
            name: "mm".into(),
            op_type: "MatMul".into(),
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["y".into()],
            attrs: HashMap::new(),
        };
        let w = HashMap::new();
        let a = Tensor::f32(vec![2, 3], (0..6).map(|x| x as f32).collect());
        let b = Tensor::f32(vec![3, 2], (0..6).map(|x| x as f32).collect());
        let c = matmul(&mm, &a, &b, &w, &HostBackend).unwrap();
        assert_eq!(c.dims, vec![2, 2]);
        // [[0,1,2],[3,4,5]] @ [[0,1],[2,3],[4,5]] = [[10,13],[28,40]]
        assert_eq!(c.as_f32(), vec![10.0, 13.0, 28.0, 40.0]);

        // Batched [2,1,3] by [2,3,1] is [2,1,1].
        let a = Tensor::f32(vec![2, 1, 3], (0..6).map(|x| x as f32).collect());
        let b = Tensor::f32(vec![2, 3, 1], (0..6).map(|x| x as f32).collect());
        let c = matmul(&mm, &a, &b, &w, &HostBackend).unwrap();
        assert_eq!(c.dims, vec![2, 1, 1]);
        assert_eq!(c.as_f32(), vec![5.0, 50.0]); // 0+1+4 ; 9+16+25
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let x = Tensor::f32(vec![2, 3], vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]);
        let node = Node {
            name: "s".into(),
            op_type: "Softmax".into(),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            attrs: [("axis".to_string(), Attribute::Int(-1))]
                .into_iter()
                .collect(),
        };
        let r = softmax(&node, &x).unwrap();
        let v = r.as_f32();
        assert!((v[0] + v[1] + v[2] - 1.0).abs() < 1e-6);
        assert!((v[3] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_normalizes_and_affines() {
        // Row [1,2,3]: mean 2, var 2/3, normalized to [-1.2247, 0, 1.2247],
        // then scale 2 and bias 1.
        let x = Tensor::f32(vec![1, 3], vec![1.0, 2.0, 3.0]);
        let scale = Tensor::f32(vec![3], vec![2.0, 2.0, 2.0]);
        let bias = Tensor::f32(vec![3], vec![1.0, 1.0, 1.0]);
        let node = Node {
            name: "ln".into(),
            op_type: "LayerNormalization".into(),
            inputs: vec!["x".into(), "s".into(), "b".into()],
            outputs: vec!["y".into()],
            attrs: [("epsilon".to_string(), Attribute::Float(0.0))]
                .into_iter()
                .collect(),
        };
        let r = layer_norm(&node, &x, &scale, &bias, &HostBackend).unwrap();
        let v = r.to_f32();
        let inv = 1.0 / (2.0f32 / 3.0).sqrt();
        assert!((v[0] - (-inv * 2.0 + 1.0)).abs() < 1e-5);
        assert!((v[1] - 1.0).abs() < 1e-5); // middle normalizes to 0 -> bias
        assert!((v[2] - (inv * 2.0 + 1.0)).abs() < 1e-5);
    }

    #[test]
    fn gemm_with_transb_and_bias() {
        // A[2,3] @ B[4,3]^T + c[4], a Conv1D-style GPT-2 projection.
        let a = Tensor::f32(vec![2, 3], (0..6).map(|x| x as f32).collect());
        let b = Tensor::f32(vec![4, 3], (0..12).map(|x| x as f32).collect());
        let c = Tensor::f32(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
        let node = Node {
            name: "g".into(),
            op_type: "Gemm".into(),
            inputs: vec!["a".into(), "b".into(), "c".into()],
            outputs: vec!["y".into()],
            attrs: [("transB".to_string(), Attribute::Int(1))]
                .into_iter()
                .collect(),
        };
        let r = gemm(&node, &[&a, &b, &c], &HashMap::new(), &HostBackend).unwrap();
        assert_eq!(r.dims, vec![2, 4]);
        // row0 [0,1,2] . rows of B: [0,1,2]->5, [3,4,5]->14, [6,7,8]->23, [9,10,11]->32; +1
        assert_eq!(r.as_f32()[..4], [6.0, 15.0, 24.0, 33.0]);
    }
}
