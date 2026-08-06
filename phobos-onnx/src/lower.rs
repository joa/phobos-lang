use anyhow::{Result, bail};

use crate::ir::Node;
use crate::shape::Dims;

/// Threads per CTA, each kernel's `@launch`.
const BLOCK_THREADS: u32 = 256;
/// The flattened elementwise kernels' `BLOCK` tile.
const ELEM_TILE: i64 = 256;
/// Output tile and k-slice for the plain f32 matmul.
const TILE_M: i64 = 64;
const TILE_N: i64 = 64;
const TILE_K: i64 = 16;
/// Rows per CTA for the row-tiled 2-D ops. The largest that divides the row
/// count is used.
const ROW_TILE_CHOICES: [i64; 5] = [16, 8, 4, 2, 1];
/// Used when a LayerNorm node carries no `epsilon` attribute.
const DEFAULT_LN_EPS: f32 = 1e-5;
/// The logistic GELU approximation is `x * sigmoid(1.702 x)`.
const GELU_SIGMOID_COEFF: f32 = 1.702;

/// A kernel parameter, in declaration order.
#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    /// A tensor edge as a memref descriptor over `view`, the dims the kernel
    /// sees, which may be a flattened view of the edge's logical shape.
    Tensor { edge: String, view: Dims },
    /// An immediate scalar.
    ScalarF32(f32),
}

/// Everything the runner needs to compile and launch one node's kernel: the
/// Phobos source, the autotune dims to pin, the grid and block, and the
/// parameters the runner turns into a memref-descriptor ABI call.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelPlan {
    /// As declared in the source, and the symbol to look up.
    pub kernel_name: String,
    pub source: String,
    /// Autotune dims to pin in the compile context, as name and value.
    pub overrides: Vec<(String, i64)>,
    pub block: u32,
    pub grid: (u32, u32, u32),
    pub params: Vec<Param>,
    /// The single output edge this kernel writes.
    pub output: String,
}

/// Lower one node given the resolved static dims of every edge.
pub fn lower_node(node: &Node, dims: &dyn Fn(&str) -> Option<Dims>) -> Result<KernelPlan> {
    let get = |edge: &str| -> Result<Dims> {
        dims(edge).ok_or_else(|| anyhow::anyhow!("edge '{edge}' has no resolved shape"))
    };

    match node.op_type.as_str() {
        "MatMul" => {
            let (a, b, out) = binary_io(node)?;
            lower_matmul(a, b, out, &get(a)?, &get(b)?)
        }
        "Add" => lower_add(node, &get),
        "Sub" => lower_elementwise_binary(node, "sub", "-", &get),
        "Mul" => lower_elementwise_binary(node, "mul", "*", &get),
        "Div" => lower_elementwise_binary(node, "div", "/", &get),
        "Relu" => lower_relu(node, &get),
        "Gelu" => lower_gelu(node, &get),
        "Softmax" => lower_softmax(node, &get),
        "LayerNormalization" => lower_layernorm(node, &get),
        // Fused ops, emitted by crate::transform.
        "PhobosFusedLinear" => lower_fused_linear(node, &get),
        "PhobosFlashAttention" => lower_flash_attention(node, &get),
        other => bail!("lowering does not support op '{other}' yet"),
    }
}

fn lower_matmul(a: &str, b: &str, out: &str, ad: &Dims, bd: &Dims) -> Result<KernelPlan> {
    if ad.len() != 2 || bd.len() != 2 || ad[1] != bd[0] {
        bail!("MatMul expects 2-D [M,K] x [K,N], got {ad:?} x {bd:?}");
    }
    let (m, n) = (ad[0], bd[1]);
    let grid = (tiles(m, TILE_M, "M")?, tiles(n, TILE_N, "N")?, 1);

    let source = format!(
        "@launch({BLOCK_THREADS})\n\
         @autotune(TILE_M in [{TILE_M}], TILE_N in [{TILE_N}], TILE_K in [{TILE_K}])\n\
         kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {{\n\
         \x20 let pm = program_id(0)\n\
         \x20 let pn = program_id(1)\n\
         \x20 var acc: tile<f32>[TILE_M, TILE_N] = 0.0\n\
         \x20 for kt in range(0, K, TILE_K) {{\n\
         \x20   var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]\n\
         \x20   var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]\n\
         \x20   acc += dot(a, b)\n\
         \x20 }}\n\
         \x20 C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc\n\
         }}\n"
    );

    Ok(KernelPlan {
        kernel_name: "matmul".into(),
        source,
        overrides: vec![
            ("TILE_M".into(), TILE_M),
            ("TILE_N".into(), TILE_N),
            ("TILE_K".into(), TILE_K),
        ],
        block: BLOCK_THREADS,
        grid,
        params: vec![
            Param::Tensor {
                edge: a.into(),
                view: ad.clone(),
            },
            Param::Tensor {
                edge: b.into(),
                view: bd.clone(),
            },
            Param::Tensor {
                edge: out.into(),
                view: vec![m, n],
            },
        ],
        output: out.into(),
    })
}

fn lower_elementwise_binary(
    node: &Node,
    kernel_name: &str,
    op: &str,
    get: &dyn Fn(&str) -> Result<Dims>,
) -> Result<KernelPlan> {
    let (a, b, out) = binary_io(node)?;
    let (ad, bd) = (get(a)?, get(b)?);
    if ad != bd {
        bail!("{kernel_name} needs matching shapes (no broadcast yet), got {ad:?} vs {bd:?}");
    }
    let n = numel(&ad);
    let grid = (tiles(n, ELEM_TILE, "N")?, 1, 1);

    let source = format!(
        "@launch({BLOCK_THREADS})\n\
         @autotune(BLOCK in [{ELEM_TILE}])\n\
         kernel {kernel_name}(A: tensor<f32>[N], B: tensor<f32>[N], Y: tensor<f32>[N]) {{\n\
         \x20 let base = program_id(0) * BLOCK\n\
         \x20 Y[base :+ BLOCK] = A[base :+ BLOCK] {op} B[base :+ BLOCK]\n\
         }}\n"
    );

    Ok(KernelPlan {
        kernel_name: kernel_name.into(),
        source,
        overrides: vec![("BLOCK".into(), ELEM_TILE)],
        block: BLOCK_THREADS,
        grid,
        params: vec![
            Param::Tensor {
                edge: a.into(),
                view: vec![n],
            },
            Param::Tensor {
                edge: b.into(),
                view: vec![n],
            },
            Param::Tensor {
                edge: out.into(),
                view: vec![n],
            },
        ],
        output: out.into(),
    })
}

fn lower_relu(node: &Node, get: &dyn Fn(&str) -> Result<Dims>) -> Result<KernelPlan> {
    let (x, out) = unary_io(node)?;
    let xd = get(x)?;
    let n = numel(&xd);
    let grid = (tiles(n, ELEM_TILE, "N")?, 1, 1);

    // tmax needs two tiles, so this maxes against a zero tile, not a scalar.
    let source = format!(
        "@launch({BLOCK_THREADS})\n\
         @autotune(BLOCK in [{ELEM_TILE}])\n\
         kernel relu(X: tensor<f32>[N], Y: tensor<f32>[N]) {{\n\
         \x20 let base = program_id(0) * BLOCK\n\
         \x20 var zero: tile<f32>[BLOCK] = 0.0\n\
         \x20 Y[base :+ BLOCK] = tmax(X[base :+ BLOCK], zero)\n\
         }}\n"
    );

    Ok(KernelPlan {
        kernel_name: "relu".into(),
        source,
        overrides: vec![("BLOCK".into(), ELEM_TILE)],
        block: BLOCK_THREADS,
        grid,
        params: vec![
            Param::Tensor {
                edge: x.into(),
                view: vec![n],
            },
            Param::Tensor {
                edge: out.into(),
                view: vec![n],
            },
        ],
        output: out.into(),
    })
}

/// Output tile for the fused matmul-epilogue kernel. Smaller than the plain
/// matmul's 64x64 because the epilogue holds an extra full tile, and two 64x64
/// f32 tiles plus GEMM staging overflow sm_75's 48 KB shared budget.
const FUSED_TILE: i64 = 32;
/// Flash-attention block sizes, rows of Q per CTA and keys per step. The
/// largest that divides the sequence length is used, capped at 32 so the score
/// and prob tiles fit the 48 KB shared budget on older GPUs.
const FLASH_TILE_CHOICES: [i64; 4] = [32, 16, 8, 4];

/// `C = act(A @ B + bias)` with `bias` a row vector, from
/// [`crate::transform`].
fn lower_fused_linear(node: &Node, get: &dyn Fn(&str) -> Result<Dims>) -> Result<KernelPlan> {
    let (a, b, bias, out) = match (node.inputs.as_slice(), node.outputs.as_slice()) {
        ([a, b, bias], [out]) => (a, b, bias, out),
        _ => bail!("FusedLinear expects inputs (A, B, bias) and one output"),
    };
    let (ad, bd) = (get(a)?, get(b)?);
    if ad.len() != 2 || bd.len() != 2 || ad[1] != bd[0] {
        bail!("FusedLinear expects 2-D [M,K] x [K,N], got {ad:?} x {bd:?}");
    }
    let (m, n) = (ad[0], bd[1]);
    let grid = (tiles(m, FUSED_TILE, "M")?, tiles(n, FUSED_TILE, "N")?, 1);

    let activation = match node.attrs.get("activation") {
        Some(crate::ir::Attribute::String(s)) => s.as_str(),
        _ => "none",
    };
    // The bias is added into `acc` in place and the activation applied over
    // as few extra tiles as possible: the [TILE_M, TILE_N] tiles dominate the
    // shared budget.
    let store = "C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N]";
    let epilogue = match activation {
        "none" => format!("  {store} = acc\n"),
        "relu" => format!(
            "  var zero: tile<f32>[TILE_M, TILE_N] = 0.0\n\
             \x20 {store} = tmax(acc, zero)\n"
        ),
        "gelu" => format!(
            "  var d: tile<f32>[TILE_M, TILE_N] = 1.0 + exp(-{GELU_SIGMOID_COEFF} * acc)\n\
             \x20 {store} = acc / d\n"
        ),
        other => bail!("FusedLinear has unknown activation '{other}'"),
    };

    let source = format!(
        "@launch({BLOCK_THREADS})\n\
         @autotune(TILE_M in [{FUSED_TILE}], TILE_N in [{FUSED_TILE}], TILE_K in [{TILE_K}])\n\
         kernel fused_linear(A: tensor<f32>[M, K], B: tensor<f32>[K, N], Bias: tensor<f32>[1, N], C: tensor<f32>[M, N]) {{\n\
         \x20 let pm = program_id(0)\n\
         \x20 let pn = program_id(1)\n\
         \x20 var acc: tile<f32>[TILE_M, TILE_N] = 0.0\n\
         \x20 for kt in range(0, K, TILE_K) {{\n\
         \x20   var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]\n\
         \x20   var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]\n\
         \x20   acc += dot(a, b)\n\
         \x20 }}\n\
         \x20 var bias: tile<f32>[1, TILE_N] = Bias[0 :+ 1, pn * TILE_N :+ TILE_N]\n\
         \x20 acc = acc + bias\n\
         {epilogue}\
         }}\n"
    );

    Ok(KernelPlan {
        kernel_name: "fused_linear".into(),
        source,
        overrides: vec![
            ("TILE_M".into(), FUSED_TILE),
            ("TILE_N".into(), FUSED_TILE),
            ("TILE_K".into(), TILE_K),
        ],
        block: BLOCK_THREADS,
        grid,
        params: vec![
            Param::Tensor {
                edge: a.into(),
                view: ad.clone(),
            },
            Param::Tensor {
                edge: b.into(),
                view: bd.clone(),
            },
            Param::Tensor {
                edge: bias.into(),
                view: vec![1, n],
            },
            Param::Tensor {
                edge: out.into(),
                view: vec![m, n],
            },
        ],
        output: out.into(),
    })
}

/// `O = softmax(scale * Q @ K^T) @ V` from [`crate::transform`]: one CTA per
/// BR-row block, streaming the keys in BC-sized steps with an online softmax.
/// The SPEC's flash-attention kernel with head dim D and the block sizes baked
/// in.
fn lower_flash_attention(node: &Node, get: &dyn Fn(&str) -> Result<Dims>) -> Result<KernelPlan> {
    let (q, k, v, out) = match (node.inputs.as_slice(), node.outputs.as_slice()) {
        ([q, k, v], [out]) => (q, k, v, out),
        _ => bail!("FlashAttention expects inputs (Q, K, V) and one output"),
    };
    let (qd, kd, vd) = (get(q)?, get(k)?, get(v)?);
    if qd.len() != 2 || kd.len() != 2 || vd.len() != 2 {
        bail!("FlashAttention expects 2-D Q/K/V (single head), got {qd:?} {kd:?} {vd:?}");
    }
    let (nq, d) = (qd[0], qd[1]);
    let (nk, dk) = (kd[0], kd[1]);
    if dk != d || vd[0] != nk || vd[1] != d {
        bail!("FlashAttention shape mismatch: Q{qd:?} K{kd:?} V{vd:?}");
    }
    let br = pick_tile(nq, "Nq")?;
    let bc = pick_tile(nk, "Nk")?;
    let scale = match node.attrs.get("scale") {
        Some(crate::ir::Attribute::Float(s)) => *s,
        _ => 1.0,
    };

    let source = format!(
        "@launch({BLOCK_THREADS})\n\
         @autotune(D in [{d}], BR in [{br}], BC in [{bc}])\n\
         kernel flash_attention(Q: tensor<f32>[Nq, D], K: tensor<f32>[Nk, D], V: tensor<f32>[Nk, D], O: tensor<f32>[Nq, D], scale: f32) {{\n\
         \x20 let pid = program_id(0)\n\
         \x20 let row = pid * BR\n\
         \x20 let q = Q[row :+ BR, :]\n\
         \x20 var acc: tile<f32>[BR, D] = 0.0\n\
         \x20 var m: tile<f32>[BR, 1] = -999999999.0\n\
         \x20 var l: tile<f32>[BR, 1] = 0.0\n\
         \x20 for kt in range(0, Nk, BC) {{\n\
         \x20   let k = K[kt :+ BC, :]\n\
         \x20   let v = V[kt :+ BC, :]\n\
         \x20   var s: tile<f32>[BR, BC] = dot_t(q, k)\n\
         \x20   s = s * scale\n\
         \x20   var mnew: tile<f32>[BR, 1] = rowmax(s)\n\
         \x20   mnew = tmax(m, mnew)\n\
         \x20   var p: tile<f32>[BR, BC] = exp(s - mnew)\n\
         \x20   var corr: tile<f32>[BR, 1] = exp(m - mnew)\n\
         \x20   l = l * corr\n\
         \x20   l += rowsum(p)\n\
         \x20   acc = acc * corr\n\
         \x20   acc += dot(p, v)\n\
         \x20   m = mnew\n\
         \x20 }}\n\
         \x20 acc = acc / l\n\
         \x20 O[row :+ BR, :] = acc\n\
         }}\n"
    );

    Ok(KernelPlan {
        kernel_name: "flash_attention".into(),
        source,
        overrides: vec![("D".into(), d), ("BR".into(), br), ("BC".into(), bc)],
        block: BLOCK_THREADS,
        grid: ((nq / br) as u32, 1, 1),
        params: vec![
            Param::Tensor {
                edge: q.into(),
                view: vec![nq, d],
            },
            Param::Tensor {
                edge: k.into(),
                view: vec![nk, d],
            },
            Param::Tensor {
                edge: v.into(),
                view: vec![nk, d],
            },
            Param::Tensor {
                edge: out.into(),
                view: vec![nq, d],
            },
            Param::ScalarF32(scale),
        ],
        output: out.into(),
    })
}

/// Largest flash block size dividing `n`.
fn pick_tile(n: i64, axis: &str) -> Result<i64> {
    FLASH_TILE_CHOICES
        .into_iter()
        .find(|t| n % t == 0)
        .ok_or_else(|| anyhow::anyhow!("{axis}={n} is not divisible by any flash block size"))
}

/// `gelu(x) = x / (1 + exp(-1.702 x))`, the logistic approximation. Flattened
/// elementwise like Relu.
fn lower_gelu(node: &Node, get: &dyn Fn(&str) -> Result<Dims>) -> Result<KernelPlan> {
    let (x, out) = unary_io(node)?;
    let n = numel(&get(x)?);
    let grid = (tiles(n, ELEM_TILE, "N")?, 1, 1);
    let c = GELU_SIGMOID_COEFF;

    let source = format!(
        "@launch({BLOCK_THREADS})\n\
         @autotune(BLOCK in [{ELEM_TILE}])\n\
         kernel gelu(X: tensor<f32>[N], Y: tensor<f32>[N]) {{\n\
         \x20 let base = program_id(0) * BLOCK\n\
         \x20 var x: tile<f32>[BLOCK] = X[base :+ BLOCK]\n\
         \x20 var d: tile<f32>[BLOCK] = 1.0 + exp(-{c} * x)\n\
         \x20 Y[base :+ BLOCK] = x / d\n\
         }}\n"
    );

    Ok(KernelPlan {
        kernel_name: "gelu".into(),
        source,
        overrides: vec![("BLOCK".into(), ELEM_TILE)],
        block: BLOCK_THREADS,
        grid,
        params: vec![
            Param::Tensor {
                edge: x.into(),
                view: vec![n],
            },
            Param::Tensor {
                edge: out.into(),
                view: vec![n],
            },
        ],
        output: out.into(),
    })
}

/// Matching shapes lower to the flattened elementwise kernel; a bias row,
/// `[.., W] + [W]` or `[.., W] + [1, W]`, to a row-tiled broadcast kernel.
fn lower_add(node: &Node, get: &dyn Fn(&str) -> Result<Dims>) -> Result<KernelPlan> {
    let (a, b, out) = binary_io(node)?;
    let (ad, bd) = (get(a)?, get(b)?);
    if ad == bd {
        return lower_elementwise_binary(node, "add", "+", get);
    }
    match bias_width(&ad, &bd) {
        Some(w) => lower_add_bias(a, b, out, &ad, w),
        None => {
            bail!("Add: only same-shape or bias-row broadcast is supported, got {ad:?} + {bd:?}")
        }
    }
}

/// The width when `bd` is a bias vector broadcasting over the last axis of
/// `ad`, `[W]` or `[1, W]` against `ad`'s trailing `W`.
fn bias_width(ad: &Dims, bd: &Dims) -> Option<i64> {
    let w = *ad.last()?;
    let is_bias =
        matches!(bd.as_slice(), [bw] if *bw == w) || matches!(bd.as_slice(), [1, bw] if *bw == w);
    is_bias.then_some(w)
}

fn lower_add_bias(a: &str, b: &str, out: &str, ad: &Dims, w: i64) -> Result<KernelPlan> {
    let (rows, tr) = row_tiling(ad, w)?;
    let source = format!(
        "@launch({BLOCK_THREADS})\n\
         kernel add_bias(A: tensor<f32>[M, {w}], B: tensor<f32>[1, {w}], Y: tensor<f32>[M, {w}]) {{\n\
         \x20 let row = program_id(0) * {tr}\n\
         \x20 var a: tile<f32>[{tr}, {w}] = A[row :+ {tr}, 0 :+ {w}]\n\
         \x20 var bias: tile<f32>[1, {w}] = B[0 :+ 1, 0 :+ {w}]\n\
         \x20 var y: tile<f32>[{tr}, {w}] = a + bias\n\
         \x20 Y[row :+ {tr}, 0 :+ {w}] = y\n\
         }}\n"
    );
    Ok(KernelPlan {
        kernel_name: "add_bias".into(),
        source,
        overrides: vec![],
        block: BLOCK_THREADS,
        grid: ((rows / tr) as u32, 1, 1),
        params: vec![
            Param::Tensor {
                edge: a.into(),
                view: vec![rows, w],
            },
            Param::Tensor {
                edge: b.into(),
                view: vec![1, w],
            },
            Param::Tensor {
                edge: out.into(),
                view: vec![rows, w],
            },
        ],
        output: out.into(),
    })
}

/// `y = exp(x - rowmax) / rowsum(...)` over the last axis.
fn lower_softmax(node: &Node, get: &dyn Fn(&str) -> Result<Dims>) -> Result<KernelPlan> {
    let (x, out) = unary_io(node)?;
    let xd = get(x)?;
    let w = *xd
        .last()
        .ok_or_else(|| anyhow::anyhow!("Softmax input is a scalar"))?;
    let (rows, tr) = row_tiling(&xd, w)?;

    let source = format!(
        "@launch({BLOCK_THREADS})\n\
         kernel softmax(X: tensor<f32>[M, {w}], Y: tensor<f32>[M, {w}]) {{\n\
         \x20 let row = program_id(0) * {tr}\n\
         \x20 var x: tile<f32>[{tr}, {w}] = X[row :+ {tr}, 0 :+ {w}]\n\
         \x20 var m: tile<f32>[{tr}, 1] = rowmax(x)\n\
         \x20 var e: tile<f32>[{tr}, {w}] = exp(x - m)\n\
         \x20 var s: tile<f32>[{tr}, 1] = rowsum(e)\n\
         \x20 var y: tile<f32>[{tr}, {w}] = e / s\n\
         \x20 Y[row :+ {tr}, 0 :+ {w}] = y\n\
         }}\n"
    );
    Ok(KernelPlan {
        kernel_name: "softmax".into(),
        source,
        overrides: vec![],
        block: BLOCK_THREADS,
        grid: ((rows / tr) as u32, 1, 1),
        params: vec![
            Param::Tensor {
                edge: x.into(),
                view: vec![rows, w],
            },
            Param::Tensor {
                edge: out.into(),
                view: vec![rows, w],
            },
        ],
        output: out.into(),
    })
}

/// `y = (x - mean) / sqrt(var + eps) * gamma + beta` over the last axis.
fn lower_layernorm(node: &Node, get: &dyn Fn(&str) -> Result<Dims>) -> Result<KernelPlan> {
    let (x, gamma, beta, out) = match (node.inputs.as_slice(), node.outputs.as_slice()) {
        ([x, g, b, ..], [out, ..]) => (x, g, b, out),
        _ => bail!("LayerNormalization expects inputs (X, scale, bias) and an output"),
    };
    let xd = get(x)?;
    let w = *xd
        .last()
        .ok_or_else(|| anyhow::anyhow!("LayerNorm input is a scalar"))?;
    let (rows, tr) = row_tiling(&xd, w)?;
    let wf = format!("{}.0", w); // float divisor for the mean/variance
    let eps = layernorm_eps(node);

    let source = format!(
        "@launch({BLOCK_THREADS})\n\
         kernel layernorm(X: tensor<f32>[M, {w}], G: tensor<f32>[1, {w}], B: tensor<f32>[1, {w}], Y: tensor<f32>[M, {w}]) {{\n\
         \x20 let row = program_id(0) * {tr}\n\
         \x20 var x: tile<f32>[{tr}, {w}] = X[row :+ {tr}, 0 :+ {w}]\n\
         \x20 var s: tile<f32>[{tr}, 1] = rowsum(x)\n\
         \x20 var mu: tile<f32>[{tr}, 1] = s / {wf}\n\
         \x20 var xc: tile<f32>[{tr}, {w}] = x - mu\n\
         \x20 var sq: tile<f32>[{tr}, {w}] = xc * xc\n\
         \x20 var vs: tile<f32>[{tr}, 1] = rowsum(sq)\n\
         \x20 var vv: tile<f32>[{tr}, 1] = vs / {wf}\n\
         \x20 var sd: tile<f32>[{tr}, 1] = sqrt(vv + {eps})\n\
         \x20 var nrm: tile<f32>[{tr}, {w}] = xc / sd\n\
         \x20 var g: tile<f32>[1, {w}] = G[0 :+ 1, 0 :+ {w}]\n\
         \x20 var gg: tile<f32>[{tr}, {w}] = nrm * g\n\
         \x20 var bb: tile<f32>[1, {w}] = B[0 :+ 1, 0 :+ {w}]\n\
         \x20 var y: tile<f32>[{tr}, {w}] = gg + bb\n\
         \x20 Y[row :+ {tr}, 0 :+ {w}] = y\n\
         }}\n"
    );
    Ok(KernelPlan {
        kernel_name: "layernorm".into(),
        source,
        overrides: vec![],
        block: BLOCK_THREADS,
        grid: ((rows / tr) as u32, 1, 1),
        params: vec![
            Param::Tensor {
                edge: x.into(),
                view: vec![rows, w],
            },
            Param::Tensor {
                edge: gamma.into(),
                view: vec![1, w],
            },
            Param::Tensor {
                edge: beta.into(),
                view: vec![1, w],
            },
            Param::Tensor {
                edge: out.into(),
                view: vec![rows, w],
            },
        ],
        output: out.into(),
    })
}

/// The `epsilon` attribute as a source float literal.
fn layernorm_eps(node: &Node) -> String {
    let eps = match node.attrs.get("epsilon") {
        Some(crate::ir::Attribute::Float(f)) => *f,
        _ => DEFAULT_LN_EPS,
    };
    // The lexer does not accept scientific notation.
    format!("{eps:.8}")
}

/// Flatten `dims` to rows of `width`, its last axis, and pick the largest
/// row-tile dividing the row count.
fn row_tiling(dims: &Dims, width: i64) -> Result<(i64, i64)> {
    if dims.last() != Some(&width) {
        bail!("expected a tensor whose last axis is {width}, got {dims:?}");
    }
    // A rank-1 tensor has an empty product, so a single row of `width`.
    let rows: i64 = dims[..dims.len() - 1].iter().product();
    if rows == 0 {
        bail!("row count is zero for shape {dims:?}");
    }
    let tr = ROW_TILE_CHOICES
        .into_iter()
        .find(|t| rows % t == 0)
        .unwrap_or(1);
    Ok((rows, tr))
}

/// The flattened element count.
fn numel(dims: &Dims) -> i64 {
    dims.iter().product()
}

/// CTAs to cover `extent` in `tile`-sized steps. The division must be exact:
/// there is no tail masking yet.
fn tiles(extent: i64, tile: i64, axis: &str) -> Result<u32> {
    if extent <= 0 || extent % tile != 0 {
        bail!("{axis}={extent} is not a positive multiple of the tile {tile}");
    }
    Ok((extent / tile) as u32)
}

fn binary_io(node: &Node) -> Result<(&str, &str, &str)> {
    match (node.inputs.as_slice(), node.outputs.as_slice()) {
        ([a, b], [out]) => Ok((a, b, out)),
        _ => bail!(
            "{} expects 2 inputs and 1 output, got {} in / {} out",
            node.op_type,
            node.inputs.len(),
            node.outputs.len()
        ),
    }
}

fn unary_io(node: &Node) -> Result<(&str, &str)> {
    match (node.inputs.as_slice(), node.outputs.as_slice()) {
        ([x], [out]) => Ok((x, out)),
        _ => bail!(
            "{} expects 1 input and 1 output, got {} in / {} out",
            node.op_type,
            node.inputs.len(),
            node.outputs.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn node(op: &str, ins: &[&str], outs: &[&str]) -> Node {
        Node {
            name: format!("{op}_node"),
            op_type: op.into(),
            inputs: ins.iter().map(|s| s.to_string()).collect(),
            outputs: outs.iter().map(|s| s.to_string()).collect(),
            attrs: HashMap::new(),
        }
    }

    fn dims_fn(map: &HashMap<String, Dims>) -> impl Fn(&str) -> Option<Dims> + '_ {
        move |e: &str| map.get(e).cloned()
    }

    #[test]
    fn lowers_matmul_grid_and_params() {
        let dims = HashMap::from([
            ("X".to_string(), vec![128, 64]),
            ("W".to_string(), vec![64, 192]),
        ]);
        let n = node("MatMul", &["X", "W"], &["Y"]);
        let plan = lower_node(&n, &dims_fn(&dims)).unwrap();

        assert_eq!(plan.kernel_name, "matmul");
        // grid is (M/TILE_M, N/TILE_N), so (128/64, 192/64) is (2, 3).
        assert_eq!(plan.grid, (2, 3, 1));
        assert_eq!(plan.block, 256);
        assert_eq!(plan.output, "Y");
        // The A, B and C descriptors, in order.
        assert_eq!(
            plan.params[0],
            Param::Tensor {
                edge: "X".into(),
                view: vec![128, 64]
            }
        );
        assert_eq!(
            plan.params[2],
            Param::Tensor {
                edge: "Y".into(),
                view: vec![128, 192]
            }
        );
        assert!(plan.source.contains("kernel matmul"));
        assert!(plan.source.contains("acc += dot(a, b)"));
    }

    #[test]
    fn matmul_rejects_mismatched_k() {
        let dims = HashMap::from([
            ("X".to_string(), vec![128, 64]),
            ("W".to_string(), vec![32, 32]),
        ]);
        let n = node("MatMul", &["X", "W"], &["Y"]);
        assert!(lower_node(&n, &dims_fn(&dims)).is_err());
    }

    #[test]
    fn lowers_add_as_flattened_elementwise() {
        let dims = HashMap::from([
            ("A".to_string(), vec![64, 64]),
            ("B".to_string(), vec![64, 64]),
        ]);
        let n = node("Add", &["A", "B"], &["Y"]);
        let plan = lower_node(&n, &dims_fn(&dims)).unwrap();

        assert_eq!(plan.kernel_name, "add");
        // 4096 elements over a 256 tile is 16 blocks.
        assert_eq!(plan.grid, (16, 1, 1));
        // Tensors are viewed as rank-1.
        assert_eq!(
            plan.params[0],
            Param::Tensor {
                edge: "A".into(),
                view: vec![4096]
            }
        );
        assert!(plan.source.contains("A[base :+ BLOCK] + B[base :+ BLOCK]"));
        assert_eq!(plan.overrides, vec![("BLOCK".to_string(), 256)]);
    }

    #[test]
    fn each_arith_op_uses_its_symbol() {
        let dims = HashMap::from([("A".to_string(), vec![256]), ("B".to_string(), vec![256])]);
        for (op, sym, name) in [
            ("Add", "+", "add"),
            ("Sub", "-", "sub"),
            ("Mul", "*", "mul"),
            ("Div", "/", "div"),
        ] {
            let n = node(op, &["A", "B"], &["Y"]);
            let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
            assert_eq!(plan.kernel_name, name);
            assert!(
                plan.source
                    .contains(&format!("A[base :+ BLOCK] {sym} B[base :+ BLOCK]")),
                "op {op} should emit symbol {sym}"
            );
        }
    }

    #[test]
    fn add_rejects_shape_mismatch() {
        let dims = HashMap::from([("A".to_string(), vec![256]), ("B".to_string(), vec![128])]);
        let n = node("Add", &["A", "B"], &["Y"]);
        assert!(lower_node(&n, &dims_fn(&dims)).is_err());
    }

    #[test]
    fn lowers_relu_as_tmax_zero() {
        let dims = HashMap::from([("X".to_string(), vec![32, 32])]);
        let n = node("Relu", &["X"], &["Y"]);
        let plan = lower_node(&n, &dims_fn(&dims)).unwrap();

        assert_eq!(plan.kernel_name, "relu");
        assert_eq!(plan.grid, (4, 1, 1)); // 1024 / 256
        assert!(plan.source.contains("var zero: tile<f32>[BLOCK] = 0.0"));
        assert!(plan.source.contains("tmax(X[base :+ BLOCK], zero)"));
        assert_eq!(plan.params.len(), 2);
    }

    #[test]
    fn non_divisible_extent_is_rejected() {
        // 300 elements is not a multiple of the 256 tile.
        let dims = HashMap::from([("A".to_string(), vec![300]), ("B".to_string(), vec![300])]);
        let n = node("Add", &["A", "B"], &["Y"]);
        assert!(lower_node(&n, &dims_fn(&dims)).is_err());
    }

    #[test]
    fn unsupported_op_is_rejected() {
        let dims = HashMap::new();
        let n = node("Conv", &["X", "W"], &["Y"]);
        assert!(lower_node(&n, &dims_fn(&dims)).is_err());
    }

    #[test]
    fn lowers_bias_add_as_row_broadcast() {
        let dims = HashMap::from([
            ("A".to_string(), vec![128, 64]),
            ("b".to_string(), vec![64]),
        ]);
        let n = node("Add", &["A", "b"], &["Y"]);
        let plan = lower_node(&n, &dims_fn(&dims)).unwrap();

        assert_eq!(plan.kernel_name, "add_bias");
        // 128 rows over a 16 row-tile is 8 blocks.
        assert_eq!(plan.grid, (8, 1, 1));
        // The bias goes in as a [1, W] descriptor for the broadcast.
        assert_eq!(
            plan.params[1],
            Param::Tensor {
                edge: "b".into(),
                view: vec![1, 64]
            }
        );
        assert!(plan.source.contains("var y: tile<f32>[16, 64] = a + bias"));
    }

    #[test]
    fn lowers_gelu_as_logistic() {
        let dims = HashMap::from([("X".to_string(), vec![64, 64])]);
        let n = node("Gelu", &["X"], &["Y"]);
        let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
        assert_eq!(plan.kernel_name, "gelu");
        assert_eq!(plan.grid, (16, 1, 1)); // 4096 / 256
        assert!(plan.source.contains("1.0 + exp(-1.702 * x)"));
        assert!(plan.source.contains("Y[base :+ BLOCK] = x / d"));
    }

    #[test]
    fn lowers_softmax_rowwise() {
        let dims = HashMap::from([("X".to_string(), vec![32, 64])]);
        let n = node("Softmax", &["X"], &["Y"]);
        let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
        assert_eq!(plan.kernel_name, "softmax");
        assert_eq!(plan.grid, (2, 1, 1)); // 32 rows / 16
        assert!(plan.source.contains("rowmax(x)"));
        assert!(plan.source.contains("exp(x - m)"));
        assert!(plan.source.contains("var y: tile<f32>[16, 64] = e / s"));
    }

    #[test]
    fn lowers_fused_linear_with_gelu_epilogue() {
        let mut n = node("PhobosFusedLinear", &["A", "B", "bias"], &["Y"]);
        n.attrs.insert(
            "activation".into(),
            crate::ir::Attribute::String("gelu".into()),
        );
        let dims = HashMap::from([
            ("A".to_string(), vec![128, 64]),
            ("B".to_string(), vec![64, 128]),
        ]);
        let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
        assert_eq!(plan.kernel_name, "fused_linear");
        assert_eq!(plan.grid, (4, 4, 1)); // 128/32, 128/32
        assert!(plan.source.contains("acc = acc + bias"));
        assert!(plan.source.contains("1.0 + exp(-1.702 * acc)"));
        // A, B, the [1, N] bias, then C.
        assert_eq!(
            plan.params[2],
            Param::Tensor {
                edge: "bias".into(),
                view: vec![1, 128]
            }
        );
        assert_eq!(plan.params.len(), 4);
    }

    #[test]
    fn lowers_flash_attention() {
        let mut n = node("PhobosFlashAttention", &["Q", "K", "V"], &["O"]);
        n.attrs
            .insert("scale".into(), crate::ir::Attribute::Float(0.125));
        let dims = HashMap::from([
            ("Q".to_string(), vec![64, 32]),
            ("K".to_string(), vec![64, 32]),
            ("V".to_string(), vec![64, 32]),
        ]);
        let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
        assert_eq!(plan.kernel_name, "flash_attention");
        assert_eq!(plan.grid, (2, 1, 1)); // Nq=64, BR=32
        assert!(plan.source.contains("dot_t(q, k)"));
        assert!(plan.source.contains("acc += dot(p, v)"));
        // Q, K, V, O, then the scale.
        assert_eq!(plan.params.len(), 5);
        assert_eq!(plan.params[4], Param::ScalarF32(0.125));
        assert_eq!(
            plan.overrides,
            vec![("D".into(), 32), ("BR".into(), 32), ("BC".into(), 32)]
        );
    }

    #[test]
    fn lowers_layernorm_with_sqrt_and_eps() {
        let mut n = node("LayerNormalization", &["X", "G", "B"], &["Y"]);
        n.attrs
            .insert("epsilon".into(), crate::ir::Attribute::Float(1e-3));
        let dims = HashMap::from([
            ("X".to_string(), vec![48, 64]),
            ("G".to_string(), vec![64]),
            ("B".to_string(), vec![64]),
        ]);
        let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
        assert_eq!(plan.kernel_name, "layernorm");
        assert_eq!(plan.grid, (3, 1, 1)); // 48 rows / 16
        assert!(plan.source.contains("sqrt(vv + 0.00100000)"));
        assert!(plan.source.contains("var mu: tile<f32>[16, 1] = s / 64.0"));
        // The scale and bias come in as [1, W] descriptors.
        assert_eq!(
            plan.params[1],
            Param::Tensor {
                edge: "G".into(),
                view: vec![1, 64]
            }
        );
        assert_eq!(plan.params.len(), 4);
    }
}
