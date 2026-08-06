use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

use anyhow::{Context, Result, bail};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

use crate::abi::{self, KernelArg};
use crate::interp::MatmulBackend;
use crate::ir::{Graph, Node, TensorData};
use crate::layout;
use crate::lower::{self, Param};
use crate::shape::{self, Dims};

/// Output tile and k-slice for the GPU matmul backend. Operands are padded up
/// to these on the host, so the kernel needs no bounds checks.
const MM_TM: usize = 32;
const MM_TN: usize = 32;
const MM_TK: usize = 16;

/// The plain tiled f32 matmul the backend launches. The dims are padded, so
/// every tile is full.
const MATMUL_SRC: &str = "\
@launch(256)
@autotune(TILE_M in [32], TILE_N in [32], TILE_K in [16])
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TILE_M, TILE_N] = 0.0
  for kt in range(0, K, TILE_K) {
    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
  }
  C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}
";

/// A compiled LayerNorm kernel for one width, with its rows-per-CTA.
struct LnKernel {
    module: Module,
    tr: usize,
}

/// A Phobos GPU matmul backend for [`crate::interp`]. Compiles the tiled matmul
/// once, then pads each call's operands to tile-aligned dims, launches, and
/// slices out the real `[m, n]`. Zero-padding K contributes zero terms to the
/// dot, and the padded M and N rows and columns are discarded.
pub struct GpuBackend {
    stream: Stream,
    module: Module,
    /// Compiled lazily, one per width.
    ln_cache: RefCell<HashMap<usize, LnKernel>>,
    /// Constant matmul operands kept device-resident under the caller's stable
    /// key: uploaded once, reused every step.
    weight_cache: RefCell<HashMap<String, DeviceBuffer<f32>>>,
    /// Non-constant operands and results, pooled by exact element count. A
    /// decode step reuses the same handful of shapes every token, so after the
    /// first this hits every time; a cudaMalloc and cudaFree pair costs more
    /// than a small matmul kernel.
    scratch: RefCell<HashMap<usize, Vec<DeviceBuffer<f32>>>>,
    /// Must be the last field: Rust drops in declaration order, and every
    /// buffer, module and stream above has to be released while the context is
    /// still alive or teardown faults.
    _ctx: cust::context::Context,
}

impl GpuBackend {
    pub fn new() -> Result<Self> {
        let _ctx = cust::quick_init().context("initializing CUDA")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let mut ctx = phobos_base::context::Context::default();
        for (name, value) in [("TILE_M", MM_TM), ("TILE_N", MM_TN), ("TILE_K", MM_TK)] {
            ctx.shape_overrides.insert(name.to_string(), value as i64);
        }
        let ptx = phobos_lang::compile(&ctx, MATMUL_SRC).context("compiling matmul kernel")?;
        let module = Module::from_ptx(&ptx, &[]).context("loading matmul PTX")?;

        Ok(GpuBackend {
            stream,
            module,
            ln_cache: RefCell::new(HashMap::new()),
            weight_cache: RefCell::new(HashMap::new()),
            scratch: RefCell::new(HashMap::new()),
            _ctx,
        })
    }

    /// A device buffer of exactly `len` elements, from the pool when possible.
    /// The contents are undefined: every caller either copies the whole buffer
    /// from the host or has the kernel write all of it before reading back.
    fn take_scratch(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        if let Some(pooled) = self.scratch.borrow_mut().get_mut(&len).and_then(Vec::pop) {
            return Ok(pooled);
        }
        // SAFETY: no caller reads before writing, see above.
        Ok(unsafe { DeviceBuffer::uninitialized(len)? })
    }

    fn return_scratch(&self, buf: DeviceBuffer<f32>) {
        self.scratch
            .borrow_mut()
            .entry(buf.len())
            .or_default()
            .push(buf);
    }

    /// The LayerNorm kernel for width `w`, compiled and cached on first use.
    fn ensure_ln(&self, w: usize) -> Result<()> {
        if self.ln_cache.borrow().contains_key(&w) {
            return Ok(());
        }
        let (src, tr) = layernorm_src(w);
        let ctx = phobos_base::context::Context::default();
        let ptx = phobos_lang::compile(&ctx, &src).context("compiling layernorm kernel")?;
        let module = Module::from_ptx(&ptx, &[]).context("loading layernorm PTX")?;
        self.ln_cache
            .borrow_mut()
            .insert(w, LnKernel { module, tr });
        Ok(())
    }
}

impl MatmulBackend for GpuBackend {
    fn matmul(&self, a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Result<Vec<f32>> {
        self.matmul_cached(a, m, k, b, n, None)
    }

    fn matmul_cached(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        b_key: Option<&str>,
    ) -> Result<Vec<f32>> {
        // All three axes pad to whole tiles. Masking a ragged N makes the
        // tiled kernel miscompute, see `examples/mm_check.rs`.
        let (mp, kp, np) = (round_up(m, MM_TM), round_up(k, MM_TK), round_up(n, MM_TN));
        let a_pad = pad(a, m, k, mp, kp);

        let mut a_dev = self.take_scratch(a_pad.len())?;
        a_dev.copy_from(&a_pad)?;
        // Left undefined: the grid covers every tile of C, so the kernel
        // writes all of it before the read-back below.
        let c_dev = self.take_scratch(mp * np)?;

        // A keyed B is a constant weight: uploaded once and kept resident.
        // Otherwise it goes up locally.
        let cache_key = b_key.map(|key| format!("{key}:{kp}x{np}"));
        let mut cache_guard: Option<std::cell::Ref<HashMap<String, DeviceBuffer<f32>>>> = None;
        let mut local_b: Option<DeviceBuffer<f32>> = None;
        let b_ptr = if let Some(key) = &cache_key {
            if !self.weight_cache.borrow().contains_key(key) {
                let b_pad = pad(b, k, n, kp, np);
                self.weight_cache
                    .borrow_mut()
                    .insert(key.clone(), DeviceBuffer::from_slice(&b_pad)?);
            }
            let guard = self.weight_cache.borrow();
            let ptr = guard[key].as_device_ptr().as_raw();
            cache_guard = Some(guard);
            ptr
        } else {
            let b_pad = pad(b, k, n, kp, np);
            let buf = DeviceBuffer::from_slice(&b_pad)?;
            let ptr = buf.as_device_ptr().as_raw();
            local_b = Some(buf);
            ptr
        };

        let mut args: Vec<KernelArg> = Vec::new();
        abi::push_tensor_descriptor(
            &mut args,
            a_dev.as_device_ptr().as_raw(),
            &[mp as i64, kp as i64],
        );
        abi::push_tensor_descriptor(&mut args, b_ptr, &[kp as i64, np as i64]);
        abi::push_tensor_descriptor(
            &mut args,
            c_dev.as_device_ptr().as_raw(),
            &[mp as i64, np as i64],
        );
        let mut slots: Vec<u64> = args.iter().map(|a| a.slot()).collect();
        let raw: Vec<*mut c_void> = slots
            .iter_mut()
            .map(|s| s as *mut u64 as *mut c_void)
            .collect();

        let func = self.module.get_function("matmul")?;
        let grid = ((mp / MM_TM) as u32, (np / MM_TN) as u32, 1u32);
        // SAFETY: raw points into slots, which outlives the call, and the
        // layout matches phobos-mlir's exploded-memref ABI at index_bitwidth
        // 32. b_ptr's owner, cache_guard or local_b, lives past the launch.
        unsafe {
            self.stream
                .launch(&func, grid, (256u32, 1, 1), 0, &raw)
                .context("matmul launch")?;
        }
        self.stream.synchronize()?;
        drop(cache_guard);
        drop(local_b);

        // Only the first m rows are real, and reading back all mp of them
        // would move a whole tile of rows to use one on a single-row call.
        let mut c_pad = vec![0.0f32; m * np];
        c_dev.index(0..m * np).copy_to(&mut c_pad)?;
        self.return_scratch(a_dev);
        self.return_scratch(c_dev);

        if np == n {
            return Ok(c_pad);
        }
        // The real [m, n] block out of the [m, np] padded rows.
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            out[i * n..(i + 1) * n].copy_from_slice(&c_pad[i * np..i * np + n]);
        }
        Ok(out)
    }

    fn layer_norm(
        &self,
        x: &[f32],
        rows: usize,
        w: usize,
        scale: &[f32],
        bias: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>> {
        self.ensure_ln(w)?;
        let cache = self.ln_cache.borrow();
        let ln = &cache[&w];
        let rp = round_up(rows, ln.tr);
        let x_pad = pad(x, rows, w, rp, w); // pad rows only; padded rows are discarded

        let x_dev = DeviceBuffer::from_slice(&x_pad)?;
        let g_dev = DeviceBuffer::from_slice(scale)?;
        let b_dev = DeviceBuffer::from_slice(bias)?;
        let y_dev = DeviceBuffer::from_slice(&vec![0.0f32; rp * w])?;

        let mut args: Vec<KernelArg> = Vec::new();
        abi::push_tensor_descriptor(
            &mut args,
            x_dev.as_device_ptr().as_raw(),
            &[rp as i64, w as i64],
        );
        abi::push_tensor_descriptor(&mut args, g_dev.as_device_ptr().as_raw(), &[1, w as i64]);
        abi::push_tensor_descriptor(&mut args, b_dev.as_device_ptr().as_raw(), &[1, w as i64]);
        abi::push_tensor_descriptor(
            &mut args,
            y_dev.as_device_ptr().as_raw(),
            &[rp as i64, w as i64],
        );
        args.push(KernelArg::F32(eps));
        let mut slots: Vec<u64> = args.iter().map(|a| a.slot()).collect();
        let raw: Vec<*mut c_void> = slots
            .iter_mut()
            .map(|s| s as *mut u64 as *mut c_void)
            .collect();

        let func = ln.module.get_function("layernorm")?;
        let grid = ((rp / ln.tr) as u32, 1u32, 1u32);
        // SAFETY: raw points into slots, which outlives the call, and the ABI
        // matches phobos-mlir's exploded-memref layout at index_bitwidth 32.
        unsafe {
            self.stream
                .launch(&func, grid, (256u32, 1, 1), 0, &raw)
                .context("layernorm launch")?;
        }
        self.stream.synchronize()?;

        let mut y_pad = vec![0.0f32; rp * w];
        y_dev.copy_to(&mut y_pad)?;
        y_pad.truncate(rows * w);
        Ok(y_pad)
    }
}

/// Rows-per-CTA for a width-`w` LayerNorm. The kernel holds two full `[tr, w]`
/// f32 tiles, the row and a squared temporary, so `2 * tr * w * 4` has to stay
/// well under the 48 KB shared budget.
fn choose_ln_tr(w: usize) -> usize {
    let max_tr = (40_000 / (2 * w * 4)).max(1);
    [16, 8, 4, 2, 1]
        .into_iter()
        .find(|&t| t <= max_tr)
        .unwrap_or(1)
}

/// The shared-lean LayerNorm kernel for width `w`: variance as
/// `E[x^2] - E[x]^2` and in-place transforms, so only two full tiles are ever
/// live.
fn layernorm_src(w: usize) -> (String, usize) {
    let tr = choose_ln_tr(w);
    let wf = format!("{w}.0");
    let src = format!(
        "@launch(256)\n\
         kernel layernorm(X: tensor<f32>[M, {w}], G: tensor<f32>[1, {w}], B: tensor<f32>[1, {w}], Y: tensor<f32>[M, {w}], eps: f32) {{\n\
         \x20 let row = program_id(0) * {tr}\n\
         \x20 var x: tile<f32>[{tr}, {w}] = X[row :+ {tr}, 0 :+ {w}]\n\
         \x20 var s: tile<f32>[{tr}, 1] = rowsum(x)\n\
         \x20 var mu: tile<f32>[{tr}, 1] = s / {wf}\n\
         \x20 var ss: tile<f32>[{tr}, 1] = rowsum(x * x)\n\
         \x20 var v: tile<f32>[{tr}, 1] = ss / {wf} - mu * mu\n\
         \x20 var sd: tile<f32>[{tr}, 1] = sqrt(v + eps)\n\
         \x20 x = x - mu\n\
         \x20 x = x / sd\n\
         \x20 var g: tile<f32>[1, {w}] = G[0 :+ 1, 0 :+ {w}]\n\
         \x20 x = x * g\n\
         \x20 var b: tile<f32>[1, {w}] = B[0 :+ 1, 0 :+ {w}]\n\
         \x20 x = x + b\n\
         \x20 Y[row :+ {tr}, 0 :+ {w}] = x\n\
         }}\n"
    );
    (src, tr)
}

fn round_up(x: usize, tile: usize) -> usize {
    x.div_ceil(tile) * tile
}

/// Zero-pad a row-major `[r, c]` matrix into `[rp, cp]`.
fn pad(src: &[f32], r: usize, c: usize, rp: usize, cp: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rp * cp];
    for i in 0..r {
        out[i * cp..i * cp + c].copy_from_slice(&src[i * c..i * c + c]);
    }
    out
}

/// Run `graph` over f32 inputs keyed by edge name, returning its outputs as
/// row-major f32 vectors.
///
/// Compute ops lower to Phobos kernels and run one per node in topological
/// order, over intermediate f32 device buffers. Layout and index ops move data
/// rather than compute on it, so [`crate::layout`] resolves them on the host;
/// that costs a round-trip apiece but keeps them rank-general. Integer tensors
/// stay host-side as i64.
pub fn run(graph: &Graph, inputs: &HashMap<String, Vec<f32>>) -> Result<HashMap<String, Vec<f32>>> {
    run_typed(graph, inputs, &HashMap::new())
}

/// [`run`] also taking integer inputs, token ids for a Gather among them, as
/// `(values, dims)` keyed by edge name.
pub fn run_typed(
    graph: &Graph,
    f32_inputs: &HashMap<String, Vec<f32>>,
    int_inputs: &HashMap<String, Vec<i64>>,
) -> Result<HashMap<String, Vec<f32>>> {
    let shapes = shape::infer(graph)?;

    // A CUDA context must stay alive for the whole run.
    let _ctx = cust::quick_init().context("initializing CUDA")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut exec = Exec {
        stream,
        shapes,
        f32s: HashMap::new(),
        ints: HashMap::new(),
    };

    for (name, tensor) in &graph.initializers {
        match &tensor.data {
            TensorData::F32(v) => exec.put_f32(name, v)?,
            TensorData::I64(v) => {
                exec.ints.insert(name.clone(), v.clone());
            }
            TensorData::I32(v) => {
                exec.ints
                    .insert(name.clone(), v.iter().map(|&x| x as i64).collect());
            }
            other => bail!("initializer '{name}' has unsupported data {other:?}"),
        }
    }
    for (name, host) in f32_inputs {
        exec.check_numel(name, host.len())?;
        exec.put_f32(name, host)?;
    }
    for (name, host) in int_inputs {
        exec.ints.insert(name.clone(), host.clone());
    }

    for node in &graph.nodes {
        exec.exec_node(node)
            .with_context(|| format!("executing node '{}' ({})", node.name, node.op_type))?;
    }

    let mut out = HashMap::new();
    for vi in &graph.outputs {
        out.insert(vi.name.clone(), exec.f32_host(&vi.name)?);
    }
    Ok(out)
}

/// f32 tensors on the device, integer tensors on the host.
struct Exec {
    stream: Stream,
    shapes: HashMap<String, Dims>,
    f32s: HashMap<String, DeviceBuffer<f32>>,
    ints: HashMap<String, Vec<i64>>,
}

impl Exec {
    fn exec_node(&mut self, node: &Node) -> Result<()> {
        match node.op_type.as_str() {
            "Reshape" | "Transpose" | "Gather" | "Concat" | "Split" => self.exec_layout(node),
            _ => self.exec_compute(node),
        }
    }

    /// Lower a compute node to a Phobos kernel, allocate its output, launch.
    fn exec_compute(&mut self, node: &Node) -> Result<()> {
        let plan = lower::lower_node(node, &|e: &str| self.shapes.get(e).cloned())
            .with_context(|| format!("lowering '{}'", node.op_type))?;

        // Kernels fully overwrite their output.
        let n = self.numel(&plan.output)?;
        self.f32s.insert(
            plan.output.clone(),
            DeviceBuffer::from_slice(&vec![0.0f32; n])?,
        );

        let ptx = compile(&plan).with_context(|| format!("compiling '{}'", plan.kernel_name))?;
        let module = Module::from_ptx(&ptx, &[])?;
        let func = module.get_function(&plan.kernel_name)?;
        self.launch(&func, &plan)
            .with_context(|| format!("launching '{}'", plan.kernel_name))?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Resolve a layout or index op on the host.
    fn exec_layout(&mut self, node: &Node) -> Result<()> {
        match node.op_type.as_str() {
            "Reshape" => {
                let (src, dst) = (&node.inputs[0], &node.outputs[0]);
                // A metadata change only, and inference already recorded the
                // shape, so the values carry straight over.
                if let Some(v) = self.ints.get(src).cloned() {
                    self.ints.insert(dst.clone(), v);
                } else {
                    let data = self.f32_host(src)?;
                    self.put_f32(dst, &data)?;
                }
            }
            "Transpose" => {
                let (src, dst) = (&node.inputs[0], &node.outputs[0]);
                let dims = self.dims(src)?;
                let perm = transpose_perm(node, dims.len());
                let (out, _) = layout::transpose(&self.f32_host(src)?, &dims, &perm)?;
                self.put_f32(dst, &out)?;
            }
            "Gather" => {
                let data = self.f32_host(&node.inputs[0])?;
                let data_dims = self.dims(&node.inputs[0])?;
                let idx = self
                    .ints
                    .get(&node.inputs[1])
                    .with_context(|| {
                        format!("Gather indices '{}' are not integer", node.inputs[1])
                    })?
                    .clone();
                let idx_dims = self.dims(&node.inputs[1])?;
                let axis = norm_axis(int_attr(node, "axis").unwrap_or(0), data_dims.len())?;
                let (out, _) = layout::gather(&data, &data_dims, &idx, &idx_dims, axis)?;
                self.put_f32(&node.outputs[0], &out)?;
            }
            "Concat" => {
                let axis_raw = int_attr(node, "axis").context("Concat needs an axis")?;
                let hosts: Vec<Vec<f32>> = node
                    .inputs
                    .iter()
                    .map(|e| self.f32_host(e))
                    .collect::<Result<_>>()?;
                let dims: Vec<Dims> = node
                    .inputs
                    .iter()
                    .map(|e| self.dims(e))
                    .collect::<Result<_>>()?;
                let axis = norm_axis(axis_raw, dims[0].len())?;
                let inputs: Vec<(&[f32], &[i64])> = hosts
                    .iter()
                    .zip(&dims)
                    .map(|(h, d)| (h.as_slice(), d.as_slice()))
                    .collect();
                let (out, _) = layout::concat(&inputs, axis)?;
                self.put_f32(&node.outputs[0], &out)?;
            }
            "Split" => {
                let dims = self.dims(&node.inputs[0])?;
                let axis = norm_axis(int_attr(node, "axis").unwrap_or(0), dims.len())?;
                let sizes: Dims = node
                    .outputs
                    .iter()
                    .map(|o| self.dims(o).map(|d| d[axis]))
                    .collect::<Result<_>>()?;
                let parts = layout::split(&self.f32_host(&node.inputs[0])?, &dims, axis, &sizes)?;
                for (out_edge, (data, _)) in node.outputs.iter().zip(parts) {
                    if !out_edge.is_empty() {
                        self.put_f32(out_edge, &data)?;
                    }
                }
            }
            other => bail!("exec_layout got non-layout op '{other}'"),
        }
        Ok(())
    }

    /// Marshal a plan's parameters into the memref ABI and launch.
    fn launch(&self, func: &cust::function::Function, plan: &lower::KernelPlan) -> Result<()> {
        let mut args: Vec<KernelArg> = Vec::new();
        for param in &plan.params {
            match param {
                Param::Tensor { edge, view } => {
                    let buf = self
                        .f32s
                        .get(edge)
                        .with_context(|| format!("no buffer for edge '{edge}'"))?;
                    let addr = buf.as_device_ptr().as_raw();
                    abi::push_tensor_descriptor(&mut args, addr, view);
                }
                Param::ScalarF32(x) => args.push(KernelArg::F32(*x)),
            }
        }

        // Each argument lives in its own 8-byte slot and the raw list points
        // into it, so `slots` must outlive the launch below.
        let mut slots: Vec<u64> = args.iter().map(|a| a.slot()).collect();
        let raw: Vec<*mut c_void> = slots
            .iter_mut()
            .map(|s| s as *mut u64 as *mut c_void)
            .collect();

        // SAFETY: `raw` points into `slots`, which outlives this call, and the
        // layout matches phobos-mlir's exploded-memref ABI at index_bitwidth
        // 32.
        unsafe {
            self.stream
                .launch(func, plan.grid, (plan.block, 1, 1), 0, &raw)
                .context("cuLaunchKernel")?;
        }
        Ok(())
    }

    /// Upload host f32 data to a fresh device buffer for `edge`.
    fn put_f32(&mut self, edge: &str, data: &[f32]) -> Result<()> {
        self.f32s
            .insert(edge.to_string(), DeviceBuffer::from_slice(data)?);
        Ok(())
    }

    /// Copy an f32 device tensor back to the host.
    fn f32_host(&self, edge: &str) -> Result<Vec<f32>> {
        let buf = self
            .f32s
            .get(edge)
            .with_context(|| format!("no f32 buffer for edge '{edge}'"))?;
        let mut host = vec![0.0f32; self.numel(edge)?];
        buf.copy_to(&mut host)?;
        Ok(host)
    }

    fn dims(&self, edge: &str) -> Result<Dims> {
        self.shapes
            .get(edge)
            .cloned()
            .with_context(|| format!("edge '{edge}' has no resolved shape"))
    }

    fn numel(&self, edge: &str) -> Result<usize> {
        Ok(self.dims(edge)?.iter().product::<i64>() as usize)
    }

    fn check_numel(&self, edge: &str, got: usize) -> Result<()> {
        let want = self.numel(edge)?;
        if got != want {
            bail!("input '{edge}' has {got} elements, shape wants {want}");
        }
        Ok(())
    }
}

/// Compile a plan's Phobos source to PTX, pinning its autotune tile dims.
fn compile(plan: &lower::KernelPlan) -> Result<String> {
    let mut ctx = phobos_base::context::Context::default();
    for (name, value) in &plan.overrides {
        ctx.shape_overrides.insert(name.clone(), *value);
    }
    phobos_lang::compile(&ctx, &plan.source)
}

/// The Transpose `perm` attribute, defaulting to reversing all axes.
fn transpose_perm(node: &Node, rank: usize) -> Vec<usize> {
    match node.attrs.get("perm") {
        Some(crate::ir::Attribute::Ints(perm)) => perm.iter().map(|&p| p as usize).collect(),
        _ => (0..rank).rev().collect(),
    }
}

fn int_attr(node: &Node, name: &str) -> Option<i64> {
    match node.attrs.get(name) {
        Some(crate::ir::Attribute::Int(i)) => Some(*i),
        _ => None,
    }
}

fn norm_axis(axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a as usize >= rank {
        bail!("axis {axis} out of range for rank {rank}");
    }
    Ok(a as usize)
}
