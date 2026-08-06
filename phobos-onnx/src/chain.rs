use std::collections::{HashMap, HashSet};
use std::ffi::c_void;

use anyhow::{Context, Result, bail};
use cust::memory::{AsyncCopyDestination, CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

use crate::abi::{self, KernelArg};
use crate::interp::{self, Data, Tensor};
use crate::ir::{Graph, Node};
use crate::lower::{self, Param};
use crate::shape::{self, Dims};

/// Output tile and k-slice for the device matmul; operands pad up to these.
const MM_TM: usize = 32;
const MM_TN: usize = 32;
const MM_TK: usize = 16;

/// The plain tiled f32 matmul the gate launches on full-tile operands.
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

/// A device-resident float tensor and its logical shape.
struct DeviceTensor {
    buf: DeviceBuffer<f32>,
    dims: Dims,
}

/// One edge's value, host or device or both, materialized lazily.
#[derive(Default)]
struct Slot {
    host: Option<Tensor>,
    device: Option<DeviceTensor>,
}

/// How much of a run stayed on the device and how much fell back.
#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    pub device_ops: usize,
    pub host_ops: usize,
    pub syncs: usize,
}

pub struct ChainExec {
    _ctx: cust::context::Context,
    stream: Stream,
    /// Keyed by source, which uniquely determines the PTX.
    kernels: HashMap<String, Module>,
    slots: HashMap<String, Slot>,
    /// Kernels launched since the last sync: a pending device write.
    dirty: bool,
    stats: Stats,
    /// What ran on-device against what fell back, by op type, so a run on a
    /// real graph maps out the coverage the later phases need.
    device_hist: HashMap<String, usize>,
    fallback_hist: HashMap<String, usize>,
    /// The plain tiled matmul kernel, compiled once.
    matmul: Module,
    /// Constant matmul operands padded to tile-aligned dims and kept resident
    /// under weight name and padded shape. Persists across `run` calls.
    padded_weights: HashMap<String, DeviceBuffer<f32>>,
    /// The edges that are graph initializers in the current run.
    weight_edges: HashSet<String>,
    /// Padding buffers whose stream ops are still in flight, freed at the next
    /// synchronize: an async copy must outlive its queued work.
    pending: Vec<DeviceBuffer<f32>>,
}

impl ChainExec {
    pub fn new() -> Result<Self> {
        let _ctx = cust::quick_init().context("initializing CUDA")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let mut ctx = phobos_base::context::Context::default();
        for (name, value) in [("TILE_M", MM_TM), ("TILE_N", MM_TN), ("TILE_K", MM_TK)] {
            ctx.shape_overrides.insert(name.to_string(), value as i64);
        }
        let ptx = phobos_lang::compile(&ctx, MATMUL_SRC).context("compiling matmul kernel")?;
        let matmul = Module::from_ptx(&ptx, &[]).context("loading matmul PTX")?;
        Ok(ChainExec {
            _ctx,
            stream,
            kernels: HashMap::new(),
            slots: HashMap::new(),
            dirty: false,
            stats: Stats::default(),
            device_hist: HashMap::new(),
            fallback_hist: HashMap::new(),
            matmul,
            padded_weights: HashMap::new(),
            weight_edges: HashSet::new(),
            pending: Vec::new(),
        })
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// What ran on-device, by op type.
    pub fn device_hist(&self) -> &HashMap<String, usize> {
        &self.device_hist
    }

    /// What fell back to the host, by op type: the coverage gap.
    pub fn fallback_hist(&self) -> &HashMap<String, usize> {
        &self.fallback_hist
    }

    /// Execute `graph` over host-tensor `inputs`, returning host tensors.
    /// Mirrors [`interp::run`] so the results can be checked against it.
    ///
    /// Float activations live on the device and supported compute ops launch
    /// cached kernels on one stream with no per-op sync, so consecutive device
    /// ops chain on stream ordering and the host syncs only to read a value
    /// back. An op with no device kernel falls back to the host interpreter,
    /// downloading its inputs and forcing a sync.
    pub fn run(
        &mut self,
        graph: &Graph,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        self.slots.clear();
        self.dirty = false;
        self.stats = Stats::default();
        self.device_hist.clear();
        self.fallback_hist.clear();
        // Weights are constant across steps, so padded_weights persists.
        self.weight_edges = graph.initializers.keys().cloned().collect();

        for (name, t) in interp::decode_initializers(graph)? {
            self.put_host(name, t);
        }
        for (name, t) in inputs {
            self.put_host(name.clone(), t.clone());
        }

        for node in &graph.nodes {
            let on_device = self
                .try_device(node)
                .with_context(|| format!("chain device node '{}' ({})", node.name, node.op_type))?;
            if on_device {
                self.stats.device_ops += 1;
                *self.device_hist.entry(node.op_type.clone()).or_default() += 1;
            } else {
                self.exec_host(node).with_context(|| {
                    format!("chain host node '{}' ({})", node.name, node.op_type)
                })?;
                self.stats.host_ops += 1;
                *self.fallback_hist.entry(node.op_type.clone()).or_default() += 1;
            }
        }

        let mut out = HashMap::new();
        for vi in &graph.outputs {
            out.insert(vi.name.clone(), self.host(&vi.name)?.clone());
        }
        Ok(out)
    }

    /// Run `node` on the device, or `Ok(false)` when no device kernel covers
    /// it and the caller has to fall back to the host.
    fn try_device(&mut self, node: &Node) -> Result<bool> {
        let op = node.op_type.as_str();
        let in_edges: Vec<String> = node
            .inputs
            .iter()
            .filter(|e| !e.is_empty())
            .cloned()
            .collect();

        // Reshaping a float tensor is metadata only, and keeps the data
        // device-resident. Its shape input is an int, so this has to come
        // before the int-input check below.
        if op == "Reshape" && !self.is_int(&in_edges[0]) {
            return self.try_reshape(node);
        }
        if op == "Transpose" && !self.is_int(&in_edges[0]) {
            return self.try_transpose(node);
        }

        // Only float compute ops with a known output shape go on-device.
        if in_edges.iter().any(|e| self.is_int(e)) {
            return Ok(false);
        }
        let in_dims: Vec<Dims> = in_edges
            .iter()
            .map(|e| self.dims(e))
            .collect::<Result<_>>()?;

        if op == "MatMul" && in_dims.len() == 2 {
            let (ad, bd) = (&in_dims[0], &in_dims[1]);
            let (m, k, n) = (
                ad[ad.len() - 2] as usize,
                ad[ad.len() - 1] as usize,
                bd[bd.len() - 1] as usize,
            );
            // A batch-1 matmul against a 2-D weight goes through the padded
            // gate, which is what carries the lm-head's M=1, N=50257.
            if bd.len() == 2
                && ad[..ad.len() - 2].iter().product::<i64>() == 1
                && bd[0] as usize == k
            {
                let mut out_dims = ad[..ad.len() - 1].to_vec();
                out_dims.push(n as i64);
                self.matmul_2d(
                    &in_edges[0],
                    &in_edges[1],
                    (m, k, n),
                    &node.outputs[0],
                    out_dims,
                )?;
                return Ok(true);
            }
            // Attention's batched matmul: matching batch dims on both sides.
            if ad.len() >= 3 && ad.len() == bd.len() && bd[bd.len() - 2] as usize == k {
                let batch_a: i64 = ad[..ad.len() - 2].iter().product();
                let batch_b: i64 = bd[..bd.len() - 2].iter().product();
                if batch_a == batch_b && batch_a > 1 {
                    let mut out_dims = ad[..ad.len() - 2].to_vec();
                    out_dims.push(m as i64);
                    out_dims.push(n as i64);
                    self.batched_matmul(
                        &in_edges[0],
                        &in_edges[1],
                        batch_a as usize,
                        (m, k, n),
                        &node.outputs[0],
                        out_dims,
                    )?;
                    return Ok(true);
                }
            }
        }

        // The projections. Only the common GPT-2 case runs on-device.
        if op == "Gemm" {
            return self.try_gemm(node);
        }

        let Some(out_dims) = device_output_dims(op, &in_dims) else {
            return Ok(false);
        };

        // Lowering needs dims for this node's input and output edges.
        let mut dmap: HashMap<String, Dims> = in_edges.iter().cloned().zip(in_dims).collect();
        dmap.insert(node.outputs[0].clone(), out_dims.clone());
        let plan = match lower::lower_node(node, &|e| dmap.get(e).cloned()) {
            Ok(p) => p,
            // Lowering bails on shapes it cannot handle yet, a broadcast Add
            // among them, so fall back rather than error.
            Err(_) => return Ok(false),
        };
        // Some lowered kernels do not JIT at every shape: lower.rs's LayerNorm
        // overflows shared memory at wide widths. A compile failure means this
        // shape is unsupported, so fall back.
        if self.ensure_kernel(&plan).is_err() {
            return Ok(false);
        }

        self.launch(&plan, &out_dims)?;
        Ok(true)
    }

    /// The plan's kernel, compiled and cached on first use.
    fn ensure_kernel(&mut self, plan: &lower::KernelPlan) -> Result<()> {
        if !self.kernels.contains_key(&plan.source) {
            let ptx = compile(plan).with_context(|| format!("compiling '{}'", plan.kernel_name))?;
            self.kernels
                .insert(plan.source.clone(), Module::from_ptx(&ptx, &[])?);
        }
        Ok(())
    }

    /// Put every tensor param on the device, allocate the output, and launch
    /// the plan's kernel on the stream without syncing.
    fn launch(&mut self, plan: &lower::KernelPlan, out_dims: &Dims) -> Result<()> {
        for param in &plan.params {
            if let Param::Tensor { edge, .. } = param
                && *edge != plan.output
            {
                self.ensure_device(edge)?;
            }
        }
        let out_numel: usize = out_dims.iter().product::<i64>() as usize;
        let out_buf = DeviceBuffer::from_slice(&vec![0.0f32; out_numel])?;
        self.slots.entry(plan.output.clone()).or_default().device = Some(DeviceTensor {
            buf: out_buf,
            dims: out_dims.clone(),
        });

        // Params go into the memref-descriptor ABI in declaration order.
        let mut args: Vec<KernelArg> = Vec::new();
        for param in &plan.params {
            match param {
                Param::Tensor { edge, view } => {
                    let dt = self.slots[edge]
                        .device
                        .as_ref()
                        .context("param not on device")?;
                    abi::push_tensor_descriptor(&mut args, dt.buf.as_device_ptr().as_raw(), view);
                }
                Param::ScalarF32(x) => args.push(KernelArg::F32(*x)),
            }
        }
        let mut slots: Vec<u64> = args.iter().map(|a| a.slot()).collect();
        let raw: Vec<*mut c_void> = slots
            .iter_mut()
            .map(|s| s as *mut u64 as *mut c_void)
            .collect();

        let func = self.kernels[&plan.source].get_function(&plan.kernel_name)?;
        // SAFETY: raw points into slots, which outlives the launch, and the
        // layout matches phobos-mlir's exploded-memref ABI at index_bitwidth
        // 32.
        unsafe {
            self.stream
                .launch(&func, plan.grid, (plan.block, 1, 1), 0, &raw)
                .with_context(|| format!("launching '{}'", plan.kernel_name))?;
        }
        self.dirty = true;
        Ok(())
    }

    /// A device-resident 2-D matmul, padding its operands to tile-aligned dims
    /// since Phobos has no tail masking yet. A constant B is padded once and
    /// kept resident. The output is the logical `[m, n]` block, and every copy
    /// is async on the stream, so this still chains.
    fn matmul_2d(
        &mut self,
        a_edge: &str,
        b_edge: &str,
        (m, k, n): (usize, usize, usize),
        out: &str,
        out_dims: Dims,
    ) -> Result<()> {
        let (mp, kp, np) = (round_up(m, MM_TM), round_up(k, MM_TK), round_up(n, MM_TN));
        // B is [K, N], resident-cached when constant and padded locally when not.
        let b_ptr = if self.weight_edges.contains(b_edge) {
            let key = format!("{b_edge}:mm:{kp}x{np}");
            self.resident_weight(&key, b_edge, false, (k, n, kp, np))?
        } else {
            self.ensure_device(b_edge)?;
            let bp = pad_2d(
                &self.stream,
                &self.slots[b_edge].device.as_ref().unwrap().buf,
                0,
                k,
                n,
                kp,
                np,
            )?;
            let p = bp.as_device_ptr().as_raw();
            self.pending.push(bp);
            p
        };
        self.matmul_core(a_edge, b_ptr, (m, k, n), (mp, kp, np), out, out_dims)
    }

    /// Pad and upload a constant weight, transposing from `[N, K]` when asked,
    /// into a resident `[kp, np]` device buffer. Happens once per weight.
    fn resident_weight(
        &mut self,
        key: &str,
        b_edge: &str,
        transpose: bool,
        (k, n, kp, np): (usize, usize, usize, usize),
    ) -> Result<u64> {
        if !self.padded_weights.contains_key(key) {
            let host = self.slots[b_edge]
                .host
                .as_ref()
                .with_context(|| format!("weight '{b_edge}' not on host"))?
                .to_f32();
            let kn = if transpose {
                transpose_2d(&host, n, k)
            } else {
                host
            };
            let padded = host_pad(&kn, k, n, kp, np);
            self.padded_weights
                .insert(key.to_string(), DeviceBuffer::from_slice(&padded)?);
        }
        Ok(self.padded_weights[key].as_device_ptr().as_raw())
    }

    /// Launch `A[m,k]` against the resident `[kp,np]` B and store the logical
    /// `[m,n]` result at `out` as `out_dims`. The copies are async, so this
    /// chains.
    fn matmul_core(
        &mut self,
        a_edge: &str,
        b_ptr: u64,
        (m, k, n): (usize, usize, usize),
        (mp, kp, np): (usize, usize, usize),
        out: &str,
        out_dims: Dims,
    ) -> Result<()> {
        self.ensure_device(a_edge)?;
        let a_pad = pad_2d(
            &self.stream,
            &self.slots[a_edge].device.as_ref().unwrap().buf,
            0,
            m,
            k,
            mp,
            kp,
        )?;
        let c_pad = DeviceBuffer::zeroed(mp * np)?;

        let mut args: Vec<KernelArg> = Vec::new();
        abi::push_tensor_descriptor(
            &mut args,
            a_pad.as_device_ptr().as_raw(),
            &[mp as i64, kp as i64],
        );
        abi::push_tensor_descriptor(&mut args, b_ptr, &[kp as i64, np as i64]);
        abi::push_tensor_descriptor(
            &mut args,
            c_pad.as_device_ptr().as_raw(),
            &[mp as i64, np as i64],
        );
        let mut slots: Vec<u64> = args.iter().map(|a| a.slot()).collect();
        let raw: Vec<*mut c_void> = slots
            .iter_mut()
            .map(|s| s as *mut u64 as *mut c_void)
            .collect();

        let func = self.matmul.get_function("matmul")?;
        let grid = ((mp / MM_TM) as u32, (np / MM_TN) as u32, 1u32);
        // SAFETY: raw outlives the launch, a_pad and c_pad live in `pending`
        // and the resident weight in `padded_weights` until the next sync, and
        // the ABI matches phobos-mlir's exploded-memref layout.
        unsafe {
            self.stream
                .launch(&func, grid, (256u32, 1, 1), 0, &raw)
                .context("matmul gate launch")?;
        }
        self.dirty = true;

        let c = extract_2d(&self.stream, &c_pad, mp, np, m, n)?;
        self.slots.entry(out.to_string()).or_default().device = Some(DeviceTensor {
            buf: c,
            dims: out_dims,
        });
        self.pending.push(a_pad);
        self.pending.push(c_pad);
        Ok(())
    }

    /// Attention's `A[batch,m,k] @ B[batch,k,n]`, as a per-head loop of padded
    /// 2-D matmuls all async on the stream. Both operands are activations, so
    /// both pad per call; the heads share one launch config.
    fn batched_matmul(
        &mut self,
        a_edge: &str,
        b_edge: &str,
        batch: usize,
        (m, k, n): (usize, usize, usize),
        out: &str,
        out_dims: Dims,
    ) -> Result<()> {
        self.ensure_device(a_edge)?;
        self.ensure_device(b_edge)?;
        let (mp, kp, np) = (round_up(m, MM_TM), round_up(k, MM_TK), round_up(n, MM_TN));
        let out_buf = DeviceBuffer::zeroed(batch * m * n)?;
        let mut transients: Vec<DeviceBuffer<f32>> = Vec::new();
        {
            let a_buf = &self.slots[a_edge].device.as_ref().unwrap().buf;
            let b_buf = &self.slots[b_edge].device.as_ref().unwrap().buf;
            let func = self.matmul.get_function("matmul")?;
            let grid = ((mp / MM_TM) as u32, (np / MM_TN) as u32, 1u32);
            for h in 0..batch {
                let a_pad = pad_2d(&self.stream, a_buf, h * m * k, m, k, mp, kp)?;
                let b_pad = pad_2d(&self.stream, b_buf, h * k * n, k, n, kp, np)?;
                let c_pad = DeviceBuffer::zeroed(mp * np)?;

                let mut args: Vec<KernelArg> = Vec::new();
                abi::push_tensor_descriptor(
                    &mut args,
                    a_pad.as_device_ptr().as_raw(),
                    &[mp as i64, kp as i64],
                );
                abi::push_tensor_descriptor(
                    &mut args,
                    b_pad.as_device_ptr().as_raw(),
                    &[kp as i64, np as i64],
                );
                abi::push_tensor_descriptor(
                    &mut args,
                    c_pad.as_device_ptr().as_raw(),
                    &[mp as i64, np as i64],
                );
                let mut slots: Vec<u64> = args.iter().map(|a| a.slot()).collect();
                let raw: Vec<*mut c_void> = slots
                    .iter_mut()
                    .map(|s| s as *mut u64 as *mut c_void)
                    .collect();
                // SAFETY: raw outlives the launch, the buffers stay in
                // `transients` and `out_buf` until the next sync, and the ABI
                // matches phobos-mlir.
                unsafe {
                    self.stream
                        .launch(&func, grid, (256u32, 1, 1), 0, &raw)
                        .context("batched matmul launch")?;
                }
                extract_2d_into(&self.stream, &c_pad, np, &out_buf, h * m * n, m, n)?;
                transients.push(a_pad);
                transients.push(b_pad);
                transients.push(c_pad);
            }
        }
        self.dirty = true;
        self.pending.append(&mut transients);
        self.slots.entry(out.to_string()).or_default().device = Some(DeviceTensor {
            buf: out_buf,
            dims: out_dims,
        });
        Ok(())
    }

    /// Run a Gemm on the device, or `Ok(false)` to fall back. Only the common
    /// GPT-2 case is covered: transA 0, alpha and beta 1, a batch-1 A, a
    /// constant B, and an optional bias.
    fn try_gemm(&mut self, node: &Node) -> Result<bool> {
        if int_attr(node, "transA").unwrap_or(0) != 0
            || float_attr(node, "alpha").unwrap_or(1.0) != 1.0
            || float_attr(node, "beta").unwrap_or(1.0) != 1.0
        {
            return Ok(false);
        }
        let tb = int_attr(node, "transB").unwrap_or(0) != 0;
        let a_edge = node.inputs[0].clone();
        let b_edge = node.inputs[1].clone();
        if !self.weight_edges.contains(&b_edge) {
            return Ok(false);
        }
        let ad = self.dims(&a_edge)?;
        let bd = self.dims(&b_edge)?;
        if bd.len() != 2 || ad[..ad.len() - 2].iter().product::<i64>() != 1 {
            return Ok(false);
        }
        let (m, k) = (ad[ad.len() - 2] as usize, ad[ad.len() - 1] as usize);
        // A Conv1D weight is [N, K] with transB 1, a plain one [K, N].
        let (n, bk) = if tb {
            (bd[0] as usize, bd[1] as usize)
        } else {
            (bd[1] as usize, bd[0] as usize)
        };
        if bk != k {
            return Ok(false);
        }
        let (mp, kp, np) = (round_up(m, MM_TM), round_up(k, MM_TK), round_up(n, MM_TN));

        let mut out_dims = ad[..ad.len() - 1].to_vec();
        out_dims.push(n as i64);
        let out = node.outputs[0].clone();
        let bias = node.inputs.get(2).filter(|e| !e.is_empty()).cloned();

        let key = format!("{b_edge}:gemm{}:{kp}x{np}", tb as u8);
        let b_ptr = self.resident_weight(&key, &b_edge, tb, (k, n, kp, np))?;

        // With a bias the matmul lands in a temp and the Add writes the output.
        let mm_target = if bias.is_some() {
            format!("{out}__gemm")
        } else {
            out.clone()
        };
        self.matmul_core(
            &a_edge,
            b_ptr,
            (m, k, n),
            (mp, kp, np),
            &mm_target,
            out_dims.clone(),
        )?;
        if let Some(bias) = bias {
            self.device_bias_add(&mm_target, &bias, &out, (m, n), &out_dims)?;
        }
        Ok(true)
    }

    /// `out = C[m,n] + bias[n]` through the lowered bias-row Add.
    fn device_bias_add(
        &mut self,
        c_edge: &str,
        bias_edge: &str,
        out: &str,
        (m, n): (usize, usize),
        out_dims: &Dims,
    ) -> Result<()> {
        let add = Node {
            name: format!("{out}__bias"),
            op_type: "Add".to_string(),
            inputs: vec![c_edge.to_string(), bias_edge.to_string()],
            outputs: vec![out.to_string()],
            attrs: HashMap::new(),
        };
        let dmap: HashMap<String, Dims> = HashMap::from([
            (c_edge.to_string(), vec![m as i64, n as i64]),
            (bias_edge.to_string(), vec![n as i64]),
            (out.to_string(), vec![m as i64, n as i64]),
        ]);
        let plan =
            lower::lower_node(&add, &|e| dmap.get(e).cloned()).context("lowering gemm bias add")?;
        self.ensure_kernel(&plan)?;
        self.launch(&plan, out_dims)
    }

    /// Reshape a float tensor, which is metadata only. A device tensor stays
    /// resident, its buffer copied under the new shape on the stream; a
    /// host-only tensor is reshaped on the host.
    fn try_reshape(&mut self, node: &Node) -> Result<bool> {
        let data_edge = node.inputs[0].clone();
        let shape_edge = node.inputs[1].clone();
        let out = node.outputs[0].clone();

        let data_dims = self.dims(&data_edge)?;
        let numel: i64 = data_dims.iter().product();
        let raw: Vec<i64> = self
            .host(&shape_edge)?
            .to_f32()
            .iter()
            .map(|&x| x as i64)
            .collect();

        // A 0 copies the input dim, a -1 infers from the element count.
        let mut target = raw;
        for (i, d) in target.iter_mut().enumerate() {
            if *d == 0 && i < data_dims.len() {
                *d = data_dims[i];
            }
        }
        let known: i64 = target.iter().filter(|&&d| d != -1).product::<i64>().max(1);
        for d in target.iter_mut() {
            if *d == -1 {
                *d = numel / known;
            }
        }
        self.relabel(&data_edge, &out, target)?;
        Ok(true)
    }

    /// A transpose reordering only size-1 axes leaves the contiguous layout
    /// alone, so it is a relabel. A real permutation of non-unit axes needs a
    /// kernel, so it falls back.
    fn try_transpose(&mut self, node: &Node) -> Result<bool> {
        let data_edge = node.inputs[0].clone();
        let dims = self.dims(&data_edge)?;
        let perm: Vec<usize> = match node.attrs.get("perm") {
            Some(crate::ir::Attribute::Ints(p)) => p.iter().map(|&x| x as usize).collect(),
            _ => (0..dims.len()).rev().collect(),
        };
        let out_dims: Vec<i64> = perm.iter().map(|&p| dims[p]).collect();
        // The data is unchanged exactly when the non-unit axes keep order.
        let keep = |d: &Dims| d.iter().copied().filter(|&x| x != 1).collect::<Vec<_>>();
        if keep(&dims) != keep(&out_dims) {
            return Ok(false);
        }
        self.relabel(&data_edge, &node.outputs[0], out_dims)?;
        Ok(true)
    }

    /// `out` is `data` under a new shape of the same element count: a stream
    /// copy of the device buffer when resident, a host reshape otherwise.
    fn relabel(&mut self, data_edge: &str, out: &str, target: Dims) -> Result<()> {
        let numel: usize = target.iter().product::<i64>() as usize;
        if self.slots[data_edge].device.is_some() {
            let out_buf = DeviceBuffer::zeroed(numel)?;
            {
                let src = &self.slots[data_edge].device.as_ref().unwrap().buf;
                // SAFETY: src stays in `slots` and out_buf moves into it, and
                // the copy is stream-ordered before any later host read.
                unsafe {
                    out_buf
                        .index(0..numel)
                        .async_copy_from(&src.index(0..numel), &self.stream)?;
                }
            }
            self.dirty = true;
            self.slots.entry(out.to_string()).or_default().device = Some(DeviceTensor {
                buf: out_buf,
                dims: target,
            });
        } else {
            let data = self.host(data_edge)?.to_f32();
            self.put_host(out.to_string(), Tensor::f32(target, data));
        }
        Ok(())
    }

    /// Run `node` on the host interpreter, materializing its inputs first.
    fn exec_host(&mut self, node: &Node) -> Result<()> {
        let mut ins: HashMap<String, Tensor> = HashMap::new();
        for e in node.inputs.iter().filter(|e| !e.is_empty()) {
            ins.insert(e.clone(), self.host(e)?.clone());
        }
        let outs = interp::run_node_host(node, &ins)?;
        for (name, t) in node.outputs.iter().zip(outs) {
            if !name.is_empty() {
                self.put_host(name.clone(), t);
            }
        }
        Ok(())
    }

    fn put_host(&mut self, edge: String, t: Tensor) {
        self.slots.insert(
            edge,
            Slot {
                host: Some(t),
                device: None,
            },
        );
    }

    /// Upload `edge`'s host data to the device unless it is already there.
    fn ensure_device(&mut self, edge: &str) -> Result<()> {
        let slot = self
            .slots
            .get_mut(edge)
            .with_context(|| format!("no slot for '{edge}'"))?;
        if slot.device.is_some() {
            return Ok(());
        }
        let host = slot
            .host
            .as_ref()
            .with_context(|| format!("'{edge}' has no data to upload"))?;
        let (data, dims) = (host.to_f32(), host.dims.clone());
        slot.device = Some(DeviceTensor {
            buf: DeviceBuffer::from_slice(&data)?,
            dims,
        });
        Ok(())
    }

    /// `edge` as a host tensor, downloading and syncing when only a device
    /// copy exists.
    fn host(&mut self, edge: &str) -> Result<&Tensor> {
        let needs_download = {
            let slot = self
                .slots
                .get(edge)
                .with_context(|| format!("no slot for '{edge}'"))?;
            slot.host.is_none()
        };
        if needs_download {
            if self.dirty {
                self.stream.synchronize()?;
                self.dirty = false;
                self.stats.syncs += 1;
                self.pending.clear();
            }
            let slot = self.slots.get(edge).unwrap();
            let dt = slot
                .device
                .as_ref()
                .with_context(|| format!("'{edge}' has no data"))?;
            let mut hostv = vec![0.0f32; dt.dims.iter().product::<i64>() as usize];
            dt.buf.copy_to(&mut hostv)?;
            let dims = dt.dims.clone();
            self.slots.get_mut(edge).unwrap().host = Some(Tensor::f32(dims, hostv));
        }
        Ok(self.slots[edge].host.as_ref().unwrap())
    }

    fn dims(&self, edge: &str) -> Result<Dims> {
        let slot = self
            .slots
            .get(edge)
            .with_context(|| format!("no slot for '{edge}'"))?;
        if let Some(h) = &slot.host {
            Ok(h.dims.clone())
        } else if let Some(d) = &slot.device {
            Ok(d.dims.clone())
        } else {
            bail!("'{edge}' has no shape")
        }
    }

    fn is_int(&self, edge: &str) -> bool {
        self.slots
            .get(edge)
            .and_then(|s| s.host.as_ref())
            .map(|t| matches!(t.data, Data::I64(_)))
            .unwrap_or(false)
    }
}

fn round_up(x: usize, tile: usize) -> usize {
    x.div_ceil(tile) * tile
}

/// Transpose a row-major `[r, c]` host matrix to `[c, r]`.
fn transpose_2d(src: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = src[i * c + j];
        }
    }
    out
}

/// Zero-pad a row-major `[r, c]` host matrix into `[rp, cp]`.
fn host_pad(src: &[f32], r: usize, c: usize, rp: usize, cp: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rp * cp];
    for i in 0..r {
        out[i * cp..i * cp + c].copy_from_slice(&src[i * c..i * c + c]);
    }
    out
}

fn int_attr(node: &Node, name: &str) -> Option<i64> {
    match node.attrs.get(name) {
        Some(crate::ir::Attribute::Int(i)) => Some(*i),
        _ => None,
    }
}

fn float_attr(node: &Node, name: &str) -> Option<f32> {
    match node.attrs.get(name) {
        Some(crate::ir::Attribute::Float(f)) => Some(*f),
        _ => None,
    }
}

/// Copy a row-major `[r, c]` device tensor into a zeroed `[rp, cp]` one, async
/// on `stream`. The padded rows and columns stay zero: they contribute zero
/// terms to the dot, or their output is sliced away.
fn pad_2d(
    stream: &Stream,
    src: &DeviceBuffer<f32>,
    src_off: usize,
    r: usize,
    c: usize,
    rp: usize,
    cp: usize,
) -> Result<DeviceBuffer<f32>> {
    let dst = DeviceBuffer::zeroed(rp * cp)?;
    // SAFETY: dst and src stay in `pending` and `slots` until the next
    // synchronize, so they outlive the queued copies.
    unsafe {
        if cp == c {
            let n = r * c;
            let s = src.index(src_off..src_off + n);
            dst.index(0..n).async_copy_from(&s, stream)?;
        } else {
            for i in 0..r {
                let s = src.index(src_off + i * c..src_off + i * c + c);
                dst.index(i * cp..i * cp + c).async_copy_from(&s, stream)?;
            }
        }
    }
    Ok(dst)
}

/// Slice the logical `[r, c]` block out of a padded `[rp, cp]` device tensor
/// into `dst` at element offset `dst_off`, async on `stream`.
fn extract_2d_into(
    stream: &Stream,
    src: &DeviceBuffer<f32>,
    cp: usize,
    dst: &DeviceBuffer<f32>,
    dst_off: usize,
    r: usize,
    c: usize,
) -> Result<()> {
    // SAFETY: as in `pad_2d`.
    unsafe {
        if cp == c {
            let n = r * c;
            let s = src.index(0..n);
            dst.index(dst_off..dst_off + n)
                .async_copy_from(&s, stream)?;
        } else {
            for i in 0..r {
                let s = src.index(i * cp..i * cp + c);
                dst.index(dst_off + i * c..dst_off + i * c + c)
                    .async_copy_from(&s, stream)?;
            }
        }
    }
    Ok(())
}

/// Slice the logical `[r, c]` block out of a padded `[rp, cp]` device tensor,
/// async on `stream`.
fn extract_2d(
    stream: &Stream,
    src: &DeviceBuffer<f32>,
    rp: usize,
    cp: usize,
    r: usize,
    c: usize,
) -> Result<DeviceBuffer<f32>> {
    let _ = rp;
    let dst = DeviceBuffer::zeroed(r * c)?;
    // SAFETY: as in `pad_2d`.
    unsafe {
        if cp == c {
            let n = r * c;
            let s = src.index(0..n);
            dst.index(0..n).async_copy_from(&s, stream)?;
        } else {
            for i in 0..r {
                let s = src.index(i * cp..i * cp + c);
                dst.index(i * c..i * c + c).async_copy_from(&s, stream)?;
            }
        }
    }
    Ok(dst)
}

/// The logical output shape of a device-supported op, so its output buffer can
/// be sized without running it on the host. `None` means unsupported.
fn device_output_dims(op: &str, ins: &[Dims]) -> Option<Dims> {
    match op {
        "Add" | "Sub" | "Mul" | "Div" => shape::broadcast(&ins[0], &ins[1]),
        "Relu" | "Gelu" | "Softmax" | "LayerNormalization" => Some(ins[0].clone()),
        "MatMul" if ins[0].len() == 2 && ins[1].len() == 2 => Some(vec![ins[0][0], ins[1][1]]),
        _ => None,
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
