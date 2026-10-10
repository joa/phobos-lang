use std::sync::Arc;

use anyhow::{Result, bail};

use crate::experts::ExpertSet;
use crate::quant::{Packed, Quant};
pub use crate::quant::quantize_row;

mod moe;

pub use moe::{ExpertsBuf, Lookahead, Moe, route};

/// A handle to backend-owned storage, so the bytes can live on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buf(pub usize);

/// A handle to backend-owned f16 storage, its length counted in elements.
///
/// Only the key and value caches use it, since decode attention is bound by
/// how fast it reads them. Kernels widen a cached element to f32 on load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HBuf(pub usize);

/// A quantized weight held as planes. See [`crate::quant::Spec::planes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QBuf(pub usize);

/// A weight held as its file's raw block bytes, plus the planes
/// [`crate::quant::Spec::raw_scales`] extracts, for a kernel that decodes it
/// itself.
///
/// Under [`Backend::constant_raw`]'s default this wraps a dense [`Buf`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawBuf(pub usize);

/// An activation quantized once for the projections that share it.
///
/// Valid only inside the pass that produced it, and only while its source
/// buffer is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QAct(pub usize);

/// Elements sharing one activation scale. Activations always quantize to
/// Q8_0, whatever the weight's format.
pub const Q8_BLOCK: usize = crate::quant::Q8_0_BLOCK;

/// The block size [`Backend::hadamard`] transforms in. The device kernel
/// supports only this size.
pub const HADAMARD_BLOCK: usize = 1024;

/// Widest output [`Backend::matmul_rows`] takes. Wider projections use the
/// `[k, n]` kernels.
pub const ROWS_MAX_N: usize = 128;

/// [`Backend::matmul_rows`] needs `k` to be a multiple of this.
pub const ROWS_TK: usize = 32;

/// Whether [`Backend::matmul_rows`] takes a `[k, n]` projection.
pub fn takes_rows(k: usize, n: usize) -> bool {
    n <= ROWS_MAX_N && k.is_multiple_of(ROWS_TK)
}

/// Guards the delta rule's L2 normalization, so an all-zero row stays zero.
pub const L2_EPS: f32 = 1e-12;

/// One side of a strided copy.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub buf: Buf,
    pub offset: usize,
    pub pitch: usize,
}

/// [`Plane`] over f16 storage, the destination of [`Backend::store_2d`].
#[derive(Clone, Copy, Debug)]
pub struct HPlane {
    pub buf: HBuf,
    pub offset: usize,
    pub pitch: usize,
}

/// How the delta net's value heads are regrouped ahead of a folded
/// projection: grouped head `k * repeat + r` reads tiled head
/// `r * groups + k`. See [`crate::hadamard`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadPerm {
    pub head_dim: usize,
    pub groups: usize,
    pub repeat: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Rope {
    pub heads: usize,
    pub head_dim: usize,
    /// Leading elements of each head that rotate. The rest pass through.
    pub rope_dim: usize,
    /// Absolute position of the first row.
    pub start_pos: usize,
}

/// A raw-weight projection `out[m, n] = a[m, k] . w` whose output starts
/// `out.1` elements into `out.0`. See [`Backend::matmul_raw_at`].
#[derive(Clone, Copy, Debug)]
pub struct RawAt {
    pub a: Buf,
    pub m: usize,
    pub k: usize,
    pub w: RawBuf,
    pub n: usize,
    pub out: (Buf, usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Attn {
    pub rows: usize,
    /// Positions already in the caches, so row `t` attends to `start_pos + t`.
    pub start_pos: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
}

impl Attn {
    /// Positions in the caches once this call's rows are appended.
    pub fn total(&self) -> usize {
        self.start_pos + self.rows
    }

    /// Query heads sharing one key/value head.
    pub fn group(&self) -> usize {
        self.n_head / self.n_kv
    }

    /// Width of one cached position: every key head side by side.
    pub fn kv_width(&self) -> usize {
        self.n_kv * self.head_dim
    }
}

/// A whole decode MLP, for a backend that can run it as one kernel.
///
/// `x` is the residual row and also the destination, since the down
/// projection adds into it. The normalization is part of the request so a
/// fused kernel can recompute it per block instead of sharing it through a
/// barrier.
#[derive(Clone, Copy, Debug)]
pub struct FusedMlp {
    pub x: Buf,
    pub d_model: usize,
    pub d_ff: usize,
    /// Gain of the normalization ahead of the projections.
    pub gain: Buf,
    pub eps: f32,
    /// Gate and up stacked, `[2 * d_ff, d_model]`.
    pub gate_up: QBuf,
    /// `[d_model, d_ff]`.
    pub down: QBuf,
}

/// [`FusedMlp`] over raw-format weights, with separate gate and up tensors.
/// Each carries its own format, since a file can mix formats.
pub struct FusedMlpRaw {
    pub x: Buf,
    pub d_model: usize,
    pub d_ff: usize,
    pub gain: Buf,
    pub eps: f32,
    pub gate: (RawBuf, Quant),
    pub up: (RawBuf, Quant),
    pub down: (RawBuf, Quant),
}

/// Attention's output epilogue as one kernel: quantize the mixed heads, then
/// run the output projection, accumulating into the residual. There is no
/// normalization ahead of it.
#[derive(Clone, Copy, Debug)]
pub struct FusedAttnOut {
    /// The mixed heads, one row, `width` wide.
    pub x: Buf,
    pub width: usize,
    /// `[d_model, width]`.
    pub w: QBuf,
    pub d_model: usize,
    /// The residual row, accumulated into.
    pub dest: Buf,
}

/// A weight a fused projection contracts against.
#[derive(Clone, Copy, Debug)]
pub enum ProjWeight {
    /// Q8_0 planes.
    Q8(QBuf),
    /// A raw format, decoded in the kernel by its own intrinsic.
    Raw(RawBuf, Quant),
}

/// One contiguous run of a projection's outputs and where to write it.
///
/// A per-run destination lets the projection write each window straight to
/// its consumer, for example the tail of the delta net's padded history,
/// instead of copying it out afterwards.
#[derive(Clone, Copy, Debug)]
pub struct ProjRun {
    /// Which of the projection's weights this run reads.
    pub weight: usize,
    /// First output of the weight this run covers.
    pub row_off: usize,
    pub width: usize,
    pub dst: Buf,
    pub dst_off: usize,
}

/// A normalization and the projection reading it, as one kernel. `x` is the
/// residual row, normalized per block as in [`FusedMlp`].
#[derive(Clone, Copy, Debug)]
pub struct FusedProject<'a> {
    pub x: Buf,
    pub d_model: usize,
    /// Gain of the normalization ahead of the projection.
    pub gain: Buf,
    pub eps: f32,
    /// Each `[out_dim, d_model]`. A raw file keeps a stacked projection's
    /// parts as separate tensors, so there can be several.
    pub weights: &'a [(ProjWeight, usize)],
    pub runs: &'a [ProjRun],
    /// The delta net's convolution and gates, if the chain continues into
    /// them.
    pub mix: Option<FusedMix>,
}

/// Which halves of a [`FusedProject`] a backend ran. The caller launches the
/// rest itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fused {
    /// The normalization, the projection and its runs.
    pub project: bool,
    /// The convolution and the gates behind them.
    pub mix: bool,
}

/// The delta net's convolution and per-head gates as the tail of a fused
/// projection. Same operands as [`Backend::delta_conv`] and
/// [`Backend::delta_gates`], run in the kernel that produced their input.
#[derive(Clone, Copy, Debug)]
pub struct FusedMix {
    pub spec: DeltaMix,
    /// `[pad + rows, channels]`, whose last position the projection writes.
    pub history: Buf,
    /// `[kernel, channels]`.
    pub taps: Buf,
    /// The raw decay and write-strength projections, as
    /// [`Backend::delta_gates`] takes them.
    pub decay: (Buf, usize),
    pub beta: (Buf, usize),
    pub rate: Buf,
    pub dt_bias: Buf,
    /// The five operands [`Backend::delta_rule`] reads.
    pub packed: Buf,
}

/// One delta-net mixing step, and the layout its fused projection uses.
#[derive(Clone, Copy, Debug)]
pub struct DeltaMix {
    pub rows: usize,
    /// Value heads. This is the head count of the packed layout, the delta
    /// rule, the gates and the readout norm.
    pub heads: usize,
    pub head_dim: usize,
    /// Query/key heads. Equal to `heads` unless the delta net is grouped.
    /// Then there are fewer, each repeated `heads / kv_heads` times, as
    /// [`ggml_repeat_4d`] does upstream.
    pub kv_heads: usize,
    /// Taps in the causal depthwise convolution.
    pub kernel: usize,
    /// Element offsets of head zero's query, key and value within one position.
    pub planes: [usize; 3],
    /// Distance between consecutive heads within a plane.
    pub head_stride: usize,
    /// L2-normalize the query and key.
    pub normalize: bool,
    /// Applied to the query after normalization, typically `1/sqrt(d)`.
    pub query_scale: f32,
}

impl DeltaMix {
    /// Elements in one packed query, key or value plane.
    pub fn span(&self) -> usize {
        self.rows * self.heads * self.head_dim
    }

    /// Elements in one packed decay or beta vector.
    pub fn gates(&self) -> usize {
        self.rows * self.heads
    }

    /// Width of one position of the fused projection: query and key
    /// `kv_heads` wide each, value `heads` wide.
    pub fn channels(&self) -> usize {
        (2 * self.kv_heads + self.heads) * self.head_dim
    }

    /// Positions the convolution carries from one call to the next.
    pub fn pad(&self) -> usize {
        self.kernel - 1
    }

    /// Elements a packed operand buffer needs.
    pub fn packed_len(&self) -> usize {
        3 * self.span() + 2 * self.gates()
    }

    /// Elements the padded convolution input needs.
    pub fn history_len(&self) -> usize {
        (self.pad() + self.rows) * self.channels()
    }
}

/// A host-held quantized weight: its two planes and the output width it was
/// uploaded with. See [`crate::quant::Spec::planes`].
type HostQuant = (Vec<i8>, Vec<f32>, usize);

/// Where a GGUF model's arithmetic happens. Operands are [`Buf`] handles, so
/// activations stay wherever the backend computes.
pub trait Backend {
    fn alloc(&self, len: usize) -> Result<Buf>;
    fn release(&self, buf: Buf);

    /// Free and total bytes on this backend's device, to check whether a
    /// model fits. `None` for a host backend, which has no fixed budget.
    fn device_memory(&self) -> Option<(usize, usize)> {
        None
    }

    /// Free and total bytes on the whole card, other programs included, for
    /// display. Defaults to [`Backend::device_memory`].
    fn card_memory(&self) -> Option<(usize, usize)> {
        self.device_memory()
    }

    /// The card this backend computes on, or `None` for a host backend.
    fn device_info(&self) -> Option<phobos_inference::DeviceInfo> {
        None
    }

    /// Hit statistics for this backend's caches. `None` means the backend
    /// has no caches, not a hit rate of zero.
    fn cache_stats(&self) -> Option<phobos_inference::CacheStats> {
        None
    }

    fn upload(&self, data: &[f32]) -> Result<Buf>;
    fn read(&self, buf: Buf, out: &mut [f32]) -> Result<()>;

    /// The index of the largest of `buf`'s first `len` elements.
    ///
    /// The default reads the vector back and reduces on the host. A device
    /// backend overrides it to reduce on the card. Ties go to the last
    /// maximum, as in [`phobos_inference::sampling::argmax`].
    fn argmax(&self, buf: Buf, len: usize) -> Result<i64> {
        let mut out = vec![0.0f32; len];
        self.read(buf, &mut out)?;
        Ok(phobos_inference::sampling::argmax(&out))
    }

    fn zeroed(&self, len: usize) -> Result<Buf> {
        self.upload(&vec![0.0f32; len])
    }

    /// [`Backend::alloc`] in f16, `len` still counted in elements.
    fn alloc_h(&self, len: usize) -> Result<HBuf>;
    fn release_h(&self, buf: HBuf);
    /// [`Backend::zeroed`] for a buffer that lives as long as the sequence.
    /// A backend may group these into shared allocations. Defaults to
    /// [`Backend::zeroed`].
    fn zeroed_state(&self, len: usize) -> Result<Buf> {
        self.zeroed(len)
    }

    fn zeroed_h(&self, len: usize) -> Result<HBuf>;

    /// Reads f16 storage back as f32. Only the checks use it; a forward pass
    /// never reads a cache on the host.
    fn read_h(&self, buf: HBuf, out: &mut [f32]) -> Result<()>;

    /// [`Backend::copy`] between f16 buffers, used when a cache grows. `len`
    /// and both offsets are even, so a backend may move whole words.
    fn copy_h(
        &self,
        src: HBuf,
        src_offset: usize,
        dst: HBuf,
        dst_offset: usize,
        len: usize,
    ) -> Result<()>;

    /// [`Backend::copy_2d`] rounding f32 to f16, for writing keys and values
    /// into the caches. Both backends must round as
    /// `phobos_base::half::f32_to_f16` does.
    fn store_2d(&self, src: Plane, dst: HPlane, rows: usize, width: usize) -> Result<()>;

    /// [`Backend::store_2d`] on two independent plane pairs in one launch,
    /// for a key and value landing in the same cache row. Defaults to two
    /// [`Backend::store_2d`] calls.
    fn store_2d_pair(
        &self,
        a: (Plane, HPlane),
        b: (Plane, HPlane),
        rows: usize,
        width: usize,
    ) -> Result<()> {
        self.store_2d(a.0, a.1, rows, width)?;
        self.store_2d(b.0, b.1, rows, width)
    }

    /// Runs [`FusedAttnOut`] as one kernel. `false` means the backend has no
    /// fused form and the caller runs the two stages itself.
    fn fused_attn_out(&self, _out: FusedAttnOut) -> Result<bool> {
        Ok(false)
    }

    /// Brackets the device-only part of a forward pass over `rows`
    /// positions. Nothing in between reads back.
    ///
    /// The GPU backend records the bracket as one CUDA graph. Eager backends
    /// ignore it.
    fn begin_pass(&self, _rows: usize) -> Result<()> {
        Ok(())
    }
    fn end_pass(&self) -> Result<()> {
        Ok(())
    }

    /// A weight uploaded once under `key` and reused afterwards. It is never
    /// released.
    fn constant(&self, key: &str, data: &[f32]) -> Result<Buf>;

    /// [`Backend::constant`] where `fill` runs only if `key` is not uploaded
    /// yet. Formats with no kernel are dequantized this way at upload.
    fn constant_lazy(&self, key: &str, fill: &dyn Fn() -> Result<Vec<f32>>) -> Result<Buf>;

    /// A quantized weight uploaded once under `key`, split into planes. Only
    /// formats with [`crate::quant::Spec::planes`] qualify.
    fn constant_quant(&self, key: &str, packed: &Packed) -> Result<QBuf>;

    /// A weight uploaded once under `key` as raw block bytes, for a format
    /// with [`crate::quant::Spec::raw_scales`]. The default uploads it dense
    /// via [`Backend::constant_lazy`].
    fn constant_raw(&self, key: &str, packed: &Packed) -> Result<RawBuf> {
        let buf = self.constant_lazy(key, &|| Ok(packed.dense()))?;
        Ok(RawBuf(buf.0))
    }

    /// [`Backend::matmul`] against a weight from [`Backend::constant_raw`],
    /// with an f32 activation. The default treats `w` as the dense buffer
    /// [`Backend::constant_raw`]'s default made.
    fn matmul_raw(&self, a: Buf, m: usize, k: usize, w: RawBuf, n: usize, out: Buf) -> Result<()> {
        self.matmul(a, m, k, Buf(w.0), n, out)
    }

    /// [`Backend::matmul_raw`] with the activation also available quantized.
    /// An integer tensor core path uses `act`; the default ignores it.
    #[allow(clippy::too_many_arguments)]
    fn matmul_raw_act(
        &self,
        _act: QAct,
        a: Buf,
        m: usize,
        k: usize,
        w: RawBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        self.matmul_raw(a, m, k, w, n, out)
    }

    /// Whether [`Backend::matmul_raw_act`] on this weight and shape reads only
    /// the quantized copy, never the f32 rows beside it. A caller that knows
    /// this can skip writing those rows.
    fn raw_act_suffices(&self, _w: RawBuf, _m: usize, _k: usize, _n: usize) -> bool {
        false
    }

    /// [`Backend::swiglu_q_rows`] without the f32 result, only its quantized
    /// copy, for a down projection that reads nothing else. `None` when this
    /// backend has no such form.
    fn swiglu_q_only(&self, _gate: Buf, _up: Buf, _rows: usize, _width: usize) -> Result<Option<QAct>> {
        Ok(None)
    }

    /// [`Backend::matmul_raw_act`] into `out` starting at an element offset,
    /// for a destination that holds other rows before it. Returns `false`
    /// when this backend or shape has no such path; the caller then projects
    /// into its own buffer and copies.
    fn matmul_raw_at(&self, _act: Option<QAct>, _at: RawAt) -> Result<bool> {
        Ok(false)
    }

    /// `out[m, n] = a[m, k] @ w[k, n]`, all row-major.
    fn matmul(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()>;

    /// [`Backend::matmul`] against a transposed weight, `w[n, k]`. Only for
    /// shapes [`takes_rows`] accepts.
    fn matmul_rows(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()>;

    /// [`Backend::matmul`] against a quantized weight. The activation is
    /// quantized to Q8_0, so the contraction is all integer. See
    /// [`quantize_row`].
    fn matmul_quant(&self, a: Buf, m: usize, k: usize, w: QBuf, n: usize, out: Buf) -> Result<()> {
        let act = self.quantize_act(a, m, k)?;
        self.matmul_quant_act(act, m, k, w, n, out)
    }

    /// Quantizes `a[m, k]` once, for the projections that share it.
    fn quantize_act(&self, a: Buf, m: usize, k: usize) -> Result<QAct>;

    /// [`Backend::matmul_quant`] against an activation quantized already.
    fn matmul_quant_act(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        w: QBuf,
        n: usize,
        out: Buf,
    ) -> Result<()>;

    /// [`Backend::matmul_quant_act`] adding into `out` rather than overwriting
    /// it.
    fn matmul_quant_add(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        w: QBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        let temp = self.alloc(m * n)?;
        self.matmul_quant_act(act, m, k, w, n, temp)?;
        self.add_into(out, temp)?;
        self.release(temp);
        Ok(())
    }

    /// Root-mean-square normalization with a per-channel gain, over each
    /// `width`-element row.
    fn rms_norm(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        out: Buf,
    ) -> Result<()>;

    /// [`Backend::rms_norm`] that also leaves the result quantized for the
    /// projection that reads it.
    fn rms_norm_q(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        out: Buf,
    ) -> Result<QAct> {
        self.rms_norm(x, rows, width, gain, eps, out)?;
        self.quantize_act(out, rows, width)
    }

    /// Runs [`FusedMlp`] as one kernel for a single-row decode step. `false`
    /// means the backend has no fused form and the caller runs the stages
    /// itself.
    fn fused_mlp(&self, _mlp: FusedMlp) -> Result<bool> {
        Ok(false)
    }

    /// Caps the expert cache at `bytes`, a user limit applied before
    /// [`Backend::budget_streamed`]. Backends with no expert cache ignore it.
    fn limit_expert_cache(&self, _bytes: usize) -> Result<()> {
        Ok(())
    }

    /// Tells an expert-streaming backend the resident weights' size and the
    /// number of expert sets, so it can size its expert cache from what is
    /// left. Called once at load, before any pass. Backends with no expert
    /// cache ignore it.
    fn budget_streamed(&self, _resident_bytes: usize, _sets: usize) -> Result<()> {
        Ok(())
    }

    /// Registers a block's expert set under `key`. A later call with the same
    /// key returns the same handle. How much of the set stays resident is up
    /// to the backend.
    fn constant_experts(&self, _key: &str, _set: &Arc<ExpertSet>) -> Result<ExpertsBuf> {
        bail!("this backend has no mixture-of-experts path")
    }

    /// A block's routed feed-forward, see [`Moe`].
    fn moe(&self, _req: Moe) -> Result<()> {
        bail!("this backend has no mixture-of-experts path")
    }

    /// [`Backend::fused_mlp`] over raw-format weights. `false` by default.
    fn fused_mlp_raw(&self, _mlp: FusedMlpRaw) -> Result<bool> {
        Ok(false)
    }

    /// Runs [`FusedProject`] as one kernel, optionally including the delta
    /// net's convolution and gates. The result says which halves ran; the
    /// caller launches the rest.
    fn fused_project(&self, _project: FusedProject) -> Result<Fused> {
        Ok(Fused::default())
    }

    /// [`Backend::swiglu`] that also leaves the result quantized for the
    /// down projection that reads it.
    fn swiglu_q(
        &self,
        gate: Buf,
        gate_at: usize,
        up: Buf,
        up_at: usize,
        out: Buf,
        len: usize,
    ) -> Result<QAct> {
        self.swiglu(gate, gate_at, up, up_at, out, len)?;
        self.quantize_act(out, 1, len)
    }

    /// [`Backend::swiglu_q`] over `rows` rows of `width`, from two dense
    /// buffers. The quantized copy is read once, by the down projection.
    fn swiglu_q_rows(&self, gate: Buf, up: Buf, out: Buf, rows: usize, width: usize) -> Result<QAct> {
        self.swiglu(gate, 0, up, 0, out, rows * width)?;
        self.quantize_act(out, rows, width)
    }

    /// `out = silu(gate) * rms_norm(x)`, also returned quantized. The delta
    /// net's gated readout.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_gated(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        gate: Buf,
        gate_at: usize,
        out: Buf,
    ) -> Result<QAct> {
        let normed = self.alloc(rows * width)?;
        self.rms_norm(x, rows, width, gain, eps, normed)?;
        self.swiglu(gate, gate_at, normed, 0, out, rows * width)?;
        self.release(normed);
        self.quantize_act(out, rows, width)
    }

    /// `acc += add`, elementwise.
    fn add_into(&self, acc: Buf, add: Buf) -> Result<()>;

    /// `x += rms_norm(y) * gain`: a mixer's output normalized onto the
    /// residual stream. `normed` is scratch of `rows * width`.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_add(
        &self,
        y: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        normed: Buf,
        x: Buf,
    ) -> Result<()> {
        self.rms_norm(y, rows, width, gain, eps, normed)?;
        self.add_into(x, normed)
    }

    /// `x[r] += bias` for each of `rows` rows of `width`: a projection's
    /// bias, broadcast down the rows.
    fn add_rows(&self, x: Buf, rows: usize, width: usize, bias: Buf) -> Result<()>;

    /// `out = silu(gate) * up` over `len` elements. The offsets allow both
    /// operands to be windows of one fused projection's output.
    fn swiglu(
        &self,
        gate: Buf,
        gate_at: usize,
        up: Buf,
        up_at: usize,
        out: Buf,
        len: usize,
    ) -> Result<()>;

    /// [`Backend::swiglu`] where the operands are planes of a wider buffer.
    /// The default copies them out densely first.
    fn swiglu_planes(
        &self,
        gate: Plane,
        up: Plane,
        out: Buf,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let (g, u) = (self.alloc(rows * width)?, self.alloc(rows * width)?);
        let dense = |buf| Plane {
            buf,
            offset: 0,
            pitch: width,
        };
        self.copy_2d(gate, dense(g), rows, width)?;
        self.copy_2d(up, dense(u), rows, width)?;
        self.swiglu(g, 0, u, 0, out, rows * width)?;
        self.release(g);
        self.release(u);
        Ok(())
    }

    /// Copies a `rows` by `width` block between two strided planes.
    fn copy_2d(&self, src: Plane, dst: Plane, rows: usize, width: usize) -> Result<()>;

    /// Rotary embedding in place over `[rows * heads, head_dim]`.
    ///
    /// `table` is `[positions, rope_dim]`. Row `p` holds position `p`'s
    /// cosines followed by its sines. Rotated pairs are
    /// `(i, i + rope_dim / 2)`.
    fn rope(&self, x: Buf, rows: usize, table: Buf, spec: Rope) -> Result<()>;

    /// [`Backend::rope`] reading a strided window of a wider buffer, such as
    /// a fused QKV output, and writing a dense `dest`. The default is
    /// [`Backend::copy_2d`] then [`Backend::rope`].
    fn rope_gather(
        &self,
        src: Plane,
        rows: usize,
        table: Buf,
        spec: Rope,
        dest: Buf,
    ) -> Result<()> {
        let width = spec.heads * spec.head_dim;
        self.copy_2d(
            src,
            Plane {
                buf: dest,
                offset: 0,
                pitch: width,
            },
            rows,
            width,
        )?;
        self.rope(dest, rows, table, spec)
    }

    /// Causal softmax attention against the key and value caches, which must
    /// already hold this call's rows.
    ///
    /// `q` is `[rows * n_head, head_dim]`, head varying fastest. The caches
    /// are f16 `[positions, n_kv * head_dim]`. Row `t` attends to positions
    /// `0 ..= start_pos + t`, and `group` query heads share each key head.
    /// `out` has `q`'s shape, in f32.
    fn attention(&self, q: Buf, keys: HBuf, values: HBuf, spec: Attn, out: Buf) -> Result<()>;

    /// `x *= sigmoid(gate)`, elementwise. The attention output gate.
    fn gate_into(&self, x: Buf, gate: Buf) -> Result<()>;

    /// The activation side of a Hadamard-folded weight. Each row of `x`,
    /// `[rows, width]`, is regrouped by `perm`, multiplied by `signs`, and
    /// put through the normalized Walsh-Hadamard transform in blocks of
    /// [`HADAMARD_BLOCK`], into `out`. See [`crate::hadamard`].
    fn hadamard(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        signs: Buf,
        perm: Option<HeadPerm>,
        out: Buf,
    ) -> Result<()>;

    /// [`Backend::hadamard`] that also returns `out` quantized.
    ///
    /// The quantized copy may be overwritten by the next call to this or
    /// [`Backend::rms_norm_hadamard_q`], so read it before then.
    fn hadamard_q(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        signs: Buf,
        perm: Option<HeadPerm>,
        out: Buf,
    ) -> Result<QAct> {
        self.hadamard(x, rows, width, signs, perm, out)?;
        self.quantize_act(out, rows, width)
    }

    /// [`Backend::rms_norm`] of `x` into `normed`, then
    /// [`Backend::hadamard_q`] of that into `out`. Folded projections read
    /// `out`, unfolded ones `normed`.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_hadamard_q(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        gain: Buf,
        eps: f32,
        signs: Buf,
        normed: Buf,
        out: Buf,
    ) -> Result<QAct> {
        self.rms_norm(x, rows, width, gain, eps, normed)?;
        self.hadamard_q(normed, rows, width, signs, None, out)
    }

    /// The causal depthwise convolution feeding the delta rule, written into
    /// the packed planes [`Backend::delta_rule`] reads.
    ///
    /// `history` is `[pad + rows, channels]`: the `pad` positions carried
    /// from the previous call, then this call's projection. Position `t`
    /// sees inputs `t - pad ..= t`. `taps` is `[kernel, channels]`,
    /// transposed from the file's layout.
    fn delta_conv(&self, history: Buf, taps: Buf, mix: DeltaMix, packed: Buf) -> Result<()>;

    /// The delta rule's per-head gates, written into `packed` after the planes.
    ///
    /// `decay_in` and `beta_in` are the raw `[rows, heads]` projections, each
    /// row `pitch` elements after the last (`heads` when dense), and `rate`
    /// and `dt_bias` are `[heads]`. The decay is
    /// `exp(rate * softplus(decay_in + dt_bias))` and the write strength is
    /// `sigmoid(beta_in)`.
    #[allow(clippy::too_many_arguments)]
    fn delta_gates(
        &self,
        decay_in: Buf,
        decay_at: usize,
        beta_in: Buf,
        beta_at: usize,
        pitch: usize,
        rate: Buf,
        dt_bias: Buf,
        mix: DeltaMix,
        packed: Buf,
    ) -> Result<()>;

    /// The gated delta rule over a block of positions, advancing `state`.
    ///
    /// `packed` holds five consecutive operands, as [`Backend::delta_conv`]
    /// and [`Backend::delta_gates`] write them: query, key and value planes,
    /// each `[rows * heads, head_dim]`, then decay and beta, each
    /// `[rows * heads]`. `out` is one more such plane. `state` is
    /// `[heads * head_dim, head_dim]`.
    ///
    /// Per position and head: `S <- decay * S`,
    /// `error <- beta * (v - k @ S)`, `S <- S + k^T @ error`, `out <- q @ S`.
    fn delta_rule(
        &self,
        packed: Buf,
        rows: usize,
        heads: usize,
        head_dim: usize,
        state: Buf,
        out: Buf,
    ) -> Result<()>;

    /// Copies `len` elements between buffers at the given offsets.
    fn copy(
        &self,
        src: Buf,
        src_offset: usize,
        dst: Buf,
        dst_offset: usize,
        len: usize,
    ) -> Result<()>;
}

pub fn read_vec(backend: &dyn Backend, buf: Buf, len: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0; len];
    backend.read(buf, &mut out)?;
    Ok(out)
}

pub mod host;

/// The fusion pass. Only the device backend uses it, but tests build it too
/// to check its output without a device.
#[cfg(any(feature = "cuda", test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) mod fuse;

#[cfg(feature = "cuda")]
pub mod device;

pub use host::HostBackend;

#[cfg(feature = "cuda")]
pub use device::DeviceBackend;
