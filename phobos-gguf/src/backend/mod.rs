use anyhow::Result;

use crate::quant::Packed;
pub use crate::quant::quantize_row;

/// A handle to backend-owned storage, so the bytes can live on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buf(pub usize);

/// A handle to backend-owned storage holding f16 rather than f32, its length
/// still counted in elements.
///
/// Only the key and value caches use it. They are the largest thing an
/// attention block owns past a few hundred positions, and the decode kernel is
/// bound by the rate it can read them, so what the format buys is bytes moved
/// rather than bytes held: grouped-query attention gives several query heads one
/// key head and each reads it separately, so a cached position is fetched once
/// per query in its group. Everything else stays f32, the queries included, and
/// the kernels widen a cached element as they load it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HBuf(pub usize);

/// A weight left quantized, held in the planes its kernels index. See
/// [`crate::quant::Spec::planes`] for what those are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QBuf(pub usize);

/// An activation quantized once for the several projections that read it. Valid
/// only inside the pass that produced it, and only while its source buffer is
/// unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QAct(pub usize);

/// Elements sharing one activation scale.
///
/// Activations are quantized to Q8_0 whatever format the weight is held in, so
/// this is the block every quantized contraction here runs on. A weight's own
/// block size comes from its [`crate::quant::Spec`] and only coincides with
/// this one because Q8_0 is the single format a kernel unpacks today.
pub const Q8_BLOCK: usize = crate::quant::Q8_0_BLOCK;

/// Guards the delta rule's L2 normalization, so an all-zero row stays zero.
pub const L2_EPS: f32 = 1e-12;

/// One side of a strided copy.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub buf: Buf,
    pub offset: usize,
    pub pitch: usize,
}

/// [`Plane`] over f16 storage: the destination [`Backend::store_2d`] writes.
#[derive(Clone, Copy, Debug)]
pub struct HPlane {
    pub buf: HBuf,
    pub offset: usize,
    pub pitch: usize,
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
/// `x` is the residual row and the destination both, since the down projection
/// adds into it, and the normalization is part of the request rather than a step
/// the caller has already taken: a fused kernel recomputes it per block instead
/// of paying a barrier to share it.
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

/// Attention's output epilogue for a backend that can run it as one kernel:
/// quantize the mixed heads, then the output projection accumulating into the
/// residual. Unlike [`FusedProject`] there is no normalization ahead of it --
/// `x` is already what the attention kernel produced.
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

/// One contiguous run of a projection's outputs, and where the caller wants it.
///
/// A stacked projection's consumers each read a window of it, and some want that
/// window elsewhere: the delta net's convolution reads its query/key/value plane
/// as the tail of a padded stream. Naming the destination per run is what lets
/// the projection write there rather than be copied out afterwards.
#[derive(Clone, Copy, Debug)]
pub struct ProjRun {
    /// First output of the weight this run covers.
    pub row_off: usize,
    pub width: usize,
    pub dst: Buf,
    pub dst_off: usize,
}

/// A normalization and the projection reading it, for a backend that can run
/// them as one kernel.
///
/// `x` is the residual row, normalized per block rather than published, for the
/// reason [`FusedMlp`] gives.
#[derive(Clone, Copy, Debug)]
pub struct FusedProject<'a> {
    pub x: Buf,
    pub d_model: usize,
    /// Gain of the normalization ahead of the projection.
    pub gain: Buf,
    pub eps: f32,
    /// `[out_dim, d_model]`, the runs being windows of its outputs.
    pub w: QBuf,
    pub out_dim: usize,
    pub runs: &'a [ProjRun],
    /// The delta net's convolution and gates, for a chain continuing past the
    /// projection into them.
    pub mix: Option<FusedMix>,
}

/// Which halves of a [`FusedProject`] a backend ran, so the caller knows what it
/// still has to launch itself. The two are separate because the tail is gated on
/// its own and a backend may cover the projection without it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fused {
    /// The normalization, the projection and its runs.
    pub project: bool,
    /// The convolution and the gates behind them.
    pub mix: bool,
}

/// The delta net's convolution and per-head gates as the tail of a fused
/// projection: [`Backend::delta_conv`] and [`Backend::delta_gates`] with the
/// same operands, run in the kernel that produced their input.
///
/// Only the convolution costs a barrier, because it reads a head's whole row of
/// the position the projection just wrote and four blocks contributed to that.
/// The gates read a window the same barrier already published, so they ride in
/// the convolution's nest for nothing.
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
    pub heads: usize,
    pub head_dim: usize,
    /// Taps in the causal depthwise convolution.
    pub kernel: usize,
    /// Element offsets of head zero's query, key and value within one position.
    pub planes: [usize; 3],
    /// Distance between consecutive heads within a plane.
    pub head_stride: usize,
    /// L2-normalize the query and key, what the delta rule is defined on.
    pub normalize: bool,
    /// Applied to the query after normalization: the `1/sqrt(d)` softmax
    /// attention also carries.
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

    /// Width of one position of the fused projection.
    pub fn channels(&self) -> usize {
        3 * self.heads * self.head_dim
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

/// Where a GGUF model's arithmetic happens. The unit of exchange is a [`Buf`]
/// handle rather than a slice, so activations stay wherever the backend
/// computes.
pub trait Backend {
    fn alloc(&self, len: usize) -> Result<Buf>;
    fn release(&self, buf: Buf);

    /// Free and total bytes on the device this backend allocates from, for a
    /// caller deciding up front whether a model fits. A backend working out of
    /// host memory reports nothing: it competes with the whole machine rather
    /// than with a fixed budget, and the operating system oversubscribes.
    fn device_memory(&self) -> Option<(usize, usize)> {
        None
    }

    fn upload(&self, data: &[f32]) -> Result<Buf>;
    fn read(&self, buf: Buf, out: &mut [f32]) -> Result<()>;

    fn zeroed(&self, len: usize) -> Result<Buf> {
        self.upload(&vec![0.0f32; len])
    }

    /// [`Backend::alloc`] in f16, `len` still counted in elements.
    fn alloc_h(&self, len: usize) -> Result<HBuf>;
    fn release_h(&self, buf: HBuf);
    fn zeroed_h(&self, len: usize) -> Result<HBuf>;

    /// Widen f16 storage back for a caller that has to look at it, which only
    /// the checks do: nothing in a forward pass reads a cache on the host.
    fn read_h(&self, buf: HBuf, out: &mut [f32]) -> Result<()>;

    /// [`Backend::copy`] between f16 buffers. Only a cache outgrowing itself
    /// needs this, so `len` and both offsets are even and a backend may lean on
    /// that to move whole words.
    fn copy_h(
        &self,
        src: HBuf,
        src_offset: usize,
        dst: HBuf,
        dst_offset: usize,
        len: usize,
    ) -> Result<()>;

    /// [`Backend::copy_2d`] rounding f32 into f16 as it goes: the projection's
    /// keys and values landing in the caches. See [`HBuf`] for why they are held
    /// narrow, and `phobos_base::half::f32_to_f16` for the rounding both
    /// backends have to agree on.
    fn store_2d(&self, src: Plane, dst: HPlane, rows: usize, width: usize) -> Result<()>;

    /// [`Backend::store_2d`] applied to two independent plane pairs in one
    /// launch: attention's value (ready as soon as the projection is) and its
    /// key (ready once rope has rotated it) landing in the same cache row.
    /// Value moves to wherever the caller makes this call, which only costs
    /// something if a reader reaches the cache before that point, and nothing
    /// does: the pass's one reader is the attention kernel, after both.
    ///
    /// A backend with no combined kernel calls [`Backend::store_2d`] twice,
    /// which is exactly what issuing them separately would have done.
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

    /// Attention's output epilogue -- quantizing the mixed heads and the
    /// output projection reading them -- as one kernel. `false` means the
    /// backend has no fused form and the caller runs the two stages itself,
    /// which is the only thing a host backend does.
    fn fused_attn_out(&self, _out: FusedAttnOut) -> Result<bool> {
        Ok(false)
    }

    /// Brackets the device-only part of a forward pass; nothing in between
    /// reads back. The GPU backend replays the bracket as one CUDA graph,
    /// backends that issue eagerly ignore both.
    fn begin_pass(&self) -> Result<()> {
        Ok(())
    }
    fn end_pass(&self) -> Result<()> {
        Ok(())
    }

    /// A weight uploaded once under `key` and reused afterwards. The backend
    /// owns it for its lifetime; it is never released.
    fn constant(&self, key: &str, data: &[f32]) -> Result<Buf>;

    /// [`Backend::constant`] whose data costs something to produce: `fill` runs
    /// only if `key` is not resident already.
    ///
    /// A weight in a format no kernel unpacks goes up this way, dequantized
    /// once at upload rather than once per token.
    fn constant_lazy(&self, key: &str, fill: &dyn Fn() -> Result<Vec<f32>>) -> Result<Buf>;

    /// A quantized weight uploaded once under `key`, split into the planes its
    /// kernels index. Only a format with [`crate::quant::Spec::planes`] can go
    /// up this way; the rest dequantize through [`Backend::constant_lazy`].
    fn constant_quant(&self, key: &str, packed: &Packed) -> Result<QBuf>;

    /// `out[m, n] = a[m, k] @ w[k, n]`, all row-major.
    fn matmul(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()>;

    /// [`Backend::matmul`] against a weight left quantized. The activation is
    /// quantized to Q8_0 whatever the weight's format is, so the contraction is
    /// integer throughout; a backend that keeps it in f32 computes something
    /// else. See [`quantize_row`].
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

    /// The normalization, both projections and the SwiGLU between them as one
    /// kernel, for a single-row decode step.
    ///
    /// `false` means the backend has no fused form and the caller runs the four
    /// stages itself, which is the only thing a host backend does.
    fn fused_mlp(&self, _mlp: FusedMlp) -> Result<bool> {
        Ok(false)
    }

    /// The normalization ahead of a mixer and the projection reading it as one
    /// kernel, each run of the projection's outputs landing where the caller
    /// asked for it, and optionally the delta net's convolution and gates behind
    /// them.
    ///
    /// What comes back says which halves ran; the caller launches the rest. A
    /// backend with no fused form leaves both unset, which is the only thing a
    /// host backend does.
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

    /// `out = silu(gate) * rms_norm(x)`, quantized as well: the delta net's
    /// gated readout.
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

    /// `out = silu(gate) * up` over `len` elements. The operands take an offset
    /// because a fused projection leaves them as two windows of one buffer.
    fn swiglu(
        &self,
        gate: Buf,
        gate_at: usize,
        up: Buf,
        up_at: usize,
        out: Buf,
        len: usize,
    ) -> Result<()>;

    /// [`Backend::swiglu`] where the two operands are planes of a wider buffer,
    /// as a fused gate-and-up projection leaves them. The default pulls them
    /// apart into dense copies.
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

    /// Copy a `rows` by `width` block between two strided planes.
    /// [`Backend::copy`] is the case where neither side is wider than the
    /// block.
    fn copy_2d(&self, src: Plane, dst: Plane, rows: usize, width: usize) -> Result<()>;

    /// Rotary embedding, in place, over `[rows * heads, head_dim]`.
    ///
    /// `table` is `[positions, rope_dim]`, each row the cosines for one
    /// absolute position followed by its sines. Row `p` must be position `p`,
    /// filled out to `start_pos + rows`. Pairs are `(i, i + rope_dim / 2)`.
    fn rope(&self, x: Buf, rows: usize, table: Buf, spec: Rope) -> Result<()>;

    /// Causal softmax attention against the key and value caches, which must
    /// already carry this call's rows.
    ///
    /// `q` is `[rows * n_head, head_dim]` with the head varying fastest, and
    /// the caches are `[positions, n_kv * head_dim]` in f16, so one cached
    /// position is a contiguous row and one head of it a column window. Row `t`
    /// attends to cache positions `0 ..= start_pos + t`, with `group` query
    /// heads sharing each key head. `out` matches `q` and stays f32, as does the
    /// arithmetic: a cached element widens on load.
    fn attention(&self, q: Buf, keys: HBuf, values: HBuf, spec: Attn, out: Buf) -> Result<()>;

    /// `x *= sigmoid(gate)`, elementwise. The attention output gate.
    fn gate_into(&self, x: Buf, gate: Buf) -> Result<()>;

    /// The causal depthwise convolution that feeds the delta rule, split into
    /// the packed planes [`Backend::delta_rule`] reads.
    ///
    /// `history` is `[pad + rows, channels]`: the `pad` positions carried from
    /// the previous call followed by this call's fused projection, so position
    /// `t` sees inputs `t - pad ..= t` and the kernel has no boundary case.
    /// `taps` is `[kernel, channels]`, transposed relative to the file so one
    /// tap across a run of channels is contiguous.
    fn delta_conv(&self, history: Buf, taps: Buf, mix: DeltaMix, packed: Buf) -> Result<()>;

    /// The delta rule's per-head gates, written into `packed` after the planes.
    ///
    /// `decay_in` and `beta_in` are the raw `[rows, heads]` projections and
    /// `rate` and `dt_bias` are `[heads]`. The decay is
    /// `exp(rate * softplus(decay_in + dt_bias))` and the write strength is
    /// `sigmoid(beta_in)`.
    #[allow(clippy::too_many_arguments)]
    fn delta_gates(
        &self,
        decay_in: Buf,
        decay_at: usize,
        beta_in: Buf,
        beta_at: usize,
        rate: Buf,
        dt_bias: Buf,
        mix: DeltaMix,
        packed: Buf,
    ) -> Result<()>;

    /// The gated delta rule over a block of positions, advancing `state`.
    ///
    /// `packed` carries all five operands consecutively, as
    /// [`Backend::delta_conv`] and [`Backend::delta_gates`] leave them: the
    /// query, key and value planes, each `[rows * heads, head_dim]` with the
    /// head varying fastest, then the decay and beta vectors, each
    /// `[rows * heads]`. `out` is a fourth `[rows * heads, head_dim]` plane and
    /// `state` is `[heads * head_dim, head_dim]`, keys down the rows and values
    /// across. For each position in order, per head:
    ///
    /// ```text
    /// S      <- decay * S
    /// error  <- beta * (v - k @ S)
    /// S      <- S + k^T @ error
    /// out    <- q @ S
    /// ```
    fn delta_rule(
        &self,
        packed: Buf,
        rows: usize,
        heads: usize,
        head_dim: usize,
        state: Buf,
        out: Buf,
    ) -> Result<()>;

    /// Copy `len` elements between buffers at the given offsets.
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

/// The fusion pass. Only the device backend consumes it, but what it emits is
/// checked without a device, so the tests build it too; there the half that
/// binds operands has no caller.
#[cfg(any(feature = "cuda", test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) mod fuse;

#[cfg(feature = "cuda")]
pub mod device;

pub use host::HostBackend;

#[cfg(feature = "cuda")]
pub use device::DeviceBackend;
