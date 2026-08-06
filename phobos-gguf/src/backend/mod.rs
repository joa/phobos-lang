use anyhow::{Result, ensure};

/// A handle to backend-owned storage, so the bytes can live on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buf(pub usize);

/// A Q8_0 weight held in quantized form: signed bytes plus per-block scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QBuf(pub usize);

/// An activation quantized once for the several projections that read it. Valid
/// only inside the pass that produced it, and only while its source buffer is
/// unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QAct(pub usize);

/// Elements sharing one Q8_0 scale.
pub const Q8_BLOCK: usize = crate::tensor::Q8_0_BLOCK;

/// Guards the delta rule's L2 normalization, so an all-zero row stays zero.
pub const L2_EPS: f32 = 1e-12;

/// One side of a strided copy.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub buf: Buf,
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
/// of paying a barrier to share it. See `docs/megakernel.md`.
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

/// A host-held Q8_0 weight: signed bytes, per-block scales, and the output
/// width it was uploaded with.
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

    /// A Q8_0 weight uploaded once under `key`. The two halves are transposed
    /// relative to each other, each for its own access pattern:
    ///
    /// - `qs` is `[n, k]`, so `qs[j * k + p]` is the byte for output `j` and
    ///   input `p`, putting the contraction axis contiguous.
    /// - `scales` is `[k / Q8_BLOCK, n]`, so `scales[(p / Q8_BLOCK) * n + j]`
    ///   scales that byte, putting one block's scales for a run of outputs
    ///   contiguous.
    fn constant_q8(&self, key: &str, qs: &[i8], scales: &[f32], k: usize, n: usize)
    -> Result<QBuf>;

    /// `out[m, n] = a[m, k] @ w[k, n]`, all row-major.
    fn matmul(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()>;

    /// [`Backend::matmul`] against a weight left in Q8_0 form. The activation is
    /// quantized per block of [`Q8_BLOCK`] too, so the contraction is integer
    /// throughout; a backend that keeps it in f32 computes something else. See
    /// [`quantize_row`].
    fn matmul_q8(&self, a: Buf, m: usize, k: usize, w: QBuf, n: usize, out: Buf) -> Result<()> {
        let act = self.quantize_act(a, m, k)?;
        self.matmul_q8_act(act, m, k, w, n, out)
    }

    /// Quantizes `a[m, k]` once, for the projections that share it.
    fn quantize_act(&self, a: Buf, m: usize, k: usize) -> Result<QAct>;

    /// [`Backend::matmul_q8`] against an activation quantized already.
    fn matmul_q8_act(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        w: QBuf,
        n: usize,
        out: Buf,
    ) -> Result<()>;

    /// [`Backend::matmul_q8_act`] adding into `out` rather than overwriting it.
    fn matmul_q8_add(
        &self,
        act: QAct,
        m: usize,
        k: usize,
        w: QBuf,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        let temp = self.alloc(m * n)?;
        self.matmul_q8_act(act, m, k, w, n, temp)?;
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
    /// the caches are `[positions, n_kv * head_dim]`, so one cached position is
    /// a contiguous row and one head of it a column window. Row `t` attends to
    /// cache positions `0 ..= start_pos + t`, with `group` query heads sharing
    /// each key head. `out` matches `q`.
    fn attention(&self, q: Buf, keys: Buf, values: Buf, spec: Attn, out: Buf) -> Result<()>;

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

/// One byte per element, one scale per block of rows. The blocks run down `k`,
/// so `k` has to tile evenly.
pub fn check_q8_shape(qs: &[i8], scales: &[f32], k: usize, n: usize) -> Result<()> {
    ensure!(
        k.is_multiple_of(Q8_BLOCK),
        "a Q8_0 weight needs k ({k}) to be a multiple of {Q8_BLOCK}"
    );
    ensure!(
        qs.len() == k * n,
        "a [{k}, {n}] Q8_0 weight needs {} bytes, got {}",
        k * n,
        qs.len()
    );
    ensure!(
        scales.len() == (k / Q8_BLOCK) * n,
        "a [{k}, {n}] Q8_0 weight needs {} scales, got {}",
        (k / Q8_BLOCK) * n,
        scales.len()
    );
    Ok(())
}

/// Quantize one block of [`Q8_BLOCK`] activations to int8 with a shared scale:
/// symmetric, round to nearest, the extreme element landing on 127.
///
/// The device kernel reproduces this and has to round the same way. Ties go to
/// even because that is what the hardware's rounding instruction does.
pub fn quantize_row(x: &[f32], qs: &mut [i8]) -> f32 {
    let absmax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let inv = 127.0 / (absmax + 1e-8);
    for (q, &v) in qs.iter_mut().zip(x) {
        *q = (v * inv).round_ties_even() as i8;
    }
    absmax / 127.0
}

pub fn read_vec(backend: &dyn Backend, buf: Buf, len: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0; len];
    backend.read(buf, &mut out)?;
    Ok(out)
}

pub mod host;

#[cfg(feature = "cuda")]
pub mod device;

pub use host::HostBackend;

#[cfg(feature = "cuda")]
pub use device::DeviceBackend;
