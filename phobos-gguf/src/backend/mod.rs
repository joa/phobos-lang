use anyhow::Result;

use crate::quant::{Packed, Quant};
pub use crate::quant::quantize_row;

/// A handle to backend-owned storage, so the bytes can live on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buf(pub usize);

/// A handle to backend-owned storage holding f16 rather than f32, its length
/// still counted in elements. Only the key and value caches use it: GQA reads
/// each cached position once per query in its group, and the decode kernel is
/// bound by the rate it reads them, so halving the format halves bytes moved.
/// Everything else stays f32; the kernels widen a cached element on load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HBuf(pub usize);

/// A weight left quantized, held in the planes its kernels index. See
/// [`crate::quant::Spec::planes`] for what those are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QBuf(pub usize);

/// A weight held in its file's raw block bytes plus the header-field planes
/// [`crate::quant::Spec::raw_scales`] pulls out of them, for a kernel that
/// decodes the rest itself. [`Backend::constant_raw`]'s default wraps a plain
/// dense [`Buf`], so this is safe to read as one whenever a backend has not
/// overridden that method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawBuf(pub usize);

/// An activation quantized once for the several projections that read it. Valid
/// only inside the pass that produced it, and only while its source buffer is
/// unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QAct(pub usize);

/// Elements sharing one activation scale. Activations quantize to Q8_0
/// whatever format the weight is held in, so this is the block every
/// quantized contraction here runs on.
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
/// `x` is the residual row and destination both, since the down projection
/// adds into it. The normalization is part of the request rather than a step
/// already taken, so a fused kernel recomputes it per block instead of
/// paying a barrier to share it.
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

/// [`FusedMlp`] over raw-format weights, which a raw file holds as two
/// separate gate and up tensors. Each carries its format, since a file mixes
/// them: the 4B's down projection is Q6_K in half its layers and Q4_K in
/// the rest.
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

/// One contiguous run of a projection's outputs, and where the caller wants
/// it. A stacked projection's consumers can want their window elsewhere (the
/// delta net's convolution reads its qkv plane as the tail of a padded
/// stream), so naming the destination per run lets the projection write
/// there instead of being copied out afterwards.
#[derive(Clone, Copy, Debug)]
pub struct ProjRun {
    /// First output of the weight this run covers.
    pub row_off: usize,
    pub width: usize,
    pub dst: Buf,
    pub dst_off: usize,
}

/// A normalization and the projection reading it, for a backend that can run
/// them as one kernel. `x` is the residual row, normalized per block rather
/// than published, for the reason [`FusedMlp`] gives.
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
/// same operands, run in the kernel that produced their input. Only the
/// convolution costs a barrier; the gates read a window it already
/// published, riding along for free.
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
    /// Value heads: the packed layout's head count, what the delta rule, the
    /// gates and the readout norm all run at.
    pub heads: usize,
    pub head_dim: usize,
    /// Query/key heads. Equal to `heads` outside a grouped-query deltanet;
    /// otherwise fewer, each repeated `heads / kv_heads` times to match, the
    /// same repetition [`ggml_repeat_4d`] does upstream.
    pub kv_heads: usize,
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

    /// Width of one position of the fused projection: query and key at
    /// `kv_heads` wide apiece, value at the full `heads`.
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

    /// The index of the largest of `buf`'s first `len` elements, without
    /// reading the rest back. The default reads the whole vector and reduces
    /// on the host; a device backend overrides it with a reduction that
    /// never leaves the card. Ties go to the last of equal maxima, matching
    /// [`phobos_inference::sampling::argmax`], which this delegates to.
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
    /// [`Backend::zeroed`] for a buffer that lives as long as the sequence: a
    /// backend where residency is per allocation wants those together rather
    /// than one each. Defaults to an ordinary zeroed buffer.
    fn zeroed_state(&self, len: usize) -> Result<Buf> {
        self.zeroed(len)
    }

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
    /// launch: attention's value and key landing in the same cache row. A
    /// backend with no combined kernel calls [`Backend::store_2d`] twice.
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

    /// Attention's output epilogue (quantize the mixed heads, then the
    /// output projection) as one kernel. `false` means the backend has no
    /// fused form and the caller runs the two stages itself.
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

    /// [`Backend::constant`] whose data costs something to produce: `fill`
    /// runs only if `key` is not resident already. A weight in a format no
    /// kernel unpacks goes up this way, dequantized once at upload rather
    /// than once per token.
    fn constant_lazy(&self, key: &str, fill: &dyn Fn() -> Result<Vec<f32>>) -> Result<Buf>;

    /// A quantized weight uploaded once under `key`, split into the planes its
    /// kernels index. Only a format with [`crate::quant::Spec::planes`] can go
    /// up this way; the rest dequantize through [`Backend::constant_lazy`].
    fn constant_quant(&self, key: &str, packed: &Packed) -> Result<QBuf>;

    /// A weight uploaded once under `key` in its raw block bytes, for a
    /// format with [`crate::quant::Spec::raw_scales`] but no kernel that
    /// unpacks it into [`Backend::constant_quant`]'s planes. The default
    /// lands it dense instead, via [`Backend::constant_lazy`]; the handle
    /// stays valid either way.
    fn constant_raw(&self, key: &str, packed: &Packed) -> Result<RawBuf> {
        let buf = self.constant_lazy(key, &|| Ok(packed.dense()))?;
        Ok(RawBuf(buf.0))
    }

    /// [`Backend::matmul`] against a weight uploaded through
    /// [`Backend::constant_raw`]. The activation stays f32: a raw kernel
    /// decodes its weight to f32 in place rather than needing a quantized
    /// activation the way [`Backend::matmul_quant`] does. The default
    /// matches [`Backend::constant_raw`]'s: `w` is a dense buffer in a
    /// [`RawBuf`] wrapper.
    fn matmul_raw(&self, a: Buf, m: usize, k: usize, w: RawBuf, n: usize, out: Buf) -> Result<()> {
        self.matmul(a, m, k, Buf(w.0), n, out)
    }

    /// [`Backend::matmul_raw`] against an activation quantized already, where
    /// the backend has a path that wants one.
    ///
    /// A raw kernel that decodes to f32 has no use for it, which is why this
    /// defaults to ignoring it. One that contracts on the integer tensor cores
    /// does, and then quantizing per weight rather than taking the caller's
    /// copy costs an `m * k` scratch slot a projection -- 163 of them for
    /// IQ1_S alone on a 27B, which is memory the weights need.
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

    /// `out[m, n] = a[m, k] @ w[k, n]`, all row-major.
    fn matmul(&self, a: Buf, m: usize, k: usize, w: Buf, n: usize, out: Buf) -> Result<()>;

    /// [`Backend::matmul`] against a weight left quantized. The activation
    /// quantizes to Q8_0 whatever the weight's format is, so the contraction
    /// is integer throughout. See [`quantize_row`].
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

    /// The normalization, both projections and the SwiGLU between them as
    /// one kernel, for a single-row decode step. `false` means the backend
    /// has no fused form and the caller runs the four stages itself.
    fn fused_mlp(&self, _mlp: FusedMlp) -> Result<bool> {
        Ok(false)
    }

    /// [`Backend::fused_mlp`] over raw-format weights, which a file keeps
    /// as separate gate and up tensors. `false` by default.
    fn fused_mlp_raw(&self, _mlp: FusedMlpRaw) -> Result<bool> {
        Ok(false)
    }

    /// The normalization ahead of a mixer and the projection reading it as
    /// one kernel, each run landing where the caller asked, and optionally
    /// the delta net's convolution and gates behind them. The result says
    /// which halves ran; the caller launches the rest.
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

    /// Rotary embedding, in place, over `[rows * heads, head_dim]`. `table`
    /// is `[positions, rope_dim]`, each row the cosines for one absolute
    /// position followed by its sines; row `p` must be position `p`. Pairs
    /// are `(i, i + rope_dim / 2)`.
    fn rope(&self, x: Buf, rows: usize, table: Buf, spec: Rope) -> Result<()>;

    /// [`Backend::rope`] against a strided window of a wider buffer, writing
    /// a dense `dest` instead of rotating in place: what a fused QKV
    /// projection's query and key both want. The default is
    /// [`Backend::copy_2d`] then [`Backend::rope`], unfused.
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
    /// already carry this call's rows. `q` is `[rows * n_head, head_dim]`
    /// with the head varying fastest; the caches are
    /// `[positions, n_kv * head_dim]` in f16. Row `t` attends to cache
    /// positions `0 ..= start_pos + t`, with `group` query heads sharing
    /// each key head. `out` matches `q` and stays f32.
    fn attention(&self, q: Buf, keys: HBuf, values: HBuf, spec: Attn, out: Buf) -> Result<()>;

    /// `x *= sigmoid(gate)`, elementwise. The attention output gate.
    fn gate_into(&self, x: Buf, gate: Buf) -> Result<()>;

    /// The causal depthwise convolution that feeds the delta rule, split
    /// into the packed planes [`Backend::delta_rule`] reads. `history` is
    /// `[pad + rows, channels]`: the `pad` positions carried from the
    /// previous call followed by this call's fused projection, so position
    /// `t` sees inputs `t - pad ..= t`. `taps` is `[kernel, channels]`,
    /// transposed relative to the file.
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
    /// `packed` carries all five operands consecutively, as
    /// [`Backend::delta_conv`] and [`Backend::delta_gates`] leave them:
    /// query, key and value planes, each `[rows * heads, head_dim]`, then
    /// decay and beta, each `[rows * heads]`. `out` is a fourth such plane;
    /// `state` is `[heads * head_dim, head_dim]`. Per position, per head:
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
