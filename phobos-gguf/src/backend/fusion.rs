// What a backend's fused decode passes take: the requests for one
// kernel over an MLP, an attention output, or a projection with its mix.

use super::*;

/// One decode row's attention operands made ready in one pass: each query
/// and key head RMS-normalized and rotated, the output gate split out
/// beside the query, and the key and value stored into the caches at the
/// rotation's `start_pos`. See [`Backend::attn_prep`].
#[derive(Clone, Copy, Debug)]
pub struct AttnPrep {
    /// The query heads and their output gate, `rope.heads` rows of
    /// `head_dim` each, at each plane's pitch.
    pub q: Plane,
    pub gate: Plane,
    /// The key and value heads, `kv_heads` dense rows of `head_dim`.
    pub k: Buf,
    pub v: Buf,
    pub kv_heads: usize,
    /// The per-head normalization gains, `head_dim` each.
    pub q_gain: Buf,
    pub k_gain: Buf,
    pub eps: f32,
    /// `[positions, rope_dim]` cosines then sines, as [`Backend::rope`] reads.
    pub table: Buf,
    pub rope: Rope,
    /// The dense query and gate, `[rope.heads, head_dim]`.
    pub q_out: Buf,
    pub gate_out: Buf,
    pub keys: HBuf,
    pub values: HBuf,
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
    /// Whether the convolution, once it has read a channel's positions,
    /// moves the last `pad` of them to the front, so `history` carries
    /// itself into the next step.
    pub shift: bool,
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
