// RMS norm and its quantizing and gated forms.

use super::*;
use phobos_kernels::launch::{CTA_THREADS, WARP_THREADS};

/// Root-mean-square normalization over rows of a fixed width, one block per row,
/// viewed as blocks of 32. The width has to be a compile-time constant for the
/// tile, so this is generated per width; a GGUF decoder only ever normalizes at
/// the model dimension. Epsilon is a plain decimal, the grammar's float literal
/// having no exponent form.
///
/// Two things come out of the reshape. A row reduction hands one row to a warp,
/// so a `[1, width]` tile sums on 32 threads of a 256-thread CTA; folded to
/// `[width / 32, 32]` it is a row per eight lanes and the partials fold once
/// more. And blocks of 32 are what a Q8_0 scale covers, so the maximum the
/// quantization needs is the same row reduction over the same tile.
pub(crate) fn rms_norm_src(width: usize, eps: f32, form: NormForm) -> String {
    let blocks = width / RMS_LANE;
    let (gated, quantized) = (form.gated(), form.quantized());
    let mut params = String::new();
    let mut body = String::new();

    // The gate multiplies the normalized row, so it replaces the plain store.
    if gated {
        params.push_str(&format!(", Z: tensor<f32>[RB, {RMS_LANE}]"));
        body.push_str(&format!(
            "  var z = Z[r * NB :+ NB, 0 :+ {RMS_LANE}]
               var y: tile<f32>[NB, {RMS_LANE}] = n * (z / (1.0 + exp(-z)))
"
        ));
    } else {
        body.push_str(&format!(
            "  var y: tile<f32>[NB, {RMS_LANE}] = n
"
        ));
    }
    body.push_str(&format!(
        "  O[r * NB :+ NB, 0 :+ {RMS_LANE}] = y
"
    ));

    // A Q8_0 scale covers 32 values, the row this tile is already folded into,
    // so the maximum is one more row reduction over it.
    if quantized {
        params.push_str(&format!(
            ", Q: tensor<i8>[RB, {RMS_LANE}], S: tensor<f32>[RB, D1]"
        ));
        body.push_str(&format!(
            "  var mx: tile<f32>[NB, 1] = rowmax(tmax(y, -y))
               var q = y * (127.0 / (mx + 0.00000001))
               Q[r * NB :+ NB, 0 :+ {RMS_LANE}] = i8(i32(round(q)))
               S[r * NB :+ NB, 0 :+ 1] = mx / 127.0
"
        ));
    }

    format!(
        "@launch({cta})
@autotune(NB in [{blocks}])
kernel {name}(X: tensor<f32>[RB, {RMS_LANE}], G: tensor<f32>[MB, {RMS_LANE}],
              O: tensor<f32>[RB, {RMS_LANE}]{params}) {{
  let r = program_id(0)
  var x = X[r * NB :+ NB, 0 :+ {RMS_LANE}]
  var sq: tile<f32>[NB, 1] = rowsum(x * x)
  var tot: tile<f32>[1, 1] = rowsum(transpose(sq))
  var inv: tile<f32>[1, 1] = 1.0 / sqrt(tot / {width}.0 + {eps:.12})
  var g = G[0 :+ NB, 0 :+ {RMS_LANE}]
  var n: tile<f32>[NB, {RMS_LANE}] = x * inv * g
{body}}}
",
        cta = norm_cta(blocks),
        name = form.kernel()
    )
}

/// Threads a normalization's CTA carries: one per value of its tile, up to the
/// usual width. The delta net normalizes 128 values at a time, which on a
/// 256-thread CTA leaves half the block idle across a grid of 8192, and sizing
/// the block to the tile does not lengthen the reduction.
pub(crate) fn norm_cta(blocks: usize) -> usize {
    (blocks * RMS_LANE).clamp(WARP_THREADS, CTA_THREADS as usize)
}

/// Values per row of the reshaped normalization tile, and of a Q8_0 block.
pub(crate) const RMS_LANE: usize = 32;

/// `out = silu(gate) * up`, with the quantized copy the projection after it
/// reads. Folded into blocks of [`RMS_LANE`] for the same reason the
/// normalization is: that is the run a Q8_0 scale covers, so the maximum is a
/// row reduction over the tile the kernel already holds.
pub(crate) fn swiglu_q_src(blocks: usize) -> String {
    format!(
        "@launch(256)
@autotune(NB in [{blocks}])
kernel swiglu_q(G: tensor<f32>[RB, {RMS_LANE}], U: tensor<f32>[RB, {RMS_LANE}],
                O: tensor<f32>[RB, {RMS_LANE}], Q: tensor<i8>[RB, {RMS_LANE}],
                S: tensor<f32>[RB, D1]) {{
  let p = program_id(0)
  var g = G[p * NB :+ NB, 0 :+ {RMS_LANE}]
  var u = U[p * NB :+ NB, 0 :+ {RMS_LANE}]
  var y: tile<f32>[NB, {RMS_LANE}] = (g / (1.0 + exp(-g))) * u
  O[p * NB :+ NB, 0 :+ {RMS_LANE}] = y
  var mx: tile<f32>[NB, 1] = rowmax(tmax(y, -y))
  var q = y * (127.0 / (mx + 0.00000001))
  Q[p * NB :+ NB, 0 :+ {RMS_LANE}] = i8(i32(round(q)))
  S[p * NB :+ NB, 0 :+ 1] = mx / 127.0
}}
"
    )
}

/// Values of a SwiGLU one CTA takes, in [`RMS_LANE`] blocks.
pub(crate) const SWIGLU_Q_BLOCKS: usize = ELEM_TILE / RMS_LANE;

/// What a normalization kernel leaves behind besides the normalized row. Both
/// extras are epilogues on the same tile, so they compose, which the delta net's
/// readout needs.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum NormForm {
    Plain,
    Quantized,
    GatedQuantized,
}

impl NormForm {
    pub(crate) fn gated(self) -> bool {
        self == NormForm::GatedQuantized
    }

    pub(crate) fn quantized(self) -> bool {
        self != NormForm::Plain
    }

    pub(crate) fn kernel(self) -> &'static str {
        match self {
            NormForm::Plain => "rms_norm",
            NormForm::Quantized => "rms_norm_q",
            NormForm::GatedQuantized => "rms_norm_gated",
        }
    }
}
