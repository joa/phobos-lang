// RMS norm and its quantizing and gated forms.

use super::*;
use phobos_kernels::launch::{CTA_THREADS, WARP_THREADS};

/// Root-mean-square normalization, one CTA per row, reshaped into blocks of
/// 32 so the row reduction runs a warp at a time and doubles as the Q8_0
/// quantization's block maximum. Generated per width since the tile size
/// must be a compile-time constant. `@dynshared`: footprint scales with
/// `width`, past the static 48 KB ceiling for wide models.
pub(crate) fn rms_norm_src(width: usize, eps: f32, form: NormForm) -> String {
    let blocks = width / RMS_LANE;
    let (gated, quantized) = (form.gated(), form.quantized());
    // The quantized, ungated norm as one `rms_norm_q_t` statement where a
    // thread can own four elements of every `4 * cta`.
    if quantized && !gated && let Some(cta) = norm_q_cta(width) {
        return format!(
            "@launch({cta})
@autotune(NB in [{blocks}])
@aligned(RB = NB, MB = NB, D1 = 1)
kernel rms_norm_q(X: tensor<f32>[RB, {RMS_LANE}], G: tensor<f32>[MB, {RMS_LANE}],
              O: tensor<f32>[RB, {RMS_LANE}], Q: tensor<i8>[RB, {RMS_LANE}], S: tensor<f32>[RB, D1]) {{
  let r = program_id(0)
  rms_norm_q_t(X[r * NB :+ NB, 0 :+ {RMS_LANE}], G[0 :+ NB, 0 :+ {RMS_LANE}], {eps:.12},
               O[r * NB :+ NB, 0 :+ {RMS_LANE}], Q[r * NB :+ NB, 0 :+ {RMS_LANE}], S[r * NB :+ NB, 0 :+ 1])
}}
"
        );
    }
    let mut params = String::new();
    let mut body = String::new();

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
@dynshared
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

/// Threads a normalization's CTA carries: one per tile value, up to the
/// usual width; sizing to the tile does not lengthen the reduction.
pub(crate) fn norm_cta(blocks: usize) -> usize {
    (blocks * RMS_LANE).clamp(WARP_THREADS, CTA_THREADS as usize)
}

/// The CTA the one-statement quantized norm runs at: the largest whole
/// number of warps up to [`CTA_THREADS`] whose `4 * cta` divides the row,
/// since `rms_norm_q_t` gives a thread four elements of every `4 * cta`.
/// 1024 and 5120 take 256 threads, 2560 takes 160. `None` for a width no
/// such CTA divides, which takes the tile passes instead.
pub(crate) fn norm_q_cta(width: usize) -> Option<usize> {
    (WARP_THREADS..=CTA_THREADS as usize)
        .rev()
        .step_by(WARP_THREADS)
        .find(|cta| width.is_multiple_of(cta * 4))
}

/// Values per row of the reshaped normalization tile, and of a Q8_0 block.
pub(crate) const RMS_LANE: usize = 32;

/// `out = silu(gate) * up`, with a quantized copy for the projection after
/// it. Folded into [`RMS_LANE`] blocks, same reasoning as `rms_norm_src`.
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

/// What a normalization kernel leaves behind besides the normalized row.
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
