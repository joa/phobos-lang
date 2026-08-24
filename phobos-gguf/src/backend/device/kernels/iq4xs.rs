// IQ4_XS matvec: decodes straight from the raw block bytes (see q2k.rs).
// Unlike the other IQ formats, quants are not grid-coded: each nibble
// indexes a fixed 16-entry codebook ([`crate::quant::iq4xs_flat_codebook`],
// `KVALUES_IQ4NL`), so one `gather` covers a whole 32-element run. The scale
// is split the way Q4_K's is: four bits from `scales_l`, two from
// `scales_h`. The gather index is built with a `var` and two sliced
// assignments (unlike a `let`-bound view, a `var` can be written into a
// slice at a time), since low/high nibbles land in different output halves.

use std::fmt::Write as _;

/// Output columns per CTA.
pub(crate) const IQ4XS_TN: usize = 8;

const RUNS: usize = 8;
const RUN: usize = 32;
const BLOCK_BYTES: usize = 136;
const SCALES_H_OFF: usize = 2;
const SCALES_L_OFF: usize = SCALES_H_OFF + 2;
const QS_OFF: usize = SCALES_L_OFF + 4;

/// [`crate::quant::iq4xs_flat_codebook`]'s length: sixteen nibble values.
pub(crate) const IQ4XS_CODEBOOK_LEN: usize = 16;

/// Byte offsets and scale-bit divisors for run `ib`.
fn run_geometry(ib: usize) -> (usize, usize, usize, usize) {
    let scale_l_off = SCALES_L_OFF + ib / 2;
    let scale_l_div = if ib.is_multiple_of(2) { 1 } else { 16 };
    let scale_h_div = 4usize.pow(ib as u32);
    let qs_off = QS_OFF + RUN / 2 * ib;
    (scale_l_off, scale_l_div, scale_h_div, qs_off)
}

/// One run's decoded value, `var idx{ib}`/`let decoded{ib} = ...`, and its
/// `out_off`.
fn decoded_run(ib: usize) -> (usize, String) {
    let (scale_l_off, scale_l_div, scale_h_div, qs_off) = run_geometry(ib);
    let out_off = ib * RUN;
    let byte = |off: usize| format!("((i32(qb[:, {off} :+ 1]) + 256) % 256)");
    let low = format!("(({} / {scale_l_div}) % 16)", byte(scale_l_off));
    let scales_h = format!("({} + {} * 256)", byte(SCALES_H_OFF), byte(SCALES_H_OFF + 1));
    let high = format!("(({scales_h} / {scale_h_div}) % 4)");
    let dl = format!("(f32(d) * f32({low} + {high} * 16 - 32))");
    let half = RUN / 2;
    let qs16 = format!("((i32(qb[:, {qs_off} :+ {half}]) + 256) % 256)");
    (
        out_off,
        format!(
            "    var idx{ib}: tile<i32>[TN, {RUN}] = 0
    idx{ib}[:, 0 :+ {half}] = {qs16} % 16
    idx{ib}[:, {half} :+ {half}] = {qs16} / 16
    let decoded{ib} = {dl} * f32(gather(codebook, idx{ib}))\n"
        ),
    )
}

pub(crate) fn iq4xs_matvec_src(tn: usize) -> String {
    let mut body = String::new();
    for ib in 0..RUNS {
        let (out_off, decode) = decoded_run(ib);
        let _ = writeln!(
            body,
            "{decode}    let a{ib} = A[pm :+ 1, kb * 256 + {out_off} :+ {RUN}]
    acc = acc + dot_t(a{ib}, decoded{ib})"
        );
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq4xs_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                    D: tensor<f16>[N, NB], CODEBOOK: tensor<i32>[1, {IQ4XS_CODEBOOK_LEN}],
                    C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  let pm = program_id(1)
  let codebook = CODEBOOK[0 :+ 1, :]
  var acc: tile<f32>[1, TN] = 0.0
  for kb in range(0, NB, 1) {{
    var qb = QB[pn * TN :+ TN, kb * {BLOCK_BYTES} :+ {BLOCK_BYTES}]
    let d = D[pn * TN :+ TN, kb :+ 1]
{body}  }}
  C[pm :+ 1, pn * TN :+ TN] = acc
}}
"
    )
}

/// [`iq4xs_matvec_src`] for `m == 1`, folding the whole decode-and-reduce
/// into one `iq4xs_qdot_t` call; see `iq1s_qdot_matvec_src`'s doc, which
/// this mirrors. `@aligned(N = TN)` is required for the same reason:
/// `iq4xs_qdot_t` demands its `qb`/`d` slices provably in bounds.
pub(crate) fn iq4xs_qdot_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq4xs_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                         D: tensor<f16>[N, NB], CODEBOOK: tensor<i32>[1, {IQ4XS_CODEBOOK_LEN}],
                         C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq4xs_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                          D[pn * TN :+ TN, :], CODEBOOK[0 :+ 1, :])
}}
"
    )
}

/// [`iq4xs_matvec_src`]'s decode, stored straight into a `[K, N]` scratch
/// instead of reduced against an activation row; see `iq1s.rs`'s
/// `iq1s_dequant_src` for why.
pub(crate) fn iq4xs_dequant_src(tn: usize) -> String {
    let mut body = String::new();
    for ib in 0..RUNS {
        let (out_off, decode) = decoded_run(ib);
        let _ = writeln!(
            body,
            "{decode}    SCRATCH[kb * 256 + {out_off} :+ {RUN}, pn * TN :+ TN] = transpose(decoded{ib})"
        );
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq4xs_dequant(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                     CODEBOOK: tensor<i32>[1, {IQ4XS_CODEBOOK_LEN}], SCRATCH: tensor<f32>[K, N]) {{
  let pn = program_id(0)
  let codebook = CODEBOOK[0 :+ 1, :]
  for kb in range(0, NB, 1) {{
    var qb = QB[pn * TN :+ TN, kb * {BLOCK_BYTES} :+ {BLOCK_BYTES}]
    let d = D[pn * TN :+ TN, kb :+ 1]
{body}  }}
}}
"
    )
}
