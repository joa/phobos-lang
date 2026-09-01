// Q2_K matvec: decodes straight from the raw block bytes, all sixteen runs
// at static offsets (run order is output order), so the block unrolls into
// one straight-line expression per run instead of a `gather`. Every raw
// byte goes through `(i32(b) + 256) % 256` before any `/` or `%`, since
// Q2_K bytes routinely have bit 7 set and a signed remainder gives the
// wrong nibble silently.
//
// Each run is one inlined expression rather than named intermediates: a
// `let` tile never releases its buffer, so named per-run steps blew the
// 48 KB shared memory ceiling by 2x. Inlined (scale reread rather than
// named), it fits in 43 KB.

use std::fmt::Write as _;

/// Output columns a CTA covers.
pub(crate) const Q2K_TN: usize = 32;

const RUN: usize = 16;
const RUNS_PER_BLOCK: usize = 256 / RUN;
const BLOCK_BYTES: usize = 84;
const QS_OFF: usize = 16;

/// Byte offset and shift divisor for run `is` (h/j/half2 decomposition,
/// matching quant/q2_k.rs::dequantize).
fn run_geometry(is: usize) -> (usize, usize) {
    let h = is / 8;
    let rem = is % 8;
    let j = rem / 2;
    let half2 = rem % 2;
    (QS_OFF + h * 32 + half2 * 16, 4usize.pow(j as u32))
}

/// One run's decoded value, `let decoded{is} = ...`, and its `out_off`.
fn decoded_run(is: usize) -> (usize, String) {
    let (qs_off, shift_div) = run_geometry(is);
    let out_off = is * RUN;
    let sc = format!("((i32(qb[:, {is} :+ 1]) + 256) % 256)");
    let q2 = format!("(((i32(qb[:, {qs_off} :+ {RUN}]) + 256) % 256 / {shift_div}) % 4)");
    (
        out_off,
        format!(
            "    let decoded{is} = f32(d) * f32({sc} % 16) * f32({q2}) - f32(dmin) * f32({sc} / 16)\n"
        ),
    )
}

pub(crate) fn q2k_matvec_src(tn: usize) -> String {
    let mut body = String::new();
    for is in 0..RUNS_PER_BLOCK {
        let (out_off, decode) = decoded_run(is);
        let _ = writeln!(
            body,
            "{decode}    let a{is} = A[pm :+ 1, kb * 256 + {out_off} :+ {RUN}]
    acc = acc + dot_t(a{is}, decoded{is})"
        );
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel q2k_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                  D: tensor<f16>[N, NB], DMIN: tensor<f16>[N, NB],
                  C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  let pm = program_id(1)
  var acc: tile<f32>[1, TN] = 0.0
  for kb in range(0, NB, 1) {{
    var qb = QB[pn * TN :+ TN, kb * {BLOCK_BYTES} :+ {BLOCK_BYTES}]
    let d = D[pn * TN :+ TN, kb :+ 1]
    let dmin = DMIN[pn * TN :+ TN, kb :+ 1]
{body}  }}
  C[pm :+ 1, pn * TN :+ TN] = acc
}}
"
    )
}

/// [`q2k_matvec_src`] for `m == 1`, folding the whole decode-and-reduce into
/// one `q2k_qdot_t` call; see `iq1s_qdot_matvec_src`'s doc, which this
/// mirrors. `@aligned(N = TN)` is required for the same reason: `q2k_qdot_t`
/// demands its `qb`/`d`/`dmin` slices provably in bounds.
pub(crate) fn q2k_qdot_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel q2k_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                       D: tensor<f16>[N, NB], DMIN: tensor<f16>[N, NB],
                       C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = q2k_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                        D[pn * TN :+ TN, :], DMIN[pn * TN :+ TN, :])
}}
"
    )
}

/// [`q2k_matvec_src`]'s decode, stored straight into a `[K, N]` scratch
/// instead of reduced against an activation row; see `iq1s.rs`'s
/// `iq1s_dequant_src` for why.
pub(crate) fn q2k_dequant_src(tn: usize) -> String {
    let mut body = String::new();
    for is in 0..RUNS_PER_BLOCK {
        let (out_off, decode) = decoded_run(is);
        let _ = writeln!(
            body,
            "{decode}    SCRATCH[kb * 256 + {out_off} :+ {RUN}, pn * TN :+ TN] = transpose(decoded{is})"
        );
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel q2k_dequant(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB], DMIN: tensor<f16>[N, NB],
                   SCRATCH: tensor<f32>[K, N]) {{
  let pn = program_id(0)
  for kb in range(0, NB, 1) {{
    var qb = QB[pn * TN :+ TN, kb * {BLOCK_BYTES} :+ {BLOCK_BYTES}]
    let d = D[pn * TN :+ TN, kb :+ 1]
    let dmin = DMIN[pn * TN :+ TN, kb :+ 1]
{body}  }}
}}
"
    )
}
