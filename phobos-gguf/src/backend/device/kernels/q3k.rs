// Q3_K matvec: same raw-byte decode as q2k.rs. Each 3-bit quant is two bits
// from `qs` plus a third from `hmask` (one bit per element, packed across
// the whole 256-element block). All runs share scale `d`, but each has its
// own signed 6-bit multiplier unpacked from three interleaved bytes (see
// quant/q3_k.rs::unpack_scales); there is no minimum term, unlike Q2_K.

use std::fmt::Write as _;

/// Output columns a CTA covers.
pub(crate) const Q3K_TN: usize = 32;

const RUN: usize = 16;
const RUNS_PER_BLOCK: usize = 256 / RUN;
const BLOCK_BYTES: usize = 110;
const QS_OFF: usize = 32;
const SCALES_OFF: usize = 96;

/// Byte offsets and shift divisors for run `is`'s `qs` field and `hmask`
/// bit; same h/j/half2 decomposition as Q2_K's `run_geometry`.
fn run_geometry(is: usize) -> (usize, usize, usize, usize) {
    let h = is / 8;
    let rem = is % 8;
    let j = rem / 2;
    let half2 = rem % 2;
    let qs_off = QS_OFF + h * 32 + half2 * 16;
    let qs_div = 4usize.pow(j as u32);
    let hm_off = half2 * 16;
    let hm_div = 2usize.pow((h * 4 + j) as u32);
    (qs_off, qs_div, hm_off, hm_div)
}

/// Byte offsets and divisors for run `is`'s signed 6-bit scale in the
/// 12-byte `scales` plane (mirrors `unpack_scales`'s cross-byte layout).
fn scale_geometry(is: usize) -> (usize, usize, usize, usize) {
    let group = is / 4;
    let c = is % 4;
    let lo_off = SCALES_OFF + if group.is_multiple_of(2) { c } else { 4 + c };
    let lo_div = if group < 2 { 1 } else { 16 };
    let hi_off = SCALES_OFF + 8 + c;
    let hi_div = 4usize.pow(group as u32);
    (lo_off, lo_div, hi_off, hi_div)
}

/// One run's decoded value, `let scale{is}`/`let decoded{is} = ...`, and its
/// `out_off`.
fn decoded_run(is: usize) -> (usize, String) {
    let (qs_off, qs_div, hm_off, hm_div) = run_geometry(is);
    let (lo_off, lo_div, hi_off, hi_div) = scale_geometry(is);
    let out_off = is * RUN;
    let lo = format!("(((i32(qb[:, {lo_off} :+ 1]) + 256) % 256) / {lo_div} % 16)");
    let hi = format!("(((i32(qb[:, {hi_off} :+ 1]) + 256) % 256) / {hi_div} % 4)");
    let low2 = format!("(((i32(qb[:, {qs_off} :+ {RUN}]) + 256) % 256) / {qs_div} % 4)");
    let bit = format!("(((i32(qb[:, {hm_off} :+ {RUN}]) + 256) % 256) / {hm_div} % 2)");
    (
        out_off,
        format!(
            "    let scale{is} = f32({lo} + {hi} * 16) - 32.0
    let decoded{is} = f32(d) * scale{is} * (f32({low2}) - 4.0 + f32({bit}) * 4.0)\n"
        ),
    )
}

pub(crate) fn q3k_matvec_src(tn: usize) -> String {
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
kernel q3k_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                  D: tensor<f16>[N, NB], C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  let pm = program_id(1)
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

/// [`q3k_matvec_src`] for `m == 1`, folding the whole decode-and-reduce into
/// one `q3k_qdot_t` call; see `iq1s_qdot_matvec_src`'s doc, which this
/// mirrors. `@aligned(N = TN)` is required for the same reason: `q3k_qdot_t`
/// demands its `qb`/`d` slices provably in bounds.
pub(crate) fn q3k_qdot_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel q3k_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                       D: tensor<f16>[N, NB], C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = q3k_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :])
}}
"
    )
}
