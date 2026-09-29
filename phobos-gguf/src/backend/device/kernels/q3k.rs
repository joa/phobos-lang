// Q3_K matvec, with the same raw-byte decode as q2k.rs.
//
// Each 3-bit quant is two bits from `qs` plus a third from `hmask`, which
// packs one bit per element across the 256-element block. All runs share
// scale `d`, and each has its own signed 6-bit multiplier spread over
// interleaved bytes (see quant/q3_k.rs::unpack_scales). Unlike Q2_K there
// is no minimum term.

use std::fmt::Write as _;

/// Output columns a CTA covers.
pub(crate) const Q3K_TN: usize = 32;

const RUN: usize = 16;
const RUNS_PER_BLOCK: usize = 256 / RUN;
// The device stride, not the packed 110. `constant_raw` pads Q3_K blocks
// to eight-byte alignment, and the two trailing bytes are never read.
const BLOCK_BYTES: usize = 112;
const QS_OFF: usize = 32;
const SCALES_OFF: usize = 96;

/// Byte offsets and shift divisors for run `is`'s `qs` bits and `hmask`
/// bit, split as in Q2_K's `run_geometry`.
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
/// 12-byte `scales` field, following `unpack_scales`.
fn scale_geometry(is: usize) -> (usize, usize, usize, usize) {
    let group = is / 4;
    let c = is % 4;
    let lo_off = SCALES_OFF + if group.is_multiple_of(2) { c } else { 4 + c };
    let lo_div = if group < 2 { 1 } else { 16 };
    let hi_off = SCALES_OFF + 8 + c;
    let hi_div = 4usize.pow(group as u32);
    (lo_off, lo_div, hi_off, hi_div)
}

/// One run's decode, `let scale{is}` and `let decoded{is} = ...`, with its
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

/// [`q3k_matvec_src`] for `m == 1`, as one `q3k_qdot_t` call; mirrors
/// `iq1s_qdot_matvec_src`. `@aligned(N = TN)` is required, because
/// `q3k_qdot_t` needs its `qb` and `d` slices provably in bounds.
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
