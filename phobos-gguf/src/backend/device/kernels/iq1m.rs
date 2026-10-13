// IQ1_M matvec, with the same raw-byte decode as iq1s.rs.
//
// IQ1_M shares IQ1_S's four-lane grid lookup, but each group of 32 carries
// its own pair of 3-bit scales, and the block has no `f16` of its own.
// `quant/iq1_m.rs::raw_scales` rebuilds the equivalent word on the host, so
// `d` here is still a plain per-block read.


use std::fmt::Write as _;

use super::IQ1S_GRID_LEN;

/// Output columns per CTA.
pub(crate) const IQ1M_TN: usize = 8;

const LANES: usize = 32;
const LANE: usize = 8;
const BLOCK_BYTES: usize = 56;
const QS_OFF: usize = 0;
const QH_OFF: usize = 32;
const SCALES_OFF: usize = 48;

/// Byte offsets and bit shifts for lane `is`, with `ib = is / 4` and
/// `l = is % 4` as in `quant/iq1_m.rs::dequantize`.
///
/// Lanes 0 and 1 read the group's first `qh` byte, lanes 2 and 3 the
/// second. Even lanes take the low nibble and odd lanes the high one.
fn run_geometry(is: usize) -> (usize, usize, usize, usize) {
    let ib = is / 4;
    let l = is % 4;
    let qs4_off = QS_OFF + 4 * ib + l;
    let qh_off = QH_OFF + 2 * ib + usize::from(l >= 2);
    let (idx_shift, bit_shift) = if l.is_multiple_of(2) { (0, 3) } else { (4, 7) };
    (qs4_off, qh_off, idx_shift, bit_shift)
}

/// Scale-word offsets for lane `is`'s group, and the shift that reaches
/// its 3-bit scale.
///
/// Two groups share one 16-bit word, six bits each. Within a group, the
/// first two lanes take the low 3 bits and the last two the high 3.
fn dl_geometry(is: usize) -> (usize, usize, usize) {
    let ib = is / 4;
    let l = is % 4;
    let sc_lo_off = SCALES_OFF + 2 * (ib / 2);
    let sc_hi_off = sc_lo_off + 1;
    let shift = 6 * (ib % 2);
    (sc_lo_off, sc_hi_off, shift + if l < 2 { 0 } else { 3 })
}

/// One lane's decoded value, `let decoded{is} = ...`, and its `out_off`.
fn decoded_lane(is: usize) -> (usize, String) {
    let (qs4_off, qh_off, idx_shift, bit_shift) = run_geometry(is);
    let (sc_lo_off, sc_hi_off, dl_shift) = dl_geometry(is);
    let out_off = is * LANE;
    let byte = |off: usize| format!("(i32(qb[:, {off} :+ 1]) & 255)");
    let qs4 = byte(qs4_off);
    let qh = byte(qh_off);
    let word = format!("({} + {} * 256)", byte(sc_lo_off), byte(sc_hi_off));
    let dl = format!("(f32(d) * f32((({word} >> {dl_shift}) & 7) * 2 + 1))");
    let idx = format!("({qs4} + (({qh} >> {idx_shift}) & 7) * 256)");
    let delta = format!("(0.125 - 0.25 * f32(({qh} >> {bit_shift}) & 1))");
    let idx8 = format!("({idx} * {LANE} + iota)");
    (
        out_off,
        format!("    let decoded{is} = {dl} * (f32(gather(grid, {idx8})) + {delta})\n"),
    )
}

pub(crate) fn iq1m_matvec_src(tn: usize) -> String {
    let mut body = String::new();
    for is in 0..LANES {
        let (out_off, decode) = decoded_lane(is);
        let a_is = format!("A[pm :+ 1, kb * 256 + {out_off} :+ {LANE}]");
        let _ = writeln!(body, "{decode}    acc += dot_t({a_is}, decoded{is})");
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq1m_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                   D: tensor<f16>[N, NB], GRID: tensor<i32>[1, {IQ1S_GRID_LEN}],
                   IOTA: tensor<i32>[1, {LANE}], C: tensor<f32>[M, N]) {{
  let pn = ctx.block.x
  let pm = ctx.block.y
  let grid = GRID[0 :+ 1, :]
  let iota = IOTA[0 :+ 1, :]
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

/// [`iq1m_matvec_src`] for `m == 1`, written with lanes; see `iq_f32.rs`.
/// [`crate::backend::device::DeviceBackend::project_raw`] ensures `N` is a
/// multiple of `tn`, and falls back to [`iq1m_matvec_src`] otherwise.
pub(crate) fn iq1m_qdot_matvec_src(tn: usize) -> String {
    super::qdot_matvec_src(super::IqFmt::Iq1m, tn)
}

/// [`iq1m_matvec_src`]'s decode, stored into a `[K, N]` scratch instead of
/// reduced against an activation row; see `iq1s_dequant_src` in `iq1s.rs`.
pub(crate) fn iq1m_dequant_src(tn: usize) -> String {
    let mut body = String::new();
    for is in 0..LANES {
        let (out_off, decode) = decoded_lane(is);
        let _ = writeln!(
            body,
            "{decode}    SCRATCH[kb * 256 + {out_off} :+ {LANE}, pn * TN :+ TN] = transpose(decoded{is})"
        );
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq1m_dequant(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                    GRID: tensor<i32>[1, {IQ1S_GRID_LEN}], IOTA: tensor<i32>[1, {LANE}],
                    SCRATCH: tensor<f32>[K, N]) {{
  let pn = ctx.block.x
  let grid = GRID[0 :+ 1, :]
  let iota = IOTA[0 :+ 1, :]
  for kb in range(0, NB, 1) {{
    var qb = QB[pn * TN :+ TN, kb * {BLOCK_BYTES} :+ {BLOCK_BYTES}]
    let d = D[pn * TN :+ TN, kb :+ 1]
{body}  }}
}}
"
    )
}

/// [`iq1m_dequant_src`]'s decode, written with lanes; see `iq_f32.rs`.
/// [`crate::backend::device::DeviceBackend::project_raw_dense`] uses
/// [`iq1m_dequant_src`] for a strip that is not a whole number of `TN`.
pub(crate) fn iq1m_qdecode_src(tn: usize) -> String {
    super::qdecode_src(super::IqFmt::Iq1m, tn)
}
