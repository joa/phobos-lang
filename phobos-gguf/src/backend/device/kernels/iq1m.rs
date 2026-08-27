// IQ1_M matvec: same raw-byte decode as iq1s.rs. IQ1_M shares IQ1_S's
// four-lane grid lookup but each group of 32 carries its own pair of 3-bit
// scales instead of one shared scale, and has no `f16` of its own;
// `quant/iq1_m.rs::raw_scales` reassembles the equivalent word host-side so
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

/// Byte offsets for lane `is` (ib = is/4, l = is%4, matching
/// quant/iq1_m.rs::dequantize). Lanes 0/2 read the group's first `qh` byte,
/// lanes 1/3 the second; index bits at bit 0 or 4, sign bit 8 higher.
fn run_geometry(is: usize) -> (usize, usize, usize, usize) {
    let ib = is / 4;
    let l = is % 4;
    let qs4_off = QS_OFF + 4 * ib + l;
    let qh_off = QH_OFF + 2 * ib + usize::from(l >= 2);
    let (idx_div, bit_div) = if l.is_multiple_of(2) { (1, 8) } else { (16, 128) };
    (qs4_off, qh_off, idx_div, bit_div)
}

/// Scale-word offsets for group `ib` and the divisor to reach lane `is`'s
/// 3-bit scale: two groups share one 16-bit word (low/high six bits each),
/// and within a group the first two lanes take the low 3 bits, last two the
/// high 3.
fn dl_geometry(is: usize) -> (usize, usize, usize) {
    let ib = is / 4;
    let l = is % 4;
    let sc_lo_off = SCALES_OFF + 2 * (ib / 2);
    let sc_hi_off = sc_lo_off + 1;
    let shift = 6 * (ib % 2);
    let dl_div = 2usize.pow((shift + if l < 2 { 0 } else { 3 }) as u32);
    (sc_lo_off, sc_hi_off, dl_div)
}

/// One lane's decoded value, `let decoded{is} = ...`, and its `out_off`.
fn decoded_lane(is: usize) -> (usize, String) {
    let (qs4_off, qh_off, idx_div, bit_div) = run_geometry(is);
    let (sc_lo_off, sc_hi_off, dl_div) = dl_geometry(is);
    let out_off = is * LANE;
    let byte = |off: usize| format!("((i32(qb[:, {off} :+ 1]) + 256) % 256)");
    let qs4 = byte(qs4_off);
    let qh = byte(qh_off);
    let word = format!("({} + {} * 256)", byte(sc_lo_off), byte(sc_hi_off));
    let dl = format!("(f32(d) * f32((({word} / {dl_div}) % 8) * 2 + 1))");
    let idx = format!("({qs4} + (({qh} / {idx_div}) % 8) * 256)");
    let delta = format!("(0.125 - 0.25 * f32(({qh} / {bit_div}) % 2))");
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
        let _ = writeln!(body, "{decode}    acc = acc + dot_t({a_is}, decoded{is})");
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq1m_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                   D: tensor<f16>[N, NB], GRID: tensor<i32>[1, {IQ1S_GRID_LEN}],
                   IOTA: tensor<i32>[1, {LANE}], C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  let pm = program_id(1)
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

/// [`iq1m_matvec_src`] for `m == 1`, folding the whole decode-and-reduce
/// into one `iq1m_qdot_t` call; see `iq1s_qdot_matvec_src`'s doc, which this
/// mirrors. `@aligned(N = TN)` is required for the same reason: `iq1m_qdot_t`
/// demands its `qb`/`d` slices provably in bounds.
/// Output tile for the dp4a variant; a warp takes two columns.
pub(crate) const IQ1M_I8_TN: usize = 64;

/// The tile for an `n` that 64 does not divide. A warp then takes two
/// columns rather than eight, which is slower but still well ahead of the
/// float path it would otherwise fall back to.
pub(crate) const IQ1M_I8_NARROW_TN: usize = 16;

/// [`iq1m_qdot_matvec_src`] against an int8-quantized activation, in dp4a.
pub(crate) fn iq1m_qdot_i8_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq1m_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                           QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                           GRID: tensor<i8>[1, {IQ1S_GRID_LEN}],
                           C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq1m_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                            QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                            GRID[0 :+ 1, :])
}}
"
    )
}

pub(crate) fn iq1m_qdot_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq1m_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                        D: tensor<f16>[N, NB], GRID: tensor<i8>[1, {IQ1S_GRID_LEN}],
                        C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq1m_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                         D[pn * TN :+ TN, :], GRID[0 :+ 1, :])
}}
"
    )
}

/// [`iq1m_matvec_src`]'s decode, stored straight into a `[K, N]` scratch
/// instead of reduced against an activation row; see `iq1s.rs`'s
/// `iq1s_dequant_src` for why.
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
  let pn = program_id(0)
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

/// [`iq1m_dequant_src`]'s decode as a single `iq1m_qdecode_t` call; see
/// `iq1s.rs`'s `iq1s_qdecode_src`, which this mirrors for IQ1_M's decode shares IQ1_S's grid outright.
pub(crate) fn iq1m_qdecode_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq1m_qdecode(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                    GRID: tensor<i8>[1, {IQ1S_GRID_LEN}],
                    SCRATCH: tensor<f32>[K, N]) {{
  let pn = program_id(0)
  SCRATCH[:, pn * TN :+ TN] = iq1m_qdecode_t(QB[pn * TN :+ TN, :],
                                             D[pn * TN :+ TN, :],
                                             GRID[0 :+ 1, :])
}}
"
    )
}
