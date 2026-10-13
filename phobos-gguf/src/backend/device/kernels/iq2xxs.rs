// IQ2_XXS matvec, with the same raw-byte decode as iq1s.rs plus a second
// `gather` for signs.
//
// The magnitude is a byte index into the grid. The sign is a 7-bit index
// into the sign table, taken from the group's four-byte `aux` field (see
// quant/iq2_xxs.rs). The scale is aux's top four bits. A sign index can
// cross a byte boundary, so it is assembled as `lo + hi * 256` first.

use std::fmt::Write as _;

/// Output columns per CTA.
pub(crate) const IQ2XXS_TN: usize = 8;

const LANES: usize = 32;
const LANE: usize = 8;
const BLOCK_BYTES: usize = 64;
const QS_OFF: usize = 0;

/// [`crate::quant::iq2xxs_flat_grid`]'s length: 256 grid entries of eight
/// lanes each.
pub(crate) const IQ2XXS_GRID_LEN: usize = 256 * 8;
/// [`crate::quant::iq2xxs_flat_signs`]'s length: eight sign multipliers for
/// each of 128 sign indices.
pub(crate) const IQ2XXS_SIGNS_LEN: usize = 128 * 8;

/// Byte offsets for lane `is`, with `ib32 = is / 4` and `l = is % 4` as in
/// `quant/iq2_xxs.rs::dequantize`. `hi_off` is `None` for `l == 0`, whose
/// sign bits fit in one byte.
fn run_geometry(is: usize) -> (usize, usize, usize, Option<usize>, usize) {
    let ib32 = is / 4;
    let l = is % 4;
    let chunk = QS_OFF + 8 * ib32;
    let aux = chunk + 4;
    let grid_off = chunk + l;
    let scale_off = aux + 3;
    let (lo_off, hi_off, shift) = match l {
        0 => (aux, None, 0),
        1 => (aux, Some(aux + 1), 7),
        2 => (aux + 1, Some(aux + 2), 6),
        _ => (aux + 2, Some(aux + 3), 5),
    };
    (grid_off, scale_off, lo_off, hi_off, shift)
}

/// One lane's decoded value, `let decoded{is} = ...`, and its `out_off`.
fn decoded_lane(is: usize) -> (usize, String) {
    let (grid_off, scale_off, lo_off, hi_off, shift) = run_geometry(is);
    let out_off = is * LANE;
    let byte = |off: usize| format!("(i32(qb[:, {off} :+ 1]) & 255)");
    let grid_idx = byte(grid_off);
    let scale = byte(scale_off);
    let lo = byte(lo_off);
    let signs_idx = match hi_off {
        None => format!("({lo} & 127)"),
        Some(hi_off) => {
            let hi = byte(hi_off);
            format!("((({lo} + {hi} * 256) >> {shift}) & 127)")
        }
    };
    let db = format!("(f32(d) * (0.5 + f32({scale} >> 4)) * 0.25)");
    let mag_idx8 = format!("({grid_idx} * {LANE} + iota)");
    let sign_idx8 = format!("({signs_idx} * {LANE} + iota)");
    (
        out_off,
        format!(
            "    let decoded{is} = {db} * f32(gather(grid, {mag_idx8})) * f32(gather(signs, {sign_idx8}))\n"
        ),
    )
}

pub(crate) fn iq2xxs_matvec_src(tn: usize) -> String {
    let mut body = String::new();
    for is in 0..LANES {
        let (out_off, decode) = decoded_lane(is);
        let a_is = format!("A[pm :+ 1, kb * 256 + {out_off} :+ {LANE}]");
        let _ = writeln!(body, "{decode}    acc += dot_t({a_is}, decoded{is})");
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq2xxs_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                     D: tensor<f16>[N, NB], GRID: tensor<i32>[1, {IQ2XXS_GRID_LEN}],
                     SIGNS: tensor<i32>[1, {IQ2XXS_SIGNS_LEN}], IOTA: tensor<i32>[1, {LANE}],
                     C: tensor<f32>[M, N]) {{
  let pn = ctx.block.x
  let pm = ctx.block.y
  let grid = GRID[0 :+ 1, :]
  let signs = SIGNS[0 :+ 1, :]
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

/// [`iq2xxs_matvec_src`] for `m == 1`, written with lanes; see `iq_f32.rs`.
/// [`crate::backend::device::DeviceBackend::project_raw`] ensures `N` is a
/// multiple of `tn`, and falls back to [`iq2xxs_matvec_src`] otherwise.
pub(crate) fn iq2xxs_qdot_matvec_src(tn: usize) -> String {
    super::qdot_matvec_src(super::IqFmt::Iq2xxs, tn)
}

/// [`iq2xxs_matvec_src`]'s decode, stored into a `[K, N]` scratch instead
/// of reduced against an activation row; see `iq1s_dequant_src` in
/// `iq1s.rs`.
pub(crate) fn iq2xxs_dequant_src(tn: usize) -> String {
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
kernel iq2xxs_dequant(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                      GRID: tensor<i32>[1, {IQ2XXS_GRID_LEN}], SIGNS: tensor<i32>[1, {IQ2XXS_SIGNS_LEN}],
                      IOTA: tensor<i32>[1, {LANE}], SCRATCH: tensor<f32>[K, N]) {{
  let pn = ctx.block.x
  let grid = GRID[0 :+ 1, :]
  let signs = SIGNS[0 :+ 1, :]
  let iota = IOTA[0 :+ 1, :]
  for kb in range(0, NB, 1) {{
    var qb = QB[pn * TN :+ TN, kb * {BLOCK_BYTES} :+ {BLOCK_BYTES}]
    let d = D[pn * TN :+ TN, kb :+ 1]
{body}  }}
}}
"
    )
}

/// [`iq2xxs_dequant_src`]'s decode, written with lanes; see `iq_f32.rs`.
/// [`crate::backend::device::DeviceBackend::project_raw_dense`] uses
/// [`iq2xxs_dequant_src`] for a strip that is not a whole number of `TN`.
pub(crate) fn iq2xxs_qdecode_src(tn: usize) -> String {
    super::qdecode_src(super::IqFmt::Iq2xxs, tn)
}

/// IQ2_XXS's prompt projection, decode and contraction in one kernel, like
/// [`super::iq1s::iq1s_qmma_src`].
pub(crate) fn iq2xxs_qmma_src(block: usize, tm: usize, tn: usize) -> String {
    format!(
        "@launch({block})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 256)
kernel iq2xxs_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                   QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                   GRID: tensor<i8>[1, {IQ2XXS_GRID_LEN}],
                   SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}],
                   C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  C[pm * TM :+ TM, pn * TN :+ TN] = iq2xxs_qmma_staged_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                         QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                         GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}
