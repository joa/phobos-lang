// IQ2_XXS matvec: same raw-byte decode as iq1s.rs, with a second batched
// `gather` for signs. Magnitude is a byte index into IQ2XXS_GRID; sign is a
// 7-bit index into KSIGNS_IQ2XS built from the group's four-byte `aux`
// field (see quant/iq2_xxs.rs). The scale lives in aux's top four bits. A
// lane's 7-bit sign index can cross a byte boundary, so it's assembled as
// `lo + hi * 256` from two corrected bytes before the div/mod.

use std::fmt::Write as _;

/// Output columns per CTA.
pub(crate) const IQ2XXS_TN: usize = 8;

const LANES: usize = 32;
const LANE: usize = 8;
const BLOCK_BYTES: usize = 66;
const QS_OFF: usize = 2;

/// [`crate::quant::iq2xxs_flat_grid`]'s length: 256 grid entries, eight
/// `i32` lanes apiece.
pub(crate) const IQ2XXS_GRID_LEN: usize = 256 * 8;
/// [`crate::quant::iq2xxs_flat_signs`]'s length: 128 sign indices, eight
/// `i32` multipliers apiece.
pub(crate) const IQ2XXS_SIGNS_LEN: usize = 128 * 8;

/// Byte offsets for lane `is` (ib32 = is/4, l = is%4, matching
/// quant/iq2_xxs.rs::dequantize). `hi_off` is `None` for `l == 0`, whose
/// sign bits fit entirely in `lo_off`'s byte.
fn run_geometry(is: usize) -> (usize, usize, usize, Option<usize>, usize) {
    let ib32 = is / 4;
    let l = is % 4;
    let chunk = QS_OFF + 8 * ib32;
    let aux = chunk + 4;
    let grid_off = chunk + l;
    let scale_off = aux + 3;
    let (lo_off, hi_off, shift_div) = match l {
        0 => (aux, None, 1),
        1 => (aux, Some(aux + 1), 128),
        2 => (aux + 1, Some(aux + 2), 64),
        _ => (aux + 2, Some(aux + 3), 32),
    };
    (grid_off, scale_off, lo_off, hi_off, shift_div)
}

/// One lane's decoded value, `let decoded{is} = ...`, and its `out_off`.
fn decoded_lane(is: usize) -> (usize, String) {
    let (grid_off, scale_off, lo_off, hi_off, shift_div) = run_geometry(is);
    let out_off = is * LANE;
    let byte = |off: usize| format!("((i32(qb[:, {off} :+ 1]) + 256) % 256)");
    let grid_idx = byte(grid_off);
    let scale = byte(scale_off);
    let lo = byte(lo_off);
    let signs_idx = match hi_off {
        None => format!("({lo} % 128)"),
        Some(hi_off) => {
            let hi = byte(hi_off);
            format!("((({lo} + {hi} * 256) / {shift_div}) % 128)")
        }
    };
    let db = format!("(f32(d) * (0.5 + f32({scale} / 16)) * 0.25)");
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
        let _ = writeln!(body, "{decode}    acc = acc + dot_t({a_is}, decoded{is})");
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq2xxs_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                     D: tensor<f16>[N, NB], GRID: tensor<i32>[1, {IQ2XXS_GRID_LEN}],
                     SIGNS: tensor<i32>[1, {IQ2XXS_SIGNS_LEN}], IOTA: tensor<i32>[1, {LANE}],
                     C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  let pm = program_id(1)
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

/// [`iq2xxs_matvec_src`] for `m == 1`, folding the whole decode-and-reduce
/// into one `iq2xxs_qdot_t` call; see `iq1s_qdot_matvec_src`'s doc, which
/// this mirrors. `@aligned(N = TN)` is required for the same reason:
/// `iq2xxs_qdot_t` demands its `qb`/`d` slices provably in bounds.
pub(crate) fn iq2xxs_qdot_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq2xxs_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                          D: tensor<f16>[N, NB], GRID: tensor<i8>[1, {IQ2XXS_GRID_LEN}],
                          SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}], C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq2xxs_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                           D[pn * TN :+ TN, :], GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}

/// [`iq2xxs_matvec_src`]'s decode, stored straight into a `[K, N]` scratch
/// instead of reduced against an activation row; see `iq1s.rs`'s
/// `iq1s_dequant_src` for why.
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
  let pn = program_id(0)
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

/// [`iq2xxs_dequant_src`]'s decode as a single `iq2xxs_qdecode_t` call; see
/// `iq1s.rs`'s `iq1s_qdecode_src`, which this mirrors with the sign table
/// alongside the magnitude grid.
pub(crate) fn iq2xxs_qdecode_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq2xxs_qdecode(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                      GRID: tensor<i8>[1, {IQ2XXS_GRID_LEN}],
                      SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}],
                      SCRATCH: tensor<f32>[K, N]) {{
  let pn = program_id(0)
  SCRATCH[:, pn * TN :+ TN] = iq2xxs_qdecode_t(QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                               GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}
