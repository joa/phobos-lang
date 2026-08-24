// IQ2_S matvec: same raw-byte decode as iq1s.rs, structurally close to
// IQ2_XXS's two batched gathers a lane, but the magnitude index is a plain
// qs byte widened by two qh bits, and the sign byte is tested directly
// against KMASK_IQ2XS rather than through IQ2_XXS's parity table (see
// quant/iq2_s.rs).

use std::fmt::Write as _;

/// Output columns per CTA.
pub(crate) const IQ2S_TN: usize = 8;

const LANES: usize = 32;
const LANE: usize = 8;
const BLOCK_BYTES: usize = 82;
const QS_OFF: usize = 2;
const SIGNS_OFF: usize = QS_OFF + 32;
const QH_OFF: usize = SIGNS_OFF + 32;
const SCALES_OFF: usize = QH_OFF + 8;

/// [`crate::quant::iq2s_flat_grid`]'s length: 1024 grid entries, eight
/// `i32` lanes apiece.
pub(crate) const IQ2S_GRID_LEN: usize = 1024 * 8;
/// [`crate::quant::iq2s_flat_signs`]'s length: every byte value, eight
/// `i32` multipliers apiece.
pub(crate) const IQ2S_SIGNS_LEN: usize = 256 * 8;

/// Byte offsets for lane `is` (ib32 = is/4, l = is%4, matching
/// quant/iq2_s.rs::dequantize).
fn run_geometry(is: usize) -> (usize, usize, usize, usize, usize) {
    let ib32 = is / 4;
    let l = is % 4;
    let grid_off = QS_OFF + 4 * ib32 + l;
    let sign_off = SIGNS_OFF + 4 * ib32 + l;
    let qh_off = QH_OFF + ib32;
    let scale_off = SCALES_OFF + ib32;
    let qh_div = 4usize.pow(l as u32);
    (grid_off, sign_off, qh_off, scale_off, qh_div)
}

/// One lane's decoded value, `let decoded{is} = ...`, and its `out_off`.
fn decoded_lane(is: usize) -> (usize, String) {
    let (grid_off, sign_off, qh_off, scale_off, qh_div) = run_geometry(is);
    let out_off = is * LANE;
    let byte = |off: usize| format!("((i32(qb[:, {off} :+ 1]) + 256) % 256)");
    let grid_byte = byte(grid_off);
    let sign_byte = byte(sign_off);
    let qh_byte = byte(qh_off);
    let scale_byte = byte(scale_off);
    let nibble = if is % 4 < 2 {
        format!("({scale_byte} % 16)")
    } else {
        format!("({scale_byte} / 16)")
    };
    let dl = format!("(f32(d) * (0.5 + f32({nibble})) * 0.25)");
    let grid_idx = format!("({grid_byte} + (({qh_byte} / {qh_div}) % 4) * 256)");
    let mag_idx8 = format!("({grid_idx} * {LANE} + iota)");
    let sign_idx8 = format!("({sign_byte} * {LANE} + iota)");
    (
        out_off,
        format!(
            "    let decoded{is} = {dl} * f32(gather(grid, {mag_idx8})) * f32(gather(signs, {sign_idx8}))\n"
        ),
    )
}

pub(crate) fn iq2s_matvec_src(tn: usize) -> String {
    let mut body = String::new();
    for is in 0..LANES {
        let (out_off, decode) = decoded_lane(is);
        let a_is = format!("A[pm :+ 1, kb * 256 + {out_off} :+ {LANE}]");
        let _ = writeln!(body, "{decode}    acc = acc + dot_t({a_is}, decoded{is})");
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq2s_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                   D: tensor<f16>[N, NB], GRID: tensor<i32>[1, {IQ2S_GRID_LEN}],
                   SIGNS: tensor<i32>[1, {IQ2S_SIGNS_LEN}], IOTA: tensor<i32>[1, {LANE}],
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

/// [`iq2s_matvec_src`] for `m == 1`, folding the whole decode-and-reduce
/// into one `iq2s_qdot_t` call; see `iq1s_qdot_matvec_src`'s doc, which this
/// mirrors. `@aligned(N = TN)` is required for the same reason: `iq2s_qdot_t`
/// demands its `qb`/`d` slices provably in bounds.
pub(crate) fn iq2s_qdot_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq2s_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                        D: tensor<f16>[N, NB], GRID: tensor<i32>[1, {IQ2S_GRID_LEN}],
                        SIGNS: tensor<i32>[1, {IQ2S_SIGNS_LEN}], C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq2s_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                         D[pn * TN :+ TN, :], GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}

/// [`iq2s_matvec_src`]'s decode, stored straight into a `[K, N]` scratch
/// instead of reduced against an activation row; see `iq1s.rs`'s
/// `iq1s_dequant_src` for why.
pub(crate) fn iq2s_dequant_src(tn: usize) -> String {
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
kernel iq2s_dequant(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                    GRID: tensor<i32>[1, {IQ2S_GRID_LEN}], SIGNS: tensor<i32>[1, {IQ2S_SIGNS_LEN}],
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
