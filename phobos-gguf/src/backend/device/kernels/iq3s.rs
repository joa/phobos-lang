// IQ3_S matvec: same raw-byte decode as iq1s.rs, IQ3_XXS's two four-wide
// grid entries a lane, but IQ2_S's direct sign byte rather than a parity
// index (see quant/iq3_s.rs), so this reuses `iq2s_flat_signs`.
//
// The two four-wide gathers land in two separate `dot_t` calls, each
// reading its iota offsets straight from the `IOTA` parameter rather than
// reslicing a `let`-bound view, which cannot be sliced again.

use std::fmt::Write as _;

use super::IQ2S_SIGNS_LEN;

/// Output columns per CTA.
pub(crate) const IQ3S_TN: usize = 8;

const LANES: usize = 32;
const HALF: usize = 4;
const BLOCK_BYTES: usize = 108;
const QS_OFF: usize = 0;
const QH_OFF: usize = QS_OFF + 64;
const SIGNS_OFF: usize = QH_OFF + 8;
const SCALES_OFF: usize = SIGNS_OFF + 32;

/// [`crate::quant::iq3s_flat_grid`]'s length: 512 grid entries, four `i32`
/// lanes apiece.
pub(crate) const IQ3S_GRID_LEN: usize = 512 * 4;

/// Byte offsets for lane `is` (o/half/l, matching
/// quant/iq3_s.rs::dequantize's group/half/lane decomposition).
fn run_geometry(is: usize) -> (usize, usize, usize, usize, usize, usize, usize) {
    let o = is / 8;
    let rem = is % 8;
    let half = rem / 4;
    let l = rem % 4;
    let grid1_off = QS_OFF + 16 * o + 8 * half + 2 * l;
    let qh_off = QH_OFF + 2 * o + half;
    let signs_off = SIGNS_OFF + 8 * o + 4 * half + l;
    let scale_off = SCALES_OFF + o;
    let (qh_div1, qh_div2) = (2usize.pow(2 * l as u32), 2usize.pow(2 * l as u32 + 1));
    (
        grid1_off,
        grid1_off + 1,
        qh_off,
        signs_off,
        scale_off,
        qh_div1,
        qh_div2,
    )
}

/// Both of one lane's decoded halves, `let decoded{is}_{half} = ...`, and
/// each half's own `out_off`.
fn decoded_lane(is: usize) -> [(usize, String); 2] {
    let (g1_off, g2_off, qh_off, signs_off, scale_off, qh_div1, qh_div2) = run_geometry(is);
    let out_off = is * 8;
    let byte = |off: usize| format!("((i32(qb[:, {off} :+ 1]) + 256) % 256)");
    let qh_byte = byte(qh_off);
    let signs_byte = byte(signs_off);
    let scale_byte = byte(scale_off);
    let nibble = if is % 8 < 4 {
        format!("({scale_byte} % 16)")
    } else {
        format!("({scale_byte} / 16)")
    };
    let db = format!("(f32(d) * f32(1 + 2 * {nibble}))");
    [(g1_off, qh_div1, 0), (g2_off, qh_div2, 4)].map(|(grid_off, qh_div, sign_shift)| {
        let half = usize::from(sign_shift != 0);
        let iota4 = format!("IOTA[0 :+ 1, 0 :+ {HALF}]");
        let grid_idx = format!("({} + (({qh_byte} / {qh_div}) % 2) * 256)", byte(grid_off));
        let grid_idx8 = format!("({grid_idx} * {HALF} + {iota4})");
        let sign_idx8 = format!("({signs_byte} * 8 + {sign_shift} + {iota4})");
        (
            out_off + half * HALF,
            format!(
                "    let decoded{is}_{half} = {db} * f32(gather(grid, {grid_idx8})) * f32(gather(signs, {sign_idx8}))\n"
            ),
        )
    })
}

pub(crate) fn iq3s_matvec_src(tn: usize) -> String {
    let mut body = String::new();
    for is in 0..LANES {
        for (half, (out_off, decode)) in decoded_lane(is).into_iter().enumerate() {
            let a_half = format!("A[pm :+ 1, kb * 256 + {out_off} :+ {HALF}]");
            let _ = writeln!(body, "{decode}    acc = acc + dot_t({a_half}, decoded{is}_{half})");
        }
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq3s_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                   D: tensor<f16>[N, NB], GRID: tensor<i32>[1, {IQ3S_GRID_LEN}],
                   SIGNS: tensor<i32>[1, {IQ2S_SIGNS_LEN}], IOTA: tensor<i32>[1, 8],
                   C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  let pm = program_id(1)
  let grid = GRID[0 :+ 1, :]
  let signs = SIGNS[0 :+ 1, :]
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

/// [`iq3s_matvec_src`] for `m == 1`, folding the whole decode-and-reduce
/// into one `iq3s_qdot_t` call; see `iq1s_qdot_matvec_src`'s doc, which this
/// mirrors. `@aligned(N = TN)` is required for the same reason: `iq3s_qdot_t`
/// demands its `qb`/`d` slices provably in bounds.
pub(crate) fn iq3s_qdot_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq3s_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                        D: tensor<f16>[N, NB], GRID: tensor<i8>[1, {IQ3S_GRID_LEN}],
                        SIGNS: tensor<i8>[1, {IQ2S_SIGNS_LEN}], C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq3s_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                         D[pn * TN :+ TN, :], GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}

/// [`iq3s_matvec_src`]'s decode, stored straight into a `[K, N]` scratch
/// instead of reduced against an activation row; see `iq1s.rs`'s
/// `iq1s_dequant_src` for why.
pub(crate) fn iq3s_dequant_src(tn: usize) -> String {
    let mut body = String::new();
    for is in 0..LANES {
        for (half, (out_off, decode)) in decoded_lane(is).into_iter().enumerate() {
            let _ = writeln!(
                body,
                "{decode}    SCRATCH[kb * 256 + {out_off} :+ {HALF}, pn * TN :+ TN] = transpose(decoded{is}_{half})"
            );
        }
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq3s_dequant(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                    GRID: tensor<i32>[1, {IQ3S_GRID_LEN}], SIGNS: tensor<i32>[1, {IQ2S_SIGNS_LEN}],
                    IOTA: tensor<i32>[1, 8], SCRATCH: tensor<f32>[K, N]) {{
  let pn = program_id(0)
  let grid = GRID[0 :+ 1, :]
  let signs = SIGNS[0 :+ 1, :]
  for kb in range(0, NB, 1) {{
    var qb = QB[pn * TN :+ TN, kb * {BLOCK_BYTES} :+ {BLOCK_BYTES}]
    let d = D[pn * TN :+ TN, kb :+ 1]
{body}  }}
}}
"
    )
}

/// [`iq3s_dequant_src`]'s decode as a single `iq3s_qdecode_t` call; see
/// `iq1s.rs`'s `iq1s_qdecode_src`, which this mirrors for IQ3_S.
pub(crate) fn iq3s_qdecode_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq3s_qdecode(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                    GRID: tensor<i8>[1, {IQ3S_GRID_LEN}],
                    SIGNS: tensor<i8>[1, {IQ2S_SIGNS_LEN}],
                    SCRATCH: tensor<f32>[K, N]) {{
  let pn = program_id(0)
  SCRATCH[:, pn * TN :+ TN] = iq3s_qdecode_t(QB[pn * TN :+ TN, :],
                                             D[pn * TN :+ TN, :],
                                             GRID[0 :+ 1, :],
                                             SIGNS[0 :+ 1, :])
}}
"
    )
}
