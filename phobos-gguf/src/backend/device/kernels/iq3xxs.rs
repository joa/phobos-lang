// IQ3_XXS matvec: same raw-byte decode as iq2xxs.rs; its scale-and-sign
// field is byte-for-byte IQ2_XXS's own `aux32` (see quant/iq3_xxs.rs), so
// this reuses IQ2_XXS's flattened sign table. Grid entries are four bytes
// (`[u32; 256]`, not IQ2_XXS's `[u64; 256]`), so each lane needs two
// four-wide gathers landing in two separate `dot_t` calls. `IOTA[0 :+ 1,
// 0 :+ 4]` reads offsets straight from the tensor parameter for both,
// since a `let`-bound view cannot be resliced.

use std::fmt::Write as _;

use super::IQ2XXS_SIGNS_LEN;

/// Output columns per CTA.
pub(crate) const IQ3XXS_TN: usize = 8;

const LANES: usize = 32;
const HALF: usize = 4;
const BLOCK_BYTES: usize = 96;
const QS_OFF: usize = 0;
const SS_OFF: usize = QS_OFF + 64;

/// [`crate::quant::iq3xxs_flat_grid`]'s length: 256 grid entries, four
/// `i32` lanes apiece.
pub(crate) const IQ3XXS_GRID_LEN: usize = 256 * 4;

/// Byte offsets for lane `is` (ib32 = is/4, l = is%4); same derivation as
/// `iq2xxs.rs::run_geometry` since both read an identical `aux32` layout.
fn run_geometry(is: usize) -> (usize, usize, usize, usize, Option<usize>, usize) {
    let ib32 = is / 4;
    let l = is % 4;
    let base = QS_OFF + 8 * ib32 + 2 * l;
    let aux = SS_OFF + 4 * ib32;
    let scale_off = aux + 3;
    let (lo_off, hi_off, shift_div) = match l {
        0 => (aux, None, 1),
        1 => (aux, Some(aux + 1), 128),
        2 => (aux + 1, Some(aux + 2), 64),
        _ => (aux + 2, Some(aux + 3), 32),
    };
    (base, base + 1, scale_off, lo_off, hi_off, shift_div)
}

/// Both of one lane's decoded halves, `let decoded{is}_{half} = ...`, and
/// each half's own `out_off`.
fn decoded_lane(is: usize) -> [(usize, String); 2] {
    let (g1_off, g2_off, scale_off, lo_off, hi_off, shift_div) = run_geometry(is);
    let out_off = is * 8;
    let byte = |off: usize| format!("((i32(qb[:, {off} :+ 1]) + 256) % 256)");
    let scale_byte = byte(scale_off);
    let lo = byte(lo_off);
    let signs_idx = match hi_off {
        None => format!("({lo} % 128)"),
        Some(hi_off) => {
            let hi = byte(hi_off);
            format!("((({lo} + {hi} * 256) / {shift_div}) % 128)")
        }
    };
    let db = format!("(f32(d) * (0.5 + f32({scale_byte} / 16)) * 0.5)");
    [(g1_off, 0), (g2_off, 4)].map(|(grid_off, sign_shift)| {
        let half = usize::from(sign_shift != 0);
        let iota4 = format!("IOTA[0 :+ 1, 0 :+ {HALF}]");
        let grid_idx8 = format!("({} * {HALF} + {iota4})", byte(grid_off));
        let sign_idx8 = format!("({signs_idx} * 8 + {sign_shift} + {iota4})");
        (
            out_off + half * HALF,
            format!(
                "    let decoded{is}_{half} = {db} * f32(gather(grid, {grid_idx8})) * f32(gather(signs, {sign_idx8}))\n"
            ),
        )
    })
}

pub(crate) fn iq3xxs_matvec_src(tn: usize) -> String {
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
kernel iq3xxs_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                     D: tensor<f16>[N, NB], GRID: tensor<i32>[1, {IQ3XXS_GRID_LEN}],
                     SIGNS: tensor<i32>[1, {IQ2XXS_SIGNS_LEN}], IOTA: tensor<i32>[1, 8],
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

/// [`iq3xxs_matvec_src`] for `m == 1`, folding the whole decode-and-reduce
/// into one `iq3xxs_qdot_t` call; see `iq1s_qdot_matvec_src`'s doc, which
/// this mirrors. `@aligned(N = TN)` is required for the same reason:
/// `iq3xxs_qdot_t` demands its `qb`/`d` slices provably in bounds.
pub(crate) fn iq3xxs_qdot_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq3xxs_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                          D: tensor<f16>[N, NB], GRID: tensor<i8>[1, {IQ3XXS_GRID_LEN}],
                          SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}], C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq3xxs_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                           D[pn * TN :+ TN, :], GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}

/// [`iq3xxs_matvec_src`]'s decode, stored straight into a `[K, N]` scratch
/// instead of reduced against an activation row; see `iq1s.rs`'s
/// `iq1s_dequant_src` for why.
pub(crate) fn iq3xxs_dequant_src(tn: usize) -> String {
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
kernel iq3xxs_dequant(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                      GRID: tensor<i32>[1, {IQ3XXS_GRID_LEN}], SIGNS: tensor<i32>[1, {IQ2XXS_SIGNS_LEN}],
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

/// [`iq3xxs_dequant_src`]'s decode as a single `iq3xxs_qdecode_t` call; see
/// `iq1s.rs`'s `iq1s_qdecode_src`, which this mirrors for IQ3_XXS.
pub(crate) fn iq3xxs_qdecode_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq3xxs_qdecode(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                      GRID: tensor<i8>[1, {IQ3XXS_GRID_LEN}],
                      SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}],
                      SCRATCH: tensor<f32>[K, N]) {{
  let pn = program_id(0)
  SCRATCH[:, pn * TN :+ TN] = iq3xxs_qdecode_t(QB[pn * TN :+ TN, :],
                                               D[pn * TN :+ TN, :],
                                               GRID[0 :+ 1, :],
                                               SIGNS[0 :+ 1, :])
}}
"
    )
}
