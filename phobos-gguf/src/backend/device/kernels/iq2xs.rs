// IQ2_XS matvec, with the same raw-byte decode as iq1s.rs.
//
// Each lane's 16-bit `qs` halfword holds a 9-bit grid index in the low bits
// and 7 bits of sign parity in the high bits. Each half of a 32-element
// group has its own scale (see quant/iq2_xs.rs). The grid is IQ2_XS's own,
// but the signs work as in IQ2_XXS, so this reuses
// [`crate::quant::iq2xxs_flat_signs`].

use std::fmt::Write as _;

use super::IQ2XXS_SIGNS_LEN;

/// Output columns per CTA.
pub(crate) const IQ2XS_TN: usize = 8;

const LANES: usize = 32;
const LANE: usize = 8;
const BLOCK_BYTES: usize = 72;
const QS_OFF: usize = 0;
const SCALES_OFF: usize = QS_OFF + 64;

/// [`crate::quant::iq2xs_flat_grid`]'s length: 512 grid entries of eight
/// lanes each.
pub(crate) const IQ2XS_GRID_LEN: usize = 512 * 8;

/// Byte offsets for lane `is`, with `ib32 = is / 4` and `l = is % 4` as in
/// `quant/iq2_xs.rs::dequantize`.
fn run_geometry(is: usize) -> (usize, usize, usize) {
    let ib32 = is / 4;
    let l = is % 4;
    let lo_off = QS_OFF + 2 * (4 * ib32 + l);
    let scale_off = SCALES_OFF + ib32;
    (lo_off, lo_off + 1, scale_off)
}

/// One lane's decoded value, `let decoded{is} = ...`, and its `out_off`.
fn decoded_lane(is: usize) -> (usize, String) {
    let (lo_off, hi_off, scale_off) = run_geometry(is);
    let out_off = is * LANE;
    let byte = |off: usize| format!("((i32(qb[:, {off} :+ 1]) + 256) % 256)");
    let q16 = format!("({} + {} * 256)", byte(lo_off), byte(hi_off));
    let scale_byte = byte(scale_off);
    let nibble = if is % 4 < 2 {
        format!("({scale_byte} % 16)")
    } else {
        format!("({scale_byte} / 16)")
    };
    let dl = format!("(f32(d) * (0.5 + f32({nibble})) * 0.25)");
    let mag_idx8 = format!("(({q16} % 512) * {LANE} + iota)");
    let sign_idx8 = format!("(({q16} / 512) * {LANE} + iota)");
    (
        out_off,
        format!(
            "    let decoded{is} = {dl} * f32(gather(grid, {mag_idx8})) * f32(gather(signs, {sign_idx8}))\n"
        ),
    )
}

pub(crate) fn iq2xs_matvec_src(tn: usize) -> String {
    let mut body = String::new();
    for is in 0..LANES {
        let (out_off, decode) = decoded_lane(is);
        let a_is = format!("A[pm :+ 1, kb * 256 + {out_off} :+ {LANE}]");
        let _ = writeln!(body, "{decode}    acc = acc + dot_t({a_is}, decoded{is})");
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq2xs_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                    D: tensor<f16>[N, NB], GRID: tensor<i32>[1, {IQ2XS_GRID_LEN}],
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

/// Output tile for the dp4a variant. A warp takes eight columns at the
/// default 256-thread CTA.
pub(crate) const IQ2XS_I8_TN: usize = 64;

/// The tile for an `n` that 64 does not divide. A warp then takes two
/// columns rather than eight, which beats the float fallback.
pub(crate) const IQ2XS_I8_NARROW_TN: usize = 16;

/// [`iq2xs_qdot_matvec_src`] against an int8-quantized activation, in dp4a.
pub(crate) fn iq2xs_qdot_i8_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256, 4)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq2xs_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                             QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                             GRID: tensor<i8>[1, {IQ2XS_GRID_LEN}],
                             SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}],
                             C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq2xs_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                              QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                              GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}

/// [`iq2xs_matvec_src`] for `m == 1`, as one `iq2xs_qdot_t` call; mirrors
/// `iq1s_qdot_matvec_src`. `@aligned(N = TN)` is required, because
/// `iq2xs_qdot_t` needs its `qb` and `d` slices provably in bounds.
pub(crate) fn iq2xs_qdot_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq2xs_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                         D: tensor<f16>[N, NB], GRID: tensor<i8>[1, {IQ2XS_GRID_LEN}],
                         SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}], C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq2xs_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                          D[pn * TN :+ TN, :], GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}

/// [`iq2xs_matvec_src`]'s decode, stored into a `[K, N]` scratch instead of
/// reduced against an activation row; see `iq1s_dequant_src` in `iq1s.rs`.
pub(crate) fn iq2xs_dequant_src(tn: usize) -> String {
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
kernel iq2xs_dequant(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                     GRID: tensor<i32>[1, {IQ2XS_GRID_LEN}], SIGNS: tensor<i32>[1, {IQ2XXS_SIGNS_LEN}],
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

/// [`iq2xs_dequant_src`]'s decode as one `iq2xs_qdecode_t` call. Mirrors
/// `iq1s_qdecode_src` in `iq1s.rs`.
pub(crate) fn iq2xs_qdecode_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq2xs_qdecode(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                     GRID: tensor<i8>[1, {IQ2XS_GRID_LEN}],
                     SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}],
                     SCRATCH: tensor<f32>[K, N]) {{
  let pn = program_id(0)
  SCRATCH[:, pn * TN :+ TN] = iq2xs_qdecode_t(QB[pn * TN :+ TN, :],
                                              D[pn * TN :+ TN, :],
                                              GRID[0 :+ 1, :],
                                              SIGNS[0 :+ 1, :])
}}
"
    )
}

/// IQ2_XS's prompt projection, decode and contraction in one kernel.
///
/// There is one scale per sixteen elements, so the tile language keeps one
/// accumulator per half block; see `qmma_signed.rs`.
pub(crate) fn iq2xs_qmma_src(block: usize, tm: usize, tn: usize) -> String {
    format!(
        "@launch({block})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 256)
kernel iq2xs_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                  QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                  GRID: tensor<i8>[1, {IQ2XS_GRID_LEN}],
                  SIGNS: tensor<i8>[1, {IQ2XXS_SIGNS_LEN}],
                  C: tensor<f32>[M, N]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm * TM :+ TM, pn * TN :+ TN] = iq2xs_qmma_staged_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                        QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                        GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}}
"
    )
}
