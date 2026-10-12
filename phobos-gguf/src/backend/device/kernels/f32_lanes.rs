// The row-major formats' decode matvecs written with lanes: each
// `<fmt>_qdot_t`'s mapping in `.ph`, a warp per column against an `f32`
// activation row, the column's lanes splitting each block, and a shuffle
// sum. The weight is stored as declared, `[N, NB * bytes]`.

use std::fmt::Write as _;

/// The kernel around a per-block body that adds into `acc`, with `col`,
/// `lane` and the block `b` in scope.
fn lanes_f32_matvec(name: &str, params: &str, aligned: &str, prelude: &str, body: &str) -> String {
    format!(
        "@launch(256)
@aligned(K = 256{aligned})
kernel {name}_qdot_lanes(A: tensor<f32>[1, K], {params}C: tensor<f32>[1, N]) {{
  let col = ctx.block.x * ctx.warps + ctx.warp
  let lane = ctx.lane
{prelude}  var acc: f32 = 0.0
  if col < N {{
    for b in range(0, K / 256, 1) {{
{body}    }}
  }}
  let total = sum<over ctx.lane>(acc)
  if lane == 0 && col < N {{
    C[0, col] = total
  }}
}}
"
    )
}

/// Q2_K: 84 bytes a block, a scale-and-min byte per sixteen-element run at
/// byte `is`, the two-bit quants from byte 16. Pass `it` gives lane `l` the
/// element `32 it + l`: run `2 it + l / 16`, the quant byte at
/// `16 + 32 (it / 4) + l`, shifted by `2 (it % 4)`.
pub(crate) fn q2k_qdot_lanes_src() -> String {
    let mut body = String::from(
        "      let blk = b * 84
      let d = f32(D[col, b])
      let dmin = f32(DMIN[col, b])
",
    );
    for it in 0..8 {
        let _ = writeln!(
            body,
            "      let sc{it} = QB[col, blk + {is} + lane / 16] & 255
      let q{it} = ((QB[col, blk + {qs} + lane] & 255) >> {sh}) & 3
      acc += (d * f32(sc{it} & 15) * f32(q{it}) - dmin * f32(sc{it} >> 4)) * A[0, b * 256 + {at} + lane]",
            is = 2 * it,
            qs = 16 + 32 * (it / 4),
            sh = 2 * (it % 4),
            at = 32 * it
        );
    }
    lanes_f32_matvec("q2k", "QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB], DMIN: tensor<f16>[N, NB], ", "", "", &body)
}

/// Q3_K: 112 bytes a block (the device pads 110): the high bits `hmask` at
/// 0, the two-bit quants at 32, the twelve bytes of signed 6-bit scales at
/// 96. Two lanes take each sixteen-element run `is`, eight elements each; an
/// element is `d * scale * (low2 - 4 + 4 bit)`.
pub(crate) fn q3k_qdot_lanes_src() -> String {
    let prelude = "  let is = lane / 2
  let y0 = (lane % 2) * 8
  let h = is / 8
  let j = (is % 8) / 2
  let half2 = is % 2
  let qo = 32 + 32 * h + 16 * half2 + y0
  let ho = 16 * half2 + y0
  let grp = is / 4
  let c = is % 4
  let lo_off = 96 + where(grp % 2 == 0, c, 4 + c)
  let lo_sh = where(grp < 2, 0, 4)
";
    let mut body = String::from(
        "      let blk = b * 112
      let sc = (((QB[col, blk + lo_off] & 255) >> lo_sh) & 15) + (((QB[col, blk + 104 + c] & 255) >> (2 * grp)) & 3) * 16 - 32
      let a = b * 256 + is * 16 + y0
",
    );
    for half in 0..2 {
        let _ = writeln!(
            body,
            "      let dot{half} = sum(f32((((QB[col, blk + qo + {at} :+ 4] & 255) >> (2 * j)) & 3) + (((QB[col, blk + ho + {at} :+ 4] & 255) >> (4 * h + j)) & 1) * 4 - 4) * A[0, a + {at} :+ 4])",
            at = 4 * half
        );
    }
    let _ = writeln!(body, "      acc += f32(D[col, b]) * f32(sc) * (dot0 + dot1)");
    lanes_f32_matvec("q3k", "QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB], ", ", RB = 16", prelude, &body)
}

/// IQ4_XS: 136 bytes a block: `scales_h` at 2, `scales_l` at 4, the nibbles
/// at 8. Lane `l` takes element `l` of each 32-element run `ib`: the low
/// nibble of byte `8 + 16 ib + l % 16` for the first sixteen, the high one
/// for the rest, through the codebook, under the run's 6-bit scale less 32.
pub(crate) fn iq4xs_qdot_lanes_src() -> String {
    let prelude = "  let shift = 4 * (lane / 16)\n";
    let mut body = String::from(
        "      let blk = b * 136
      let d = f32(D[col, b])
      let sh = (QB[col, blk + 2] & 255) | ((QB[col, blk + 3] & 255) << 8)
",
    );
    for ib in 0..8 {
        let _ = writeln!(
            body,
            "      let sl{ib} = ((QB[col, blk + {l_off}] & 255) >> {l_sh}) & 15
      let s{ib} = sl{ib} + ((sh >> {h_sh}) & 3) * 16 - 32
      let nib{ib} = ((QB[col, blk + {qs} + lane % 16] & 255) >> shift) & 15
      acc += d * f32(s{ib}) * f32(CODEBOOK[0, nib{ib}]) * A[0, b * 256 + {at} + lane]",
            l_off = 4 + ib / 2,
            l_sh = 4 * (ib % 2),
            h_sh = 2 * ib,
            qs = 8 + 16 * ib,
            at = 32 * ib
        );
    }
    lanes_f32_matvec(
        "iq4xs",
        "QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB], CODEBOOK: tensor<i32>[1, 16], ",
        "",
        prelude,
        &body,
    )
}
