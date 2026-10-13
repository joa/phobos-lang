// The K-quants' decode matvecs, written with lanes over the block decodes
// in `fuse/kquant_block.rs`: four lanes a column, eight columns a warp.
//
// Q4_K and Q5_K read `d` and `dmin` from the block header. Q6_K's `d` lives
// only in the grouped scale plane (see `Quant::device_block`).

use crate::backend::fuse::{kquant_block_call, kquant_lanes_block, q4k_block_src, q5k_block_src, q6k_block_src};
use crate::quant::Quant;

/// Columns a CTA of [`q4k_qdot_lanes_src`] covers: eight a warp.
pub(crate) const Q4K_LANES_TN: usize = 64;

/// The Q4_K decode matvec written with lanes, over [`q4k_block_src`]. `QB`
/// is the grouped weight as the device holds it, `[N/8, NB * 8 * 144]`.
pub fn q4k_qdot_lanes_src() -> String {
    format!(
        "{}
@launch(256)
@aligned(K = 256, GW = 1152)
kernel q4k_qdot_lanes(AQ: tensor<i8>[1, K], AS: tensor<f32>[1, KB],
                      QB: tensor<i8>[G, GW], C: tensor<f32>[1, N]) {{
  let col = ctx.block.x * 64 + ctx.warp * 8 + ctx.lane / 4
  let q = ctx.lane % 4
  let g = col / 8
  let c8 = col % 8
  var acc: f32 = 0.0
  for b in range(0, K / 256, 1) {{
    q4k_block(QB, AQ, AS, 0, g, c8, q, acc, b)
  }}
  let total = sum<over ctx.lane, width 4>(acc)
  if q == 0 && col < N {{
    C[0, col] = total
  }}
}}
",
        q4k_block_src()
    )
}



/// The Q5_K decode matvec written with lanes, over [`q5k_block_src`].
pub(crate) fn q5k_qdot_lanes_src() -> String {
    format!(
        "{}
@launch(256)
@aligned(K = 256, GW = 1408)
kernel q5k_qdot_lanes(AQ: tensor<i8>[1, K], AS: tensor<f32>[1, KB],
                      QB: tensor<i8>[G, GW], C: tensor<f32>[1, N]) {{
  let col = ctx.block.x * 64 + ctx.warp * 8 + ctx.lane / 4
  let q = ctx.lane % 4
  let g = col / 8
  let c8 = col % 8
  var acc: f32 = 0.0
  for b in range(0, K / 256, 1) {{
    q5k_block(QB, AQ, AS, 0, g, c8, q, acc, b)
  }}
  let total = sum<over ctx.lane, width 4>(acc)
  if q == 0 && col < N {{
    C[0, col] = total
  }}
}}
",
        q5k_block_src()
    )
}


/// The Q6_K decode matvec written with lanes, over [`q6k_block_src`]. Its
/// `d` is in the grouped plane `D`, `[N/8, NB * 8]`.
pub(crate) fn q6k_qdot_lanes_src() -> String {
    format!(
        "{}
@launch(256)
@aligned(K = 256, GW = 1664, DW = 8)
kernel q6k_qdot_lanes(AQ: tensor<i8>[1, K], AS: tensor<f32>[1, KB],
                      QB: tensor<i8>[G, GW], D: tensor<f16>[G, DW], C: tensor<f32>[1, N]) {{
  let col = ctx.block.x * 64 + ctx.warp * 8 + ctx.lane / 4
  let q = ctx.lane % 4
  let h = q / 2
  let l0 = 16 * (q % 2)
  let g = col / 8
  let c8 = col % 8
  var acc: f32 = 0.0
  for b in range(0, K / 256, 1) {{
    q6k_block(QB, D, AQ, AS, 0, g, c8, q, h, l0, acc, b)
  }}
  let total = sum<over ctx.lane, width 4>(acc)
  if q == 0 && col < N {{
    C[0, col] = total
  }}
}}
",
        q6k_block_src()
    )
}

/// The format prefix of a K-quant with a split matvec.
fn kquant_prefix(quant: Quant) -> Option<&'static str> {
    match quant {
        Quant::Q4_K => Some("q4k"),
        Quant::Q5_K => Some("q5k"),
        Quant::Q6_K => Some("q6k"),
        _ => None,
    }
}

/// The kernel [`kquant_lanes_split_src`] names.
pub(crate) fn kquant_lanes_split_name(quant: Quant) -> &'static str {
    match quant {
        Quant::Q4_K => "q4k_lanes_split",
        Quant::Q5_K => "q5k_lanes_split",
        _ => "q6k_lanes_split",
    }
}

/// The split-K form of a K-quant's lanes matvec, `s` slices of `k`, for an
/// `n` that [`Q4K_LANES_TN`] divides.
///
/// Program `(pn, ps)` walks slice `ps` of `k`'s blocks for the CTA's 64
/// columns into column `ps * n + col` of a `[1, s * n]` partial row, which
/// `qgemm_reduce` sums. Q6_K also takes its grouped scale plane `D`.
pub(crate) fn kquant_lanes_split_src(quant: Quant, n: usize, k: usize, s: usize) -> Option<String> {
    let name = kquant_prefix(quant)?;
    let (block, bytes, setup) = kquant_lanes_block(name);
    let sb = k / 256 / s;
    let gw = 8 * (k / 256) * bytes;
    let d = match quant {
        Quant::Q6_K => "D: tensor<f16>[G, DW], ",
        _ => "",
    };
    let call = kquant_block_call(name, "QB", "D", "AQ", "AS", "0", "g", "acc", "b");
    Some(format!(
        "{block}
@launch(256)
@aligned(K = 256, GW = {gw}, DW = 8)
kernel {name}_lanes_split(AQ: tensor<i8>[1, K], AS: tensor<f32>[1, KB],
                      QB: tensor<i8>[G, GW], {d}P: tensor<f32>[1, NP]) {{
  let ps = ctx.block.y
  let col = ctx.block.x * 64 + ctx.warp * 8 + ctx.lane / 4
  let q = ctx.lane % 4
{setup}  let g = col / 8
  let c8 = col % 8
  var acc: f32 = 0.0
  for b in range(ps * {sb}, ps * {sb} + {sb}) {{
    {call}
  }}
  let total = sum<over ctx.lane, width 4>(acc)
  if q == 0 {{
    P[0, ps * {n} + col] = total
  }}
}}
"
    ))
}
