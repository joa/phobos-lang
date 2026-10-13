// The mixture-of-experts feed-forward's kernels: the router's top-k, the
// slot-indexed decode matvec, and the weighted combine.
//
// Every expert a token reads is in a cache slot by the time its kernel
// runs, so the matvec's only per-token input besides the activation is a
// slot table.

use crate::backend::fuse::{kquant_block_call, kquant_lanes_block};

/// Experts a token goes through, baked into the kernels as `U`.
pub(crate) const MOE_USED: usize = 8;

/// Output tile of the slot matvec, the same as `kquant.rs`'s wide tile.
pub(crate) const MOE_QDOT_TN: usize = 64;

/// Columns one CTA of the combine adds.
pub(crate) const MOE_COMBINE_TN: usize = 256;

/// Softmax and top-k over each row of router logits, one warp per row.
///
/// Writes the `MOE_USED` largest probabilities in descending order: ids to
/// `TOPK` and weights, renormalized to sum to one, to `W`. This matches
/// llama.cpp's `build_moe_ffn`, and the host's `route`: a tie goes to the
/// lower index. `IO`, the iota row, is no longer read.
///
/// The lanes stride the experts. Softmax is monotonic, so the picks are the
/// largest logits, and pick `j` is the largest after pick `j - 1` in
/// (value descending, index ascending) order, so nothing is marked between
/// picks. The softmax denominator cancels in the renormalization, which
/// leaves `exp(l - max)` over the picks' sum.
pub fn moe_topk_src() -> String {
    "@launch(32)
kernel moe_topk(L: tensor<f32>[R, E], IO: tensor<f32>[1, E], TOPK: tensor<i32>[R, U], W: tensor<f32>[R, U]) {
  let r = ctx.block.x
  var top: f32 = -1.0e30
  for k in range(ctx.lane, E, ctx.lanes) {
    top = max(top, L[r, k])
  }
  let m = max<over ctx.lane>(top)
  var pv: f32 = 1.0e30
  var pi = -1
  var picked: f32 = 0.0
  for j in range(0, U, 1) {
    var bv: f32 = -1.0e30
    var bi = E
    for k in range(ctx.lane, E, ctx.lanes) {
      let l = L[r, k]
      let after = l < pv || (l == pv && k > pi)
      let better = l > bv || (l == bv && k < bi)
      if after && better {
        bv = l
        bi = k
      }
    }
    let v = max<over ctx.lane>(bv)
    let i = min<over ctx.lane>(where(bv == v, bi, E))
    let e = exp(v - m)
    picked += e
    if ctx.lane == 0 {
      W[r, j] = e
      TOPK[r, j] = i32(i)
    }
    pv = v
    pi = i
  }
  if ctx.lane == 0 {
    for j in range(0, U, 1) {
      W[r, j] = W[r, j] / picked
    }
  }
}
"
    .to_string()
}

/// `kquant.rs`'s lanes decode matvec over a cache slab.
///
/// Program `j` on the second grid axis reads slot `SLOT[j]`, `NE` rows per
/// slot, and writes row `j` of `C`. `AQ` and `AS` hold `U` rows of
/// quantized activation. `act_row` picks the row a program reads: `0` when
/// all experts share one (gate and up), `j` when each has its own (down).
pub fn moe_qdot_src(name: &str, ne: usize, act_row: &str) -> String {
    let (block, bytes, setup) = kquant_lanes_block(name);
    let call = kquant_block_call(name, "CB", "CD", "AQ", "AS", act_row, "g", "acc", "b");
    format!(
        "{block}
@launch(256)
@autotune(NE in [{ne}])
@aligned(N = {MOE_QDOT_TN}, NS = NE, K = 256, RB = {bytes})
kernel {name}_moe_qdot(AQ: tensor<i8>[U, K], AS: tensor<f32>[U, KB], SLOT: tensor<i32>[1, U],
                      CB: tensor<i8>[NS, RB], CD: tensor<f16>[NS, NB], C: tensor<f32>[U, N]) {{
  let j = ctx.block.y
{lanes}{setup}  var acc: f32 = 0.0
  for b in range(0, K / 256, 1) {{
    {call}
  }}
  let total = sum<over ctx.lane, width 4>(acc)
  if q == 0 {{
    C[j, col] = total
  }}
}}
",
        lanes = slot_lanes()
    )
}

/// Gate, up and SwiGLU in one launch.
///
/// Program `j` on the second grid axis contracts the shared activation row
/// against slot `SLOT[j]` of both the gate and up slabs, which share a
/// format. It writes `silu(gate) * up` as row `j` of `H`.
pub fn moe_gateup_src(name: &str, ne: usize) -> String {
    let (block, bytes, setup) = kquant_lanes_block(name);
    let gate = kquant_block_call(name, "GB", "GD", "AQ", "AS", "0", "g", "ag", "b");
    let up = kquant_block_call(name, "UB", "UD", "AQ", "AS", "0", "g", "au", "b");
    format!(
        "{block}
@launch(256)
@autotune(NE in [{ne}])
@aligned(N = {MOE_QDOT_TN}, NS = NE, K = 256, RB = {bytes})
kernel {name}_moe_gateup(AQ: tensor<i8>[U, K], AS: tensor<f32>[U, KB], SLOT: tensor<i32>[1, U],
                        GB: tensor<i8>[NS, RB], GD: tensor<f16>[NS, NB],
                        UB: tensor<i8>[NS, RB], UD: tensor<f16>[NS, NB], H: tensor<f32>[U, N]) {{
  let j = ctx.block.y
{lanes}{setup}  var ag: f32 = 0.0
  var au: f32 = 0.0
  for b in range(0, K / 256, 1) {{
    {gate}
    {up}
  }}
  let gv = sum<over ctx.lane, width 4>(ag)
  let uv = sum<over ctx.lane, width 4>(au)
  if q == 0 {{
    H[j, col] = gv / (1.0 + exp(0.0 - gv)) * uv
  }}
}}
",
        lanes = slot_lanes()
    )
}

/// A slot kernel's lanes: four to a column, eight columns a warp, and `g`
/// the first row of the column's group of eight in slot `SLOT[j]`.
fn slot_lanes() -> &'static str {
    "  let col = ctx.block.x * 64 + ctx.warp * 8 + ctx.lane / 4
  let q = ctx.lane % 4
  let g = SLOT[0, j] * NE + col / 8 * 8
  let c8 = col % 8
"
}

/// `X += sum_j W[j] * C[j, :] + sigmoid(G) * S`: the routed experts' down
/// rows weighted by the router, plus the gated shared expert, added into
/// the residual. One CTA per tile of columns.
pub(crate) fn moe_combine_src() -> String {
    format!(
        "@launch(256)
@autotune(TN in [{MOE_COMBINE_TN}], U in [{MOE_USED}])
@aligned(N = TN)
kernel moe_combine(W: tensor<f32>[1, U], C: tensor<f32>[U, N], G: tensor<f32>[1, 1], S: tensor<f32>[1, N],
                   X: tensor<f32>[1, N]) {{
  let pn = ctx.block.x
  var acc: tile<f32>[1, TN] = X[0 :+ 1, pn * TN :+ TN]
  for j in range(0, U, 1) {{
    let w = W[0, j]
    acc += C[j :+ 1, pn * TN :+ TN] * w
  }}
  var g: tile<f32>[1, 1] = G[0 :+ 1, 0 :+ 1]
  g = 1.0 / (1.0 + exp(0.0 - g))
  acc += S[0 :+ 1, pn * TN :+ TN] * g
  X[0 :+ 1, pn * TN :+ TN] = acc
}}
"
    )
}
