// The mixture-of-experts feed-forward's kernels: the router's top-k, the
// slot-indexed decode matvec, and the weighted combine. Every expert a
// token reads is in a cache slot by the time its kernel runs (the misses
// were copied in at the block's sync point), so the matvec takes a slot
// table and nothing else that varies per token.

use super::quant::qdot_i8_cta;

/// Experts a token goes through, baked into the kernels as `U`.
pub(crate) const MOE_USED: usize = 8;

/// Columns a warp of the slot matvec owns, as `kquant.rs`'s wide tile.
pub(crate) const MOE_QDOT_TN: usize = 64;

/// Columns one CTA of the combine adds.
pub(crate) const MOE_COMBINE_TN: usize = 256;

/// The halving tree `argmax.rs` uses, over a `[1, W]` (value, index) pair
/// held in `v` and `i`, folding the upper half into the lower.
fn halving_tree(width: usize) -> String {
    let mut tree = String::new();
    let mut half = width / 2;
    while half >= 1 {
        tree.push_str(&format!(
            "    i[0 :+ 1, 0 :+ {half}] = argsel(v[0 :+ 1, {half} :+ {half}], v[0 :+ 1, 0 :+ {half}], \
             i[0 :+ 1, {half} :+ {half}], i[0 :+ 1, 0 :+ {half}])\n    \
             v[0 :+ 1, 0 :+ {half}] = tmax(v[0 :+ 1, {half} :+ {half}], v[0 :+ 1, 0 :+ {half}])\n"
        ));
        half /= 2;
    }
    tree
}

/// Softmax over each row of `n_expert` router logits, one program a row,
/// the `MOE_USED` largest probabilities in descending order as `TOPK` (i32)
/// and `W` (renormalized to sum to one), the order llama.cpp's
/// `build_moe_ffn` uses. `IO` is the iota row `argmax` carries, `[0,
/// n_expert)` as f32. A picked entry is masked to -1, which no probability
/// reaches, so it is not picked twice. `n_expert` must be a power of two
/// for the tree.
pub(crate) fn moe_topk_src(n_expert: usize) -> String {
    assert!(n_expert.is_power_of_two(), "the top-k tree wants a power of two, got {n_expert}");
    format!(
        "@launch(256)
@autotune(NE in [{n_expert}], U in [{MOE_USED}])
@aligned(E = NE)
kernel moe_topk(L: tensor<f32>[R, E], IO: tensor<f32>[1, E], TOPK: tensor<i32>[R, U], W: tensor<f32>[R, U]) {{
  let r = program_id(0)
  var p: tile<f32>[1, NE] = L[r :+ 1, 0 :+ NE]
  let m = rowmax(p)
  var ex: tile<f32>[1, NE] = exp(p - m)
  let total = rowsum(ex)
  p = ex / total
  var picked: tile<f32>[1, 1] = 0.0
  var neg: tile<f32>[1, NE] = -1.0
  for j in range(0, U, 1) {{
    var v: tile<f32>[1, NE] = p
    var i: tile<f32>[1, NE] = IO[0 :+ 1, 0 :+ NE]
{tree}    W[r :+ 1, j :+ 1] = v[0 :+ 1, 0 :+ 1]
    TOPK[r :+ 1, j :+ 1] = i32(i[0 :+ 1, 0 :+ 1])
    picked = picked + v[0 :+ 1, 0 :+ 1]
    p = argsel(p, v[0 :+ 1, 0 :+ 1], neg, p)
  }}
  var w: tile<f32>[1, U] = W[r :+ 1, 0 :+ U]
  W[r :+ 1, 0 :+ U] = w / picked
}}
",
        tree = halving_tree(n_expert)
    )
}

/// The decode matvec of `kquant.rs` over a cache slab: program `j` of the
/// second grid axis reads slot `SLOT[j]`, `NE` rows apiece, and writes row
/// `j` of `C`. `AQ`/`AS` carry `U` rows of quantized activation; `act_row`
/// picks which one a program reads: `0` when every expert reads the same
/// row (gate and up), `j` when each reads its own (down, fed by the SwiGLU
/// of its gate and up). `resident` is the register budget's CTA count, as
/// `kquant.rs` sets it per format.
pub(crate) fn moe_qdot_src(name: &str, ne: usize, act_row: &str, resident: usize) -> String {
    let cta = qdot_i8_cta(MOE_QDOT_TN);
    let min_blocks = (1024 / cta * resident / 4).max(1);
    format!(
        "@launch({cta}, {min_blocks})
@autotune(TN in [{MOE_QDOT_TN}], NE in [{ne}])
@aligned(N = TN, NS = NE)
kernel {name}_moe_qdot(AQ: tensor<i8>[U, K], AS: tensor<f32>[U, KB], SLOT: tensor<i32>[1, U],
                      CB: tensor<i8>[NS, RB], CD: tensor<f16>[NS, NB], C: tensor<f32>[U, N]) {{
  let pn = program_id(0)
  let j = program_id(1)
  let r = SLOT[0, j] * NE + pn * TN
  C[j :+ 1, pn * TN :+ TN] = {name}_qdot_i8_t(AQ[{act_row} :+ 1, :], AS[{act_row} :+ 1, :],
                                              CB[r :+ TN, :], CD[r :+ TN, :])
}}
"
    )
}

/// Gate, up and the SwiGLU in one launch: program `j` of the second grid
/// axis contracts the shared activation row against slot `SLOT[j]` of the
/// gate slab and of the up slab, both `NE` rows an expert in one format,
/// and writes `silu(gate) * up` as row `j` of `H`. Two launches and a
/// SwiGLU fewer a block than the separate form.
pub(crate) fn moe_gateup_src(name: &str, ne: usize, resident: usize) -> String {
    let cta = qdot_i8_cta(MOE_QDOT_TN);
    let min_blocks = (1024 / cta * resident / 4).max(1);
    format!(
        "@launch({cta}, {min_blocks})
@autotune(TN in [{MOE_QDOT_TN}], NE in [{ne}])
@aligned(N = TN, NS = NE)
kernel {name}_moe_gateup(AQ: tensor<i8>[U, K], AS: tensor<f32>[U, KB], SLOT: tensor<i32>[1, U],
                        GB: tensor<i8>[NS, RB], GD: tensor<f16>[NS, NB],
                        UB: tensor<i8>[NS, RB], UD: tensor<f16>[NS, NB], H: tensor<f32>[U, N]) {{
  let pn = program_id(0)
  let j = program_id(1)
  let r = SLOT[0, j] * NE + pn * TN
  var g: tile<f32>[1, TN] = {name}_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :], GB[r :+ TN, :], GD[r :+ TN, :])
  var u: tile<f32>[1, TN] = {name}_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :], UB[r :+ TN, :], UD[r :+ TN, :])
  H[j :+ 1, pn * TN :+ TN] = g / (1.0 + exp(0.0 - g)) * u
}}
"
    )
}

/// `X += sum_j W[j] * C[j, :] + sigmoid(G) * S`: the routed experts'
/// down rows weighted by the router, plus the shared expert's row scaled by
/// its gate's logit, into the residual. One CTA a tile of columns.
pub(crate) fn moe_combine_src() -> String {
    format!(
        "@launch(256)
@autotune(TN in [{MOE_COMBINE_TN}], U in [{MOE_USED}])
@aligned(N = TN)
kernel moe_combine(W: tensor<f32>[1, U], C: tensor<f32>[U, N], G: tensor<f32>[1, 1], S: tensor<f32>[1, N],
                   X: tensor<f32>[1, N]) {{
  let pn = program_id(0)
  var acc: tile<f32>[1, TN] = X[0 :+ 1, pn * TN :+ TN]
  for j in range(0, U, 1) {{
    let w = W[0, j]
    acc = acc + C[j :+ 1, pn * TN :+ TN] * w
  }}
  var g: tile<f32>[1, 1] = G[0 :+ 1, 0 :+ 1]
  g = 1.0 / (1.0 + exp(0.0 - g))
  acc = acc + S[0 :+ 1, pn * TN :+ TN] * g
  X[0 :+ 1, pn * TN :+ TN] = acc
}}
"
    )
}
