// The activation side of a Hadamard-folded weight; see `crate::hadamard`.
//
// The Sylvester matrix of 1024 is the Kronecker square of the one of 32, so a
// 1024-element block viewed as a [32, 32] tile `X` (element `32 a + b` at row
// `a`, column `b`) transforms as `H X H` for the normalized 32-point `H`,
// symmetric and its own inverse. The row buffer is handed over as `[rows *
// width / 32, 32]`, a block is 32 consecutive rows of it, and a program takes
// one block. The delta net's regrouped heads are gathered a head at a time,
// `head_dim / 32` rows apiece, from wherever the tiled order keeps them.
// Both extents are whole blocks by construction, which `@aligned` promises.

use crate::backend::{HADAMARD_BLOCK, HeadPerm};

/// The side of the tile a block is viewed as.
pub(crate) const HADAMARD_SIDE: usize = 32;

/// The normalized 32-point transform, row-major.
pub(crate) fn hadamard_matrix() -> Vec<f32> {
    let scale = (HADAMARD_SIDE as f32).sqrt().recip();
    let mut h = vec![0.0f32; HADAMARD_SIDE * HADAMARD_SIDE];
    for (i, v) in h.iter_mut().enumerate() {
        let (r, c) = (i / HADAMARD_SIDE, i % HADAMARD_SIDE);
        *v = if (r & c).count_ones() % 2 == 1 { -scale } else { scale };
    }
    h
}

/// The kernel for rows `width` wide, regrouped by `perm` where given.
pub(crate) fn hadamard_src(width: usize, perm: Option<HeadPerm>) -> String {
    let side = HADAMARD_SIDE;
    let blocks = width / HADAMARD_BLOCK;
    let load = match perm {
        None => format!("  var x = X[p * {side} :+ {side}, :]\n"),
        Some(perm) => {
            let span = perm.head_dim / side;
            let heads = HADAMARD_BLOCK / perm.head_dim;
            let row_rows = width / side;
            let (groups, repeat) = (perm.groups, perm.repeat);
            format!(
                "  let row = p / {blocks}
  var x: tile<f32>[{side}, {side}] = 0.0
  for i in range(0, {heads}) {{
    let c = b * {heads} + i
    let src = row * {row_rows} + ((c % {repeat}) * {groups} + c / {repeat}) * {span}
    x[i * {span} :+ {span}, :] = X[src :+ {span}, :]
  }}
"
            )
        }
    };
    format!(
        "@launch(256)
@aligned(R = {side}, SR = {side})
kernel hadamard(X: tensor<f32>[R, {side}], S: tensor<f32>[SR, {side}],
                H: tensor<f32>[{side}, {side}], Y: tensor<f32>[R, {side}]) {{
  let p = program_id(0)
  let b = p % {blocks}
{load}  var s = S[b * {side} :+ {side}, :]
  var h = H[:, :]
  var t = dot(x * s, h)
  var y = dot(h, t)
  Y[p * {side} :+ {side}, :] = y
}}
"
    )
}
