// The activation side of a Hadamard-folded weight; see `crate::hadamard`.
//
// The 1024-point Sylvester matrix is the Kronecker square of the 32-point
// one. So a 1024-element block, viewed as a [32, 32] tile `X` (element
// `32 a + b` at row `a`, column `b`), transforms as `H X H`. Here `H` is
// the normalized 32-point matrix, which is symmetric and its own inverse.
//
// The row buffer is passed as `[rows * width / 32, 32]`. A block is 32
// consecutive rows of it, and a program takes one block. The delta net's
// regrouped heads are gathered one head at a time, `head_dim / 32` rows
// each. Both extents are whole blocks, as `@aligned` promises.

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

/// What the kernel does around the transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum HadamardForm {
    /// The transform alone.
    Plain,
    /// Also writes a Q8_0 copy of the output, one 32-element block per tile
    /// row.
    Quantized,
    /// Quantized, with an RMS norm first. The payload is the epsilon's f32
    /// bits. The normalized row is written out as well.
    Normed(u32),
}

impl HadamardForm {
    /// The kernel's name, distinct per form so a trace tells them apart.
    pub(crate) fn kernel(self) -> &'static str {
        match self {
            HadamardForm::Plain => "hadamard",
            HadamardForm::Quantized => "hadamard_q",
            HadamardForm::Normed(_) => "rms_norm_hadamard_q",
        }
    }
}

/// A Hadamard kernel's cache key: width, head regrouping and form.
pub(crate) type HadamardKey = (usize, Option<HeadPerm>, HadamardForm);

/// Widest row [`HadamardForm::Normed`] accepts. Each program stages its
/// whole row for the sum of squares, beside its block's five tiles.
pub(crate) const HADAMARD_NORM_MAX_WIDTH: usize = 5120;

/// The kernel for rows `width` wide, regrouped by `perm` where given.
pub(crate) fn hadamard_src(width: usize, perm: Option<HeadPerm>, form: HadamardForm) -> String {
    let side = HADAMARD_SIDE;
    let blocks = width / HADAMARD_BLOCK;
    let row_rows = width / side;
    let load = match (perm, form) {
        (None, HadamardForm::Normed(bits)) => format!(
            "  let row = p / {blocks}
  var xr = X[row * {row_rows} :+ {row_rows}, :]
  var sq: tile<f32>[{row_rows}, 1] = rowsum(xr * xr)
  var tot: tile<f32>[1, 1] = rowsum(transpose(sq))
  var inv: tile<f32>[1, 1] = 1.0 / sqrt(tot / {width}.0 + {eps:.12})
  var g = G[b * {side} :+ {side}, :]
  var x: tile<f32>[{side}, {side}] = X[p * {side} :+ {side}, :] * inv * g
  N[p * {side} :+ {side}, :] = x
",
            eps = f32::from_bits(bits)
        ),
        (None, _) => format!("  var x = X[p * {side} :+ {side}, :]
"),
        (Some(perm), _) => {
            let span = perm.head_dim / side;
            let heads = HADAMARD_BLOCK / perm.head_dim;
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
    let (mut params, mut store) = (String::new(), String::new());
    if let HadamardForm::Normed(_) = form {
        params.push_str(&format!(",
                G: tensor<f32>[SR, {side}], N: tensor<f32>[R, {side}]"));
    }
    if form != HadamardForm::Plain {
        // Same arithmetic as `quantize`.
        params.push_str(&format!(",
                Q: tensor<i8>[R, {side}], D: tensor<f32>[R, D1]"));
        store = format!(
            "  var mx: tile<f32>[{side}, 1] = rowmax(tmax(y, -y))
  var inq = 127.0 / (mx + 0.00000001)
  var yq = y * inq
  Q[p * {side} :+ {side}, :] = i8(i32(round(yq)))
  D[p * {side} :+ {side}, 0 :+ 1] = mx / 127.0
"
        );
    }
    format!(
        "@launch(256)
@aligned(R = {side}, SR = {side}, D1 = 1)
kernel {name}(X: tensor<f32>[R, {side}], S: tensor<f32>[SR, {side}],
                H: tensor<f32>[{side}, {side}], Y: tensor<f32>[R, {side}]{params}) {{
  let p = program_id(0)
  let b = p % {blocks}
{load}  var s = S[b * {side} :+ {side}, :]
  var h = H[:, :]
  var t = dot(x * s, h)
  var y = dot(h, t)
  Y[p * {side} :+ {side}, :] = y
{store}}}
",
        name = form.kernel()
    )
}
