// Prompt attention on the tensor cores with the scores never written out.

use crate::backend::Attn;

/// Query rows of one [`attn_flash_src`] program.
pub(crate) const FLASH_TILE: usize = 64;

/// Keys of one [`attn_flash_src`] step. Narrower than the query tile so its
/// scores and values leave shared memory for two programs a multiprocessor.
pub(crate) const FLASH_KEYS: usize = 32;

/// Output columns one [`attn_flash_src`] program accumulates. Its
/// accumulator and scores stay in the tensor cores' registers only while
/// they fit 128 `f32` a lane, so a wider head splits its values across the
/// grid and recomputes the scores per slice.
pub(crate) const FLASH_VALUE_COLS: usize = 128;

/// Whether [`attn_flash_src`] takes this prompt: whole tiles of rows and
/// cache, and a head dimension the value slices and the contraction step
/// divide.
pub(crate) fn attn_flash_fits(spec: Attn) -> bool {
    spec.rows.is_multiple_of(FLASH_TILE)
        && spec.start_pos.is_multiple_of(FLASH_TILE)
        && spec.head_dim.is_multiple_of(FLASH_VALUE_COLS.min(spec.head_dim))
        && spec.head_dim.is_multiple_of(32)
}

/// Causal attention for a prompt, one query tile, value slice and head per
/// program, as FlashAttention-2: the scores of a key step, contracted over
/// the head dimension in steps, take the running softmax in registers and
/// feed the mix through one shared tile. The query tile's own keys come
/// last, masked. The queries arrive narrowed to f16 and the scores round to
/// it on the way into the mix; the softmax and both accumulations stay f32.
pub(crate) fn attn_flash_src(head_dim: usize, group: usize) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    let (tile, keys, cols) = (FLASH_TILE, FLASH_KEYS, FLASH_VALUE_COLS.min(head_dim));
    format!(
        "@tensorcore
@launch(128)
@dynshared
@autotune(BR in [{tile}], BC in [{keys}], TK in [32], D in [{head_dim}], DV in [{cols}], G in [{group}])
@aligned(R = BR, NK = BR, QW = D, KW = D)
kernel attn_flash(Q: tensor<f16>[R, QW], K: tensor<f16>[NK, KW], V: tensor<f16>[NK, KW],
                  O: tensor<f32>[R, QW]) {{
  let qt = ctx.block.x
  let vs = ctx.block.y
  let h = ctx.block.z
  let qrow = qt * BR
  let qcol = h * D
  let kcol = h / G * D
  let vcol = kcol + vs * DV

  var acc: tile<f32>[BR, DV] = 0.0
  var m: tile<f32>[BR, 1] = -1.0e30
  var l: tile<f32>[BR, 1] = 0.0

  let base = NK - R + qrow
  for kt in range(0, base, BC) {{
    var s: tile<f32>[BR, BC] = 0.0
    for dk in range(0, D, TK) {{
      var q = Q[qrow :+ BR, qcol + dk :+ TK]
      var k = K[kt :+ BC, kcol + dk :+ TK]
      s += dot_t(q, k)
    }}
    s = s * {scale:.9}
    var mn: tile<f32>[BR, 1] = rowmax(s)
    mn = tmax(m, mn)
    s = exp(s - mn)
    var corr: tile<f32>[BR, 1] = exp(m - mn)
    l = l * corr + rowsum(s)
    var v = V[kt :+ BC, vcol :+ DV]
    acc = acc * corr
    acc += dot(s, v)
    m = mn
  }}

  for dt in range(0, BR, BC) {{
    var ds: tile<f32>[BR, BC] = 0.0
    for dk in range(0, D, TK) {{
      var q = Q[qrow :+ BR, qcol + dk :+ TK]
      var k = K[base + dt :+ BC, kcol + dk :+ TK]
      ds += dot_t(q, k)
    }}
    ds = where(iota(1) + dt <= iota(0), ds * {scale:.9}, -1.0e30)
    var dm: tile<f32>[BR, 1] = rowmax(ds)
    dm = tmax(m, dm)
    ds = exp(ds - dm)
    var dcorr: tile<f32>[BR, 1] = exp(m - dm)
    l = l * dcorr + rowsum(ds)
    var dv = V[base + dt :+ BC, vcol :+ DV]
    acc = acc * dcorr
    acc += dot(ds, dv)
    m = dm
  }}

  acc = acc / l
  O[qrow :+ BR, qcol + vs * DV :+ DV] = acc
}}
"
    )
}
