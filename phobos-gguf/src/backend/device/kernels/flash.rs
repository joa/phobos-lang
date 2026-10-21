// Prompt attention on the tensor cores with the scores never written out.

use crate::backend::Attn;

/// Query rows of one [`attn_flash_src`] program.
pub(crate) const FLASH_TILE: usize = 64;

/// Keys of one [`attn_flash_src`] step. Narrower than the query tile so the
/// streamed form's scores and values leave shared memory for two programs a
/// multiprocessor.
pub(crate) const FLASH_KEYS: usize = 32;

/// The form and key step of [`attn_flash_src`] this chip takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FlashShape {
    pub(crate) form: FlashForm,
    pub(crate) keys: usize,
    /// Asserts the key loop's double-buffered staging, which the compiler
    /// otherwise declines for the resident form.
    pub(crate) pipeline: bool,
}

impl FlashShape {
    /// The shape for `chip`. `PHOBOS_ATTN_FLASH_SHAPE` overrides it, to sweep
    /// without a rebuild: `streamed` or `resident`, then a key step that
    /// divides [`FLASH_TILE`], then `p` to assert the pipeline, such as
    /// `resident16p`.
    pub(crate) fn for_chip(chip: &str) -> Self {
        let form = if chip == "sm_75" { FlashForm::Streamed } else { FlashForm::Resident };
        let chosen = Self { form, keys: FLASH_KEYS, pipeline: false };
        let Ok(spec) = std::env::var("PHOBOS_ATTN_FLASH_SHAPE") else {
            return chosen;
        };
        let (form, keys) = match (spec.strip_prefix("streamed"), spec.strip_prefix("resident")) {
            (Some(keys), _) => (FlashForm::Streamed, keys),
            (_, Some(keys)) => (FlashForm::Resident, keys),
            _ => return chosen,
        };
        let (keys, pipeline) = keys.strip_suffix('p').map_or((keys, false), |keys| (keys, true));
        match keys.parse::<usize>() {
            Ok(keys) if keys >= 16 && FLASH_TILE.is_multiple_of(keys) => Self { form, keys, pipeline },
            _ => chosen,
        }
    }
}

/// How [`attn_flash_src`] reads the queries and keys, chosen by the chip's
/// shared memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FlashForm {
    /// The query and key tiles staged a contraction step at a time, both
    /// again every key step, in shared memory sm_75 has.
    Streamed,
    /// The query tile held for the whole program and each key tile staged
    /// once, the full head wide. A value slice is the whole head, so the
    /// scores are computed once. Needs the larger shared memory of sm_80 on.
    Resident,
}

impl FlashForm {
    /// Output columns one program accumulates. Its accumulator and scores
    /// stay in the tensor cores' registers only while they fit the lane
    /// budget, so the streamed form splits a wide head's values across the
    /// grid and recomputes the scores per slice.
    pub(crate) fn value_cols(self, head_dim: usize) -> usize {
        match self {
            Self::Streamed => head_dim.min(128),
            Self::Resident => head_dim,
        }
    }
}

/// Whether [`attn_flash_src`] takes this prompt: whole tiles of rows and
/// cache, and a head dimension the value slices and the contraction step
/// divide.
pub(crate) fn attn_flash_fits(spec: Attn, form: FlashForm) -> bool {
    spec.rows.is_multiple_of(FLASH_TILE)
        && spec.start_pos.is_multiple_of(FLASH_TILE)
        && spec.head_dim.is_multiple_of(form.value_cols(spec.head_dim))
        && spec.head_dim.is_multiple_of(32)
}

/// The scores `name` of the query tile against the key tile at row `at`.
fn scores(form: FlashForm, name: &str, at: &str) -> String {
    match form {
        FlashForm::Streamed => format!(
            "var {name}: tile<f32>[BR, BC] = 0.0
    for dk in range(0, D, TK) {{
      var q = Q[qrow :+ BR, qcol + dk :+ TK]
      var k = K[{at} :+ BC, kcol + dk :+ TK]
      {name} += dot_t(q, k)
    }}"
        ),
        FlashForm::Resident => format!(
            "var k = K[{at} :+ BC, kcol :+ D]
    var {name}: tile<f32>[BR, BC] = dot_t(q, k)"
        ),
    }
}

/// Causal attention for a prompt, one query tile, value slice and head per
/// program, as FlashAttention-2: the scores of a key step take the running
/// softmax in registers and feed the mix through one shared tile. The query
/// tile's own keys come last, masked. The queries arrive narrowed to f16 and
/// the scores round to it on the way into the mix; the softmax and both
/// accumulations stay f32.
pub(crate) fn attn_flash_src(head_dim: usize, group: usize, shape: FlashShape) -> String {
    let form = shape.form;
    let scale = (head_dim as f32).sqrt().recip();
    let (tile, keys, cols) = (FLASH_TILE, shape.keys, form.value_cols(head_dim));
    let query = if form == FlashForm::Resident { "  var q = Q[qrow :+ BR, qcol :+ D]\n" } else { "" };
    let (step, diag) = (scores(form, "s", "kt"), scores(form, "ds", "base + dt"));
    let pipeline = if shape.pipeline { "@pipeline(on)\n" } else { "" };
    format!(
        "@tensorcore
@launch(128)
@dynshared
{pipeline}@autotune(BR in [{tile}], BC in [{keys}], TK in [32], D in [{head_dim}], DV in [{cols}], G in [{group}])
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
{query}  var acc: tile<f32>[BR, DV] = 0.0
  var m: tile<f32>[BR, 1] = -1.0e30
  var l: tile<f32>[BR, 1] = 0.0

  let base = NK - R + qrow
  for kt in range(0, base, BC) {{
    {step}
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
    {diag}
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
