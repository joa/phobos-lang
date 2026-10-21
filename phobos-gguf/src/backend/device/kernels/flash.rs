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
        // Measured on an sm_86 A40, the resident form in 16-key steps wins,
        // small enough for two programs a multiprocessor; no other chip has
        // been measured, and elsewhere the streamed form is the known one.
        let chosen = match chip {
            "sm_86" => Self { form: FlashForm::Resident, keys: 16, pipeline: false },
            _ => Self { form: FlashForm::Streamed, keys: FLASH_KEYS, pipeline: false },
        };
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
    /// scores are computed once, in about three times the shared memory.
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
///
/// With `splits` over one, program `(qt, sp * slices + vs, h)` takes the
/// `sp`-th share of the whole key tiles below the query tile, and the last
/// share the query tile's own keys too. Each writes its unnormalized
/// accumulator to `P` and its row max and sum to `ML`, columns `2h` and
/// `2h + 1`, all at row `sp * R + qrow`, for [`attn_merge_src`] to combine.
pub(crate) fn attn_flash_src(head_dim: usize, group: usize, shape: FlashShape, splits: usize) -> String {
    let form = shape.form;
    let scale = (head_dim as f32).sqrt().recip();
    let (tile, keys, cols) = (FLASH_TILE, shape.keys, form.value_cols(head_dim));
    let slices = head_dim / cols;
    let query = if form == FlashForm::Resident { "  var q = Q[qrow :+ BR, qcol :+ D]\n" } else { "" };
    let (step, diag) = (scores(form, "s", "kt"), scores(form, "ds", "base + dt"));
    let pipeline = if shape.pipeline { "@pipeline(on)\n" } else { "" };
    let split = splits > 1;
    let (outputs, place, range, diag_len, store) = if split {
        (
            "P: tensor<f32>[PR, QW], ML: tensor<f32>[PR, MW]",
            "  let sp = ctx.block.y / SL\n  let vs = ctx.block.y - sp * SL",
            "  let tiles = base / BC\n  let per = (tiles + SP - 1) / SP\n  let k0 = min(sp * per, tiles) * BC\n  let k1 = min(sp * per + per, tiles) * BC\n  for kt in range(k0, k1, BC) {",
            "BR * (sp / (SP - 1))",
            "  let prow = sp * R + qrow\n  P[prow :+ BR, qcol + vs * DV :+ DV] = acc\n  ML[prow :+ BR, 2 * h :+ 1] = m\n  ML[prow :+ BR, 2 * h + 1 :+ 1] = l",
        )
    } else {
        (
            "O: tensor<f32>[R, QW]",
            "  let vs = ctx.block.y",
            "  for kt in range(0, base, BC) {",
            "BR",
            "  acc = acc / l\n  O[qrow :+ BR, qcol + vs * DV :+ DV] = acc",
        )
    };
    let consts = if split { format!(", SP in [{splits}], SL in [{slices}]") } else { String::new() };
    let aligned = if split { ", PR = BR" } else { "" };
    format!(
        "@tensorcore
@launch(128)
@dynshared
{pipeline}@autotune(BR in [{tile}], BC in [{keys}], TK in [32], D in [{head_dim}], DV in [{cols}], G in [{group}]{consts})
@aligned(R = BR, NK = BR, QW = D, KW = D{aligned})
kernel attn_flash(Q: tensor<f16>[R, QW], K: tensor<f16>[NK, KW], V: tensor<f16>[NK, KW],
                  {outputs}) {{
  let qt = ctx.block.x
{place}
  let h = ctx.block.z
  let qrow = qt * BR
  let qcol = h * D
  let kcol = h / G * D
  let vcol = kcol + vs * DV
{query}  var acc: tile<f32>[BR, DV] = 0.0
  var m: tile<f32>[BR, 1] = -1.0e30
  var l: tile<f32>[BR, 1] = 0.0

  let base = NK - R + qrow
{range}
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

  for dt in range(0, {diag_len}, BC) {{
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

{store}
}}
"
    )
}

/// Rows one program of [`attn_merge_src`] combines.
pub(crate) const FLASH_MERGE_ROWS: usize = 16;

/// The splits of [`attn_flash_src`] combined: each split's accumulator
/// weighed by `exp(m_s - m)` against the largest row max `m`, divided by
/// the sums weighed alike. One program a block of rows and a head.
pub(crate) fn attn_merge_src(head_dim: usize, splits: usize) -> String {
    let rows = FLASH_MERGE_ROWS;
    format!(
        "@launch(128)
@autotune(TR in [{rows}], D in [{head_dim}], SP in [{splits}])
@aligned(R = TR, PR = TR, QW = D)
kernel attn_merge(P: tensor<f32>[PR, QW], ML: tensor<f32>[PR, MW], O: tensor<f32>[R, QW]) {{
  let r0 = ctx.block.x * TR
  let h = ctx.block.y
  var mx: tile<f32>[TR, 1] = -1.0e30
  for s in range(0, SP) {{
    mx = tmax(mx, ML[s * R + r0 :+ TR, 2 * h :+ 1])
  }}
  var den: tile<f32>[TR, 1] = 0.0
  var acc: tile<f32>[TR, D] = 0.0
  for s in range(0, SP) {{
    var w: tile<f32>[TR, 1] = exp(ML[s * R + r0 :+ TR, 2 * h :+ 1] - mx)
    den += w * ML[s * R + r0 :+ TR, 2 * h + 1 :+ 1]
    acc += P[s * R + r0 :+ TR, h * D :+ D] * w
  }}
  O[r0 :+ TR, h * D :+ D] = acc / den
}}
"
    )
}

/// Programs a multiprocessor the split aims the grid at: a few waves of
/// two each, so the last wave's tail is short.
const FLASH_PROGRAMS_PER_SM: usize = 8;

/// Keys a split takes at least, so its share outweighs the merge.
const FLASH_SPLIT_KEYS: usize = 1024;

/// Most splits of one query tile's keys.
const FLASH_MAX_SPLITS: usize = 8;

/// How many ways [`attn_flash_src`] splits this prompt's keys on a card of
/// `sms` multiprocessors: enough programs for a few waves, while each share
/// keeps [`FLASH_SPLIT_KEYS`] keys. One at a shallow depth.
pub(crate) fn flash_splits(spec: Attn, form: FlashForm, sms: usize) -> usize {
    let programs = spec.rows / FLASH_TILE * (spec.head_dim / form.value_cols(spec.head_dim)) * spec.n_head;
    let wanted = (FLASH_PROGRAMS_PER_SM * sms).div_ceil(programs.max(1));
    wanted.min(spec.start_pos / FLASH_SPLIT_KEYS).clamp(1, FLASH_MAX_SPLITS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(rows: usize, start_pos: usize, n_head: usize) -> Attn {
        Attn { rows, start_pos, n_head, n_kv: 4, head_dim: 256 }
    }

    #[test]
    fn a_deep_prompt_splits_its_keys_to_fill_the_card() {
        // The 27B on an A40: 8 query tiles of 24 heads, a wave and a bit.
        assert_eq!(flash_splits(spec(512, 32256, 24), FlashForm::Resident, 84), 4);
        // Shallow: every share would be short.
        assert_eq!(flash_splits(spec(512, 512, 24), FlashForm::Resident, 84), 1);
        assert_eq!(flash_splits(spec(512, 2048, 24), FlashForm::Resident, 84), 2);
        // A grid that already fills the card stays whole.
        assert_eq!(flash_splits(spec(4096, 32256, 24), FlashForm::Resident, 84), 1);
    }
}
