// A raw add-projection split along k across a wide grid.

use super::*;

/// Most slices [`Emit::proj_add_raw_split`] cuts `k` into.
const SPLIT_MAX: usize = 8;

/// Shortest `k` [`Emit::proj_add_raw_split`] splits. A shorter contraction
/// keeps its blocks brief, and the barrier would cost more than it saves.
const SPLIT_MIN_K: usize = 4096;

impl Emit {
    /// A raw-format [`Stage::ProjAddRaw`] split along `k`, when its tiles
    /// alone would leave most of the grid idle: each block walks a slice of
    /// `k` for one tile into f32 partials, then after a grid barrier the
    /// tiles add their partials onto `y`. `None` when the stage is not one,
    /// or its shape or the grid does not call for it.
    ///
    /// The partials take the scales plane of a scratch entry, the one f32
    /// scratch the pass allocates.
    pub(super) fn proj_add_raw_split(&mut self, key: &ChainKey, s: usize) -> Result<Option<bool>> {
        let Stage::ProjAddRaw { a, w, y, width } = key.stages[s] else {
            return Ok(None);
        };
        if !matches!(key.vals[w.0], Kind::Raw { .. }) {
            return Ok(None);
        }
        let (k, blocks) = (key.len_of(a), self.blocks as usize);
        let tiles = width / RAW_UNIT;
        if !width.is_multiple_of(RAW_UNIT) || self.shared.contains(&a) || 2 * tiles > blocks {
            return Ok(None);
        }
        if k < SPLIT_MIN_K || !k.is_multiple_of(256) {
            return Ok(None);
        }
        let nb = k / 256;
        let Some(splits) = (2..=SPLIT_MAX).rev().find(|&sp| nb.is_multiple_of(sp) && tiles * sp <= blocks) else {
            return Ok(None);
        };
        let (aq, asc) = self.quant(a, k, false);
        let (rq, rd, fmt) = self.raw_weight(key, w)?;
        let yn = self.given(y, key.len_of(y), View::Flat);
        let at = self.claim_scratch(Scratch {
            bytes: Q8_BLOCK,
            scales: splits * width,
        });
        let pp = format!("P{s}");
        self.declare_scratch(at, true, &pp, "f32", [1, (splits * width) as i64]);
        let rt = self.tune_const("RT".into(), RAW_UNIT);
        let sb = nb / splits;
        let units = tiles * splits;
        let iters = units.div_ceil(blocks);
        let _ = write!(
            self.body,
            "
  for k{s} in range(0, {iters}) {{
    let u{s} = p + k{s} * ctx.blocks.x
    if u{s} < {units} {{
      let ps{s} = u{s} / {tiles}
      let t{s} = (u{s} - ps{s} * {tiles}) * {rt}
"
        );
        let tot = self.raw_lanes(s, fmt, &rq, &rd, (&aq, &asc), &format!("t{s}"), &format!("ps{s} * {sb}"), sb);
        self.store_lanes(&pp, &format!("ps{s} * {width} + t{s}"), &tot, RAW_UNIT);
        let _ = write!(self.body, "    }}
  }}
");
        self.land_lane_stores();
        let sum = (0..splits)
            .map(|sp| format!("{pp}[0 :+ 1, {} + r{s} :+ {rt}]", sp * width))
            .collect::<Vec<_>>()
            .join(" + ");
        let _ = write!(
            self.body,
            "
  for j{s} in range(0, {}) {{
    let v{s} = p + j{s} * ctx.blocks.x
    if v{s} < {tiles} {{
      let r{s} = v{s} * {rt}
      {yn}[0 :+ 1, r{s} :+ {rt}] += {sum}
    }}
  }}
",
            tiles.div_ceil(blocks)
        );
        Ok(Some(true))
    }
}
