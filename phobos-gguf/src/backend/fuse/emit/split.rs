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
        let Kind::Raw { quant, .. } = key.vals[w.0] else {
            return Ok(None);
        };
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
        self.scratch.push(Scratch {
            bytes: Q8_BLOCK,
            scales: splits * width,
        });
        let at = self.scratch.len() - 1;
        let pp = self.slot(format!("P{s}"), "f32", [1, (splits * width) as i64], Bound::ScratchScales(at));
        let rt = self.tune_const("RT".into(), RAW_UNIT);
        // A slice's bytes and scales start at eight times its block offset,
        // since the weight is grouped by eight rows.
        let sb = nb / splits;
        let (sk, skb, srb) = (sb * 256, sb * 256 / Q8_BLOCK, sb * quant.device_block_bytes());
        let (gk, gb) = (8 * srb, 8 * sb);
        let units = tiles * splits;
        let iters = units.div_ceil(blocks);
        let _ = write!(
            self.body,
            "
  for k{s} in range(0, {iters}) {{
    let u{s} = p + k{s} * BLOCKS
    if u{s} < {units} {{
      let ps{s} = u{s} / {tiles}
      let t{s} = (u{s} - ps{s} * {tiles}) * {rt}
      {pp}[0 :+ 1, ps{s} * {width} + t{s} :+ {rt}] = {fmt}_qdot_i8_t({aq}[0 :+ 1, ps{s} * {sk} :+ {sk}],
          {asc}[0 :+ 1, ps{s} * {skb} :+ {skb}], {rq}[t{s} :+ {rt}, ps{s} * {gk} :+ {srb}],
          {rd}[t{s} :+ {rt}, ps{s} * {gb} :+ {sb}])
    }}
  }}
"
        );
        self.barrier();
        let sum = (0..splits)
            .map(|sp| format!("{pp}[0 :+ 1, {} + r{s} :+ {rt}]", sp * width))
            .collect::<Vec<_>>()
            .join(" + ");
        let _ = write!(
            self.body,
            "
  for j{s} in range(0, {}) {{
    let v{s} = p + j{s} * BLOCKS
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
