// IQ2_XXS's decode geometry, shared by the contraction (`iq2xxs_qdot.rs`) and
// the expansion (`qdecode.rs`). Two table lookups a lane, a magnitude grid and
// a sign table, and unlike IQ1_S a real branch on the lane's byte split:
// `l == 0` has no `hi` byte.

use super::*;

// IQ2_XXS block layout (phobos-gguf/src/quant/iq2_xxs.rs): 66 bytes, the
// qs/aux plane at byte 2.
pub(super) const IQ2XXS_BLOCK_BYTES: i64 = 66;
const IQ2XXS_QS_OFF: i64 = 2;
pub(super) const IQ2XXS_LANE: i64 = 8;

/// Lane geometry: byte offsets within a block, and the lane's element offset.
/// `hi_off` re-reads `lo_off` at `l == 0`; the select below drops it.
pub(super) struct Iq2xxsLane<'c> {
    l: Value<'c, 'c>,
    grid_off: Value<'c, 'c>,
    scale_off: Value<'c, 'c>,
    lo_off: Value<'c, 'c>,
    hi_off: Value<'c, 'c>,
    pub(super) zero_idx: Value<'c, 'c>,
    eight_idx: Value<'c, 'c>,
    pub(super) k_lane_off: Value<'c, 'c>,
}

/// Per-block state: the scaled magnitude, and the lane's grid and sign
/// entries as one `vector<8xi8>` apiece.
pub(super) struct Iq2xxsBlock<'c> {
    pub(super) db: Value<'c, 'c>,
    pub(super) grid_v: Value<'c, 'c>,
    pub(super) signs_v: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {
    /// The byte offsets warp lane `lane` reads, and its element offset.
    /// `hi_off` folds to a re-read of `lo_off` at `l == 0` (harmless, in
    /// bounds); `arith.select` drops its contribution below.
    pub(super) fn iq2xxs_lane(
        &mut self,
        body: &Block<'c>,
        lane: Value<'c, 'c>,
    ) -> Result<Iq2xxsLane<'c>> {
        let four = self.const_index(body, 4)?;
        let eight_idx = self.const_index(body, 8)?;
        let zero_idx = self.const_index(body, 0)?;
        let one_idx = self.const_index(body, 1)?;
        let ib32 = self.divui(body, lane, four)?;
        let l = self.remui(body, lane, four)?;
        let is_l0 = self.push(
            body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, l, zero_idx, self.loc),
        )?;

        // chunk = QS_OFF + 8*ib32; aux = chunk+4; grid_off = chunk+l;
        // scale_off = aux+3.
        let chunk = self.addi(
            body,
            self.const_index(body, IQ2XXS_QS_OFF)?,
            self.muli(body, ib32, eight_idx)?,
        )?;
        let aux = self.addi(body, chunk, four)?;
        let grid_off = self.addi(body, chunk, l)?;
        let three_idx = self.const_index(body, 3)?;
        let scale_off = self.addi(body, aux, three_idx)?;

        // lo_off = aux + (max(l, 1) - 1): aux+0 for l in {0,1}, aux+1 for
        // l=2, aux+2 for l=3, matching run_geometry's match arms.
        let l_or_1 = self.push(body, arith::select(is_l0, one_idx, l, self.loc))?;
        let lo_off = self.addi(body, aux, self.subi(body, l_or_1, one_idx)?)?;
        let hi_off = self.addi(body, aux, l)?;

        let k_lane_off = self.muli(body, lane, self.const_index(body, IQ2XXS_LANE)?)?;
        Ok(Iq2xxsLane {
            l,
            grid_off,
            scale_off,
            lo_off,
            hi_off,
            zero_idx,
            eight_idx,
            k_lane_off,
        })
    }

    /// The lane's per-block decode state for the block `at` names.
    pub(super) fn iq2xxs_block(
        &mut self,
        kb: &Block<'c>,
        geom: &Iq2xxsLane<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        tables: &QTables<'_, 'c>,
        at: &BlockAt<'c>,
    ) -> Result<Iq2xxsBlock<'c>> {
        let (i32_t, f32_t) = (self.i32_t, self.f32_t);
        let grid_idx = self.qbyte(kb, qb, at.j, at.off, geom.grid_off)?;
        let scale = self.qbyte(kb, qb, at.j, at.off, geom.scale_off)?;
        let lo = self.qbyte(kb, qb, at.j, at.off, geom.lo_off)?;
        let hi = self.qbyte(kb, qb, at.j, at.off, geom.hi_off)?;

        let is_l0_i32 = self.push(
            kb,
            arith::cmpi(
                self.ctx,
                arith::CmpiPredicate::Eq,
                geom.l,
                self.const_index(kb, 0)?,
                self.loc,
            ),
        )?;
        let zero_i32 = self.zero_scalar(kb, i32_t)?;
        let hi_term = self.push(kb, arith::select(is_l0_i32, zero_i32, hi, self.loc))?;
        let c256 = self.const_index(kb, 256)?;
        let c256_i32 = self.numeric_cast(kb, c256, i32_t)?;
        let hi_shifted = self.push(kb, arith::muli(hi_term, c256_i32, self.loc))?;
        let word = self.push(kb, arith::addi(lo, hi_shifted, self.loc))?;

        // shift_div = 1 for l == 0, else 1 << (8 - l): 128, 64, 32 for l =
        // 1, 2, 3, matching run_geometry's match arms.
        let eight_i32 = self.numeric_cast(kb, self.const_index(kb, 8)?, i32_t)?;
        let l_i32 = self.numeric_cast(kb, geom.l, i32_t)?;
        let exp = self.push(kb, arith::subi(eight_i32, l_i32, self.loc))?;
        let zero_exp = self.zero_scalar(kb, i32_t)?;
        let exp = self.push(kb, arith::select(is_l0_i32, zero_exp, exp, self.loc))?;
        let one_i32 = self.const_i32(kb, 1)?;
        let shift_div = self.push(kb, arith::shli(one_i32, exp, self.loc))?;

        let c128 = self.const_i32(kb, 128)?;
        let signs_idx = self.push(kb, arith::divui(word, shift_div, self.loc))?;
        let signs_idx = self.push(kb, arith::remui(signs_idx, c128, self.loc))?;
        let signs_idx = self.numeric_cast(kb, signs_idx, self.index_t)?;
        let signs_idx8 = self.push(kb, arith::muli(signs_idx, geom.eight_idx, self.loc))?;

        let c16 = self.const_i32(kb, 16)?;
        let scale_shifted = self.push(kb, arith::divui(scale, c16, self.loc))?;
        let scale_f32 = self.numeric_cast(kb, scale_shifted, f32_t)?;
        let half = self.const_f32(kb, 0.5)?;
        let quarter = self.const_f32(kb, 0.25)?;
        let d_val = self.push(kb, memref::load(d.mem, &[at.j, at.blk], self.loc))?;
        let d_f32 = self.numeric_cast(kb, d_val, f32_t)?;
        let sc = self.push(kb, arith::addf(half, scale_f32, self.loc))?;
        let sc = self.push(kb, arith::mulf(sc, quarter, self.loc))?;
        let db = self.push(kb, arith::mulf(d_f32, sc, self.loc))?;

        let grid_idx = self.numeric_cast(kb, grid_idx, self.index_t)?;
        let grid_idx8 = self.push(kb, arith::muli(grid_idx, geom.eight_idx, self.loc))?;
        // Eight contiguous bytes apiece: one load each, not one per element.
        let vec_t = Type::vector(&[IQ2XXS_LANE as u64], self.i8_t);
        let z = geom.zero_idx;
        let grid_v = self.vec_load_al(kb, tables.grid.mem, &[z, grid_idx8], vec_t, 8)?;
        let signs_v = self.vec_load_al(kb, tables.signs.mem, &[z, signs_idx8], vec_t, 8)?;
        Ok(Iq2xxsBlock { db, grid_v, signs_v })
    }

    /// The lane's `y`-th decoded weight, from the entries already in
    /// registers: a grid magnitude times its sign.
    pub(super) fn iq2xxs_decoded(
        &mut self,
        kb: &Block<'c>,
        blk: &Iq2xxsBlock<'c>,
        y: i64,
    ) -> Result<Value<'c, 'c>> {
        let (i8_t, f32_t) = (self.i8_t, self.f32_t);
        let mag_val = self.vec_extract(kb, blk.grid_v, &[y], i8_t)?;
        let sign_val = self.vec_extract(kb, blk.signs_v, &[y], i8_t)?;
        let mag_f32 = self.numeric_cast(kb, mag_val, f32_t)?;
        let sign_f32 = self.numeric_cast(kb, sign_val, f32_t)?;
        let decoded = self.push(kb, arith::mulf(blk.db, mag_f32, self.loc))?;
        self.push(kb, arith::mulf(decoded, sign_f32, self.loc))
    }
}
