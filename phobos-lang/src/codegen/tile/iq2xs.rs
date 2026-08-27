// IQ2_XS's decode geometry, shared by the contraction (`iq2xs_qdot.rs`) and
// the expansion (`qdecode.rs`). A 16-bit qs halfword splits into a magnitude
// index and a sign index via `% 512` and `/ 512`; the sign table is IQ2_XXS's
// own, reused outright.

use super::*;

// IQ2_XS block layout (phobos-gguf/src/quant/iq2_xs.rs): 74 bytes, qs at
// byte 2 (two bytes a lane), scales at byte 66.
pub(super) const IQ2XS_BLOCK_BYTES: i64 = 74;
const IQ2XS_QS_OFF: i64 = 2;
const IQ2XS_SCALES_OFF: i64 = 66;
pub(super) const IQ2XS_LANE: i64 = 8;

/// Lane geometry: byte offsets, and which half of the scale byte the lane
/// takes.
pub(super) struct Iq2xsLane<'c> {
    lo_off: Value<'c, 'c>,
    hi_off: Value<'c, 'c>,
    scale_off: Value<'c, 'c>,
    l_lt_2: Value<'c, 'c>,
    pub(super) k_lane_off: Value<'c, 'c>,
}

/// Per-block state: the scaled magnitude, and the lane's grid and sign
/// entries as one `vector<8xi8>` apiece.
pub(super) struct Iq2xsBlock<'c> {
    pub(super) dl: Value<'c, 'c>,
    pub(super) grid_v: Value<'c, 'c>,
    pub(super) signs_v: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {
    /// The byte offsets warp lane `lane` reads, and its element offset.
    pub(super) fn iq2xs_lane(
        &mut self,
        body: &Block<'c>,
        lane: Value<'c, 'c>,
    ) -> Result<Iq2xsLane<'c>> {
        let four = self.const_index(body, 4)?;
        let two_idx = self.const_index(body, 2)?;
        let one_idx = self.const_index(body, 1)?;
        let ib32 = self.divui(body, lane, four)?;
        let l = self.remui(body, lane, four)?;

        let lo_off = self.addi(
            body,
            self.const_index(body, IQ2XS_QS_OFF)?,
            self.muli(body, lane, two_idx)?,
        )?;
        let hi_off = self.addi(body, lo_off, one_idx)?;
        let scale_off = self.addi(body, self.const_index(body, IQ2XS_SCALES_OFF)?, ib32)?;

        let l_lt_2 = self.push(
            body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, l, two_idx, self.loc),
        )?;

        let k_lane_off = self.muli(body, lane, self.const_index(body, IQ2XS_LANE)?)?;
        Ok(Iq2xsLane {
            lo_off,
            hi_off,
            scale_off,
            l_lt_2,
            k_lane_off,
        })
    }

    /// The lane's per-block decode state for the block `at` names.
    pub(super) fn iq2xs_block(
        &mut self,
        kb: &Block<'c>,
        geom: &Iq2xsLane<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        tables: &QTables<'_, 'c>,
        at: &BlockAt<'c>,
    ) -> Result<Iq2xsBlock<'c>> {
        let f32_t = self.f32_t;
        let lo = self.qbyte(kb, qb, at.j, at.off, geom.lo_off)?;
        let hi = self.qbyte(kb, qb, at.j, at.off, geom.hi_off)?;
        let scale_byte = self.qbyte(kb, qb, at.j, at.off, geom.scale_off)?;

        let c256 = self.const_i32(kb, 256)?;
        let hi_shifted = self.push(kb, arith::muli(hi, c256, self.loc))?;
        let q16 = self.push(kb, arith::addi(lo, hi_shifted, self.loc))?;

        let c16 = self.const_i32(kb, 16)?;
        let lo_nibble = self.push(kb, arith::remui(scale_byte, c16, self.loc))?;
        let hi_nibble = self.push(kb, arith::divui(scale_byte, c16, self.loc))?;
        let nibble = self.push(kb, arith::select(geom.l_lt_2, lo_nibble, hi_nibble, self.loc))?;
        let nibble_f32 = self.numeric_cast(kb, nibble, f32_t)?;
        let half = self.const_f32(kb, 0.5)?;
        let quarter = self.const_f32(kb, 0.25)?;
        let d_val = self.push(kb, memref::load(d.mem, &[at.j, at.blk], self.loc))?;
        let d_f32 = self.numeric_cast(kb, d_val, f32_t)?;
        let sc = self.push(kb, arith::addf(half, nibble_f32, self.loc))?;
        let sc = self.push(kb, arith::mulf(sc, quarter, self.loc))?;
        let dl = self.push(kb, arith::mulf(d_f32, sc, self.loc))?;

        let c512 = self.const_i32(kb, 512)?;
        let mag_idx = self.push(kb, arith::remui(q16, c512, self.loc))?;
        let sign_idx = self.push(kb, arith::divui(q16, c512, self.loc))?;
        let mag_idx = self.numeric_cast(kb, mag_idx, self.index_t)?;
        let sign_idx = self.numeric_cast(kb, sign_idx, self.index_t)?;
        let eight_idx = self.const_index(kb, 8)?;
        let mag_idx8 = self.push(kb, arith::muli(mag_idx, eight_idx, self.loc))?;
        let sign_idx8 = self.push(kb, arith::muli(sign_idx, eight_idx, self.loc))?;
        // Eight contiguous bytes apiece: one load each, not one per element.
        let vec_t = Type::vector(&[IQ2XS_LANE as u64], self.i8_t);
        let z = self.const_index(kb, 0)?;
        let grid_v = self.vec_load_al(kb, tables.grid.mem, &[z, mag_idx8], vec_t, 8)?;
        let signs_v = self.vec_load_al(kb, tables.signs.mem, &[z, sign_idx8], vec_t, 8)?;
        Ok(Iq2xsBlock { dl, grid_v, signs_v })
    }

    /// The lane's `y`-th decoded weight, from the entries already in
    /// registers: a grid magnitude times its sign.
    pub(super) fn iq2xs_decoded(
        &mut self,
        kb: &Block<'c>,
        blk: &Iq2xsBlock<'c>,
        y: i64,
    ) -> Result<Value<'c, 'c>> {
        let (i8_t, f32_t) = (self.i8_t, self.f32_t);
        let mag_val = self.vec_extract(kb, blk.grid_v, &[y], i8_t)?;
        let sign_val = self.vec_extract(kb, blk.signs_v, &[y], i8_t)?;
        let mag_f32 = self.numeric_cast(kb, mag_val, f32_t)?;
        let sign_f32 = self.numeric_cast(kb, sign_val, f32_t)?;
        let decoded = self.push(kb, arith::mulf(blk.dl, mag_f32, self.loc))?;
        self.push(kb, arith::mulf(decoded, sign_f32, self.loc))
    }
}
