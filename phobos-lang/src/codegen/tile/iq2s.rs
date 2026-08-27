// IQ2_S's decode geometry, shared by the contraction (`iq2s_qdot.rs`) and the
// expansion (`qdecode.rs`). Two table lookups a lane; the sign table is keyed
// by the raw byte, so unlike IQ2_XXS there is no `l == 0` branch to fold away.

use super::*;

// IQ2_S block layout (phobos-gguf/src/quant/iq2_s.rs): 82 bytes, qs at byte
// 2, signs at byte 34, qh at byte 66, scales at byte 74.
pub(super) const IQ2S_BLOCK_BYTES: i64 = 80;
const IQ2S_QS_OFF: i64 = 0;
const IQ2S_SIGNS_OFF: i64 = 32;
const IQ2S_QH_OFF: i64 = 64;
const IQ2S_SCALES_OFF: i64 = 72;
pub(super) const IQ2S_LANE: i64 = 8;

/// Lane geometry: byte offsets, the divisor picking the lane's two qh bits,
/// and which half of the scale byte it takes.
pub(super) struct Iq2sLane<'c> {
    grid_off: Value<'c, 'c>,
    sign_off: Value<'c, 'c>,
    qh_off: Value<'c, 'c>,
    scale_off: Value<'c, 'c>,
    qh_div: Value<'c, 'c>,
    l_lt_2: Value<'c, 'c>,
    pub(super) k_lane_off: Value<'c, 'c>,
}

/// Per-block state: the scaled magnitude, and the lane's grid and sign
/// entries as one `vector<8xi8>` apiece.
pub(super) struct Iq2sBlock<'c> {
    pub(super) dl: Value<'c, 'c>,
    pub(super) grid_v: Value<'c, 'c>,
    pub(super) signs_v: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {
    /// The byte offsets warp lane `lane` reads, and its element offset.
    pub(super) fn iq2s_lane(&mut self, body: &Block<'c>, lane: Value<'c, 'c>) -> Result<Iq2sLane<'c>> {
        let i32_t = self.i32_t;
        let four = self.const_index(body, 4)?;
        let two_idx = self.const_index(body, 2)?;
        let ib32 = self.divui(body, lane, four)?;
        let l = self.remui(body, lane, four)?;

        let grid_off = self.addi(body, self.const_index(body, IQ2S_QS_OFF)?, lane)?;
        let sign_off = self.addi(body, self.const_index(body, IQ2S_SIGNS_OFF)?, lane)?;
        let qh_off = self.addi(body, self.const_index(body, IQ2S_QH_OFF)?, ib32)?;
        let scale_off = self.addi(body, self.const_index(body, IQ2S_SCALES_OFF)?, ib32)?;

        // qh_div = 1 << (2 * l): 4^l.
        let l_i32 = self.numeric_cast(body, l, i32_t)?;
        let two_i32 = self.const_i32(body, 2)?;
        let qh_shift = self.push(body, arith::muli(l_i32, two_i32, self.loc))?;
        let one_i32 = self.const_i32(body, 1)?;
        let qh_div = self.push(body, arith::shli(one_i32, qh_shift, self.loc))?;

        // The scale nibble: lane % 4 < 2 takes the low nibble, else the high.
        let l_lt_2 = self.push(
            body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, l, two_idx, self.loc),
        )?;

        let k_lane_off = self.muli(body, lane, self.const_index(body, IQ2S_LANE)?)?;
        Ok(Iq2sLane {
            grid_off,
            sign_off,
            qh_off,
            scale_off,
            qh_div,
            l_lt_2,
            k_lane_off,
        })
    }

    /// The lane's per-block decode state for the block `at` names.
    pub(super) fn iq2s_block(
        &mut self,
        kb: &Block<'c>,
        geom: &Iq2sLane<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        tables: &QTables<'_, 'c>,
        at: &BlockAt<'c>,
    ) -> Result<Iq2sBlock<'c>> {
        let f32_t = self.f32_t;
        let grid_byte = self.qbyte(kb, qb, at.j, at.off, geom.grid_off)?;
        let sign_byte = self.qbyte(kb, qb, at.j, at.off, geom.sign_off)?;
        let qh_byte = self.qbyte(kb, qb, at.j, at.off, geom.qh_off)?;
        let scale_byte = self.qbyte(kb, qb, at.j, at.off, geom.scale_off)?;

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

        let c4 = self.const_i32(kb, 4)?;
        let c256 = self.const_i32(kb, 256)?;
        let qh_bits = self.push(kb, arith::divui(qh_byte, geom.qh_div, self.loc))?;
        let qh_bits = self.push(kb, arith::remui(qh_bits, c4, self.loc))?;
        let qh_bits = self.push(kb, arith::muli(qh_bits, c256, self.loc))?;
        let grid_idx = self.push(kb, arith::addi(grid_byte, qh_bits, self.loc))?;
        let grid_idx = self.numeric_cast(kb, grid_idx, self.index_t)?;
        let eight_idx = self.const_index(kb, 8)?;
        let grid_idx8 = self.push(kb, arith::muli(grid_idx, eight_idx, self.loc))?;
        let sign_idx = self.numeric_cast(kb, sign_byte, self.index_t)?;
        let sign_idx8 = self.push(kb, arith::muli(sign_idx, eight_idx, self.loc))?;
        // Eight contiguous bytes apiece: one load each, not one per element.
        let vec_t = Type::vector(&[IQ2S_LANE as u64], self.i8_t);
        let z = self.const_index(kb, 0)?;
        let grid_v = self.vec_load_al(kb, tables.grid.mem, &[z, grid_idx8], vec_t, 8)?;
        let signs_v = self.vec_load_al(kb, tables.signs.mem, &[z, sign_idx8], vec_t, 8)?;
        Ok(Iq2sBlock { dl, grid_v, signs_v })
    }

    /// The lane's `y`-th decoded weight, from the entries already in
    /// registers: a grid magnitude times its sign.
    pub(super) fn iq2s_decoded(
        &mut self,
        kb: &Block<'c>,
        blk: &Iq2sBlock<'c>,
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
