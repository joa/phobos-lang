// IQ3_S's decode geometry, shared by the contraction (`iq3s_qdot.rs`) and the
// expansion (`qdecode.rs`). IQ3_XXS's two four-wide grid entries again, but
// with IQ2_S's direct-byte sign table: no lo/hi assembly at all, a lane reads
// its sign index outright.

use super::*;

// IQ3_S block layout (phobos-gguf/src/quant/iq3_s.rs): 110 bytes, qs at
// byte 2 (two grid-index bytes a lane), qh at byte 66 (one byte a half),
// signs at byte 74, scales at byte 106.
pub(super) const IQ3S_BLOCK_BYTES: i64 = 108;
const IQ3S_QS_OFF: i64 = 0;
const IQ3S_QH_OFF: i64 = 64;
const IQ3S_SIGNS_OFF: i64 = 72;
const IQ3S_SCALES_OFF: i64 = 104;
/// Elements a lane takes from one of its two grid entries.
pub(super) const IQ3S_HALF: i64 = 4;

/// Elements a lane owns: both halves of its grid entry.
pub(super) const IQ3S_LANE: i64 = 2 * IQ3S_HALF;

/// Lane geometry: byte offsets, the two divisors picking the lane's ninth
/// grid-index bits, and which half of the scale byte it takes.
pub(super) struct Iq3sLane<'c> {
    grid1_off: Value<'c, 'c>,
    grid2_off: Value<'c, 'c>,
    qh_off: Value<'c, 'c>,
    signs_off: Value<'c, 'c>,
    scale_off: Value<'c, 'c>,
    qh_div1: Value<'c, 'c>,
    qh_div2: Value<'c, 'c>,
    half_is_0: Value<'c, 'c>,
    pub(super) zero_idx: Value<'c, 'c>,
    pub(super) k_lane_off: Value<'c, 'c>,
}

/// Per-block state: the scaled magnitude, two four-wide grid entries and one
/// eight-wide sign entry, each one vector load.
pub(super) struct Iq3sBlock<'c> {
    pub(super) db: Value<'c, 'c>,
    pub(super) g1_v: Value<'c, 'c>,
    pub(super) g2_v: Value<'c, 'c>,
    pub(super) signs_v: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {
    /// The byte offsets warp lane `lane` reads, and its element offset.
    pub(super) fn iq3s_lane(&mut self, body: &Block<'c>, lane: Value<'c, 'c>) -> Result<Iq3sLane<'c>> {
        let i32_t = self.i32_t;
        let eight_idx = self.const_index(body, 8)?;
        let four_idx = self.const_index(body, 4)?;
        let two_idx = self.const_index(body, 2)?;
        let one_idx = self.const_index(body, 1)?;
        let o = self.divui(body, lane, eight_idx)?;
        let rem = self.remui(body, lane, eight_idx)?;
        let half = self.divui(body, rem, four_idx)?;
        let l = self.remui(body, rem, four_idx)?;

        // grid1_off = QS_OFF + 2*lane (16*o + 8*half + 2*l == 2*lane).
        let grid1_off = self.addi(body, self.const_index(body, IQ3S_QS_OFF)?, self.muli(body, lane, two_idx)?)?;
        let grid2_off = self.addi(body, grid1_off, one_idx)?;
        // qh_off = QH_OFF + 2*o + half.
        let qh_off = self.addi(
            body,
            self.addi(body, self.const_index(body, IQ3S_QH_OFF)?, self.muli(body, o, two_idx)?)?,
            half,
        )?;
        // signs_off = SIGNS_OFF + lane (8*o + 4*half + l == lane).
        let signs_off = self.addi(body, self.const_index(body, IQ3S_SIGNS_OFF)?, lane)?;
        let scale_off = self.addi(body, self.const_index(body, IQ3S_SCALES_OFF)?, o)?;

        // qh_div1 = 1 << (2*l), qh_div2 = 1 << (2*l + 1).
        let l_i32 = self.numeric_cast(body, l, i32_t)?;
        let two_i32 = self.const_i32(body, 2)?;
        let one_i32 = self.const_i32(body, 1)?;
        let exp1 = self.push(body, arith::muli(l_i32, two_i32, self.loc))?;
        let exp2 = self.push(body, arith::addi(exp1, one_i32, self.loc))?;
        let qh_div1 = self.push(body, arith::shli(one_i32, exp1, self.loc))?;
        let qh_div2 = self.push(body, arith::shli(one_i32, exp2, self.loc))?;

        // Scale nibble: half == 0 takes the low nibble, else the high.
        let zero_idx = self.const_index(body, 0)?;
        let half_is_0 = self.push(
            body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, half, zero_idx, self.loc),
        )?;

        let k_lane_off = self.muli(body, lane, eight_idx)?;
        Ok(Iq3sLane {
            grid1_off,
            grid2_off,
            qh_off,
            signs_off,
            scale_off,
            qh_div1,
            qh_div2,
            half_is_0,
            zero_idx,
            k_lane_off,
        })
    }

    /// The lane's per-block decode state for the block `at` names.
    pub(super) fn iq3s_block(
        &mut self,
        kb: &Block<'c>,
        geom: &Iq3sLane<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        tables: &QTables<'_, 'c>,
        at: &BlockAt<'c>,
    ) -> Result<Iq3sBlock<'c>> {
        let f32_t = self.f32_t;
        let g1_byte = self.qbyte(kb, qb, at.j, at.off, geom.grid1_off)?;
        let g2_byte = self.qbyte(kb, qb, at.j, at.off, geom.grid2_off)?;
        let qh_byte = self.qbyte(kb, qb, at.j, at.off, geom.qh_off)?;
        let signs_byte = self.qbyte(kb, qb, at.j, at.off, geom.signs_off)?;
        let scale_byte = self.qbyte(kb, qb, at.j, at.off, geom.scale_off)?;

        let c16 = self.const_i32(kb, 16)?;
        let lo_nibble = self.push(kb, arith::remui(scale_byte, c16, self.loc))?;
        let hi_nibble = self.push(kb, arith::divui(scale_byte, c16, self.loc))?;
        let nibble = self.push(kb, arith::select(geom.half_is_0, lo_nibble, hi_nibble, self.loc))?;
        let c2 = self.const_i32(kb, 2)?;
        let c1 = self.const_i32(kb, 1)?;
        let db_word = self.push(kb, arith::muli(nibble, c2, self.loc))?;
        let db_word = self.push(kb, arith::addi(db_word, c1, self.loc))?;
        let db_f32 = self.numeric_cast(kb, db_word, f32_t)?;
        let d_val = self.push(kb, memref::load(d.mem, &[at.d_row, at.d_col], self.loc))?;
        let d_f32 = self.numeric_cast(kb, d_val, f32_t)?;
        let db = self.push(kb, arith::mulf(d_f32, db_f32, self.loc))?;

        let c256 = self.const_i32(kb, 256)?;
        let four_idx_kb = self.const_index(kb, 4)?;
        let signs_idx = self.numeric_cast(kb, signs_byte, self.index_t)?;
        let eight_idx_kb = self.const_index(kb, 8)?;
        let signs_base = self.push(kb, arith::muli(signs_idx, eight_idx_kb, self.loc))?;

        // The ninth index bit comes from qh, one bit per grid entry.
        let widen = |cg: &mut Self, grid_byte, qh_div| -> Result<Value<'c, 'c>> {
            let bit = cg.push(kb, arith::divui(qh_byte, qh_div, cg.loc))?;
            let bit = cg.push(kb, arith::remui(bit, c2, cg.loc))?;
            let hi = cg.push(kb, arith::muli(bit, c256, cg.loc))?;
            let idx = cg.push(kb, arith::addi(grid_byte, hi, cg.loc))?;
            let idx = cg.numeric_cast(kb, idx, cg.index_t)?;
            cg.push(kb, arith::muli(idx, four_idx_kb, cg.loc))
        };
        let g1_base = widen(self, g1_byte, geom.qh_div1)?;
        let g2_base = widen(self, g2_byte, geom.qh_div2)?;
        // Four bytes a grid entry, eight for the sign entry: three loads.
        let z = self.const_index(kb, 0)?;
        let g_t = Type::vector(&[IQ3S_HALF as u64], self.i8_t);
        let s_t = Type::vector(&[2 * IQ3S_HALF as u64], self.i8_t);
        let g1_v = self.vec_load_al(kb, tables.grid.mem, &[z, g1_base], g_t, 4)?;
        let g2_v = self.vec_load_al(kb, tables.grid.mem, &[z, g2_base], g_t, 4)?;
        let signs_v = self.vec_load_al(kb, tables.signs.mem, &[z, signs_base], s_t, 8)?;
        Ok(Iq3sBlock {
            db,
            g1_v,
            g2_v,
            signs_v,
        })
    }

    /// One decoded weight: element `y` of grid entry `entry` (0 or 1), signed
    /// by the matching half of the lane's sign entry. Both are already in
    /// registers, so this is arithmetic only.
    pub(super) fn iq3s_decoded(
        &mut self,
        kb: &Block<'c>,
        blk: &Iq3sBlock<'c>,
        entry: i64,
        y: i64,
    ) -> Result<Value<'c, 'c>> {
        let (i8_t, f32_t) = (self.i8_t, self.f32_t);
        let g_v = if entry == 0 { blk.g1_v } else { blk.g2_v };
        let mag_val = self.vec_extract(kb, g_v, &[y], i8_t)?;
        let sign_val = self.vec_extract(kb, blk.signs_v, &[entry * IQ3S_HALF + y], i8_t)?;
        let mag_f32 = self.numeric_cast(kb, mag_val, f32_t)?;
        let sign_f32 = self.numeric_cast(kb, sign_val, f32_t)?;
        let decoded = self.push(kb, arith::mulf(blk.db, mag_f32, self.loc))?;
        self.push(kb, arith::mulf(decoded, sign_f32, self.loc))
    }
}
