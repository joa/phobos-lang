// IQ1_M's decode geometry, shared by the contraction (`iq1m_qdot.rs`) and the
// expansion (`qdecode.rs`). Mirrors IQ1_S's grid lookup, but each group of 32
// has its own pair of 3-bit scales and the grid index and sign bit both come
// from one qh byte per lane, so the per-lane offsets need `arith.select`s
// where a source generator would have written per-lane branches.

use super::*;

// IQ1_M block layout (phobos-gguf/src/quant/iq1_m.rs): 56 bytes, qs at byte
// 0, qh at byte 32, scale pairs at byte 48.
pub(super) const IQ1M_BLOCK_BYTES: i64 = 56;
const IQ1M_QH_OFF: i64 = 32;
const IQ1M_SCALES_OFF: i64 = 48;
pub(super) const IQ1M_LANE: i64 = 8;

/// Lane geometry, IQ1_S's with `arith.select`s: the divisors picking this
/// lane's grid index, sign bit and group scale out of one qh byte.
pub(super) struct Iq1mLane<'c> {
    qh_off: Value<'c, 'c>,
    sc_lo_off: Value<'c, 'c>,
    sc_hi_off: Value<'c, 'c>,
    idx_div: Value<'c, 'c>,
    bit_div: Value<'c, 'c>,
    dl_div: Value<'c, 'c>,
    c1_i32: Value<'c, 'c>,
    c8_i32: Value<'c, 'c>,
    pub(super) lane: Value<'c, 'c>,
    pub(super) zero_idx: Value<'c, 'c>,
    pub(super) k_lane_off: Value<'c, 'c>,
}

/// Per-block state: the group scale, the sign offset, and the lane's grid
/// entry as one `vector<8xi8>` (IQ1_S's table, shared outright).
pub(super) struct Iq1mBlock<'c> {
    pub(super) dl: Value<'c, 'c>,
    pub(super) delta: Value<'c, 'c>,
    pub(super) grid_v: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {
    /// The byte offsets warp lane `lane` reads, and its element offset.
    pub(super) fn iq1m_lane(&mut self, body: &Block<'c>, lane: Value<'c, 'c>) -> Result<Iq1mLane<'c>> {
        let i32_t = self.i32_t;
        let four = self.const_index(body, 4)?;
        let two_idx = self.const_index(body, 2)?;
        let zero_idx = self.const_index(body, 0)?;
        let one_idx = self.const_index(body, 1)?;
        let ib = self.divui(body, lane, four)?;
        let l = self.remui(body, lane, four)?;

        // qh_off = QH_OFF + 2*ib + (l >= 2 ? 1 : 0).
        let l_ge_2 = self.push(
            body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Uge, l, two_idx, self.loc),
        )?;
        let qh_extra = self.push(body, arith::select(l_ge_2, one_idx, zero_idx, self.loc))?;
        let qh_off = self.addi(
            body,
            self.addi(body, self.const_index(body, IQ1M_QH_OFF)?, self.muli(body, ib, two_idx)?)?,
            qh_extra,
        )?;

        // idx_div, bit_div = l % 2 == 0 ? (1, 8) : (16, 128).
        let l_mod2 = self.remui(body, l, two_idx)?;
        let l_even = self.push(
            body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, l_mod2, zero_idx, self.loc),
        )?;
        let (c1_i32, c8_i32, c16_i32, c128_i32) = (
            self.const_i32(body, 1)?,
            self.const_i32(body, 8)?,
            self.const_i32(body, 16)?,
            self.const_i32(body, 128)?,
        );
        let idx_div = self.push(body, arith::select(l_even, c1_i32, c16_i32, self.loc))?;
        let bit_div = self.push(body, arith::select(l_even, c8_i32, c128_i32, self.loc))?;

        // sc_lo_off = SCALES_OFF + 2*(ib/2); sc_hi_off = sc_lo_off + 1.
        let ib_half = self.divui(body, ib, two_idx)?;
        let sc_lo_off = self.addi(
            body,
            self.const_index(body, IQ1M_SCALES_OFF)?,
            self.muli(body, ib_half, two_idx)?,
        )?;
        let sc_hi_off = self.addi(body, sc_lo_off, one_idx)?;

        // dl_div = 1 << (6*(ib%2) + (l < 2 ? 0 : 3)).
        let ib_mod2 = self.remui(body, ib, two_idx)?;
        let ib_odd = self.push(
            body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ne, ib_mod2, zero_idx, self.loc),
        )?;
        let six_i32 = self.const_i32(body, 6)?;
        let zero_i32 = self.zero_scalar(body, i32_t)?;
        let shift = self.push(body, arith::select(ib_odd, six_i32, zero_i32, self.loc))?;
        let l_lt_2 = self.push(
            body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, l, two_idx, self.loc),
        )?;
        let three_i32 = self.const_i32(body, 3)?;
        let exp_extra = self.push(body, arith::select(l_lt_2, zero_i32, three_i32, self.loc))?;
        let dl_exp = self.push(body, arith::addi(shift, exp_extra, self.loc))?;
        let dl_div = self.push(body, arith::shli(c1_i32, dl_exp, self.loc))?;

        let k_lane_off = self.muli(body, lane, self.const_index(body, IQ1M_LANE)?)?;
        Ok(Iq1mLane {
            qh_off,
            sc_lo_off,
            sc_hi_off,
            idx_div,
            bit_div,
            dl_div,
            c1_i32,
            c8_i32,
            lane,
            zero_idx,
            k_lane_off,
        })
    }

    /// The lane's per-block decode state for the block `at` names.
    pub(super) fn iq1m_block(
        &mut self,
        kb: &Block<'c>,
        geom: &Iq1mLane<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
        at: &BlockAt<'c>,
    ) -> Result<Iq1mBlock<'c>> {
        let f32_t = self.f32_t;
        let (c1_i32, c8_i32) = (geom.c1_i32, geom.c8_i32);
        // qs4_off = lane exactly (QS_OFF == 0, and 4*(is/4)+is%4 == is).
        let qs4 = self.qbyte(kb, qb, at.j, at.off, geom.lane)?;
        let qh = self.qbyte(kb, qb, at.j, at.off, geom.qh_off)?;
        let sc_lo = self.qbyte(kb, qb, at.j, at.off, geom.sc_lo_off)?;
        let sc_hi = self.qbyte(kb, qb, at.j, at.off, geom.sc_hi_off)?;

        let c256_i32 = self.const_i32(kb, 256)?;
        let word = self.push(kb, arith::muli(sc_hi, c256_i32, self.loc))?;
        let word = self.push(kb, arith::addi(sc_lo, word, self.loc))?;

        let c2_i32 = self.const_i32(kb, 2)?;
        let dl_word = self.push(kb, arith::divui(word, geom.dl_div, self.loc))?;
        let dl_word = self.push(kb, arith::remui(dl_word, c8_i32, self.loc))?;
        let dl_word = self.push(kb, arith::muli(dl_word, c2_i32, self.loc))?;
        let dl_word = self.push(kb, arith::addi(dl_word, c1_i32, self.loc))?;
        let dl_f32 = self.numeric_cast(kb, dl_word, f32_t)?;
        let d_val = self.push(kb, memref::load(d.mem, &[at.d_row, at.d_col], self.loc))?;
        let d_f32 = self.numeric_cast(kb, d_val, f32_t)?;
        let dl = self.push(kb, arith::mulf(d_f32, dl_f32, self.loc))?;

        let idx_lo = self.push(kb, arith::divui(qh, geom.idx_div, self.loc))?;
        let idx_lo = self.push(kb, arith::remui(idx_lo, c8_i32, self.loc))?;
        let idx_lo = self.push(kb, arith::muli(idx_lo, c256_i32, self.loc))?;
        let idx = self.push(kb, arith::addi(qs4, idx_lo, self.loc))?;
        let idx = self.numeric_cast(kb, idx, self.index_t)?;
        let eight_idx = self.const_index(kb, 8)?;
        let idx8 = self.push(kb, arith::muli(idx, eight_idx, self.loc))?;

        let bit = self.push(kb, arith::divui(qh, geom.bit_div, self.loc))?;
        let bit = self.push(kb, arith::remui(bit, c2_i32, self.loc))?;
        let bit_f32 = self.numeric_cast(kb, bit, f32_t)?;
        let c0125 = self.const_f32(kb, 0.125)?;
        let c025 = self.const_f32(kb, 0.25)?;
        let bit_term = self.push(kb, arith::mulf(c025, bit_f32, self.loc))?;
        let delta = self.push(kb, arith::subf(c0125, bit_term, self.loc))?;
        // Eight bytes at an eight-byte-aligned offset: one `ld.global.v2.b32`.
        let grid_t = Type::vector(&[IQ1M_LANE as u64], self.i8_t);
        let grid_v = self.vec_load_al(kb, grid.mem, &[geom.zero_idx, idx8], grid_t, 8)?;
        Ok(Iq1mBlock { dl, delta, grid_v })
    }

    /// The lane's `y`-th decoded weight, from the grid entry already in
    /// registers. The grid value is signed, so it widens sign-extending.
    pub(super) fn iq1m_decoded(
        &mut self,
        kb: &Block<'c>,
        blk: &Iq1mBlock<'c>,
        y: i64,
    ) -> Result<Value<'c, 'c>> {
        let (i8_t, f32_t) = (self.i8_t, self.f32_t);
        let grid_val = self.vec_extract(kb, blk.grid_v, &[y], i8_t)?;
        let grid_f32 = self.numeric_cast(kb, grid_val, f32_t)?;
        let plus_delta = self.push(kb, arith::addf(grid_f32, blk.delta, self.loc))?;
        self.push(kb, arith::mulf(blk.dl, plus_delta, self.loc))
    }
}
