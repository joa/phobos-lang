// IQ1_S's decode geometry, shared by the contraction (`iq1s_qdot.rs`) and the
// expansion (`qdecode.rs`). Both walk the format the same way: a thread
// owns one of the 32 format lanes of a 256-element block and decodes its eight
// elements from a 9-bit grid index and a 3-bit group scale.

use super::*;

// IQ1_S block layout (phobos-gguf/src/quant/iq1_s.rs): 50 bytes, qs at byte
// 2, qh at byte 34, 32 lanes of 8 elements.
pub(super) const IQ1S_BLOCK_BYTES: i64 = 48;
const IQ1S_QS_OFF: i64 = 0;
const IQ1S_QH_OFF: i64 = 32;
pub(super) const IQ1S_LANE: i64 = 8;
const IQ1S_DELTA: f32 = 0.125;

/// Lane geometry: byte offsets within a block, and where the lane's eight
/// elements sit in the 256-element k-block. Emitted once, outside the k loop.
pub(super) struct Iq1sLane<'c> {
    qs_off: Value<'c, 'c>,
    qh_lo_off: Value<'c, 'c>,
    qh_hi_off: Value<'c, 'c>,
    shift: Value<'c, 'c>,
    pub(super) k_lane_off: Value<'c, 'c>,
}

/// Per-block state: the group scale, the offset every grid entry takes, and
/// the lane's whole grid entry as one `vector<8xi8>`.
pub(super) struct Iq1sBlock<'c> {
    pub(super) dl: Value<'c, 'c>,
    pub(super) delta: Value<'c, 'c>,
    pub(super) grid_v: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {
    /// The byte offsets warp lane `lane` reads, and its element offset.
    pub(super) fn iq1s_lane(
        &mut self,
        body: &Block<'c>,
        lane: Value<'c, 'c>,
    ) -> Result<Iq1sLane<'c>> {
        let i32_t = self.i32_t;
        let four = self.const_index(body, 4)?;
        let three = self.const_index(body, 3)?;
        let two = self.const_index(body, 2)?;
        let one = self.const_index(body, 1)?;
        let ib = self.divui(body, lane, four)?;
        let l = self.remui(body, lane, four)?;
        let qs_off = self.addi(body, self.const_index(body, IQ1S_QS_OFF)?, lane)?;
        let qh_lo_off = self.addi(
            body,
            self.const_index(body, IQ1S_QH_OFF)?,
            self.muli(body, ib, two)?,
        )?;
        let qh_hi_off = self.addi(body, qh_lo_off, one)?;
        let shift = self.muli(body, l, three)?;
        let shift = self.numeric_cast(body, shift, i32_t)?;
        let k_lane_off = self.muli(body, lane, self.const_index(body, IQ1S_LANE)?)?;
        Ok(Iq1sLane {
            qs_off,
            qh_lo_off,
            qh_hi_off,
            shift,
            k_lane_off,
        })
    }

    /// The lane's per-block decode state for the block `at` names.
    pub(super) fn iq1s_block(
        &mut self,
        kb: &Block<'c>,
        geom: &Iq1sLane<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
        at: &BlockAt<'c>,
    ) -> Result<Iq1sBlock<'c>> {
        let (i32_t, f32_t) = (self.i32_t, self.f32_t);
        let qs = self.qbyte(kb, qb, at.j, at.off, geom.qs_off)?;
        let qh_lo = self.qbyte(kb, qb, at.j, at.off, geom.qh_lo_off)?;
        let qh_hi = self.qbyte(kb, qb, at.j, at.off, geom.qh_hi_off)?;
        let c256 = self.const_index(kb, 256)?;
        let c256_i32 = self.numeric_cast(kb, c256, i32_t)?;
        let qh_hi_shifted = self.push(kb, arith::muli(qh_hi, c256_i32, self.loc))?;
        let qh = self.push(kb, arith::addi(qh_lo, qh_hi_shifted, self.loc))?;

        let c8 = self.const_index(kb, 8)?;
        let c8_i32 = self.numeric_cast(kb, c8, i32_t)?;
        let c4096 = self.const_i32(kb, 4096)?;
        let c32768 = self.const_i32(kb, 32768)?;
        let c2 = self.const_i32(kb, 2)?;
        let c1 = self.const_i32(kb, 1)?;

        let d_val = self.push(kb, memref::load(d.mem, &[at.d_row, at.d_col], self.loc))?;
        let d_f32 = self.numeric_cast(kb, d_val, f32_t)?;
        let sc = self.push(kb, arith::divui(qh, c4096, self.loc))?;
        let sc = self.push(kb, arith::remui(sc, c8_i32, self.loc))?;
        let sc = self.push(kb, arith::muli(sc, c2, self.loc))?;
        let sc = self.push(kb, arith::addi(sc, c1, self.loc))?;
        let sc_f32 = self.numeric_cast(kb, sc, f32_t)?;
        let dl = self.push(kb, arith::mulf(d_f32, sc_f32, self.loc))?;

        let sign_bit = self.push(kb, arith::divui(qh, c32768, self.loc))?;
        let sign_bit = self.push(kb, arith::remui(sign_bit, c2, self.loc))?;
        let sign_f32 = self.numeric_cast(kb, sign_bit, f32_t)?;
        let delta_c = self.const_f32(kb, f64::from(IQ1S_DELTA))?;
        let twice_delta_c = self.const_f32(kb, f64::from(2.0 * IQ1S_DELTA))?;
        let sign_term = self.push(kb, arith::mulf(twice_delta_c, sign_f32, self.loc))?;
        let delta = self.push(kb, arith::subf(delta_c, sign_term, self.loc))?;

        let shift_div = self.push(kb, arith::shrui(qh, geom.shift, self.loc))?;
        let shift_div = self.push(kb, arith::remui(shift_div, c8_i32, self.loc))?;
        let shift_term = self.push(kb, arith::muli(shift_div, c256_i32, self.loc))?;
        let base_idx = self.push(kb, arith::addi(qs, shift_term, self.loc))?;
        let base_idx = self.numeric_cast(kb, base_idx, self.index_t)?;
        let base_idx8 = self.push(kb, arith::muli(base_idx, c8, self.loc))?;
        // Eight bytes at an eight-byte-aligned offset: one `ld.global.b64`.
        let zero = self.const_index(kb, 0)?;
        let grid_t = Type::vector(&[IQ1S_LANE as u64], self.i8_t);
        let grid_v = self.vec_load_al(kb, grid.mem, &[zero, base_idx8], grid_t, 8)?;
        Ok(Iq1sBlock { dl, delta, grid_v })
    }

    /// The lane's `y`-th decoded weight, taken from the grid entry already in
    /// registers. The grid value is signed, so it widens with `extsi`, not the
    /// zero-extend the unsigned block fields take.
    pub(super) fn iq1s_decoded(
        &mut self,
        kb: &Block<'c>,
        blk: &Iq1sBlock<'c>,
        y: i64,
    ) -> Result<Value<'c, 'c>> {
        let (i8_t, f32_t) = (self.i8_t, self.f32_t);
        let grid_val = self.vec_extract(kb, blk.grid_v, &[y], i8_t)?;
        let grid_f32 = self.numeric_cast(kb, grid_val, f32_t)?;
        let plus_delta = self.push(kb, arith::addf(grid_f32, blk.delta, self.loc))?;
        self.push(kb, arith::mulf(blk.dl, plus_delta, self.loc))
    }
}
