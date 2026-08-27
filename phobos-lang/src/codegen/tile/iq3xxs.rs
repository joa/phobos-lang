// IQ3_XXS's decode geometry, shared by the contraction (`iq3xxs_qdot.rs`) and
// the expansion (`qdecode.rs`). Two four-wide grid lookups a lane (grid
// entries are 4 bytes wide here, against IQ2_XXS's one eight-wide lookup) plus
// a sign-table lookup. The scale/sign geometry is IQ2_XXS's outright: the
// aux32 field is byte-for-byte the same.

use super::*;

// IQ3_XXS block layout (phobos-gguf/src/quant/iq3_xxs.rs): 98 bytes, qs at
// byte 2 (two grid-index bytes a lane), aux32 scale/sign field at byte 66.
pub(super) const IQ3XXS_BLOCK_BYTES: i64 = 96;
const IQ3XXS_QS_OFF: i64 = 0;
const IQ3XXS_SS_OFF: i64 = 64;
/// Elements a lane takes from one of its two grid entries.
pub(super) const IQ3XXS_HALF: i64 = 4;

/// Elements a lane owns: both halves of its grid entry.
pub(super) const IQ3XXS_LANE: i64 = 2 * IQ3XXS_HALF;

/// Lane geometry: byte offsets within a block, and the lane's element offset.
/// Scale and sign fields are IQ2_XXS's, byte for byte.
pub(super) struct Iq3xxsLane<'c> {
    l: Value<'c, 'c>,
    g1_off: Value<'c, 'c>,
    g2_off: Value<'c, 'c>,
    scale_off: Value<'c, 'c>,
    lo_off: Value<'c, 'c>,
    hi_off: Value<'c, 'c>,
    eight_idx: Value<'c, 'c>,
    pub(super) zero_idx: Value<'c, 'c>,
    pub(super) k_lane_off: Value<'c, 'c>,
}

/// Per-block state: the scaled magnitude, two four-wide grid entries and one
/// eight-wide sign entry, each one vector load.
pub(super) struct Iq3xxsBlock<'c> {
    pub(super) db: Value<'c, 'c>,
    pub(super) g1_v: Value<'c, 'c>,
    pub(super) g2_v: Value<'c, 'c>,
    pub(super) signs_v: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {
    /// The byte offsets warp lane `lane` reads, and its element offset.
    /// `hi_off` folds to a re-read of `lo_off` at `l == 0`, as in
    /// `iq2xxs_lane`.
    pub(super) fn iq3xxs_lane(
        &mut self,
        body: &Block<'c>,
        lane: Value<'c, 'c>,
    ) -> Result<Iq3xxsLane<'c>> {
        let four = self.const_index(body, 4)?;
        let eight_idx = self.const_index(body, 8)?;
        let zero_idx = self.const_index(body, 0)?;
        let one_idx = self.const_index(body, 1)?;
        let two_idx = self.const_index(body, 2)?;
        let ib32 = self.divui(body, lane, four)?;
        let l = self.remui(body, lane, four)?;
        let is_l0 = self.push(
            body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, l, zero_idx, self.loc),
        )?;

        // base = QS_OFF + 8*ib32 + 2*l; g1_off = base, g2_off = base + 1.
        let base = self.addi(
            body,
            self.addi(body, self.const_index(body, IQ3XXS_QS_OFF)?, self.muli(body, ib32, eight_idx)?)?,
            self.muli(body, l, two_idx)?,
        )?;
        let g1_off = base;
        let g2_off = self.addi(body, base, one_idx)?;

        // aux = SS_OFF + 4*ib32; scale_off = aux + 3.
        let aux = self.addi(body, self.const_index(body, IQ3XXS_SS_OFF)?, self.muli(body, ib32, four)?)?;
        let three_idx = self.const_index(body, 3)?;
        let scale_off = self.addi(body, aux, three_idx)?;

        // Same lo/hi/shift-div folding as iq2xxs_lane: lo_off = aux +
        // (max(l, 1) - 1), hi_off = aux + l (a harmless re-read at l == 0).
        let l_or_1 = self.push(body, arith::select(is_l0, one_idx, l, self.loc))?;
        let lo_off = self.addi(body, aux, self.subi(body, l_or_1, one_idx)?)?;
        let hi_off = self.addi(body, aux, l)?;

        let k_lane_off = self.muli(body, lane, eight_idx)?;
        Ok(Iq3xxsLane {
            l,
            g1_off,
            g2_off,
            scale_off,
            lo_off,
            hi_off,
            eight_idx,
            zero_idx,
            k_lane_off,
        })
    }

    /// The lane's per-block decode state for the block `at` names.
    pub(super) fn iq3xxs_block(
        &mut self,
        kb: &Block<'c>,
        geom: &Iq3xxsLane<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        tables: &QTables<'_, 'c>,
        at: &BlockAt<'c>,
    ) -> Result<Iq3xxsBlock<'c>> {
        let (i32_t, f32_t) = (self.i32_t, self.f32_t);
        let g1_byte = self.qbyte(kb, qb, at.j, at.off, geom.g1_off)?;
        let g2_byte = self.qbyte(kb, qb, at.j, at.off, geom.g2_off)?;
        let scale_byte = self.qbyte(kb, qb, at.j, at.off, geom.scale_off)?;
        let lo = self.qbyte(kb, qb, at.j, at.off, geom.lo_off)?;
        let hi = self.qbyte(kb, qb, at.j, at.off, geom.hi_off)?;

        let is_l0_kb = self.push(
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
        let hi_term = self.push(kb, arith::select(is_l0_kb, zero_i32, hi, self.loc))?;
        let c256 = self.const_i32(kb, 256)?;
        let hi_shifted = self.push(kb, arith::muli(hi_term, c256, self.loc))?;
        let word = self.push(kb, arith::addi(lo, hi_shifted, self.loc))?;

        let eight_i32 = self.numeric_cast(kb, self.const_index(kb, 8)?, i32_t)?;
        let l_i32 = self.numeric_cast(kb, geom.l, i32_t)?;
        let exp = self.push(kb, arith::subi(eight_i32, l_i32, self.loc))?;
        let zero_exp = self.zero_scalar(kb, i32_t)?;
        let exp = self.push(kb, arith::select(is_l0_kb, zero_exp, exp, self.loc))?;
        let one_i32 = self.const_i32(kb, 1)?;
        let shift_div = self.push(kb, arith::shli(one_i32, exp, self.loc))?;

        let c128 = self.const_i32(kb, 128)?;
        let signs_idx = self.push(kb, arith::divui(word, shift_div, self.loc))?;
        let signs_idx = self.push(kb, arith::remui(signs_idx, c128, self.loc))?;
        let signs_idx = self.numeric_cast(kb, signs_idx, self.index_t)?;
        let signs_base = self.push(kb, arith::muli(signs_idx, geom.eight_idx, self.loc))?;

        let c16 = self.const_i32(kb, 16)?;
        let scale_shifted = self.push(kb, arith::divui(scale_byte, c16, self.loc))?;
        let scale_f32 = self.numeric_cast(kb, scale_shifted, f32_t)?;
        let half_c = self.const_f32(kb, 0.5)?;
        let d_val = self.push(kb, memref::load(d.mem, &[at.j, at.blk], self.loc))?;
        let d_f32 = self.numeric_cast(kb, d_val, f32_t)?;
        let sc = self.push(kb, arith::addf(half_c, scale_f32, self.loc))?;
        let sc = self.push(kb, arith::mulf(sc, half_c, self.loc))?;
        let db = self.push(kb, arith::mulf(d_f32, sc, self.loc))?;

        let four_idx = self.const_index(kb, 4)?;
        let g1_idx = self.numeric_cast(kb, g1_byte, self.index_t)?;
        let g2_idx = self.numeric_cast(kb, g2_byte, self.index_t)?;
        let g1_base = self.push(kb, arith::muli(g1_idx, four_idx, self.loc))?;
        let g2_base = self.push(kb, arith::muli(g2_idx, four_idx, self.loc))?;
        // Four bytes a grid entry, eight for the sign entry: three loads.
        let z = self.const_index(kb, 0)?;
        let g_t = Type::vector(&[IQ3XXS_HALF as u64], self.i8_t);
        let s_t = Type::vector(&[2 * IQ3XXS_HALF as u64], self.i8_t);
        let g1_v = self.vec_load_al(kb, tables.grid.mem, &[z, g1_base], g_t, 4)?;
        let g2_v = self.vec_load_al(kb, tables.grid.mem, &[z, g2_base], g_t, 4)?;
        let signs_v = self.vec_load_al(kb, tables.signs.mem, &[z, signs_base], s_t, 8)?;
        Ok(Iq3xxsBlock {
            db,
            g1_v,
            g2_v,
            signs_v,
        })
    }

    /// One decoded weight: element `y` of grid entry `entry` (0 or 1), signed
    /// by the matching half of the lane's sign entry. Both are already in
    /// registers, so this is arithmetic only.
    pub(super) fn iq3xxs_decoded(
        &mut self,
        kb: &Block<'c>,
        blk: &Iq3xxsBlock<'c>,
        entry: i64,
        y: i64,
    ) -> Result<Value<'c, 'c>> {
        let (i8_t, f32_t) = (self.i8_t, self.f32_t);
        let g_v = if entry == 0 { blk.g1_v } else { blk.g2_v };
        let mag_val = self.vec_extract(kb, g_v, &[y], i8_t)?;
        let sign_val = self.vec_extract(kb, blk.signs_v, &[entry * IQ3XXS_HALF + y], i8_t)?;
        let mag_f32 = self.numeric_cast(kb, mag_val, f32_t)?;
        let sign_f32 = self.numeric_cast(kb, sign_val, f32_t)?;
        let decoded = self.push(kb, arith::mulf(blk.db, mag_f32, self.loc))?;
        self.push(kb, arith::mulf(decoded, sign_f32, self.loc))
    }
}
