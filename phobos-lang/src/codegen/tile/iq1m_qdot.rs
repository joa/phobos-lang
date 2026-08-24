// Fused IQ1_M dot: grid-table decode folded into the contraction. Mirrors
// iq1s_qdot.rs's grid lookup, but each group of 32 has its own pair of 3-bit
// scales and the grid index/sign bit come from one qh byte per lane.

use super::*;

// IQ1_M block layout (phobos-gguf/src/quant/iq1_m.rs): 56 bytes, qs at byte
// 0, qh at byte 32, scale pairs at byte 48.
const IQ1M_BLOCK_BYTES: i64 = 56;
const IQ1M_QH_OFF: i64 = 32;
const IQ1M_SCALES_OFF: i64 = 48;
const IQ1M_LANE: i64 = 8;

impl<'c> Codegen<'c> {
    /// IQ1_M's matvec contraction with the decode folded in. Same shape as
    /// `tile_iq1s_qdot_t` (warp owns an output, register accumulator, one
    /// closing shuffle), but the per-lane offsets need `arith.select`s
    /// instead of a source generator's per-lane branches.
    pub(in crate::codegen) fn tile_iq1m_qdot_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        for (v, what) in [
            (a, "iq1m_qdot_t a"),
            (qb, "iq1m_qdot_t qb"),
            (d, "iq1m_qdot_t d"),
            (grid, "iq1m_qdot_t grid"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if a.elem != self.f32_t || qb.elem != self.i8_t {
            bail!("iq1m_qdot_t contracts an f32 activation against IQ1_M's raw i8 bytes");
        }
        if d.elem != self.f16_t {
            bail!("iq1m_qdot_t's block scale must be f16");
        }
        if grid.elem != self.i32_t {
            bail!("iq1m_qdot_t's grid table must hold i32 lanes");
        }
        if self.cta_threads % WARP != 0 {
            bail!("iq1m_qdot_t needs a CTA that is a whole number of warps");
        }
        let cols = qb.shape[0];
        if cols == DYN {
            bail!("iq1m_qdot_t needs a static output width");
        }
        if a.shape[1] != DYN && a.shape[1] % 256 != 0 {
            bail!("iq1m_qdot_t needs a whole number of 256-element blocks");
        }
        self.check_shapes(&[cols], &[d.shape[0]], "iq1m_qdot_t d rows")?;

        let out = self.alloc_tile_shaped(block, self.f32_t, &[1, cols])?;
        let (i32_t, f32_t) = (self.i32_t, self.f32_t);

        let warp_w = self.const_index(block, WARP)?;
        let total = self.const_index(block, cols * WARP)?;
        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;

        let body = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body.argument(0)?.into());
        let j = self.divui(&body, li, warp_w)?;
        let lane = self.remui(&body, li, warp_w)?;

        let four = self.const_index(&body, 4)?;
        let two_idx = self.const_index(&body, 2)?;
        let zero_idx = self.const_index(&body, 0)?;
        let one_idx = self.const_index(&body, 1)?;
        let ib = self.divui(&body, lane, four)?;
        let l = self.remui(&body, lane, four)?;

        // qh_off = QH_OFF + 2*ib + (l >= 2 ? 1 : 0).
        let l_ge_2 = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Uge, l, two_idx, self.loc),
        )?;
        let qh_extra = self.push(&body, arith::select(l_ge_2, one_idx, zero_idx, self.loc))?;
        let qh_off = self.addi(
            &body,
            self.addi(&body, self.const_index(&body, IQ1M_QH_OFF)?, self.muli(&body, ib, two_idx)?)?,
            qh_extra,
        )?;

        // idx_div, bit_div = l % 2 == 0 ? (1, 8) : (16, 128).
        let l_mod2 = self.remui(&body, l, two_idx)?;
        let l_even = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, l_mod2, zero_idx, self.loc),
        )?;
        let (c1_i32, c8_i32, c16_i32, c128_i32) = (
            self.push(&body, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 1).into(), self.loc))?,
            self.push(&body, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 8).into(), self.loc))?,
            self.push(&body, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 16).into(), self.loc))?,
            self.push(&body, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 128).into(), self.loc))?,
        );
        let idx_div = self.push(&body, arith::select(l_even, c1_i32, c16_i32, self.loc))?;
        let bit_div = self.push(&body, arith::select(l_even, c8_i32, c128_i32, self.loc))?;

        // sc_lo_off = SCALES_OFF + 2*(ib/2); sc_hi_off = sc_lo_off + 1.
        let ib_half = self.divui(&body, ib, two_idx)?;
        let sc_lo_off = self.addi(
            &body,
            self.const_index(&body, IQ1M_SCALES_OFF)?,
            self.muli(&body, ib_half, two_idx)?,
        )?;
        let sc_hi_off = self.addi(&body, sc_lo_off, one_idx)?;

        // dl_div = 1 << (6*(ib%2) + (l < 2 ? 0 : 3)).
        let ib_mod2 = self.remui(&body, ib, two_idx)?;
        let ib_odd = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ne, ib_mod2, zero_idx, self.loc),
        )?;
        let six_i32 = self.push(&body, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 6).into(), self.loc))?;
        let zero_i32 = self.zero_scalar(&body, i32_t)?;
        let shift = self.push(&body, arith::select(ib_odd, six_i32, zero_i32, self.loc))?;
        let l_lt_2 = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, l, two_idx, self.loc),
        )?;
        let three_i32 = self.push(&body, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 3).into(), self.loc))?;
        let exp_extra = self.push(&body, arith::select(l_lt_2, zero_i32, three_i32, self.loc))?;
        let dl_exp = self.push(&body, arith::addi(shift, exp_extra, self.loc))?;
        let dl_div = self.push(&body, arith::shli(c1_i32, dl_exp, self.loc))?;

        let k_lane_off = self.muli(&body, lane, self.const_index(&body, IQ1M_LANE)?)?;

        let step = self.const_index(&body, 256)?;
        let zero_k = self.const_index(&body, 0)?;
        let kd = if a.shape[1] == DYN {
            let one = self.const_index(&body, 1)?;
            self.push(&body, memref::dim(a.mem, one, self.loc))?
        } else {
            self.const_index(&body, a.shape[1])?
        };
        let init = self.zero_scalar(&body, f32_t)?;
        let blk_bytes = self.const_index(&body, IQ1M_BLOCK_BYTES)?;

        let kb = Block::new(&[(self.index_t, self.loc), (f32_t, self.loc)]);
        let kbase = detach(kb.argument(0)?.into());
        let carry = detach(kb.argument(1)?.into());
        let blk = self.divui(&kb, kbase, step)?;
        let blk_off = self.muli(&kb, blk, blk_bytes)?;

        let load_u8 = |cg: &mut Self, at: &Block<'c>, off: Value<'c, 'c>| -> Result<Value<'c, 'c>> {
            let byte_off = cg.addi(at, blk_off, off)?;
            let byte = cg.push(at, memref::load(qb.mem, &[j, byte_off], cg.loc))?;
            cg.push(
                at,
                OperationBuilder::new("arith.extui", cg.loc)
                    .add_operands(&[byte])
                    .add_results(&[i32_t])
                    .build()?,
            )
        };
        // qs4_off = lane exactly (QS_OFF == 0, and 4*(is/4)+is%4 == is).
        let qs4 = load_u8(self, &kb, lane)?;
        let qh = load_u8(self, &kb, qh_off)?;
        let sc_lo = load_u8(self, &kb, sc_lo_off)?;
        let sc_hi = load_u8(self, &kb, sc_hi_off)?;

        let c256_i32 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 256).into(), self.loc))?;
        let word = self.push(&kb, arith::muli(sc_hi, c256_i32, self.loc))?;
        let word = self.push(&kb, arith::addi(sc_lo, word, self.loc))?;

        let c2_i32 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 2).into(), self.loc))?;
        let dl_word = self.push(&kb, arith::divui(word, dl_div, self.loc))?;
        let dl_word = self.push(&kb, arith::remui(dl_word, c8_i32, self.loc))?;
        let dl_word = self.push(&kb, arith::muli(dl_word, c2_i32, self.loc))?;
        let dl_word = self.push(&kb, arith::addi(dl_word, c1_i32, self.loc))?;
        let dl_f32 = self.numeric_cast(&kb, dl_word, f32_t)?;
        let d_val = self.push(&kb, memref::load(d.mem, &[j, blk], self.loc))?;
        let d_f32 = self.numeric_cast(&kb, d_val, f32_t)?;
        let dl = self.push(&kb, arith::mulf(d_f32, dl_f32, self.loc))?;

        let idx_lo = self.push(&kb, arith::divui(qh, idx_div, self.loc))?;
        let idx_lo = self.push(&kb, arith::remui(idx_lo, c8_i32, self.loc))?;
        let idx_lo = self.push(&kb, arith::muli(idx_lo, c256_i32, self.loc))?;
        let idx = self.push(&kb, arith::addi(qs4, idx_lo, self.loc))?;
        let idx = self.numeric_cast(&kb, idx, self.index_t)?;
        let eight_idx = self.const_index(&kb, 8)?;
        let idx8 = self.push(&kb, arith::muli(idx, eight_idx, self.loc))?;

        let bit = self.push(&kb, arith::divui(qh, bit_div, self.loc))?;
        let bit = self.push(&kb, arith::remui(bit, c2_i32, self.loc))?;
        let bit_f32 = self.numeric_cast(&kb, bit, f32_t)?;
        let c0125 = self.push(&kb, arith::constant(self.ctx, FloatAttribute::new(self.ctx, f32_t, 0.125).into(), self.loc))?;
        let c025 = self.push(&kb, arith::constant(self.ctx, FloatAttribute::new(self.ctx, f32_t, 0.25).into(), self.loc))?;
        let bit_term = self.push(&kb, arith::mulf(c025, bit_f32, self.loc))?;
        let delta = self.push(&kb, arith::subf(c0125, bit_term, self.loc))?;

        let mut partial = carry;
        let k_off = self.addi(&kb, kbase, k_lane_off)?;
        let zero_idx_kb = self.const_index(&kb, 0)?;
        for y in 0..IQ1M_LANE {
            let yc = self.const_index(&kb, y)?;
            let grid_at = self.addi(&kb, idx8, yc)?;
            let grid_val = self.push(&kb, memref::load(grid.mem, &[zero_idx_kb, grid_at], self.loc))?;
            let grid_f32 = self.numeric_cast(&kb, grid_val, f32_t)?;
            let plus_delta = self.push(&kb, arith::addf(grid_f32, delta, self.loc))?;
            let decoded = self.push(&kb, arith::mulf(dl, plus_delta, self.loc))?;
            let a_at = self.addi(&kb, k_off, yc)?;
            let a_val = self.push(&kb, memref::load(a.mem, &[zero_idx_kb, a_at], self.loc))?;
            let prod = self.push(&kb, arith::mulf(decoded, a_val, self.loc))?;
            partial = self.push(&kb, arith::addf(partial, prod, self.loc))?;
        }
        kb.append_operation(scf::r#yield(&[partial], self.loc));

        let kr = Region::new();
        kr.append_block(kb);
        let mut acc = self.push(
            &body,
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&[zero_k, kd, step, init])
                .add_results(&[f32_t])
                .add_regions([kr])
                .build()?,
        )?;

        let mut mask = WARP / 2;
        while mask >= 1 {
            let other = self.shfl_xor_f32(&body, acc, mask)?;
            acc = self.push(&body, arith::addf(acc, other, self.loc))?;
            mask /= 2;
        }

        let is_lead = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, zero_idx, self.loc),
        )?;
        let store = Block::new(&[]);
        store.append_operation(memref::store(acc, out.mem, &[zero_idx, j], self.loc));
        store.append_operation(scf::r#yield(&[], self.loc));
        let sr = Region::new();
        sr.append_block(store);
        body.append_operation(scf::r#if(is_lead, &[], sr, Region::new(), self.loc));
        body.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));
        self.barrier(block)?;
        Ok(out)
    }
}
