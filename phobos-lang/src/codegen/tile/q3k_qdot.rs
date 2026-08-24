// Fused Q3_K dot: static-offset decode folded into the contraction. Same
// warp mapping as q2k_qdot.rs, but a six-bit signed scale per run unpacked
// from three interleaved bytes, and a third quant bit from a separate
// hmask plane.

use super::*;

// Q3_K block layout (phobos-gguf/src/quant/q3_k.rs): 110 bytes, hmask at
// byte 0, qs at byte 32, scales at byte 96. No minimum term, unlike Q2_K.
const Q3K_BLOCK_BYTES: i64 = 110;
const Q3K_QS_OFF: i64 = 32;
const Q3K_SCALES_OFF: i64 = 96;
const Q3K_RUN: i64 = 16;
const Q3K_RUNS: i64 = 16;

impl<'c> Codegen<'c> {
    /// Q3_K's matvec contraction with the decode folded in. Same lane
    /// mapping as `tile_q2k_qdot_t`, same overall shape as
    /// `tile_iq1s_qdot_t`.
    pub(in crate::codegen) fn tile_q3k_qdot_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        for (v, what) in [
            (a, "q3k_qdot_t a"),
            (qb, "q3k_qdot_t qb"),
            (d, "q3k_qdot_t d"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if a.elem != self.f32_t || qb.elem != self.i8_t {
            bail!("q3k_qdot_t contracts an f32 activation against Q3_K's raw i8 bytes");
        }
        if d.elem != self.f16_t {
            bail!("q3k_qdot_t's block scale must be f16");
        }
        if self.cta_threads % WARP != 0 {
            bail!("q3k_qdot_t needs a CTA that is a whole number of warps");
        }
        let cols = qb.shape[0];
        if cols == DYN {
            bail!("q3k_qdot_t needs a static output width");
        }
        if a.shape[1] != DYN && a.shape[1] % 256 != 0 {
            bail!("q3k_qdot_t needs a whole number of 256-element blocks");
        }
        self.check_shapes(&[cols], &[d.shape[0]], "q3k_qdot_t d rows")?;

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

        let sixteen_idx = self.const_index(&body, Q3K_RUN)?;
        let eight_idx = self.const_index(&body, 8)?;
        let two_idx = self.const_index(&body, 2)?;
        let four_idx = self.const_index(&body, 4)?;
        let y = self.remui(&body, lane, sixteen_idx)?;
        let lane_half = self.divui(&body, lane, sixteen_idx)?;

        let step = self.const_index(&body, 256)?;
        let zero_k = self.const_index(&body, 0)?;
        let kd = if a.shape[1] == DYN {
            let one = self.const_index(&body, 1)?;
            self.push(&body, memref::dim(a.mem, one, self.loc))?
        } else {
            self.const_index(&body, a.shape[1])?
        };
        let init = self.zero_scalar(&body, f32_t)?;
        let blk_bytes = self.const_index(&body, Q3K_BLOCK_BYTES)?;

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

        let d_val = self.push(&kb, memref::load(d.mem, &[j, blk], self.loc))?;
        let d_f32 = self.numeric_cast(&kb, d_val, f32_t)?;

        let mut partial = carry;
        let zero_idx_kb = self.const_index(&kb, 0)?;
        for iter in 0..(Q3K_RUNS / 2) {
            let iter_c = self.const_index(&kb, iter * 2)?;
            let is = self.addi(&kb, iter_c, lane_half)?;

            let h = self.divui(&kb, is, eight_idx)?;
            let rem = self.remui(&kb, is, eight_idx)?;
            let jj = self.divui(&kb, rem, two_idx)?;
            let half2 = self.remui(&kb, rem, two_idx)?;

            // qs_off = QS_OFF + h*32 + half2*16; qs_div = 4^j.
            let thirtytwo = self.const_index(&kb, 32)?;
            let qs_off = self.addi(
                &kb,
                self.addi(&kb, self.const_index(&kb, Q3K_QS_OFF)?, self.muli(&kb, h, thirtytwo)?)?,
                self.muli(&kb, half2, sixteen_idx)?,
            )?;
            let jj_i32 = self.numeric_cast(&kb, jj, i32_t)?;
            let two_i32 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 2).into(), self.loc))?;
            let one_i32 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 1).into(), self.loc))?;
            let qs_shift = self.push(&kb, arith::muli(jj_i32, two_i32, self.loc))?;
            let qs_div = self.push(&kb, arith::shli(one_i32, qs_shift, self.loc))?;

            // hm_off = half2*16; hm_div = 2^(h*4+j).
            let hm_off = self.muli(&kb, half2, sixteen_idx)?;
            let h_i32 = self.numeric_cast(&kb, h, i32_t)?;
            let four_i32 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 4).into(), self.loc))?;
            let h4 = self.push(&kb, arith::muli(h_i32, four_i32, self.loc))?;
            let hm_shift = self.push(&kb, arith::addi(h4, jj_i32, self.loc))?;
            let hm_div = self.push(&kb, arith::shli(one_i32, hm_shift, self.loc))?;

            // group = is/4, c = is%4.
            let group = self.divui(&kb, is, four_idx)?;
            let c = self.remui(&kb, is, four_idx)?;
            let group_mod2 = self.remui(&kb, group, two_idx)?;
            let zero_idx2 = self.const_index(&kb, 0)?;
            let group_even = self.push(
                &kb,
                arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, group_mod2, zero_idx2, self.loc),
            )?;
            // lo_off = SCALES_OFF + (group even ? c : 4 + c).
            let c_plus4 = self.addi(&kb, c, four_idx)?;
            let lo_extra = self.push(&kb, arith::select(group_even, c, c_plus4, self.loc))?;
            let lo_off = self.addi(&kb, self.const_index(&kb, Q3K_SCALES_OFF)?, lo_extra)?;
            // lo_div = group < 2 ? 1 : 16.
            let two_idx2 = self.const_index(&kb, 2)?;
            let group_lt2 = self.push(
                &kb,
                arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, group, two_idx2, self.loc),
            )?;
            let one_i32b = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 1).into(), self.loc))?;
            let c16 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 16).into(), self.loc))?;
            let lo_div = self.push(&kb, arith::select(group_lt2, one_i32b, c16, self.loc))?;
            // hi_off = SCALES_OFF + 8 + c; hi_div = 4^group.
            let eight_c = self.const_index(&kb, Q3K_SCALES_OFF + 8)?;
            let hi_off = self.addi(&kb, eight_c, c)?;
            let group_i32 = self.numeric_cast(&kb, group, i32_t)?;
            let hi_shift = self.push(&kb, arith::muli(group_i32, two_i32, self.loc))?;
            let hi_div = self.push(&kb, arith::shli(one_i32, hi_shift, self.loc))?;

            let lo_byte = load_u8(self, &kb, lo_off)?;
            let lo_val = self.push(&kb, arith::divui(lo_byte, lo_div, self.loc))?;
            let c4 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 4).into(), self.loc))?;
            let lo_val = self.push(&kb, arith::remui(lo_val, c16, self.loc))?;
            let hi_byte = load_u8(self, &kb, hi_off)?;
            let hi_val = self.push(&kb, arith::divui(hi_byte, hi_div, self.loc))?;
            let hi_val = self.push(&kb, arith::remui(hi_val, c4, self.loc))?;

            let hi_val_shifted = self.push(&kb, arith::muli(hi_val, c16, self.loc))?;
            let scale_word = self.push(&kb, arith::addi(lo_val, hi_val_shifted, self.loc))?;
            let scale_f32 = self.numeric_cast(&kb, scale_word, f32_t)?;
            let c32f = self.push(&kb, arith::constant(self.ctx, FloatAttribute::new(self.ctx, f32_t, 32.0).into(), self.loc))?;
            let scale = self.push(&kb, arith::subf(scale_f32, c32f, self.loc))?;

            let qs_byte_off = self.addi(&kb, qs_off, y)?;
            let qs_byte = load_u8(self, &kb, qs_byte_off)?;
            let low2 = self.push(&kb, arith::divui(qs_byte, qs_div, self.loc))?;
            let low2 = self.push(&kb, arith::remui(low2, c4, self.loc))?;
            let low2_f32 = self.numeric_cast(&kb, low2, f32_t)?;

            let hm_byte_off = self.addi(&kb, hm_off, y)?;
            let hm_byte = load_u8(self, &kb, hm_byte_off)?;
            let bit = self.push(&kb, arith::divui(hm_byte, hm_div, self.loc))?;
            let c2 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 2).into(), self.loc))?;
            let bit = self.push(&kb, arith::remui(bit, c2, self.loc))?;
            let bit_f32 = self.numeric_cast(&kb, bit, f32_t)?;

            let c4f = self.push(&kb, arith::constant(self.ctx, FloatAttribute::new(self.ctx, f32_t, 4.0).into(), self.loc))?;
            let bit4 = self.push(&kb, arith::mulf(bit_f32, c4f, self.loc))?;
            let low2_minus4 = self.push(&kb, arith::subf(low2_f32, c4f, self.loc))?;
            let quant = self.push(&kb, arith::addf(low2_minus4, bit4, self.loc))?;

            let dscale = self.push(&kb, arith::mulf(d_f32, scale, self.loc))?;
            let decoded = self.push(&kb, arith::mulf(dscale, quant, self.loc))?;

            let local_off = self.const_index(&kb, iter * 32)?;
            let a_at = self.addi(&kb, self.addi(&kb, kbase, local_off)?, lane)?;
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

        let zero = self.const_index(&body, 0)?;
        let is_lead = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, zero, self.loc),
        )?;
        let store = Block::new(&[]);
        store.append_operation(memref::store(acc, out.mem, &[zero, j], self.loc));
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
