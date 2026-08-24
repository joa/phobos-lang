// Fused Q2_K dot: static-offset decode folded into the contraction. Unlike
// the grid-coded IQ formats, Q2_K needs no table lookup. A run is 16
// elements (half a warp), so this maps two runs onto one warp per pass.

use super::*;

// Q2_K block layout (phobos-gguf/src/quant/q2_k.rs): 84 bytes, scale/min
// byte plane at byte 0 (one byte a run), two-bit quant plane at byte 16.
const Q2K_BLOCK_BYTES: i64 = 84;
const Q2K_QS_OFF: i64 = 16;
const Q2K_RUN: i64 = 16;
const Q2K_RUNS: i64 = 16;

impl<'c> Codegen<'c> {
    /// Q2_K's matvec contraction with the decode folded in. Same shape as
    /// `tile_iq1s_qdot_t`; differs only in the lane mapping.
    pub(in crate::codegen) fn tile_q2k_qdot_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        dmin: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        for (v, what) in [
            (a, "q2k_qdot_t a"),
            (qb, "q2k_qdot_t qb"),
            (d, "q2k_qdot_t d"),
            (dmin, "q2k_qdot_t dmin"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if a.elem != self.f32_t || qb.elem != self.i8_t {
            bail!("q2k_qdot_t contracts an f32 activation against Q2_K's raw i8 bytes");
        }
        if d.elem != self.f16_t || dmin.elem != self.f16_t {
            bail!("q2k_qdot_t's block scale and minimum must be f16");
        }
        if self.cta_threads % WARP != 0 {
            bail!("q2k_qdot_t needs a CTA that is a whole number of warps");
        }
        let cols = qb.shape[0];
        if cols == DYN {
            bail!("q2k_qdot_t needs a static output width");
        }
        if a.shape[1] != DYN && a.shape[1] % 256 != 0 {
            bail!("q2k_qdot_t needs a whole number of 256-element blocks");
        }
        self.check_shapes(&[cols], &[d.shape[0]], "q2k_qdot_t d rows")?;
        self.check_shapes(&[cols], &[dmin.shape[0]], "q2k_qdot_t dmin rows")?;

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

        let sixteen_idx = self.const_index(&body, Q2K_RUN)?;
        let eight_idx = self.const_index(&body, 8)?;
        let two_idx = self.const_index(&body, 2)?;
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
        let blk_bytes = self.const_index(&body, Q2K_BLOCK_BYTES)?;

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
        let dmin_val = self.push(&kb, memref::load(dmin.mem, &[j, blk], self.loc))?;
        let dmin_f32 = self.numeric_cast(&kb, dmin_val, f32_t)?;

        let mut partial = carry;
        let zero_idx_kb = self.const_index(&kb, 0)?;
        for iter in 0..(Q2K_RUNS / 2) {
            let iter_c = self.const_index(&kb, iter * 2)?;
            let is = self.addi(&kb, iter_c, lane_half)?;

            let h = self.divui(&kb, is, eight_idx)?;
            let rem = self.remui(&kb, is, eight_idx)?;
            let jj = self.divui(&kb, rem, two_idx)?;
            let half2 = self.remui(&kb, rem, two_idx)?;

            // qs_off = QS_OFF + h*32 + half2*16.
            let thirtytwo = self.const_index(&kb, 32)?;
            let qs_off = self.addi(
                &kb,
                self.addi(&kb, self.const_index(&kb, Q2K_QS_OFF)?, self.muli(&kb, h, thirtytwo)?)?,
                self.muli(&kb, half2, sixteen_idx)?,
            )?;
            // shift_div = 4^j = 1 << (2*j).
            let jj_i32 = self.numeric_cast(&kb, jj, i32_t)?;
            let two_i32 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 2).into(), self.loc))?;
            let shift = self.push(&kb, arith::muli(jj_i32, two_i32, self.loc))?;
            let one_i32 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 1).into(), self.loc))?;
            let shift_div = self.push(&kb, arith::shli(one_i32, shift, self.loc))?;

            let sc_byte = load_u8(self, &kb, is)?;
            let c16 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 16).into(), self.loc))?;
            let c4 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 4).into(), self.loc))?;
            let low_nib = self.push(&kb, arith::remui(sc_byte, c16, self.loc))?;
            let hi_nib = self.push(&kb, arith::divui(sc_byte, c16, self.loc))?;
            let low_nib_f32 = self.numeric_cast(&kb, low_nib, f32_t)?;
            let hi_nib_f32 = self.numeric_cast(&kb, hi_nib, f32_t)?;
            let term1 = self.push(&kb, arith::mulf(d_f32, low_nib_f32, self.loc))?;
            let term2 = self.push(&kb, arith::mulf(dmin_f32, hi_nib_f32, self.loc))?;

            let qs_byte_off = self.addi(&kb, qs_off, y)?;
            let q_byte = load_u8(self, &kb, qs_byte_off)?;
            let q2 = self.push(&kb, arith::divui(q_byte, shift_div, self.loc))?;
            let q2 = self.push(&kb, arith::remui(q2, c4, self.loc))?;
            let q2_f32 = self.numeric_cast(&kb, q2, f32_t)?;

            let decoded = self.push(&kb, arith::mulf(term1, q2_f32, self.loc))?;
            let decoded = self.push(&kb, arith::subf(decoded, term2, self.loc))?;

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
