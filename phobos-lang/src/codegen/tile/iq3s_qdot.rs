// Fused IQ3_S dot: two four-wide grid-table lookups a lane plus a
// sign-table lookup, folded into the contraction.

use super::*;

// IQ3_S block layout (phobos-gguf/src/quant/iq3_s.rs): 110 bytes, qs at
// byte 2 (two grid-index bytes a lane), qh at byte 66 (one byte a half),
// signs at byte 74, scales at byte 106.
const IQ3S_BLOCK_BYTES: i64 = 110;
const IQ3S_QS_OFF: i64 = 2;
const IQ3S_QH_OFF: i64 = 66;
const IQ3S_SIGNS_OFF: i64 = 74;
const IQ3S_SCALES_OFF: i64 = 106;
const IQ3S_HALF: i64 = 4;

impl<'c> Codegen<'c> {
    /// IQ3_S's matvec contraction with its two-grid-entry decode folded in.
    /// Same shape as `tile_iq1s_qdot_t`.
    pub(in crate::codegen) fn tile_iq3s_qdot_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
        signs: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        for (v, what) in [
            (a, "iq3s_qdot_t a"),
            (qb, "iq3s_qdot_t qb"),
            (d, "iq3s_qdot_t d"),
            (grid, "iq3s_qdot_t grid"),
            (signs, "iq3s_qdot_t signs"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if a.elem != self.f32_t || qb.elem != self.i8_t {
            bail!("iq3s_qdot_t contracts an f32 activation against IQ3_S's raw i8 bytes");
        }
        if d.elem != self.f16_t {
            bail!("iq3s_qdot_t's block scale must be f16");
        }
        if grid.elem != self.i32_t || signs.elem != self.i32_t {
            bail!("iq3s_qdot_t's grid and sign tables must hold i32 lanes");
        }
        if self.cta_threads % WARP != 0 {
            bail!("iq3s_qdot_t needs a CTA that is a whole number of warps");
        }
        let cols = qb.shape[0];
        if cols == DYN {
            bail!("iq3s_qdot_t needs a static output width");
        }
        if a.shape[1] != DYN && a.shape[1] % 256 != 0 {
            bail!("iq3s_qdot_t needs a whole number of 256-element blocks");
        }
        self.check_shapes(&[cols], &[d.shape[0]], "iq3s_qdot_t d rows")?;

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

        let eight_idx = self.const_index(&body, 8)?;
        let four_idx = self.const_index(&body, 4)?;
        let two_idx = self.const_index(&body, 2)?;
        let one_idx = self.const_index(&body, 1)?;
        let o = self.divui(&body, lane, eight_idx)?;
        let rem = self.remui(&body, lane, eight_idx)?;
        let half = self.divui(&body, rem, four_idx)?;
        let l = self.remui(&body, rem, four_idx)?;

        // grid1_off = QS_OFF + 2*lane (16*o + 8*half + 2*l == 2*lane).
        let grid1_off = self.addi(&body, self.const_index(&body, IQ3S_QS_OFF)?, self.muli(&body, lane, two_idx)?)?;
        let grid2_off = self.addi(&body, grid1_off, one_idx)?;
        // qh_off = QH_OFF + 2*o + half.
        let qh_off = self.addi(
            &body,
            self.addi(&body, self.const_index(&body, IQ3S_QH_OFF)?, self.muli(&body, o, two_idx)?)?,
            half,
        )?;
        // signs_off = SIGNS_OFF + lane (8*o + 4*half + l == lane).
        let signs_off = self.addi(&body, self.const_index(&body, IQ3S_SIGNS_OFF)?, lane)?;
        let scale_off = self.addi(&body, self.const_index(&body, IQ3S_SCALES_OFF)?, o)?;

        // qh_div1 = 1 << (2*l), qh_div2 = 1 << (2*l + 1).
        let l_i32 = self.numeric_cast(&body, l, i32_t)?;
        let two_i32 = self.push(&body, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 2).into(), self.loc))?;
        let one_i32 = self.push(&body, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 1).into(), self.loc))?;
        let exp1 = self.push(&body, arith::muli(l_i32, two_i32, self.loc))?;
        let exp2 = self.push(&body, arith::addi(exp1, one_i32, self.loc))?;
        let qh_div1 = self.push(&body, arith::shli(one_i32, exp1, self.loc))?;
        let qh_div2 = self.push(&body, arith::shli(one_i32, exp2, self.loc))?;

        // Scale nibble: half == 0 takes the low nibble, else the high.
        let zero_idx = self.const_index(&body, 0)?;
        let half_is_0 = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, half, zero_idx, self.loc),
        )?;

        let k_lane_off = self.muli(&body, lane, eight_idx)?;

        let step = self.const_index(&body, 256)?;
        let zero_k = self.const_index(&body, 0)?;
        let kd = if a.shape[1] == DYN {
            let one = self.const_index(&body, 1)?;
            self.push(&body, memref::dim(a.mem, one, self.loc))?
        } else {
            self.const_index(&body, a.shape[1])?
        };
        let init = self.zero_scalar(&body, f32_t)?;
        let blk_bytes = self.const_index(&body, IQ3S_BLOCK_BYTES)?;

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
        let g1_byte = load_u8(self, &kb, grid1_off)?;
        let g2_byte = load_u8(self, &kb, grid2_off)?;
        let qh_byte = load_u8(self, &kb, qh_off)?;
        let signs_byte = load_u8(self, &kb, signs_off)?;
        let scale_byte = load_u8(self, &kb, scale_off)?;

        let c16 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 16).into(), self.loc))?;
        let lo_nibble = self.push(&kb, arith::remui(scale_byte, c16, self.loc))?;
        let hi_nibble = self.push(&kb, arith::divui(scale_byte, c16, self.loc))?;
        let nibble = self.push(&kb, arith::select(half_is_0, lo_nibble, hi_nibble, self.loc))?;
        let c2 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 2).into(), self.loc))?;
        let c1 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 1).into(), self.loc))?;
        let db_word = self.push(&kb, arith::muli(nibble, c2, self.loc))?;
        let db_word = self.push(&kb, arith::addi(db_word, c1, self.loc))?;
        let db_f32 = self.numeric_cast(&kb, db_word, f32_t)?;
        let d_val = self.push(&kb, memref::load(d.mem, &[j, blk], self.loc))?;
        let d_f32 = self.numeric_cast(&kb, d_val, f32_t)?;
        let db = self.push(&kb, arith::mulf(d_f32, db_f32, self.loc))?;

        let c256 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 256).into(), self.loc))?;
        let four_idx_kb = self.const_index(&kb, 4)?;
        let signs_idx = self.numeric_cast(&kb, signs_byte, self.index_t)?;
        let eight_idx_kb = self.const_index(&kb, 8)?;
        let signs_base = self.push(&kb, arith::muli(signs_idx, eight_idx_kb, self.loc))?;

        let widen = |cg: &mut Self, at: &Block<'c>, grid_byte: Value<'c, 'c>, qh_div: Value<'c, 'c>| -> Result<Value<'c, 'c>> {
            let bit = cg.push(at, arith::divui(qh_byte, qh_div, cg.loc))?;
            let bit = cg.push(at, arith::remui(bit, c2, cg.loc))?;
            let hi = cg.push(at, arith::muli(bit, c256, cg.loc))?;
            let idx = cg.push(at, arith::addi(grid_byte, hi, cg.loc))?;
            let idx = cg.numeric_cast(at, idx, cg.index_t)?;
            cg.push(at, arith::muli(idx, four_idx_kb, cg.loc))
        };
        let g1_base = widen(self, &kb, g1_byte, qh_div1)?;
        let g2_base = widen(self, &kb, g2_byte, qh_div2)?;

        let mut partial = carry;
        let k_off = self.addi(&kb, kbase, k_lane_off)?;
        let zero_idx_kb = self.const_index(&kb, 0)?;
        for (half_i, (grid_base, sign_shift)) in [(g1_base, 0i64), (g2_base, 4i64)].into_iter().enumerate() {
            let sign_off = self.const_index(&kb, sign_shift)?;
            let half_off = self.const_index(&kb, (half_i as i64) * IQ3S_HALF)?;
            for y in 0..IQ3S_HALF {
                let yc = self.const_index(&kb, y)?;
                let mag_at = self.addi(&kb, grid_base, yc)?;
                let sign_at = self.addi(&kb, self.addi(&kb, signs_base, sign_off)?, yc)?;
                let mag_val = self.push(&kb, memref::load(grid.mem, &[zero_idx_kb, mag_at], self.loc))?;
                let sign_val = self.push(&kb, memref::load(signs.mem, &[zero_idx_kb, sign_at], self.loc))?;
                let mag_f32 = self.numeric_cast(&kb, mag_val, f32_t)?;
                let sign_f32 = self.numeric_cast(&kb, sign_val, f32_t)?;
                let decoded = self.push(&kb, arith::mulf(db, mag_f32, self.loc))?;
                let decoded = self.push(&kb, arith::mulf(decoded, sign_f32, self.loc))?;
                let a_at = self.addi(&kb, self.addi(&kb, k_off, half_off)?, yc)?;
                let a_val = self.push(&kb, memref::load(a.mem, &[zero_idx_kb, a_at], self.loc))?;
                let prod = self.push(&kb, arith::mulf(decoded, a_val, self.loc))?;
                partial = self.push(&kb, arith::addf(partial, prod, self.loc))?;
            }
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
