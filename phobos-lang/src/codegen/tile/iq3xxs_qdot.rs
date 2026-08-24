// Fused IQ3_XXS dot: two four-wide grid-table lookups a lane (grid entries
// are 4 bytes wide here, vs IQ2_XXS's one eight-wide lookup) plus a
// sign-table lookup, folded into the contraction. Reuses IQ2_XXS's
// scale/sign geometry (aux32 field is byte-for-byte the same).

use super::*;

// IQ3_XXS block layout (phobos-gguf/src/quant/iq3_xxs.rs): 98 bytes, qs at
// byte 2 (two grid-index bytes a lane), aux32 scale/sign field at byte 66.
const IQ3XXS_BLOCK_BYTES: i64 = 98;
const IQ3XXS_QS_OFF: i64 = 2;
const IQ3XXS_SS_OFF: i64 = 66;
const IQ3XXS_HALF: i64 = 4;

impl<'c> Codegen<'c> {
    /// IQ3_XXS's matvec contraction with its two-grid-entry decode folded
    /// in. Same shape as `tile_iq1s_qdot_t`; reuses `tile_iq2xxs_qdot_t`'s
    /// `arith.select`-folded lo/hi byte geometry.
    pub(in crate::codegen) fn tile_iq3xxs_qdot_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
        signs: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        for (v, what) in [
            (a, "iq3xxs_qdot_t a"),
            (qb, "iq3xxs_qdot_t qb"),
            (d, "iq3xxs_qdot_t d"),
            (grid, "iq3xxs_qdot_t grid"),
            (signs, "iq3xxs_qdot_t signs"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if a.elem != self.f32_t || qb.elem != self.i8_t {
            bail!("iq3xxs_qdot_t contracts an f32 activation against IQ3_XXS's raw i8 bytes");
        }
        if d.elem != self.f16_t {
            bail!("iq3xxs_qdot_t's block scale must be f16");
        }
        if grid.elem != self.i32_t || signs.elem != self.i32_t {
            bail!("iq3xxs_qdot_t's grid and sign tables must hold i32 lanes");
        }
        if self.cta_threads % WARP != 0 {
            bail!("iq3xxs_qdot_t needs a CTA that is a whole number of warps");
        }
        let cols = qb.shape[0];
        if cols == DYN {
            bail!("iq3xxs_qdot_t needs a static output width");
        }
        if a.shape[1] != DYN && a.shape[1] % 256 != 0 {
            bail!("iq3xxs_qdot_t needs a whole number of 256-element blocks");
        }
        self.check_shapes(&[cols], &[d.shape[0]], "iq3xxs_qdot_t d rows")?;

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
        let eight_idx = self.const_index(&body, 8)?;
        let zero_idx = self.const_index(&body, 0)?;
        let one_idx = self.const_index(&body, 1)?;
        let two_idx = self.const_index(&body, 2)?;
        let ib32 = self.divui(&body, lane, four)?;
        let l = self.remui(&body, lane, four)?;
        let is_l0 = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, l, zero_idx, self.loc),
        )?;

        // base = QS_OFF + 8*ib32 + 2*l; g1_off = base, g2_off = base + 1.
        let base = self.addi(
            &body,
            self.addi(&body, self.const_index(&body, IQ3XXS_QS_OFF)?, self.muli(&body, ib32, eight_idx)?)?,
            self.muli(&body, l, two_idx)?,
        )?;
        let g1_off = base;
        let g2_off = self.addi(&body, base, one_idx)?;

        // aux = SS_OFF + 4*ib32; scale_off = aux + 3.
        let aux = self.addi(&body, self.const_index(&body, IQ3XXS_SS_OFF)?, self.muli(&body, ib32, four)?)?;
        let three_idx = self.const_index(&body, 3)?;
        let scale_off = self.addi(&body, aux, three_idx)?;

        // Same lo/hi/shift-div folding as iq2xxs_qdot_t: lo_off = aux +
        // (max(l, 1) - 1), hi_off = aux + l (a harmless re-read at l == 0).
        let l_or_1 = self.push(&body, arith::select(is_l0, one_idx, l, self.loc))?;
        let lo_off = self.addi(&body, aux, self.subi(&body, l_or_1, one_idx)?)?;
        let hi_off = self.addi(&body, aux, l)?;

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
        let blk_bytes = self.const_index(&body, IQ3XXS_BLOCK_BYTES)?;

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
        let g1_byte = load_u8(self, &kb, g1_off)?;
        let g2_byte = load_u8(self, &kb, g2_off)?;
        let scale_byte = load_u8(self, &kb, scale_off)?;
        let lo = load_u8(self, &kb, lo_off)?;
        let hi = load_u8(self, &kb, hi_off)?;

        let is_l0_kb = self.push(
            &kb,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, l, self.const_index(&kb, 0)?, self.loc),
        )?;
        let zero_i32 = self.zero_scalar(&kb, i32_t)?;
        let hi_term = self.push(&kb, arith::select(is_l0_kb, zero_i32, hi, self.loc))?;
        let c256 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 256).into(), self.loc))?;
        let hi_shifted = self.push(&kb, arith::muli(hi_term, c256, self.loc))?;
        let word = self.push(&kb, arith::addi(lo, hi_shifted, self.loc))?;

        let eight_i32 = self.numeric_cast(&kb, self.const_index(&kb, 8)?, i32_t)?;
        let l_i32 = self.numeric_cast(&kb, l, i32_t)?;
        let exp = self.push(&kb, arith::subi(eight_i32, l_i32, self.loc))?;
        let zero_exp = self.zero_scalar(&kb, i32_t)?;
        let exp = self.push(&kb, arith::select(is_l0_kb, zero_exp, exp, self.loc))?;
        let one_i32 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 1).into(), self.loc))?;
        let shift_div = self.push(&kb, arith::shli(one_i32, exp, self.loc))?;

        let c128 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 128).into(), self.loc))?;
        let signs_idx = self.push(&kb, arith::divui(word, shift_div, self.loc))?;
        let signs_idx = self.push(&kb, arith::remui(signs_idx, c128, self.loc))?;
        let signs_idx = self.numeric_cast(&kb, signs_idx, self.index_t)?;
        let signs_base = self.push(&kb, arith::muli(signs_idx, eight_idx, self.loc))?;

        let c16 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 16).into(), self.loc))?;
        let scale_shifted = self.push(&kb, arith::divui(scale_byte, c16, self.loc))?;
        let scale_f32 = self.numeric_cast(&kb, scale_shifted, f32_t)?;
        let half_c = self.push(&kb, arith::constant(self.ctx, FloatAttribute::new(self.ctx, f32_t, 0.5).into(), self.loc))?;
        let d_val = self.push(&kb, memref::load(d.mem, &[j, blk], self.loc))?;
        let d_f32 = self.numeric_cast(&kb, d_val, f32_t)?;
        let sc = self.push(&kb, arith::addf(half_c, scale_f32, self.loc))?;
        let sc = self.push(&kb, arith::mulf(sc, half_c, self.loc))?;
        let db = self.push(&kb, arith::mulf(d_f32, sc, self.loc))?;

        let four_idx = self.const_index(&kb, 4)?;
        let g1_idx = self.numeric_cast(&kb, g1_byte, self.index_t)?;
        let g2_idx = self.numeric_cast(&kb, g2_byte, self.index_t)?;
        let g1_base = self.push(&kb, arith::muli(g1_idx, four_idx, self.loc))?;
        let g2_base = self.push(&kb, arith::muli(g2_idx, four_idx, self.loc))?;

        let mut partial = carry;
        let k_off = self.addi(&kb, kbase, k_lane_off)?;
        let zero_idx_kb = self.const_index(&kb, 0)?;
        for (half, (grid_base, sign_shift)) in [(g1_base, 0i64), (g2_base, 4i64)].into_iter().enumerate() {
            let sign_off = self.const_index(&kb, sign_shift)?;
            let half_off = self.const_index(&kb, (half as i64) * IQ3XXS_HALF)?;
            for y in 0..IQ3XXS_HALF {
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
