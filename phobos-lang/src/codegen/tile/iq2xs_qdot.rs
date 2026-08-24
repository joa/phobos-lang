// Fused IQ2_XS dot: two grid-table lookups a lane, folded into the
// contraction. A 16-bit qs halfword splits into a magnitude index and a
// sign index via % 512 / / 512.

use super::*;

// IQ2_XS block layout (phobos-gguf/src/quant/iq2_xs.rs): 74 bytes, qs at
// byte 2 (two bytes a lane), scales at byte 66.
const IQ2XS_BLOCK_BYTES: i64 = 74;
const IQ2XS_QS_OFF: i64 = 2;
const IQ2XS_SCALES_OFF: i64 = 66;
const IQ2XS_LANE: i64 = 8;

impl<'c> Codegen<'c> {
    /// IQ2_XS's matvec contraction with its magnitude-grid and sign-table
    /// lookups folded in (the sign table is IQ2_XXS's own, reused outright).
    /// Same shape as `tile_iq1s_qdot_t`.
    pub(in crate::codegen) fn tile_iq2xs_qdot_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
        signs: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        for (v, what) in [
            (a, "iq2xs_qdot_t a"),
            (qb, "iq2xs_qdot_t qb"),
            (d, "iq2xs_qdot_t d"),
            (grid, "iq2xs_qdot_t grid"),
            (signs, "iq2xs_qdot_t signs"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if a.elem != self.f32_t || qb.elem != self.i8_t {
            bail!("iq2xs_qdot_t contracts an f32 activation against IQ2_XS's raw i8 bytes");
        }
        if d.elem != self.f16_t {
            bail!("iq2xs_qdot_t's block scale must be f16");
        }
        if grid.elem != self.i32_t || signs.elem != self.i32_t {
            bail!("iq2xs_qdot_t's grid and sign tables must hold i32 lanes");
        }
        if self.cta_threads % WARP != 0 {
            bail!("iq2xs_qdot_t needs a CTA that is a whole number of warps");
        }
        let cols = qb.shape[0];
        if cols == DYN {
            bail!("iq2xs_qdot_t needs a static output width");
        }
        if a.shape[1] != DYN && a.shape[1] % 256 != 0 {
            bail!("iq2xs_qdot_t needs a whole number of 256-element blocks");
        }
        self.check_shapes(&[cols], &[d.shape[0]], "iq2xs_qdot_t d rows")?;

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
        let one_idx = self.const_index(&body, 1)?;
        let ib32 = self.divui(&body, lane, four)?;
        let l = self.remui(&body, lane, four)?;

        let lo_off = self.addi(
            &body,
            self.const_index(&body, IQ2XS_QS_OFF)?,
            self.muli(&body, lane, two_idx)?,
        )?;
        let hi_off = self.addi(&body, lo_off, one_idx)?;
        let scale_off = self.addi(&body, self.const_index(&body, IQ2XS_SCALES_OFF)?, ib32)?;

        let l_lt_2 = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, l, two_idx, self.loc),
        )?;

        let k_lane_off = self.muli(&body, lane, self.const_index(&body, IQ2XS_LANE)?)?;

        let step = self.const_index(&body, 256)?;
        let zero_k = self.const_index(&body, 0)?;
        let kd = if a.shape[1] == DYN {
            let one = self.const_index(&body, 1)?;
            self.push(&body, memref::dim(a.mem, one, self.loc))?
        } else {
            self.const_index(&body, a.shape[1])?
        };
        let init = self.zero_scalar(&body, f32_t)?;
        let blk_bytes = self.const_index(&body, IQ2XS_BLOCK_BYTES)?;

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
        let lo = load_u8(self, &kb, lo_off)?;
        let hi = load_u8(self, &kb, hi_off)?;
        let scale_byte = load_u8(self, &kb, scale_off)?;

        let c256 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 256).into(), self.loc))?;
        let hi_shifted = self.push(&kb, arith::muli(hi, c256, self.loc))?;
        let q16 = self.push(&kb, arith::addi(lo, hi_shifted, self.loc))?;

        let c16 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 16).into(), self.loc))?;
        let lo_nibble = self.push(&kb, arith::remui(scale_byte, c16, self.loc))?;
        let hi_nibble = self.push(&kb, arith::divui(scale_byte, c16, self.loc))?;
        let nibble = self.push(&kb, arith::select(l_lt_2, lo_nibble, hi_nibble, self.loc))?;
        let nibble_f32 = self.numeric_cast(&kb, nibble, f32_t)?;
        let half = self.push(&kb, arith::constant(self.ctx, FloatAttribute::new(self.ctx, f32_t, 0.5).into(), self.loc))?;
        let quarter = self.push(&kb, arith::constant(self.ctx, FloatAttribute::new(self.ctx, f32_t, 0.25).into(), self.loc))?;
        let d_val = self.push(&kb, memref::load(d.mem, &[j, blk], self.loc))?;
        let d_f32 = self.numeric_cast(&kb, d_val, f32_t)?;
        let sc = self.push(&kb, arith::addf(half, nibble_f32, self.loc))?;
        let sc = self.push(&kb, arith::mulf(sc, quarter, self.loc))?;
        let dl = self.push(&kb, arith::mulf(d_f32, sc, self.loc))?;

        let c512 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 512).into(), self.loc))?;
        let mag_idx = self.push(&kb, arith::remui(q16, c512, self.loc))?;
        let sign_idx = self.push(&kb, arith::divui(q16, c512, self.loc))?;
        let mag_idx = self.numeric_cast(&kb, mag_idx, self.index_t)?;
        let sign_idx = self.numeric_cast(&kb, sign_idx, self.index_t)?;
        let eight_idx = self.const_index(&kb, 8)?;
        let mag_idx8 = self.push(&kb, arith::muli(mag_idx, eight_idx, self.loc))?;
        let sign_idx8 = self.push(&kb, arith::muli(sign_idx, eight_idx, self.loc))?;

        let mut partial = carry;
        let k_off = self.addi(&kb, kbase, k_lane_off)?;
        let zero_idx_kb = self.const_index(&kb, 0)?;
        for y in 0..IQ2XS_LANE {
            let yc = self.const_index(&kb, y)?;
            let mag_at = self.addi(&kb, mag_idx8, yc)?;
            let sign_at = self.addi(&kb, sign_idx8, yc)?;
            let mag_val = self.push(&kb, memref::load(grid.mem, &[zero_idx_kb, mag_at], self.loc))?;
            let sign_val = self.push(&kb, memref::load(signs.mem, &[zero_idx_kb, sign_at], self.loc))?;
            let mag_f32 = self.numeric_cast(&kb, mag_val, f32_t)?;
            let sign_f32 = self.numeric_cast(&kb, sign_val, f32_t)?;
            let decoded = self.push(&kb, arith::mulf(dl, mag_f32, self.loc))?;
            let decoded = self.push(&kb, arith::mulf(decoded, sign_f32, self.loc))?;
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
