// Fused IQ1_S dot: grid-table decode folded into the contraction.

use super::*;

// IQ1_S block layout (phobos-gguf/src/quant/iq1_s.rs): 50 bytes, qs at byte
// 2, qh at byte 34, 32 lanes of 8 elements (a warp's width, so warp lane l
// decodes format lane l directly).
const IQ1S_BLOCK_BYTES: i64 = 50;
const IQ1S_QS_OFF: i64 = 2;
const IQ1S_QH_OFF: i64 = 34;
const IQ1S_LANE: i64 = 8;
const IQ1S_DELTA: f32 = 0.125;

impl<'c> Codegen<'c> {
    /// IQ1_S's matvec contraction with the decode folded in. A warp owns one
    /// output; its 32 lanes divide the contraction into a register
    /// accumulator, synchronized once via a closing shuffle instead of
    /// staging every intermediate through shared memory with a barrier each.
    /// Each lane reads its own 9-bit grid index and 3-bit group scale from
    /// the block's qs/qh bytes.
    pub(in crate::codegen) fn tile_iq1s_qdot_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        for (v, what) in [
            (a, "iq1s_qdot_t a"),
            (qb, "iq1s_qdot_t qb"),
            (d, "iq1s_qdot_t d"),
            (grid, "iq1s_qdot_t grid"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if a.elem != self.f32_t || qb.elem != self.i8_t {
            bail!("iq1s_qdot_t contracts an f32 activation against IQ1_S's raw i8 bytes");
        }
        if d.elem != self.f16_t {
            bail!("iq1s_qdot_t's block scale must be f16");
        }
        if grid.elem != self.i32_t {
            bail!("iq1s_qdot_t's grid table must hold i32 lanes");
        }
        if self.cta_threads % WARP != 0 {
            bail!("iq1s_qdot_t needs a CTA that is a whole number of warps");
        }
        let cols = qb.shape[0];
        if cols == DYN {
            bail!("iq1s_qdot_t needs a static output width");
        }
        if a.shape[1] != DYN && a.shape[1] % 256 != 0 {
            bail!("iq1s_qdot_t needs a whole number of 256-element blocks");
        }
        self.check_shapes(&[cols], &[d.shape[0]], "iq1s_qdot_t d rows")?;

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
        let three = self.const_index(&body, 3)?;
        let two = self.const_index(&body, 2)?;
        let one = self.const_index(&body, 1)?;
        let ib = self.divui(&body, lane, four)?;
        let l = self.remui(&body, lane, four)?;
        let qs_off = self.addi(&body, self.const_index(&body, IQ1S_QS_OFF)?, lane)?;
        let qh_lo_off = self.addi(
            &body,
            self.const_index(&body, IQ1S_QH_OFF)?,
            self.muli(&body, ib, two)?,
        )?;
        let qh_hi_off = self.addi(&body, qh_lo_off, one)?;
        let shift = self.muli(&body, l, three)?;
        let shift = self.numeric_cast(&body, shift, i32_t)?;
        let k_lane_off = self.muli(&body, lane, self.const_index(&body, IQ1S_LANE)?)?;

        let step = self.const_index(&body, 256)?;
        let zero_k = self.const_index(&body, 0)?;
        let kd = if a.shape[1] == DYN {
            let one = self.const_index(&body, 1)?;
            self.push(&body, memref::dim(a.mem, one, self.loc))?
        } else {
            self.const_index(&body, a.shape[1])?
        };
        let init = self.zero_scalar(&body, f32_t)?;
        let blk_bytes = self.const_index(&body, IQ1S_BLOCK_BYTES)?;

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
        let qs = load_u8(self, &kb, qs_off)?;
        let qh_lo = load_u8(self, &kb, qh_lo_off)?;
        let qh_hi = load_u8(self, &kb, qh_hi_off)?;
        let c256 = self.const_index(&kb, 256)?;
        let c256_i32 = self.numeric_cast(&kb, c256, i32_t)?;
        let qh_hi_shifted = self.push(&kb, arith::muli(qh_hi, c256_i32, self.loc))?;
        let qh = self.push(&kb, arith::addi(qh_lo, qh_hi_shifted, self.loc))?;

        let c8 = self.const_index(&kb, 8)?;
        let c8_i32 = self.numeric_cast(&kb, c8, i32_t)?;
        let c4096 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 4096).into(), self.loc))?;
        let c32768 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 32768).into(), self.loc))?;
        let c2 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 2).into(), self.loc))?;
        let c1 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 1).into(), self.loc))?;

        let d_val = self.push(&kb, memref::load(d.mem, &[j, blk], self.loc))?;
        let d_f32 = self.numeric_cast(&kb, d_val, f32_t)?;
        let sc = self.push(&kb, arith::divui(qh, c4096, self.loc))?;
        let sc = self.push(&kb, arith::remui(sc, c8_i32, self.loc))?;
        let sc = self.push(&kb, arith::muli(sc, c2, self.loc))?;
        let sc = self.push(&kb, arith::addi(sc, c1, self.loc))?;
        let sc_f32 = self.numeric_cast(&kb, sc, f32_t)?;
        let dl = self.push(&kb, arith::mulf(d_f32, sc_f32, self.loc))?;

        let sign_bit = self.push(&kb, arith::divui(qh, c32768, self.loc))?;
        let sign_bit = self.push(&kb, arith::remui(sign_bit, c2, self.loc))?;
        let sign_f32 = self.numeric_cast(&kb, sign_bit, f32_t)?;
        let delta_c = self.push(&kb, arith::constant(self.ctx, FloatAttribute::new(self.ctx, f32_t, f64::from(IQ1S_DELTA)).into(), self.loc))?;
        let twice_delta_c = self.push(&kb, arith::constant(self.ctx, FloatAttribute::new(self.ctx, f32_t, f64::from(2.0 * IQ1S_DELTA)).into(), self.loc))?;
        let sign_term = self.push(&kb, arith::mulf(twice_delta_c, sign_f32, self.loc))?;
        let delta = self.push(&kb, arith::subf(delta_c, sign_term, self.loc))?;

        let shift_div = self.push(&kb, arith::shrui(qh, shift, self.loc))?;
        let shift_div = self.push(&kb, arith::remui(shift_div, c8_i32, self.loc))?;
        let shift_term = self.push(&kb, arith::muli(shift_div, c256_i32, self.loc))?;
        let base_idx = self.push(&kb, arith::addi(qs, shift_term, self.loc))?;
        let base_idx = self.numeric_cast(&kb, base_idx, self.index_t)?;
        let base_idx8 = self.push(&kb, arith::muli(base_idx, c8, self.loc))?;

        let mut partial = carry;
        let k_off = self.addi(&kb, kbase, k_lane_off)?;
        for y in 0..IQ1S_LANE {
            let yc = self.const_index(&kb, y)?;
            let grid_at = self.addi(&kb, base_idx8, yc)?;
            let zero = self.const_index(&kb, 0)?;
            let grid_val = self.push(&kb, memref::load(grid.mem, &[zero, grid_at], self.loc))?;
            let grid_f32 = self.numeric_cast(&kb, grid_val, f32_t)?;
            let plus_delta = self.push(&kb, arith::addf(grid_f32, delta, self.loc))?;
            let decoded = self.push(&kb, arith::mulf(dl, plus_delta, self.loc))?;
            let a_at = self.addi(&kb, k_off, yc)?;
            let zero_row = self.const_index(&kb, 0)?;
            let a_val = self.push(&kb, memref::load(a.mem, &[zero_row, a_at], self.loc))?;
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
