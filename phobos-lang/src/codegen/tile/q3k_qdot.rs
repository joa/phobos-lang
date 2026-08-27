// Fused Q3_K dot: static-offset decode folded into the contraction. Same
// warp mapping as q2k_qdot.rs, but a six-bit signed scale per run unpacked
// from three interleaved bytes, and a third quant bit from a separate
// hmask plane.

use super::*;

// Q3_K block layout (phobos-gguf/src/quant/q3_k.rs): hmask at byte 0, qs at
// byte 32, scales at byte 96. No minimum term, unlike Q2_K. The blocks are
// 110 bytes on disk and on the host; the device upload pads them to 112 so
// every block base is a multiple of eight, which is what the two vector loads
// below need. See `Quant::device_block_bytes`.
const Q3K_BLOCK_BYTES: i64 = 112;
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

        // A lane owns eight consecutive elements of one run: run `is`, from
        // `lane / 2`, and elements `y0 = (lane % 2) * 8` upwards. Sixteen runs
        // of sixteen over thirty-two lanes covers the whole 256-element block
        // in a single pass, and eight consecutive elements make the qs bytes,
        // the hmask bytes and the activations one wide load each. The earlier
        // map gave a lane one element per run and strided its eight by 32, so
        // nothing could merge: nineteen load instructions for the same 42
        // bytes this shape reads in seven, which left the kernel stalled on
        // `long_scoreboard` at a fiftieth of the bandwidth its siblings reach.
        //
        // Every index below comes from `is`, so it is invariant across the
        // k-loop and computed once here rather than eight times a block.
        let two_idx = self.const_index(&body, 2)?;
        let four_idx = self.const_index(&body, 4)?;
        let eight_idx = self.const_index(&body, 8)?;
        let sixteen_idx = self.const_index(&body, Q3K_RUN)?;
        let zero_idx = self.const_index(&body, 0)?;
        let lanes_per_run = self.const_index(&body, WARP / Q3K_RUNS)?;
        let is = self.divui(&body, lane, lanes_per_run)?;
        let y0 = self.muli(&body, self.remui(&body, lane, lanes_per_run)?, eight_idx)?;

        let h = self.divui(&body, is, eight_idx)?;
        let rem = self.remui(&body, is, eight_idx)?;
        let jj = self.divui(&body, rem, two_idx)?;
        let half2 = self.remui(&body, rem, two_idx)?;
        let half2_16 = self.muli(&body, half2, sixteen_idx)?;

        // qs_off = QS_OFF + h*32 + half2*16 + y0 and hm_off = half2*16 + y0.
        // Every term is a multiple of eight, and so is the block stride, which
        // is what lets the two loads below promise align-8.
        let thirtytwo = self.const_index(&body, 32)?;
        let qs_base = self.addi(
            &body,
            self.const_index(&body, Q3K_QS_OFF)?,
            self.muli(&body, h, thirtytwo)?,
        )?;
        let qs_off = self.addi(&body, self.addi(&body, qs_base, half2_16)?, y0)?;
        let hm_off = self.addi(&body, half2_16, y0)?;

        let jj_i32 = self.numeric_cast(&body, jj, i32_t)?;
        let h_i32 = self.numeric_cast(&body, h, i32_t)?;
        let two_i32 = self.const_i32(&body, 2)?;
        let four_i32 = self.const_i32(&body, 4)?;
        let qs_shift = self.push(&body, arith::muli(jj_i32, two_i32, self.loc))?;
        let h4 = self.push(&body, arith::muli(h_i32, four_i32, self.loc))?;
        let hm_shift = self.push(&body, arith::addi(h4, jj_i32, self.loc))?;

        // The run's six-bit scale, a low nibble and two high bits that sit in
        // separate bytes: `group` picks the pair, `c` the byte within it.
        let group = self.divui(&body, is, four_idx)?;
        let c = self.remui(&body, is, four_idx)?;
        let group_mod2 = self.remui(&body, group, two_idx)?;
        let group_even = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, group_mod2, zero_idx, self.loc),
        )?;
        let c_plus4 = self.addi(&body, c, four_idx)?;
        let lo_extra = self.push(&body, arith::select(group_even, c, c_plus4, self.loc))?;
        let lo_off = self.addi(&body, self.const_index(&body, Q3K_SCALES_OFF)?, lo_extra)?;
        let hi_off = self.addi(&body, self.const_index(&body, Q3K_SCALES_OFF + 8)?, c)?;
        let group_lt2 = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, group, two_idx, self.loc),
        )?;
        let zero_i32 = self.const_i32(&body, 0)?;
        let lo_shift = self.push(&body, arith::select(group_lt2, zero_i32, four_i32, self.loc))?;
        let group_i32 = self.numeric_cast(&body, group, i32_t)?;
        let hi_shift = self.push(&body, arith::muli(group_i32, two_i32, self.loc))?;

        // Where the lane's eight activations start. `y0` is 0 or 8, so the
        // byte address is 0 or 32 past a block boundary: a multiple of sixteen
        // either way, which is what the two `vector<4xf32>` loads need.
        let act_off = self.addi(&body, self.muli(&body, is, sixteen_idx)?, y0)?;

        let step = self.const_index(&body, 256)?;
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
        let zero_idx_kb = self.const_index(&kb, 0)?;

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

        let c2_i32 = self.const_i32(&kb, 2)?;
        let c4_i32 = self.const_i32(&kb, 4)?;
        let c16_i32 = self.const_i32(&kb, 16)?;

        let lo_byte = load_u8(self, &kb, lo_off)?;
        let lo_val = self.push(&kb, arith::shrui(lo_byte, lo_shift, self.loc))?;
        let lo_val = self.push(&kb, arith::remui(lo_val, c16_i32, self.loc))?;
        let hi_byte = load_u8(self, &kb, hi_off)?;
        let hi_val = self.push(&kb, arith::shrui(hi_byte, hi_shift, self.loc))?;
        let hi_val = self.push(&kb, arith::remui(hi_val, c4_i32, self.loc))?;
        let hi_val = self.push(&kb, arith::muli(hi_val, c16_i32, self.loc))?;
        let scale_word = self.push(&kb, arith::addi(lo_val, hi_val, self.loc))?;
        let scale_f32 = self.numeric_cast(&kb, scale_word, f32_t)?;
        let c32f = self.const_f32(&kb, 32.0)?;
        let scale = self.push(&kb, arith::subf(scale_f32, c32f, self.loc))?;
        let dscale = self.push(&kb, arith::mulf(d_f32, scale, self.loc))?;

        // The lane's whole share of the block: two eight-byte weight loads and
        // two sixteen-byte activation loads.
        let i8_t = self.i8_t;
        let byte_vec_t = Type::vector(&[8], i8_t);
        let qs_at = self.addi(&kb, blk_off, qs_off)?;
        let qs_v = self.vec_load_al(&kb, qb.mem, &[j, qs_at], byte_vec_t, 8)?;
        let hm_at = self.addi(&kb, blk_off, hm_off)?;
        let hm_v = self.vec_load_al(&kb, qb.mem, &[j, hm_at], byte_vec_t, 8)?;

        let act_vec_t = Type::vector(&[ACT_VEC as u64], f32_t);
        let a_at = self.addi(&kb, kbase, act_off)?;
        let a_lo = self.vec_load(&kb, a.mem, &[zero_idx_kb, a_at], act_vec_t)?;
        let a_at_hi = self.addi(&kb, a_at, self.const_index(&kb, ACT_VEC)?)?;
        let a_hi = self.vec_load(&kb, a.mem, &[zero_idx_kb, a_at_hi], act_vec_t)?;

        let c4f = self.const_f32(&kb, 4.0)?;
        let mut partial = carry;
        for t in 0..(2 * ACT_VEC) {
            let qs_b = self.vec_extract(&kb, qs_v, &[t], i8_t)?;
            let qs_i = self.push(
                &kb,
                OperationBuilder::new("arith.extui", self.loc)
                    .add_operands(&[qs_b])
                    .add_results(&[i32_t])
                    .build()?,
            )?;
            let low2 = self.push(&kb, arith::shrui(qs_i, qs_shift, self.loc))?;
            let low2 = self.push(&kb, arith::remui(low2, c4_i32, self.loc))?;
            let low2_f32 = self.numeric_cast(&kb, low2, f32_t)?;

            let hm_b = self.vec_extract(&kb, hm_v, &[t], i8_t)?;
            let hm_i = self.push(
                &kb,
                OperationBuilder::new("arith.extui", self.loc)
                    .add_operands(&[hm_b])
                    .add_results(&[i32_t])
                    .build()?,
            )?;
            let bit = self.push(&kb, arith::shrui(hm_i, hm_shift, self.loc))?;
            let bit = self.push(&kb, arith::remui(bit, c2_i32, self.loc))?;
            let bit_f32 = self.numeric_cast(&kb, bit, f32_t)?;

            let bit4 = self.push(&kb, arith::mulf(bit_f32, c4f, self.loc))?;
            let low2_minus4 = self.push(&kb, arith::subf(low2_f32, c4f, self.loc))?;
            let quant = self.push(&kb, arith::addf(low2_minus4, bit4, self.loc))?;
            let decoded = self.push(&kb, arith::mulf(dscale, quant, self.loc))?;

            let av = if t < ACT_VEC { a_lo } else { a_hi };
            let a_val = self.vec_extract(&kb, av, &[t % ACT_VEC], f32_t)?;
            let prod = self.push(&kb, arith::mulf(decoded, a_val, self.loc))?;
            partial = self.push(&kb, arith::addf(partial, prod, self.loc))?;
        }
        kb.append_operation(scf::r#yield(&[partial], self.loc));

        let kr = Region::new();
        kr.append_block(kb);
        let mut acc = self.push(
            &body,
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&[zero_idx, kd, step, init])
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
