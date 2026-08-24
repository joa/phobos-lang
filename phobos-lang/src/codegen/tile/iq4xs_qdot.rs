// Fused IQ4_XS dot: a fixed 16-entry codebook lookup folded into the
// contraction. Not grid-coded like the other formats here: a block is eight
// 32-element runs, and a warp's 32 lanes map onto one run (RUN == WARP).
// A run's scale is uniform across it, computed once and shared by every
// lane; only the codebook nibble index varies by lane.

use super::*;

// IQ4_XS block layout (phobos-gguf/src/quant/iq4_xs.rs): 136 bytes,
// scales_h at byte 2, scales_l at byte 4, qs at byte 8.
const IQ4XS_BLOCK_BYTES: i64 = 136;
const IQ4XS_SCALES_H_OFF: i64 = 2;
const IQ4XS_SCALES_L_OFF: i64 = 4;
const IQ4XS_QS_OFF: i64 = 8;
const IQ4XS_RUNS: i64 = 8;
const IQ4XS_RUN: i64 = 32;

/// Four offsets for run `ib` (see
/// [`crate::backend::device::kernels::iq4xs::run_geometry`]), compile-time
/// constants here since `ib` is a Rust loop index, not a lane id.
fn run_geometry(ib: i64) -> (i64, i64, i64, i64) {
    let scale_l_off = IQ4XS_SCALES_L_OFF + ib / 2;
    let scale_l_div = if ib % 2 == 0 { 1 } else { 16 };
    let scale_h_div = 4i64.pow(ib as u32);
    let qs_off = IQ4XS_QS_OFF + (IQ4XS_RUN / 2) * ib;
    (scale_l_off, scale_l_div, scale_h_div, qs_off)
}

impl<'c> Codegen<'c> {
    /// IQ4_XS's matvec contraction with the codebook lookup folded in. Same
    /// warp-owns-an-output shape as `tile_iq1s_qdot_t`, but the lane maps to
    /// an element position within a 32-wide run, not a format lane.
    pub(in crate::codegen) fn tile_iq4xs_qdot_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        codebook: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        for (v, what) in [
            (a, "iq4xs_qdot_t a"),
            (qb, "iq4xs_qdot_t qb"),
            (d, "iq4xs_qdot_t d"),
            (codebook, "iq4xs_qdot_t codebook"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if a.elem != self.f32_t || qb.elem != self.i8_t {
            bail!("iq4xs_qdot_t contracts an f32 activation against IQ4_XS's raw i8 bytes");
        }
        if d.elem != self.f16_t {
            bail!("iq4xs_qdot_t's block scale must be f16");
        }
        if codebook.elem != self.i32_t {
            bail!("iq4xs_qdot_t's codebook must hold i32 values");
        }
        if self.cta_threads % WARP != 0 {
            bail!("iq4xs_qdot_t needs a CTA that is a whole number of warps");
        }
        let cols = qb.shape[0];
        if cols == DYN {
            bail!("iq4xs_qdot_t needs a static output width");
        }
        if a.shape[1] != DYN && a.shape[1] % 256 != 0 {
            bail!("iq4xs_qdot_t needs a whole number of 256-element blocks");
        }
        self.check_shapes(&[cols], &[d.shape[0]], "iq4xs_qdot_t d rows")?;

        let out = self.alloc_tile_shaped(block, self.f32_t, &[1, cols])?;
        let (i32_t, f32_t) = (self.i32_t, self.f32_t);

        let warp_w = self.const_index(block, WARP)?;
        let total = self.const_index(block, cols * WARP)?;
        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;

        let body = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body.argument(0)?.into());
        let j = self.divui(&body, li, warp_w)?;
        // `y`: this lane's position within a 32-element run (RUN == WARP).
        let y = self.remui(&body, li, warp_w)?;

        let sixteen_idx = self.const_index(&body, 16)?;
        let y_mod16 = self.remui(&body, y, sixteen_idx)?;
        let y_lt_16 = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, y, sixteen_idx, self.loc),
        )?;

        let step = self.const_index(&body, 256)?;
        let zero_k = self.const_index(&body, 0)?;
        let kd = if a.shape[1] == DYN {
            let one = self.const_index(&body, 1)?;
            self.push(&body, memref::dim(a.mem, one, self.loc))?
        } else {
            self.const_index(&body, a.shape[1])?
        };
        let init = self.zero_scalar(&body, f32_t)?;
        let blk_bytes = self.const_index(&body, IQ4XS_BLOCK_BYTES)?;

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
        let load_u8_const = |cg: &mut Self, at: &Block<'c>, off: i64| -> Result<Value<'c, 'c>> {
            let off_v = cg.const_index(at, off)?;
            load_u8(cg, at, off_v)
        };

        // scales_h is one halfword, the same for every run in the block.
        let sh_lo = load_u8_const(self, &kb, IQ4XS_SCALES_H_OFF)?;
        let sh_hi = load_u8_const(self, &kb, IQ4XS_SCALES_H_OFF + 1)?;
        let c256 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 256).into(), self.loc))?;
        let sh_hi_shifted = self.push(&kb, arith::muli(sh_hi, c256, self.loc))?;
        let scales_h = self.push(&kb, arith::addi(sh_lo, sh_hi_shifted, self.loc))?;

        let d_val = self.push(&kb, memref::load(d.mem, &[j, blk], self.loc))?;
        let d_f32 = self.numeric_cast(&kb, d_val, f32_t)?;

        let c16 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 16).into(), self.loc))?;
        let c4 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 4).into(), self.loc))?;
        let c32 = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, 32).into(), self.loc))?;

        let mut partial = carry;
        let zero_idx_kb = self.const_index(&kb, 0)?;
        let k_off = self.addi(&kb, kbase, y)?;
        for ib in 0..IQ4XS_RUNS {
            let (scale_l_off, scale_l_div, scale_h_div, qs_off) = run_geometry(ib);

            let sl_byte = load_u8_const(self, &kb, scale_l_off)?;
            let sl_div_c = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, scale_l_div).into(), self.loc))?;
            let low = self.push(&kb, arith::divui(sl_byte, sl_div_c, self.loc))?;
            let low = self.push(&kb, arith::remui(low, c16, self.loc))?;

            let sh_div_c = self.push(&kb, arith::constant(self.ctx, IntegerAttribute::new(i32_t, scale_h_div).into(), self.loc))?;
            let high = self.push(&kb, arith::divui(scales_h, sh_div_c, self.loc))?;
            let high = self.push(&kb, arith::remui(high, c4, self.loc))?;
            let high16 = self.push(&kb, arith::muli(high, c16, self.loc))?;

            let scale_word = self.push(&kb, arith::addi(low, high16, self.loc))?;
            let scale_word = self.push(&kb, arith::subi(scale_word, c32, self.loc))?;
            let scale_f32 = self.numeric_cast(&kb, scale_word, f32_t)?;
            let dl = self.push(&kb, arith::mulf(d_f32, scale_f32, self.loc))?;

            let qs_base = self.const_index(&kb, qs_off)?;
            let qs_byte_off = self.addi(&kb, qs_base, y_mod16)?;
            let qs_byte = load_u8(self, &kb, qs_byte_off)?;
            let lo_nib = self.push(&kb, arith::remui(qs_byte, c16, self.loc))?;
            let hi_nib = self.push(&kb, arith::divui(qs_byte, c16, self.loc))?;
            let nibble = self.push(&kb, arith::select(y_lt_16, lo_nib, hi_nib, self.loc))?;
            let nibble_idx = self.numeric_cast(&kb, nibble, self.index_t)?;
            let cb_val = self.push(&kb, memref::load(codebook.mem, &[zero_idx_kb, nibble_idx], self.loc))?;
            let cb_f32 = self.numeric_cast(&kb, cb_val, f32_t)?;
            let decoded = self.push(&kb, arith::mulf(dl, cb_f32, self.loc))?;

            let run_off = self.const_index(&kb, ib * IQ4XS_RUN)?;
            let a_at = self.addi(&kb, k_off, run_off)?;
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
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, y, zero, self.loc),
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
