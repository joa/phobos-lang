// Fused IQ2_S dot: two grid-table lookups a lane, folded into the
// contraction. The decode itself lives in `iq2s.rs`, shared with `qdecode.rs`.

use super::iq2s::{IQ2S_BLOCK_BYTES, IQ2S_LANE};
use super::*;

impl<'c> Codegen<'c> {
    /// IQ2_S's matvec contraction with its magnitude-grid and sign-table
    /// lookups folded in. Same shape as `tile_iq1s_qdot_t`.
    pub(in crate::codegen) fn tile_iq2s_qdot_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
        signs: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        for (v, what) in [
            (a, "iq2s_qdot_t a"),
            (qb, "iq2s_qdot_t qb"),
            (d, "iq2s_qdot_t d"),
            (grid, "iq2s_qdot_t grid"),
            (signs, "iq2s_qdot_t signs"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if a.elem != self.f32_t || qb.elem != self.i8_t {
            bail!("iq2s_qdot_t contracts an f32 activation against IQ2_S's raw i8 bytes");
        }
        if d.elem != self.f16_t {
            bail!("iq2s_qdot_t's block scale must be f16");
        }
        if grid.elem != self.i8_t || signs.elem != self.i8_t {
            bail!("iq2s_qdot_t's grid and sign tables must hold packed i8 lanes");
        }
        if self.cta_threads % WARP != 0 {
            bail!("iq2s_qdot_t needs a CTA that is a whole number of warps");
        }
        let cols = qb.shape[0];
        if cols == DYN {
            bail!("iq2s_qdot_t needs a static output width");
        }
        if a.shape[1] != DYN && a.shape[1] % 256 != 0 {
            bail!("iq2s_qdot_t needs a whole number of 256-element blocks");
        }
        self.check_shapes(&[cols], &[d.shape[0]], "iq2s_qdot_t d rows")?;

        let out = self.alloc_tile_shaped(block, self.f32_t, &[1, cols])?;
        let f32_t = self.f32_t;

        let warp_w = self.const_index(block, WARP)?;
        let total = self.const_index(block, cols * WARP)?;
        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;

        let body = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body.argument(0)?.into());
        let j = self.divui(&body, li, warp_w)?;
        let lane = self.remui(&body, li, warp_w)?;

        let geom = self.iq2s_lane(&body, lane)?;

        let step = self.const_index(&body, 256)?;
        let zero_k = self.const_index(&body, 0)?;
        let kd = if a.shape[1] == DYN {
            let one = self.const_index(&body, 1)?;
            self.push(&body, memref::dim(a.mem, one, self.loc))?
        } else {
            self.const_index(&body, a.shape[1])?
        };
        let init = self.zero_scalar(&body, f32_t)?;
        let blk_bytes = self.const_index(&body, IQ2S_BLOCK_BYTES)?;

        let kb = Block::new(&[(self.index_t, self.loc), (f32_t, self.loc)]);
        let kbase = detach(kb.argument(0)?.into());
        let carry = detach(kb.argument(1)?.into());
        let blk = self.divui(&kb, kbase, step)?;
        let at = self.raw_block_at(&kb, j, blk, blk_bytes)?;
        let dec = self.iq2s_block(&kb, &geom, qb, d, &QTables { grid, signs }, &at)?;

        let mut partial = carry;
        let k_off = self.addi(&kb, kbase, geom.k_lane_off)?;
        let zero_idx_kb = self.const_index(&kb, 0)?;
        let a_t = Type::vector(&[ACT_VEC as u64], f32_t);
        for g in 0..IQ2S_LANE / ACT_VEC {
            let goff = self.const_index(&kb, g * ACT_VEC)?;
            let a_at = self.addi(&kb, k_off, goff)?;
            let a_v = self.vec_load(&kb, a.mem, &[zero_idx_kb, a_at], a_t)?;
            for i in 0..ACT_VEC {
                let decoded = self.iq2s_decoded(&kb, &dec, g * ACT_VEC + i)?;
                let a_val = self.vec_extract(&kb, a_v, &[i], f32_t)?;
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
