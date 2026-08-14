// The four-way integer dot product, for cards without integer
// tensor cores and for single-row contractions.

use super::*;

impl<'c> Codegen<'c> {
    pub(super) fn tile_matmul_t_dp4a(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        kd: i64,
    ) -> Result<bool> {
        let applies = a.elem == self.i8_t
            && b.elem == self.i8_t
            && out.elem == self.i32_t
            && kd % 4 == 0
            && a.swizzle.is_none()
            && b.swizzle.is_none()
            && !a.is_masked()
            && !b.is_masked()
            && self.has_dp4a();
        if !applies {
            return Ok(false);
        }

        let i32_t = self.i32_t;
        let vec4_i8 = Type::vector(&[4], self.i8_t);
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let (i, j) = (idx[0], idx[1]);
            let slot_t = MemRefType::new(i32_t, &[], None, None);
            let slot = cg.push(blk, memref::alloca(cg.ctx, slot_t, &[], &[], None, cg.loc))?;
            let zero = cg.zero_scalar(blk, i32_t)?;
            blk.append_operation(memref::store(zero, slot, &[], cg.loc));

            // One iteration per group of four bytes.
            let lo = cg.const_index(blk, 0)?;
            let hi = cg.const_index(blk, kd / 4)?;
            let st = cg.const_index(blk, 1)?;
            let kb = Block::new(&[(cg.index_t, cg.loc)]);
            let group = detach(kb.argument(0)?.into());
            let four = cg.const_index(&kb, 4)?;
            let k = cg.push(&kb, arith::muli(group, four, cg.loc))?;

            let va = cg.vec_load_al(&kb, a.mem, &[i, k], vec4_i8, 4)?;
            let vb = cg.vec_load_al(&kb, b.mem, &[j, k], vec4_i8, 4)?;
            let cur = cg.push(&kb, memref::load(slot, &[], cg.loc))?;
            let acc = cg.dot4_accumulate(&kb, va, vb, cur)?;
            kb.append_operation(memref::store(acc, slot, &[], cg.loc));
            kb.append_operation(scf::r#yield(&[], cg.loc));
            let region = Region::new();
            region.append_block(kb);
            blk.append_operation(scf::r#for(lo, hi, st, region, cg.loc));

            let fin = cg.push(blk, memref::load(slot, &[], cg.loc))?;
            blk.append_operation(memref::store(fin, out.mem, idx, cg.loc));
            Ok(())
        })?;
        Ok(true)
    }
}
