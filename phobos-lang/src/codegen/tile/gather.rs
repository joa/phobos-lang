// gather(TABLE, IDX): a per-element table lookup, out[i] = TABLE[IDX[i]].
// Unlike other elementwise tile ops, each output element reads from a
// data-dependent source rather than the same offset it writes to.
//
// A table may be rank-1 or rank-2 with a leading dim of 1: kernel parameters
// are always rank-2 (`push_descriptor`'s fixed two-extent descriptor), and
// `A[0, :]` can't reach rank-1 since point and slice subscripts don't mix.

use super::*;

impl<'c> Codegen<'c> {
    pub(in crate::codegen) fn tile_gather(
        &mut self,
        block: &Block<'c>,
        table: &MemVal<'c>,
        idx: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        let leading_one = table.shape.len() == 2 && table.shape[0] == 1;
        if table.shape.len() != 1 && !leading_one {
            bail!("gather's table must be a rank-1 tensor, or rank-2 with a leading dim of 1");
        }
        if idx.shape.contains(&DYN) {
            bail!("gather needs a static index tile shape");
        }
        if !self.is_int(idx.elem) {
            bail!("gather's index tile must hold integers");
        }
        if idx.is_masked() {
            bail!("gather's index tile must be a fully in-bounds slice");
        }

        let out = self.alloc_tile_shaped(block, table.elem, &idx.shape)?;

        self.distribute(block, &out, 1, true, |cg, blk, at| {
            let i = cg.load_scalar(blk, idx, at)?;
            let indices = if leading_one {
                let zero = cg.const_index(blk, 0)?;
                vec![zero, i]
            } else {
                vec![i]
            };
            let v = cg.push(blk, memref::load(table.mem, &indices, cg.loc))?;
            blk.append_operation(memref::store(v, out.mem, at, cg.loc));
            Ok(())
        })?;

        Ok(out)
    }
}
