// Row argmax: the column of each row's largest element, the first on a tie.

use super::reduce::RowSrc;
use super::*;

impl<'c> Codegen<'c> {
    /// `[rows, 1]` i32 column indices of each row's maximum over `src`, whose
    /// elements are `elem`. Lanes of a warp share a row as in
    /// [`Self::rowreduce`]; with one lane per row there is no shuffle.
    pub(in crate::codegen) fn rowargmax(
        &mut self,
        block: &Block<'c>,
        src: &RowSrc<'_, 'c>,
        elem: Type<'c>,
    ) -> Result<MemVal<'c>> {
        let [rows, cols] = src.shape();
        if rows == DYN || cols == DYN {
            bail!("rowargmax needs a static tile shape");
        }
        if !self.is_float(elem) && !self.is_int(elem) {
            bail!("rowargmax needs a numeric element type, got {elem}");
        }
        let out = self.alloc_tile_shaped(block, self.i32_t, &[rows, 1])?;
        // gpu.shuffle carries a 32-bit payload.
        let lanes = match self.reduce_lanes(rows) {
            Some(lanes) if elem == self.f32_t || elem == self.i32_t => lanes,
            _ => 1,
        };

        let total = self.const_index(block, rows * lanes)?;
        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;
        let body = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body.argument(0)?.into());
        let lanes_v = self.const_index(&body, lanes)?;
        let row = self.divui(&body, li, lanes_v)?;
        let lane = self.remui(&body, li, lanes_v)?;

        // This lane's strided slice of the row. A strict `>` keeps the first
        // column of a tie, since each lane walks its columns in order.
        let init_v = self.reduce_identity(&body, elem, Reduce::Max)?;
        let init_i = self.const_i32(&body, 0)?;
        let hi = self.const_index(&body, cols)?;
        let jb = Block::new(&[
            (self.index_t, self.loc),
            (elem, self.loc),
            (self.i32_t, self.loc),
        ]);
        let j = detach(jb.argument(0)?.into());
        let (best_v, best_i) = (detach(jb.argument(1)?.into()), detach(jb.argument(2)?.into()));
        let x = self.row_elem(&jb, src, row, j, elem)?;
        let j32 = self.push(&jb, arith::index_cast(j, self.i32_t, self.loc))?;
        let better = self.order(&jb, elem, x, best_v, true)?;
        let nv = self.push(&jb, arith::select(better, x, best_v, self.loc))?;
        let ni = self.push(&jb, arith::select(better, j32, best_i, self.loc))?;
        jb.append_operation(scf::r#yield(&[nv, ni], self.loc));
        let jr = Region::new();
        jr.append_block(jb);
        let fold = body.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&[lane, hi, lanes_v, init_v, init_i])
                .add_results(&[elem, self.i32_t])
                .add_regions([jr])
                .build()?,
        );
        let mut v = detach(fold.result(0)?.into());
        let mut i = detach(fold.result(1)?.into());

        // The butterfly: take the other lane's pick when it is larger, or
        // equal at a smaller column.
        let mut mask = lanes / 2;
        while mask >= 1 {
            let ov = self.shfl_xor(&body, v, mask)?;
            let oi = self.shfl_xor(&body, i, mask)?;
            let larger = self.order(&body, elem, ov, v, true)?;
            let equal = self.order(&body, elem, ov, v, false)?;
            let earlier = self.push(
                &body,
                arith::cmpi(self.ctx, arith::CmpiPredicate::Slt, oi, i, self.loc),
            )?;
            let tie = self.push(&body, arith::andi(equal, earlier, self.loc))?;
            let take = self.push(&body, arith::ori(larger, tie, self.loc))?;
            v = self.push(&body, arith::select(take, ov, v, self.loc))?;
            i = self.push(&body, arith::select(take, oi, i, self.loc))?;
            mask /= 2;
        }

        let zero = self.const_index(&body, 0)?;
        let is_lead = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, zero, self.loc),
        )?;
        let then = Block::new(&[]);
        then.append_operation(memref::store(i, out.mem, &[row, zero], self.loc));
        then.append_operation(scf::r#yield(&[], self.loc));
        let tr = Region::new();
        tr.append_block(then);
        body.append_operation(scf::r#if(is_lead, &[], tr, Region::new(), self.loc));
        body.append_operation(scf::r#yield(&[], self.loc));
        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));
        self.barrier(block)?;
        Ok(out)
    }

    /// `a > b`, or `a == b` without `greater`, in `elem`'s own comparison.
    fn order(
        &self,
        block: &Block<'c>,
        elem: Type<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        greater: bool,
    ) -> Result<Value<'c, 'c>> {
        let op = if self.is_float(elem) {
            let pred = if greater { arith::CmpfPredicate::Ogt } else { arith::CmpfPredicate::Oeq };
            arith::cmpf(self.ctx, pred, a, b, self.loc)
        } else {
            let pred = if greater { arith::CmpiPredicate::Sgt } else { arith::CmpiPredicate::Eq };
            arith::cmpi(self.ctx, pred, a, b, self.loc)
        };
        self.push(block, op)
    }
}
