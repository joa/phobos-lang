// Row reductions, serial and warp-cooperative.

use super::*;

impl<'c> Codegen<'c> {
    /// Reduces a rank-2 tile over its last (column) dim into a [rows, 1]
    /// column vector. When the CTA has threads to spare, lanes of a warp
    /// cooperate on each row ([`Self::rowreduce_warp`]); otherwise one thread
    /// sweeps each row serially ([`Self::rowreduce_serial`]).
    pub(in crate::codegen) fn tile_rowreduce(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        kind: Reduce,
    ) -> Result<MemVal<'c>> {
        if src.shape.len() != 2 {
            bail!("row reduction expects a rank-2 tile");
        }
        let (rows, cols) = (src.shape[0], src.shape[1]);
        if rows == DYN || cols == DYN {
            bail!("row reduction needs a static tile shape");
        }
        let elem = src.elem;
        if !self.is_float(elem) {
            bail!("row reduction needs a float element type");
        }
        let out = self.alloc_tile_shaped(block, elem, &[rows, 1])?;
        match self.reduce_lanes(rows) {
            // gpu.shuffle carries a 32-bit payload; f16 keeps the serial path.
            Some(lanes) if elem == self.f32_t => {
                self.rowreduce_warp(block, src, &out, kind, lanes)?
            }
            _ => self.rowreduce_serial(block, src, &out, kind)?,
        }
        Ok(out)
    }

    /// The fold identity: 0 for sum; for max, smaller than any finite input
    /// (the first column overwrites it). f16 saturates at -65504, so use a
    /// representable floor.
    pub(super) fn reduce_identity(
        &self,
        block: &Block<'c>,
        elem: Type<'c>,
        kind: Reduce,
    ) -> Result<Value<'c, 'c>> {
        match kind {
            Reduce::Sum => self.zero_scalar(block, elem),
            Reduce::Max => {
                let floor = if elem == self.f16_t {
                    -65504.0
                } else {
                    -3.0e38
                };
                self.push(
                    block,
                    arith::constant(
                        self.ctx,
                        FloatAttribute::new(self.ctx, elem, floor).into(),
                        self.loc,
                    ),
                )
            }
        }
    }

    /// Lanes cooperating on each row of a warp-shuffled row reduction: the
    /// largest power of two the CTA can spend per row, capped at the warp
    /// width, such that rows * lanes covers whole warps (shfl.sync stalls
    /// unless every lane of a participating warp reaches it). None when only
    /// a single lane per row fits; the serial path covers that.
    pub(super) fn reduce_lanes(&self, rows: i64) -> Option<i64> {
        let per_row = self.cta_threads / rows.max(1);
        if per_row < 2 {
            return None;
        }
        let mut lanes = 1i64 << per_row.min(32).ilog2();
        while lanes >= 2 {
            if (rows * lanes) % 32 == 0 {
                return Some(lanes);
            }
            lanes /= 2;
        }
        None
    }

    /// Each output row is owned by one thread, which sweeps the columns with
    /// a scalar accumulator (a thread-private rank-0 alloca).
    pub(super) fn rowreduce_serial(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        out: &MemVal<'c>,
        kind: Reduce,
    ) -> Result<()> {
        let cols = src.shape[1];
        let elem = src.elem;
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let i = idx[0];
            // Thread-private scalar accumulator.
            let slot_t = MemRefType::new(elem, &[], None, None);
            let slot = cg.push(blk, memref::alloca(cg.ctx, slot_t, &[], &[], None, cg.loc))?;
            let init = cg.reduce_identity(blk, elem, kind)?;
            blk.append_operation(memref::store(init, slot, &[], cg.loc));

            let lo = cg.const_index(blk, 0)?;
            let hi = cg.const_index(blk, cols)?;
            let st = cg.const_index(blk, 1)?;
            let jb = Block::new(&[(cg.index_t, cg.loc)]);
            let j = detach(jb.argument(0)?.into());
            let v = cg.push(&jb, memref::load(src.mem, &[i, j], cg.loc))?;
            let cur = cg.push(&jb, memref::load(slot, &[], cg.loc))?;
            let nv = match kind {
                Reduce::Sum => cg.push(&jb, arith::addf(cur, v, cg.loc))?,
                Reduce::Max => cg.fmax(&jb, cur, v)?,
            };
            jb.append_operation(memref::store(nv, slot, &[], cg.loc));
            jb.append_operation(scf::r#yield(&[], cg.loc));
            let region = Region::new();
            region.append_block(jb);
            blk.append_operation(scf::r#for(lo, hi, st, region, cg.loc));

            let fin = cg.push(blk, memref::load(slot, &[], cg.loc))?;
            blk.append_operation(memref::store(fin, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// Warp-cooperative row reduction: lanes consecutive lanes fold one row
    /// (each folds a strided slice of the columns as an scf.for iter_arg),
    /// then a gpu.shuffle xor butterfly combines the partials and lane 0 of
    /// the group stores the row result.
    ///
    /// Safety of the shuffle: rows * lanes covers whole warps and the block
    /// dim is a warp multiple, so any warp reaching the shuffle has all 32
    /// lanes present. The xor masks stay below lanes, so a lanes-aligned
    /// group never exchanges outside itself.
    pub(super) fn rowreduce_warp(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        out: &MemVal<'c>,
        kind: Reduce,
        lanes: i64,
    ) -> Result<()> {
        let (rows, cols) = (src.shape[0], src.shape[1]);
        let total = self.const_index(block, rows * lanes)?;
        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;

        let body = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body.argument(0)?.into());
        let lanes_v = self.const_index(&body, lanes)?;
        let row = self.divui(&body, li, lanes_v)?;
        let lane = self.remui(&body, li, lanes_v)?;

        // This lane's strided slice of the row, folded in a register.
        let init = self.reduce_identity(&body, self.f32_t, kind)?;
        let hi = self.const_index(&body, cols)?;
        let jb = Block::new(&[(self.index_t, self.loc), (self.f32_t, self.loc)]);
        let j = detach(jb.argument(0)?.into());
        let cur = detach(jb.argument(1)?.into());
        let v = self.push(&jb, memref::load(src.mem, &[row, j], self.loc))?;
        let nv = match kind {
            Reduce::Sum => self.push(&jb, arith::addf(cur, v, self.loc))?,
            Reduce::Max => self.fmax(&jb, cur, v)?,
        };
        jb.append_operation(scf::r#yield(&[nv], self.loc));
        let jr = Region::new();
        jr.append_block(jb);
        let mut acc = self.push(
            &body,
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&[lane, hi, lanes_v, init])
                .add_results(&[self.f32_t])
                .add_regions([jr])
                .build()?,
        )?;

        let mut mask = lanes / 2;
        while mask >= 1 {
            let other = self.shfl_xor_f32(&body, acc, mask)?;
            acc = match kind {
                Reduce::Sum => self.push(&body, arith::addf(acc, other, self.loc))?,
                Reduce::Max => self.fmax(&body, acc, other)?,
            };
            mask /= 2;
        }

        let zero = self.const_index(&body, 0)?;
        let is_lead = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, zero, self.loc),
        )?;
        let then = Block::new(&[]);
        then.append_operation(memref::store(acc, out.mem, &[row, zero], self.loc));
        then.append_operation(scf::r#yield(&[], self.loc));
        let tr = Region::new();
        tr.append_block(then);
        body.append_operation(scf::r#if(is_lead, &[], tr, Region::new(), self.loc));

        body.append_operation(scf::r#yield(&[], self.loc));
        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));
        self.barrier(block)?;
        Ok(())
    }
}
