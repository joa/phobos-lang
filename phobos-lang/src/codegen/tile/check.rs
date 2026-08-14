// Shape agreement, bounds masks, and distributing a tile over the CTA.
//
// `distribute` is what turns a tile-shaped operation into the thread-
// distributed loop nest the rest of the emitter writes into.

use super::*;

impl<'c> Codegen<'c> {
    pub(in crate::codegen) fn check_matmul_shapes(
        &self,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if a.shape.len() != 2 || b.shape.len() != 2 || out.shape.len() != 2 {
            bail!("dot expects rank-2 tiles");
        }
        self.check_shapes(&[a.shape[1]], &[b.shape[0]], "dot contraction dim")?;
        self.check_shapes(&[a.shape[0], b.shape[1]], &out.shape, "dot result")
    }

    pub(in crate::codegen) fn check_matmul_elems(
        &self,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        what: &str,
    ) -> Result<()> {
        if a.elem == b.elem && a.elem == out.elem {
            return Ok(());
        }

        // The accumulator has to hold both operands: their join, or something
        // wider of the same kind. Equal width is not enough, since f16 and bf16
        // are both 16b and neither holds the other.
        let holds = self.numeric_join(a.elem, b.elem).is_some_and(|join| {
            if join == out.elem {
                return true;
            }
            match (self.is_float(join), self.is_float(out.elem)) {
                (true, true) => self.float_bits(out.elem) > self.float_bits(join),
                (false, false) => self.elem_bytes(out.elem) > self.elem_bytes(join),
                // An integer contraction into a float accumulator, or the
                // reverse, would silently change what the sum means.
                _ => false,
            }
        });

        if holds {
            Ok(())
        } else {
            bail!(
                "{what}: cannot accumulate {} and {} operands into {}",
                a.elem,
                b.elem,
                out.elem
            )
        }
    }

    pub(in crate::codegen) fn check_shapes(&self, a: &[i64], b: &[i64], what: &str) -> Result<()> {
        let ok = a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(&x, &y)| x == DYN || y == DYN || x == y);

        if !ok {
            bail!(
                "{what}: shape mismatch ({} vs {})",
                fmt_shape(a),
                fmt_shape(b)
            );
        }

        Ok(())
    }

    /// Conjunction of `offset + idx[d] < extent` over the masked dims of a
    /// tensor slice, or None when the mask is empty (every dim in bounds).
    /// The offset and extent values were materialized where the slice was
    /// taken, so they dominate the distributed loop body that calls this.
    pub(in crate::codegen) fn bounds_pred(
        &self,
        block: &Block<'c>,
        mask: &[Option<(Value<'c, 'c>, Value<'c, 'c>)>],
        idx: &[Value<'c, 'c>],
    ) -> Result<Option<Value<'c, 'c>>> {
        let mut pred: Option<Value<'c, 'c>> = None;

        for (d, entry) in mask.iter().enumerate() {
            let Some((off, extent)) = entry else { continue };
            let global = self.addi(block, *off, idx[d])?;
            let in_bounds = self.push(
                block,
                arith::cmpi(
                    self.ctx,
                    arith::CmpiPredicate::Ult,
                    global,
                    *extent,
                    self.loc,
                ),
            )?;

            pred = Some(match pred {
                Some(p) => self.push(block, arith::andi(p, in_bounds, self.loc))?,
                None => in_bounds,
            });
        }

        Ok(pred)
    }

    /// Stages a partially out-of-bounds slice into a fresh, fully in-bounds
    /// tile: in-bounds elements are copied, out-of-bounds ones read as zero.
    /// The load index is clamped to 0 on any masked dim that overflows, so no
    /// access ever leaves the tensor, then a select substitutes zero for the
    /// clamped reads. Downstream ops then treat the result as an ordinary
    /// dense tile.
    pub(in crate::codegen) fn materialize_masked(
        &mut self,
        block: &Block<'c>,
        view: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        if view.shape.contains(&DYN) {
            bail!("a masked tensor slice needs a static shape");
        }

        let dst = self.alloc_tile_shaped(block, view.elem, &view.shape)?;
        let mask = view.mask.clone();
        let src = view.mem;
        let dst_mem = dst.mem;
        let elem = view.elem;

        self.distribute(block, &dst, 1, true, |cg, blk, idx| {
            let zero_idx = cg.const_index(blk, 0)?;
            let mut safe = idx.to_vec();
            let mut pred: Option<Value<'c, 'c>> = None;

            for (d, entry) in mask.iter().enumerate() {
                let Some((off, extent)) = entry else { continue };
                let global = cg.addi(blk, *off, idx[d])?;
                let in_bounds = cg.push(
                    blk,
                    arith::cmpi(cg.ctx, arith::CmpiPredicate::Ult, global, *extent, cg.loc),
                )?;

                safe[d] = cg.select(blk, in_bounds, idx[d], zero_idx)?;
                pred = Some(match pred {
                    Some(p) => cg.push(blk, arith::andi(p, in_bounds, cg.loc))?,
                    None => in_bounds,
                });
            }

            let loaded = cg.push(blk, memref::load(src, &safe, cg.loc))?;
            let val = match pred {
                Some(p) => {
                    let zero = cg.zero_scalar(blk, elem)?;
                    cg.select(blk, p, loaded, zero)?
                }
                None => loaded,
            };

            blk.append_operation(memref::store(val, dst_mem, idx, cg.loc));

            Ok(())
        })?;

        Ok(dst)
    }

    /// arith.select(cond, a, b): a when cond is true, else b.
    pub(in crate::codegen) fn select(
        &self,
        block: &Block<'c>,
        cond: Value<'c, 'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(block, arith::select(cond, a, b, self.loc))
    }

    /// Emits body once per element of out, or once per width-element innermost
    /// segment when width > 1 (the caller guarantees a static, width-divisible
    /// innermost extent), distributed across the CTA.
    ///
    /// sync=false skips the trailing barrier, for pipelined prefetch copies that
    /// get synced by their iteration's closing barrier instead.
    pub(in crate::codegen) fn distribute(
        &mut self,
        block: &Block<'c>,
        out: &MemVal<'c>,
        width: i64,
        sync: bool,
        body: impl FnOnce(&mut Self, &Block<'c>, &[Value<'c, 'c>]) -> Result<()>,
    ) -> Result<()> {
        let mut sizes = self.tile_sizes(block, out)?;
        let rank = sizes.len();

        if width > 1 {
            sizes[rank - 1] = self.const_index(block, out.shape[rank - 1] / width)?;
        }

        let mut total = sizes[0];

        for &s in &sizes[1..] {
            total = self.muli(block, total, s)?;
        }

        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;

        let body_block = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body_block.argument(0)?.into());

        // Last dim varies fastest, so adjacent threads touch adjacent elements.
        let mut idx = vec![li; rank];

        if rank > 1 {
            let mut rem = li;

            for i in (1..rank).rev() {
                idx[i] = self.remui(&body_block, rem, sizes[i])?;
                rem = self.divui(&body_block, rem, sizes[i])?;
            }

            idx[0] = rem;
        }

        if width > 1 {
            let w = self.const_index(&body_block, width)?;
            idx[rank - 1] = self.muli(&body_block, idx[rank - 1], w)?;
        }

        // A masked output writes only the in-bounds elements: the whole body
        // runs under an scf.if guarding offset + local index < extent. Callers
        // scalarize masked writes, so the guard is exact per element. The
        // trailing barrier stays outside it, being CTA-uniform where the guard
        // is not.
        if let Some(pred) = self.bounds_pred(&body_block, &out.mask, &idx)? {
            let then = Block::new(&[]);
            body(self, &then, &idx)?;
            then.append_operation(scf::r#yield(&[], self.loc));
            let then_region = Region::new();
            then_region.append_block(then);
            body_block.append_operation(scf::r#if(pred, &[], then_region, Region::new(), self.loc));
        } else {
            body(self, &body_block, &idx)?;
        }

        body_block.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body_block);
        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));

        // Barriers cannot deadlock here: the language never exposes thread ids,
        // so every scalar value, and so all control flow, is CTA-uniform.
        if sync {
            self.barrier(block)?;
        }

        Ok(())
    }

    /// Vector width for an elementwise op over these buffers: 4 when every
    /// buffer has f32 elements, provably 16B-aligned rows, and a static
    /// innermost extent divisible by 4; otherwise 1 (scalar).
    pub(in crate::codegen) fn elementwise_width(&self, mvs: &[&MemVal<'c>]) -> i64 {
        // A masked buffer is scalarized so the per-element store guard is
        // exact (a partial vector could straddle the bounds).
        if mvs.iter().any(|m| m.is_masked()) {
            return 1;
        }
        let ok = mvs.iter().all(|m| {
            let last = *m.shape.last().expect("tile values are not rank-0");
            m.elem == self.f32_t && m.vectorizes(4) && last != DYN && last % 4 == 0
        });
        if ok { 4 } else { 1 }
    }

    /// Per-axis index for reading a broadcast operand: axes where the operand
    /// has extent 1 but the output is wider read index 0.
    pub(in crate::codegen) fn bc_index(
        &self,
        block: &Block<'c>,
        idx: &[Value<'c, 'c>],
        out_shape: &[i64],
        src_shape: &[i64],
    ) -> Result<Vec<Value<'c, 'c>>> {
        idx.iter()
            .enumerate()
            .map(|(d, &ix)| {
                if src_shape[d] == 1 && out_shape[d] != 1 {
                    self.const_index(block, 0)
                } else {
                    Ok(ix)
                }
            })
            .collect()
    }
}
