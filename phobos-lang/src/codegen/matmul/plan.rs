// The k-loop shared by the three paths, and the epilogue's scaling.

use super::*;

impl<'c> Codegen<'c> {
    /// Runs the fused matmul's k-loop. Both the vector and WMMA paths use it.
    /// With `@pipeline`, it double-buffers by unrolling two iterations: the
    /// first half computes from one buffer pair while prefetching the next
    /// into the other, the second half does the same for the odd iteration
    /// (its accumulators pass through a CTA-uniform scf.if). Without it, this
    /// just stages, barriers, accumulates, barriers in place.
    ///
    /// Returns the finished accumulators; stage prefetches one iteration
    /// into a pair, half emits one pipelined half, mac accumulates one pair.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn matmul_kloop(
        &mut self,
        block: &Block<'c>,
        bounds: (Value<'c, 'c>, Value<'c, 'c>, Value<'c, 'c>),
        regs: &[Value<'c, 'c>],
        a_bufs: &[MemVal<'c>],
        b_bufs: &[MemVal<'c>],
        stage: impl Fn(&mut Self, &Block<'c>, Value<'c, 'c>, &MemVal<'c>, &MemVal<'c>) -> Result<()>,
        half: impl Fn(
            &mut Self,
            &Block<'c>,
            Value<'c, 'c>,
            (&MemVal<'c>, &MemVal<'c>),
            (&MemVal<'c>, &MemVal<'c>),
            &[Value<'c, 'c>],
        ) -> Result<Vec<Value<'c, 'c>>>,
        mac: impl Fn(
            &mut Self,
            &Block<'c>,
            &MemVal<'c>,
            &MemVal<'c>,
            &[Value<'c, 'c>],
        ) -> Result<Vec<Value<'c, 'c>>>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let (lo, hi, st) = bounds;

        if !self.pipeline_assert {
            return self.carry_loop(block, lo, hi, st, regs, |cg, body, kt, accs| {
                stage(cg, body, kt, &a_bufs[0], &b_bufs[0])?;

                // publish the staging, accumulate, then retire this
                // iteration's reads before the next one overwrites.
                cg.barrier(body)?;
                let next = mac(cg, body, &a_bufs[0], &b_bufs[0], accs)?;
                cg.barrier(body)?;
                Ok(next)
            });
        }

        // prologue: stage lo into the first pair, then unroll by two
        stage(self, block, lo, &a_bufs[0], &b_bufs[0])?;
        self.barrier(block)?;
        let two = self.const_index(block, 2)?;
        let st2 = self.muli(block, st, two)?;
        self.carry_loop(block, lo, hi, st2, regs, |cg, body, kt, accs| {
            let next = cg.addi(body, kt, st)?;
            let half_a = half(
                cg,
                body,
                next,
                (&a_bufs[0], &b_bufs[0]),
                (&a_bufs[1], &b_bufs[1]),
                accs,
            )?;

            // half_b: when iteration kt + st exists.
            let have_b = cg.push(
                body,
                arith::cmpi(cg.ctx, arith::CmpiPredicate::Slt, next, hi, cg.loc),
            )?;
            let then_block = Block::new(&[]);
            let next2 = cg.addi(&then_block, next, st)?;
            let half_b = half(
                cg,
                &then_block,
                next2,
                (&a_bufs[1], &b_bufs[1]),
                (&a_bufs[0], &b_bufs[0]),
                &half_a,
            )?;

            then_block.append_operation(scf::r#yield(&half_b, cg.loc));
            let then_region = Region::new();
            then_region.append_block(then_block);

            let else_block = Block::new(&[]);
            else_block.append_operation(scf::r#yield(&half_a, cg.loc));
            let else_region = Region::new();
            else_region.append_block(else_block);

            let types: Vec<Type<'c>> = half_a.iter().map(|v| v.r#type()).collect();
            let op =
                body.append_operation(scf::r#if(have_b, &types, then_region, else_region, cg.loc));

            (0..half_a.len())
                .map(|i| Ok(detach(op.result(i)?.into())))
                .collect()
        })
    }

    /// Applies precomputed GEMM scaling to one accumulator value, producing
    /// alpha*acc + beta*prev_load, or alpha*acc, or just acc. prev_load (the
    /// prior C value) is only loaded when beta is present. Works the same on
    /// scalars and vectors since the arith ops are element-wise.
    pub(in crate::codegen) fn apply_scaling(
        &mut self,
        block: &Block<'c>,
        acc: Value<'c, 'c>,
        alpha: Option<Value<'c, 'c>>,
        beta: Option<Value<'c, 'c>>,
        load_prev: impl FnOnce(&mut Self) -> Result<Value<'c, 'c>>,
    ) -> Result<Value<'c, 'c>> {
        let f32 = self.f32_t;
        match (alpha, beta) {
            (Some(a), Some(b)) => {
                let prev = load_prev(self)?;
                let ar = self.push(block, self.elem_arith(BinOp::Mul, f32, a, acc)?)?;
                let br = self.push(block, self.elem_arith(BinOp::Mul, f32, b, prev)?)?;
                self.push(block, self.elem_arith(BinOp::Add, f32, ar, br)?)
            }
            (Some(a), None) => self.push(block, self.elem_arith(BinOp::Mul, f32, a, acc)?),
            _ => Ok(acc),
        }
    }

    /// The warp's block origin (m0, n0) on a wm x wn warp grid of bm x bn
    /// element blocks, with surplus warps clamped onto the last block.
    /// Returns (tid, w, wt, m0, n0) so callers can derive lane coordinates.
    pub(in crate::codegen) fn warp_block_origin(
        &self,
        block: &Block<'c>,
        wm: i64,
        wn: i64,
        bm: i64,
        bn: i64,
    ) -> Result<(
        Value<'c, 'c>,
        Value<'c, 'c>,
        Value<'c, 'c>,
        Value<'c, 'c>,
        Value<'c, 'c>,
    )> {
        let tid = self.thread_id(block)?;
        let w = self.const_index(block, 32)?;
        let warp_id = self.divui(block, tid, w)?;
        let wmax = self.const_index(block, wm * wn - 1)?;
        let wt = self.minsi(block, warp_id, wmax)?;
        let wn_v = self.const_index(block, wn)?;
        let q = self.divui(block, wt, wn_v)?;
        let r = self.remui(block, wt, wn_v)?;
        let bm_v = self.const_index(block, bm)?;
        let bn_v = self.const_index(block, bn)?;
        let m0 = self.muli(block, q, bm_v)?;
        let n0 = self.muli(block, r, bn_v)?;

        Ok((tid, w, wt, m0, n0))
    }

    /// The (row, col) origin of 16x16 fragment (fi, fj) within the warp's
    /// block at (m0, n0).
    pub(super) fn frag_origin(
        &self,
        block: &Block<'c>,
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        fi: i64,
        fj: i64,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let c_fi = self.const_index(block, fi * 16)?;
        let c_fj = self.const_index(block, fj * 16)?;
        Ok((self.addi(block, m0, c_fi)?, self.addi(block, n0, c_fj)?))
    }
}
