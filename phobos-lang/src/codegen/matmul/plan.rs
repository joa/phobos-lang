// The k-loop shared by the three paths, and the epilogue's scaling.

use super::*;

impl<'c> Codegen<'c> {
    /// Runs the fused matmul's k-loop and returns the finished accumulators.
    ///
    /// Without `@pipeline`, each iteration stages, barriers, accumulates and
    /// barriers again. With it, the loop double-buffers: each half computes
    /// from one buffer pair while prefetching into the other, so a turn is
    /// an even number of halves, `unroll` rounded up to even. Either way
    /// every iteration of a turn after its first sits in a CTA-uniform
    /// scf.if, for a count the turn does not divide.
    ///
    /// `stage` stages one iteration into a pair, `half` emits one pipelined
    /// half, and `mac` accumulates from one pair.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn matmul_kloop(
        &mut self,
        block: &Block<'c>,
        bounds: (Value<'c, 'c>, Value<'c, 'c>, Value<'c, 'c>),
        unroll: i64,
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

        if a_bufs.len() < 2 {
            let turn = if unroll > 1 {
                let n = self.const_index(block, unroll)?;
                self.muli(block, st, n)?
            } else {
                st
            };
            // The first barrier publishes the staging. The second retires
            // this iteration's reads before the next overwrites them.
            let one = |cg: &mut Self, body: &Block<'c>, kt, accs: &[Value<'c, 'c>]| {
                stage(cg, body, kt, &a_bufs[0], &b_bufs[0])?;
                cg.barrier(body)?;
                let next = mac(cg, body, &a_bufs[0], &b_bufs[0], accs)?;
                cg.barrier(body)?;
                Ok(next)
            };
            return self.carry_loop(block, lo, hi, turn, regs, |cg, body, kt, accs| {
                let mut accs = one(cg, body, kt, accs)?;
                let mut at = kt;
                for _ in 1..unroll {
                    at = cg.addi(body, at, st)?;
                    accs = cg.guarded_iteration(body, at, hi, &accs, |cg, then, accs| one(cg, then, at, accs))?;
                }
                Ok(accs)
            });
        }

        // Prologue: stage lo into the first pair, then a turn of halves.
        stage(self, block, lo, &a_bufs[0], &b_bufs[0])?;
        self.barrier(block)?;
        let halves = unroll + unroll % 2;
        let n = self.const_index(block, halves)?;
        let turn = self.muli(block, st, n)?;
        self.carry_loop(block, lo, hi, turn, regs, |cg, body, kt, accs| {
            let pair = |i: i64| {
                let i = (i % 2) as usize;
                (&a_bufs[i], &b_bufs[i])
            };
            let mut at = cg.addi(body, kt, st)?;
            let mut accs = half(cg, body, at, pair(0), pair(1), accs)?;
            // Half i runs only when iteration kt + i * st exists.
            for i in 1..halves {
                if i > 1 {
                    at = cg.addi(body, at, st)?;
                }
                accs = cg.guarded_iteration(body, at, hi, &accs, |cg, then, accs| {
                    let prefetch = cg.addi(then, at, st)?;
                    half(cg, then, prefetch, pair(i), pair(i + 1), accs)
                })?;
            }
            Ok(accs)
        })
    }

    /// Emits `if at < hi { f(accs) } else { accs }` and returns its results.
    /// The condition is CTA-uniform, as [`Self::guarded_prefetch`] says.
    fn guarded_iteration(
        &mut self,
        block: &Block<'c>,
        at: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
        f: impl FnOnce(&mut Self, &Block<'c>, &[Value<'c, 'c>]) -> Result<Vec<Value<'c, 'c>>>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let more = self.push(block, arith::cmpi(self.ctx, arith::CmpiPredicate::Slt, at, hi, self.loc))?;
        let then_block = Block::new(&[]);
        let ran = f(self, &then_block, accs)?;
        then_block.append_operation(scf::r#yield(&ran, self.loc));
        let then_region = Region::new();
        then_region.append_block(then_block);

        let else_block = Block::new(&[]);
        else_block.append_operation(scf::r#yield(accs, self.loc));
        let else_region = Region::new();
        else_region.append_block(else_block);

        let types: Vec<Type<'c>> = ran.iter().map(|v| v.r#type()).collect();
        let op = block.append_operation(scf::r#if(more, &types, then_region, else_region, self.loc));
        (0..ran.len()).map(|i| Ok(detach(op.result(i)?.into()))).collect()
    }

    /// Applies precomputed GEMM scaling to one accumulator value: alpha*acc +
    /// beta*prev, alpha*acc, or just acc. The prior C value is loaded only
    /// when beta is present. Works on scalars and vectors alike.
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
    /// Returns (tid, w, wt, m0, n0), where w is the warp size and wt the
    /// clamped warp id.
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
