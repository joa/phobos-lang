// The register-accumulator matmul, for shapes with no tensor core path.

use super::*;

impl<'c> Codegen<'c> {
    /// The lane's whole accumulator is one tm x tn f32 vector.
    pub(super) fn reg_seed(
        &mut self,
        block: &Block<'c>,
        plan: &GemmPlan<'c>,
        init: Value<'c, 'c>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let GemmPath::Reg { tm, tn, .. } = plan.path else {
            bail!("reg_seed on a tensor-core plan");
        };
        let init = self.coerce(block, init, self.f32_t)?;
        let acc_t = Type::vector(&[tm as u64, tn as u64], self.f32_t);
        Ok(vec![self.vec_broadcast(block, init, acc_t)?])
    }

    pub(super) fn reg_loop(
        &mut self,
        block: &Block<'c>,
        mut acc: GemmAcc<'c>,
        (lo, hi, st): (Value<'c, 'c>, Value<'c, 'c>, Value<'c, 'c>),
        src: &GemmSource<'_>,
    ) -> Result<GemmAcc<'c>> {
        let GemmPlan { m, n, kk, .. } = acc.plan;
        let GemmPath::Reg { tm, tn, lm, ln } = acc.plan.path else {
            bail!("reg_loop on a tensor-core plan");
        };
        let (tiles_m, tiles_n) = (m / tm, n / tn);

        // The lane's sub-tile origin: the warp's block origin (surplus warps
        // clamped onto the last block, as in tile_matmul) plus the lane's
        // position on the lm x ln lane grid of tm x tn sub-tiles.
        let origin = self.warp_block_origin(block, tiles_m / lm, tiles_n / ln, lm * tm, ln * tn)?;
        let (tid, w, wt, wm0, wn0) = origin;
        let lane = self.remui(block, tid, w)?;
        let ln_v = self.const_index(block, ln)?;
        let lane_m = self.divui(block, lane, ln_v)?;
        let lane_n = self.remui(block, lane, ln_v)?;
        let tm_v = self.const_index(block, tm)?;
        let tn_v = self.const_index(block, tn)?;
        let off_m = self.muli(block, lane_m, tm_v)?;
        let off_n = self.muli(block, lane_n, tn_v)?;
        let m0 = self.addi(block, wm0, off_m)?;
        let n0 = self.addi(block, wn0, off_n)?;

        // staging buffers; a k-major (kk x m).
        let pairs = self.staging_pairs();
        let (a_bufs, b_bufs) = self.alloc_staging_pairs(pairs, |cg| {
            Ok((
                cg.alloc_tile_shaped(block, cg.f32_t, &[kk, m])?,
                cg.alloc_tile_shaped(block, cg.f32_t, &[kk, n])?,
            ))
        })?;

        let dims = (kk, tm, tn);
        let finals = self.matmul_kloop(
            block,
            (lo, hi, st),
            &acc.regs,
            &a_bufs,
            &b_bufs,
            |cg, body, kt, a, b| cg.fused_stage(body, src, kt, a, b, false),
            |cg, body, piv, cur, dst, accs| {
                cg.fused_half(body, src, piv, hi, cur, dst, dims, m0, n0, accs)
            },
            |cg, body, a, b, accs| cg.register_mac(body, a, b, dims, m0, n0, accs),
        )?;
        acc.regs = finals;
        // The lane's own origin is what the drain indexes by.
        acc.origin = Some((tid, w, wt, m0, n0));
        Ok(acc)
    }

    /// Each lane writes its finished sub-tile straight to C, optionally
    /// applying alpha*acc + beta*prev_load.
    pub(super) fn reg_store(
        &mut self,
        block: &Block<'c>,
        acc: GemmAcc<'c>,
        view: MemVal<'c>,
        alpha: Option<GemmScale<'c>>,
        beta: Option<GemmScale<'c>>,
    ) -> Result<()> {
        let GemmPath::Reg { tm, tn, .. } = acc.plan.path else {
            bail!("reg_store on a tensor-core plan");
        };
        let (_, _, _, m0, n0) = Self::gemm_finals(&acc)?;
        let row_t = Type::vector(&[tn as u64], self.f32_t);

        // pre-compute alpha/beta broadcasts once
        let (alpha_row, beta_row) =
            self.epilogue_scaling(block, alpha, beta, row_t, view.vectorizes(4))?;

        for i in 0..tm {
            let ci = self.const_index(block, i)?;
            let mi = self.addi(block, m0, ci)?;
            let row = self.vec_extract(block, acc.regs[0], &[i], row_t)?;

            if view.vectorizes(4) {
                let out_row = self.apply_scaling(block, row, alpha_row, beta_row, |cg| {
                    cg.vec_load(block, view.mem, &[mi, n0], row_t)
                })?;

                self.vec_store(block, out_row, view.mem, &[mi, n0])?;
            } else {
                for j in 0..tn {
                    let cj = self.const_index(block, j)?;
                    let nj = self.addi(block, n0, cj)?;
                    let e = self.vec_extract(block, row, &[j], self.f32_t)?;
                    let out_e = self.apply_scaling(block, e, alpha_row, beta_row, |cg| {
                        cg.push(block, memref::load(view.mem, &[mi, nj], cg.loc))
                    })?;

                    block.append_operation(memref::store(out_e, view.mem, &[mi, nj], self.loc));
                }
            }
        }
        Ok(())
    }

    /// The fused k-loop, done as a vector contraction. The lane's accumulator rides the loop as
    /// one vector<TMxTNxf32> iter_arg. Each iteration grabs a CxTM and a CxTN chunk and folds
    /// them in with a single vector.contract over k. C is the largest of 4, 2, 1 that divides
    /// TILE_K; a_t and b_sh are both k-major, so the chunk rows are contiguous loads.
    ///
    /// Doing C k-steps per iteration cuts the loop overhead by C. The contract lowers to
    /// broadcast plus vector.fma rank-1 updates, the same fma.rn stream as the scalar form
    /// without the extract/insert noise in the IR.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn register_mac(
        &mut self,
        block: &Block<'c>,
        a_t: &MemVal<'c>,
        b_sh: &MemVal<'c>,
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let (kk, tm, tn) = dims;
        let chunk = [4, 2, 1]
            .into_iter()
            .find(|c| kk % c == 0)
            .expect("1 divides everything");

        let a_row_t = Type::vector(&[tm as u64], self.f32_t);
        let b_row_t = Type::vector(&[tn as u64], self.f32_t);
        let lhs_t = Type::vector(&[chunk as u64, tm as u64], self.f32_t);
        let rhs_t = Type::vector(&[chunk as u64, tn as u64], self.f32_t);
        let lo = self.const_index(block, 0)?;
        let hi = self.const_index(block, kk)?;
        let st = self.const_index(block, chunk)?;

        self.carry_loop(block, lo, hi, st, accs, |cg, lblk, k, accs| {
            let zero = cg.zero_scalar(lblk, cg.f32_t)?;
            let mut lhs = cg.vec_broadcast(lblk, zero, lhs_t)?;
            let mut rhs = cg.vec_broadcast(lblk, zero, rhs_t)?;

            for j in 0..chunk {
                let c = cg.const_index(lblk, j)?;
                let kj = cg.addi(lblk, k, c)?;

                let a_row = cg.vec_load(lblk, a_t.mem, &[kj, m0], a_row_t)?;
                lhs = cg.vec_insert(lblk, a_row, lhs, &[j])?;

                let b_row = cg.vec_load(lblk, b_sh.mem, &[kj, n0], b_row_t)?;
                rhs = cg.vec_insert(lblk, b_row, rhs, &[j])?;
            }

            Ok(vec![cg.vec_contract(lblk, lhs, rhs, accs[0], true)?])
        })
    }
}
