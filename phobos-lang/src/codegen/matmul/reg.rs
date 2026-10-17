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
        unroll: i64,
        src: &GemmSource<'_>,
    ) -> Result<GemmAcc<'c>> {
        let GemmPlan { m, n, kk, .. } = acc.plan;
        let GemmPath::Reg { tm, tn, lm, ln } = acc.plan.path else {
            bail!("reg_loop on a tensor-core plan");
        };
        let (tiles_m, tiles_n) = (m / tm, n / tn);

        // The lane's sub-tile origin is the warp's block origin plus the
        // lane's place on the lm x ln grid of tm x tn sub-tiles. Surplus
        // warps clamp onto the last block.
        let origin = self.warp_block_origin(block, tiles_m / lm, tiles_n / ln, lm * tm, ln * tn)?;
        let (tid, w, wt, wm0, wn0) = origin;
        let lane = self.remui(block, tid, w)?;
        let ln_v = self.const_index(block, ln)?;
        let lane_m = self.divui(block, lane, ln_v)?;
        let lane_n = self.remui(block, lane, ln_v)?;
        let (group, split) = Self::reg_col_groups(tn, ln);
        let tm_v = self.const_index(block, tm)?;
        let tn_v = self.const_index(block, group)?;
        let off_m = self.muli(block, lane_m, tm_v)?;
        let off_n = self.muli(block, lane_n, tn_v)?;
        let m0 = self.addi(block, wm0, off_m)?;
        let n0 = self.addi(block, wn0, off_n)?;

        // Staging buffers, with a k-major (kk x m).
        let pairs = self.staging_pairs();
        let (a_bufs, b_bufs) = self.alloc_staging_pairs(pairs, |cg| {
            Ok((
                cg.alloc_tile_shaped(block, cg.f32_t, &[kk, m])?,
                cg.alloc_tile_shaped(block, cg.f32_t, &[kk, n])?,
            ))
        })?;

        let dims = (kk, tm, tn, split);
        let finals = self.matmul_kloop(
            block,
            (lo, hi, st),
            unroll,
            &acc.regs,
            &a_bufs,
            &b_bufs,
            |cg, body, kt, a, b| cg.fused_stage(body, src, kt, a, b, false),
            |cg, body, piv, cur, dst, accs| {
                cg.fused_half(body, src, piv, (hi, st), cur, dst, dims, m0, n0, accs)
            },
            |cg, body, a, b, accs| cg.register_mac(body, a, b, dims, m0, n0, accs),
        )?;
        acc.regs = finals;
        // The drain indexes by the lane's own origin.
        acc.origin = Some((tid, w, wt, m0, n0));
        Ok(acc)
    }

    /// A lane's columns of a `tn`-wide sub-tile on a lane grid `ln` wide, as
    /// (group width, distance between groups). An eight-wide row is two
    /// groups of four, `4 * ln` apart, so at one k the warp's lanes read B in
    /// consecutive 16-byte runs, which fall in distinct banks; one run of
    /// eight a lane would put lanes `i` and `i + 4` in the same banks. A
    /// narrower row is one group, with distance 0.
    fn reg_col_groups(tn: i64, ln: i64) -> (i64, i64) {
        if tn > 4 { (4, 4 * ln) } else { (tn, 0) }
    }

    /// Each lane writes its finished sub-tile straight to C, optionally
    /// computing alpha*acc + beta*prev.
    ///
    /// A surplus warp, clamped onto the last block by
    /// [`Self::warp_block_origin`], holds the same sub-tiles as that block's
    /// own warp and writes nothing: two read-modify-writes of one C would
    /// apply beta twice whenever one lands before the other reads.
    pub(super) fn reg_store(
        &mut self,
        block: &Block<'c>,
        acc: GemmAcc<'c>,
        view: MemVal<'c>,
        alpha: Option<GemmScale<'c>>,
        beta: Option<GemmScale<'c>>,
    ) -> Result<()> {
        let GemmPath::Reg { tm, tn, lm, ln } = acc.plan.path else {
            bail!("reg_store on a tensor-core plan");
        };
        let (tid, w, wt, m0, n0) = Self::gemm_finals(&acc)?;
        let row_t = Type::vector(&[tn as u64], self.f32_t);
        let (group, split) = Self::reg_col_groups(tn, ln);
        let group_t = Type::vector(&[group as u64], self.f32_t);

        let (alpha_row, beta_row) =
            self.epilogue_scaling(block, alpha, beta, group_t, view.vectorizes(4))?;

        // Warps that own a block, of the CTA's.
        let owners = (acc.plan.m / tm / lm) * (acc.plan.n / tn / ln);
        let guarded = Block::new(&[]);
        let into = if owners < self.cta_threads / 32 { &guarded } else { block };
        // Each group of the row's elements with the column it starts at.
        let mut groups = vec![(0, n0)];
        if split > 0 {
            let c = self.const_index(into, split)?;
            groups.push((group, self.addi(into, n0, c)?));
        }
        for i in 0..tm {
            let ci = self.const_index(into, i)?;
            let mi = self.addi(into, m0, ci)?;
            let row = self.vec_extract(into, acc.regs[0], &[i], row_t)?;

            for &(first, col) in &groups {
                let part = if split > 0 {
                    let lanes: Vec<i64> = (first..first + group).collect();
                    self.vec_shuffle(into, row, row, &lanes, group_t)?
                } else {
                    row
                };
                if view.vectorizes(4) {
                    let out = self.apply_scaling(into, part, alpha_row, beta_row, |cg| {
                        cg.vec_load(into, view.mem, &[mi, col], group_t)
                    })?;

                    self.vec_store(into, out, view.mem, &[mi, col])?;
                } else {
                    for j in 0..group {
                        let cj = self.const_index(into, j)?;
                        let nj = self.addi(into, col, cj)?;
                        let e = self.vec_extract(into, part, &[j], self.f32_t)?;
                        let out_e = self.apply_scaling(into, e, alpha_row, beta_row, |cg| {
                            cg.push(into, memref::load(view.mem, &[mi, nj], cg.loc))
                        })?;

                        into.append_operation(memref::store(out_e, view.mem, &[mi, nj], self.loc));
                    }
                }
            }
        }
        if owners < self.cta_threads / 32 {
            let warp = self.divui(block, tid, w)?;
            let owner = self.push(block, arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, warp, wt, self.loc))?;
            guarded.append_operation(scf::r#yield(&[], self.loc));
            let region = Region::new();
            region.append_block(guarded);
            block.append_operation(scf::r#if(owner, &[], region, Region::new(), self.loc));
        }
        Ok(())
    }

    /// The fused k-loop body as a vector contraction.
    ///
    /// The lane's accumulator is one vector<TMxTNxf32> iter_arg. Each
    /// iteration loads a C x TM and a C x TN chunk and folds them in with one
    /// vector.contract over k. C is the largest of 4, 2, 1 that divides kk.
    /// Both buffers are k-major, so the chunk rows are contiguous loads.
    /// `dims` is (kk, tm, tn, split): a nonzero split reads each B row as
    /// two halves that far apart, see [`Self::reg_col_groups`].
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn register_mac(
        &mut self,
        block: &Block<'c>,
        a_t: &MemVal<'c>,
        b_sh: &MemVal<'c>,
        dims: (i64, i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let (kk, tm, tn, split) = dims;
        let n1 = if split > 0 {
            let c = self.const_index(block, split)?;
            Some(self.addi(block, n0, c)?)
        } else {
            None
        };
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

                let b_row = match n1 {
                    Some(n1) => {
                        let half_t = Type::vector(&[tn as u64 / 2], cg.f32_t);
                        let lo = cg.vec_load(lblk, b_sh.mem, &[kj, n0], half_t)?;
                        let hi = cg.vec_load(lblk, b_sh.mem, &[kj, n1], half_t)?;
                        let lanes: Vec<i64> = (0..tn).collect();
                        cg.vec_shuffle(lblk, lo, hi, &lanes, b_row_t)?
                    }
                    None => cg.vec_load(lblk, b_sh.mem, &[kj, n0], b_row_t)?,
                };
                rhs = cg.vec_insert(lblk, b_row, rhs, &[j])?;
            }

            Ok(vec![cg.vec_contract(lblk, lhs, rhs, accs[0], true)?])
        })
    }
}
