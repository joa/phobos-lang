// The register-accumulator matmul, for shapes with no tensor core path.

use super::*;

impl<'c> Codegen<'c> {
    pub(in crate::codegen) fn emit_register_matmul(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
    ) -> Result<()> {
        let (shape, kk) = self.fusion_dims(p)?;
        let (m, n) = (shape[0], shape[1]);

        if self.has_wmma() && self.wmma_plan(m, n, kk).is_some() {
            return if self.has_mma_sync() {
                self.emit_mma_sync_matmul(block, p)
            } else {
                self.emit_wmma_matmul(block, p)
            };
        }

        let (tm, tn) = self.sub_tile(m, n);
        let (tiles_m, tiles_n) = (m / tm, n / tn);
        let (lm, ln) = Self::lane_grid(tiles_m, tiles_n, tm, tn)
            .ok_or_else(|| anyhow!("matmul fusion without a lane grid"))?;

        let init = self.emit_scalar(block, p.init)?;
        let init = self.coerce(block, init, self.f32_t)?;

        // the lane's whole accumulator is one TMxTN vector
        let acc_t = Type::vector(&[tm as u64, tn as u64], self.f32_t);
        let regs = vec![self.vec_broadcast(block, init, acc_t)?];

        let (lo, hi, st, iv_div) = self.loop_bounds(block, p.start, p.end, p.step)?;

        // The lane's sub-tile origin: the warp's block origin (surplus warps
        // clamped onto the last block, as in tile_matmul) plus the lane's
        // position on the lm x ln lane grid of tm x tn sub-tiles.
        let (tid, w, _, wm0, wn0) =
            self.warp_block_origin(block, tiles_m / lm, tiles_n / ln, lm * tm, ln * tn)?;
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
        let pairs = if self.pipeline { 2 } else { 1 };
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
            &regs,
            &a_bufs,
            &b_bufs,
            |cg, body, kt, a, b| cg.fused_stage(body, p, kt, iv_div, a, b, false),
            |cg, body, piv, cur, dst, accs| {
                cg.fused_half(body, p, iv_div, piv, hi, cur, dst, dims, m0, n0, accs)
            },
            |cg, body, a, b, accs| cg.register_mac(body, a, b, dims, m0, n0, accs),
        )?;

        // epilogue: each lane writes its finished sub-tile straight to C,
        // optionally applying alpha*acc + beta*prev_load
        let view = self.epilogue_view(block, p, &shape)?;
        let row_t = Type::vector(&[tn as u64], self.f32_t);

        // pre-compute alpha/beta broadcasts once
        let (alpha_row, beta_row) = self.epilogue_scaling(block, p, row_t, view.vectorizes(4))?;

        for i in 0..tm {
            let ci = self.const_index(block, i)?;
            let mi = self.addi(block, m0, ci)?;
            let row = self.vec_extract(block, finals[0], &[i], row_t)?;

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

    /// Evaluates one staged operand slice with the fusion's k-loop iv bound
    /// to kt.
    pub(super) fn emit_kt_slice(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        slice: &Expr,
        kt: Value<'c, 'c>,
        iv_div: i64,
    ) -> Result<MemVal<'c>> {
        self.scopes.push(HashMap::new());
        self.bind(
            p.kt,
            Binding::Let {
                value: kt,
                div: iv_div,
            },
        );
        let src = self.emit_expr(block, slice);
        self.scopes.pop();

        match src? {
            Rv::Tile(mv) => Ok(mv),
            Rv::Scalar(_) => bail!("staged value must be a tensor slice"),
        }
    }

    /// The fused k-loop, done as a vector contraction. The lane's accumulator
    /// rides the loop as one vector<TMxTNxf32> iter_arg. Each iteration grabs a
    /// CxTM and a CxTN chunk and folds them in with a single vector.contract
    /// over k. C is the largest of 4, 2, 1 that divides TILE_K; a_t and b_sh
    /// are both k-major, so the chunk rows are contiguous loads.
    ///
    /// Doing C k-steps per iteration cuts the loop overhead by C. The contract
    /// lowers to broadcast + vector.fma rank-1 updates, which on NVPTX is the
    /// same fma.rn stream as the scalar form, just without the extract/insert
    /// noise in the IR.
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
