// Moving contraction operands into shared memory or registers, including the
// double-buffered forms, and draining results out through a slab.

use super::*;

impl<'c> Codegen<'c> {
    /// Picks the f16 staging strategy for a tensor-core k-loop, as
    /// (stage_async, reg_stage). Both need f16 operands. `stage_async` uses
    /// cp.async. `reg_stage` is the fallback without cp.async and needs tiles
    /// that divide evenly across the CTA.
    pub(in crate::codegen) fn staging_mode(
        &self,
        src: &GemmSource<'_>,
        m: i64,
        kk: i64,
        n: i64,
    ) -> (bool, bool) {
        let (a_elem, b_elem) = self.gemm_operand_elems(src);
        let both_f16 = a_elem == Some(self.f16_t) && b_elem == Some(self.f16_t);
        let stage_async = self.has_cp_async() && both_f16; // TODO(joa): probably too conservative
        let reg_stage = !self.has_cp_async()
            && both_f16
            && self.reg_stage_divides(m, kk)
            && self.reg_stage_divides(kk, n);
        (stage_async, reg_stage)
    }

    pub(in crate::codegen) fn alloc_staging_pairs(
        &mut self,
        pairs: usize,
        mut make: impl FnMut(&mut Self) -> Result<(MemVal<'c>, MemVal<'c>)>,
    ) -> Result<(Vec<MemVal<'c>>, Vec<MemVal<'c>>)> {
        let mut a_bufs = Vec::with_capacity(pairs);
        let mut b_bufs = Vec::with_capacity(pairs);
        for _ in 0..pairs {
            let (a, b) = make(self)?;
            a_bufs.push(a);
            b_bufs.push(b);
        }
        Ok((a_bufs, b_bufs))
    }

    /// Stages one iteration's a (k-major) and b tiles into shared, without a
    /// barrier. With `async_copy` the copies are cp.async, and the caller
    /// creates the group and waits on it.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn fused_stage(
        &mut self,
        block: &Block<'c>,
        src: &GemmSource<'_>,
        kt: Value<'c, 'c>,
        a_buf: &MemVal<'c>,
        b_buf: &MemVal<'c>,
        async_copy: bool,
    ) -> Result<()> {
        let (a_src, b_src) = self.gemm_operands(block, src, kt)?;

        self.tile_copy_transposed(block, &a_src, a_buf, async_copy)?;
        self.tile_copy(block, &b_src, b_buf, false, async_copy)
    }

    /// The preheader-staged f16 buffer for a dot operand, if an enclosing
    /// loop hoisted it (see `ir/build/hoist.rs`). The caller then skips both
    /// the in-loop staging and the release.
    pub(in crate::codegen) fn hoisted_stage(&self, src: &MemVal<'c>) -> Option<MemVal<'c>> {
        self.hoisted_stages
            .iter()
            .rev()
            .flatten()
            .find(|(v, _)| *v == src.mem)
            .map(|(_, buf)| buf.clone())
    }

    /// Emits `if guard_iv < hi { prefetch(...) }`, guarding the prefetch of
    /// one more iteration.
    ///
    /// The condition is CTA-uniform, so a barrier inside would be safe. That
    /// holds because every loop bound comes from block-uniform values:
    /// literals, `program_id`, `memref.dim`, the zero return of
    /// `warp_partial` and `grid_barrier`, and arithmetic over those.
    /// `codegen::tests::pipeline::atomic_add_cannot_reach_a_loop_bound` checks
    /// this.
    pub(in crate::codegen) fn guarded_prefetch(
        &mut self,
        block: &Block<'c>,
        guard_iv: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        prefetch: impl FnOnce(&mut Self, &Block<'c>) -> Result<()>,
    ) -> Result<()> {
        let more = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Slt, guard_iv, hi, self.loc),
        )?;
        let then_block = Block::new(&[]);
        prefetch(self, &then_block)?;
        then_block.append_operation(scf::r#yield(&[], self.loc));
        let then_region = Region::new();
        then_region.append_block(then_block);
        block.append_operation(scf::r#if(more, &[], then_region, Region::new(), self.loc));
        Ok(())
    }

    /// The staged f16 shared buffer for one tile-dot operand, and whether it
    /// was hoisted.
    ///
    /// A hoisted operand reuses the preheader copy, and the caller must not
    /// release it. Otherwise a fresh buffer is staged here, without a barrier.
    pub(in crate::codegen) fn dot_stage(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        shape: &[i64],
        swizzled: bool,
    ) -> Result<(MemVal<'c>, bool)> {
        if let Some(buf) = self.hoisted_stage(src) {
            return Ok((buf, true));
        }
        let buf = if swizzled {
            self.alloc_tile_swizzled(block, self.f16_t, shape)?
        } else {
            self.alloc_tile_shaped(block, self.f16_t, shape)?
        };
        self.stage_to_f16(block, src, &buf, false)?;
        Ok((buf, false))
    }

    /// One unrolled half of a pipelined loop.
    ///
    /// Prefetches iteration `prefetch_iv` under a guard, runs `mac` on the
    /// resident buffers, then waits and barriers. The barrier publishes the
    /// prefetch and retires the resident reads before the next half
    /// overwrites them.
    pub(super) fn pipelined_half(
        &mut self,
        block: &Block<'c>,
        prefetch_iv: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        use_async: bool,
        stage: impl FnOnce(&mut Self, &Block<'c>) -> Result<()>,
        mac: impl FnOnce(&mut Self) -> Result<Vec<Value<'c, 'c>>>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        self.guarded_prefetch(block, prefetch_iv, hi, stage)?;

        // The group sits outside the guard, since tokens cannot leave an
        // scf.if region. Waiting on an empty group is a no-op.
        let group = if use_async {
            Some(self.async_create_group(block)?)
        } else {
            None
        };

        let next = mac(self)?;

        if let Some(group) = group {
            self.async_wait(block, group)?;
        }
        self.barrier(block)?;
        Ok(next)
    }

    /// The vector path's pipelined half: prefetch into dst, register MAC from
    /// cur. See [`Self::pipelined_half`].
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn fused_half(
        &mut self,
        block: &Block<'c>,
        src: &GemmSource<'_>,
        prefetch_iv: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        cur: (&MemVal<'c>, &MemVal<'c>),
        dst: (&MemVal<'c>, &MemVal<'c>),
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let use_async = self.has_cp_async();
        self.pipelined_half(
            block,
            prefetch_iv,
            hi,
            use_async,
            |cg, then| cg.fused_stage(then, src, prefetch_iv, dst.0, dst.1, use_async),
            |cg| cg.register_mac(block, cur.0, cur.1, dims, m0, n0, accs),
        )
    }

    /// Stages an operand into an f16 shared buffer for the tensor cores.
    ///
    /// An f32 source is rounded with [`Self::tile_copy_f16`], an f16 source is
    /// copied as is, and anything else is an error. `async_copy` applies only
    /// to the f16 copy, since rounding cannot be a raw cp.async. The caller
    /// adds the barrier.
    pub(in crate::codegen) fn stage_to_f16(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        dst: &MemVal<'c>,
        async_copy: bool,
    ) -> Result<()> {
        if dst.elem != self.f16_t {
            bail!("WMMA staging destination must be f16");
        }

        if src.elem == self.f32_t {
            self.tile_copy_f16(block, src, dst)
        } else if src.elem == self.f16_t {
            self.tile_copy(block, src, dst, false, async_copy)
        } else {
            bail!("WMMA operands must be f16 or f32, got {}", src.elem)
        }
    }

    /// dst[...] = f16(src[...]), staging an f32 slice into an f16 shared
    /// buffer. Uses 4-wide vectors when the source rows are provably aligned.
    /// The caller adds the barrier.
    pub(in crate::codegen) fn tile_copy_f16(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        dst: &MemVal<'c>,
    ) -> Result<()> {
        if src.elem != self.f32_t || dst.elem != self.f16_t {
            bail!("f16 staging needs an f32 source and an f16 destination");
        }

        let last = *dst.shape.last().expect("tile values are not rank-0");
        let width = if src.vectorizes(4) && last != DYN && last % 4 == 0 {
            4
        } else {
            1
        };

        let v32_t = Type::vector(&[4], self.f32_t);
        let v16_t = Type::vector(&[4], self.f16_t);

        self.distribute(block, dst, width, false, |cg, blk, idx| {
            // Only the destination is swizzled.
            let didx = cg.swizzled_index(blk, dst, idx)?;

            if width > 1 {
                let v = cg.vec_load(blk, src.mem, idx, v32_t)?;
                let h = cg.truncf(blk, v, v16_t)?;
                cg.vec_store_al(blk, h, dst.mem, &didx, 8)?;
            } else {
                let e = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
                let h = cg.truncf(blk, e, cg.f16_t)?;
                blk.append_operation(memref::store(h, dst.mem, &didx, cg.loc));
            }

            Ok(())
        })
    }

    pub(in crate::codegen) fn truncf(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("arith.truncf", self.loc)
                .add_operands(&[value])
                .add_results(&[t])
                .build()?,
        )
    }

    /// The lane's read slice of its warp's 16x16 slab, see [`SlabDrain`].
    /// Returns (srow, lrow, lcol) for the warp tile at slab row `slab0`.
    pub(super) fn slab_lane_slice(
        &self,
        block: &Block<'c>,
        slab0: Value<'c, 'c>,
        lane: Value<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>, Value<'c, 'c>)> {
        let two = self.const_index(block, 2)?;
        let lrow = self.divui(block, lane, two)?;
        let lhalf = self.remui(block, lane, two)?;
        let eight = self.const_index(block, 8)?;
        let lcol = self.muli(block, lhalf, eight)?;
        let srow = self.addi(block, slab0, lrow)?;
        Ok((srow, lrow, lcol))
    }

    /// Copies the warp's slab out to C for fragment (fi, fj), computing
    /// alpha*acc + beta*prev. The caller adds the barriers around it.
    pub(super) fn drain_slab_tile(
        &mut self,
        block: &Block<'c>,
        d: &SlabDrain<'c>,
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        fi: i64,
        fj: i64,
    ) -> Result<()> {
        let (mb, nb) = self.frag_origin(block, m0, n0, fi, fj)?;
        let mi = self.addi(block, mb, d.lrow)?;
        let nbl = self.addi(block, nb, d.lcol)?;
        let row_t = Type::vector(&[4], self.f32_t);
        let row_f16_t = Type::vector(&[4], self.f16_t);

        for h in 0..2 {
            let c_h = self.const_index(block, h * 4)?;
            let sc = self.addi(block, d.lcol, c_h)?;
            let nj = self.addi(block, nbl, c_h)?;

            match d.mode {
                DrainMode::VecF32 => {
                    let v = self.vec_load(block, d.slab.mem, &[d.srow, sc], row_t)?;
                    let out_v = self.apply_scaling(block, v, d.alpha, d.beta, |cg| {
                        cg.vec_load(block, d.view.mem, &[mi, nj], row_t)
                    })?;

                    self.vec_store(block, out_v, d.view.mem, &[mi, nj])?;
                }
                DrainMode::VecF16 => {
                    let raw = self.vec_load_al(block, d.slab.mem, &[d.srow, sc], row_f16_t, 8)?;
                    let acc_v = self.vec_extf(block, raw, row_t)?;
                    let out_v = self.apply_scaling(block, acc_v, d.alpha, d.beta, |cg| {
                        let c = cg.vec_load_al(block, d.view.mem, &[mi, nj], row_f16_t, 8)?;
                        cg.vec_extf(block, c, row_t)
                    })?;
                    let out_v = self.vec_truncf(block, out_v, row_f16_t)?;

                    self.vec_store_al(block, out_v, d.view.mem, &[mi, nj], 8)?;
                }
                DrainMode::Scalar => {
                    for e in 0..4 {
                        let c_e = self.const_index(block, e)?;
                        let sce = self.addi(block, sc, c_e)?;
                        let ne = self.addi(block, nj, c_e)?;
                        let x = self.load_as(block, d.slab.mem, &[d.srow, sce], self.f32_t)?;
                        let out_e = self.apply_scaling(block, x, d.alpha, d.beta, |cg| {
                            cg.load_as(block, d.view.mem, &[mi, ne], cg.f32_t)
                        })?;
                        let out_e = self.coerce(block, out_e, d.view.elem)?;

                        block.append_operation(memref::store(
                            out_e,
                            d.view.mem,
                            &[mi, ne],
                            self.loc,
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    pub(in crate::codegen) fn reg_stage_divides(&self, rows: i64, cols: i64) -> bool {
        let lane_elems = self.cta_threads * HALF_VEC;
        cols % HALF_VEC == 0 && rows * cols >= lane_elems && (rows * cols) % lane_elems == 0
    }
}
