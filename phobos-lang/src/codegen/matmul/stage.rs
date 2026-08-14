// Getting operands into shared memory or registers for a contraction,
// including the double-buffered and slab-drained forms.

use super::*;

impl<'c> Codegen<'c> {
    /// Picks the f16 staging strategy for a tensor-core k-loop. Returns
    /// (stage_async, reg_stage): the first is true when cp.async is available
    /// for f16 operands, the second when register staging works (the tile has
    /// to divide evenly by the CTA).
    pub(in crate::codegen) fn staging_mode(
        &self,
        p: &MatmulFusion<'_>,
        m: i64,
        kk: i64,
        n: i64,
    ) -> (bool, bool) {
        let both_f16 = self.slice_tensor_elem(p.a_slice) == Some(self.f16_t)
            && self.slice_tensor_elem(p.b_slice) == Some(self.f16_t);
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
    /// barrier. With async_copy the transfers are cp.async and the caller owns
    /// the group and wait.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn fused_stage(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        kt: Value<'c, 'c>,
        iv_div: i64,
        a_buf: &MemVal<'c>,
        b_buf: &MemVal<'c>,
        async_copy: bool,
    ) -> Result<()> {
        let a_src = self.emit_kt_slice(block, p, p.a_slice, kt, iv_div)?;
        let b_src = self.emit_kt_slice(block, p, p.b_slice, kt, iv_div)?;

        self.tile_copy_transposed(block, &a_src, a_buf, async_copy)?;
        self.tile_copy(block, &b_src, b_buf, false, async_copy)
    }

    /// Returns the staged f16 shared buffer for one tile-dot operand: the
    /// preheader copy when an enclosing loop hoisted this operand (see
    /// codegen/hoist.rs), else a fresh pooled buffer staged here without a
    /// barrier. The flag is true for the hoisted case, where the caller
    /// must skip the release; the loop epilogue owns that buffer.
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

    /// One unrolled half of a pipelined loop: a guarded, barrier-free prefetch
    /// of iteration prefetch_iv, accumulation from the resident buffers via
    /// mac, then a closing wait and barrier. The barrier publishes the prefetch
    /// and retires the resident reads before the next half overwrites them.
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

        // cp.async group outside the guard (an empty group is a no-op wait;
        // tokens can't cross scf.if regions).
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

    /// The vector path's pipelined half: prefetch into dst, register-MAC from
    /// cur (see [`Self::pipelined_half`]).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn fused_half(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        iv_div: i64,
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
            |cg, then| cg.fused_stage(then, p, prefetch_iv, iv_div, dst.0, dst.1, use_async),
            |cg| cg.register_mac(block, cur.0, cur.1, dims, m0, n0, accs),
        )
    }

    /// Stages an operand into an f16 shared buffer for WMMA. An f32 source is
    /// rounded down ([`Self::tile_copy_f16`]), an f16 source is copied as-is,
    /// and anything else is rejected. async_copy only applies to the straight
    /// f16 copy; the f32 round-down can't be a raw cp.async byte transfer. Never
    /// emits a barrier; the caller owns synchronization.
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

    /// dst[...] = f16(src[...]): stages an f32 slice into an f16 shared buffer,
    /// rounding each element once (arith.truncf). Vectorized as 4xf32 loads /
    /// 4xf16 (8-byte) stores when the source rows are provably aligned. Never
    /// emits a barrier; the caller owns sync.
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
            // Read the (unswizzled) source, round, store to the swizzled column.
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

    /// The lane's read slice of its warp's 16x16 slab tile (see
    /// [`SlabDrain`]). Returns (srow, lrow, lcol) for the warp tile at slab
    /// row slab0.
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

    /// Copies the warp's slab out to C for fragment (fi, fj), applying
    /// alpha*acc + beta*prev_load. The caller owns the surrounding barriers.
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
