// `@tensorcore` via the wmma intrinsics: 16x16 fragments, used from
// sm_80 on and as the fallback when mma.sync does not apply.

use super::*;

impl<'c> Codegen<'c> {
    /// Warp grid (wm x wn, with wm*wn the launch ABI's warp count) laid over
    /// the matmul's 16x16-fragment grid. Returns None when the shape doesn't
    /// split into whole fragments owned by whole warps. We pick the
    /// factorization that loads the fewest fragments per k-step (fm + fn for
    /// fm*fn computes), breaking ties toward wider warp blocks since those give
    /// contiguous b reads and epilogue stores.
    pub(in crate::codegen) fn wmma_plan(&self, m: i64, n: i64, kk: i64) -> Option<(i64, i64)> {
        if m == DYN || n == DYN || kk == DYN || m % 16 != 0 || n % 16 != 0 || kk % 16 != 0 {
            return None;
        }

        let warps = self.cta_threads / 32;
        let (gm, gn) = (m / 16, n / 16);

        (1..=warps)
            .filter(|wm| warps % wm == 0)
            .map(|wm| (wm, warps / wm))
            .filter(|&(wm, wn)| gm % wm == 0 && gn % wn == 0)
            .min_by_key(|&(wm, wn)| (gm / wm + gn / wn, gm / wm))
    }

    /// Whether to pad the WMMA staging buffers.
    ///
    /// More padding means a larger CTA footprint, so fewer CTAs fit per SM. We
    /// skip it when kernel registers (@launch) are the limiting factor instead.
    pub(super) fn wmma_should_pad(
        &self,
        m: i64,
        kk: i64,
        n: i64,
        acc_elem: Type<'c>,
        pairs: i64,
    ) -> bool {
        // TODO(joa): @autotune this
        let Some(maxnreg) = self.launch.and_then(|l| l.max_nreg) else {
            return true;
        };

        let sm_bytes = self.isa.smem_per_sm();
        let warps = self.cta_threads / 32;
        let warp_blocks = (self.isa.max_warps_per_sm() / warps).max(1);
        let reg_blocks = (self.isa.regs_per_sm() / (self.cta_threads * maxnreg)).max(1);
        let acc_bytes = if acc_elem == self.f16_t { 2 } else { 4 }; // TODO(joa): support for other data types
        let slab = warps * 16 * 16 * acc_bytes;

        // f16 staging, both operands, doubled under @pipeline; plus the slab
        let smem = |ak: i64, bn: i64| pairs * (m * ak + kk * bn) * 2 + slab;
        let blocks = |bytes: i64| (sm_bytes / bytes).min(warp_blocks).min(reg_blocks);

        blocks(smem(kk + WMMA_SMEM_PAD, n + WMMA_SMEM_PAD)) >= blocks(smem(kk, n))
    }

    pub(in crate::codegen) fn emit_wmma_matmul(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
    ) -> Result<()> {
        let (shape, kk) = self.fusion_dims(p)?;
        let (m, n) = (shape[0], shape[1]);
        let (wm, wn) = self
            .wmma_plan(m, n, kk)
            .ok_or_else(|| anyhow!("wmma matmul without a warp grid"))?;

        // fragments per warp: fmxfn block of the 16x16-fragment grid
        let (fm, fnn) = ((m / 16) / wm, (n / 16) / wn);

        // the warp's accumulators, seeded with the init scalar rounded to the accumulator type
        let acc_elem = self.scalar_type(p.acc_scalar);
        let init = self.emit_scalar(block, p.init)?;
        let init = self.coerce(block, init, acc_elem)?;
        let c_frag_t = self.wmma_c_type(acc_elem)?;
        let mut regs = Vec::with_capacity((fm * fnn) as usize);

        for _ in 0..fm * fnn {
            regs.push(self.wmma_const_frag(block, init, c_frag_t)?);
        }

        // f16 staging, padded against bank conflicts when the CTA budget allows
        let pairs = if self.pipeline_assert { 2 } else { 1 };
        let pad = self.wmma_should_pad(m, kk, n, acc_elem, pairs as i64);
        let alloc = if pad {
            Self::alloc_tile_padded
        } else {
            Self::alloc_tile_shaped
        };

        let (finals, (tid, w, wt, m0, n0)) = self.tc_matmul_kloop(
            block,
            p,
            (m, n, kk),
            (wm, wn),
            &regs,
            false,
            |cg, blk, shape| alloc(cg, blk, cg.f16_t, shape),
        )?;

        // epilogue: each warp drains its fragments through its 16x16 slab,
        // optionally applying alpha*acc + beta*prev_load
        let view = self.epilogue_view(block, p, &shape)?;

        // The slab matches the accumulator fragments' element type; the f32
        // vector drain only applies when both the slab and C are f32.
        let slab = self.alloc_tile_shaped(block, acc_elem, &[(self.cta_threads / 32) * 16, 16])?;
        let slab_f32 = acc_elem == self.f32_t;
        let sixteen = self.const_index(block, 16)?;
        let slab0 = self.muli(block, wt, sixteen)?;
        let zero = self.const_index(block, 0)?;

        let lane = self.remui(block, tid, w)?;
        let (srow, lrow, lcol) = self.slab_lane_slice(block, slab0, lane)?;
        let row_t = Type::vector(&[4], self.f32_t);

        // The 128-bit vector drain needs an f32 slab and a 16B-aligned f32 C
        // row (aligned on its own is element-agnostic: an f16 C is 8B-aligned
        // but takes the scalar, rounding store).
        let vec_drain = slab_f32 && view.elem == self.f32_t && view.vectorizes(4);

        // pre-compute alpha/beta broadcasts once
        let (alpha, beta) = self.epilogue_scaling(block, p, row_t, vec_drain)?;
        let drain = SlabDrain {
            slab: slab.clone(),
            view,
            srow,
            lrow,
            lcol,
            alpha,
            beta,
            mode: if vec_drain {
                DrainMode::VecF32
            } else {
                DrainMode::Scalar
            },
        };

        for fi in 0..fm {
            for fj in 0..fnn {
                let frag = finals[(fi * fnn + fj) as usize];

                self.wmma_store(block, frag, slab.mem, &[slab0, zero], 16)?;

                // publish the slab and read it out before the next iteration overwrites it
                self.barrier(block)?;
                self.drain_slab_tile(block, &drain, m0, n0, fi, fj)?;
                self.barrier(block)?;
            }
        }
        Ok(())
    }

    /// Drives the tensor-core k-loop shared by the WMMA and mma.sync paths:
    /// the loop bounds, the warp's fragment-block origin (surplus warps clamp
    /// onto the last block), the staging pairs (a as [m, kk], b as [kk, n],
    /// via alloc, double-buffered under @pipeline), and [`Self::matmul_kloop`]
    /// with WMMA staging and the tensor-core MAC. Returns the finished
    /// accumulators and the origin (as [`Self::warp_block_origin`]) for the
    /// epilogue.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub(super) fn tc_matmul_kloop(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        (m, n, kk): (i64, i64, i64),
        (wm, wn): (i64, i64),
        regs: &[Value<'c, 'c>],
        mma_sync: bool,
        alloc: impl Fn(&mut Self, &Block<'c>, &[i64]) -> Result<MemVal<'c>>,
    ) -> Result<(
        Vec<Value<'c, 'c>>,
        (
            Value<'c, 'c>,
            Value<'c, 'c>,
            Value<'c, 'c>,
            Value<'c, 'c>,
            Value<'c, 'c>,
        ),
    )> {
        let (fm, fnn) = ((m / 16) / wm, (n / 16) / wn);
        let dims = (kk, fm, fnn);

        let (lo, hi, st, iv_div) = self.loop_bounds(block, p.start, p.end, p.step)?;
        let origin = self.warp_block_origin(block, wm, wn, fm * 16, fnn * 16)?;
        let (_, _, _, m0, n0) = origin;

        let pairs = if self.pipeline_assert { 2 } else { 1 };
        let (a_bufs, b_bufs) = self.alloc_staging_pairs(pairs, |cg| {
            Ok((alloc(cg, block, &[m, kk])?, alloc(cg, block, &[kk, n])?))
        })?;

        let (stage_async, reg_stage) = self.staging_mode(p, m, kk, n);

        let finals = self.matmul_kloop(
            block,
            (lo, hi, st),
            regs,
            &a_bufs,
            &b_bufs,
            |cg, body, kt, a, b| cg.wmma_stage(body, p, kt, iv_div, a, b, false),
            |cg, body, piv, cur, dst, accs| {
                cg.wmma_half(
                    body,
                    p,
                    iv_div,
                    piv,
                    hi,
                    st,
                    cur,
                    dst,
                    dims,
                    m0,
                    n0,
                    accs,
                    stage_async,
                    reg_stage,
                    mma_sync,
                )
            },
            |cg, body, a, b, accs| cg.tc_mac(body, a, b, dims, m0, n0, accs, false, mma_sync),
        )?;

        Ok((finals, origin))
    }

    /// Tile-by-tile matmul on the tensor cores: out = a @ b (NN), or
    /// out = a @ b.T (NT, transpose_b), contracting the last dim of both
    /// operands. f16 inputs, f32 accumulate; the f32 result goes straight into
    /// the (shared) out tile. Returns false without emitting anything when
    /// @tensorcore is off or the shapes don't split into whole 16x16 tiles
    /// owned by whole warps, so the caller can fall back to the vector path.
    ///
    /// With accumulate (out += a @ b) the accumulator fragments start from out
    /// instead of zero, folding the running sum into the tensor-core MAC. This
    /// relies on the launch matching cta_threads (no surplus warps), which the
    /// warp grid (wm * wn == warps) and the launch ABI guarantee; a surplus
    /// warp would double-count.
    pub(in crate::codegen) fn wmma_dot(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        transpose_b: bool,
        accumulate: bool,
    ) -> Result<bool> {
        let (m, n) = (out.shape[0], out.shape[1]);
        let kk = a.shape[1];
        let Some((wm, wn)) = self.has_wmma().then(|| self.wmma_plan(m, n, kk)).flatten() else {
            return Ok(false);
        };

        // The tensor cores accumulate in f32: the output (and so the running
        // sum for +=) must be f32. Operands may be f16 or f32; f32 is
        // rounded to f16 when staged. Anything else stays on the vector path.
        if out.elem != self.f32_t || !self.is_f16_or_f32(a.elem) || !self.is_f16_or_f32(b.elem) {
            return Ok(false);
        }
        let (fm, fnn) = ((m / 16) / wm, (n / 16) / wn);
        let dims = (kk, fm, fnn);

        // thread-level mma.sync + ldmatrix
        if self.has_mma_sync() {
            return self.mma_sync_dot(block, a, b, out, transpose_b, accumulate, wm, wn);
        }

        // f16 staging buffers, mirroring each operand's natural layout: a as
        // [m, k], b as [k, n] (NN) or [n, k] (NT). No transposing copy; the
        // NT B fragment is transposed in the wmma load instead.
        let (a_buf, a_hoisted) = self.dot_stage(block, a, &[m, kk], false)?;
        let (b_buf, b_hoisted) = self.dot_stage(block, b, &b.shape.clone(), false)?;
        self.barrier(block)?;

        // The warp's fragment-block origin; surplus warps clamp onto the
        // last block and recompute it (identical writes, benign).
        let (_, _, _, m0, n0) = self.warp_block_origin(block, wm, wn, fm * 16, fnn * 16)?;

        // Accumulators: the running out fragments for +=, else zero.
        let c_frag_t = self.wmma_c_type(self.f32_t)?;
        let mut regs = Vec::with_capacity((fm * fnn) as usize);

        if accumulate {
            for fi in 0..fm {
                for fj in 0..fnn {
                    let (mb, nb) = self.frag_origin(block, m0, n0, fi, fj)?;
                    regs.push(self.wmma_load(block, out, &[mb, nb], c_frag_t, false)?);
                }
            }
        } else {
            let zero = self.zero_scalar(block, self.f32_t)?;

            for _ in 0..fm * fnn {
                regs.push(self.wmma_const_frag(block, zero, c_frag_t)?);
            }
        }
        let finals = self.wmma_mac(block, &a_buf, &b_buf, dims, m0, n0, &regs, transpose_b)?;

        // Each warp stores its fragments straight to its disjoint slice of
        // the shared output (lead dimension = the tile's row stride). A
        // closing barrier publishes them before downstream reads.
        for fi in 0..fm {
            for fj in 0..fnn {
                let frag = finals[(fi * fnn + fj) as usize];
                let (mb, nb) = self.frag_origin(block, m0, n0, fi, fj)?;
                self.wmma_store(block, frag, out.mem, &[mb, nb], n)?;
            }
        }
        self.barrier(block)?;

        // The staging is dead past the MAC; the closing barrier orders its
        // reads before any pooled reuse. Hoisted buffers outlive the loop.
        if !a_hoisted {
            self.release(&a_buf);
        }
        if !b_hoisted {
            self.release(&b_buf);
        }
        Ok(true)
    }

    /// Stages one iteration's a and b slices into shared as f16, without a
    /// barrier (the caller owns synchronization).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn wmma_stage(
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

        self.stage_to_f16(block, &a_src, a_buf, async_copy)?;
        self.stage_to_f16(block, &b_src, b_buf, async_copy)
    }

    /// The tensor-core pipelined half: prefetch into dst, tensor-core MAC from
    /// cur (see [`Self::pipelined_half`]). reg_stage selects the sm_75
    /// register-staged variant (see [`Self::wmma_half_reg_staged`]).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn wmma_half(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        iv_div: i64,
        prefetch_iv: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        st: Value<'c, 'c>,
        cur: (&MemVal<'c>, &MemVal<'c>),
        dst: (&MemVal<'c>, &MemVal<'c>),
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
        async_copy: bool,
        reg_stage: bool,
        mma_sync: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        if reg_stage {
            return self.wmma_half_reg_staged(
                block,
                p,
                iv_div,
                prefetch_iv,
                hi,
                st,
                cur,
                dst,
                dims,
                m0,
                n0,
                accs,
                mma_sync,
            );
        }

        self.pipelined_half(
            block,
            prefetch_iv,
            hi,
            async_copy,
            |cg, then| cg.wmma_stage(then, p, prefetch_iv, iv_div, dst.0, dst.1, async_copy),
            |cg| cg.tc_mac(block, cur.0, cur.1, dims, m0, n0, accs, false, mma_sync),
        )
    }

    /// The sm_75 register-staged pipeline half. The synchronous path goes
    /// load -> shared -> barrier -> compute, so the shared store stalls on the
    /// global load before any math runs. This one issues the next tile's global
    /// loads into registers, runs the current tile's WMMA compute while they're
    /// in flight, then stores the registers to shared under the same prefetch
    /// guard. The loads are unconditional (so they overlap the compute) with a
    /// clamped, in-bounds index, and the guarded store skips the redundant final
    /// read. Assumes the launch block is cta_threads (the WMMA launch ABI).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn wmma_half_reg_staged(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        iv_div: i64,
        prefetch_iv: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        st: Value<'c, 'c>,
        cur: (&MemVal<'c>, &MemVal<'c>),
        dst: (&MemVal<'c>, &MemVal<'c>),
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
        mma_sync: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        // Clamp the prefetch to the last valid tile start: the load runs even on
        // the final iteration (no "next" tile), where it harmlessly re-reads the
        // last tile since the guarded store below skips it.
        let last = self.subi(block, hi, st)?;
        let safe = self.minsi(block, prefetch_iv, last)?;

        // Issue the next tile's global loads into registers (held across compute).
        let a_loaded = self.wmma_load_operand(block, p, p.a_slice, safe, iv_div, dst.0)?;
        let b_loaded = self.wmma_load_operand(block, p, p.b_slice, safe, iv_div, dst.1)?;

        // Compute the current tile while the loads are in flight.
        let next = self.tc_mac(block, cur.0, cur.1, dims, m0, n0, accs, false, mma_sync)?;

        // Commit the prefetched tile to shared (only when it was real), then a
        // closing barrier publishes it and retires the reads cur will reuse.
        self.guarded_prefetch(block, prefetch_iv, hi, |cg, then| {
            cg.wmma_store_operand(then, &a_loaded, dst.0)?;
            cg.wmma_store_operand(then, &b_loaded, dst.1)
        })?;

        self.barrier(block)?;

        Ok(next)
    }

    /// Unrolled per-thread vector<[`HALF_VEC`]xf16> loads of one staged operand
    /// from global into registers (no shared store). Returns each loaded vector
    /// with its [row, col] destination index, for [`Self::wmma_store_operand`]
    /// to write after the compute. The per-thread count is static (the caller
    /// gates on [`Self::reg_stage_divides`]) and the CTA stride is cta_threads.
    pub(super) fn wmma_load_operand(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        slice: &Expr,
        kt: Value<'c, 'c>,
        iv_div: i64,
        dst: &MemVal<'c>,
    ) -> Result<Vec<(Value<'c, 'c>, [Value<'c, 'c>; 2])>> {
        let src = self.emit_kt_slice(block, p, slice, kt, iv_div)?;
        let (rows, cols) = (dst.shape[0], dst.shape[1]);
        let inner = cols / HALF_VEC; // HALF_VEC-wide vectors per row
        let per = (rows * inner) / self.cta_threads;
        let vec_t = Type::vector(&[HALF_VEC as u64], self.f16_t);
        let tid = self.thread_id(block)?;
        let inner_v = self.const_index(block, inner)?;
        let vec_w = self.const_index(block, HALF_VEC)?;
        let mut loaded = Vec::with_capacity(per as usize);

        for i in 0..per {
            // linear vector index tid + i*cta_threads row-major
            let off = self.const_index(block, i * self.cta_threads)?;
            let lin = self.addi(block, tid, off)?;
            let r = self.divui(block, lin, inner_v)?;
            let c4 = self.remui(block, lin, inner_v)?;
            let c = self.muli(block, c4, vec_w)?;
            // 8-byte f16 vector load from the global slice
            let v = self.vec_load_al(block, src.mem, &[r, c], vec_t, 8)?;
            loaded.push((v, [r, c]));
        }

        Ok(loaded)
    }

    pub(super) fn wmma_store_operand(
        &mut self,
        block: &Block<'c>,
        loaded: &[(Value<'c, 'c>, [Value<'c, 'c>; 2])],
        dst: &MemVal<'c>,
    ) -> Result<()> {
        for (v, idx) in loaded {
            let didx = self.swizzled_index(block, dst, idx)?;
            self.vec_store_al(block, *v, dst.mem, &didx, 8)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn wmma_mac(
        &mut self,
        block: &Block<'c>,
        a_buf: &MemVal<'c>,
        b_buf: &MemVal<'c>,
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
        transpose_b: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let (kk, fm, fnn) = dims;
        let a_frag_t = self.wmma_a_type()?;
        let b_frag_t = self.wmma_b_type()?;
        let c_frag_t = accs
            .first()
            .map(|v| v.r#type())
            .ok_or_else(|| anyhow!("wmma_mac needs at least one accumulator fragment"))?;
        let mut accs = accs.to_vec();

        for ks in 0..kk / 16 {
            let k_v = self.const_index(block, ks * 16)?;
            let mut a_frags = Vec::with_capacity(fm as usize);

            for fi in 0..fm {
                let c = self.const_index(block, fi * 16)?;
                let mi = self.addi(block, m0, c)?;
                a_frags.push(self.wmma_load(block, a_buf, &[mi, k_v], a_frag_t, false)?);
            }

            let mut b_frags = Vec::with_capacity(fnn as usize);
            for fj in 0..fnn {
                let c = self.const_index(block, fj * 16)?;
                let nj = self.addi(block, n0, c)?;
                // NN reads b[k, n]; NT reads the same logical fragment out of
                // the [n, k] buffer at [n, k], transposed in the load.
                let idx = if transpose_b { [nj, k_v] } else { [k_v, nj] };
                b_frags.push(self.wmma_load(block, b_buf, &idx, b_frag_t, transpose_b)?);
            }

            for fi in 0..fm {
                for fj in 0..fnn {
                    let i = (fi * fnn + fj) as usize;

                    accs[i] = self.wmma_compute(
                        block,
                        a_frags[fi as usize],
                        b_frags[fj as usize],
                        accs[i],
                        c_frag_t,
                    )?;
                }
            }
        }

        Ok(accs)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn tc_mac(
        &mut self,
        block: &Block<'c>,
        a_buf: &MemVal<'c>,
        b_buf: &MemVal<'c>,
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
        transpose_b: bool,
        mma_sync: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        if mma_sync {
            self.mma_sync_mac(block, a_buf, b_buf, dims, m0, n0, accs, transpose_b)
        } else {
            self.wmma_mac(block, a_buf, b_buf, dims, m0, n0, accs, transpose_b)
        }
    }
}
