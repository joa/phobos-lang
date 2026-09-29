// `@tensorcore` via mma.sync, the sm_75+ path. Needs 64-bit index lowering
// and its own fragment addressing, see `ldmatrix_frag`.

use super::*;

impl<'c> Codegen<'c> {
    /// One m16n8 vector<2x2x{acc}> accumulator per (fi, fj, n8), seeded
    /// from init.
    pub(super) fn mma_sync_seed(
        &mut self,
        block: &Block<'c>,
        plan: &GemmPlan<'c>,
        init: Value<'c, 'c>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let GemmPath::MmaSync { wm, wn } = plan.path else {
            bail!("mma_sync_seed on another path's plan");
        };
        let (fm, fnn) = ((plan.m / 16) / wm, (plan.n / 16) / wn);
        let init = self.coerce(block, init, plan.acc_elem)?;
        let acc_t = Type::vector(&[2, 2], plan.acc_elem);
        let seed = self.vec_broadcast(block, init, acc_t)?;
        Ok(vec![seed; (fm * fnn * 2) as usize])
    }

    /// Drains the mma.sync accumulators to C.
    ///
    /// Each lane's m16n8 D fragment (vector<2x2x{acc}>) covers (row, col)
    /// pairs of the 16x8 tile, see [`Self::mma_sync_dfrag_base`]. The warp
    /// scatters them into its private 16x16 shared slab. The lanes then copy
    /// the slab to C between barriers, computing alpha*acc + beta*prev. The
    /// drain is shared with the WMMA epilogue.
    pub(super) fn mma_sync_store(
        &mut self,
        block: &Block<'c>,
        acc: GemmAcc<'c>,
        view: MemVal<'c>,
        alpha: Option<GemmScale<'c>>,
        beta: Option<GemmScale<'c>>,
    ) -> Result<()> {
        let GemmPath::MmaSync { wm, wn } = acc.plan.path else {
            bail!("mma_sync_store on another path's plan");
        };
        let acc_elem = acc.plan.acc_elem;
        let (fm, fnn) = ((acc.plan.m / 16) / wm, (acc.plan.n / 16) / wn);
        let (tid, _, wt, m0, n0) = Self::gemm_finals(&acc)?;
        let finals = acc.regs;

        let slab = self.alloc_tile_shaped(block, acc_elem, &[(self.cta_threads / 32) * 16, 16])?;
        let slab_f32 = acc_elem == self.f32_t;
        let w = self.const_index(block, 32)?;
        let sixteen = self.const_index(block, 16)?;
        let slab0 = self.muli(block, wt, sixteen)?;

        // The lane's D-fragment coordinates within an m16n8 tile.
        let lane = self.remui(block, tid, w)?;
        let eight = self.const_index(block, 8)?;
        let (gid, dcol) = self.mma_sync_dfrag_base(block, lane)?;
        let srow_d = self.addi(block, slab0, gid)?;

        let (srow, lrow, lcol) = self.slab_lane_slice(block, slab0, lane)?;
        let row_t = Type::vector(&[4], self.f32_t);

        // Coalesced 4-wide drains: an f32 slab to an f32 C, or an f16 slab
        // through an f32 scale to an f16 C. Both need a 4-aligned output row
        // pitch, else the drain is scalar. The scaling vectors are always f32.
        let vec_f32 = slab_f32 && view.elem == self.f32_t && view.vectorizes(4);
        let vec_f16 = !slab_f32 && view.elem == self.f16_t && view.vectorizes(4);
        let (alpha, beta) = self.epilogue_scaling(block, alpha, beta, row_t, vec_f32 || vec_f16)?;
        let drain = SlabDrain {
            slab: slab.clone(),
            view,
            srow,
            lrow,
            lcol,
            alpha,
            beta,
            mode: if vec_f32 {
                DrainMode::VecF32
            } else if vec_f16 {
                DrainMode::VecF16
            } else {
                DrainMode::Scalar
            },
        };

        for fi in 0..fm {
            for fj in 0..fnn {
                // Scatter both n8 D fragments into the warp's 16x16 slab.
                for nn in 0..2 {
                    let frag = finals[((fi * fnn + fj) * 2 + nn) as usize];
                    let ncol = self.const_index(block, nn * 8)?;
                    let scol0 = self.addi(block, ncol, dcol)?;
                    for di in 0..2 {
                        let rbase = if di == 0 {
                            srow_d
                        } else {
                            self.addi(block, srow_d, eight)?
                        };
                        for dj in 0..2 {
                            let e = self.vec_extract(block, frag, &[di, dj], acc_elem)?;
                            let cj = self.const_index(block, dj)?;
                            let sc = self.addi(block, scol0, cj)?;
                            block.append_operation(memref::store(
                                e,
                                slab.mem,
                                &[rbase, sc],
                                self.loc,
                            ));
                        }
                    }
                }

                // The first barrier publishes the scatter. The second keeps
                // the next fragment from overwriting the slab mid-drain.
                self.barrier(block)?;
                self.drain_slab_tile(block, &drain, m0, n0, fi, fj)?;
                self.barrier(block)?;
            }
        }

        Ok(())
    }

    /// The lane's m16n8 D-fragment base within an mma.sync tile, as
    /// (row, col). The row is lane / 4, plus 8 for the second half. The column
    /// is 2 * (lane % 4), plus 1 for the second element of a pair.
    pub(super) fn mma_sync_dfrag_base(
        &self,
        block: &Block<'c>,
        lane: Value<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let four = self.const_index(block, 4)?;
        let two = self.const_index(block, 2)?;
        let gid = self.divui(block, lane, four)?;
        let tig = self.remui(block, lane, four)?;
        let dcol = self.muli(block, tig, two)?;

        Ok((gid, dcol))
    }

    /// The mma.sync counterpart of [`Self::wmma_dot`].
    ///
    /// Stages both operands as swizzled f16, accumulates per-lane
    /// vector<2x2xf32> fragments with nvgpu.mma.sync, and scatters the D
    /// fragments into the shared out tile. `+=` seeds the accumulators from
    /// out.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mma_sync_dot(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        transpose_b: bool,
        accumulate: bool,
        wm: i64,
        wn: i64,
    ) -> Result<bool> {
        let (m, n) = (out.shape[0], out.shape[1]);
        let kk = a.shape[1];
        let (fm, fnn) = ((m / 16) / wm, (n / 16) / wn);
        let dims = (kk, fm, fnn);

        // Swizzled f16 staging in each operand's natural layout: a as [m, k],
        // b as [k, n] (NN) or [n, k] (NT). ldmatrix reads consecutive rows,
        // which would hit the same shared banks. The swizzle permutes the
        // column per row, the same way on store and load.
        let (a_buf, a_hoisted) = self.dot_stage(block, a, &[m, kk], true)?;
        let (b_buf, b_hoisted) = self.dot_stage(block, b, &b.shape.clone(), true)?;
        self.barrier(block)?;

        // The warp's fragment-block origin. Surplus warps clamp onto the last.
        let (_, _, _, m0, n0) = self.warp_block_origin(block, wm, wn, fm * 16, fnn * 16)?;

        // Accumulators start from out for +=, else from zero.
        let acc_t = Type::vector(&[2, 2], self.f32_t);
        let regs = if accumulate {
            self.mma_sync_load_frags(block, out, (fm, fnn), m0, n0)?
        } else {
            let zero = self.zero_scalar(block, self.f32_t)?;
            let seed = self.vec_broadcast(block, zero, acc_t)?;

            vec![seed; (fm * fnn * 2) as usize]
        };

        let finals = self.mma_sync_mac(block, &a_buf, &b_buf, dims, m0, n0, &regs, transpose_b)?;

        self.mma_sync_store_frags(block, out, &finals, (fm, fnn), m0, n0)?;
        self.barrier(block)?;

        // The barrier above orders the staging reads before any reuse.
        // Hoisted buffers outlive the loop and are not released here.
        if !a_hoisted {
            self.release(&a_buf);
        }
        if !b_hoisted {
            self.release(&b_buf);
        }

        Ok(true)
    }

    /// Walks the warp's m16n8 D fragments over the tile at (m0, n0).
    ///
    /// Calls `f` with each fragment's register index and, per element, its
    /// (di, dj) position and [row, col] address. The scatter and the gather
    /// both use this walk, so their addressing always matches.
    pub(in crate::codegen) fn for_each_dfrag(
        &mut self,
        block: &Block<'c>,
        warp_frags: (i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        mut f: impl FnMut(&mut Self, usize, &[([i64; 2], [Value<'c, 'c>; 2])]) -> Result<()>,
    ) -> Result<()> {
        let (fm, fnn) = warp_frags;
        let tid = self.thread_id(block)?;
        let w = self.const_index(block, 32)?;
        let eight = self.const_index(block, 8)?;
        let lane = self.remui(block, tid, w)?;
        let (gid, dcol) = self.mma_sync_dfrag_base(block, lane)?;

        for fi in 0..fm {
            for fj in 0..fnn {
                let (mb, nb) = self.frag_origin(block, m0, n0, fi, fj)?;
                let mrow0 = self.addi(block, mb, gid)?;

                for nn in 0..2 {
                    let ncol = self.const_index(block, nn * 8)?;
                    let ncb = self.addi(block, nb, ncol)?;
                    let ncb = self.addi(block, ncb, dcol)?;
                    let mut elems = Vec::with_capacity(4);

                    for di in 0..2 {
                        let mrow = if di == 0 {
                            mrow0
                        } else {
                            self.addi(block, mrow0, eight)?
                        };

                        for dj in 0..2 {
                            let cj = self.const_index(block, dj)?;
                            let col = self.addi(block, ncb, cj)?;
                            elems.push(([di, dj], [mrow, col]));
                        }
                    }

                    f(self, ((fi * fnn + fj) * 2 + nn) as usize, &elems)?;
                }
            }
        }

        Ok(())
    }

    /// Scatters each lane's m16n8 D fragment (vector<2x2xf32>) to its
    /// (row, col) spots in the shared out tile. The caller adds the barrier.
    pub(super) fn mma_sync_store_frags(
        &mut self,
        block: &Block<'c>,
        dst: &MemVal<'c>,
        finals: &[Value<'c, 'c>],
        warp_frags: (i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
    ) -> Result<()> {
        self.for_each_dfrag(block, warp_frags, m0, n0, |cg, i, elems| {
            for ([di, dj], addr) in elems {
                let e = cg.vec_extract(block, finals[i], &[*di, *dj], cg.f32_t)?;
                block.append_operation(memref::store(e, dst.mem, addr, cg.loc));
            }
            Ok(())
        })
    }

    /// Seeds the per-lane D fragments from the shared out tile for `+=`. The
    /// inverse of [`Self::mma_sync_store_frags`].
    pub(super) fn mma_sync_load_frags(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        warp_frags: (i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let acc_t = Type::vector(&[2, 2], self.f32_t);
        let mut regs = Vec::with_capacity((warp_frags.0 * warp_frags.1 * 2) as usize);

        self.for_each_dfrag(block, warp_frags, m0, n0, |cg, _, elems| {
            let zero = cg.zero_scalar(block, cg.f32_t)?;
            let mut frag = cg.vec_broadcast(block, zero, acc_t)?;

            for ([di, dj], addr) in elems {
                let x = cg.load_as(block, src.mem, addr, cg.f32_t)?;
                frag = cg.vec_insert(block, x, frag, &[*di, *dj])?;
            }

            regs.push(frag);
            Ok(())
        })?;

        Ok(regs)
    }

    /// One k-tile of mma.sync from the staged f16 buffers.
    ///
    /// The hardware shape is m16n8kK, with K = 8 on Turing and 16 on Ampere+.
    /// Each 16x16 fragment splits into two n8 halves, so the warp owns
    /// fm * fnn * 2 accumulators. Each K-deep step loads A and B with ldmatrix
    /// and accumulates with nvgpu.mma.sync. With `transpose_b`, B is staged
    /// [n, k] and read without the transpose flag.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn mma_sync_mac(
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
        let mma_k = self.mma_sync_k()?;
        // 8x8 tiles per fragment: A spans m16 x K, B spans K x n8. Each lane
        // holds two f16 per tile.
        let a_tiles = 2 * (mma_k / 8);
        let b_tiles = mma_k / 8;
        let a_frag_t = Type::vector(&[a_tiles as u64, 2], self.f16_t);
        let b_frag_t = Type::vector(&[b_tiles as u64, 2], self.f16_t);
        let acc_t = accs
            .first()
            .map(|v| v.r#type())
            .ok_or_else(|| anyhow!("mma_sync_mac needs at least one accumulator"))?;
        let tid = self.thread_id(block)?;
        let w = self.const_index(block, 32)?;
        let lane = self.remui(block, tid, w)?;
        let mut accs = accs.to_vec();

        for ks in 0..kk / mma_k {
            let kbase = self.const_index(block, ks * mma_k)?;

            // A fragments: one m16 x K block per fi.
            let mut a_frags = Vec::with_capacity(fm as usize);
            for fi in 0..fm {
                let c = self.const_index(block, fi * 16)?;
                let mi = self.addi(block, m0, c)?;
                a_frags.push(self.ldmatrix_frag(
                    block,
                    a_buf,
                    &[mi, kbase],
                    a_tiles,
                    false,
                    a_frag_t,
                    lane,
                )?);
            }

            // B fragments: one K x n8 block per (fj, n8). NN reads the [k, n]
            // buffer transposed. NT reads the [n, k] buffer as is.
            let mut b_frags = Vec::with_capacity((fnn * 2) as usize);
            for fj in 0..fnn {
                for nn in 0..2 {
                    let c = self.const_index(block, fj * 16 + nn * 8)?;
                    let nj = self.addi(block, n0, c)?;
                    let (idx, trans) = if transpose_b {
                        ([nj, kbase], false)
                    } else {
                        ([kbase, nj], true)
                    };
                    b_frags.push(
                        self.ldmatrix_frag(block, b_buf, &idx, b_tiles, trans, b_frag_t, lane)?,
                    );
                }
            }

            for fi in 0..fm {
                for fj in 0..fnn {
                    for nn in 0..2 {
                        let i = ((fi * fnn + fj) * 2 + nn) as usize;
                        let shape = self.mma_shape(16, 8, mma_k)?;
                        accs[i] = self.mma_sync(
                            block,
                            a_frags[fi as usize],
                            b_frags[(fj * 2 + nn) as usize],
                            accs[i],
                            shape,
                            acc_t,
                        )?;
                    }
                }
            }
        }
        Ok(accs)
    }

    /// The warp's [`Self::ldmatrix`] operand at the warp-tile origin `indices`
    /// (row, col).
    ///
    /// The lowering does not spread addresses across lanes, so the per-lane
    /// offset is added here. Each lane supplies the start of one 8-element
    /// row. Non-transpose A is 16 rows by num_tiles/2 k-tiles of 8: row =
    /// lane % 16, column = (lane / 16) % (num_tiles/2) * 8. Transpose B is
    /// num_tiles 8x8 tiles stacked along k: row = lane % (8 * num_tiles),
    /// column unchanged. The column is then swizzled to match the staging
    /// store.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn ldmatrix_frag(
        &self,
        block: &Block<'c>,
        buf: &MemVal<'c>,
        indices: &[Value<'c, 'c>],
        num_tiles: i64,
        transpose: bool,
        frag_t: Type<'c>,
        lane: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        // Lanes past the used address count (8 per tile) still have their
        // address dereferenced. The row modulus keeps them inside the block,
        // or a read near the end of the buffer could fault.
        let (row_off, col_off) = if transpose {
            let rows = self.const_index(block, 8 * num_tiles)?;
            (self.remui(block, lane, rows)?, None)
        } else {
            let rows = self.const_index(block, (8 * num_tiles).min(16))?;
            let row = self.remui(block, lane, rows)?;
            let r16 = self.const_index(block, 16)?;
            let col_tiles = self.const_index(block, (num_tiles / 2).max(1))?;
            let g = self.divui(block, lane, r16)?;
            let gt = self.remui(block, g, col_tiles)?;
            let eight = self.const_index(block, 8)?;
            (row, Some(self.muli(block, gt, eight)?))
        };

        let row = self.addi(block, indices[0], row_off)?;
        let col = match col_off {
            Some(c) => self.addi(block, indices[1], c)?,
            None => indices[1],
        };

        // Identity when the buffer is unswizzled.
        let col = self.swizzle_col(block, buf, row, col)?;

        self.ldmatrix(block, buf.mem, [row, col], num_tiles, transpose, frag_t)
    }
}
