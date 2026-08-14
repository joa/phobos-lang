// `@tensorcore` via mma.sync, the sm_75 path. Needs 64-bit index
// lowering and its own fragment addressing; see `ldmatrix_frag`.

use super::*;

impl<'c> Codegen<'c> {
    pub(in crate::codegen) fn emit_mma_sync_matmul(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
    ) -> Result<()> {
        let (shape, kk) = self.fusion_dims(p)?;
        let (m, n) = (shape[0], shape[1]);
        let (wm, wn) = self
            .wmma_plan(m, n, kk)
            .ok_or_else(|| anyhow!("mma.sync matmul without a warp grid"))?;
        let (fm, fnn) = ((m / 16) / wm, (n / 16) / wn);

        // the warp's accumulators: one m16n8 vector<2x2x{acc}> per (fi, fj, n8),
        // initialized with the scalar rounded to the accumulator type
        let acc_elem = self.scalar_type(p.acc_scalar);
        let init = self.emit_scalar(block, p.init)?;
        let init = self.coerce(block, init, acc_elem)?;
        let acc_t = Type::vector(&[2, 2], acc_elem);
        let seed = self.vec_broadcast(block, init, acc_t)?;
        let regs = vec![seed; (fm * fnn * 2) as usize];

        // f16 staging, XOR-swizzled (not padded): the ldmatrix reads stride
        // consecutive rows, which alias shared banks unpadded, so the column
        // is permuted per row to spread them, at zero extra SM. The store and
        // load MUST apply the same swizzle.
        let (finals, (tid, _, wt, m0, n0)) = self.tc_matmul_kloop(
            block,
            p,
            (m, n, kk),
            (wm, wn),
            &regs,
            true,
            |cg, blk, shape| cg.alloc_tile_swizzled(blk, cg.f16_t, shape),
        )?;

        self.mma_sync_epilogue(
            block,
            p,
            &shape,
            finals,
            (fm, fnn),
            acc_elem,
            tid,
            wt,
            m0,
            n0,
        )
    }

    /// Drains the mma.sync accumulators to C. Each lane's m16n8 D fragment
    /// (vector<2x2x{acc}>) maps to known (row, col) pairs of the 16x8 tile
    /// (row = laneId/4 [+8], col = 2*(laneId%4) [+1]). The warp scatters them
    /// into its private 16x16 shared slab, then the lanes copy slab to C between
    /// barriers, applying alpha*acc + beta*prev_load. The drain matches the WMMA
    /// epilogue (lane = half a slab row, two 4-vectors) so the f32 fast path and
    /// the f16 rounding path can be shared.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mma_sync_epilogue(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        shape: &[i64],
        finals: Vec<Value<'c, 'c>>,
        warp_frags: (i64, i64),
        acc_elem: Type<'c>,
        tid: Value<'c, 'c>,
        wt: Value<'c, 'c>,
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
    ) -> Result<()> {
        let (fm, fnn) = warp_frags;
        let view = self.epilogue_view(block, p, shape)?;

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

        // Coalesced 4-wide drains: an f32 slab straight to an f32 C, or an f16
        // slab through an f32 scale to an f16 C (the gemm_fp16.ph case). Both
        // need an aligned (multiple-of-4 row pitch) output, otherwise we drain
        // scalar. The scaling vectors are f32 either way (the f16 path scales
        // after extf).
        let vec_f32 = slab_f32 && view.elem == self.f32_t && view.vectorizes(4);
        let vec_f16 = !slab_f32 && view.elem == self.f16_t && view.vectorizes(4);
        let (alpha, beta) = self.epilogue_scaling(block, p, row_t, vec_f32 || vec_f16)?;
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

                // publish the scatter, drain it to C, and retire the reads
                // before the next fragment overwrites the slab
                self.barrier(block)?;
                self.drain_slab_tile(block, &drain, m0, n0, fi, fj)?;
                self.barrier(block)?;
            }
        }

        Ok(())
    }

    /// The lane's m16n8 D-fragment base within an mma.sync tile. The row is
    /// laneId / 4 (the second 8-row half adds 8) and the column is
    /// 2 * (laneId % 4) (the second element of a pair adds 1).
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

    /// The mma.sync counterpart of the legacy [`Self::wmma_dot`] body (taken
    /// when mma_sync() holds): swizzled f16 staging, per-lane vector<2x2xf32>
    /// accumulators folded with nvgpu.mma.sync, and a direct scatter of the D
    /// fragments to the shared out tile. += seeds the accumulators from out
    /// (folding the running sum into the MAC), the same as the WMMA path.
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

        // Swizzled (not padded) f16 staging, matching each operand's natural
        // layout: a as [m, k], b as [k, n] (NN) or [n, k] (NT). The ldmatrix
        // reads stride consecutive rows, which alias shared banks when unpadded.
        // The swizzle permutes the column per row, the same way on store and
        // load, so it spreads the banks out for free.
        let (a_buf, a_hoisted) = self.dot_stage(block, a, &[m, kk], true)?;
        let (b_buf, b_hoisted) = self.dot_stage(block, b, &b.shape.clone(), true)?;
        self.barrier(block)?;

        // The warp's fragment-block origin; surplus warps clamp onto the last.
        let (_, _, _, m0, n0) = self.warp_block_origin(block, wm, wn, fm * 16, fnn * 16)?;

        // Accumulators: the running out D fragments for +=, else zero.
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

    /// Walks the warp's m16n8 D fragments over the tile at (m0, n0). For each
    /// (fi, fj, n8) fragment it computes the lane's four element addresses
    /// (see [`Self::mma_sync_dfrag_base`]) and yields the fragment's register
    /// index with each element's (di, dj) position and [row, col] address.
    /// The scatter and its inverse gather share this walk, so their
    /// addressing cannot drift apart.
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

    /// Scatters each lane's m16n8 D fragment (vector<2x2xf32>) straight to its
    /// (row, col) spot in the shared out tile. This is the dot's barrier-free
    /// publish, the mma.sync take on a straight-to-shared
    /// subgroup_mma_store_matrix. The caller barriers afterward.
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

    /// Seeds the per-lane D fragments from the shared out tile (the += running
    /// sum), the inverse of [`Self::mma_sync_store_frags`]: each lane reads its
    /// four (row, col) elements per m16n8 tile into a vector<2x2xf32>.
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

    /// One k-tile of mma.sync computes from the staged f16 buffers. The legacy
    /// WMMA MAC works in m16n16k16 fragments, but the hardware mma.sync shape is
    /// m16n8kK (K = 8 on Turing, 16 on Ampere+), so each logical 16x16 fragment
    /// splits into two n-sub-tiles of 8, and the warp owns fm * fnn * 2
    /// vector<2x2x{acc}> accumulators (one per (fi, fj, n8)). Each kK-deep step
    /// ldmatrix-loads the warp's A and B fragments and folds them in with
    /// nvgpu.mma.sync. With transpose_b the B operand is staged [n, k] (the
    /// dot_t layout) and read without the transpose flip.
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
        // Per-lane fragment register counts: A spans m16 x kK as 8x8 tiles, B
        // spans kK x n8, both two f16 per tile. C/D is the m16n8 accumulator.
        let a_tiles = 2 * (mma_k / 8);
        let b_tiles = mma_k / 8;
        let a_frag_t = Type::vector(&[a_tiles as u64, 2], self.f16_t);
        let b_frag_t = Type::vector(&[b_tiles as u64, 2], self.f16_t);
        let acc_t = accs
            .first()
            .map(|v| v.r#type())
            .ok_or_else(|| anyhow!("mma_sync_mac needs at least one accumulator"))?;
        // The lane index spreads the ldmatrix reads across the warp.
        let tid = self.thread_id(block)?;
        let w = self.const_index(block, 32)?;
        let lane = self.remui(block, tid, w)?;
        let mut accs = accs.to_vec();

        for ks in 0..kk / mma_k {
            let kbase = self.const_index(block, ks * mma_k)?;

            // A fragments: one m16 x kK block per fi.
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

            // B fragments: one kK x n8 block per (fj, n8). NN reads b[k, n]
            // transposed (k-major buffer, col-major operand); NT reads the
            // [n, k] buffer straight.
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
    /// (row, col). The lowering just does a plain strided-element-pointer on
    /// them and doesn't spread the work across the warp's lanes, so we fold the
    /// per-lane address offset in here (each ldmatrix lane holds the start
    /// address of one 8-element row).
    ///
    /// The offset depends on how the operand's tiles are laid out. Non-transpose
    /// A is 16 rows by num_tiles/2 8-wide k-tiles, so the row is lane % 16 and
    /// the k-column is (lane / 16) % (num_tiles/2) * 8. Transpose B is num_tiles
    /// 8x8 tiles stacked along k, so the row is lane % (8 * num_tiles) and the
    /// column is just the warp-tile column. On sm_75 these collapse to A
    /// lane % 16 / col 0 and B lane % 8. Since the offset rides the index, any
    /// swizzle (a later phase) has to ride the staging store, not this load.
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
        // Fold the per-lane (row, col) offset into the warp-tile origin.
        // Lanes past the consumed address count (8 per tile) still compute an
        // address the hardware dereferences even though the value is unused,
        // so the row modulus clamps them into the block: an operand in the
        // last rows of the final shared buffer would otherwise read past the
        // CTA's shared window and fault (observed on sm_75 with ldmatrix.x1
        // once buffer pooling shrank the window).
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

        // Read back the swizzled column the staging store wrote (identity when
        // the buffer is unswizzled).
        let col = self.swizzle_col(block, buf, row, col)?;

        self.ldmatrix(block, buf.mem, [row, col], num_tiles, transpose, frag_t)
    }
}
