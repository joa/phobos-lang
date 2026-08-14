// The Q8_0 tensor-core matmul used by a prompt's projections.

use super::*;

impl<'c> Codegen<'c> {
    /// The warp's patch of `m8n8k16` tiles: how many down, how many across.
    ///
    /// Grown from one tile a side, alternating so the patch stays square, and
    /// stopped when the patch would leave warps of the CTA with nothing to do.
    /// A square patch is what makes the operand loads pay: `rm` by `rn` tiles
    /// issue `2 * rm * rn` tensor instructions against `2 * (rm + rn)` loads.
    pub(super) fn qmma_patch(&self, rt: i64, ct: i64) -> (i64, i64) {
        let warps = self.cta_threads / WARP;
        let (mut rm, mut rn) = (1, 1);
        while rm * rn < QMMA_TILES {
            let grow_rows = rm < rn;
            let (try_m, try_n) = if grow_rows {
                (rm * 2, rn)
            } else {
                (rm, rn * 2)
            };
            let fits = rt % try_m == 0 && ct % try_n == 0;
            if !fits || (rt / try_m) * (ct / try_n) < warps {
                // The other direction may still have room.
                let (alt_m, alt_n) = if grow_rows {
                    (rm, rn * 2)
                } else {
                    (rm * 2, rn)
                };
                if alt_m * alt_n <= QMMA_TILES
                    && rt % alt_m == 0
                    && ct % alt_n == 0
                    && (rt / alt_m) * (ct / alt_n) >= warps
                {
                    (rm, rn) = (alt_m, alt_n);
                    continue;
                }
                break;
            }
            (rm, rn) = (try_m, try_n);
        }
        (rm, rn)
    }

    /// out[i, j] = sum_b (sum_{k in block b} a[i, k] * w[j, k]) * asc[i, b] * wsc[b, j]:
    /// the batched Q8_0 contraction on the integer tensor cores, block scales
    /// included, as one operation.
    ///
    /// This is to a prompt pass what [`Self::tile_qdot_t`] is to a decode step,
    /// and it exists for the same reason. Written in the tile language the
    /// contraction has to stop every 32 elements of `k` to apply the scales,
    /// which puts the accumulator in shared memory and stages both operands
    /// there per block: for a `[64, 64]` tile the accumulator alone is 16 KB,
    /// so the tile cannot even be built.
    ///
    /// Folding the scales in lets the whole of `k` stay inside one operation,
    /// so the accumulators are registers and live across it, both operands are
    /// read straight from global memory in the layout the `m8n8k16` fragments
    /// already want, and there is no barrier in the loop at all.
    ///
    /// The weight scales are indexed `[block, out]` here and `[out, block]` in
    /// `qdot_t`, which is not an inconsistency: a lane of this kernel holds two
    /// neighbouring output columns of one block, so `[block, out]` puts its two
    /// scales next to each other and a warp's eight columns in one sector.
    ///
    /// Like `qdot_t` this assumes `k` is a whole number of Q8_0 blocks, which
    /// is what the format guarantees.
    pub(in crate::codegen) fn tile_qmma_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        asc: &MemVal<'c>,
        w: &MemVal<'c>,
        wsc: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        let (md, nd) = (a.shape[0], w.shape[0]);
        if md == DYN || nd == DYN {
            bail!("qmma_t needs a static output shape");
        }
        let out = self.alloc_tile_shaped(block, self.f32_t, &[md, nd])?;
        self.qmma_t_into(block, a, asc, w, wsc, &out)?;
        Ok(out)
    }

    /// [`Self::tile_qmma_t`] writing an existing destination rather than a
    /// fresh tile.
    ///
    /// The destination is normally a slice of the output tensor, which is what
    /// makes this worth having: the accumulators are already in registers, and
    /// going through a shared tile on the way out costs a `[128, 64]` f32
    /// buffer, which is 32 KB and holds the kernel to one CTA per
    /// multiprocessor. Writing global directly leaves the occupancy to the
    /// register file.
    pub(in crate::codegen) fn qmma_t_into(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        asc: &MemVal<'c>,
        w: &MemVal<'c>,
        wsc: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        if out.elem != self.f32_t {
            bail!("qmma_t produces f32");
        }
        if out.is_masked() {
            bail!("qmma_t needs a fully in-bounds destination");
        }
        for (v, what) in [
            (a, "qmma_t a"),
            (asc, "qmma_t a scales"),
            (w, "qmma_t w"),
            (wsc, "qmma_t w scales"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
            if v.swizzle.is_some() {
                bail!("{what} must not be swizzled");
            }
        }
        if a.elem != self.i8_t || w.elem != self.i8_t {
            bail!("qmma_t contracts int8 operands");
        }
        if asc.elem != self.f32_t || wsc.elem != self.f32_t {
            bail!("qmma_t scales must be f32");
        }
        if !self.has_int8_mma() {
            bail!("qmma_t needs the integer tensor cores (sm_75 or later)");
        }
        let (md, nd) = (a.shape[0], w.shape[0]);
        if md == DYN || nd == DYN {
            bail!("qmma_t needs a static output shape");
        }
        self.check_shapes(&[md, nd], &out.shape, "qmma_t destination")?;
        if md % IMMA_TILE != 0 || nd % IMMA_TILE != 0 {
            bail!("qmma_t needs an output tile that is a multiple of {IMMA_TILE} both ways");
        }
        self.check_shapes(&[md], &[asc.shape[0]], "qmma_t a scale rows")?;
        self.check_shapes(&[nd], &[wsc.shape[1]], "qmma_t w scale columns")?;
        if self.cta_threads % WARP != 0 {
            bail!("qmma_t needs a CTA that is a whole number of warps");
        }

        let (i32_t, f32_t) = (self.i32_t, self.f32_t);
        let vec4_i8 = Type::vector(&[4], self.i8_t);
        let frag_t = Type::vector(&[1, 4], self.i8_t);
        let acc_t = Type::vector(&[1, 2], i32_t);

        let one = self.const_index(block, 1)?;
        let kd = if a.shape[1] == DYN {
            self.push(block, memref::dim(a.mem, one, self.loc))?
        } else {
            self.const_index(block, a.shape[1])?
        };

        // The lane's place in the fragments, as in [`Self::tile_matmul_t_imma`]:
        // both operands are read from row lane / 4 four bytes on, and the two
        // accumulator elements land in columns 2 * (lane % 4) and one past.
        let tid = self.thread_id(block)?;
        let warp_w = self.const_index(block, WARP)?;
        let warp = self.divui(block, tid, warp_w)?;
        let lane = self.remui(block, tid, warp_w)?;
        let four = self.const_index(block, 4)?;
        let quad = self.divui(block, lane, four)?;
        let in_quad = self.remui(block, lane, four)?;
        let k_off = self.muli(block, in_quad, four)?;
        let two = self.const_index(block, 2)?;
        let d_col = self.muli(block, in_quad, two)?;

        let (rt, ct) = (md / IMMA_TILE, nd / IMMA_TILE);
        let (rm, rn) = self.qmma_patch(rt, ct);
        let patches = (rt / rm) * (ct / rn);
        let warps = self.const_index(block, self.cta_threads / WARP)?;
        let total = self.const_index(block, patches)?;
        let across = self.const_index(block, ct / rn)?;

        let tb = Block::new(&[(self.index_t, self.loc)]);
        let u = detach(tb.argument(0)?.into());
        let ui = self.divui(&tb, u, across)?;
        let uj = self.remui(&tb, u, across)?;
        let patch_m = self.const_index(&tb, rm * IMMA_TILE)?;
        let patch_n = self.const_index(&tb, rn * IMMA_TILE)?;
        let i0 = self.muli(&tb, ui, patch_m)?;
        let j0 = self.muli(&tb, uj, patch_n)?;
        let row_base = self.addi(&tb, i0, quad)?;
        let col_base = self.addi(&tb, j0, d_col)?;
        let frag_base = self.addi(&tb, j0, quad)?;

        // Row of A and of the accumulator, row of W, and output column, one per
        // tile of the patch. All are loop-invariant, so they are built once.
        let mut a_rows = Vec::with_capacity(rm as usize);
        for r in 0..rm {
            let off = self.const_index(&tb, r * IMMA_TILE)?;
            a_rows.push(self.addi(&tb, row_base, off)?);
        }
        let (mut w_rows, mut out_cols) = (Vec::new(), Vec::new());
        for c in 0..rn {
            let off = self.const_index(&tb, c * IMMA_TILE)?;
            w_rows.push(self.addi(&tb, frag_base, off)?);
            out_cols.push(self.addi(&tb, col_base, off)?);
        }

        let lanes = (rm * rn * 2) as usize;
        let mut args = vec![(self.index_t, self.loc)];
        args.extend(std::iter::repeat_n((f32_t, self.loc), lanes));
        let kb = Block::new(&args);
        let k = detach(kb.argument(0)?.into());
        let mut accs = Vec::with_capacity(lanes);
        for slot in 0..lanes {
            accs.push(detach(kb.argument(slot + 1)?.into()));
        }

        let blk_w = self.const_index(&kb, Q8_BLOCK)?;
        let b = self.divui(&kb, k, blk_w)?;
        let k_from = self.addi(&kb, k, k_off)?;
        let halves = Q8_BLOCK / IMMA_K;
        let mut k_cols = Vec::with_capacity(halves as usize);
        for h in 0..halves {
            let off = self.const_index(&kb, h * IMMA_K)?;
            k_cols.push(self.addi(&kb, k_from, off)?);
        }

        let mut a_frags = Vec::new();
        for row in &a_rows {
            for col in &k_cols {
                let v = self.vec_load_al(&kb, a.mem, &[*row, *col], vec4_i8, 4)?;
                a_frags.push(self.vec_shape_cast(&kb, v, frag_t)?);
            }
        }
        let mut w_frags = Vec::new();
        for row in &w_rows {
            for col in &k_cols {
                let v = self.vec_load_al(&kb, w.mem, &[*row, *col], vec4_i8, 4)?;
                w_frags.push(self.vec_shape_cast(&kb, v, frag_t)?);
            }
        }

        let zero_i = self.zero_scalar(&kb, i32_t)?;
        let empty = self.vec_broadcast(&kb, zero_i, acc_t)?;
        let shape = self.mma_shape(IMMA_TILE, IMMA_TILE, IMMA_K)?;

        // A weight scale belongs to an output column, not to a row of the
        // patch, so loading it inside the row loop would fetch each one `rm`
        // times. The patch is square and square is where the tensor work pays,
        // so that is eight redundant loads out of every nine.
        let mut w_scales = Vec::with_capacity(rn as usize * 2);
        for out_col in &out_cols {
            for dj in 0..2 {
                let off = self.const_index(&kb, dj)?;
                let col = self.addi(&kb, *out_col, off)?;
                w_scales.push(self.push(&kb, memref::load(wsc.mem, &[b, col], self.loc))?);
            }
        }

        let mut next = Vec::with_capacity(lanes);
        for r in 0..rm as usize {
            let sa = self.push(&kb, memref::load(asc.mem, &[a_rows[r], b], self.loc))?;
            for c in 0..rn as usize {
                let mut sum = empty;
                for h in 0..halves as usize {
                    sum = self.mma_sync(
                        &kb,
                        a_frags[r * halves as usize + h],
                        w_frags[c * halves as usize + h],
                        sum,
                        shape,
                        acc_t,
                    )?;
                }
                for dj in 0..2 {
                    let sw = w_scales[c * 2 + dj as usize];
                    let scale = self.push(&kb, arith::mulf(sa, sw, self.loc))?;
                    let raw = self.vec_extract(&kb, sum, &[0, dj], i32_t)?;
                    let as_f = self.small_int_to_f32(&kb, raw)?;
                    let slot = (r * rn as usize + c) * 2 + dj as usize;
                    // A mul and an add here are mul.rn and add.rn, which ptxas
                    // may not contract; at a patch's size that is 128 wasted
                    // instructions against the same block's 128 tensor ones.
                    next.push(self.elem_mac(&kb, f32_t, as_f, scale, accs[slot])?);
                }
            }
        }
        kb.append_operation(scf::r#yield(&next, self.loc));

        let zero_k = self.const_index(&tb, 0)?;
        let step = self.const_index(&tb, Q8_BLOCK)?;
        let init = self.zero_scalar(&tb, f32_t)?;
        let mut operands = vec![zero_k, kd, step];
        operands.extend(std::iter::repeat_n(init, lanes));
        let kr = Region::new();
        kr.append_block(kb);
        let loop_op = tb.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&operands)
                .add_results(&vec![f32_t; lanes])
                .add_regions([kr])
                .build()?,
        );

        for (r, row) in a_rows.iter().enumerate() {
            for (c, out_col) in out_cols.iter().enumerate() {
                for dj in 0..2 {
                    let off = self.const_index(&tb, dj)?;
                    let col = self.addi(&tb, *out_col, off)?;
                    let slot = (r * rn as usize + c) * 2 + dj as usize;
                    let value = detach(loop_op.result(slot)?.into());
                    tb.append_operation(memref::store(value, out.mem, &[*row, col], self.loc));
                }
            }
        }
        tb.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(tb);
        block.append_operation(scf::r#for(warp, total, warps, region, self.loc));
        self.barrier(block)?;
        Ok(())
    }
}
