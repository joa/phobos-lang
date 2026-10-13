use super::*;

/// See [`Codegen::imma_lane`].
struct ImmaLane<'c> {
    rows: Vec<Value<'c, 'c>>,
    cols: Vec<Value<'c, 'c>>,
    w_rows: Vec<Value<'c, 'c>>,
    k_off: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {

    /// The fragments of `fa` with each element combined by `op` with its
    /// row's value in `col`. Rebinds nothing.
    pub(super) fn frag_scale_raw(
        &mut self,
        block: &Block<'c>,
        fa: &FragAcc<'c>,
        op: BinOp,
        col: &MemVal<'c>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        if fa.imma {
            bail!("internal: row scaling of an int8 fragment accumulator");
        }
        let (fm, fnn) = fa.warp_frags();
        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        let zero = self.const_index(block, 0)?;
        let mut frags = fa.frags.clone();
        self.for_each_dfrag(block, (fm, fnn), m0, n0, |cg, i, elems| {
            for ([di, dj], [row, _]) in elems {
                let c = cg.load_as(block, col.mem, &[*row, zero], cg.f32_t)?;
                let e = cg.vec_extract(block, frags[i], &[*di, *dj], cg.f32_t)?;
                let r = cg.push(block, cg.elem_arith(op, cg.f32_t, e, c)?)?;
                frags[i] = cg.vec_insert(block, r, frags[i], &[*di, *dj])?;
            }
            Ok(())
        })?;
        Ok(frags)
    }

    /// The fragments of `fa += dot(a, b)`. Rebinds nothing.
    pub(super) fn frag_dot_raw(
        &mut self,
        block: &Block<'c>,
        fa: &FragAcc<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        if fa.imma {
            bail!("internal: f16 dot into an int8 fragment accumulator");
        }
        let kk = a.shape[1];
        self.check_shapes(&a.shape, &[fa.m, kk], "fragment dot lhs")?;
        self.check_shapes(&b.shape, &[kk, fa.n], "fragment dot rhs")?;
        let (fm, fnn) = fa.warp_frags();

        let (a_buf, a_hoisted) = self.dot_stage(block, a, &[fa.m, kk], true)?;
        let (b_buf, b_hoisted) = self.dot_stage(block, b, &b.shape.clone(), true)?;
        self.barrier(block)?;

        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        let finals = self.mma_sync_mac(
            block,
            &a_buf,
            &b_buf,
            (kk, fm, fnn),
            m0,
            n0,
            &fa.frags,
            false,
        )?;

        // Nothing to publish, but the barrier orders the ldmatrix reads
        // before the staging is reused. Hoisted buffers are not released.
        self.barrier(block)?;
        if !a_hoisted {
            self.release(&a_buf);
        }
        if !b_hoisted {
            self.release(&b_buf);
        }
        Ok(finals)
    }

    /// A lane's place in its warp's patch of 8x8 tiles, as in
    /// [`Self::tile_matmul_t_imma`]: the accumulator row of each patch row,
    /// the first of its two columns in each patch column, the row of the
    /// transposed operand each patch column reads, and its k offset within
    /// a 16-wide step.
    fn imma_lane(&mut self, block: &Block<'c>, fa: &FragAcc<'c>) -> Result<ImmaLane<'c>> {
        let (fm, fnn) = fa.warp_frags();
        let (.., m0, n0) =
            self.warp_block_origin(block, fa.wm, fa.wn, fm * IMMA_TILE, fnn * IMMA_TILE)?;
        let tid = self.thread_id(block)?;
        let warp_w = self.const_index(block, WARP)?;
        let lane = self.remui(block, tid, warp_w)?;
        let four = self.const_index(block, 4)?;
        let quad = self.divui(block, lane, four)?;
        let in_quad = self.remui(block, lane, four)?;
        let k_off = self.muli(block, in_quad, four)?;
        let two = self.const_index(block, 2)?;
        let d_col = self.muli(block, in_quad, two)?;
        let row0 = self.addi(block, m0, quad)?;
        let col0 = self.addi(block, n0, d_col)?;
        let w_row0 = self.addi(block, n0, quad)?;
        let at = |base: Value<'c, 'c>, count: i64| -> Result<Vec<Value<'c, 'c>>> {
            (0..count)
                .map(|t| {
                    let off = self.const_index(block, t * IMMA_TILE)?;
                    self.addi(block, base, off)
                })
                .collect()
        };
        Ok(ImmaLane {
            rows: at(row0, fm)?,
            cols: at(col0, fnn)?,
            w_rows: at(w_row0, fnn)?,
            k_off,
        })
    }

    /// Four bytes of an int8 operand's row as an `m8n8k16` A or B fragment.
    fn imma_word(
        &self,
        block: &Block<'c>,
        v: &MemVal<'c>,
        row: Value<'c, 'c>,
        col: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let word = if v.vectorizes(4) {
            self.vec_load_al(block, v.mem, &[row, col], Type::vector(&[4], self.i8_t), 4)?
        } else {
            let mut bytes = Vec::with_capacity(4);
            for i in 0..4 {
                let off = self.const_index(block, i)?;
                let c = self.addi(block, col, off)?;
                bytes.push(self.push(block, memref::load(v.mem, &[row, c], self.loc))?);
            }
            self.push(
                block,
                OperationBuilder::new("vector.from_elements", self.loc)
                    .add_operands(&bytes)
                    .add_results(&[Type::vector(&[4], self.i8_t)])
                    .build()?,
            )?
        };
        self.vec_shape_cast(block, word, Type::vector(&[1, 4], self.i8_t))
    }

    /// The fragments of `fa += f32(dot_t(a, w)) * s1 * s2` for int8 `a` and
    /// `w`, one scale a `[m, 1]` row broadcast and the other `[1, n]`. Each
    /// 8x8 tile's sum stays in i32 over the whole k, and its elements are
    /// converted, scaled and added to the accumulator in registers.
    /// Rebinds nothing.
    pub(super) fn frag_dot_scaled_raw(
        &mut self,
        block: &Block<'c>,
        fa: &FragAcc<'c>,
        tiles: &[MemVal<'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let [a, w, s1, s2] = tiles else {
            bail!("fragment scaled dot takes four operands");
        };
        if !fa.imma {
            bail!("internal: scaled dot on a 16x16 fragment accumulator");
        }
        let kk = a.shape[1];
        self.check_shapes(&a.shape, &[fa.m, kk], "fragment scaled dot lhs")?;
        self.check_shapes(&w.shape, &[fa.n, kk], "fragment scaled dot rhs")?;
        if a.elem != self.i8_t || w.elem != self.i8_t || kk % IMMA_K != 0 {
            bail!("fragment scaled dot contracts int8 operands over a k that is a multiple of {IMMA_K}");
        }
        for v in tiles {
            if v.is_masked() || v.swizzle.is_some() {
                bail!("internal: fragment scaled dot reached a partial or swizzled operand");
            }
        }
        let is_row = |s: &MemVal<'c>| s.shape == [fa.m, 1];
        let (fm, fnn) = fa.warp_frags();
        let lane = self.imma_lane(block, fa)?;
        let zero = self.const_index(block, 0)?;

        let steps = kk / IMMA_K;
        let mut k_cols = Vec::with_capacity(steps as usize);
        for h in 0..steps {
            let off = self.const_index(block, h * IMMA_K)?;
            k_cols.push(self.addi(block, lane.k_off, off)?);
        }
        let mut a_frags = Vec::new();
        for &row in &lane.rows {
            for &col in &k_cols {
                a_frags.push(self.imma_word(block, a, row, col)?);
            }
        }
        let mut w_frags = Vec::new();
        for &row in &lane.w_rows {
            for &col in &k_cols {
                w_frags.push(self.imma_word(block, w, row, col)?);
            }
        }

        // Each scale's value for every slot of the patch: a row scale by
        // patch row, a column scale by output column.
        let scale_at = |s: &MemVal<'c>| -> Result<Vec<Value<'c, 'c>>> {
            let mut by_slot = Vec::with_capacity((fm * fnn * 2) as usize);
            if is_row(s) {
                for &row in &lane.rows {
                    let v = self.load_as(block, s.mem, &[row, zero], self.f32_t)?;
                    by_slot.extend(std::iter::repeat_n(v, (fnn * 2) as usize));
                }
            } else {
                let mut by_col = Vec::with_capacity((fnn * 2) as usize);
                for &col in &lane.cols {
                    for dj in 0..2 {
                        let off = self.const_index(block, dj)?;
                        let c = self.addi(block, col, off)?;
                        by_col.push(self.load_as(block, s.mem, &[zero, c], self.f32_t)?);
                    }
                }
                for _ in 0..fm {
                    by_slot.extend_from_slice(&by_col);
                }
            }
            Ok(by_slot)
        };
        let first = scale_at(s1)?;
        let second = scale_at(s2)?;

        // The magic-constant conversion holds for |sum| < 2^22.
        let small = kk * 128 * 128 < 1 << 22;
        let acc_t = Type::vector(&[1, 2], self.i32_t);
        let zero_i = self.zero_scalar(block, self.i32_t)?;
        let empty = self.vec_broadcast(block, zero_i, acc_t)?;
        let shape = self.mma_shape(IMMA_TILE, IMMA_TILE, IMMA_K)?;
        let mut frags = fa.frags.clone();
        for r in 0..fm as usize {
            for c in 0..fnn as usize {
                let mut sum = empty;
                for h in 0..steps as usize {
                    let (af, wf) = (a_frags[r * steps as usize + h], w_frags[c * steps as usize + h]);
                    sum = self.mma_sync(block, af, wf, sum, shape, acc_t)?;
                }
                for dj in 0..2 {
                    let slot = (r * fnn as usize + c) * 2 + dj;
                    let raw = self.vec_extract(block, sum, &[0, dj as i64], self.i32_t)?;
                    let as_f = if small {
                        self.small_int_to_f32(block, raw)?
                    } else {
                        self.push(block, arith::sitofp(raw, self.f32_t, self.loc))?
                    };
                    let t = self.push(block, arith::mulf(as_f, first[slot], self.loc))?;
                    frags[slot] = self.elem_mac(block, self.f32_t, t, second[slot], frags[slot])?;
                }
            }
        }
        // Orders the reads before a shared operand is rewritten. Elided
        // when nothing is.
        self.barrier(block)?;
        Ok(frags)
    }

    /// `<tensor slice> = acc`: scatters each lane's fragment elements
    /// straight to the target, converting to its element type. Like
    /// [`Self::mma_sync_store_frags`], but without going through shared.
    pub(super) fn frag_store(
        &mut self,
        block: &Block<'c>,
        target: &MemVal<'c>,
        fa: &FragAcc<'c>,
    ) -> Result<()> {
        self.check_shapes(&[fa.m, fa.n], &target.shape, "fragment store")?;
        if fa.imma {
            let lane = self.imma_lane(block, fa)?;
            for (r, &row) in lane.rows.iter().enumerate() {
                for (c, &col) in lane.cols.iter().enumerate() {
                    for dj in 0..2 {
                        let off = self.const_index(block, dj)?;
                        let at = self.addi(block, col, off)?;
                        let e = fa.frags[(r * lane.cols.len() + c) * 2 + dj as usize];
                        let e = self.coerce(block, e, target.elem)?;
                        block.append_operation(memref::store(e, target.mem, &[row, at], self.loc));
                    }
                }
            }
            return Ok(());
        }
        let (fm, fnn) = fa.warp_frags();
        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        self.for_each_dfrag(block, (fm, fnn), m0, n0, |cg, i, elems| {
            for ([di, dj], addr) in elems {
                let e = cg.vec_extract(block, fa.frags[i], &[*di, *dj], cg.f32_t)?;
                let e = cg.coerce(block, e, target.elem)?;
                block.append_operation(memref::store(e, target.mem, addr, cg.loc));
            }
            Ok(())
        })
    }

}
