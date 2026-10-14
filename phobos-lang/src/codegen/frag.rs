use super::*;

/// An operand of the scaled dot. When it is a slice, `from` is the buffer
/// it was sliced from and where it starts there, which is what is read: a
/// slice taken each iteration has its offset rebuilt per load, and the
/// buffer it came from does not.
pub(super) struct ImmaOperand<'c> {
    pub(super) view: MemVal<'c>,
    pub(super) from: Option<SliceOrigin<'c>>,
}

/// Where a slice starts in the buffer it was taken from, and that buffer's
/// swizzle.
pub(super) struct SliceOrigin<'c> {
    pub(super) mem: Value<'c, 'c>,
    pub(super) starts: [Value<'c, 'c>; 2],
    pub(super) swizzle: Option<Swizzle>,
}

/// A scaled dot's column scales for the patch's output columns, two a
/// lane's column, and whether its sums convert exactly through the magic
/// constant. See [`Codegen::frag_dot_scaled_raw`].
struct ScaledCols<'a, 'c> {
    first: &'a [Value<'c, 'c>],
    second: &'a [Value<'c, 'c>],
    small: bool,
}

/// The weight of a scaled dot: an operand in memory, or a generated tile
/// whose block each lane evaluates for the fragment words it holds.
pub(super) enum ImmaWeight<'a, 'c> {
    Stored(ImmaOperand<'c>),
    Generated { ir: &'a crate::ir::Ir, op: crate::ir::OpId },
}

/// See [`Codegen::imma_lane`].
struct ImmaLane<'c> {
    rows: Vec<Value<'c, 'c>>,
    cols: Vec<Value<'c, 'c>>,
    w_rows: Vec<Value<'c, 'c>>,
    k_off: Value<'c, 'c>,
    /// The warp's patch origin, and the lane's row of an `ldmatrix` over
    /// four tiles from it.
    m0: Value<'c, 'c>,
    n0: Value<'c, 'c>,
    lane: Value<'c, 'c>,
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
            m0,
            n0,
            lane,
        })
    }

    /// Where element `[row, col]` of an operand is read: its memref and
    /// indices into it.
    fn imma_at(
        &self,
        block: &Block<'c>,
        v: &ImmaOperand<'c>,
        row: Value<'c, 'c>,
        col: Value<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, [Value<'c, 'c>; 2])> {
        let (mem, swizzle, [row, col]) = match &v.from {
            Some(f) => (f.mem, f.swizzle, [self.addi(block, f.starts[0], row)?, self.addi(block, f.starts[1], col)?]),
            None => (v.view.mem, v.view.swizzle, [row, col]),
        };
        // A fragment's word, and an ldmatrix row, sit inside one granule.
        Ok((mem, [row, self.swizzle_by(block, swizzle, row, col)?]))
    }

    /// Four bytes of an int8 operand's row as an `m8n8k16` A or B fragment.
    fn imma_word(
        &self,
        block: &Block<'c>,
        v: &ImmaOperand<'c>,
        row: Value<'c, 'c>,
        col: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let (mem, at) = self.imma_at(block, v, row, col)?;
        let word = if v.view.vectorizes(4) {
            self.vec_load_al(block, mem, &at, Type::vector(&[4], self.i8_t), 4)?
        } else {
            let mut bytes = Vec::with_capacity(4);
            for i in 0..4 {
                let off = self.const_index(block, i)?;
                let c = self.addi(block, at[1], off)?;
                bytes.push(self.push(block, memref::load(mem, &[at[0], c], self.loc))?);
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

    /// A lane's two adjacent columns of an operand as f32, from an even
    /// `col`: one 8-byte load when the view keeps them aligned. The pair
    /// sits inside one swizzle granule, as a fragment's word does.
    fn imma_pair(
        &self,
        block: &Block<'c>,
        v: &ImmaOperand<'c>,
        row: Value<'c, 'c>,
        col: Value<'c, 'c>,
    ) -> Result<[Value<'c, 'c>; 2]> {
        if v.view.elem == self.f32_t && v.view.vectorizes(2) {
            let (mem, at) = self.imma_at(block, v, row, col)?;
            let pair = self.vec_load_al(block, mem, &at, Type::vector(&[2], self.f32_t), 8)?;
            return Ok([self.vec_extract(block, pair, &[0], self.f32_t)?, self.vec_extract(block, pair, &[1], self.f32_t)?]);
        }
        let mut pair = Vec::with_capacity(2);
        for dj in 0..2 {
            let off = self.const_index(block, dj)?;
            let c = self.addi(block, col, off)?;
            let (mem, at) = self.imma_at(block, v, row, c)?;
            pair.push(self.load_as(block, mem, &at, self.f32_t)?);
        }
        Ok([pair[0], pair[1]])
    }

    /// An operand's fragments for its `rows.len()` tiles from row `origin`,
    /// tile-major then by k step. From shared memory four tiles at a time
    /// are one `ldmatrix.x4` a k step, lane `l` addressing row `origin + l`
    /// of the four: its non-transposed 16-bit layout is the `m8n8k16`
    /// fragment's. Otherwise a word load each.
    fn imma_frags(
        &self,
        block: &Block<'c>,
        v: &ImmaOperand<'c>,
        rows: &[Value<'c, 'c>],
        origin: Value<'c, 'c>,
        lane: Value<'c, 'c>,
        k_cols: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let steps = k_cols.len();
        let frag_t = Type::vector(&[1, 4], self.i8_t);
        let ldmatrix = v.view.shared && v.view.vectorizes(16);
        if !(ldmatrix && rows.len().is_multiple_of(4)) {
            let mut frags = Vec::with_capacity(rows.len() * steps);
            for &row in rows {
                for &col in k_cols {
                    frags.push(self.imma_word(block, v, row, col)?);
                }
            }
            return Ok(frags);
        }
        let quad_t = Type::vector(&[4, 4], self.i8_t);
        let four_i8 = Type::vector(&[4], self.i8_t);
        let mut frags = vec![None; rows.len() * steps];
        for g in 0..rows.len() / 4 {
            let off = self.const_index(block, (4 * g) as i64 * IMMA_TILE)?;
            let first = self.addi(block, origin, off)?;
            let row = self.addi(block, first, lane)?;
            for h in 0..steps {
                let col = self.const_index(block, h as i64 * IMMA_K)?;
                let (mem, at) = self.imma_at(block, v, row, col)?;
                let quad = self.ldmatrix(block, mem, at, 4, false, quad_t)?;
                for t in 0..4 {
                    let word = self.vec_extract(block, quad, &[t as i64], four_i8)?;
                    frags[(4 * g + t) * steps + h] = Some(self.vec_shape_cast(block, word, frag_t)?);
                }
            }
        }
        Ok(frags.into_iter().map(|f| f.expect("every tile and step loaded")).collect())
    }

    /// The fragments of `fa += r * c`, or `fa -= r * c` with `negate`, for a
    /// `[m, 1]` column `r` and a `[1, n]` row `c`: each element adds its
    /// row's value of `r` times its column's of `c`. Rebinds nothing.
    pub(super) fn frag_outer_raw(
        &mut self,
        block: &Block<'c>,
        fa: &FragAcc<'c>,
        r: &ImmaOperand<'c>,
        c: &ImmaOperand<'c>,
        negate: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        if !fa.imma {
            bail!("internal: an outer product on a 16x16 fragment accumulator");
        }
        self.check_shapes(&r.view.shape, &[fa.m, 1], "fragment outer product column")?;
        self.check_shapes(&c.view.shape, &[1, fa.n], "fragment outer product row")?;
        let (fm, fnn) = fa.warp_frags();
        let lane = self.imma_lane(block, fa)?;
        let zero = self.const_index(block, 0)?;
        let mut cols = Vec::with_capacity((fnn * 2) as usize);
        for &col in &lane.cols {
            cols.extend(self.imma_pair(block, c, zero, col)?);
        }
        let mut frags = fa.frags.clone();
        for i in 0..fm as usize {
            let (mem, at) = self.imma_at(block, r, lane.rows[i], zero)?;
            let mut row = self.load_as(block, mem, &at, self.f32_t)?;
            if negate {
                row = self.push(block, arith::negf(row, self.loc))?;
            }
            for (j, &col) in cols.iter().enumerate() {
                let slot = i * cols.len() + j;
                frags[slot] = self.elem_mac(block, self.f32_t, row, col, frags[slot])?;
            }
        }
        Ok(frags)
    }

    /// The fragments of `fa += f32(dot_t(a, w)) * s1 * s2` for int8 `a` and
    /// `w`, one scale a `[m, 1]` row broadcast and the other `[1, n]`. Each
    /// 8x8 tile's sum stays in i32 over the whole k, and its elements are
    /// converted and added as `acc + d * (s1 * s2)` in registers. With the
    /// scale product independent of the mma, ptxas issues every operand load
    /// ahead of the first one. Rebinds nothing.
    pub(super) fn frag_dot_scaled_raw(
        &mut self,
        block: &Block<'c>,
        fa: &FragAcc<'c>,
        a: &ImmaOperand<'c>,
        w: &ImmaWeight<'_, 'c>,
        [s1, s2]: [&ImmaOperand<'c>; 2],
    ) -> Result<Vec<Value<'c, 'c>>> {
        if !fa.imma {
            bail!("internal: scaled dot on a 16x16 fragment accumulator");
        }
        let kk = a.view.shape[1];
        self.check_shapes(&a.view.shape, &[fa.m, kk], "fragment scaled dot lhs")?;
        let (w_shape, w_i8) = match w {
            ImmaWeight::Stored(w) => (w.view.shape.clone(), w.view.elem == self.i8_t),
            ImmaWeight::Generated { ir, op } => match ir.ty(ir.result(*op)) {
                crate::ir::Type::Dist(d) => (d.shape.clone(), d.elem == crate::ir::Scalar::I8),
                _ => bail!("internal: a generated weight that is not a generated tile"),
            },
        };
        self.check_shapes(&w_shape, &[fa.n, kk], "fragment scaled dot rhs")?;
        if a.view.elem != self.i8_t || !w_i8 || kk % IMMA_K != 0 {
            bail!("fragment scaled dot contracts int8 operands over a k that is a multiple of {IMMA_K}");
        }
        let stored = match w {
            ImmaWeight::Stored(w) => Some(&w.view),
            ImmaWeight::Generated { .. } => None,
        };
        for v in [Some(&a.view), stored, Some(&s1.view), Some(&s2.view)].into_iter().flatten() {
            if v.is_masked() {
                bail!("internal: fragment scaled dot reached a partial operand");
            }
        }
        let is_row = |s: &ImmaOperand<'c>| s.view.shape == [fa.m, 1];
        let (fm, fnn) = fa.warp_frags();
        let lane = self.imma_lane(block, fa)?;
        let zero = self.const_index(block, 0)?;

        let steps = kk / IMMA_K;
        let mut k_cols = Vec::with_capacity(steps as usize);
        for h in 0..steps {
            let off = self.const_index(block, h * IMMA_K)?;
            k_cols.push(self.addi(block, lane.k_off, off)?);
        }
        let a_frags = self.imma_frags(block, a, &lane.rows, lane.m0, lane.lane, &k_cols)?;
        let w_frags = match w {
            ImmaWeight::Stored(w) => self.imma_frags(block, w, &lane.w_rows, lane.n0, lane.lane, &k_cols)?,
            ImmaWeight::Generated { ir, op } => self.generated_frags(block, ir, *op, &lane.w_rows, &k_cols)?,
        };

        // A column scale's values for the patch's output columns, loaded up
        // front; a row scale is loaded with its patch row, as qmma_t does.
        let cols_of = |s: &ImmaOperand<'c>| -> Result<Vec<Value<'c, 'c>>> {
            let mut by_col = Vec::with_capacity((fnn * 2) as usize);
            for &col in &lane.cols {
                by_col.extend(self.imma_pair(block, s, zero, col)?);
            }
            Ok(by_col)
        };
        let first_cols = if is_row(s1) { Vec::new() } else { cols_of(s1)? };
        let second_cols = if is_row(s2) { Vec::new() } else { cols_of(s2)? };

        // The magic-constant conversion holds for |sum| < 2^22.
        let small = kk * 128 * 128 < 1 << 22;
        let acc_t = Type::vector(&[1, 2], self.i32_t);
        let zero_i = self.zero_scalar(block, self.i32_t)?;
        let empty = self.vec_broadcast(block, zero_i, acc_t)?;
        let shape = self.mma_shape(IMMA_TILE, IMMA_TILE, IMMA_K)?;
        let row_of = |cg: &Self, s: &ImmaOperand<'c>, r: usize| -> Result<Option<Value<'c, 'c>>> {
            if !is_row(s) {
                return Ok(None);
            }
            let (mem, at) = cg.imma_at(block, s, lane.rows[r], zero)?;
            Ok(Some(cg.load_as(block, mem, &at, cg.f32_t)?))
        };
        let scales = ScaledCols {
            first: &first_cols,
            second: &second_cols,
            small,
        };
        let mut frags = fa.frags.clone();
        let (fm, fnn, steps) = (fm as usize, fnn as usize, steps as usize);
        if self.int8_mma_k() == 32 && fm % 2 == 0 {
            // From Blackwell on, `m8n8k16` runs as a half-used `m16n8k16`.
            // Two row tiles over two k steps go through one `m16n8k32`
            // instead, or over one step through a whole `m16n8k16`, its
            // fragments the `m8n8k16` ones regrouped as `qgemm_t`'s: A as
            // rows (g, g + 8) of the first step, then of the second, B as
            // the first step, then the second. Its two rows of sums are the
            // two tiles' own.
            let ks = if steps % 2 == 0 { 2 } else { 1 };
            let four_i8 = Type::vector(&[4], self.i8_t);
            let a_t = Type::vector(&[2 * ks as u64, 4], self.i8_t);
            let b_t = Type::vector(&[ks as u64, 4], self.i8_t);
            let d_t = Type::vector(&[2, 2], self.i32_t);
            let wide = self.mma_shape(2 * IMMA_TILE, IMMA_TILE, ks as i64 * IMMA_K)?;
            let zero_d = self.vec_broadcast(block, zero_i, d_t)?;
            let zero_i8 = self.zero_scalar(block, self.i8_t)?;
            for rp in 0..fm / 2 {
                let rows = [2 * rp, 2 * rp + 1];
                let row_scales = [
                    [row_of(self, s1, rows[0])?, row_of(self, s2, rows[0])?],
                    [row_of(self, s1, rows[1])?, row_of(self, s2, rows[1])?],
                ];
                for c in 0..fnn {
                    let mut d = zero_d;
                    for hp in 0..steps / ks {
                        let mut a = self.vec_broadcast(block, zero_i8, a_t)?;
                        let mut b = self.vec_broadcast(block, zero_i8, b_t)?;
                        for i in 0..ks {
                            let h = ks * hp + i;
                            for (j, &r) in rows.iter().enumerate() {
                                let piece = self.vec_shape_cast(block, a_frags[r * steps + h], four_i8)?;
                                a = self.vec_insert(block, piece, a, &[(2 * i + j) as i64])?;
                            }
                            let piece = self.vec_shape_cast(block, w_frags[c * steps + h], four_i8)?;
                            b = self.vec_insert(block, piece, b, &[i as i64])?;
                        }
                        d = self.mma_sync(block, a, b, d, wide, d_t)?;
                    }
                    for (j, &r) in rows.iter().enumerate() {
                        let slot = (r * fnn + c) * 2;
                        let sums = (d, j as i64);
                        self.scaled_dot_epilogue(block, &scales, row_scales[j], c, sums, &mut frags[slot..slot + 2])?;
                    }
                }
            }
        } else {
            for r in 0..fm {
                let row_scales = [row_of(self, s1, r)?, row_of(self, s2, r)?];
                for c in 0..fnn {
                    let mut sum = empty;
                    for h in 0..steps {
                        let (af, wf) = (a_frags[r * steps + h], w_frags[c * steps + h]);
                        sum = self.mma_sync(block, af, wf, sum, shape, acc_t)?;
                    }
                    let slot = (r * fnn + c) * 2;
                    self.scaled_dot_epilogue(block, &scales, row_scales, c, (sum, 0), &mut frags[slot..slot + 2])?;
                }
            }
        }
        // Orders the reads before a shared operand is rewritten. Elided
        // when nothing is.
        self.barrier(block)?;
        Ok(frags)
    }

    /// A lane's two sums of one 8x8 tile in patch column `c`, row `j` of the
    /// mma's `d`, into its two accumulator elements, as
    /// `acc + f32(d) * (s1 * s2)`: a row scale from `rows`, a column scale
    /// from `cols`.
    fn scaled_dot_epilogue(
        &mut self,
        block: &Block<'c>,
        cols: &ScaledCols<'_, 'c>,
        rows: [Option<Value<'c, 'c>>; 2],
        c: usize,
        (d, j): (Value<'c, 'c>, i64),
        acc: &mut [Value<'c, 'c>],
    ) -> Result<()> {
        for (dj, acc) in acc.iter_mut().enumerate() {
            let raw = self.vec_extract(block, d, &[j, dj as i64], self.i32_t)?;
            let as_f = if cols.small {
                self.small_int_to_f32(block, raw)?
            } else {
                self.push(block, arith::sitofp(raw, self.f32_t, self.loc))?
            };
            let first = rows[0].unwrap_or_else(|| cols.first[c * 2 + dj]);
            let second = rows[1].unwrap_or_else(|| cols.second[c * 2 + dj]);
            let t = self.push(block, arith::mulf(first, second, self.loc))?;
            *acc = self.elem_mac(block, self.f32_t, as_f, t, *acc)?;
        }
        Ok(())
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
            // A lane's two adjacent columns go out as one 8-byte store when
            // the rows keep them aligned, as `qgemm_t`'s do.
            let paired = target.elem == self.f32_t && target.vectorizes(2) && target.swizzle.is_none();
            let pair_t = Type::vector(&[2], self.f32_t);
            for (r, &row) in lane.rows.iter().enumerate() {
                for (c, &col) in lane.cols.iter().enumerate() {
                    let slot = (r * lane.cols.len() + c) * 2;
                    if paired {
                        let v = self.vec_broadcast(block, fa.frags[slot], pair_t)?;
                        let v = self.vec_insert(block, fa.frags[slot + 1], v, &[1])?;
                        self.vec_store_al(block, v, target.mem, &[row, col], 8)?;
                        continue;
                    }
                    for dj in 0..2 {
                        let off = self.const_index(block, dj)?;
                        let at = self.addi(block, col, off)?;
                        let e = fa.frags[slot + dj as usize];
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
