use super::*;
use crate::codegen::matmul::AOperand;

/// The A operand of an f16 fragment dot: a tile, or another accumulator's
/// fragments whose warps own the same rows.
pub(super) enum DotA<'c> {
    Tile(MemVal<'c>),
    Frags(FragAcc<'c>),
}

/// A `[rows, 1]` column an accumulator's rows read: a shared tile, or the
/// lane's own rows in registers (`RegLayout::MmaRows`).
pub(super) enum RowCol<'c> {
    Tile(MemVal<'c>),
    Regs(Vec<Value<'c, 'c>>),
}

/// The lane's slot in a fragment-row column for element row half `di` of
/// D fragment `i` in a warp patch `fnn` fragments wide: two rows a 16-row
/// block, in `for_each_dfrag`'s order.
pub(super) fn row_slot(i: usize, fnn: i64, di: i64) -> usize {
    i / (fnn as usize * 2) * 2 + di as usize
}

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

/// An operand of the int8 accumulator's updates: a tile in memory, or a
/// generated tile whose block each lane evaluates for the elements it holds.
pub(super) enum FragIn<'a, 'c> {
    Stored(ImmaOperand<'c>),
    Generated {
        ir: &'a crate::ir::Ir,
        op: crate::ir::OpId,
        shape: Vec<i64>,
        elem: crate::ir::Scalar,
    },
}

impl<'c> FragIn<'_, 'c> {
    fn shape(&self) -> &[i64] {
        match self {
            FragIn::Stored(v) => &v.view.shape,
            FragIn::Generated { shape, .. } => shape,
        }
    }

    fn stored(&self) -> Option<&MemVal<'c>> {
        match self {
            FragIn::Stored(v) => Some(&v.view),
            FragIn::Generated { .. } => None,
        }
    }
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
        col: &RowCol<'c>,
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
                let c = match col {
                    RowCol::Tile(t) => cg.load_as(block, t.mem, &[*row, zero], cg.f32_t)?,
                    RowCol::Regs(vals) => vals[row_slot(i, fnn, *di)],
                };
                let e = cg.vec_extract(block, frags[i], &[*di, *dj], cg.f32_t)?;
                let r = cg.push(block, cg.elem_arith(op, cg.f32_t, e, c)?)?;
                frags[i] = cg.vec_insert(block, r, frags[i], &[*di, *dj])?;
            }
            Ok(())
        })?;
        Ok(frags)
    }

    /// The fragments of `fa += dot(a, b)`, or of `fa += dot_t(a, b)` with
    /// `transpose`. Rebinds nothing.
    pub(super) fn frag_dot_raw(
        &mut self,
        block: &Block<'c>,
        fa: &FragAcc<'c>,
        a: &DotA<'c>,
        b: &MemVal<'c>,
        transpose: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        if fa.imma {
            bail!("internal: f16 dot into an int8 fragment accumulator");
        }
        let kk = match a {
            DotA::Tile(a) => {
                self.check_shapes(&a.shape, &[fa.m, a.shape[1]], "fragment dot lhs")?;
                a.shape[1]
            }
            DotA::Frags(src) => {
                if src.imma || src.m != fa.m || src.wm != fa.wm || src.wn != 1 || fa.wn != 1 {
                    bail!("internal: a fragment operand whose warps do not own the dot's rows");
                }
                src.n
            }
        };
        let b_want = if transpose { [fa.n, kk] } else { [kk, fa.n] };
        self.check_shapes(&b.shape, &b_want, "fragment dot rhs")?;
        let (fm, fnn) = fa.warp_frags();

        let (a_buf, a_kept) = match a {
            DotA::Tile(a) => {
                let (buf, kept) = self.frag_dot_operand(block, a)?;
                (Some(buf), kept)
            }
            DotA::Frags(_) => (None, true),
        };
        let (b_buf, b_kept) = self.frag_dot_operand(block, b)?;
        if !(a_kept && b_kept) {
            self.barrier(block)?;
        }

        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        let a_op = match (a, &a_buf) {
            (DotA::Frags(src), _) => AOperand::Frags(src),
            (DotA::Tile(_), Some(buf)) => AOperand::Staged(buf),
            (DotA::Tile(_), None) => unreachable!("a tile operand is staged or kept"),
        };
        let finals = self.mma_sync_mac(
            block,
            a_op,
            &b_buf,
            (kk, fm, fnn),
            m0,
            n0,
            &fa.frags,
            transpose,
        )?;

        // Nothing to publish, but the barrier orders the ldmatrix reads
        // before the staging, or the tile read in place, is written again.
        // Hoisted and in-place buffers are not released.
        self.barrier(block)?;
        if let (false, Some(a_buf)) = (a_kept, &a_buf) {
            self.release(a_buf);
        }
        if !b_kept {
            self.release(&b_buf);
        }
        Ok(finals)
    }

    /// The buffer `ldmatrix` reads for an operand of the f16 fragment dot,
    /// and whether it is kept rather than staged here: the operand itself
    /// when it is an f16 shared tile laid out for the reads, else the
    /// preheader's copy or a fresh swizzled one.
    fn frag_dot_operand(&mut self, block: &Block<'c>, v: &MemVal<'c>) -> Result<(MemVal<'c>, bool)> {
        if self.ldmatrix_reads_in_place(v) {
            return Ok((v.clone(), true));
        }
        self.dot_stage(block, v, &v.shape.clone(), true)
    }

    /// Whether `ldmatrix` can read `v` where it is, as fast as a staged
    /// copy: f16 in shared memory, unmasked and unpadded, every row on a
    /// 16-byte boundary, and swizzled, since unswizzled rows of a multiple
    /// of 128 bytes conflict eight ways.
    fn ldmatrix_reads_in_place(&self, v: &MemVal<'c>) -> bool {
        let [_, cols] = v.shape[..] else { return false };
        v.shared
            && v.elem == self.f16_t
            && v.swizzle.is_some()
            && v.row_stride.is_none()
            && !v.is_masked()
            && cols != DYN
            && cols % 8 == 0
            && (v.align_div == 0 || v.align_div % 8 == 0)
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

    /// Whether an operand holds int8.
    fn frag_in_is_i8(&self, v: &FragIn<'_, 'c>) -> bool {
        match v {
            FragIn::Stored(v) => v.view.elem == self.i8_t,
            FragIn::Generated { elem, .. } => *elem == crate::ir::Scalar::I8,
        }
    }

    /// [`Self::imma_frags`] of either kind of operand.
    fn frag_in_frags(
        &mut self,
        block: &Block<'c>,
        v: &FragIn<'_, 'c>,
        rows: &[Value<'c, 'c>],
        origin: Value<'c, 'c>,
        lane: Value<'c, 'c>,
        k_cols: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        match v {
            FragIn::Stored(v) => self.imma_frags(block, v, rows, origin, lane, k_cols),
            FragIn::Generated { ir, op, .. } => self.generated_frags(block, ir, *op, rows, k_cols),
        }
    }

    /// [`Self::imma_pair`] of either kind of operand.
    fn frag_in_pair(
        &mut self,
        block: &Block<'c>,
        v: &FragIn<'_, 'c>,
        row: Value<'c, 'c>,
        col: Value<'c, 'c>,
    ) -> Result<[Value<'c, 'c>; 2]> {
        match v {
            FragIn::Stored(v) => self.imma_pair(block, v, row, col),
            FragIn::Generated { ir, op, .. } => {
                let one = self.const_index(block, 1)?;
                let next = self.addi(block, col, one)?;
                let first = self.eval_generate(block, ir, *op, row, col)?;
                Ok([self.coerce(block, first, self.f32_t)?, {
                    let second = self.eval_generate(block, ir, *op, row, next)?;
                    self.coerce(block, second, self.f32_t)?
                }])
            }
        }
    }

    /// Element `[row, col]` of either kind of operand, as f32.
    fn frag_in_elem(
        &mut self,
        block: &Block<'c>,
        v: &FragIn<'_, 'c>,
        row: Value<'c, 'c>,
        col: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        match v {
            FragIn::Stored(v) => {
                let (mem, at) = self.imma_at(block, v, row, col)?;
                self.load_as(block, mem, &at, self.f32_t)
            }
            FragIn::Generated { ir, op, .. } => {
                let e = self.eval_generate(block, ir, *op, row, col)?;
                self.coerce(block, e, self.f32_t)
            }
        }
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
        r: &FragIn<'_, 'c>,
        c: &FragIn<'_, 'c>,
        negate: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        if !fa.imma {
            bail!("internal: an outer product on a 16x16 fragment accumulator");
        }
        self.check_shapes(r.shape(), &[fa.m, 1], "fragment outer product column")?;
        self.check_shapes(c.shape(), &[1, fa.n], "fragment outer product row")?;
        let (fm, fnn) = fa.warp_frags();
        let lane = self.imma_lane(block, fa)?;
        let zero = self.const_index(block, 0)?;
        let mut cols = Vec::with_capacity((fnn * 2) as usize);
        for &col in &lane.cols {
            cols.extend(self.frag_in_pair(block, c, zero, col)?);
        }
        let mut frags = fa.frags.clone();
        for i in 0..fm as usize {
            let mut row = self.frag_in_elem(block, r, lane.rows[i], zero)?;
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
        [a, w, s1, s2]: [&FragIn<'_, 'c>; 4],
    ) -> Result<Vec<Value<'c, 'c>>> {
        if !fa.imma {
            bail!("internal: scaled dot on a 16x16 fragment accumulator");
        }
        let kk = a.shape()[1];
        self.check_shapes(a.shape(), &[fa.m, kk], "fragment scaled dot lhs")?;
        self.check_shapes(w.shape(), &[fa.n, kk], "fragment scaled dot rhs")?;
        if !(self.frag_in_is_i8(a) && self.frag_in_is_i8(w)) || kk % IMMA_K != 0 {
            bail!("fragment scaled dot contracts int8 operands over a k that is a multiple of {IMMA_K}");
        }
        for v in [a, w, s1, s2].into_iter().filter_map(FragIn::stored) {
            if v.is_masked() {
                bail!("internal: fragment scaled dot reached a partial operand");
            }
        }
        let is_row = |s: &FragIn<'_, 'c>| s.shape() == [fa.m, 1];
        let (fm, fnn) = fa.warp_frags();
        let lane = self.imma_lane(block, fa)?;
        let zero = self.const_index(block, 0)?;

        let steps = kk / IMMA_K;
        let mut k_cols = Vec::with_capacity(steps as usize);
        for h in 0..steps {
            let off = self.const_index(block, h * IMMA_K)?;
            k_cols.push(self.addi(block, lane.k_off, off)?);
        }
        let a_frags = self.frag_in_frags(block, a, &lane.rows, lane.m0, lane.lane, &k_cols)?;
        let w_frags = self.frag_in_frags(block, w, &lane.w_rows, lane.n0, lane.lane, &k_cols)?;

        // A column scale's values for the patch's output columns, loaded up
        // front; a row scale is loaded with its patch row, as qmma_t does.
        let cols_of = |cg: &mut Self, s: &FragIn<'_, 'c>| -> Result<Vec<Value<'c, 'c>>> {
            let mut by_col = Vec::with_capacity((fnn * 2) as usize);
            for &col in &lane.cols {
                by_col.extend(cg.frag_in_pair(block, s, zero, col)?);
            }
            Ok(by_col)
        };
        let first_cols = if is_row(s1) { Vec::new() } else { cols_of(self, s1)? };
        let second_cols = if is_row(s2) { Vec::new() } else { cols_of(self, s2)? };

        // The magic-constant conversion holds for |sum| < 2^22.
        let small = kk * 128 * 128 < 1 << 22;
        let acc_t = Type::vector(&[1, 2], self.i32_t);
        let zero_i = self.zero_scalar(block, self.i32_t)?;
        let empty = self.vec_broadcast(block, zero_i, acc_t)?;
        let shape = self.mma_shape(IMMA_TILE, IMMA_TILE, IMMA_K)?;
        let row_of = |cg: &mut Self, s: &FragIn<'_, 'c>, r: usize| -> Result<Option<Value<'c, 'c>>> {
            if !is_row(s) {
                return Ok(None);
            }
            Ok(Some(cg.frag_in_elem(block, s, lane.rows[r], zero)?))
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
        // A lane's two adjacent columns go out as one store where the rows
        // keep them aligned. A swizzle moves whole granules, so the pair
        // stays adjacent.
        let paired = target.vectorizes(2);
        let pair_t = Type::vector(&[2], target.elem);
        let bytes = 2 * i64::from(self.elem_bytes(target.elem).unwrap_or(4));
        self.for_each_dfrag(block, (fm, fnn), m0, n0, |cg, i, elems| {
            for pair in elems.chunks(2) {
                let ([di, _], [row, col]) = pair[0];
                let col = cg.swizzle_col(block, target, row, col)?;
                let vals = pair
                    .iter()
                    .map(|([di, dj], _)| {
                        let e = cg.vec_extract(block, fa.frags[i], &[*di, *dj], cg.f32_t)?;
                        cg.coerce(block, e, target.elem)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if paired && pair.len() == 2 && pair[1].0 == [di, 1] {
                    let v = cg.vec_broadcast(block, vals[0], pair_t)?;
                    let v = cg.vec_insert(block, vals[1], v, &[1])?;
                    cg.vec_store_al(block, v, target.mem, &[row, col], bytes)?;
                    continue;
                }
                for (k, &e) in vals.iter().enumerate() {
                    let off = cg.const_index(block, k as i64)?;
                    let at = cg.addi(block, col, off)?;
                    block.append_operation(memref::store(e, target.mem, &[row, at], cg.loc));
                }
            }
            Ok(())
        })
    }

}
