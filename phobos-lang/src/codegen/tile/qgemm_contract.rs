// The contraction of a `qgemm_t` stage: each warp's 32 x 32 patch over the
// stage's four 32-element groups, through the integer tensor cores, then the
// scale epilogue.
//
// Where the chip has `m16n8k32`, one `ldmatrix.x4` loads a whole A fragment,
// 16 rows over both 16-element halves of a group, and another two whole B
// fragments of 8 columns, so the registers it fills are the operands as they
// stand. Before sm_80, `m8n8k16` takes each 8 x 8 tile and half apart.

use super::qgemm::{CHUNK_BYTES, Lanes, PATCH, QGEMM_KT, QgFormat};
use super::*;

/// A group's `m16n8k32` operands: the A quad of each 16-row pair, then the
/// B quad of each pair of column tiles.
type Quads<'c> = (Vec<Value<'c, 'c>>, Vec<Value<'c, 'c>>);

impl<'c> Codegen<'c> {
    /// The warp's patch over one stage: four 32-element groups, each with two
    /// `ldmatrix.x4` per operand, the patch's tensor tiles, and the scale
    /// epilogue. `sa` is the activation scale stage, and whether each scale
    /// has the group's activation sum beside it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qgemm_contract(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        lanes: &Lanes<'c>,
        a: (&MemVal<'c>, Option<Value<'c, 'c>>),
        w_st: &MemVal<'c>,
        sa: (&MemVal<'c>, bool),
        sw_st: &MemVal<'c>,
        planes: (Option<&MemVal<'c>>, Option<&MemVal<'c>>),
        accs: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let (mw_st, as_st) = planes;
        let (sa_st, pairs) = sa;
        let (i32_t, f32_t) = (self.i32_t, self.f32_t);
        let acc_t = Type::vector(&[1, 2], i32_t);
        let two_f32 = Type::vector(&[2], f32_t);
        let zero_i = self.zero_scalar(block, i32_t)?;
        let empty = self.vec_broadcast(block, zero_i, acc_t)?;
        let halves = Q8_BLOCK / IMMA_K;
        let split = fmt.scale_run() == IMMA_K;
        let per_col = if split { halves } else { 1 };
        let (rows, cols) = self.qgemm_patch_coords(block, lanes)?;
        let chunk_w = self.const_index(block, CHUNK_BYTES)?;
        // The activation stage, and the half of it this turn reads.
        let (a_st, a_half) = a;
        let k32 = lanes.k32.as_ref();
        let a_row = k32.map_or(lanes.ldm_a_row, |l| l.a_row);
        let a_row = match a_half {
            Some(half) => self.addi(block, a_row, half)?,
            None => a_row,
        };
        // Each operand's stage as bytes, and the lane's base in it: its row,
        // with its swizzle phase (and the half it reads, for whole
        // fragments) in the chunk bits. A row is a whole bank period, so a
        // group's chunk is the base xor a constant.
        let a_flat = self.tile_flat(block, a_st)?;
        let w_flat = self.tile_flat(block, w_st)?;
        let kt = self.const_index(block, QGEMM_KT)?;
        let base = |cg: &mut Self, row: Value<'c, 'c>, phase: Value<'c, 'c>| -> Result<Value<'c, 'c>> {
            let row_b = cg.muli(block, row, kt)?;
            let phase_b = cg.muli(block, phase, chunk_w)?;
            cg.addi(block, row_b, phase_b)
        };
        let (a_base, w_base) = match k32 {
            Some(k32) => {
                let ax = self.push(block, arith::xori(k32.a_half, lanes.ldm_xor, self.loc))?;
                let wx = self.push(block, arith::xori(k32.w_half, lanes.ldm_xor, self.loc))?;
                (base(self, a_row, ax)?, base(self, k32.w_row, wx)?)
            }
            None => (base(self, a_row, lanes.ldm_xor)?, base(self, lanes.ldm_w_row, lanes.ldm_xor)?),
        };
        let zero = self.const_index(block, 0)?;

        let mut accs = accs.to_vec();
        for s in 0..(QGEMM_KT / Q8_BLOCK) {
            let quads = match k32 {
                Some(_) => Some(self.qgemm_quads_k32(block, (&a_flat, a_base), (&w_flat, w_base), s)?),
                None => None,
            };
            // The weight scales this group needs: one per column, or one per
            // column and half.
            let mut sw = Vec::with_capacity((PATCH * per_col) as usize);
            for col in &cols {
                for h in 0..per_col {
                    let s_idx = self.const_index(block, s * per_col + h)?;
                    sw.push(self.vec_load_al(block, sw_st.mem, &[s_idx, *col], two_f32, 8)?);
                }
            }
            let s_idx = self.const_index(block, s)?;
            // The activation scales, one per row, and with them the row sums
            // when the stage holds the two as a pair.
            let mut sa = Vec::with_capacity(PATCH as usize);
            let mut asum = Vec::with_capacity(PATCH as usize);
            for row in &rows {
                if pairs {
                    let col = self.const_index(block, 2 * s)?;
                    let pair = self.vec_load_al(block, sa_st.mem, &[*row, col], two_f32, 8)?;
                    sa.push(self.vec_extract(block, pair, &[0], f32_t)?);
                    asum.push(self.vec_extract(block, pair, &[1], f32_t)?);
                } else {
                    sa.push(self.push(block, memref::load(sa_st.mem, &[*row, s_idx], self.loc))?);
                }
            }
            // The minimum term's operands, if the format has one: column
            // minimums, loaded like the scales, and row sums from their own
            // plane, loaded like the activation scales.
            let mut mw = Vec::with_capacity(PATCH as usize);
            if let Some(mw_st) = mw_st {
                for col in &cols {
                    mw.push(self.vec_load_al(block, mw_st.mem, &[s_idx, *col], two_f32, 8)?);
                }
            }
            if let Some(as_st) = as_st {
                for row in &rows {
                    let raw = self.push(block, memref::load(as_st.mem, &[*row, s_idx], self.loc))?;
                    asum.push(self.small_int_to_f32(block, raw)?);
                }
            }

            // -(sa * asum) per row: the minimum term's factor, shared by
            // every column of the patch. With no sums from a plane or a pair,
            // the `m16n8k32` contraction finds each row pair's as it goes.
            let neg_min = |cg: &mut Self, r: usize, sum: Value<'c, 'c>| -> Result<Value<'c, 'c>> {
                let p = cg.push(block, arith::mulf(sa[r], sum, cg.loc))?;
                cg.push(block, arith::negf(p, cg.loc))
            };
            let mut neg_mins = Vec::with_capacity(PATCH as usize);
            for (r, &sum) in asum.iter().enumerate() {
                neg_mins.push(neg_min(self, r, sum)?);
            }

            // One tile's epilogue over its accumulators: one over both halves
            // when a scale covers the group, otherwise one per half, each the
            // vector and the row of it this tile's lane values sit in.
            let epilogue = |cg: &mut Self, accs: &mut [Value<'c, 'c>], neg_mins: &[Value<'c, 'c>], r: i64, c: i64, sums: &[(Value<'c, 'c>, i64)]| -> Result<()> {
                for dj in 0..2i64 {
                    let slot = ((r * PATCH + c) * 2 + dj) as usize;
                    let sw_of = |cg: &mut Self, h: usize| {
                        cg.vec_extract(block, sw[c as usize * per_col as usize + h], &[dj], f32_t)
                    };
                    let raw_of = |cg: &mut Self, h: usize| -> Result<Value<'c, 'c>> {
                        let (sum, row) = sums[h];
                        let raw = cg.vec_extract(block, sum, &[row, dj], i32_t)?;
                        cg.small_int_to_f32(block, raw)
                    };
                    accs[slot] = if split {
                        // acc += sa * (c0 * sw0 + c1 * sw1).
                        let (w0, w1) = (sw_of(cg, 0)?, sw_of(cg, 1)?);
                        let (c0, c1) = (raw_of(cg, 0)?, raw_of(cg, 1)?);
                        let inner = cg.push(block, arith::mulf(c0, w0, cg.loc))?;
                        let inner = cg.elem_mac(block, f32_t, c1, w1, inner)?;
                        cg.elem_mac(block, f32_t, inner, sa[r as usize], accs[slot])?
                    } else if mw_st.is_some() {
                        // acc += (sa * sw) * c - (sa * asum) * mw: the
                        // run's minimum, weighted by the activation's sum.
                        let w0 = sw_of(cg, 0)?;
                        let as_f = raw_of(cg, 0)?;
                        let mw0 = cg.vec_extract(block, mw[c as usize], &[dj], f32_t)?;
                        let scale = cg.push(block, arith::mulf(sa[r as usize], w0, cg.loc))?;
                        let acc = cg.elem_mac(block, f32_t, as_f, scale, accs[slot])?;
                        cg.elem_mac(block, f32_t, neg_mins[r as usize], mw0, acc)?
                    } else {
                        let w0 = sw_of(cg, 0)?;
                        let scale = cg.push(block, arith::mulf(sa[r as usize], w0, cg.loc))?;
                        let as_f = raw_of(cg, 0)?;
                        cg.elem_mac(block, f32_t, as_f, scale, accs[slot])?
                    };
                }
                Ok(())
            };
            if let Some((a4, w4)) = &quads {
                for (rp, &a) in a4.iter().enumerate() {
                    if mw_st.is_some() && asum.is_empty() {
                        for (j, sum) in self.qgemm_row_sums_k32(block, a)?.into_iter().enumerate() {
                            neg_mins.push(neg_min(self, 2 * rp + j, sum)?);
                        }
                    }
                    let r0 = 2 * rp as i64;
                    let each = |cg: &mut Self, j: i64, c: i64, sums: &[(Value<'c, 'c>, i64)]| {
                        epilogue(cg, &mut accs, &neg_mins, r0 + j, c, sums)
                    };
                    self.qgemm_pair_k32(block, a, w4, split, each)?;
                }
                continue;
            }

            let four_i8 = Type::vector(&[4], self.i8_t);
            let frag_t = Type::vector(&[1, 4], self.i8_t);
            let quad_t = Type::vector(&[PATCH as u64, 4], self.i8_t);
            let shape = self.mma_shape(IMMA_TILE, IMMA_TILE, IMMA_K)?;
            let (mut a_frags, mut w_frags) = (Vec::new(), Vec::new());
            for h in 0..halves {
                let off = self.const_index(block, (s * halves + h) * CHUNK_BYTES)?;
                let a_at = self.push(block, arith::xori(a_base, off, self.loc))?;
                let w_at = self.push(block, arith::xori(w_base, off, self.loc))?;
                let a4 = self.ldmatrix(block, a_flat.mem, [zero, a_at], PATCH, false, quad_t)?;
                let w4 = self.ldmatrix(block, w_flat.mem, [zero, w_at], PATCH, false, quad_t)?;
                for t in 0..PATCH {
                    let a = self.vec_extract(block, a4, &[t], four_i8)?;
                    a_frags.push(self.vec_shape_cast(block, a, frag_t)?);
                    let w = self.vec_extract(block, w4, &[t], four_i8)?;
                    w_frags.push(self.vec_shape_cast(block, w, frag_t)?);
                }
            }
            let frag = |h: i64, t: i64| (h * PATCH + t) as usize;
            // The tile's mma, then its epilogue, one tile at a time.
            for r in 0..PATCH {
                for c in 0..PATCH {
                    let mut sums = Vec::with_capacity(per_col as usize);
                    let mut sum = empty;
                    for h in 0..halves {
                        let seed = if split { empty } else { sum };
                        sum = self.mma_sync(block, a_frags[frag(h, r)], w_frags[frag(h, c)], seed, shape, acc_t)?;
                        if split {
                            sums.push((sum, 0));
                        }
                    }
                    if !split {
                        sums.push((sum, 0));
                    }
                    epilogue(self, &mut accs, &neg_mins, r, c, &sums)?;
                }
            }
        }
        Ok(accs)
    }

    /// Group `s`'s operands for `m16n8k32`: per 16-row pair of the patch,
    /// its A fragment over both halves of the group, and per pair of column
    /// tiles, both their B fragments, each one `ldmatrix.x4`. Each operand
    /// is its stage as bytes with the lane's base in it.
    fn qgemm_quads_k32(
        &mut self,
        block: &Block<'c>,
        a: (&MemVal<'c>, Value<'c, 'c>),
        w: (&MemVal<'c>, Value<'c, 'c>),
        s: i64,
    ) -> Result<Quads<'c>> {
        let quad_t = Type::vector(&[4, 4], self.i8_t);
        let zero = self.const_index(block, 0)?;
        // The group's chunk pair, from the lane's base; a row pair on is a
        // whole number of bank periods.
        let quads = |cg: &mut Self, (mem, base): (&MemVal<'c>, Value<'c, 'c>)| -> Result<Vec<Value<'c, 'c>>> {
            let off = cg.const_index(block, s * Q8_BLOCK)?;
            let at = cg.push(block, arith::xori(base, off, cg.loc))?;
            (0..PATCH / 2)
                .map(|p| {
                    let rows = cg.const_index(block, p * 2 * IMMA_TILE * QGEMM_KT)?;
                    let at = cg.addi(block, at, rows)?;
                    cg.ldmatrix(block, mem.mem, [zero, at], 4, false, quad_t)
                })
                .collect()
        };
        Ok((quads(self, a)?, quads(self, w)?))
    }

    /// The activation's sum over the group for the lane's two rows of a row
    /// pair, as f32: the pair's A fragment against a B of ones, whose `d`
    /// holds the two rows, every column alike.
    fn qgemm_row_sums_k32(&mut self, block: &Block<'c>, a: Value<'c, 'c>) -> Result<[Value<'c, 'c>; 2]> {
        let (i8_t, i32_t) = (self.i8_t, self.i32_t);
        let one = self.push(
            block,
            arith::constant(self.ctx, IntegerAttribute::new(i8_t, 1).into(), self.loc),
        )?;
        let ones = self.vec_broadcast(block, one, Type::vector(&[2, 4], i8_t))?;
        let d_t = Type::vector(&[2, 2], i32_t);
        let zero_i = self.zero_scalar(block, i32_t)?;
        let zero_d = self.vec_broadcast(block, zero_i, d_t)?;
        let shape = self.mma_shape(2 * IMMA_TILE, IMMA_TILE, Q8_BLOCK)?;
        let d = self.mma_sync(block, a, ones, zero_d, shape, d_t)?;
        let mut sums = [zero_i; 2];
        for (row, sum) in sums.iter_mut().enumerate() {
            let raw = self.vec_extract(block, d, &[row as i64, 0], i32_t)?;
            *sum = self.small_int_to_f32(block, raw)?;
        }
        Ok(sums)
    }

    /// One row pair of a group through `m16n8k32`, from its A quad `a` and
    /// the group's B quads, or `m16n8k16` per half for a format with a scale
    /// per half. Each 8 x 8 tile's sums go to `each` with its row in the
    /// pair and its column tile as soon as they are summed: the mma's `d`
    /// and the row of it the tile's values sit in, one per scale.
    fn qgemm_pair_k32(
        &mut self,
        block: &Block<'c>,
        a: Value<'c, 'c>,
        w4: &[Value<'c, 'c>],
        split: bool,
        mut each: impl FnMut(&mut Self, i64, i64, &[(Value<'c, 'c>, i64)]) -> Result<()>,
    ) -> Result<()> {
        let (i8_t, i32_t) = (self.i8_t, self.i32_t);
        let four_i8 = Type::vector(&[4], i8_t);
        let d_t = Type::vector(&[2, 2], i32_t);
        let zero_i = self.zero_scalar(block, i32_t)?;
        let zero_d = self.vec_broadcast(block, zero_i, d_t)?;
        let zero_i8 = self.zero_scalar(block, i8_t)?;
        let halves = Q8_BLOCK / IMMA_K;
        // `count` matrices of a quad from `first` on, as an operand.
        let pick = |cg: &mut Self, quad: Value<'c, 'c>, first: i64, count: i64| -> Result<Value<'c, 'c>> {
            let t = Type::vector(&[count as u64, 4], i8_t);
            let mut v = cg.vec_broadcast(block, zero_i8, t)?;
            for i in 0..count {
                let piece = cg.vec_extract(block, quad, &[first + i], four_i8)?;
                v = cg.vec_insert(block, piece, v, &[i])?;
            }
            Ok(v)
        };
        for c in 0..PATCH {
            // B's quad holds column tile 2p at both halves, then 2p + 1.
            let (w, b0) = (w4[(c / 2) as usize], 2 * (c % 2));
            let mut sums0 = Vec::new();
            let mut sums1 = Vec::new();
            if split {
                // A's quad is both row tiles at the first half, then at
                // the second.
                let shape = self.mma_shape(2 * IMMA_TILE, IMMA_TILE, IMMA_K)?;
                for h in 0..halves {
                    let a = pick(self, a, 2 * h, 2)?;
                    let b = pick(self, w, b0 + h, 1)?;
                    let d = self.mma_sync(block, a, b, zero_d, shape, d_t)?;
                    sums0.push((d, 0));
                    sums1.push((d, 1));
                }
            } else {
                let shape = self.mma_shape(2 * IMMA_TILE, IMMA_TILE, Q8_BLOCK)?;
                let b = pick(self, w, b0, 2)?;
                let d = self.mma_sync(block, a, b, zero_d, shape, d_t)?;
                sums0.push((d, 0));
                sums1.push((d, 1));
            }
            each(self, 0, c, &sums0)?;
            each(self, 1, c, &sums1)?;
        }
        Ok(())
    }
}
