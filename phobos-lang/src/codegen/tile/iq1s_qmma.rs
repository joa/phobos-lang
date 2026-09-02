// IQ1_S on the integer tensor cores, with the grid decode folded in.
//
// This is to a prompt pass what `iq1s_qdot.rs` is to a decode step. The
// expansion path it replaces is not slow because its decode is slow: measured
// per instruction the two bodies are the same size, and both move DRAM at
// 67 GB/s. It is slow because it writes the expanded weight out and reads it
// back, 23.8 GB against the 2.3 GB the format itself is, so it pays 11.4x the
// bytes for the same arithmetic. Contracting straight out of the registers the
// decode already lands in costs none of that, and leaves no scratch on a card
// where the weights are 6.37 GiB of 8.
//
// Two properties of the format are what make it fit `mma.m8n8k16.s8` at all.
// A grid byte is -1, 0 or 1 and the delta is a quarter of the step between
// them, so `dl * (g + delta)` is `(dl / 8) * (8g +- 1)` and the weight is an
// exact int8 in -9..9: no activation row sums, unlike the `dp4a` decode path,
// which carries the delta as a second dot product. And a group of 32 elements
// shares one scale, which is exactly one Q8_0 activation block, so both scales
// land on the same k step and the accumulators stay in registers.

use super::iq1s::IQ1S_BLOCK_BYTES;
use super::*;

/// Where `qh` starts inside a device IQ1_S block, after the 32 `qs` bytes.
const IQ1S_QH_OFF: i64 = 32;

/// Elements a `qs` byte and its three `qh` bits decode to.
const IQ1S_LANE: i64 = 8;

impl<'c> Codegen<'c> {
    /// out[i, j] = sum_b (sum_{k in group b} a[i, k] * w[j, k]), with `w`
    /// decoded from IQ1_S rather than read: the batched contraction, the grid
    /// lookup and both scales as one operation.
    ///
    /// `grid` is the signed table, `8 * g +- 1` already folded, indexed by the
    /// 11-bit grid index and the group's sign bit together. Folding the delta
    /// into the table rather than the kernel is what keeps the decode to a
    /// single four-byte load per fragment: see `iq1s_signed_grid` on the host.
    pub(in crate::codegen) fn tile_iq1s_qmma_t(
        &mut self,
        block: &Block<'c>,
        aq: &MemVal<'c>,
        asc: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        let (md, nd) = (aq.shape[0], qb.shape[0]);
        if md == DYN || nd == DYN {
            bail!("iq1s_qmma_t needs a static output shape");
        }
        let out = self.alloc_tile_shaped(block, self.f32_t, &[md, nd])?;
        self.iq1s_qmma_t_into(block, aq, asc, qb, d, grid, &out)?;
        Ok(out)
    }

    /// [`Self::tile_iq1s_qmma_t`] writing an existing destination, so the
    /// accumulators go straight to global from the registers they are in. See
    /// [`Self::qmma_t_into`], whose patch and fragment geometry this shares.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn iq1s_qmma_t_into(
        &mut self,
        block: &Block<'c>,
        aq: &MemVal<'c>,
        asc: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        for (v, what) in [
            (aq, "iq1s_qmma_t a"),
            (asc, "iq1s_qmma_t a scales"),
            (qb, "iq1s_qmma_t qb"),
            (d, "iq1s_qmma_t d"),
            (grid, "iq1s_qmma_t grid"),
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
        if aq.elem != self.i8_t || qb.elem != self.i8_t || grid.elem != self.i8_t {
            bail!("iq1s_qmma_t contracts an int8 activation against IQ1_S's raw bytes");
        }
        if asc.elem != self.f32_t {
            bail!("iq1s_qmma_t's activation scale must be f32");
        }
        if d.elem != self.f16_t {
            bail!("iq1s_qmma_t's block scale must be f16");
        }
        if out.elem != self.f32_t {
            bail!("iq1s_qmma_t produces f32");
        }
        if out.is_masked() {
            bail!("iq1s_qmma_t needs a fully in-bounds destination");
        }
        if !self.has_int8_mma() {
            bail!("iq1s_qmma_t needs the integer tensor cores (sm_75 or later)");
        }
        if self.cta_threads % WARP != 0 {
            bail!("iq1s_qmma_t needs a CTA that is a whole number of warps");
        }
        let (md, nd) = (aq.shape[0], qb.shape[0]);
        if md == DYN || nd == DYN {
            bail!("iq1s_qmma_t needs a static output shape");
        }
        self.check_shapes(&[md, nd], &out.shape, "iq1s_qmma_t destination")?;
        if md % IMMA_TILE != 0 || nd % IMMA_TILE != 0 {
            bail!("iq1s_qmma_t needs an output tile that is a multiple of {IMMA_TILE} both ways");
        }
        self.check_shapes(&[md], &[asc.shape[0]], "iq1s_qmma_t a scale rows")?;
        self.check_shapes(&[nd], &[d.shape[0]], "iq1s_qmma_t d rows")?;
        if aq.shape[1] != DYN && aq.shape[1] % 256 != 0 {
            bail!("iq1s_qmma_t needs a whole number of 256-element blocks");
        }

        let (i32_t, f32_t) = (self.i32_t, self.f32_t);
        let vec4_i8 = Type::vector(&[4], self.i8_t);
        let frag_t = Type::vector(&[1, 4], self.i8_t);
        let acc_t = Type::vector(&[1, 2], i32_t);

        let one = self.const_index(block, 1)?;
        let kd = if aq.shape[1] == DYN {
            self.push(block, memref::dim(aq.mem, one, self.loc))?
        } else {
            self.const_index(block, aq.shape[1])?
        };

        // The lane's place in the fragments, as in `qmma_t_into`: both operands
        // are read from row lane / 4 four bytes on, and the two accumulator
        // elements land in columns 2 * (lane % 4) and one past.
        let tid = self.thread_id(block)?;
        let warp_w = self.const_index(block, WARP)?;
        let warp = self.divui(block, tid, warp_w)?;
        let lane = self.remui(block, tid, warp_w)?;
        let four = self.const_index(block, 4)?;
        let two = self.const_index(block, 2)?;
        let quad = self.divui(block, lane, four)?;
        let in_quad = self.remui(block, lane, four)?;
        let k_off = self.muli(block, in_quad, four)?;
        let d_col = self.muli(block, in_quad, two)?;

        // A fragment is four consecutive k of one column, so it is half of one
        // eight-element format lane. Which half, and which of the group's four
        // lanes, are fixed for the thread: element `in_quad * 4 + h * 16` of
        // the group sits in format lane `in_quad / 2 + h * 2`, low half when
        // `in_quad` is even. Both are loop-invariant; only the block moves.
        let half = self.remui(block, in_quad, two)?;
        let half_off = self.muli(block, half, four)?;
        let lane_of = |cg: &mut Self, blk: &Block<'c>, h: i64| -> Result<Value<'c, 'c>> {
            let base = cg.divui(blk, in_quad, two)?;
            let off = cg.const_index(blk, h * 2)?;
            cg.addi(blk, base, off)
        };
        let fmt_lane: Vec<Value<'c, 'c>> = (0..Q8_BLOCK / IMMA_K)
            .map(|h| lane_of(self, block, h))
            .collect::<Result<_>>()?;
        // The three bits of `qh` that extend the grid index sit at 3 * l.
        let mut fmt_shift = Vec::with_capacity(fmt_lane.len());
        for l in &fmt_lane {
            let three = self.const_index(block, 3)?;
            let s = self.muli(block, *l, three)?;
            fmt_shift.push(self.numeric_cast(block, s, i32_t)?);
        }

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

        // One k step is one 32-element group, which is one activation block and
        // one weight scale at the same time.
        let blk_w = self.const_index(&kb, Q8_BLOCK)?;
        let b = self.divui(&kb, k, blk_w)?;
        let groups = self.const_index(&kb, 256 / Q8_BLOCK)?;
        let blk = self.divui(&kb, b, groups)?;
        let ib = self.remui(&kb, b, groups)?;
        let blk_bytes = self.const_index(&kb, IQ1S_BLOCK_BYTES)?;
                let qs_group = self.muli(&kb, ib, four)?;
        let qh_group = {
            let base = self.const_index(&kb, IQ1S_QH_OFF)?;
            let two_ib = self.muli(&kb, ib, two)?;
            self.addi(&kb, base, two_ib)?
        };
        let qh_group_hi = self.addi(&kb, qh_group, one)?;

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
                let v = self.vec_load_al(&kb, aq.mem, &[*row, *col], vec4_i8, 4)?;
                a_frags.push(self.vec_shape_cast(&kb, v, frag_t)?);
            }
        }

        // The weight fragment, decoded rather than loaded. `qh` is per column
        // and per group, so it serves both halves of the k step; `qs` is per
        // format lane, so each half takes its own byte.
        let c8_i32 = self.const_i32(&kb, 8)?;
        let c256_i32 = self.const_i32(&kb, 256)?;
        let c32768 = self.const_i32(&kb, 32768)?;
        let c2_i32 = self.const_i32(&kb, 2)?;
        let zero_row = self.const_index(&kb, 0)?;
        let grid_t = Type::vector(&[4], self.i8_t);
        let mut w_frags = Vec::new();
        for row in &w_rows {
            let qh_lo = self.qbyte_grouped(&kb, qb, *row, blk, blk_bytes, qh_group)?;
            let qh_hi = self.qbyte_grouped(&kb, qb, *row, blk, blk_bytes, qh_group_hi)?;
            let qh_hi = self.push(&kb, arith::muli(qh_hi, c256_i32, self.loc))?;
            let qh = self.push(&kb, arith::addi(qh_lo, qh_hi, self.loc))?;
            let sign = self.push(&kb, arith::divui(qh, c32768, self.loc))?;
            let sign = self.push(&kb, arith::remui(sign, c2_i32, self.loc))?;
            for h in 0..halves as usize {
                let qs_off = self.addi(&kb, qs_group, fmt_lane[h])?;
                let qs = self.qbyte_grouped(&kb, qb, *row, blk, blk_bytes, qs_off)?;
                let hi = self.push(&kb, arith::shrui(qh, fmt_shift[h], self.loc))?;
                let hi = self.push(&kb, arith::remui(hi, c8_i32, self.loc))?;
                let hi = self.push(&kb, arith::muli(hi, c256_i32, self.loc))?;
                let idx = self.push(&kb, arith::addi(qs, hi, self.loc))?;
                // The sign picks between the two foldings of the same entry, so
                // it is the table's low index bit and costs no arithmetic here.
                let idx = self.push(&kb, arith::muli(idx, c2_i32, self.loc))?;
                let idx = self.push(&kb, arith::addi(idx, sign, self.loc))?;
                let idx = self.numeric_cast(&kb, idx, self.index_t)?;
                let entry = self.muli(&kb, idx, self.const_index(&kb, IQ1S_LANE)?)?;
                let at = self.addi(&kb, entry, half_off)?;
                let v = self.vec_load_al(&kb, grid.mem, &[zero_row, at], grid_t, 4)?;
                w_frags.push(self.vec_shape_cast(&kb, v, frag_t)?);
            }
        }

        let zero_i = self.zero_scalar(&kb, i32_t)?;
        let empty = self.vec_broadcast(&kb, zero_i, acc_t)?;
        let shape = self.mma_shape(IMMA_TILE, IMMA_TILE, IMMA_K)?;

        // The weight scale belongs to an output column, which is not the column
        // whose fragment this lane holds, so it takes its own `qh`. An eighth
        // is the delta fold: the table carries `8g +- 1` where the weight is
        // `dl * (g +- 1/8)`.
        let eighth = self.const_f32(&kb, 0.125)?;
        let c4096 = self.const_i32(&kb, 4096)?;
        let c1_i32 = self.const_i32(&kb, 1)?;
        let mut w_scales = Vec::with_capacity(rn as usize * 2);
        for out_col in &out_cols {
            for dj in 0..2 {
                let off = self.const_index(&kb, dj)?;
                let col = self.addi(&kb, *out_col, off)?;
                let lo = self.qbyte_grouped(&kb, qb, col, blk, blk_bytes, qh_group)?;
                let hi = self.qbyte_grouped(&kb, qb, col, blk, blk_bytes, qh_group_hi)?;
                let hi = self.push(&kb, arith::muli(hi, c256_i32, self.loc))?;
                let qh = self.push(&kb, arith::addi(lo, hi, self.loc))?;
                let sc = self.push(&kb, arith::divui(qh, c4096, self.loc))?;
                let sc = self.push(&kb, arith::remui(sc, c8_i32, self.loc))?;
                let sc = self.push(&kb, arith::muli(sc, c2_i32, self.loc))?;
                let sc = self.push(&kb, arith::addi(sc, c1_i32, self.loc))?;
                let sc = self.numeric_cast(&kb, sc, f32_t)?;
                let dat = self.raw_block_at(&kb, col, blk, blk_bytes)?;
                let dv = self.push(&kb, memref::load(d.mem, &[dat.d_row, dat.d_col], self.loc))?;
                let dv = self.numeric_cast(&kb, dv, f32_t)?;
                let dl = self.push(&kb, arith::mulf(dv, sc, self.loc))?;
                w_scales.push(self.push(&kb, arith::mulf(dl, eighth, self.loc))?);
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

impl<'c> Codegen<'c> {
    /// [`Self::iq1s_qmma_t_into`] with the decoded weight staged through shared
    /// memory rather than decoded again in every warp that needs it.
    ///
    /// `qmma_patch` caps a warp's patch at [`QMMA_TILES`] tiles, so a 128-row
    /// output tile takes two patches and each column of the weight is decoded
    /// twice over. Staging decodes it once for the whole CTA and turns a
    /// fragment from about seventeen instructions into one shared load.
    ///
    /// What it costs is the loop nest. The register form carries its
    /// accumulators across `k` inside a patch loop; staging needs every warp at
    /// the same `k` at once, so the patch loop has to go, and it only can when
    /// there is exactly one patch a warp. That is why this is a second entry
    /// point rather than a flag on the first: the caller picks it by name,
    /// which also keeps the two apart in the kernel cache, where a flag read
    /// during codegen would not.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn iq1s_qmma_staged_into(
        &mut self,
        block: &Block<'c>,
        aq: &MemVal<'c>,
        asc: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        let (md, nd) = (aq.shape[0], qb.shape[0]);
        if md == DYN || nd == DYN {
            bail!("iq1s_qmma_t needs a static output shape");
        }
        if !self.has_int8_mma() {
            bail!("iq1s_qmma_t needs the integer tensor cores (sm_75 or later)");
        }
        if self.cta_threads % WARP != 0 {
            bail!("iq1s_qmma_t needs a CTA that is a whole number of warps");
        }
        self.check_shapes(&[md, nd], &out.shape, "iq1s_qmma_t destination")?;
        if md % IMMA_TILE != 0 || nd % IMMA_TILE != 0 {
            bail!("iq1s_qmma_t needs an output tile that is a multiple of {IMMA_TILE} both ways");
        }

        let (rt, ct) = (md / IMMA_TILE, nd / IMMA_TILE);
        let (rm, rn) = self.qmma_patch(rt, ct);
        let patches = (rt / rm) * (ct / rn);
        let warps = self.cta_threads / WARP;
        if patches != warps {
            bail!("staged iq1s_qmma_t needs one patch a warp, got {patches} for {warps} warps");
        }
        // A staging lane is one format lane: eight elements of one column.
        let entries = nd * (Q8_BLOCK / IQ1S_LANE);
        if entries % self.cta_threads != 0 {
            bail!("staged iq1s_qmma_t needs the CTA to divide the {entries} lanes it stages");
        }

        let (i32_t, f32_t) = (self.i32_t, self.f32_t);
        let vec4_i8 = Type::vector(&[4], self.i8_t);
        let frag_t = Type::vector(&[1, 4], self.i8_t);
        let acc_t = Type::vector(&[1, 2], i32_t);
        let lane_t = Type::vector(&[IQ1S_LANE as u64], self.i8_t);
        let stage = self.alloc_tile_shaped(block, self.i8_t, &[nd, Q8_BLOCK])?;

        let one = self.const_index(block, 1)?;
        let kd = if aq.shape[1] == DYN {
            self.push(block, memref::dim(aq.mem, one, self.loc))?
        } else {
            self.const_index(block, aq.shape[1])?
        };

        let tid = self.thread_id(block)?;
        let warp_w = self.const_index(block, WARP)?;
        let warp = self.divui(block, tid, warp_w)?;
        let lane = self.remui(block, tid, warp_w)?;
        let four = self.const_index(block, 4)?;
        let two = self.const_index(block, 2)?;
        let quad = self.divui(block, lane, four)?;
        let in_quad = self.remui(block, lane, four)?;
        let k_off = self.muli(block, in_quad, four)?;
        let d_col = self.muli(block, in_quad, two)?;

        // The warp's patch straight from its index. No patch loop: that is the
        // guard above, and the whole reason staging is expressible here.
        let across = self.const_index(block, ct / rn)?;
        let ui = self.divui(block, warp, across)?;
        let uj = self.remui(block, warp, across)?;
        let patch_m = self.const_index(block, rm * IMMA_TILE)?;
        let patch_n = self.const_index(block, rn * IMMA_TILE)?;
        let i0 = self.muli(block, ui, patch_m)?;
        let j0 = self.muli(block, uj, patch_n)?;
        let row_base = self.addi(block, i0, quad)?;
        let col_base = self.addi(block, j0, d_col)?;
        let frag_base = self.addi(block, j0, quad)?;

        let mut a_rows = Vec::with_capacity(rm as usize);
        for r in 0..rm {
            let off = self.const_index(block, r * IMMA_TILE)?;
            a_rows.push(self.addi(block, row_base, off)?);
        }
        let (mut w_rows, mut out_cols) = (Vec::new(), Vec::new());
        for c in 0..rn {
            let off = self.const_index(block, c * IMMA_TILE)?;
            w_rows.push(self.addi(block, frag_base, off)?);
            out_cols.push(self.addi(block, col_base, off)?);
        }
        // Staging lanes this thread owns, strided by the CTA so that
        // consecutive threads write consecutive columns.
        let per_thread = entries / self.cta_threads;
        let mut mine = Vec::with_capacity(per_thread as usize);
        for e in 0..per_thread {
            let off = self.const_index(block, e * self.cta_threads)?;
            mine.push(self.addi(block, tid, off)?);
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
        let groups = self.const_index(&kb, 256 / Q8_BLOCK)?;
        let blk = self.divui(&kb, b, groups)?;
        let ib = self.remui(&kb, b, groups)?;
        let blk_bytes = self.const_index(&kb, IQ1S_BLOCK_BYTES)?;
                let qs_group = self.muli(&kb, ib, four)?;
        let qh_group = {
            let base = self.const_index(&kb, IQ1S_QH_OFF)?;
            let two_ib = self.muli(&kb, ib, two)?;
            self.addi(&kb, base, two_ib)?
        };
        let qh_group_hi = self.addi(&kb, qh_group, one)?;

        let c8_i32 = self.const_i32(&kb, 8)?;
        let c256_i32 = self.const_i32(&kb, 256)?;
        let c32768 = self.const_i32(&kb, 32768)?;
        let c2_i32 = self.const_i32(&kb, 2)?;
        let c4096 = self.const_i32(&kb, 4096)?;
        let c1_i32 = self.const_i32(&kb, 1)?;
        let three_i32 = self.const_i32(&kb, 3)?;
        let zero_row = self.const_index(&kb, 0)?;
        let eight = self.const_index(&kb, IQ1S_LANE)?;

        // Stage the block: every column's four format lanes, decoded once.
        for entry in &mine {
            let j = self.divui(&kb, *entry, four)?;
            let l = self.remui(&kb, *entry, four)?;
            let qh_lo = self.qbyte_grouped(&kb, qb, j, blk, blk_bytes, qh_group)?;
            let qh_hi = self.qbyte_grouped(&kb, qb, j, blk, blk_bytes, qh_group_hi)?;
            let qh_hi = self.push(&kb, arith::muli(qh_hi, c256_i32, self.loc))?;
            let qh = self.push(&kb, arith::addi(qh_lo, qh_hi, self.loc))?;
            let sign = self.push(&kb, arith::divui(qh, c32768, self.loc))?;
            let sign = self.push(&kb, arith::remui(sign, c2_i32, self.loc))?;
            let qs_off = self.addi(&kb, qs_group, l)?;
            let qs = self.qbyte_grouped(&kb, qb, j, blk, blk_bytes, qs_off)?;
            let shift = self.numeric_cast(&kb, l, i32_t)?;
            let shift = self.push(&kb, arith::muli(shift, three_i32, self.loc))?;
            let hi = self.push(&kb, arith::shrui(qh, shift, self.loc))?;
            let hi = self.push(&kb, arith::remui(hi, c8_i32, self.loc))?;
            let hi = self.push(&kb, arith::muli(hi, c256_i32, self.loc))?;
            let idx = self.push(&kb, arith::addi(qs, hi, self.loc))?;
            // The sign picks between the two foldings of the same entry, so it
            // is the table's low index bit and costs no arithmetic here.
            let idx = self.push(&kb, arith::muli(idx, c2_i32, self.loc))?;
            let idx = self.push(&kb, arith::addi(idx, sign, self.loc))?;
            let idx = self.numeric_cast(&kb, idx, self.index_t)?;
            let at = self.muli(&kb, idx, eight)?;
            let v = self.vec_load_al(&kb, grid.mem, &[zero_row, at], lane_t, 8)?;
            let dst = self.muli(&kb, l, eight)?;
            self.vec_store_al(&kb, v, stage.mem, &[j, dst], 8)?;
        }
        self.barrier(&kb)?;

        let k_from = self.addi(&kb, k, k_off)?;
        let halves = Q8_BLOCK / IMMA_K;
        let (mut k_cols, mut stage_cols) = (Vec::new(), Vec::new());
        for h in 0..halves {
            let off = self.const_index(&kb, h * IMMA_K)?;
            k_cols.push(self.addi(&kb, k_from, off)?);
            stage_cols.push(self.addi(&kb, k_off, off)?);
        }

        let mut a_frags = Vec::new();
        for row in &a_rows {
            for col in &k_cols {
                let v = self.vec_load_al(&kb, aq.mem, &[*row, *col], vec4_i8, 4)?;
                a_frags.push(self.vec_shape_cast(&kb, v, frag_t)?);
            }
        }
        // One shared load where the register form spends a decode.
        let mut w_frags = Vec::new();
        for row in &w_rows {
            for col in &stage_cols {
                let v = self.vec_load_al(&kb, stage.mem, &[*row, *col], vec4_i8, 4)?;
                w_frags.push(self.vec_shape_cast(&kb, v, frag_t)?);
            }
        }

        let zero_i = self.zero_scalar(&kb, i32_t)?;
        let empty = self.vec_broadcast(&kb, zero_i, acc_t)?;
        let shape = self.mma_shape(IMMA_TILE, IMMA_TILE, IMMA_K)?;

        let eighth = self.const_f32(&kb, 0.125)?;
        let mut w_scales = Vec::with_capacity(rn as usize * 2);
        for out_col in &out_cols {
            for dj in 0..2 {
                let off = self.const_index(&kb, dj)?;
                let col = self.addi(&kb, *out_col, off)?;
                let lo = self.qbyte_grouped(&kb, qb, col, blk, blk_bytes, qh_group)?;
                let hi = self.qbyte_grouped(&kb, qb, col, blk, blk_bytes, qh_group_hi)?;
                let hi = self.push(&kb, arith::muli(hi, c256_i32, self.loc))?;
                let qh = self.push(&kb, arith::addi(lo, hi, self.loc))?;
                let sc = self.push(&kb, arith::divui(qh, c4096, self.loc))?;
                let sc = self.push(&kb, arith::remui(sc, c8_i32, self.loc))?;
                let sc = self.push(&kb, arith::muli(sc, c2_i32, self.loc))?;
                let sc = self.push(&kb, arith::addi(sc, c1_i32, self.loc))?;
                let sc = self.numeric_cast(&kb, sc, f32_t)?;
                let dat = self.raw_block_at(&kb, col, blk, blk_bytes)?;
                let dv = self.push(&kb, memref::load(d.mem, &[dat.d_row, dat.d_col], self.loc))?;
                let dv = self.numeric_cast(&kb, dv, f32_t)?;
                let dl = self.push(&kb, arith::mulf(dv, sc, self.loc))?;
                w_scales.push(self.push(&kb, arith::mulf(dl, eighth, self.loc))?);
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
                    next.push(self.elem_mac(&kb, f32_t, as_f, scale, accs[slot])?);
                }
            }
        }
        // The next turn of the loop overwrites what this one staged.
        self.barrier(&kb)?;
        kb.append_operation(scf::r#yield(&next, self.loc));

        let zero_k = self.const_index(block, 0)?;
        let step = self.const_index(block, Q8_BLOCK)?;
        let init = self.zero_scalar(block, f32_t)?;
        let mut operands = vec![zero_k, kd, step];
        operands.extend(std::iter::repeat_n(init, lanes));
        let kr = Region::new();
        kr.append_block(kb);
        let loop_op = block.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&operands)
                .add_results(&vec![f32_t; lanes])
                .add_regions([kr])
                .build()?,
        );

        for (r, row) in a_rows.iter().enumerate() {
            for (c, out_col) in out_cols.iter().enumerate() {
                for dj in 0..2 {
                    let off = self.const_index(block, dj)?;
                    let col = self.addi(block, *out_col, off)?;
                    let slot = (r * rn as usize + c) * 2 + dj as usize;
                    let value = detach(loop_op.result(slot)?.into());
                    block.append_operation(memref::store(value, out.mem, &[*row, col], self.loc));
                }
            }
        }
        self.barrier(block)?;
        Ok(())
    }
}
