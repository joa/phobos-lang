// The fused Q8_0 dot: scales folded into the contraction.
//
// The weight is either one `i8` per element or Q5_0's blocks without their
// scale (`QWeight::Q50`), 20 bytes per 32 elements: the `qh` word, then 16
// bytes whose byte `e` holds element `e` in its low nibble and `e + 16` in
// its high one. Bit `e` of `qh` is element `e`'s fifth bit. Four elements
// widen to the `q - 16` bytes the contraction reads with no carry across
// bytes: the four `qh` bits spread one to a byte, and a clear bit puts 0xF0
// over the nibble.

use super::*;

/// Bytes of a Q5_0 block without its two-byte scale.
const Q50_BLOCK_BYTES: i64 = 20;

/// How `qdot_t` and `qmma_t` hold their weight.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::codegen) enum QWeight {
    /// One `i8` per element.
    I8,
    /// Q5_0 blocks without their scale; see the module comment.
    Q50,
}

impl<'c> Codegen<'c> {
    /// Weight bytes `k .. k + 4` of row `j` as `vector<4xi8>`. `k` is a
    /// multiple of four.
    pub(super) fn qweight_word(
        &self,
        block: &Block<'c>,
        w: &MemVal<'c>,
        form: QWeight,
        j: Value<'c, 'c>,
        k: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let vec4_i8 = Type::vector(&[4], self.i8_t);
        if form == QWeight::I8 {
            return self.vec_load_al(block, w.mem, &[j, k], vec4_i8, 4);
        }
        let word_t = Type::vector(&[1], self.i32_t);
        let word_at = |at: Value<'c, 'c>| -> Result<Value<'c, 'c>> {
            let v = self.vec_load_al(block, w.mem, &[j, at], vec4_i8, 4)?;
            let v = self.vec_bitcast(block, v, word_t)?;
            self.vec_extract(block, v, &[0], self.i32_t)
        };
        let blk = self.const_index(block, Q8_BLOCK)?;
        let half_blk = self.const_index(block, Q8_BLOCK / 2)?;
        let four = self.const_index(block, 4)?;
        let b = self.divui(block, k, blk)?;
        let e = self.remui(block, k, blk)?;
        let bytes = self.const_index(block, Q50_BLOCK_BYTES)?;
        let base = self.muli(block, b, bytes)?;
        let qh = word_at(base)?;
        let at = self.remui(block, e, half_blk)?;
        let qs_at = self.addi(block, base, four)?;
        let qs_at = self.addi(block, qs_at, at)?;
        let qs = word_at(qs_at)?;

        let half = self.divui(block, e, half_blk)?;
        let shift = self.muli(block, half, four)?;
        let shift = self.numeric_cast(block, shift, self.i32_t)?;
        let low = self.kq_bits_dyn(block, qs, shift, 0x0F0F_0F0F)?;
        let e = self.numeric_cast(block, e, self.i32_t)?;
        let bits = self.kq_bits_dyn(block, qh, e, 0xF)?;
        let spread_mul = self.const_i32(block, 0x0020_4081)?;
        let spread = self.push(block, arith::muli(bits, spread_mul, self.loc))?;
        let ones = self.const_i32(block, 0x0101_0101)?;
        let spread = self.push(block, arith::andi(spread, ones, self.loc))?;
        let clear = self.push(block, arith::xori(spread, ones, self.loc))?;
        let f0 = self.const_i32(block, 0xF0)?;
        let high = self.push(block, arith::muli(clear, f0, self.loc))?;
        let word = self.push(block, arith::ori(low, high, self.loc))?;
        let word = self.vec_broadcast(block, word, word_t)?;
        self.vec_bitcast(block, word, vec4_i8)
    }

    /// out[i, j] = sum_b (sum_{k in block b} a[i, k] * w[j, k]) * asc[i, b] * wsc[j, b]:
    /// the whole Q8_0 contraction, block scales included, as one operation.
    ///
    /// A warp owns one output and its lanes split `k`, so the 32 lanes read
    /// 512 contiguous bytes of one weight row. Nothing is staged, the
    /// accumulator is a register, and the only synchronization is the closing
    /// butterfly shuffle. Each lane takes 16 bytes, four `dp4a` under one
    /// scale pair, since 16 divides the 32-element block.
    ///
    /// The scales are indexed `[row, block]`, so a lane's scale load is
    /// contiguous with its neighbours'.
    pub(in crate::codegen) fn tile_qdot_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        asc: &MemVal<'c>,
        w: &MemVal<'c>,
        wsc: &MemVal<'c>,
        form: QWeight,
    ) -> Result<MemVal<'c>> {
        for (v, what) in [
            (a, "qdot_t a"),
            (asc, "qdot_t a scales"),
            (w, "qdot_t w"),
            (wsc, "qdot_t w scales"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if a.elem != self.i8_t || w.elem != self.i8_t {
            bail!("qdot_t contracts int8 operands");
        }
        if asc.elem != self.f32_t || wsc.elem != self.f32_t {
            bail!("qdot_t scales must be f32");
        }
        if !self.has_dp4a() {
            bail!("qdot_t needs dp4a (sm_61 or later)");
        }
        let (rows, cols) = (a.shape[0], w.shape[0]);
        if rows == DYN || cols == DYN {
            bail!("qdot_t needs a static output shape");
        }
        self.check_shapes(&[rows], &[asc.shape[0]], "qdot_t a scale rows")?;
        self.check_shapes(&[cols], &[wsc.shape[0]], "qdot_t w scale rows")?;
        if self.cta_threads % WARP != 0 {
            bail!("qdot_t needs a CTA that is a whole number of warps");
        }

        let out = self.alloc_tile_shaped(block, self.f32_t, &[rows, cols])?;
        let (i32_t, f32_t, vec4_i8) = (self.i32_t, self.f32_t, Type::vector(&[4], self.i8_t));

        // The contraction length: static when the slice fixes it, otherwise
        // the operand's own extent.
        let one = self.const_index(block, 1)?;
        let kd = if a.shape[1] == DYN {
            self.push(block, memref::dim(a.mem, one, self.loc))?
        } else {
            self.const_index(block, a.shape[1])?
        };

        let lane_w = self.const_index(block, WARP)?;
        let total = self.const_index(block, rows * cols * WARP)?;
        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;

        // A warp per output element. The CTA size and `total` are both warp
        // multiples, so a warp is wholly inside or outside this loop and every
        // lane reaches the shuffle.
        let body = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body.argument(0)?.into());
        let unit = self.divui(&body, li, lane_w)?;
        let lane = self.remui(&body, li, lane_w)?;
        let ncols = self.const_index(&body, cols)?;
        let i = self.divui(&body, unit, ncols)?;
        let j = self.remui(&body, unit, ncols)?;

        let step = self.const_index(&body, QDOT_STEP)?;
        let lane_bytes = self.const_index(&body, QDOT_LANE)?;
        let lane_off = self.muli(&body, lane, lane_bytes)?;
        let zero_k = self.const_index(&body, 0)?;
        let init = self.zero_scalar(&body, f32_t)?;

        let kb = Block::new(&[(self.index_t, self.loc), (f32_t, self.loc)]);
        let base = detach(kb.argument(0)?.into());
        let carry = detach(kb.argument(1)?.into());
        let koff = self.addi(&kb, base, lane_off)?;
        // A lane's chunk divides the Q8_0 block, so one predicate covers it.
        let live = self.push(
            &kb,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, koff, kd, self.loc),
        )?;

        let then = Block::new(&[]);
        let mut dots = self.zero_scalar(&then, i32_t)?;
        for c in 0..QDOT_LANE / 4 {
            let at = self.const_index(&then, c * 4)?;
            let k = self.addi(&then, koff, at)?;
            let va = self.vec_load_al(&then, a.mem, &[i, k], vec4_i8, 4)?;
            let vb = self.qweight_word(&then, w, form, j, k)?;
            dots = self.dot4_accumulate(&then, va, vb, dots)?;
        }
        let blk_w = self.const_index(&then, Q8_BLOCK)?;
        let b = self.divui(&then, koff, blk_w)?;
        let sa = self.push(&then, memref::load(asc.mem, &[i, b], self.loc))?;
        let sw = self.push(&then, memref::load(wsc.mem, &[j, b], self.loc))?;
        let as_f = self.numeric_cast(&then, dots, f32_t)?;
        let scaled = self.push(&then, arith::mulf(as_f, sa, self.loc))?;
        let scaled = self.push(&then, arith::mulf(scaled, sw, self.loc))?;
        let summed = self.push(&then, arith::addf(carry, scaled, self.loc))?;
        then.append_operation(scf::r#yield(&[summed], self.loc));

        let otherwise = Block::new(&[]);
        otherwise.append_operation(scf::r#yield(&[carry], self.loc));

        let (tr, er) = (Region::new(), Region::new());
        tr.append_block(then);
        er.append_block(otherwise);
        let next = self.push(&kb, scf::r#if(live, &[f32_t], tr, er, self.loc))?;
        kb.append_operation(scf::r#yield(&[next], self.loc));

        let kr = Region::new();
        kr.append_block(kb);
        let mut acc = self.push(
            &body,
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&[zero_k, kd, step, init])
                .add_results(&[f32_t])
                .add_regions([kr])
                .build()?,
        )?;

        let mut mask = WARP / 2;
        while mask >= 1 {
            let other = self.shfl_xor_f32(&body, acc, mask)?;
            acc = self.push(&body, arith::addf(acc, other, self.loc))?;
            mask /= 2;
        }

        let zero = self.const_index(&body, 0)?;
        let is_lead = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, zero, self.loc),
        )?;
        let store = Block::new(&[]);
        store.append_operation(memref::store(acc, out.mem, &[i, j], self.loc));
        store.append_operation(scf::r#yield(&[], self.loc));
        let sr = Region::new();
        sr.append_block(store);
        body.append_operation(scf::r#if(is_lead, &[], sr, Region::new(), self.loc));
        body.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));
        self.barrier(block)?;
        Ok(out)
    }
}
