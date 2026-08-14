// Integer tensor-core contraction (mma.sync on i8).

use super::*;

impl<'c> Codegen<'c> {
    /// `dot_t` over int8 operands on the integer tensor cores.
    ///
    /// Unlike the f16 tensor-core path there is no staging buffer and no
    /// ldmatrix: the m8n8k16 fragment layout is already what `dot_t` holds in
    /// memory. A lane's A register is four contiguous bytes of row `lane / 4`,
    /// its B register is four contiguous bytes of row `lane / 4` of the [n, k]
    /// operand, and both are exactly the four bytes `dp4a` would have read.
    /// One `mma.sync` folds sixteen products per lane where `dp4a` folds four.
    ///
    /// Returns false when it does not apply, leaving the caller on the dp4a
    /// path: the tensor core issues whole 8x8 output tiles over a k that is a
    /// multiple of 16, and there are no integer tensor cores before Turing. A
    /// masked output would need the store guarded per element, which is what
    /// the generic paths already do.
    pub(super) fn tile_matmul_t_imma(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        kd: i64,
    ) -> Result<bool> {
        let (md, nd) = (out.shape[0], out.shape[1]);
        let applies = a.elem == self.i8_t
            && b.elem == self.i8_t
            && out.elem == self.i32_t
            // Positive as well as divisible: DYN is i64::MIN, which every one
            // of these divides.
            && md > 0
            && nd > 0
            && md % 8 == 0
            && nd % 8 == 0
            && kd % 16 == 0
            && a.swizzle.is_none()
            && b.swizzle.is_none()
            && !a.is_masked()
            && !b.is_masked()
            && !out.is_masked()
            && self.has_int8_mma();
        if !applies {
            return Ok(false);
        }

        let i32_t = self.i32_t;
        let vec4_i8 = Type::vector(&[4], self.i8_t);
        let frag_t = Type::vector(&[1, 4], self.i8_t);
        let acc_t = Type::vector(&[1, 2], i32_t);

        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;
        let warp_size = self.const_index(block, 32)?;
        let warp = self.divui(block, tid, warp_size)?;
        let warps = self.divui(block, bdim, warp_size)?;
        let lane = self.remui(block, tid, warp_size)?;

        // The lane's place in the fragments: both operands are read from row
        // lane / 4, four bytes starting at column 4 * (lane % 4), and the two
        // accumulator elements land in columns 2 * (lane % 4) and one past.
        let four = self.const_index(block, 4)?;
        let quad = self.divui(block, lane, four)?;
        let in_quad = self.remui(block, lane, four)?;
        let k_off = self.muli(block, in_quad, four)?;
        let two = self.const_index(block, 2)?;
        let d_col = self.muli(block, in_quad, two)?;

        // One 8x8 output tile per warp, row-major so neighbouring warps share
        // the A rows they read.
        let eight = self.const_index(block, 8)?;
        let n_tiles = self.const_index(block, nd / 8)?;
        let total = self.const_index(block, (md / 8) * (nd / 8))?;

        let tb = Block::new(&[(self.index_t, self.loc)]);
        let t = detach(tb.argument(0)?.into());
        let ti = self.divui(&tb, t, n_tiles)?;
        let tj = self.remui(&tb, t, n_tiles)?;
        let i0 = self.muli(&tb, ti, eight)?;
        let j0 = self.muli(&tb, tj, eight)?;
        let a_row = self.addi(&tb, i0, quad)?;
        let b_row = self.addi(&tb, j0, quad)?;

        let zero = self.zero_scalar(&tb, i32_t)?;
        let init = self.vec_broadcast(&tb, zero, acc_t)?;
        let lo = self.const_index(&tb, 0)?;
        let hi = self.const_index(&tb, kd)?;
        let st = self.const_index(&tb, 16)?;

        let kb = Block::new(&[(self.index_t, self.loc), (acc_t, self.loc)]);
        let k = detach(kb.argument(0)?.into());
        let acc = detach(kb.argument(1)?.into());
        let k_col = self.addi(&kb, k, k_off)?;
        let va = self.vec_load_al(&kb, a.mem, &[a_row, k_col], vec4_i8, 4)?;
        let vb = self.vec_load_al(&kb, b.mem, &[b_row, k_col], vec4_i8, 4)?;
        let va = self.vec_shape_cast(&kb, va, frag_t)?;
        let vb = self.vec_shape_cast(&kb, vb, frag_t)?;
        let shape = self.mma_shape(IMMA_TILE, IMMA_TILE, IMMA_K)?;
        let next = self.mma_sync(&kb, va, vb, acc, shape, acc_t)?;
        kb.append_operation(scf::r#yield(&[next], self.loc));
        let k_region = Region::new();
        k_region.append_block(kb);
        let fin = self.push(
            &tb,
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&[lo, hi, st, init])
                .add_results(&[acc_t])
                .add_regions([k_region])
                .build()?,
        )?;

        let out_col = self.addi(&tb, j0, d_col)?;
        for dj in 0..2 {
            let e = self.vec_extract(&tb, fin, &[0, dj], i32_t)?;
            let off = self.const_index(&tb, dj)?;
            let col = self.addi(&tb, out_col, off)?;
            tb.append_operation(memref::store(e, out.mem, &[a_row, col], self.loc));
        }
        tb.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(tb);
        block.append_operation(scf::r#for(warp, total, warps, region, self.loc));
        self.barrier(block)?;
        Ok(true)
    }

    /// A small signed integer as an f32, without the conversion instruction.
    ///
    /// Adding 1.5 * 2^23 to `value` as an integer lands it in the mantissa of
    /// that float, so the bits are already the f32 of `1.5 * 2^23 + value` and
    /// subtracting the constant back off leaves the value exactly. It holds for
    /// `|value| < 2^22`, which a Q8_0 block guarantees: 32 products of two
    /// int8s cannot exceed 32 * 127 * 127, about an eighth of the room.
    ///
    /// This is worth doing rather than a `cvt` because on Turing conversions
    /// issue at a quarter of the arithmetic rate, one per eight cycles against
    /// one per cycle, and the quantized matmul does one per accumulator per
    /// block: at a patch's size that is 128 of them against the same block's
    /// 128 tensor instructions, which run one per four cycles.
    pub(super) fn small_int_to_f32(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        const MAGIC_BITS: i64 = 0x4B40_0000;
        const MAGIC: f64 = 12_582_912.0;
        let bias = self.push(
            block,
            arith::constant(
                self.ctx,
                IntegerAttribute::new(self.i32_t, MAGIC_BITS).into(),
                self.loc,
            ),
        )?;
        let shifted = self.addi(block, value, bias)?;
        let bits = self.push(block, arith::bitcast(shifted, self.f32_t, self.loc))?;
        let magic = self.push(
            block,
            arith::constant(
                self.ctx,
                FloatAttribute::new(self.ctx, self.f32_t, MAGIC).into(),
                self.loc,
            ),
        )?;
        self.push(block, arith::subf(bits, magic, self.loc))
    }
}
