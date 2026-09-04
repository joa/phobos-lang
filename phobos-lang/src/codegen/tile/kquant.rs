// What the K-quants, Q4_K, Q5_K and Q6_K, contribute to the staged
// projection (`qgemm.rs`) and the decode matvec (`qdot_i8_reg.rs`), and the
// arithmetic the two share. No tables: every quant is a nibble plus, for
// Q5_K, one bit of a 32-byte `qh` plane, or, for Q6_K, two bits of a
// 64-byte one. The scales are in the block, six-bit indices packed twelve
// bytes for Q4_K and Q5_K (`q4_k::scale_min` on the host), sixteen signed
// bytes for Q6_K.
//
// Q4_K and Q5_K carry a minimum: a run decodes as `d * sc * q - dmin * m`,
// and against a Q8_0 activation `a = sa * aq` the contraction is
// `sa * (d * sc * A - dmin * m * S)` with `A` the dp4a chain and `S` the
// activation's own sum over the run, which neither kernel had before. The
// staged projection sums each row's group at stage time; the decode matvec
// sums the row once in a prologue (`kquant_qdot.rs`). Q6_K has no minimum,
// and its `q - 32` folds into the byte before the dp4a, exactly and without
// a carry: `(qh2 + 14) & 15` is `qh2 - 2 mod 16`, so
// `ql | (((qh2 + 0x0E0E0E0E) & 0x0F0F0F0F) << 4)` is `q - 32` as an i8.
//
// Quants of at most six bits are the same bytes signed or unsigned, so
// they ride the signed dp4a and mma as they are; the per-run dot products
// stay under 2^22, so the exact add-and-subtract conversion applies.

use super::qgemm::{Lanes, Stage, TileAt};
use super::*;

/// Bytes of a 32-element activation group summed for the minimum term.
pub(super) const KQ_GROUP: i64 = 32;

impl QgFormat {
    /// Whether the format is one of the three K-quants here.
    pub(in crate::codegen) fn is_kquant(self) -> bool {
        matches!(self, Self::Q4k | Self::Q5k | Self::Q6k)
    }

    /// Whether a run subtracts a minimum, and so the contraction needs the
    /// activation's run sums.
    pub(in crate::codegen) fn has_min(self) -> bool {
        matches!(self, Self::Q4k | Self::Q5k)
    }

    /// Whether `d` (and `dmin`) sit in the block header, in the same load
    /// as the scales, so the scale plane is uploaded but never read.
    pub(in crate::codegen) fn d_in_block(self) -> bool {
        matches!(self, Self::Q4k | Self::Q5k)
    }

    /// Blocks the decode matvec's register pipeline holds ahead. Two for
    /// every format; Q4_K at three measured slower.
    pub(in crate::codegen) fn pipeline_depth(self) -> usize {
        2
    }
}

impl<'c> Codegen<'c> {
    /// `(word >> shift) & mask` for a runtime `shift`.
    pub(super) fn kq_bits_dyn(
        &self,
        block: &Block<'c>,
        word: Value<'c, 'c>,
        shift: Value<'c, 'c>,
        mask: i64,
    ) -> Result<Value<'c, 'c>> {
        let shifted = self.push(block, arith::shrui(word, shift, self.loc))?;
        let m = self.const_i32(block, mask)?;
        self.push(block, arith::andi(shifted, m, self.loc))
    }

    /// The `f16` in the low (`hi` false) or high half of a word, as f32.
    pub(super) fn kq_f16_of(
        &self,
        block: &Block<'c>,
        word: Value<'c, 'c>,
        hi: bool,
    ) -> Result<Value<'c, 'c>> {
        let word = if hi {
            let sixteen = self.const_i32(block, 16)?;
            self.push(block, arith::shrui(word, sixteen, self.loc))?
        } else {
            word
        };
        let i16_t: Type<'c> = IntegerType::new(self.ctx, 16).into();
        let bits = self.push(
            block,
            OperationBuilder::new("arith.trunci", self.loc)
                .add_operands(&[word])
                .add_results(&[i16_t])
                .build()?,
        )?;
        let half = self.push(block, arith::bitcast(bits, self.f16_t, self.loc))?;
        self.numeric_cast(block, half, self.f32_t)
    }

    /// Byte `idx` (a runtime 0..7) of the pair `lo`, `hi`, zero-extended.
    pub(super) fn kq_byte_at(
        &self,
        block: &Block<'c>,
        lo: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        idx: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let picked = self.byte_permute(block, lo, hi, idx)?;
        let mask = self.const_i32(block, 0xFF)?;
        self.push(block, arith::andi(picked, mask, self.loc))
    }

    /// Byte `idx` (a runtime 0..7) of the pair, sign-extended: the selector
    /// names the byte once and its sign three times.
    pub(super) fn kq_sbyte_at(
        &self,
        block: &Block<'c>,
        lo: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        idx: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let eight = self.const_i32(block, 8)?;
        let signed = self.push(block, arith::ori(idx, eight, self.loc))?;
        let spread = self.const_i32(block, 0x1110)?;
        let upper = self.push(block, arith::muli(signed, spread, self.loc))?;
        let sel = self.push(block, arith::ori(idx, upper, self.loc))?;
        self.byte_permute(block, lo, hi, sel)
    }

    /// Q4_K's and Q5_K's six-bit scale and minimum of run `r` (a runtime
    /// 0..7) from the three words of the twelve scale bytes, both branches
    /// of `q4_k::scale_min` computed and one selected on `r < 4`.
    pub(super) fn kq_scale_min(
        &self,
        block: &Block<'c>,
        w: [Value<'c, 'c>; 3],
        r: Value<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let four = self.const_i32(block, 4)?;
        let six = self.const_i32(block, 6)?;
        let c63 = self.const_i32(block, 63)?;
        let c15 = self.const_i32(block, 15)?;
        let r_plus = self.push(block, arith::addi(r, four, self.loc))?;
        let r_minus = self.push(block, arith::subi(r, four, self.loc))?;
        // r < 4: byte r and byte r + 4, six bits each.
        let sc_lo = self.kq_byte_at(block, w[0], w[1], r)?;
        let sc_lo = self.push(block, arith::andi(sc_lo, c63, self.loc))?;
        let m_lo = self.kq_byte_at(block, w[0], w[1], r_plus)?;
        let m_lo = self.push(block, arith::andi(m_lo, c63, self.loc))?;
        // r >= 4: the low nibbles of byte r + 4 (the third word), the top
        // two bits of bytes r - 4 and r.
        let b_high = self.kq_byte_at(block, w[2], w[2], r_minus)?;
        let b_lower = self.kq_byte_at(block, w[0], w[1], r_minus)?;
        let b_same = self.kq_byte_at(block, w[0], w[1], r)?;
        let sc_hi = {
            let low4 = self.push(block, arith::andi(b_high, c15, self.loc))?;
            let top = self.push(block, arith::shrui(b_lower, six, self.loc))?;
            let top = self.push(block, arith::shli(top, four, self.loc))?;
            self.push(block, arith::ori(low4, top, self.loc))?
        };
        let m_hi = {
            let low4 = self.push(block, arith::shrui(b_high, four, self.loc))?;
            let top = self.push(block, arith::shrui(b_same, six, self.loc))?;
            let top = self.push(block, arith::shli(top, four, self.loc))?;
            self.push(block, arith::ori(low4, top, self.loc))?
        };
        let low = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, r, four, self.loc),
        )?;
        Ok((
            self.select(block, low, sc_lo, sc_hi)?,
            self.select(block, low, m_lo, m_hi)?,
        ))
    }

    /// Four Q4_K or Q5_K quants of a `qs` word as bytes: the nibbles `shift`
    /// (a runtime 0 or 4) selects, plus, for Q5_K, bit `bit` of the matching
    /// `qh` word as the fifth.
    pub(super) fn kq_nibbles(
        &self,
        block: &Block<'c>,
        qs: Value<'c, 'c>,
        shift: Value<'c, 'c>,
        qh: Option<(Value<'c, 'c>, Value<'c, 'c>)>,
    ) -> Result<Value<'c, 'c>> {
        let low = self.kq_bits_dyn(block, qs, shift, 0x0F0F_0F0F)?;
        let Some((qh, bit)) = qh else {
            return Ok(low);
        };
        let fifth = self.kq_bits_dyn(block, qh, bit, 0x0101_0101)?;
        let four = self.const_i32(block, 4)?;
        let fifth = self.push(block, arith::shli(fifth, four, self.loc))?;
        self.push(block, arith::ori(low, fifth, self.loc))
    }

    /// Four Q6_K quants as signed bytes, `q - 32`: the nibbles of `ql` that
    /// `nib_shift` selects and the two bits of `qh` that `qh_shift` does,
    /// the offset folded in without a carry (see the module comment).
    pub(super) fn kq_q6_bytes(
        &self,
        block: &Block<'c>,
        ql: Value<'c, 'c>,
        qh: Value<'c, 'c>,
        nib_shift: Value<'c, 'c>,
        qh_shift: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let low = self.kq_bits_dyn(block, ql, nib_shift, 0x0F0F_0F0F)?;
        let two = self.kq_bits_dyn(block, qh, qh_shift, 0x0303_0303)?;
        let fourteen = self.const_i32(block, 0x0E0E_0E0E)?;
        let nibble = self.const_i32(block, 0x0F0F_0F0F)?;
        let four = self.const_i32(block, 4)?;
        let high = self.push(block, arith::addi(two, fourteen, self.loc))?;
        let high = self.push(block, arith::andi(high, nibble, self.loc))?;
        let high = self.push(block, arith::shli(high, four, self.loc))?;
        self.push(block, arith::ori(low, high, self.loc))
    }

    /// Sixteen bytes at a sixteen-aligned offset of the column's block, as
    /// four words.
    pub(super) fn qg_u128(
        &self,
        block: &Block<'c>,
        qb: &MemVal<'c>,
        lanes: &Lanes<'c>,
        off: Value<'c, 'c>,
    ) -> Result<[Value<'c, 'c>; 4]> {
        let v = self.vec_load_al(
            block,
            qb.mem,
            &[lanes.qb_row, off],
            Type::vector(&[16], self.i8_t),
            16,
        )?;
        let v = self.vec_bitcast(block, v, Type::vector(&[4], self.i32_t))?;
        Ok([
            self.vec_extract(block, v, &[0], self.i32_t)?,
            self.vec_extract(block, v, &[1], self.i32_t)?,
            self.vec_extract(block, v, &[2], self.i32_t)?,
            self.vec_extract(block, v, &[3], self.i32_t)?,
        ])
    }

    /// The staged projection's reads for a K-quant thread: its column's
    /// block header and the bytes of its 32-element group, then the group
    /// index the decode needs as a runtime value.
    pub(super) fn kq_gemm_load(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        lanes: &Lanes<'c>,
        qb: &MemVal<'c>,
        at: &TileAt<'c>,
        regs: &mut Vec<Value<'c, 'c>>,
    ) -> Result<()> {
        let two = self.const_index(block, 2)?;
        let four = self.const_index(block, 4)?;
        match fmt {
            QgFormat::Q4k | QgFormat::Q5k => {
                // d, dmin and the twelve scale bytes: one sixteen-byte load.
                regs.extend(self.qg_u128(block, qb, lanes, at.blk_off)?);
                if fmt == QgFormat::Q5k {
                    // The whole qh plane; run `ib` takes bit `ib` of it.
                    for off in [16, 32] {
                        let at_qh = self.qg_at(block, at, off, 0)?;
                        regs.extend(self.qg_u128(block, qb, lanes, at_qh)?);
                    }
                }
                // Runs 2p and 2p + 1 share the 32-byte plane p, one the low
                // nibbles and one the high, so a group is the whole plane.
                let qs_base = if fmt == QgFormat::Q5k { 48 } else { 16 };
                let pair = self.divui(block, at.ib, two)?;
                let thirty_two = self.const_index(block, 32)?;
                let plane = self.muli(block, pair, thirty_two)?;
                let base = self.const_index(block, qs_base)?;
                let off = self.addi(block, plane, base)?;
                let off = self.addi(block, at.blk_off, off)?;
                for half in [0, 16] {
                    let step = self.const_index(block, half)?;
                    let at_qs = self.addi(block, off, step)?;
                    regs.extend(self.qg_u128(block, qb, lanes, at_qs)?);
                }
                regs.push(self.numeric_cast(block, at.ib, self.i32_t)?);
            }
            QgFormat::Q6k => {
                // Group `ib` is quarter `ib % 4` of the 128-element group
                // `ib / 4`: 32 `ql` bytes at `64 g + 32 (quarter % 2)`,
                // nibble `quarter / 2`; the 32 `qh` bytes of the group at
                // `128 + 32 g`, bits `2 quarter`.
                let g = self.divui(block, at.ib, four)?;
                let parity = self.remui(block, at.ib, two)?;
                let sixty_four = self.const_index(block, 64)?;
                let thirty_two = self.const_index(block, 32)?;
                let ql = self.muli(block, g, sixty_four)?;
                let half = self.muli(block, parity, thirty_two)?;
                let ql = self.addi(block, ql, half)?;
                let ql = self.addi(block, at.blk_off, ql)?;
                for off in [0, 16] {
                    let step = self.const_index(block, off)?;
                    let at_ql = self.addi(block, ql, step)?;
                    regs.extend(self.qg_u128(block, qb, lanes, at_ql)?);
                }
                let qh = self.muli(block, g, thirty_two)?;
                let qh_base = self.const_index(block, 128)?;
                let qh = self.addi(block, qh, qh_base)?;
                let qh = self.addi(block, at.blk_off, qh)?;
                for off in [0, 16] {
                    let step = self.const_index(block, off)?;
                    let at_qh = self.addi(block, qh, step)?;
                    regs.extend(self.qg_u128(block, qb, lanes, at_qh)?);
                }
                // The two signed scale bytes of runs 2 ib and 2 ib + 1.
                let sc = self.qg_at(block, at, 192, 2)?;
                regs.push(self.qg_u16(block, qb, lanes, sc)?);
                let quarter = self.remui(block, at.ib, four)?;
                regs.push(self.numeric_cast(block, quarter, self.i32_t)?);
            }
            _ => bail!("{} is not a K-quant", fmt.intrinsic()),
        }
        Ok(())
    }

    /// Decode the thread's group from what [`Self::kq_gemm_load`] read: four
    /// octets into the stage, the scale, and the minimum where there is one.
    pub(super) fn kq_gemm_decode(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        stage: &Stage<'_, 'c>,
        regs: &[Value<'c, 'c>],
    ) -> Result<()> {
        let dv = stage.dv;
        match fmt {
            QgFormat::Q4k | QgFormat::Q5k => {
                let (hdr, rest) = regs.split_at(4);
                let (qh, rest) = if fmt == QgFormat::Q5k {
                    rest.split_at(8)
                } else {
                    rest.split_at(0)
                };
                let (qs, ib) = (&rest[..8], rest[8]);
                let dmin = self.kq_f16_of(block, hdr[0], true)?;
                let (sc, m) = self.kq_scale_min(block, [hdr[1], hdr[2], hdr[3]], ib)?;
                let sc = self.small_int_to_f32(block, sc)?;
                let sw = self.push(block, arith::mulf(dv, sc, self.loc))?;
                self.qgemm_put_scale(block, stage, 1, 0, sw)?;
                let m = self.small_int_to_f32(block, m)?;
                let mw = self.push(block, arith::mulf(dmin, m, self.loc))?;
                self.qgemm_put_min(block, stage, mw)?;
                // The low nibbles for an even run, the high for an odd.
                let one = self.const_i32(block, 1)?;
                let two = self.const_i32(block, 2)?;
                let parity = self.push(block, arith::andi(ib, one, self.loc))?;
                let shift = self.push(block, arith::shli(parity, two, self.loc))?;
                for l in 0..4usize {
                    let fifth = |i: usize| (fmt == QgFormat::Q5k).then(|| (qh[i], ib));
                    let w0 = self.kq_nibbles(block, qs[2 * l], shift, fifth(2 * l))?;
                    let w1 = self.kq_nibbles(block, qs[2 * l + 1], shift, fifth(2 * l + 1))?;
                    self.qgemm_put_octet(block, stage, l as i64, w0, w1)?;
                }
            }
            QgFormat::Q6k => {
                let (ql, rest) = regs.split_at(8);
                let (qh, rest) = rest.split_at(8);
                let (sc16, quarter) = (rest[0], rest[1]);
                // Sign-extend each scale byte: shift it to the top, then
                // arithmetic shift back.
                let c24 = self.const_i32(block, 24)?;
                for h in 0..2i64 {
                    let up = self.const_i32(block, 24 - 8 * h)?;
                    let sc = self.push(block, arith::shli(sc16, up, self.loc))?;
                    let sc = self.push(block, arith::shrsi(sc, c24, self.loc))?;
                    let sc = self.small_int_to_f32(block, sc)?;
                    let sw = self.push(block, arith::mulf(dv, sc, self.loc))?;
                    self.qgemm_put_scale(block, stage, 2, h, sw)?;
                }
                let one = self.const_i32(block, 1)?;
                let two = self.const_i32(block, 2)?;
                let nib_shift = self.push(block, arith::shrui(quarter, one, self.loc))?;
                let nib_shift = self.push(block, arith::shli(nib_shift, two, self.loc))?;
                let qh_shift = self.push(block, arith::shli(quarter, one, self.loc))?;
                for l in 0..4usize {
                    let w0 = self.kq_q6_bytes(block, ql[2 * l], qh[2 * l], nib_shift, qh_shift)?;
                    let w1 =
                        self.kq_q6_bytes(block, ql[2 * l + 1], qh[2 * l + 1], nib_shift, qh_shift)?;
                    self.qgemm_put_octet(block, stage, l as i64, w0, w1)?;
                }
            }
            _ => bail!("{} is not a K-quant", fmt.intrinsic()),
        }
        Ok(())
    }

    /// The sum of the four bytes of each of `words`, as one i32: `dp4a`
    /// against a word of ones.
    pub(super) fn kq_byte_sum(
        &self,
        block: &Block<'c>,
        words: &[Value<'c, 'c>],
        seed: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let quad_t = Type::vector(&[4], self.i8_t);
        let one_i32 = Type::vector(&[1], self.i32_t);
        let ones = self.const_i32(block, 0x0101_0101)?;
        let ones = self.vec_broadcast(block, ones, one_i32)?;
        let ones = self.vec_bitcast(block, ones, quad_t)?;
        let mut acc = seed;
        for &w in words {
            let bytes = self.vec_broadcast(block, w, one_i32)?;
            let bytes = self.vec_bitcast(block, bytes, quad_t)?;
            acc = self.dot4_accumulate(block, bytes, ones, acc)?;
        }
        Ok(acc)
    }
}
