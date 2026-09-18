// PTQ1_0 in the staged projection (`qgemm.rs`) and the decode matvec
// (`qdot_i8_reg.rs`). The device block is 256 weights in 56 bytes: quarter
// `q` owns the three words at `12 q`, the tail byte `48 + q`, and the `f16`
// scale at `52 + 2 (q / 2)`. Trit `n` of byte `b` of word `w` is weight
// `4 (5 w + n) + b` of the quarter, so one trit of a word is a `dp4a` operand;
// the tail byte's four trits are weights 60..64. See `phobos-gguf`'s
// `quant/ptq1_0.rs`, which lays the file's blocks out this way at upload.
//
// A byte holds five trits base three, scaled so trit `n` is the top trit of
// `b * 3^n mod 256`: `((b * 3^n mod 256) * 3) >> 8`. Two bytes to a word in
// 16-bit lanes carry no multiply into each other, so a word decodes as two
// lane pairs, one `* 3`, one `prmt` of the high bytes and one mask a trit.
// The trits are 0..2 for weights `d * (t - 1)`: the matvec takes `t` to the
// `dp4a` and subtracts the activation's sum with one more against -1 bytes,
// and the projection writes `t - 1` as `(t + 0x7F) ^ 0x80` a byte, which
// never carries.

use super::qdot_i8_reg::Piece;
use super::qgemm::{Lanes, Stage, TileAt};
use super::*;

/// Where a device block keeps the quarters' tails and scales.
const PT_TAILS: i64 = 48;
const PT_SCALES: i64 = 52;

impl<'c> Codegen<'c> {
    /// The first `count` trit planes of `word`: word `n` holds trit `n` of
    /// each of its four bytes, 0..2 a byte.
    fn pt_planes(&self, block: &Block<'c>, word: Value<'c, 'c>, count: usize) -> Result<Vec<Value<'c, 'c>>> {
        let zero = self.const_i32(block, 0)?;
        let lo_sel = self.const_i32(block, 0x4140)?;
        let hi_sel = self.const_i32(block, 0x4342)?;
        let lanes = [
            self.byte_permute(block, word, zero, lo_sel)?,
            self.byte_permute(block, word, zero, hi_sel)?,
        ];
        self.pt_advance(block, lanes, count)
    }

    /// Trits off two lane pairs, each 16-bit lane `x mod 256` of some byte
    /// times a power of three: one plane a step.
    fn pt_advance(&self, block: &Block<'c>, mut lanes: [Value<'c, 'c>; 2], count: usize) -> Result<Vec<Value<'c, 'c>>> {
        let three = self.const_i32(block, 3)?;
        let mask = self.const_i32(block, 0x00FF_00FF)?;
        let tops = self.const_i32(block, 0x7531)?;
        let mut planes = Vec::with_capacity(count);
        for n in 0..count {
            let lo = self.push(block, arith::muli(lanes[0], three, self.loc))?;
            let hi = self.push(block, arith::muli(lanes[1], three, self.loc))?;
            planes.push(self.byte_permute(block, lo, hi, tops)?);
            if n + 1 < count {
                lanes = [
                    self.push(block, arith::andi(lo, mask, self.loc))?,
                    self.push(block, arith::andi(hi, mask, self.loc))?,
                ];
            }
        }
        Ok(planes)
    }

    /// Quarter `q`'s tail byte of `tails` as one word of its four trits: the
    /// byte times 1, 3, 9 and 27 in four lanes, then one step of
    /// [`Self::pt_advance`].
    fn pt_tail(&self, block: &Block<'c>, tails: Value<'c, 'c>, q: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let eight = self.const_i32(block, 8)?;
        let shift = self.push(block, arith::muli(q, eight, self.loc))?;
        let byte = self.kq_bits_dyn(block, tails, shift, 0xFF)?;
        let mask = self.const_i32(block, 0x00FF_00FF)?;
        let mut lanes = Vec::with_capacity(2);
        for powers in [0x0003_0001, 0x001B_0009] {
            let powers = self.const_i32(block, powers)?;
            let spread = self.push(block, arith::muli(byte, powers, self.loc))?;
            lanes.push(self.push(block, arith::andi(spread, mask, self.loc))?);
        }
        Ok(self.pt_advance(block, [lanes[0], lanes[1]], 1)?[0])
    }

    /// Trits 0..2 a byte as the weights' signed `t - 1`.
    fn pt_signed(&self, block: &Block<'c>, trits: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let bias = self.const_i32(block, 0x7F7F_7F7F)?;
        let flip = self.const_i32(block, 0x8080_8080u32 as i32 as i64)?;
        let biased = self.push(block, arith::addi(trits, bias, self.loc))?;
        self.push(block, arith::xori(biased, flip, self.loc))
    }

    /// Quarter `q`'s scale out of the word holding both, as f32.
    fn pt_scale(&self, block: &Block<'c>, scales: Value<'c, 'c>, q: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let one = self.const_i32(block, 1)?;
        let four = self.const_i32(block, 4)?;
        let half = self.push(block, arith::shrui(q, one, self.loc))?;
        let shift = self.push(block, arith::shli(half, four, self.loc))?;
        let bits = self.kq_bits_dyn(block, scales, shift, 0xFFFF)?;
        self.kq_f16_of(block, bits, false)
    }

    /// The staged projection's reads for a PTQ1_0 thread: the three words
    /// of its group's quarter, the tails and the scales, then the quarter
    /// and which half of it the group is.
    pub(super) fn pt_gemm_load(
        &mut self,
        block: &Block<'c>,
        lanes: &Lanes<'c>,
        qb: &MemVal<'c>,
        at: &TileAt<'c>,
        regs: &mut Vec<Value<'c, 'c>>,
    ) -> Result<()> {
        let two = self.const_index(block, 2)?;
        let q = self.divui(block, at.ib, two)?;
        let twelve = self.const_index(block, 12)?;
        let base = self.muli(block, q, twelve)?;
        let base = self.addi(block, at.blk_off, base)?;
        for w in 0..3 {
            let off = self.const_index(block, 4 * w)?;
            let off = self.addi(block, base, off)?;
            regs.push(self.qg_u32(block, qb, lanes, off)?);
        }
        for field in [PT_TAILS, PT_SCALES] {
            let off = self.const_index(block, field)?;
            let off = self.addi(block, at.blk_off, off)?;
            regs.push(self.qg_u32(block, qb, lanes, off)?);
        }
        let half = self.remui(block, at.ib, two)?;
        regs.push(self.numeric_cast(block, q, self.i32_t)?);
        regs.push(self.numeric_cast(block, half, self.i32_t)?);
        Ok(())
    }

    /// Decode the thread's group from what [`Self::pt_gemm_load`] read. The
    /// first half of a quarter is word 0's five planes and word 1's first
    /// three; the second is word 1's last two, word 2's five and the tail.
    pub(super) fn pt_gemm_decode(&mut self, block: &Block<'c>, stage: &Stage<'_, 'c>, regs: &[Value<'c, 'c>]) -> Result<()> {
        let [w0, w1, w2, tails, scales, q, half] = regs else {
            bail!("ptq1_qgemm_t decodes seven registers, got {}", regs.len());
        };
        let zero = self.const_i32(block, 0)?;
        let first = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, *half, zero, self.loc),
        )?;
        let outer = self.select(block, first, *w0, *w2)?;
        let f = self.pt_planes(block, outer, 5)?;
        let m = self.pt_planes(block, *w1, 5)?;
        let tail = self.pt_tail(block, *tails, *q)?;
        let lead = [f[0], f[1], f[2], f[3], f[4], m[0], m[1], m[2]];
        let trail = [m[3], m[4], f[0], f[1], f[2], f[3], f[4], tail];
        let mut words = Vec::with_capacity(8);
        for (a, b) in lead.into_iter().zip(trail) {
            let t = self.select(block, first, a, b)?;
            words.push(self.pt_signed(block, t)?);
        }
        for l in 0..4 {
            self.qgemm_put_octet(block, stage, l as i64, words[2 * l], words[2 * l + 1])?;
        }
        let d = self.pt_scale(block, *scales, *q)?;
        self.qgemm_put_scale(block, stage, 1, 0, d)
    }

    /// The loads of a decode lane's quarter: its three words, then the
    /// tails and scales the column's four lanes share through L1.
    pub(super) fn pt_pieces(&mut self, body: &Block<'c>, quarter: Value<'c, 'c>) -> Result<Vec<Piece<'c>>> {
        let twelve = self.const_index(body, 12)?;
        let base = self.muli(body, quarter, twelve)?;
        let mut pieces = Vec::with_capacity(4);
        for w in 0..3 {
            let off = self.const_index(body, 4 * w)?;
            pieces.push(Piece { off: self.addi(body, base, off)?, width: 4 });
        }
        pieces.push(Piece { off: self.const_index(body, PT_TAILS)?, width: 8 });
        Ok(pieces)
    }

    /// `carry` plus one block of a decode lane's quarter, `regs` as
    /// [`Self::pt_pieces`] laid them out, against the activations from
    /// `k_off`: two groups of eight `dp4a` against the trits and eight
    /// against -1, whose sum is the exact `sum (t - 1) a`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pt_qdot_block(
        &mut self,
        kb: &Block<'c>,
        regs: &[Value<'c, 'c>],
        aq: &MemVal<'c>,
        asc: &MemVal<'c>,
        k_off: Value<'c, 'c>,
        carry: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let [w0, w1, w2, tails, scales] = regs else {
            bail!("ptq1_qdot_i8_t decodes five registers, got {}", regs.len());
        };
        let (act, sa) = self.qr_act(kb, QgFormat::Ptq1, aq, asc, k_off)?;
        let block_w = self.const_index(kb, 256)?;
        let quarter_w = self.const_index(kb, 64)?;
        let in_block = self.remui(kb, k_off, block_w)?;
        let q = self.divui(kb, in_block, quarter_w)?;
        let q = self.numeric_cast(kb, q, self.i32_t)?;

        let mut words = Vec::with_capacity(16);
        for w in [*w0, *w1, *w2] {
            words.extend(self.pt_planes(kb, w, 5)?);
        }
        words.push(self.pt_tail(kb, *tails, q)?);
        let quad_t = Type::vector(&[4], self.i8_t);
        let one_i32 = Type::vector(&[1], self.i32_t);
        let as_bytes = |cg: &Self, w: Value<'c, 'c>| -> Result<Value<'c, 'c>> {
            let v = cg.vec_broadcast(kb, w, one_i32)?;
            cg.vec_bitcast(kb, v, quad_t)
        };
        let minus = self.const_i32(kb, -1)?;
        let minus = as_bytes(self, minus)?;

        let d = self.pt_scale(kb, *scales, q)?;
        let mut acc = carry;
        for (g, &s) in sa.iter().enumerate() {
            let mut dot = self.zero_scalar(kb, self.i32_t)?;
            for o in 8 * g..8 * g + 8 {
                let t = as_bytes(self, words[o])?;
                dot = self.dot4_accumulate(kb, t, act[o], dot)?;
                dot = self.dot4_accumulate(kb, minus, act[o], dot)?;
            }
            let dot = self.small_int_to_f32(kb, dot)?;
            let weight = self.push(kb, arith::mulf(d, s, self.loc))?;
            acc = self.elem_mac(kb, self.f32_t, dot, weight, acc)?;
        }
        Ok(acc)
    }
}
