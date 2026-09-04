// What each raw format contributes to the staged projection in `qgemm.rs`:
// the bytes a thread reads for its column and 32-element group, and how they
// become four octets of int8 weights plus one scale a group or a half.
// IQ1_S and IQ1_M decode by `prmt` from a two-bit grid into `8g +- 1`
// bytes; the IQ2 and IQ3 grids are int8 already, signed by a 0/-1 byte mask
// as `(m ^ mask) + (mask & 0x01010101)`. The K-quants, Q4_K, Q5_K and Q6_K,
// have no tables and decode in `kquant.rs`.

use super::qgemm::{Lanes, Stage, TileAt};
use super::*;

/// A raw format the staged projection decodes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::codegen) enum QgFormat {
    Iq1s,
    Iq1m,
    Iq2xxs,
    Iq2xs,
    Iq2s,
    Iq3xxs,
    Iq3s,
    Q4k,
    Q5k,
    Q6k,
}

/// IQ1_S's grid at two bits a lane: 2048 entries of eight.
pub(in crate::codegen) const IQ1_GRID2_BYTES: i64 = 2048 * 2;

/// The same grid at a nibble a lane, `g + 1` in 0..2, a `prmt` selector as
/// it is; the matvecs read this one.
pub(in crate::codegen) const IQ1_GRID4_BYTES: i64 = 2048 * 4;

/// The byte tables for the nibble grid: selector `g + 1` is 0 for -1, 1 for
/// 0, 2 for +1, and the byte is `8g + 1` or `8g - 1`.
pub(super) const IQ1_LUT4_PLUS: i64 = 0x0009_01F9;
pub(super) const IQ1_LUT4_MINUS: i64 = 0x0007_FFF7;

/// The byte tables `prmt` applies to a ternary lane. A code is `g & 3`: 0 for
/// a zero, 1 for +1, 3 for -1, and byte `code` of the table is `8g + 1` or
/// `8g - 1`. Written as the i32 the register holds, low byte first.
const IQ1_LUT_PLUS: i64 = 0xF900_0901u32 as i32 as i64;
const IQ1_LUT_MINUS: i64 = 0xF700_07FFu32 as i32 as i64;

impl QgFormat {
    pub(in crate::codegen) fn from_intrinsic(callee: &str) -> Option<Self> {
        Some(match callee {
            "iq1s_qgemm_t" => Self::Iq1s,
            "iq1m_qgemm_t" => Self::Iq1m,
            "iq2xxs_qgemm_t" => Self::Iq2xxs,
            "iq2xs_qgemm_t" => Self::Iq2xs,
            "iq2s_qgemm_t" => Self::Iq2s,
            "iq3xxs_qgemm_t" => Self::Iq3xxs,
            "iq3s_qgemm_t" => Self::Iq3s,
            "q4k_qgemm_t" => Self::Q4k,
            "q5k_qgemm_t" => Self::Q5k,
            "q6k_qgemm_t" => Self::Q6k,
            _ => return None,
        })
    }

    pub(in crate::codegen) fn intrinsic(self) -> &'static str {
        match self {
            Self::Iq1s => "iq1s_qgemm_t",
            Self::Iq1m => "iq1m_qgemm_t",
            Self::Iq2xxs => "iq2xxs_qgemm_t",
            Self::Iq2xs => "iq2xs_qgemm_t",
            Self::Iq2s => "iq2s_qgemm_t",
            Self::Iq3xxs => "iq3xxs_qgemm_t",
            Self::Iq3s => "iq3s_qgemm_t",
            Self::Q4k => "q4k_qgemm_t",
            Self::Q5k => "q5k_qgemm_t",
            Self::Q6k => "q6k_qgemm_t",
        }
    }

    /// The format a `<fmt>_qdot_i8_t` call names.
    pub(in crate::codegen) fn from_qdot_i8(callee: &str) -> Option<Self> {
        Some(match callee {
            "iq1s_qdot_i8_t" => Self::Iq1s,
            "iq1m_qdot_i8_t" => Self::Iq1m,
            "iq2xxs_qdot_i8_t" => Self::Iq2xxs,
            "iq2xs_qdot_i8_t" => Self::Iq2xs,
            "iq2s_qdot_i8_t" => Self::Iq2s,
            "iq3xxs_qdot_i8_t" => Self::Iq3xxs,
            "iq3s_qdot_i8_t" => Self::Iq3s,
            "q4k_qdot_i8_t" => Self::Q4k,
            "q5k_qdot_i8_t" => Self::Q5k,
            "q6k_qdot_i8_t" => Self::Q6k,
            _ => return None,
        })
    }

    pub(in crate::codegen) fn qdot_i8_intrinsic(self) -> &'static str {
        match self {
            Self::Iq1s => "iq1s_qdot_i8_t",
            Self::Iq1m => "iq1m_qdot_i8_t",
            Self::Iq2xxs => "iq2xxs_qdot_i8_t",
            Self::Iq2xs => "iq2xs_qdot_i8_t",
            Self::Iq2s => "iq2s_qdot_i8_t",
            Self::Iq3xxs => "iq3xxs_qdot_i8_t",
            Self::Iq3s => "iq3s_qdot_i8_t",
            Self::Q4k => "q4k_qdot_i8_t",
            Self::Q5k => "q5k_qdot_i8_t",
            Self::Q6k => "q6k_qdot_i8_t",
        }
    }

    /// The device block: the file's, less the leading `d` the IQ formats
    /// carry and no device kernel reads, and less Q6_K's trailing one.
    pub(in crate::codegen) fn block_bytes(self) -> i64 {
        match self {
            Self::Iq1s => 48,
            Self::Iq1m => 56,
            Self::Iq2xxs => 64,
            Self::Iq2xs => 72,
            Self::Iq2s => 80,
            Self::Iq3xxs => 96,
            Self::Iq3s => 108,
            Self::Q4k => 144,
            Self::Q5k => 176,
            Self::Q6k => 208,
        }
    }

    /// The table operands after `(qb, d)`, by size in bytes: the magnitude
    /// grid, then the 0/-1 sign masks for the formats that carry signs apart.
    pub(in crate::codegen) fn tables(self) -> &'static [i64] {
        match self {
            Self::Iq1s | Self::Iq1m => &[IQ1_GRID2_BYTES],
            Self::Iq2xxs => &[256 * 8, 128 * 8],
            Self::Iq2xs => &[512 * 8, 128 * 8],
            Self::Iq2s => &[1024 * 8, 256 * 8],
            Self::Iq3xxs => &[256 * 4, 128 * 8],
            Self::Iq3s => &[512 * 4, 256 * 8],
            Self::Q4k | Self::Q5k | Self::Q6k => &[],
        }
    }

    /// The tables a format's decode matvec reads: the ternary formats take
    /// the nibble grid there, the rest the same tables as the projection.
    pub(in crate::codegen) fn qdot_tables(self) -> &'static [i64] {
        match self {
            Self::Iq1s | Self::Iq1m => &[IQ1_GRID4_BYTES],
            _ => self.tables(),
        }
    }

    /// Elements one weight scale covers: a whole group, or half of one.
    pub(in crate::codegen) fn scale_run(self) -> i64 {
        match self {
            Self::Iq1m | Self::Iq2xs | Self::Iq2s | Self::Q6k => 16,
            _ => 32,
        }
    }
}

impl<'c> Codegen<'c> {
    /// A byte of the column's block, zero-extended.
    pub(super) fn qg_u8(&self, block: &Block<'c>, qb: &MemVal<'c>, lanes: &Lanes<'c>, off: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let byte = self.push(block, memref::load(qb.mem, &[lanes.qb_row, off], self.loc))?;
        self.extui(block, byte, self.i32_t)
    }

    /// Two bytes, at an even offset, zero-extended.
    pub(super) fn qg_u16(&self, block: &Block<'c>, qb: &MemVal<'c>, lanes: &Lanes<'c>, off: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let i16_t: Type<'c> = IntegerType::new(self.ctx, 16).into();
        let v = self.vec_load_al(block, qb.mem, &[lanes.qb_row, off], Type::vector(&[2], self.i8_t), 2)?;
        let v = self.vec_bitcast(block, v, Type::vector(&[1], i16_t))?;
        let v = self.vec_extract(block, v, &[0], i16_t)?;
        self.extui(block, v, self.i32_t)
    }

    /// Four bytes at a four-aligned offset, as one word.
    pub(super) fn qg_u32(&self, block: &Block<'c>, qb: &MemVal<'c>, lanes: &Lanes<'c>, off: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let v = self.vec_load_al(block, qb.mem, &[lanes.qb_row, off], Type::vector(&[4], self.i8_t), 4)?;
        let v = self.vec_bitcast(block, v, Type::vector(&[1], self.i32_t))?;
        self.vec_extract(block, v, &[0], self.i32_t)
    }

    /// Eight bytes at an eight-aligned offset, as two words.
    pub(super) fn qg_u64(
        &self,
        block: &Block<'c>,
        qb: &MemVal<'c>,
        lanes: &Lanes<'c>,
        off: Value<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let v = self.vec_load_al(block, qb.mem, &[lanes.qb_row, off], Type::vector(&[8], self.i8_t), 8)?;
        let v = self.vec_bitcast(block, v, Type::vector(&[2], self.i32_t))?;
        Ok((
            self.vec_extract(block, v, &[0], self.i32_t)?,
            self.vec_extract(block, v, &[1], self.i32_t)?,
        ))
    }

    /// `blk_off + base + ib * stride`, the byte offset of a group's field.
    pub(super) fn qg_at(&self, block: &Block<'c>, at: &TileAt<'c>, base: i64, stride: i64) -> Result<Value<'c, 'c>> {
        let base = self.const_index(block, base)?;
        let stride = self.const_index(block, stride)?;
        let off = self.muli(block, at.ib, stride)?;
        let off = self.addi(block, off, base)?;
        self.addi(block, at.blk_off, off)
    }

    /// Entry `idx` of a table of `width`-byte entries in shared memory, as
    /// `width / 4` words.
    pub(super) fn qg_entry(
        &self,
        block: &Block<'c>,
        tab: &MemVal<'c>,
        idx: Value<'c, 'c>,
        width: i64,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let idx = self.numeric_cast(block, idx, self.index_t)?;
        let w = self.const_index(block, width)?;
        let at = self.muli(block, idx, w)?;
        let zero = self.const_index(block, 0)?;
        let bytes = Type::vector(&[width as u64], self.i8_t);
        let v = self.vec_load_al(block, tab.mem, &[zero, at], bytes, width)?;
        let words = self.vec_bitcast(block, v, Type::vector(&[(width / 4) as u64], self.i32_t))?;
        (0..width / 4)
            .map(|i| self.vec_extract(block, words, &[i], self.i32_t))
            .collect()
    }

    /// Byte `l` of a word, zero-extended.
    pub(super) fn qg_byte(&self, block: &Block<'c>, word: Value<'c, 'c>, l: i64) -> Result<Value<'c, 'c>> {
        self.qg_bits(block, word, 8 * l, 0xFF)
    }

    /// `(word >> shift) & mask`.
    pub(super) fn qg_bits(&self, block: &Block<'c>, word: Value<'c, 'c>, shift: i64, mask: i64) -> Result<Value<'c, 'c>> {
        let shifted = if shift == 0 {
            word
        } else {
            let s = self.const_i32(block, shift)?;
            self.push(block, arith::shrui(word, s, self.loc))?
        };
        let m = self.const_i32(block, mask)?;
        self.push(block, arith::andi(shifted, m, self.loc))
    }

    /// The masked bytes of `word` negated: `(w ^ mask) + (mask & 0x01010101)`,
    /// exact wherever a masked byte is nonzero.
    pub(super) fn qg_negate(&self, block: &Block<'c>, word: Value<'c, 'c>, mask: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let ones = self.const_i32(block, 0x0101_0101)?;
        let flipped = self.push(block, arith::xori(word, mask, self.loc))?;
        let carry = self.push(block, arith::andi(mask, ones, self.loc))?;
        self.push(block, arith::addi(flipped, carry, self.loc))
    }

    /// A ternary grid entry, eight two-bit codes, as the two `8g +- 1` words
    /// the byte table `lut` describes.
    pub(super) fn qg_ternary(
        &self,
        block: &Block<'c>,
        tab: &MemVal<'c>,
        idx: Value<'c, 'c>,
        lut: Value<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let i16_t: Type<'c> = IntegerType::new(self.ctx, 16).into();
        let idx = self.numeric_cast(block, idx, self.index_t)?;
        let two = self.const_index(block, 2)?;
        let at = self.muli(block, idx, two)?;
        let zero = self.const_index(block, 0)?;
        let e = self.vec_load_al(block, tab.mem, &[zero, at], Type::vector(&[2], self.i8_t), 2)?;
        let e = self.vec_bitcast(block, e, Type::vector(&[1], i16_t))?;
        let e = self.vec_extract(block, e, &[0], i16_t)?;
        let e = self.extui(block, e, self.i32_t)?;
        // Spread eight two-bit codes into eight nibbles: three doublings of
        // the gap between them.
        let x = self.qg_spread(block, e, 8, 0x00FF_00FF)?;
        let x = self.qg_spread(block, x, 4, 0x0F0F_0F0F)?;
        let x = self.qg_spread(block, x, 2, 0x3333_3333)?;
        let w0 = self.byte_permute(block, lut, lut, x)?;
        let c16 = self.const_i32(block, 16)?;
        let x_hi = self.push(block, arith::shrui(x, c16, self.loc))?;
        let w1 = self.byte_permute(block, lut, lut, x_hi)?;
        Ok((w0, w1))
    }

    /// A ternary grid entry from the nibble grid, one word already in
    /// selector form, as the two `8g +- 1` words the byte table `lut`
    /// describes.
    pub(super) fn qg_ternary4(
        &self,
        block: &Block<'c>,
        tab: &MemVal<'c>,
        idx: Value<'c, 'c>,
        lut: Value<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let e = self.qg_entry(block, tab, idx, 4)?[0];
        let w0 = self.byte_permute(block, lut, lut, e)?;
        let c16 = self.const_i32(block, 16)?;
        let e_hi = self.push(block, arith::shrui(e, c16, self.loc))?;
        let w1 = self.byte_permute(block, lut, lut, e_hi)?;
        Ok((w0, w1))
    }

    /// `(x | (x << by)) & mask`: one step of spreading packed fields apart.
    pub(super) fn qg_spread(&self, block: &Block<'c>, x: Value<'c, 'c>, by: i64, mask: i64) -> Result<Value<'c, 'c>> {
        let by = self.const_i32(block, by)?;
        let mask = self.const_i32(block, mask)?;
        let shifted = self.push(block, arith::shli(x, by, self.loc))?;
        let both = self.push(block, arith::ori(x, shifted, self.loc))?;
        self.push(block, arith::andi(both, mask, self.loc))
    }

    /// The byte table for a delta whose sign bit is `neg`.
    pub(super) fn qg_lut(&self, block: &Block<'c>, neg: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let zero = self.const_i32(block, 0)?;
        let is_neg = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ne, neg, zero, self.loc),
        )?;
        let plus = self.const_i32(block, IQ1_LUT_PLUS)?;
        let minus = self.const_i32(block, IQ1_LUT_MINUS)?;
        self.select(block, is_neg, minus, plus)
    }

    /// `d * (0.5 + n) * q` for an integer `n`, in the order the register
    /// kernels compute it.
    pub(super) fn qg_half_scale(&self, block: &Block<'c>, dv: Value<'c, 'c>, n: Value<'c, 'c>, q: f64) -> Result<Value<'c, 'c>> {
        // A few bits wide, so the add-and-subtract conversion is exact and
        // skips the quarter-rate `cvt`.
        let n = self.small_int_to_f32(block, n)?;
        let half = self.const_f32(block, 0.5)?;
        let q = self.const_f32(block, q)?;
        let sc = self.push(block, arith::addf(half, n, self.loc))?;
        let sc = self.push(block, arith::mulf(sc, q, self.loc))?;
        self.push(block, arith::mulf(dv, sc, self.loc))
    }

    /// `d * (2 n + 1) / 8`: the IQ1 group scale, with the delta's eighth
    /// folded in.
    pub(super) fn qg_odd_eighth(&self, block: &Block<'c>, dv: Value<'c, 'c>, n: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let two = self.const_i32(block, 2)?;
        let one = self.const_i32(block, 1)?;
        let sc = self.push(block, arith::muli(n, two, self.loc))?;
        let sc = self.push(block, arith::addi(sc, one, self.loc))?;
        let sc = self.small_int_to_f32(block, sc)?;
        let dl = self.push(block, arith::mulf(dv, sc, self.loc))?;
        let eighth = self.const_f32(block, 0.125)?;
        self.push(block, arith::mulf(dl, eighth, self.loc))
    }

    /// The low or high nibble of a scale byte for half `h` of a group.
    pub(super) fn qg_nibble(&self, block: &Block<'c>, byte: Value<'c, 'c>, h: i64) -> Result<Value<'c, 'c>> {
        self.qg_bits(block, byte, 4 * h, 0xF)
    }

    /// The thread's format-specific reads for the tile at `at`, appended to
    /// `regs` in the order `qgemm_fmt_decode` takes them back.
    pub(super) fn qgemm_fmt_load(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        lanes: &Lanes<'c>,
        qb: &MemVal<'c>,
        at: &TileAt<'c>,
        regs: &mut Vec<Value<'c, 'c>>,
    ) -> Result<()> {
        match fmt {
            QgFormat::Q4k | QgFormat::Q5k | QgFormat::Q6k => {
                self.kq_gemm_load(block, fmt, lanes, qb, at, regs)?;
            }
            QgFormat::Iq1s => {
                let qs = self.qg_at(block, at, 0, 4)?;
                let qh = self.qg_at(block, at, 32, 2)?;
                regs.push(self.qg_u32(block, qb, lanes, qs)?);
                regs.push(self.qg_u16(block, qb, lanes, qh)?);
            }
            QgFormat::Iq1m => {
                let qs = self.qg_at(block, at, 0, 4)?;
                let qh = self.qg_at(block, at, 32, 2)?;
                regs.push(self.qg_u32(block, qb, lanes, qs)?);
                regs.push(self.qg_u16(block, qb, lanes, qh)?);
                // The scale word is shared by two groups: bytes 48 + 2 (ib / 2).
                let two = self.const_index(block, 2)?;
                let pair = self.divui(block, at.ib, two)?;
                let pair2 = self.muli(block, pair, two)?;
                let base = self.const_index(block, 48)?;
                let sc = self.addi(block, pair2, base)?;
                let sc = self.addi(block, at.blk_off, sc)?;
                regs.push(self.qg_u16(block, qb, lanes, sc)?);
            }
            QgFormat::Iq2xxs => {
                let off = self.qg_at(block, at, 0, 8)?;
                let (lo, hi) = self.qg_u64(block, qb, lanes, off)?;
                regs.push(lo);
                regs.push(hi);
            }
            QgFormat::Iq2xs => {
                let off = self.qg_at(block, at, 0, 8)?;
                let (lo, hi) = self.qg_u64(block, qb, lanes, off)?;
                regs.push(lo);
                regs.push(hi);
                let sc = self.qg_at(block, at, 64, 1)?;
                regs.push(self.qg_u8(block, qb, lanes, sc)?);
            }
            QgFormat::Iq2s => {
                let qs = self.qg_at(block, at, 0, 4)?;
                let signs = self.qg_at(block, at, 32, 4)?;
                let qh = self.qg_at(block, at, 64, 1)?;
                let sc = self.qg_at(block, at, 72, 1)?;
                regs.push(self.qg_u32(block, qb, lanes, qs)?);
                regs.push(self.qg_u32(block, qb, lanes, signs)?);
                regs.push(self.qg_u8(block, qb, lanes, qh)?);
                regs.push(self.qg_u8(block, qb, lanes, sc)?);
            }
            QgFormat::Iq3xxs => {
                let qs = self.qg_at(block, at, 0, 8)?;
                let (lo, hi) = self.qg_u64(block, qb, lanes, qs)?;
                regs.push(lo);
                regs.push(hi);
                let aux = self.qg_at(block, at, 64, 4)?;
                regs.push(self.qg_u32(block, qb, lanes, aux)?);
            }
            QgFormat::Iq3s => {
                // A 108-byte block is only four-aligned, so the eight grid
                // bytes come as two words.
                let qs = self.qg_at(block, at, 0, 8)?;
                let qs_hi = self.qg_at(block, at, 4, 8)?;
                regs.push(self.qg_u32(block, qb, lanes, qs)?);
                regs.push(self.qg_u32(block, qb, lanes, qs_hi)?);
                let qh = self.qg_at(block, at, 64, 1)?;
                regs.push(self.qg_u8(block, qb, lanes, qh)?);
                let signs = self.qg_at(block, at, 72, 4)?;
                regs.push(self.qg_u32(block, qb, lanes, signs)?);
                // One scale byte covers two groups, a nibble each.
                let two = self.const_index(block, 2)?;
                let pair = self.divui(block, at.ib, two)?;
                let base = self.const_index(block, 104)?;
                let sc = self.addi(block, pair, base)?;
                let sc = self.addi(block, at.blk_off, sc)?;
                regs.push(self.qg_u8(block, qb, lanes, sc)?);
                // And which nibble: the group's parity.
                let parity = self.remui(block, at.ib, two)?;
                regs.push(self.numeric_cast(block, parity, self.i32_t)?);
            }
        }
        Ok(())
    }

    /// Decode the thread's group from the registers `qgemm_fmt_load` filled
    /// and write its four octets and its scale or scales into the stage.
    pub(super) fn qgemm_fmt_decode(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        stage: &Stage<'_, 'c>,
        regs: &[Value<'c, 'c>],
    ) -> Result<()> {
        let dv = stage.dv;
        match fmt {
            QgFormat::Q4k | QgFormat::Q5k | QgFormat::Q6k => {
                self.kq_gemm_decode(block, fmt, stage, regs)?;
            }
            QgFormat::Iq1s => {
                let (qs, qh) = (regs[0], regs[1]);
                let n = self.qg_bits(block, qh, 12, 7)?;
                let sw = self.qg_odd_eighth(block, dv, n)?;
                self.qgemm_put_scale(block, stage, 1, 0, sw)?;
                let neg = self.qg_bits(block, qh, 15, 1)?;
                let lut = self.qg_lut(block, neg)?;
                for l in 0..4 {
                    let lo = self.qg_byte(block, qs, l)?;
                    let hi = self.qg_bits(block, qh, 3 * l, 7)?;
                    let idx = self.qg_join(block, lo, hi, 8)?;
                    let (w0, w1) = self.qg_ternary(block, &stage.tabs[0], idx, lut)?;
                    self.qgemm_put_octet(block, stage, l, w0, w1)?;
                }
            }
            QgFormat::Iq1m => {
                let (qs, qh, word) = (regs[0], regs[1], regs[2]);
                // The two three-bit scales sit at 6 (ib % 2) and three above.
                let two = self.const_index(block, 2)?;
                let parity = self.remui(block, stage.lanes.g, two)?;
                let parity = self.numeric_cast(block, parity, self.i32_t)?;
                let six = self.const_i32(block, 6)?;
                let shift = self.push(block, arith::muli(parity, six, self.loc))?;
                let seven = self.const_i32(block, 7)?;
                let three = self.const_i32(block, 3)?;
                for h in 0..2 {
                    let s = if h == 0 {
                        shift
                    } else {
                        self.push(block, arith::addi(shift, three, self.loc))?
                    };
                    let n = self.push(block, arith::shrui(word, s, self.loc))?;
                    let n = self.push(block, arith::andi(n, seven, self.loc))?;
                    let sw = self.qg_odd_eighth(block, dv, n)?;
                    self.qgemm_put_scale(block, stage, 2, h, sw)?;
                }
                for l in 0..4 {
                    // Octet l takes byte l / 2 of qh: the low nibble for an
                    // even l, the high for an odd, three index bits then the
                    // delta's sign.
                    let lo = self.qg_byte(block, qs, l)?;
                    let nib = 8 * (l / 2) + 4 * (l % 2);
                    let hi = self.qg_bits(block, qh, nib, 7)?;
                    let neg = self.qg_bits(block, qh, nib + 3, 1)?;
                    let lut = self.qg_lut(block, neg)?;
                    let idx = self.qg_join(block, lo, hi, 8)?;
                    let (w0, w1) = self.qg_ternary(block, &stage.tabs[0], idx, lut)?;
                    self.qgemm_put_octet(block, stage, l, w0, w1)?;
                }
            }
            QgFormat::Iq2xxs => {
                let (grid_bytes, aux) = (regs[0], regs[1]);
                let n = self.qg_bits(block, aux, 28, 0xF)?;
                let sw = self.qg_half_scale(block, dv, n, 0.25)?;
                self.qgemm_put_scale(block, stage, 1, 0, sw)?;
                for l in 0..4 {
                    let idx = self.qg_byte(block, grid_bytes, l)?;
                    let sidx = self.qg_bits(block, aux, 7 * l, 127)?;
                    self.qg_put_signed(block, stage, l, idx, 8, sidx)?;
                }
            }
            QgFormat::Iq2xs => {
                let (lo, hi, sc) = (regs[0], regs[1], regs[2]);
                for h in 0..2 {
                    let n = self.qg_nibble(block, sc, h)?;
                    let sw = self.qg_half_scale(block, dv, n, 0.25)?;
                    self.qgemm_put_scale(block, stage, 2, h, sw)?;
                }
                for l in 0..4 {
                    // Halfword l: nine bits of grid index, seven of signs.
                    let word = if l < 2 { lo } else { hi };
                    let q16 = self.qg_bits(block, word, 16 * (l % 2), 0xFFFF)?;
                    let idx = self.qg_bits(block, q16, 0, 511)?;
                    let sidx = self.qg_bits(block, q16, 9, 127)?;
                    self.qg_put_signed(block, stage, l, idx, 8, sidx)?;
                }
            }
            QgFormat::Iq2s => {
                let (qs, signs, qh, sc) = (regs[0], regs[1], regs[2], regs[3]);
                for h in 0..2 {
                    let n = self.qg_nibble(block, sc, h)?;
                    let sw = self.qg_half_scale(block, dv, n, 0.25)?;
                    self.qgemm_put_scale(block, stage, 2, h, sw)?;
                }
                for l in 0..4 {
                    let lo = self.qg_byte(block, qs, l)?;
                    let hi = self.qg_bits(block, qh, 2 * l, 3)?;
                    let idx = self.qg_join(block, lo, hi, 8)?;
                    let sidx = self.qg_byte(block, signs, l)?;
                    self.qg_put_signed(block, stage, l, idx, 8, sidx)?;
                }
            }
            QgFormat::Iq3xxs => {
                let (lo, hi, aux) = (regs[0], regs[1], regs[2]);
                let n = self.qg_bits(block, aux, 28, 0xF)?;
                let sw = self.qg_half_scale(block, dv, n, 0.5)?;
                self.qgemm_put_scale(block, stage, 1, 0, sw)?;
                for l in 0..4 {
                    // Octet l is grid bytes 2l and 2l + 1, four lanes apiece.
                    let word = if l < 2 { lo } else { hi };
                    let g1 = self.qg_byte(block, word, 2 * (l % 2))?;
                    let g2 = self.qg_byte(block, word, 2 * (l % 2) + 1)?;
                    let sidx = self.qg_bits(block, aux, 7 * l, 127)?;
                    self.qg_put_signed_pair(block, stage, l, g1, g2, sidx)?;
                }
            }
            QgFormat::Iq3s => {
                let (lo, hi, qh, signs, sc, parity) = (regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
                let nib_lo = self.qg_nibble(block, sc, 0)?;
                let nib_hi = self.qg_nibble(block, sc, 1)?;
                let zero = self.const_i32(block, 0)?;
                let odd = self.push(
                    block,
                    arith::cmpi(self.ctx, arith::CmpiPredicate::Ne, parity, zero, self.loc),
                )?;
                let n = self.select(block, odd, nib_hi, nib_lo)?;
                let two = self.const_i32(block, 2)?;
                let one = self.const_i32(block, 1)?;
                let n = self.push(block, arith::muli(n, two, self.loc))?;
                let n = self.push(block, arith::addi(n, one, self.loc))?;
                let n = self.numeric_cast(block, n, self.f32_t)?;
                let sw = self.push(block, arith::mulf(dv, n, self.loc))?;
                self.qgemm_put_scale(block, stage, 1, 0, sw)?;
                for l in 0..4 {
                    let word = if l < 2 { lo } else { hi };
                    let b1 = self.qg_byte(block, word, 2 * (l % 2))?;
                    let b2 = self.qg_byte(block, word, 2 * (l % 2) + 1)?;
                    let h1 = self.qg_bits(block, qh, 2 * l, 1)?;
                    let h2 = self.qg_bits(block, qh, 2 * l + 1, 1)?;
                    let g1 = self.qg_join(block, b1, h1, 8)?;
                    let g2 = self.qg_join(block, b2, h2, 8)?;
                    let sidx = self.qg_byte(block, signs, l)?;
                    self.qg_put_signed_pair(block, stage, l, g1, g2, sidx)?;
                }
            }
        }
        Ok(())
    }

    /// `lo | (hi << shift)`.
    pub(super) fn qg_join(&self, block: &Block<'c>, lo: Value<'c, 'c>, hi: Value<'c, 'c>, shift: i64) -> Result<Value<'c, 'c>> {
        let s = self.const_i32(block, shift)?;
        let hi = self.push(block, arith::shli(hi, s, self.loc))?;
        self.push(block, arith::ori(lo, hi, self.loc))
    }

    /// An octet from one eight-byte magnitude entry and one eight-byte sign
    /// mask entry.
    fn qg_put_signed(
        &mut self,
        block: &Block<'c>,
        stage: &Stage<'_, 'c>,
        l: i64,
        idx: Value<'c, 'c>,
        width: i64,
        sidx: Value<'c, 'c>,
    ) -> Result<()> {
        let mags = self.qg_entry(block, &stage.tabs[0], idx, width)?;
        let masks = self.qg_entry(block, &stage.tabs[1], sidx, 8)?;
        let w0 = self.qg_negate(block, mags[0], masks[0])?;
        let w1 = self.qg_negate(block, mags[1], masks[1])?;
        self.qgemm_put_octet(block, stage, l, w0, w1)
    }

    /// An octet from two four-byte magnitude entries and one sign mask entry.
    fn qg_put_signed_pair(
        &mut self,
        block: &Block<'c>,
        stage: &Stage<'_, 'c>,
        l: i64,
        g1: Value<'c, 'c>,
        g2: Value<'c, 'c>,
        sidx: Value<'c, 'c>,
    ) -> Result<()> {
        let m0 = self.qg_entry(block, &stage.tabs[0], g1, 4)?;
        let m1 = self.qg_entry(block, &stage.tabs[0], g2, 4)?;
        let masks = self.qg_entry(block, &stage.tabs[1], sidx, 8)?;
        let w0 = self.qg_negate(block, m0[0], masks[0])?;
        let w1 = self.qg_negate(block, m1[0], masks[1])?;
        self.qgemm_put_octet(block, stage, l, w0, w1)
    }

    /// Zero-extend an integer.
    pub(super) fn extui(&self, block: &Block<'c>, v: Value<'c, 'c>, to: Type<'c>) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("arith.extui", self.loc)
                .add_operands(&[v])
                .add_results(&[to])
                .build()?,
        )
    }
}
