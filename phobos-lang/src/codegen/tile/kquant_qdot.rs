// The K-quants' side of `<fmt>_qdot_i8_t` (`qdot_i8_reg.rs`): what a lane's
// quarter of a block loads, where its 64 activations sit, the activation
// run sums the minimum term needs, and the decode of one block. The
// arithmetic is `kquant.rs`'s.
//
// A Q4_K or Q5_K quarter is runs `2 q` and `2 q + 1`: one contiguous 32-byte
// span of `qs`, low nibbles then high, so the lane's activations are the
// 64 contiguous elements the intrinsic already loads. A Q6_K block is
// interleaved across two 128-element groups, so a lane takes sixteen
// elements of each of a group's four quarters instead: `128 g + l0 +
// {0..16} + 32 {0, 1, 2, 3}` for `g = q / 2`, `l0 = 16 (q % 2)`, four runs
// of sixteen that are each exactly one Q6_K scale run, from three sixteen-
// byte loads (`ql` at `64 g + l0` and 32 past it, `qh` at `128 + 32 g +
// l0`), with the activations four sixteen-byte loads 32 apart.

use super::kquant::KQ_GROUP;
use super::qdot_i8_reg::Piece;
use super::*;

/// Groups of 32 the activation-sum prologue can hold: `k` up to 32768,
/// four kilobytes of shared memory. `K` is dynamic, so the tile is sized
/// here and the backend refuses a wider row.
pub(in crate::codegen) const KQ_MAX_GROUPS: i64 = 1024;

impl<'c> Codegen<'c> {
    /// The loads of a K-quant lane's quarter, sixteen bytes each.
    pub(super) fn kq_pieces(
        &mut self,
        body: &Block<'c>,
        fmt: QgFormat,
        quarter: Value<'c, 'c>,
    ) -> Result<Vec<Piece<'c>>> {
        let c = |cg: &mut Self, v: i64| cg.const_index(body, v);
        let piece = |off, width| Piece { off, width };
        Ok(match fmt {
            QgFormat::Q4k | QgFormat::Q5k => {
                // The header, shared by the column's four lanes through L1;
                // Q5_K's whole qh plane likewise; then the quarter's span.
                let qs_base = if fmt == QgFormat::Q5k { 48 } else { 16 };
                let thirty_two = c(self, 32)?;
                let span = self.muli(body, quarter, thirty_two)?;
                let mut pieces = vec![piece(c(self, 0)?, 16)];
                if fmt == QgFormat::Q5k {
                    pieces.push(piece(c(self, 16)?, 16));
                    pieces.push(piece(c(self, 32)?, 16));
                }
                for half in [0, 16] {
                    let base = c(self, qs_base + half)?;
                    pieces.push(piece(self.addi(body, span, base)?, 16));
                }
                pieces
            }
            QgFormat::Q6k => {
                let (two, sixteen, thirty_two, sixty_four) = (c(self, 2)?, c(self, 16)?, c(self, 32)?, c(self, 64)?);
                let g = self.divui(body, quarter, two)?;
                let l0 = self.remui(body, quarter, two)?;
                let l0 = self.muli(body, l0, sixteen)?;
                let ql = self.muli(body, g, sixty_four)?;
                let ql = self.addi(body, ql, l0)?;
                let ql_hi = self.addi(body, ql, thirty_two)?;
                let qh = self.muli(body, g, thirty_two)?;
                let qh = self.addi(body, qh, l0)?;
                let qh_base = c(self, 128)?;
                let qh = self.addi(body, qh, qh_base)?;
                vec![
                    piece(ql, 16),
                    piece(ql_hi, 16),
                    piece(qh, 16),
                    // The sixteen scales, shared by the column's lanes.
                    piece(c(self, 192)?, 16),
                ]
            }
            _ => bail!("{} is not a K-quant", fmt.qdot_i8_intrinsic()),
        })
    }

    /// Where a lane's activations start within the block: 64 apart for the
    /// contiguous formats, the interleaved offset for Q6_K.
    pub(super) fn kq_lane_k_off(
        &mut self,
        body: &Block<'c>,
        fmt: QgFormat,
        quarter: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        if fmt != QgFormat::Q6k {
            let sixty_four = self.const_index(body, 64)?;
            return self.muli(body, quarter, sixty_four);
        }
        let (two, sixteen, c128) = (
            self.const_index(body, 2)?,
            self.const_index(body, 16)?,
            self.const_index(body, 128)?,
        );
        let g = self.divui(body, quarter, two)?;
        let l0 = self.remui(body, quarter, two)?;
        let g = self.muli(body, g, c128)?;
        let l0 = self.muli(body, l0, sixteen)?;
        self.addi(body, g, l0)
    }

    /// Sum every 32-element group of the activation row into `sums`, one
    /// i32 a group, the CTA's threads striding over the groups. The caller
    /// barriers after.
    pub(super) fn kq_sum_prologue(
        &mut self,
        block: &Block<'c>,
        aq: &MemVal<'c>,
        sums: &MemVal<'c>,
        kd: Value<'c, 'c>,
        tid: Value<'c, 'c>,
        bdim: Value<'c, 'c>,
    ) -> Result<()> {
        let group_w = self.const_index(block, KQ_GROUP)?;
        let groups = self.divui(block, kd, group_w)?;
        let body = Block::new(&[(self.index_t, self.loc)]);
        let g = detach(body.argument(0)?.into());
        let zero = self.const_index(&body, 0)?;
        let thirty_two = self.const_index(&body, KQ_GROUP)?;
        let sixteen = self.const_index(&body, 16)?;
        let at = self.muli(&body, g, thirty_two)?;
        let at_hi = self.addi(&body, at, sixteen)?;
        let bytes16 = Type::vector(&[16], self.i8_t);
        let words_t = Type::vector(&[4], self.i32_t);
        let mut words = Vec::with_capacity(8);
        for at in [at, at_hi] {
            let v = self.vec_load_al(&body, aq.mem, &[zero, at], bytes16, 16)?;
            let v = self.vec_bitcast(&body, v, words_t)?;
            for w in 0..4 {
                words.push(self.vec_extract(&body, v, &[w], self.i32_t)?);
            }
        }
        let seed = self.zero_scalar(&body, self.i32_t)?;
        let sum = self.kq_byte_sum(&body, &words, seed)?;
        body.append_operation(memref::store(sum, sums.mem, &[zero, g], self.loc));
        body.append_operation(scf::r#yield(&[], self.loc));
        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(tid, groups, bdim, region, self.loc));
        Ok(())
    }

    /// A word's four bytes as a `dp4a` operand.
    fn kq_as_bytes(&self, block: &Block<'c>, w: Value<'c, 'c>) -> Result<Value<'c, 'c>> {
        let v = self.vec_broadcast(block, w, Type::vector(&[1], self.i32_t))?;
        self.vec_bitcast(block, v, Type::vector(&[4], self.i8_t))
    }

    /// `carry` plus one block of a K-quant lane's quarter, `regs` as
    /// [`Self::kq_pieces`] laid it out (plus the plane's `d` last for
    /// Q6_K), against the activations from `k_off`. `tabs[0]` is the run
    /// sums for a format with a minimum.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn kq_qdot_block(
        &mut self,
        kb: &Block<'c>,
        fmt: QgFormat,
        tabs: &[MemVal<'c>],
        regs: &[Value<'c, 'c>],
        aq: &MemVal<'c>,
        asc: &MemVal<'c>,
        k_off: Value<'c, 'c>,
        carry: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let f32_t = self.f32_t;
        let (act, sa) = self.qr_act(kb, fmt, aq, asc, k_off)?;
        let block_w = self.const_index(kb, 256)?;
        let in_block = self.remui(kb, k_off, block_w)?;
        let mut acc = carry;
        match fmt {
            QgFormat::Q4k | QgFormat::Q5k => {
                let (hdr, rest) = regs.split_at(4);
                let (qh, qs) = if fmt == QgFormat::Q5k { rest.split_at(8) } else { rest.split_at(0) };
                let dv = self.kq_f16_of(kb, hdr[0], false)?;
                let dmin = self.kq_f16_of(kb, hdr[0], true)?;
                let sixty_four = self.const_index(kb, 64)?;
                let quarter = self.divui(kb, in_block, sixty_four)?;
                let quarter = self.numeric_cast(kb, quarter, self.i32_t)?;
                let group_w = self.const_index(kb, KQ_GROUP)?;
                let group0 = self.divui(kb, k_off, group_w)?;
                let zero = self.const_index(kb, 0)?;
                let two = self.const_i32(kb, 2)?;
                let r0 = self.push(kb, arith::muli(quarter, two, self.loc))?;
                for h in 0..2usize {
                    // Run 2 q + h: its scale and minimum, its activation sum.
                    let h_i32 = self.const_i32(kb, h as i64)?;
                    let r = self.push(kb, arith::addi(r0, h_i32, self.loc))?;
                    let (sc, m) = self.kq_scale_min(kb, [hdr[1], hdr[2], hdr[3]], r)?;
                    let h_idx = self.const_index(kb, h as i64)?;
                    let group = self.addi(kb, group0, h_idx)?;
                    let sum = self.push(kb, memref::load(tabs[0].mem, &[zero, group], self.loc))?;
                    let shift = self.const_i32(kb, 4 * h as i64)?;
                    let mut dot = self.zero_scalar(kb, self.i32_t)?;
                    for o in 0..4usize {
                        for w in 0..2usize {
                            let idx = 2 * o + w;
                            let fifth = (fmt == QgFormat::Q5k).then(|| (qh[idx], r));
                            let q = self.kq_nibbles(kb, qs[idx], shift, fifth)?;
                            let q = self.kq_as_bytes(kb, q)?;
                            dot = self.dot4_accumulate(kb, q, act[8 * h + idx], dot)?;
                        }
                    }
                    let dot = self.small_int_to_f32(kb, dot)?;
                    let sum = self.small_int_to_f32(kb, sum)?;
                    let sc = self.small_int_to_f32(kb, sc)?;
                    let m = self.small_int_to_f32(kb, m)?;
                    let weight = self.push(kb, arith::mulf(dv, sa[h], self.loc))?;
                    let weight = self.push(kb, arith::mulf(weight, sc, self.loc))?;
                    let min = self.push(kb, arith::mulf(dmin, sa[h], self.loc))?;
                    let min = self.push(kb, arith::mulf(min, m, self.loc))?;
                    let min = self.push(kb, arith::negf(min, self.loc))?;
                    acc = self.elem_mac(kb, f32_t, dot, weight, acc)?;
                    acc = self.elem_mac(kb, f32_t, sum, min, acc)?;
                }
            }
            QgFormat::Q6k => {
                let (ql_lo, rest) = regs.split_at(4);
                let (ql_hi, rest) = rest.split_at(4);
                let (qh, rest) = rest.split_at(4);
                let (sc, rest) = rest.split_at(4);
                let dv = self.numeric_cast(kb, rest[0], f32_t)?;
                let (c16, c128) = (self.const_index(kb, 16)?, self.const_index(kb, 128)?);
                let g = self.divui(kb, in_block, c128)?;
                let l0 = self.remui(kb, in_block, c128)?;
                let l0 = self.divui(kb, l0, c16)?;
                let zero_idx = self.const_index(kb, 0)?;
                let first = self.push(
                    kb,
                    arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, g, zero_idx, self.loc),
                )?;
                // The scales of runs 8 g + l0 / 16 + 2 i: bytes l0 / 16 + 2 i
                // of the word pair group g owns.
                let lo = self.select(kb, first, sc[0], sc[2])?;
                let hi = self.select(kb, first, sc[1], sc[3])?;
                let b0 = self.numeric_cast(kb, l0, self.i32_t)?;
                for i in 0..4usize {
                    let two_i = self.const_i32(kb, 2 * i as i64)?;
                    let b = self.push(kb, arith::addi(b0, two_i, self.loc))?;
                    let scale = self.kq_sbyte_at(kb, lo, hi, b)?;
                    let ql = if i % 2 == 0 { ql_lo } else { ql_hi };
                    let nib_shift = self.const_i32(kb, 4 * (i as i64 / 2))?;
                    let qh_shift = self.const_i32(kb, 2 * i as i64)?;
                    let mut dot = self.zero_scalar(kb, self.i32_t)?;
                    for w in 0..4usize {
                        let q = self.kq_q6_bytes(kb, ql[w], qh[w], nib_shift, qh_shift)?;
                        let q = self.kq_as_bytes(kb, q)?;
                        dot = self.dot4_accumulate(kb, q, act[4 * i + w], dot)?;
                    }
                    let dot = self.small_int_to_f32(kb, dot)?;
                    let scale = self.small_int_to_f32(kb, scale)?;
                    let weight = self.push(kb, arith::mulf(dv, sa[i], self.loc))?;
                    let weight = self.push(kb, arith::mulf(weight, scale, self.loc))?;
                    acc = self.elem_mac(kb, f32_t, dot, weight, acc)?;
                }
            }
            _ => bail!("{} is not a K-quant", fmt.qdot_i8_intrinsic()),
        }
        Ok(acc)
    }
}
