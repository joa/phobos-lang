// Shuffles and the CTA exchange. Both move a value as 32-bit words: one for
// a type of 32 bits or fewer, two for a 64-bit one.

use anyhow::Result;
use melior::dialect::{arith, memref, scf};
use melior::ir::{Block, BlockLike, Region, RegionLike, Type, Value, ValueLike, attribute::IntegerAttribute};

use super::Lowered;
use crate::codegen::Codegen;
use crate::ast::lanes::Mode;
use crate::ir::{Ir, OpId, OpKind};

/// How a value was packed into words, to undo it.
#[derive(Clone, Copy)]
enum Packing<'c> {
    /// An `i32`.
    Word,
    /// Bitcast from a 32-bit float.
    Float(Type<'c>),
    /// Widened from a narrower integer; floats are bitcast to one first.
    Narrow { int: Type<'c>, float: Option<Type<'c>> },
    /// An `index` that lowers to 32 bits.
    Index32,
    /// Split from 64 bits, low word first; floats and `index` are cast to
    /// `i64` first.
    Wide(Type<'c>),
}

impl<'c> Codegen<'c> {
    pub(super) fn emit_lane_op(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        match ir.kind(op).clone() {
            OpKind::Shuffle { mode, width } => {
                let v = self.scalar_ir(ir.operand(op, 0))?;
                let src = self.scalar_ir(ir.operand(op, 1))?;
                let (src, keep) = self.within_piece(block, mode, src, width)?;
                let src = self.push(block, arith::index_cast(src, self.i32_t, self.loc))?;
                let (words, packing) = self.pack(block, v)?;
                let words = words
                    .into_iter()
                    .map(|w| self.isa.shuffle(self, block, w, mode, src))
                    .collect::<Result<Vec<_>>>()?;
                let mut out = self.unpack(block, &words, packing)?;
                if let Some(keep) = keep {
                    out = self.push(block, arith::select(keep, v, out, self.loc))?;
                }
                self.set_ir(ir.result(op), Lowered::Scalar(out));
            }
            // Every thread of the group computed the same value; nothing
            // to emit.
            OpKind::Converged(_) => {
                let v = self.scalar_ir(ir.operand(op, 0))?;
                self.set_ir(ir.result(op), Lowered::Scalar(v));
            }
            OpKind::Exchange => self.exchange(block, ir, op)?,
            other => unreachable!("{} is not a lane op", other.name()),
        }
        Ok(())
    }

    /// A shuffle runs across the whole warp; a `width`-lane piece of it is
    /// arithmetic on the source lane. Returns the source for the hardware,
    /// and for `down` and `up` the condition under which the lane keeps its
    /// own value because the source is past its piece. An xor mask under
    /// the width stays in the piece by itself.
    fn within_piece(
        &self,
        block: &Block<'c>,
        mode: Mode,
        src: Value<'c, 'c>,
        width: i64,
    ) -> Result<(Value<'c, 'c>, Option<Value<'c, 'c>>)> {
        if width == self.isa.lanes() || mode == Mode::Xor {
            return Ok((src, None));
        }
        let lane = self.lane_id(block)?;
        let w = self.const_index(block, width)?;
        let at = self.remui(block, lane, w)?;
        Ok(match mode {
            Mode::From => {
                let base = self.subi(block, lane, at)?;
                let off = self.remui(block, src, w)?;
                (self.addi(block, base, off)?, None)
            }
            Mode::Down => {
                let to = self.addi(block, at, src)?;
                let past = arith::cmpi(self.ctx, arith::CmpiPredicate::Uge, to, w, self.loc);
                (src, Some(self.push(block, past)?))
            }
            Mode::Up => {
                let past = arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, at, src, self.loc);
                (src, Some(self.push(block, past)?))
            }
            Mode::Xor => unreachable!("returned above"),
        })
    }

    /// Lane 0 of each warp writes the warp's value to shared memory, and
    /// every thread reads all of them back. The barrier before keeps the
    /// scratch from overwriting what an earlier reader of its bytes still
    /// needs, and the one after keeps the next writer of those bytes off
    /// the reads.
    fn exchange(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        let v = self.scalar_ir(ir.operand(op, 0))?;
        let warps = ir.results(op).len() as i64;
        let (words, packing) = self.pack(block, v)?;
        let scratch = self.alloc_tile_shaped(block, self.i32_t, &[words.len() as i64, warps])?;
        self.barrier(block)?;
        let tid = self.thread_id(block)?;
        let lanes = self.const_index(block, self.isa.lanes())?;
        let lane = self.remui(block, tid, lanes)?;
        let warp = self.divui(block, tid, lanes)?;
        let zero = self.const_index(block, 0)?;
        let lead = self.push(block, arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, zero, self.loc))?;
        let then = Block::new(&[]);
        for (i, &w) in words.iter().enumerate() {
            let row = self.const_index(&then, i as i64)?;
            then.append_operation(memref::store(w, scratch.mem, &[row, warp], self.loc));
        }
        then.append_operation(scf::r#yield(&[], self.loc));
        let region = Region::new();
        region.append_block(then);
        block.append_operation(scf::r#if(lead, &[], region, Region::new(), self.loc));
        self.barrier(block)?;
        for (w, &res) in ir.results(op).iter().enumerate() {
            let col = self.const_index(block, w as i64)?;
            let mut got = Vec::new();
            for i in 0..words.len() {
                let row = self.const_index(block, i as i64)?;
                got.push(self.push(block, memref::load(scratch.mem, &[row, col], self.loc))?);
            }
            let out = self.unpack(block, &got, packing)?;
            self.set_ir(res, Lowered::Scalar(out));
        }
        self.barrier(block)?;
        self.release(&scratch);
        Ok(())
    }

    fn pack(&self, block: &Block<'c>, v: Value<'c, 'c>) -> Result<(Vec<Value<'c, 'c>>, Packing<'c>)> {
        let t = v.r#type();
        let loc = self.loc;
        if t == self.i32_t {
            return Ok((vec![v], Packing::Word));
        }
        if t == self.f32_t {
            let w = self.push(block, arith::bitcast(v, self.i32_t, loc))?;
            return Ok((vec![w], Packing::Float(t)));
        }
        if t == self.f16_t || t == self.bf16_t || t == self.i8_t || t == self.i16_t || t == self.bool_t {
            let float = (t == self.f16_t || t == self.bf16_t).then_some(t);
            let int = if float.is_some() { self.i16_t } else { t };
            let bits = match float {
                Some(_) => self.push(block, arith::bitcast(v, int, loc))?,
                None => v,
            };
            let w = self.push(block, arith::extui(bits, self.i32_t, loc))?;
            return Ok((vec![w], Packing::Narrow { int, float }));
        }
        if t == self.index_t && self.isa.index_bits() == 32 {
            let w = self.push(block, arith::index_cast(v, self.i32_t, loc))?;
            return Ok((vec![w], Packing::Index32));
        }
        let wide = if t == self.index_t {
            self.push(block, arith::index_cast(v, self.i64_t, loc))?
        } else if t == self.f64_t {
            self.push(block, arith::bitcast(v, self.i64_t, loc))?
        } else {
            v
        };
        let lo = self.push(block, arith::trunci(wide, self.i32_t, loc))?;
        let k32 = self.push(
            block,
            arith::constant(self.ctx, IntegerAttribute::new(self.i64_t, 32).into(), loc),
        )?;
        let hi = self.push(block, arith::shrui(wide, k32, loc))?;
        let hi = self.push(block, arith::trunci(hi, self.i32_t, loc))?;
        Ok((vec![lo, hi], Packing::Wide(t)))
    }

    fn unpack(&self, block: &Block<'c>, words: &[Value<'c, 'c>], packing: Packing<'c>) -> Result<Value<'c, 'c>> {
        let loc = self.loc;
        Ok(match packing {
            Packing::Word => words[0],
            Packing::Float(t) => self.push(block, arith::bitcast(words[0], t, loc))?,
            Packing::Index32 => self.push(block, arith::index_cast(words[0], self.index_t, loc))?,
            Packing::Narrow { int, float } => {
                let bits = self.push(block, arith::trunci(words[0], int, loc))?;
                match float {
                    Some(f) => self.push(block, arith::bitcast(bits, f, loc))?,
                    None => bits,
                }
            }
            Packing::Wide(t) => {
                let lo = self.push(block, arith::extui(words[0], self.i64_t, loc))?;
                let hi = self.push(block, arith::extui(words[1], self.i64_t, loc))?;
                let k32 = self.push(
                    block,
                    arith::constant(self.ctx, IntegerAttribute::new(self.i64_t, 32).into(), loc),
                )?;
                let hi = self.push(block, arith::shli(hi, k32, loc))?;
                let wide = self.push(block, arith::ori(lo, hi, loc))?;
                if t == self.index_t {
                    self.push(block, arith::index_cast(wide, t, loc))?
                } else if t == self.f64_t {
                    self.push(block, arith::bitcast(wide, t, loc))?
                } else {
                    wide
                }
            }
        })
    }
}
