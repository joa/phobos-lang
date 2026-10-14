// A generated tile read where a consumer needs it: the `Generate` op's block
// lowered once per element block, with the row and unit bound to the
// consumer's own indices.

use anyhow::{Result, bail};
use melior::ir::{Block, Type, Value};

use super::Lowered;
use crate::codegen::Codegen;
use crate::codegen::frag::{FragIn, ImmaOperand};
use crate::ir::{Ir, OpId, ValueId};

impl<'c> Codegen<'c> {
    /// An int8 accumulator update's operand `v`: generated, or a tile.
    pub(super) fn frag_in<'a>(&self, ir: &'a Ir, v: ValueId) -> Result<FragIn<'a, 'c>> {
        if let Lowered::Generated(op) = self.get_ir(v)? {
            let crate::ir::Type::Dist(d) = ir.ty(v) else {
                bail!("internal: a generated operand that is not a generated tile");
            };
            return Ok(FragIn::Generated {
                ir,
                op: *op,
                shape: d.shape.clone(),
                elem: d.elem,
            });
        }
        let view = self.tile_ir(ir, v)?;
        let from = self.slice_origin(ir, v)?;
        Ok(FragIn::Stored(ImmaOperand { view, from }))
    }

    /// Releases the tiles among `ins`.
    pub(super) fn release_frag_ins(&mut self, ins: &[FragIn<'_, 'c>]) {
        for v in ins {
            if let FragIn::Stored(v) = v {
                self.release(&v.view);
            }
        }
    }

    /// The value of `op`'s block at `(row, unit)`.
    pub(in crate::codegen) fn eval_generate(
        &mut self,
        block: &Block<'c>,
        ir: &Ir,
        op: OpId,
        row: Value<'c, 'c>,
        unit: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let [region] = ir.blocks_of(op)[..] else {
            bail!("internal: a generate without its block");
        };
        let [row_arg, unit_arg] = ir.args(region)[..] else {
            bail!("internal: a generate block without a row and a unit");
        };
        self.set_ir(row_arg, Lowered::Scalar(row));
        self.set_ir(unit_arg, Lowered::Scalar(unit));
        let mut value = None;
        for &inner in ir.ops(region) {
            if ir.kind(inner).is_terminator() {
                value = ir.operands(inner).first().copied();
            } else {
                self.emit_op_inner(block, ir, inner)?;
            }
        }
        let Some(value) = value else {
            bail!("internal: a generate block yields nothing");
        };
        self.scalar_ir(value)
    }

    /// The `m8n8k16` B fragments of a generated int8 weight, tile-major then
    /// by k step as `imma_frags` lays them out: for row `rows[t]` and k
    /// column `k_cols[h]`, the word of bytes `k_cols[h] .. + 3`.
    pub(in crate::codegen) fn generated_frags(
        &mut self,
        block: &Block<'c>,
        ir: &Ir,
        op: OpId,
        rows: &[Value<'c, 'c>],
        k_cols: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let four = self.const_index(block, 4)?;
        let frag_t = Type::vector(&[1, 4], self.i8_t);
        let mut frags = Vec::with_capacity(rows.len() * k_cols.len());
        for &row in rows {
            for &col in k_cols {
                let unit = self.divui(block, col, four)?;
                let word = self.eval_generate(block, ir, op, row, unit)?;
                let word = self.numeric_cast(block, word, self.i32_t)?;
                let word = self.vec_broadcast(block, word, Type::vector(&[1], self.i32_t))?;
                let bytes = self.vec_bitcast(block, word, Type::vector(&[4], self.i8_t))?;
                frags.push(self.vec_shape_cast(block, bytes, frag_t)?);
            }
        }
        Ok(frags)
    }
}
