// A synchronous stage whose readers all sit past an explicit `barrier()` in
// its block loads its chunks into registers where it is written, and stores
// them into its buffer just before that barrier. The loads' latency then
// hides behind whatever runs in between, a weight's decode in the staged
// projections, without holding the chunks across a whole loop turn as the
// pipeline's register prefetch does.

use anyhow::Result;
use melior::ir::{Block, Value};

use super::pipeline::{RegCopy, fenced_before_reads};
use crate::codegen::{Codegen, MemVal};
use crate::ir::{BlockId, Ir, OpId, OpKind, Space, Type};

/// A stage's chunks, loaded and waiting for the barrier in `block`.
pub(in crate::codegen) struct Deferred<'c> {
    block: BlockId,
    dst: MemVal<'c>,
    plan: RegCopy<'c>,
    vals: Vec<Value<'c, 'c>>,
}

impl<'c> Codegen<'c> {
    /// Loads the stage `op` copies from `src` into registers, to be stored
    /// into `dst` at the barrier ahead of its readers. False, with nothing
    /// emitted, when the stage does not qualify.
    pub(super) fn defer_stage(
        &mut self,
        block: &Block<'c>,
        ir: &Ir,
        op: OpId,
        src: &MemVal<'c>,
        dst: &MemVal<'c>,
    ) -> Result<bool> {
        if !defers(ir, op) {
            return Ok(false);
        }
        let Some(plan) = self.reg_copy_plan(src, dst) else {
            return Ok(false);
        };
        let plan = self.reg_copy_indexed(block, plan)?;
        let vals = self.reg_copy_load(block, src, &plan)?;
        self.deferred.push(Deferred {
            block: ir.parent_block(op),
            dst: dst.clone(),
            plan,
            vals,
        });
        Ok(true)
    }

    /// Stores the chunks waiting for the barrier `op`, ahead of it.
    pub(super) fn flush_deferred(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        let at = ir.parent_block(op);
        let (now, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.deferred).into_iter().partition(|d| d.block == at);
        self.deferred = later;
        for d in now {
            // Located again here rather than held across the work between.
            let plan = self.reg_copy_indexed(block, d.plan)?;
            self.reg_copy_store(block, &d.dst, &plan, &d.vals)?;
        }
        Ok(())
    }
}

/// Whether the stage `op` can be stored late: a `barrier()` in its block
/// comes before anything reads it, and nothing before that barrier writes
/// global memory, which the stage's source might be.
fn defers(ir: &Ir, op: OpId) -> bool {
    let ops = ir.ops(ir.parent_block(op));
    let rest = &ops[ir.position(op) + 1..];
    let Some(fence) = rest.iter().position(|&o| matches!(ir.kind(o), OpKind::Barrier)) else {
        return false;
    };
    fenced_before_reads(ir, rest, vec![ir.result(op)]) && !rest[..fence].iter().any(|&o| writes_global(ir, o))
}

/// Whether `op`, or an op nested in it, writes a tensor or a view of one.
pub(super) fn writes_global(ir: &Ir, op: OpId) -> bool {
    let operands = ir.operands(op);
    let writes = ir.kind(op).writes(operands.len()).into_iter().any(|i| match ir.ty(operands[i]) {
        Type::Tensor(_) => true,
        Type::Tile(t) => t.space == Space::Global,
        _ => false,
    });
    writes || ir.blocks_of(op).iter().any(|&b| ir.ops(b).iter().any(|&o| writes_global(ir, o)))
}
