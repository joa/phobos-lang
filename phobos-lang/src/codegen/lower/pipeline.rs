use std::collections::HashMap;

use anyhow::{Result, bail};
use melior::{
    dialect::{arith, scf},
    ir::{Block, BlockLike, Region, RegionLike},
};

use super::Lowered;
use crate::codegen::{Codegen, MemVal, detach};
use crate::ir::{Ir, OpId, OpKind, Pipeline, ValueId};

impl<'c> Codegen<'c> {
    pub(super) fn emit_pipelined_for_ir(
        &mut self,
        block: &Block<'c>,
        ir: &Ir,
        op: OpId,
        info: Pipeline,
    ) -> Result<()> {
        let lo = self.scalar_ir(ir.operand(op, 0))?;
        let hi = self.scalar_ir(ir.operand(op, 1))?;
        let st = self.scalar_ir(ir.operand(op, 2))?;
        let body = ir.blocks_of(op)[0];
        let iv = ir.args(body)[0];
        let ops = ir.ops(body);

        // The prefix ends at the last of the staged prefix's stage ops.
        let mut stages = Vec::with_capacity(info.staged);
        for &o in ops {
            if matches!(ir.kind(o), OpKind::Stage(_)) {
                stages.push(o);
                if stages.len() == info.staged {
                    break;
                }
            }
        }
        if stages.len() != info.staged {
            bail!("a pipelined loop with fewer stage ops than its prefix");
        }
        let end = ops.iter().position(|o| *o == stages[info.staged - 1]).expect("found above") + 1;
        let (prefix, rest) = ops.split_at(end);

        // Prologue: stage iteration 0 into each pair's first buffer.
        let mut bufs0 = Vec::with_capacity(info.staged);
        let mut bufs1 = Vec::with_capacity(info.staged);
        self.lowered.push(HashMap::new());
        self.set_ir(iv, Lowered::Scalar(lo));
        let prologue = (|| {
            for &o in prefix {
                if matches!(ir.kind(o), OpKind::Stage(_)) {
                    let src = self.tile_ir(ir, ir.operand(o, 0))?;
                    let b0 = self.alloc_tile_shaped(block, src.elem, &src.shape)?;
                    bufs1.push(self.alloc_tile_shaped(block, src.elem, &src.shape)?);
                    self.tile_copy(block, &src, &b0, true, false)?;
                    bufs0.push(b0);
                } else {
                    self.emit_op(block, ir, o)?;
                }
            }
            Ok::<(), anyhow::Error>(())
        })();
        self.lowered.pop();
        prologue?;

        let body_block = Block::new(&[(self.index_t, self.loc)]);
        let mlir_iv = detach(body_block.argument(0)?.into());
        let next = self.addi(&body_block, mlir_iv, st)?;

        let half = Half {
            ir,
            iv,
            prefix,
            rest,
            stages: &stages,
            hi,
            ends_with_tile_op: info.ends_with_tile_op,
        };
        // Half A: compute iteration iv from bufs0, prefetch iv+st -> bufs1.
        self.emit_pipeline_stage_ir(&body_block, &half, mlir_iv, next, &bufs0, &bufs1)?;

        // Half B (when iteration iv+st exists): compute it from bufs1,
        // prefetch iv+2*st -> bufs0.
        let have_b = self.push(
            &body_block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Slt, next, hi, self.loc),
        )?;
        let half_b = Block::new(&[]);
        let next2 = self.addi(&half_b, next, st)?;
        self.emit_pipeline_stage_ir(&half_b, &half, next, next2, &bufs1, &bufs0)?;
        half_b.append_operation(scf::r#yield(&[], self.loc));
        let half_b_region = Region::new();
        half_b_region.append_block(half_b);
        body_block.append_operation(scf::r#if(
            have_b,
            &[],
            half_b_region,
            Region::new(),
            self.loc,
        ));
        body_block.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body_block);
        let two = self.const_index(block, 2)?;
        let st2 = self.muli(block, st, two)?;
        block.append_operation(scf::r#for(lo, hi, st2, region, self.loc));
        Ok(())
    }

    /// One unrolled half: a guarded prefetch (no barrier) of `prefetch_iv`
    /// into `dst`, compute of `compute_iv` from `cur`, one closing barrier.
    fn emit_pipeline_stage_ir(
        &mut self,
        block: &Block<'c>,
        half: &Half<'c, '_>,
        compute_iv: melior::ir::Value<'c, 'c>,
        prefetch_iv: melior::ir::Value<'c, 'c>,
        cur: &[MemVal<'c>],
        dst: &[MemVal<'c>],
    ) -> Result<()> {
        let ir = half.ir;
        let use_async = self.has_cp_async();
        self.guarded_prefetch(block, prefetch_iv, half.hi, |cg, then| {
            cg.lowered.push(HashMap::new());
            cg.set_ir(half.iv, Lowered::Scalar(prefetch_iv));
            let mut d = dst.iter();
            let out = (|| {
                for &o in half.prefix {
                    if matches!(ir.kind(o), OpKind::Stage(_)) {
                        let src = cg.tile_ir(ir, ir.operand(o, 0))?;
                        let buf = d.next().expect("one buffer per stage");
                        cg.tile_copy(then, &src, buf, false, use_async)?;
                    } else {
                        cg.emit_op(then, ir, o)?;
                    }
                }
                Ok::<(), anyhow::Error>(())
            })();
            cg.lowered.pop();
            out
        })?;

        // cp.async: commit everything this thread issued in the prefetch into
        // one group.
        let group = if use_async {
            Some(self.async_create_group(block)?)
        } else {
            None
        };

        // Compute against cur.
        self.lowered.push(HashMap::new());
        self.set_ir(half.iv, Lowered::Scalar(compute_iv));
        for (&stage, c) in half.stages.iter().zip(cur) {
            self.set_ir(ir.result(stage), Lowered::Tile(c.clone()));
        }
        let compute = self.emit_ops(block, ir, half.rest);
        self.lowered.pop();
        compute?;

        if let Some(group) = group {
            self.async_wait(block, group)?;
            self.barrier(block)?;
            return Ok(());
        }
        if !half.ends_with_tile_op {
            self.barrier(block)?;
        }
        Ok(())
    }
}

/// What both halves of a pipelined loop share.
struct Half<'c, 'o> {
    ir: &'o Ir,
    iv: ValueId,
    prefix: &'o [OpId],
    rest: &'o [OpId],
    stages: &'o [OpId],
    hi: melior::ir::Value<'c, 'c>,
    ends_with_tile_op: bool,
}
