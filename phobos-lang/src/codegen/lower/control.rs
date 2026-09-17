use anyhow::{Result, anyhow, bail};
use melior::{
    dialect::{arith, scf},
    ir::{
        Attribute, Block, BlockLike, Region, RegionLike, Type, ValueLike,
        attribute::IntegerAttribute, operation::OperationBuilder,
    },
};

use super::Lowered;
use crate::codegen::matmul::GemmSource;
use crate::codegen::{Codegen, FragAcc, detach};
use crate::ir::{self, Bounds, ForInfo, Ir, OpId, OpKind};

impl<'c> Codegen<'c> {
    pub(super) fn emit_control_op(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        match ir.kind(op).clone() {
            OpKind::For(info) => self.emit_for_ir(block, ir, op, info),
            OpKind::While => self.emit_while_ir(block, ir, op),
            OpKind::If => self.emit_if_ir(block, ir, op),
            other => bail!("{} is not a control op", other.name()),
        }
    }

    /// The hoisted frame is pushed around the loop; the planner keeps its
    /// buffers alive through it, as the loop's operands.
    fn emit_for_ir(&mut self, block: &Block<'c>, ir: &Ir, op: OpId, info: ForInfo) -> Result<()> {
        let first_hoisted = info.bound_operands() + info.carried;
        let mut frame = Vec::new();
        for &buf in &ir.operands(op)[first_hoisted..] {
            let stage = ir
                .def_op(buf)
                .ok_or_else(|| anyhow!("a hoisted operand that no op made"))?;
            let src = self.tile_ir(ir, ir.operand(stage, 0))?;
            frame.push((src.mem, self.tile_ir(ir, buf)?));
        }
        self.hoisted_stages.push(frame);
        let result = self.emit_for_inner_ir(block, ir, op, info);
        self.hoisted_stages.pop();
        result
    }

    fn emit_for_inner_ir(&mut self, block: &Block<'c>, ir: &Ir, op: OpId, info: ForInfo) -> Result<()> {
        if info.carried > 0 {
            let first = ir.operand(op, info.bound_operands());
            if matches!(ir.ty(first), ir::Type::Gemm(_)) {
                return self.emit_gemm_for_ir(block, ir, op);
            }
            return self.emit_frag_for_ir(block, ir, op, info);
        }
        if let Some(pipeline) = info.pipeline {
            self.pipelined_any = true;
            return self.emit_pipelined_for_ir(block, ir, op, pipeline);
        }
        let body = ir.blocks_of(op)[0];
        match info.bounds {
            Bounds::Affine { lo, hi, step } => {
                let body_block = Block::new(&[(self.index_t, self.loc)]);
                let iv = detach(body_block.argument(0)?.into());
                self.set_ir(ir.args(body)[0], Lowered::Scalar(iv));
                self.emit_ops(&body_block, ir, ir.ops(body))?;
                body_block.append_operation(OperationBuilder::new("affine.yield", self.loc).build()?);

                let region = Region::new();
                region.append_block(body_block);
                let bound_map = |v: i64| {
                    Attribute::parse(self.ctx, &format!("affine_map<() -> ({v})>"))
                        .ok_or_else(|| anyhow!("failed to parse affine bound map for {v}"))
                };
                let op = OperationBuilder::new("affine.for", self.loc)
                    .add_attributes(&[
                        (self.id("lowerBoundMap"), bound_map(lo)?),
                        (self.id("upperBoundMap"), bound_map(hi)?),
                        (
                            self.id("step"),
                            IntegerAttribute::new(self.index_t, step).into(),
                        ),
                        (self.id("operandSegmentSizes"), self.i32_array(&[0, 0, 0])?),
                    ])
                    .add_regions([region])
                    .build()?;
                block.append_operation(op);
            }
            Bounds::Dynamic => {
                let lo = self.scalar_ir(ir.operand(op, 0))?;
                let hi = self.scalar_ir(ir.operand(op, 1))?;
                let st = self.scalar_ir(ir.operand(op, 2))?;
                let body_block = Block::new(&[(self.index_t, self.loc)]);
                let iv = detach(body_block.argument(0)?.into());
                self.set_ir(ir.args(body)[0], Lowered::Scalar(iv));
                self.emit_ops(&body_block, ir, ir.ops(body))?;
                body_block.append_operation(scf::r#yield(&[], self.loc));
                let region = Region::new();
                region.append_block(body_block);
                block.append_operation(scf::r#for(lo, hi, st, region, self.loc));

                // The masked replay of the last chunk; see `emit_split_for`.
                if info.ragged {
                    let full = hi;
                    let end = self.scalar_ir(ir.operand(op, 3))?;
                    let ragged = self.push(
                        block,
                        arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, full, end, self.loc),
                    )?;
                    let then_block = Block::new(&[]);
                    self.emit_ops(&then_block, ir, ir.ops(ir.blocks_of(op)[1]))?;
                    then_block.append_operation(scf::r#yield(&[], self.loc));
                    let region = Region::new();
                    region.append_block(then_block);
                    block.append_operation(scf::r#if(ragged, &[], region, Region::new(), self.loc));
                }
            }
        }
        Ok(())
    }

    /// The fragment accumulators ride the loop as iter args, one MLIR
    /// value per fragment.
    fn emit_frag_for_ir(&mut self, block: &Block<'c>, ir: &Ir, op: OpId, info: ForInfo) -> Result<()> {
        let lo = self.scalar_ir(ir.operand(op, 0))?;
        let hi = self.scalar_ir(ir.operand(op, 1))?;
        let st = self.scalar_ir(ir.operand(op, 2))?;
        let body = ir.blocks_of(op)[0];
        let carried = &ir.operands(op)[3..3 + info.carried];
        let accs: Vec<FragAcc<'c>> = carried
            .iter()
            .map(|&v| self.frags_ir(v))
            .collect::<Result<_>>()?;
        let mut inits = Vec::new();
        for fa in &accs {
            inits.extend(fa.frags.iter().copied());
        }

        let mut args = vec![(self.index_t, self.loc)];
        args.extend(inits.iter().map(|v| (v.r#type(), self.loc)));
        let body_block = Block::new(&args);
        let iv = detach(body_block.argument(0)?.into());
        self.set_ir(ir.args(body)[0], Lowered::Scalar(iv));
        let mut off = 1;
        for (k, fa) in accs.iter().enumerate() {
            let mut frags = Vec::with_capacity(fa.frags.len());
            for j in 0..fa.frags.len() {
                frags.push(detach(body_block.argument(off + j)?.into()));
            }
            off += fa.frags.len();
            self.set_ir(
                ir.args(body)[1 + k],
                Lowered::Frags(FragAcc {
                    frags,
                    ..fa.clone()
                }),
            );
        }

        self.emit_ops(&body_block, ir, ir.ops(body))?;
        let mut finals = Vec::with_capacity(inits.len());
        for v in self.yielded(ir, body)? {
            finals.extend(self.frags_ir(v)?.frags);
        }
        body_block.append_operation(scf::r#yield(&finals, self.loc));

        let region = Region::new();
        region.append_block(body_block);
        let types: Vec<Type<'c>> = inits.iter().map(|v| v.r#type()).collect();
        let mut operands = vec![lo, hi, st];
        operands.extend_from_slice(&inits);
        let loop_op = block.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&operands)
                .add_results(&types)
                .add_regions([region])
                .build()?,
        );

        let mut off = 0;
        for (k, fa) in accs.iter().enumerate() {
            let mut frags = Vec::with_capacity(fa.frags.len());
            for j in 0..fa.frags.len() {
                frags.push(detach(loop_op.result(off + j)?.into()));
            }
            off += fa.frags.len();
            self.set_ir(
                ir.results(op)[k],
                Lowered::Frags(FragAcc {
                    frags,
                    ..fa.clone()
                }),
            );
        }
        Ok(())
    }

    fn emit_while_ir(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        let (before_ir, after_ir) = (ir.blocks_of(op)[0], ir.blocks_of(op)[1]);
        let before = Block::new(&[]);
        self.emit_ops(&before, ir, ir.ops(before_ir))?;
        let c = self.scalar_ir(self.yielded(ir, before_ir)?[0])?;
        before.append_operation(scf::condition(c, &[], self.loc));

        let after = Block::new(&[]);
        self.emit_ops(&after, ir, ir.ops(after_ir))?;
        after.append_operation(scf::r#yield(&[], self.loc));

        let before_region = Region::new();
        before_region.append_block(before);
        let after_region = Region::new();
        after_region.append_block(after);
        block.append_operation(scf::r#while(
            &[],
            &[],
            before_region,
            after_region,
            self.loc,
        ));
        Ok(())
    }

    fn emit_if_ir(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        let c = self.scalar_ir(ir.operand(op, 0))?;
        let blocks = ir.blocks_of(op);
        let then_block = Block::new(&[]);
        self.emit_ops(&then_block, ir, ir.ops(blocks[0]))?;
        then_block.append_operation(scf::r#yield(&[], self.loc));
        let then_region = Region::new();
        then_region.append_block(then_block);

        let else_region = Region::new();
        if let Some(&els) = blocks.get(1) {
            let else_block = Block::new(&[]);
            self.emit_ops(&else_block, ir, ir.ops(els))?;
            else_block.append_operation(scf::r#yield(&[], self.loc));
            else_region.append_block(else_block);
        }
        block.append_operation(scf::r#if(c, &[], then_region, else_region, self.loc));
        Ok(())
    }

    /// The register matmul's k-loop: the seeded accumulator rides the loop
    /// the emitter builds itself, staging each iteration's operands by
    /// lowering the body's slices afresh, and comes out as the loop's
    /// result. See `matmul::gemm`.
    fn emit_gemm_for_ir(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        let lo = self.scalar_ir(ir.operand(op, 0))?;
        let hi = self.scalar_ir(ir.operand(op, 1))?;
        let st = self.scalar_ir(ir.operand(op, 2))?;
        let acc = self.gemm_ir(ir.operand(op, 3))?;
        let body = ir.blocks_of(op)[0];
        let ops = ir.ops(body);
        let Some((&yield_op, rest)) = ops.split_last() else {
            bail!("a gemm loop with an empty body");
        };
        let Some((&dot, prefix)) = rest.split_last() else {
            bail!("a gemm loop whose body has no gemm_dot");
        };
        if !matches!(ir.kind(dot), OpKind::GemmDot) || !matches!(ir.kind(yield_op), OpKind::Yield) {
            bail!("a gemm loop's body must end in gemm_dot and yield");
        }
        let src = GemmSource::Graph {
            ir,
            iv: ir.args(body)[0],
            prefix,
            a: ir.operand(dot, 1),
            b: ir.operand(dot, 2),
        };
        // The fused-GEMM backend doubles its buffers whenever the attribute
        // is on, so this really did pipeline.
        if self.pipeline_assert {
            self.pipelined_any = true;
        }
        let acc = self.gemm_loop(block, acc, (lo, hi, st), &src)?;
        self.set_ir(ir.results(op)[0], Lowered::Gemm(acc));
        Ok(())
    }
}
