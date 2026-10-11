use anyhow::{Result, bail};
use melior::{
    dialect::{arith, memref},
    ir::{
        Block, BlockLike, ValueLike, attribute::FloatAttribute, operation::OperationBuilder,
        r#type::MemRefType,
    },
};

use super::Lowered;
use crate::ast::UnOp;
use crate::codegen::Codegen;
use crate::ir::{self, IndexOp, Ir, Literal, OpId, OpKind};

impl<'c> Codegen<'c> {
    pub(super) fn emit_scalar_op(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        let kind = ir.kind(op).clone();
        let value = match kind {
            OpKind::Const(Literal::Int(n)) => self.const_index(block, n)?,
            OpKind::Const(Literal::Float(v)) => self.push(
                block,
                arith::constant(
                    self.ctx,
                    FloatAttribute::new(self.ctx, self.f32_t, v).into(),
                    self.loc,
                ),
            )?,
            OpKind::Const(Literal::Bool(b)) => self.const_bool(block, b)?,
            OpKind::Unary(un) => {
                let v = self.scalar_ir(ir.operand(op, 0))?;
                let t = v.r#type();
                match un {
                    UnOp::Neg if self.is_float(t) => self.push(block, arith::negf(v, self.loc))?,
                    UnOp::Neg => {
                        let zero = self.const_index(block, 0)?;
                        self.subi(block, zero, v)?
                    }
                    UnOp::Not => {
                        let one = self.const_bool(block, true)?;
                        self.push(block, arith::xori(v, one, self.loc))?
                    }
                    UnOp::BitNot => {
                        let ones = self.const_index(block, -1)?;
                        self.push(block, arith::xori(v, ones, self.loc))?
                    }
                }
            }
            OpKind::Binary(bin) => {
                let a = self.scalar_ir(ir.operand(op, 0))?;
                let b = self.scalar_ir(ir.operand(op, 1))?;
                self.emit_binop(block, bin, a, b)?
            }
            OpKind::IndexOp(iop) => {
                let a = self.scalar_ir(ir.operand(op, 0))?;
                let b = self.scalar_ir(ir.operand(op, 1))?;
                match iop {
                    IndexOp::DivU => self.divui(block, a, b)?,
                    IndexOp::RemU => self.remui(block, a, b)?,
                    IndexOp::CmpUlt => self.push(
                        block,
                        arith::cmpi(self.ctx, arith::CmpiPredicate::Ult, a, b, self.loc),
                    )?,
                }
            }
            OpKind::Cast(want) => {
                let v = self.scalar_ir(ir.operand(op, 0))?;
                let want = self.ir_scalar_type(want);
                if Self::ir_unsigned(ir, ir.operand(op, 0)) {
                    self.unsigned_cast(block, v, want)?
                } else {
                    self.numeric_cast(block, v, want)?
                }
            }
            OpKind::IndexCast(want) => {
                let v = self.scalar_ir(ir.operand(op, 0))?;
                let want = self.ir_scalar_type(want);
                if Self::ir_unsigned(ir, ir.operand(op, 0)) {
                    self.push(
                        block,
                        OperationBuilder::new("arith.index_castui", self.loc)
                            .add_operands(&[v])
                            .add_results(&[want])
                            .build()?,
                    )?
                } else {
                    self.push(block, arith::index_cast(v, want, self.loc))?
                }
            }
            OpKind::Math(step) => {
                let v = self.scalar_ir(ir.operand(op, 0))?;
                let step = self.ir_elem_step(step);
                let r = self.apply_elem_step(block, v, step)?;
                let ir::Type::Scalar(want) = ir.ty(ir.result(op)) else {
                    bail!("math to a non-scalar");
                };
                let want = self.ir_scalar_type(*want);
                self.numeric_cast(block, r, want)?
            }
            OpKind::Select => {
                let c = self.scalar_ir(ir.operand(op, 0))?;
                let a = self.scalar_ir(ir.operand(op, 1))?;
                let b = self.scalar_ir(ir.operand(op, 2))?;
                self.select(block, c, a, b)?
            }
            OpKind::ProgramId(d) => self.block_id(block, ["x", "y", "z"][d as usize])?,
            OpKind::Dim(d) => {
                let mem = self.tile_ir(ir, ir.operand(op, 0))?.mem;
                let pos = self.const_index(block, d as i64)?;
                self.push(block, memref::dim(mem, pos, self.loc))?
            }
            OpKind::Load => {
                let mem = ir.operand(op, 0);
                let indices = self.indices_ir(ir, op, 1)?;
                match ir.ty(mem) {
                    ir::Type::Tile(t) if t.space == ir::Space::Private => {
                        let slot = self.slot_ir(mem)?;
                        self.push(block, memref::load(slot, &[], self.loc))?
                    }
                    _ => {
                        let mv = self.tile_ir(ir, mem)?;
                        self.push(block, memref::load(mv.mem, &indices, self.loc))?
                    }
                }
            }
            OpKind::Store => {
                let v = self.scalar_ir(ir.operand(op, 0))?;
                let mem = ir.operand(op, 1);
                let indices = self.indices_ir(ir, op, 2)?;
                match ir.ty(mem) {
                    ir::Type::Tile(t) if t.space == ir::Space::Private => {
                        let slot = self.slot_ir(mem)?;
                        block.append_operation(memref::store(v, slot, &[], self.loc));
                    }
                    _ => {
                        let mv = self.tile_ir(ir, mem)?;
                        block.append_operation(memref::store(v, mv.mem, &indices, self.loc));
                    }
                }
                return Ok(());
            }
            OpKind::AtomicAdd => {
                let mv = self.tile_ir(ir, ir.operand(op, 0))?;
                let idx = self.scalar_ir(ir.operand(op, 1))?;
                let val = self.scalar_ir(ir.operand(op, 2))?;
                let rank = mv.shape.len();
                self.cta_atomic(block, val.r#type(), |cg, blk| {
                    cg.atomic_add_raw(blk, mv.mem, rank, idx, val)
                })?
            }
            OpKind::AtomicCas => {
                let mv = self.tile_ir(ir, ir.operand(op, 0))?;
                let idx = self.scalar_ir(ir.operand(op, 1))?;
                let expected = self.scalar_ir(ir.operand(op, 2))?;
                let val = self.scalar_ir(ir.operand(op, 3))?;
                let rank = mv.shape.len();
                self.cta_atomic(block, val.r#type(), |cg, blk| {
                    cg.atomic_cas_raw(blk, mv.mem, rank, idx, expected, val)
                })?
            }
            OpKind::Barrier => {
                self.barrier(block)?;
                return Ok(());
            }
            OpKind::Trap => {
                block.append_operation(OperationBuilder::new("llvm.intr.trap", self.loc).build()?);
                return Ok(());
            }
            OpKind::LoadAcquire | OpKind::StoreRelease => {
                let mv = self.tile_ir(ir, ir.operand(op, 0))?;
                let idx = self.scalar_ir(ir.operand(op, 1))?;
                let mut at = vec![idx];
                for _ in 1..mv.shape.len() {
                    at.push(self.const_index(block, 0)?);
                }
                if matches!(ir.kind(op), OpKind::StoreRelease) {
                    let v = self.scalar_ir(ir.operand(op, 2))?;
                    self.barrier(block)?;
                    self.leader(block, |cg, blk| cg.store_release_sys(blk, &mv, &at, v))?;
                    self.const_index(block, 0)?
                } else {
                    let v = self.cta_atomic(block, self.i32_t, |cg, blk| cg.load_acquire_sys(blk, &mv, &at))?;
                    self.fence_sys(block)?;
                    v
                }
            }
            OpKind::GridBarrier => {
                let mv = self.tile_ir(ir, ir.operand(op, 0))?;
                self.grid_barrier_raw(block, mv.mem, mv.shape.len())?;
                self.const_index(block, 0)?
            }
            OpKind::Alloc => {
                // The private slot of a `var` scalar. Shared buffers are tile
                // ops.
                let ir::Type::Tile(t) = ir.ty(ir.result(op)) else {
                    bail!("alloc of a non-tile");
                };
                let slot_t = MemRefType::new(self.ir_scalar_type(t.elem), &[], None, None);
                let slot = self.push(
                    block,
                    memref::alloca(self.ctx, slot_t, &[], &[], None, self.loc),
                )?;
                self.set_ir(ir.result(op), Lowered::Slot(slot));
                return Ok(());
            }
            other => bail!("{} is not a scalar op", other.name()),
        };
        self.set_ir(ir.result(op), Lowered::Scalar(value));
        Ok(())
    }

    /// The index operands of `op` from `from` on.
    fn indices_ir(&self, ir: &Ir, op: OpId, from: usize) -> Result<Vec<melior::ir::Value<'c, 'c>>> {
        ir.operands(op)[from..]
            .iter()
            .map(|&v| self.scalar_ir(v))
            .collect()
    }
}
