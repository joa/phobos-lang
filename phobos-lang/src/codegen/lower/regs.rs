// Register tiles: each thread's share of a tile as values. Element `e` of
// thread `t` sits at row-major position `t + e * threads`, the mapping a
// plain sweep uses, so reading a shared tile at the same element needs no
// exchange between threads.

use anyhow::{Result, bail};
use melior::dialect::memref;
use melior::ir::{Block, BlockLike, Type, Value};

use super::Lowered;
use crate::codegen::{Codegen, FragAcc};
use crate::ir::{self, Ir, OpId, OpKind, ValueId};

#[derive(Clone)]
pub(in crate::codegen) struct RegTile<'c> {
    pub(in crate::codegen) vals: Vec<Value<'c, 'c>>,
    pub(in crate::codegen) elem: Type<'c>,
    pub(in crate::codegen) shape: Vec<i64>,
}

impl<'c> Codegen<'c> {
    pub(super) fn regs_ir(&self, v: ValueId) -> Result<RegTile<'c>> {
        match self.get_ir(v)? {
            Lowered::Regs(r) => Ok(r.clone()),
            _ => bail!("{v} is not a register tile"),
        }
    }

    /// The registers a loop carries for `v`, a fragment accumulator or a
    /// register tile.
    pub(super) fn carried_vals(&self, v: ValueId) -> Result<Vec<Value<'c, 'c>>> {
        match self.get_ir(v)? {
            Lowered::Frags(fa) => Ok(fa.frags.clone()),
            Lowered::Regs(r) => Ok(r.vals.clone()),
            _ => bail!("{v} is not carried in registers"),
        }
    }

    /// `like` with its registers replaced by `vals`.
    pub(super) fn with_vals(&self, like: &Lowered<'c>, vals: Vec<Value<'c, 'c>>) -> Result<Lowered<'c>> {
        Ok(match like {
            Lowered::Frags(fa) => Lowered::Frags(FragAcc { frags: vals, ..fa.clone() }),
            Lowered::Regs(r) => Lowered::Regs(RegTile { vals, ..r.clone() }),
            _ => bail!("not carried in registers"),
        })
    }

    pub(super) fn emit_regs_op(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        match ir.kind(op) {
            OpKind::RegsFill => {
                let ty = self.regs_type(ir, ir.result(op))?;
                let v = self.scalar_ir(ir.operand(op, 0))?;
                let v = self.coerce(block, v, ty.elem)?;
                let count = self.regs_per_thread(&ty.shape);
                let vals = vec![v; count];
                self.set_ir(ir.result(op), Lowered::Regs(RegTile { vals, ..ty }));
            }
            OpKind::RegsFused(tree) => {
                let ty = self.regs_type(ir, ir.result(op))?;
                let vec_t = Type::vector(&[4], ty.elem);
                let mut leaves = Vec::new();
                let mut vals = Vec::new();
                for e in 0..self.regs_per_thread(&ty.shape) {
                    let idx = self.reg_index(block, &ty.shape, e)?;
                    // Tile leaves are the same for every element; keep one copy.
                    let mut seen = Vec::new();
                    let fused = self.fused_from_tree(ir, op, tree, ir.result(op), &mut seen, Some(e))?;
                    if e == 0 {
                        leaves = seen;
                    }
                    let v = self.eval_fused(block, &fused, &idx, &ty.shape, 1, vec_t)?;
                    vals.push(self.coerce(block, v, ty.elem)?);
                }
                for leaf in &leaves {
                    self.release(leaf);
                }
                self.set_ir(ir.result(op), Lowered::Regs(RegTile { vals, ..ty }));
            }
            OpKind::RegsStore => {
                let regs = self.regs_ir(ir.operand(op, 0))?;
                let dst = self.tile_ir(ir, ir.operand(op, 1))?;
                for (e, &v) in regs.vals.iter().enumerate() {
                    let idx = self.reg_index(block, &regs.shape, e)?;
                    let v = self.coerce(block, v, dst.elem)?;
                    block.append_operation(memref::store(v, dst.mem, &idx, self.loc));
                }
                self.barrier(block)?;
            }
            other => bail!("{} is not a register-tile op", other.name()),
        }
        Ok(())
    }

    /// The register tile `v` is typed as, with no values yet.
    fn regs_type(&self, ir: &Ir, v: ValueId) -> Result<RegTile<'c>> {
        let ir::Type::Regs(r) = ir.ty(v) else {
            bail!("{v} is not typed as a register tile");
        };
        Ok(RegTile {
            vals: Vec::new(),
            elem: self.ir_scalar_type(r.elem),
            shape: r.shape.clone(),
        })
    }

    fn regs_per_thread(&self, shape: &[i64]) -> usize {
        (shape.iter().product::<i64>() / self.cta_threads) as usize
    }

    /// The coordinates of this thread's element `e`.
    fn reg_index(&self, block: &Block<'c>, shape: &[i64], e: usize) -> Result<Vec<Value<'c, 'c>>> {
        let tid = self.thread_id(block)?;
        let off = self.const_index(block, e as i64 * self.cta_threads)?;
        let mut rest = self.addi(block, tid, off)?;
        let mut idx = vec![rest; shape.len()];
        for d in (1..shape.len()).rev() {
            let extent = self.const_index(block, shape[d])?;
            idx[d] = self.remui(block, rest, extent)?;
            rest = self.divui(block, rest, extent)?;
        }
        idx[0] = rest;
        Ok(idx)
    }
}
