// Register tiles: each thread's share of a tile as values. Element `e` of
// thread `t` sits at row-major position `t + e * threads`, the mapping a
// plain sweep uses, so reading a shared tile at the same element needs no
// exchange between threads. A fragment-row column instead holds the rows a
// lane holds of f16-path fragments, see `RegLayout::MmaRows`.

use anyhow::{Result, bail};
use melior::dialect::{arith, memref, scf};
use melior::ir::{Block, BlockLike, Region, RegionLike, Type, Value};

use super::{Fill, Lowered};
use crate::codegen::{Codegen, FragAcc, Reduce};
use crate::ir::{self, Ir, OpId, OpKind, ValueId};

#[derive(Clone)]
pub(in crate::codegen) struct RegTile<'c> {
    pub(in crate::codegen) vals: Vec<Value<'c, 'c>>,
    pub(in crate::codegen) elem: Type<'c>,
    pub(in crate::codegen) shape: Vec<i64>,
    /// The warps of a fragment-row column; None for a blocked tile.
    pub(in crate::codegen) rows_wm: Option<i64>,
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
                let count = self.regs_per_thread(&ty);
                let vals = vec![v; count];
                self.set_ir(ir.result(op), Lowered::Regs(RegTile { vals, ..ty }));
            }
            OpKind::RegsFused(tree) => {
                let ty = self.regs_type(ir, ir.result(op))?;
                let vec_t = Type::vector(&[4], ty.elem);
                let mut leaves = Vec::new();
                let mut vals = Vec::new();
                for e in 0..self.regs_per_thread(&ty) {
                    let idx = self.reg_index(block, &ty, e)?;
                    // Tile leaves are the same for every element; keep one copy.
                    let mut seen = Vec::new();
                    let fused = self.fused_from_tree(ir, op, tree, ir.result(op), &mut seen, Fill::RegElem(e))?;
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
                // A fragment-row column's quad holds each row four times;
                // its first lane stores it.
                let (body, guard) = match regs.rows_wm {
                    Some(_) => {
                        let tid = self.thread_id(block)?;
                        let four = self.const_index(block, 4)?;
                        let in_quad = self.remui(block, tid, four)?;
                        let zero = self.const_index(block, 0)?;
                        let lead = self.push(block, arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, in_quad, zero, self.loc))?;
                        (Block::new(&[]), Some(lead))
                    }
                    None => (Block::new(&[]), None),
                };
                let at: &Block<'c> = if guard.is_some() { &body } else { block };
                for (e, &v) in regs.vals.iter().enumerate() {
                    let idx = self.reg_index(at, &regs, e)?;
                    let v = self.coerce(at, v, dst.elem)?;
                    at.append_operation(memref::store(v, dst.mem, &idx, self.loc));
                }
                if let Some(lead) = guard {
                    body.append_operation(scf::r#yield(&[], self.loc));
                    let region = Region::new();
                    region.append_block(body);
                    block.append_operation(scf::r#if(lead, &[], region, Region::new(), self.loc));
                }
                self.barrier(block)?;
            }
            OpKind::RegsReduce(how) => {
                let regs = self.regs_ir(ir.operand(op, 0))?;
                let kind = match how {
                    ir::Reduce::RowSum => Reduce::Sum,
                    ir::Reduce::RowMax => Reduce::Max,
                    other => bail!("{other:?} of a register tile"),
                };
                let mut idx = Vec::new();
                for e in 0..regs.vals.len() {
                    idx.push(self.reg_index(block, &regs, e)?);
                }
                let out = self.regs_rowreduce(block, &regs.vals, &idx, &regs.shape, regs.elem, kind)?;
                self.set_ir(ir.result(op), Lowered::Tile(out));
            }
            other => bail!("{} is not a register-tile op", other.name()),
        }
        Ok(())
    }

    /// The register tile `v` is typed as, with no values yet.
    fn regs_type(&self, ir: &Ir, v: ValueId) -> Result<RegTile<'c>> {
        let Some(r) = ir.ty(v).regs() else {
            bail!("{v} is not typed as a register tile");
        };
        Ok(RegTile {
            vals: Vec::new(),
            elem: self.ir_scalar_type(r.elem),
            shape: r.shape.clone(),
            rows_wm: ir.ty(v).mma_rows(),
        })
    }

    fn regs_per_thread(&self, ty: &RegTile<'c>) -> usize {
        match ty.rows_wm {
            Some(wm) => (ty.shape[0] / 16 / wm * 2) as usize,
            None => (ty.shape.iter().product::<i64>() / self.cta_threads) as usize,
        }
    }

    /// The coordinates of this thread's element `e`.
    fn reg_index(&self, block: &Block<'c>, ty: &RegTile<'c>, e: usize) -> Result<Vec<Value<'c, 'c>>> {
        let shape = &ty.shape[..];
        if let Some(wm) = ty.rows_wm {
            let rows = shape[0] / wm;
            let (.., m0, _) = self.warp_block_origin(block, wm, 1, rows, 1)?;
            let tid = self.thread_id(block)?;
            let w = self.const_index(block, 32)?;
            let lane = self.remui(block, tid, w)?;
            let (gid, _) = self.mma_sync_dfrag_base(block, lane)?;
            let off = self.const_index(block, 16 * (e as i64 / 2) + 8 * (e as i64 % 2))?;
            let row = self.addi(block, m0, off)?;
            let row = self.addi(block, row, gid)?;
            return Ok(vec![row, self.const_index(block, 0)?]);
        }
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
