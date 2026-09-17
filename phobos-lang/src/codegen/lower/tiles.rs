use anyhow::{Result, bail};
use melior::ir::Block;

use super::{Lowered, dims};
use crate::ast::{AssignOp, BinOp};
use crate::codegen::elemwise::{ElemStep, Fused};
use crate::codegen::{Codegen, MemVal, Reduce as CgReduce};
use crate::ir::{self, Ir, Map, OpId, OpKind, Reduce, Tree};

impl<'c> Codegen<'c> {
    pub(super) fn emit_tile_op(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        let kind = ir.kind(op).clone();
        match kind {
            OpKind::AssumeAlign => {
                let mut mv = self.tile_ir(ir, ir.operand(op, 0))?;
                mv.mem = self.assume_align(block, mv.mem, 16)?;
                self.set_ir(ir.result(op), Lowered::Tile(mv));
            }
            OpKind::Alloc => {
                let ir::Type::Tile(t) = ir.ty(ir.result(op)) else {
                    bail!("alloc of a non-tile");
                };
                if t.space == ir::Space::Private {
                    return self.emit_scalar_op(block, ir, op);
                }
                let elem = self.ir_scalar_type(t.elem);
                let shape = dims(&t.shape);
                let mv = if t.layout.row_stride.is_some() {
                    self.alloc_tile_padded(block, elem, &shape)?
                } else if t.layout.swizzle.is_some() {
                    self.alloc_tile_swizzled(block, elem, &shape)?
                } else {
                    self.alloc_tile_shaped(block, elem, &shape)?
                };
                self.set_ir(ir.result(op), Lowered::Tile(mv));
            }
            OpKind::Slice(s) => {
                let src = self.tile_ir(ir, ir.operand(op, 0))?;
                let rank = s.rank();
                let offsets = (0..rank)
                    .map(|d| self.scalar_ir(ir.operand(op, 1 + d)))
                    .collect::<Result<Vec<_>>>()?;
                let dyn_sizes = (1 + rank..1 + rank + s.dyn_sizes())
                    .map(|i| self.scalar_ir(ir.operand(op, i)))
                    .collect::<Result<Vec<_>>>()?;
                let mut mask = vec![None; rank];
                for (d, slot) in mask.iter_mut().enumerate() {
                    if let Some(i) = s.mask_operand(d) {
                        *slot = Some((offsets[d], self.scalar_ir(ir.operand(op, i))?));
                    }
                }
                let ir::Type::Tile(t) = ir.ty(ir.result(op)) else {
                    bail!("slice to a non-tile");
                };
                let mv = self.subview_raw(
                    block,
                    &src,
                    &offsets,
                    &dyn_sizes,
                    dims(&s.sizes),
                    mask,
                    t.layout.align_div,
                )?;
                self.set_ir(ir.result(op), Lowered::Tile(mv));
            }
            OpKind::Flat => {
                let src = self.tile_ir(ir, ir.operand(op, 0))?;
                let mv = self.tile_flat(block, &src)?;
                self.set_ir(ir.result(op), Lowered::Tile(mv));
            }
            OpKind::Stage(stage) => {
                let src = self.tile_ir(ir, ir.operand(op, 0))?;
                let tile = if stage.pad {
                    self.alloc_tile_padded(block, src.elem, &src.shape)?
                } else {
                    self.alloc_tile_shaped(block, src.elem, &src.shape)?
                };
                self.tile_copy(block, &src, &tile, stage.sync, false)?;
                self.set_ir(ir.result(op), Lowered::Tile(tile));
            }
            OpKind::Materialize => {
                let view = self.tile_ir(ir, ir.operand(op, 0))?;
                let mv = self.materialize_masked(block, &view)?;
                self.set_ir(ir.result(op), Lowered::Tile(mv));
            }
            OpKind::HoistStage => {
                let src = self.tile_ir(ir, ir.operand(op, 0))?;
                let buf = if self.has_mma_sync() {
                    self.alloc_tile_swizzled(block, self.f16_t, &src.shape)?
                } else {
                    self.alloc_tile_shaped(block, self.f16_t, &src.shape)?
                };
                self.stage_to_f16(block, &src, &buf, false)?;
                self.set_ir(ir.result(op), Lowered::Tile(buf));
            }
            OpKind::Copy { sync } => {
                let src = self.tile_ir(ir, ir.operand(op, 0))?;
                let dst = self.tile_ir(ir, ir.operand(op, 1))?;
                self.tile_copy(block, &src, &dst, sync, false)?;
                self.release(&src);
            }
            OpKind::Convert => {
                let src = self.tile_ir(ir, ir.operand(op, 0))?;
                let dst = self.tile_ir(ir, ir.operand(op, 1))?;
                self.tile_convert(block, &src, &dst)?;
                self.release(&src);
            }
            OpKind::Fill => {
                let v = self.scalar_ir(ir.operand(op, 0))?;
                let dst = self.tile_ir(ir, ir.operand(op, 1))?;
                self.tile_fill(block, v, &dst)?;
            }
            OpKind::Accumulate => {
                let src = self.tile_ir(ir, ir.operand(op, 0))?;
                let dst = self.tile_ir(ir, ir.operand(op, 1))?;
                self.tile_binary(block, BinOp::Add, &dst, &src, &dst)?;
                self.release(&src);
            }
            OpKind::Map(map) => {
                let out = self.emit_map(block, ir, op, map, None)?;
                self.set_ir(ir.result(op), Lowered::Tile(out));
            }
            OpKind::MapInto(map) => {
                let n = ir.operands(op).len();
                let dst = self.tile_ir(ir, ir.operand(op, n - 1))?;
                self.emit_map(block, ir, op, map, Some(dst))?;
            }
            OpKind::Fused(tree) => {
                let n = ir.operands(op).len();
                let dst_id = ir.operand(op, n - 1);
                let target = self.tile_ir(ir, dst_id)?;
                let mut leaves = Vec::new();
                let fused = self.fused_from_tree(ir, op, &tree, dst_id, &mut leaves)?;
                self.fused_sweep(block, &target, &fused, &leaves)?;
                for leaf in &leaves {
                    self.release(leaf);
                }
            }
            OpKind::Chain(steps) => {
                let src = self.tile_ir(ir, ir.operand(op, 0))?;
                let dst = self.tile_ir(ir, ir.operand(op, 1))?;
                let steps: Vec<ElemStep<'c>> = steps.iter().map(|s| self.ir_elem_step(*s)).collect();
                self.elem_chain_sweep(block, &src, &dst, &steps)?;
                self.release(&src);
            }
            OpKind::ScaledAdd => {
                let s1 = self.scalar_ir(ir.operand(op, 0))?;
                let t1 = self.tile_ir(ir, ir.operand(op, 1))?;
                let s2 = self.scalar_ir(ir.operand(op, 2))?;
                let t2 = self.tile_ir(ir, ir.operand(op, 3))?;
                let dst = self.tile_ir(ir, ir.operand(op, 4))?;
                self.tile_scaled_add_into(block, s1, &t1, s2, &t2, &dst)?;
                self.release(&t1);
                self.release(&t2);
            }
            OpKind::Reduce(how) => {
                let t = self.tile_ir(ir, ir.operand(op, 0))?;
                let out = match how {
                    Reduce::RowMax => self.tile_rowreduce(block, &t, CgReduce::Max)?,
                    Reduce::RowSum => self.tile_rowreduce(block, &t, CgReduce::Sum)?,
                    Reduce::CumSum => self.tile_cumsum(block, &t)?,
                    Reduce::Tril => self.tile_tril(block, &t)?,
                    Reduce::Transpose => self.tile_transpose(block, &t)?,
                };
                if how != Reduce::Tril {
                    self.release(&t);
                }
                self.set_ir(ir.result(op), Lowered::Tile(out));
            }
            OpKind::Dot { transpose } => {
                let a = self.tile_ir(ir, ir.operand(op, 0))?;
                let b = self.tile_ir(ir, ir.operand(op, 1))?;
                let ir::Type::Tile(t) = ir.ty(ir.result(op)) else {
                    bail!("dot to a non-tile");
                };
                let elem = self.ir_scalar_type(t.elem);
                let out = self.alloc_tile_shaped(block, elem, &dims(&t.shape))?;
                if transpose {
                    if !self.wmma_dot(block, &a, &b, &out, true, false)? {
                        self.tile_matmul_t(block, &a, &b, &out)?;
                    }
                } else {
                    self.check_matmul_shapes(&a, &b, &out)?;
                    if !self.wmma_dot(block, &a, &b, &out, false, false)? {
                        let zero = self.zero_scalar(block, out.elem)?;
                        self.tile_fill(block, zero, &out)?;
                        self.tile_matmul(block, &a, &b, &out)?;
                    }
                }
                self.release(&a);
                self.release(&b);
                self.set_ir(ir.result(op), Lowered::Tile(out));
            }
            OpKind::DotInto {
                transpose,
                accumulate,
                aliased,
            } => {
                let a = self.tile_ir(ir, ir.operand(op, 0))?;
                let b = self.tile_ir(ir, ir.operand(op, 1))?;
                let target = self.tile_ir(ir, ir.operand(op, 2))?;
                let assign = if accumulate { AssignOp::Add } else { AssignOp::Set };
                if aliased {
                    self.dot_via_temp(block, &a, &b, &target, transpose, assign)?;
                } else {
                    self.dot_into(block, &a, &b, &target, transpose, assign)?;
                }
            }
            other => bail!("{} is not a tile op", other.name()),
        }
        Ok(())
    }

    /// `Map` into a fresh buffer of the result's type, or `MapInto` a
    /// given one.
    fn emit_map(
        &mut self,
        block: &Block<'c>,
        ir: &Ir,
        op: OpId,
        map: Map,
        into: Option<MemVal<'c>>,
    ) -> Result<MemVal<'c>> {
        let fresh = |cg: &mut Self| -> Result<MemVal<'c>> {
            let ir::Type::Tile(t) = ir.ty(ir.result(op)) else {
                bail!("map to a non-tile");
            };
            let elem = cg.ir_scalar_type(t.elem);
            cg.alloc_tile_shaped(block, elem, &dims(&t.shape))
        };
        match map {
            Map::Binary(bop) => {
                let a = self.tile_ir(ir, ir.operand(op, 0))?;
                let b = self.tile_ir(ir, ir.operand(op, 1))?;
                let out = match into {
                    Some(dst) => dst,
                    None => fresh(self)?,
                };
                self.tile_binary_dispatch(block, bop, &a, &b, &out)?;
                self.release(&a);
                self.release(&b);
                Ok(out)
            }
            Map::Scalar { op: bop, scalar_left } => {
                let tile = self.tile_ir(ir, ir.operand(op, 0))?;
                let scalar = self.scalar_ir(ir.operand(op, 1))?;
                let out = match into {
                    Some(dst) => dst,
                    None => self.alloc_tile_shaped(block, tile.elem, &tile.shape)?,
                };
                self.tile_scalar_into(block, bop, &tile, scalar, scalar_left, &out)?;
                self.release(&tile);
                Ok(out)
            }
            Map::Unary(step) => {
                let t = self.tile_ir(ir, ir.operand(op, 0))?;
                match (step, into) {
                    (ir::ElemStep::Exp, Some(dst)) => {
                        self.tile_exp_into(block, &t, &dst)?;
                        Ok(dst)
                    }
                    (_, Some(_)) => bail!("no in-place form of {step}"),
                    (ir::ElemStep::Cast(want), None) => {
                        let want = self.ir_scalar_type(want);
                        let out = self.tile_cast(block, &t, want)?;
                        self.release(&t);
                        Ok(out)
                    }
                    (ir::ElemStep::Exp, None) => self.tile_exp(block, &t),
                    (ir::ElemStep::Log, None) => self.tile_log(block, &t),
                    (ir::ElemStep::Round, None) => self.tile_round(block, &t),
                    (ir::ElemStep::Sqrt, None) => self.tile_sqrt(block, &t),
                    (ir::ElemStep::Tanh, None) => self.tile_tanh(block, &t),
                }
            }
            Map::Max => {
                let a = self.tile_ir(ir, ir.operand(op, 0))?;
                let b = self.tile_ir(ir, ir.operand(op, 1))?;
                let out = match into {
                    Some(dst) => dst,
                    None => fresh(self)?,
                };
                self.tile_max_bc(block, &a, &b, &out)?;
                self.release(&a);
                self.release(&b);
                Ok(out)
            }
        }
    }

    /// The dot arm of a store, past the self-alias check.
    fn dot_into(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        target: &MemVal<'c>,
        transpose: bool,
        op: AssignOp,
    ) -> Result<()> {
        if self.wmma_dot(block, a, b, target, transpose, op == AssignOp::Add)? {
            self.release(a);
            self.release(b);
            return Ok(());
        }
        if transpose {
            if op == AssignOp::Add {
                bail!("`tile += dot_t(...)` is not supported; assign with `=`");
            }
            self.tile_matmul_t(block, a, b, target)?;
        } else {
            if op == AssignOp::Set {
                let zero = self.zero_scalar(block, target.elem)?;
                self.tile_fill(block, zero, target)?;
            }
            self.tile_matmul(block, a, b, target)?;
        }
        self.release(a);
        self.release(b);
        Ok(())
    }

    /// `p = dot(p, p)` routed through a temp, as `store_tile` routes it.
    fn dot_via_temp(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        target: &MemVal<'c>,
        transpose: bool,
        op: AssignOp,
    ) -> Result<()> {
        let temp = self.alloc_tile_shaped(block, target.elem, &target.shape)?;
        if transpose {
            self.tile_matmul_t(block, a, b, &temp)?;
        } else {
            let zero = self.zero_scalar(block, temp.elem)?;
            self.tile_fill(block, zero, &temp)?;
            self.tile_matmul(block, a, b, &temp)?;
        }
        self.release(a);
        self.release(b);
        match op {
            AssignOp::Set => self.tile_copy(block, &temp, target, true, false)?,
            AssignOp::Add => self.tile_binary(block, BinOp::Add, target, &temp, target)?,
        }
        self.release(&temp);
        Ok(())
    }

    fn ir_elem_step(&self, step: ir::ElemStep) -> ElemStep<'c> {
        match step {
            ir::ElemStep::Cast(s) => ElemStep::Cast(self.ir_scalar_type(s)),
            ir::ElemStep::Round => ElemStep::Round,
            ir::ElemStep::Sqrt => ElemStep::Sqrt,
            ir::ElemStep::Exp => ElemStep::Exp,
            ir::ElemStep::Log => ElemStep::Log,
            ir::ElemStep::Tanh => ElemStep::Tanh,
        }
    }

    /// The emitter's fused tree from the op's, with every tile leaf that
    /// is not the target collected for release, once per mention.
    fn fused_from_tree(
        &self,
        ir: &Ir,
        op: OpId,
        tree: &Tree,
        target: ir::ValueId,
        leaves: &mut Vec<MemVal<'c>>,
    ) -> Result<Fused<'c>> {
        Ok(match tree {
            Tree::Leaf(i) => {
                let v = ir.operand(op, *i);
                let mv = self.tile_ir(ir, v)?;
                if v != target {
                    leaves.push(mv.clone());
                }
                Fused::Tile(mv)
            }
            Tree::Scalar(i) => Fused::Scalar(self.scalar_ir(ir.operand(op, *i))?),
            Tree::Unary(step, x) => Fused::Unary(
                self.ir_elem_step(*step),
                Box::new(self.fused_from_tree(ir, op, x, target, leaves)?),
            ),
            Tree::Binary(bop, a, b) => Fused::Binary(
                *bop,
                Box::new(self.fused_from_tree(ir, op, a, target, leaves)?),
                Box::new(self.fused_from_tree(ir, op, b, target, leaves)?),
            ),
            Tree::Max(a, b) => Fused::Max(
                Box::new(self.fused_from_tree(ir, op, a, target, leaves)?),
                Box::new(self.fused_from_tree(ir, op, b, target, leaves)?),
            ),
        })
    }
}
