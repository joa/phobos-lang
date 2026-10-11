// A dot whose product only feeds an elementwise update, lowered as one op:
// the update runs on each element as the product is written out. A path
// with no such write-out, the tensor cores or a dynamic shape, emits the dot
// and then the update's sweep, as it would have unfused.

use anyhow::Result;
use melior::ir::{Block, Type, Value};

use crate::ast::AssignOp;
use crate::codegen::elemwise::Fused;
use crate::codegen::{Codegen, DYN, MemVal};
use crate::ir::{Ir, OpId, Tree};

/// What `fused_from_tree` makes of a leaf that is not a plain tile.
#[derive(Clone, Copy)]
pub(super) enum Fill<'a, 'c> {
    None,
    /// The element of the register tile being built.
    RegElem(usize),
    /// A fused dot's product, filled in by the write-out.
    Hole,
    /// A fused dot's product, already in `tile`; `release` when the tile is
    /// a temporary.
    Product { tile: &'a MemVal<'c>, release: bool },
}

impl<'c> Codegen<'c> {
    pub(super) fn emit_dot_fused(
        &mut self,
        block: &Block<'c>,
        ir: &Ir,
        op: OpId,
        transpose: bool,
        tree: &Tree,
        in_place: bool,
    ) -> Result<()> {
        let a = self.tile_ir(ir, ir.operand(op, 0))?;
        let b = self.tile_ir(ir, ir.operand(op, 1))?;
        let dst_id = *ir.operands(op).last().expect("verified: a destination");
        let dst = self.tile_ir(ir, dst_id)?;

        if !self.fused_write_out(&a, &b, &dst, transpose) {
            let product = if in_place {
                dst.clone()
            } else {
                self.alloc_tile_shaped(block, dst.elem, &dst.shape)?
            };
            self.dot_into(block, &a, &b, &product, transpose, AssignOp::Set)?;
            let mut leaves = Vec::new();
            let fill = Fill::Product {
                tile: &product,
                release: !in_place,
            };
            let fused = self.fused_from_tree(ir, op, tree, dst_id, &mut leaves, fill)?;
            self.fused_sweep(block, &dst, &fused, &leaves)?;
            for leaf in &leaves {
                self.release(leaf);
            }
            return Ok(());
        }

        let mut leaves = Vec::new();
        let fused = self.fused_from_tree(ir, op, tree, dst_id, &mut leaves, Fill::Hole)?;
        Self::check_sweep_leaves(&dst, &leaves)?;
        let (elem, shape) = (dst.elem, dst.shape.clone());
        let vec_t = Type::vector(&[4], elem);
        let epilogue = |cg: &mut Self, blk: &Block<'c>, product: Value<'c, 'c>, idx: &[Value<'c, 'c>]| {
            let v = cg.eval_fused(blk, &fused.fill_hole(&Fused::Scalar(product)), idx, &shape, 1, vec_t)?;
            cg.coerce(blk, v, elem)
        };
        if transpose {
            self.tile_matmul_t_with(block, &a, &b, &dst, false, Some(&epilogue))?;
        } else {
            self.tile_matmul_with(block, &a, &b, &dst, Some(&epilogue))?;
        }
        self.release(&a);
        self.release(&b);
        for leaf in &leaves {
            self.release(leaf);
        }
        Ok(())
    }

    /// Whether the dot takes a path that writes each element out from a
    /// register: neither the tensor cores nor the integer `dot_t` paths, and
    /// a static shape.
    fn fused_write_out(&self, a: &MemVal<'c>, b: &MemVal<'c>, dst: &MemVal<'c>, transpose: bool) -> bool {
        if self.wmma_takes(a, b, dst) || dst.shape.contains(&DYN) {
            return false;
        }
        if transpose {
            a.shape[1] != DYN && !(a.elem == self.i8_t && b.elem == self.i8_t)
        } else {
            true
        }
    }
}
