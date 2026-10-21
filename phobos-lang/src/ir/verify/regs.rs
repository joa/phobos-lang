// The typing rules of the register-tile ops.

use super::Verifier;
use crate::ir::{OpId, OpKind, Type};

impl Verifier<'_> {
    pub(super) fn regs(&mut self, op: OpId, kind: &OpKind) {
        let ir = self.ir;
        match kind {
            OpKind::RegsFill => {
                if self.want_operands(op, 1) {
                    self.want_scalar(op, 0, "fill");
                }
                self.want_regs_result(op);
            }
            OpKind::RegsFused(tree) => {
                let Some(shape) = self.want_regs_result(op) else {
                    return;
                };
                let layout = ir.ty(ir.result(op)).regs().map(|r| r.layout);
                for (i, &v) in ir.operands(op).iter().enumerate() {
                    if let Some(r) = ir.ty(v).regs()
                        && (r.shape != shape || Some(r.layout) != layout)
                    {
                        self.err(op, format!("register operand {i} has another shape or layout"));
                    }
                }
                self.check_tree(op, tree);
            }
            OpKind::RegsStore => {
                self.want_results(op, 0);
                if !self.want_operands(op, 2) {
                    return;
                }
                let (Some(r), Type::Tile(t)) = (ir.ty(ir.operand(op, 0)).regs(), ir.ty(ir.operand(op, 1))) else {
                    self.err(op, "stores a register tile into a tile");
                    return;
                };
                if t.static_shape().as_deref() != Some(&r.shape[..]) {
                    self.err(op, "the destination has another shape");
                }
            }
            _ => unreachable!("only register-tile ops reach here"),
        }
    }

    /// The register tile an op yields, by shape.
    fn want_regs_result(&mut self, op: OpId) -> Option<Vec<i64>> {
        if !self.want_results(op, 1) {
            return None;
        }
        match self.ir.ty(self.ir.result(op)) {
            t @ Type::Dist(_) => t.regs().map(|r| r.shape.clone()),
            other => {
                self.err(op, format!("result must be a register tile, is {other}"));
                None
            }
        }
    }
}
