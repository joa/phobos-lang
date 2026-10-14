// The typing rules of the per-thread ops.

use super::Verifier;
use crate::ir::{OpId, OpKind, Scalar, Type};

impl Verifier<'_> {
    pub(super) fn lanes(&mut self, op: OpId, kind: &OpKind) {
        let ir = self.ir;
        match kind {
            OpKind::Shuffle { .. } | OpKind::Converged(_) | OpKind::Exchange => {
                let want = if matches!(kind, OpKind::Shuffle { .. }) { 2 } else { 1 };
                if self.want_operands(op, want) {
                    let t = self.scalar(ir.operand(op, 0));
                    if t.is_none() {
                        self.err(op, "operand must be a scalar");
                    }
                    if ir.results(op).iter().any(|&r| self.scalar(r) != t) {
                        self.err(op, "results must have the operand's type");
                    }
                }
            }
            OpKind::LaneLoad { len, .. } => {
                if !matches!(ir.operands(op).first().map(|&t| ir.ty(t)), Some(Type::Tensor(_) | Type::Tile(_))) {
                    self.err(op, "operand must be a tensor or a tile");
                }
                if ir.results(op).len() != *len as usize {
                    self.err(op, "one result per element");
                }
            }
            OpKind::Dot4 { a, b } => {
                let bytes = a.operands() + b.operands();
                if self.want_operands(op, bytes + 1) && self.want_results(op, 1) {
                    let i8s = ir.operands(op)[..bytes].iter().all(|&v| self.scalar(v) == Some(Scalar::I8));
                    if !i8s || self.scalar(ir.operand(op, bytes)) != Some(Scalar::Index) {
                        self.err(op, "takes i8 elements and an index accumulator");
                    }
                }
            }
            OpKind::Word | OpKind::HalfWord | OpKind::Dp4a | OpKind::BytePerm | OpKind::PopCount => {
                let (want, elem) = match kind {
                    OpKind::Word => (4, Scalar::I8),
                    OpKind::HalfWord => (2, Scalar::I8),
                    OpKind::PopCount => (1, Scalar::Index),
                    _ => (3, Scalar::Index),
                };
                if self.want_operands(op, want) && self.want_results(op, 1) {
                    if ir.operands(op).iter().any(|&v| self.scalar(v) != Some(elem)) {
                        self.err(op, format!("operands must be {elem}"));
                    }
                    if self.scalar(ir.result(op)) != Some(Scalar::Index) {
                        self.err(op, "result must be index");
                    }
                }
            }
            OpKind::F16Bits => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    if self.scalar(ir.operand(op, 0)) != Some(Scalar::Index) {
                        self.err(op, "operand must be index");
                    }
                    if self.scalar(ir.result(op)) != Some(Scalar::F16) {
                        self.err(op, "result must be f16");
                    }
                }
            }
            OpKind::LaneStore { len, words, .. } => {
                let operands = ir.operands(op);
                if !matches!(operands.first().map(|&t| ir.ty(t)), Some(Type::Tensor(_) | Type::Tile(_))) {
                    self.err(op, "operand must be a tensor or a tile");
                }
                let values = if *words { *len as usize / 4 } else { *len as usize };
                if operands.len() < values + 1 {
                    self.err(op, "one value per element, or per word");
                }
                self.want_results(op, 0);
            }
            other => unreachable!("{} is not a per-thread op", other.name()),
        }
    }
}
