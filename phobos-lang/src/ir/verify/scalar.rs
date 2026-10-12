// The typing rules of the scalar ops, which compute one value from others.

use super::Verifier;
use crate::ir::{IndexOp, Literal, OpId, OpKind, Scalar, Type, UnOp};

impl Verifier<'_> {
    pub(super) fn scalar_kind(&mut self, op: OpId, kind: &OpKind) {
        let ir = self.ir;
        match kind {
            OpKind::Const(lit) => {
                self.want_operands(op, 0);
                if self.want_results(op, 1) {
                    let want = match lit {
                        Literal::Int(_) => Scalar::Index,
                        Literal::Float(_) => Scalar::F32,
                        Literal::Bool(_) => Scalar::Bool,
                    };
                    if self.scalar(ir.result(op)) != Some(want) {
                        self.err(op, format!("result must be {want}"));
                    }
                }
            }
            OpKind::Unary(un) => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    let s = self.want_scalar(op, 0, "operand");
                    let ok = match (un, s) {
                        (UnOp::Neg, Some(s)) => s.is_numeric(),
                        (UnOp::Not, Some(s)) => s == Scalar::Bool,
                        (UnOp::BitNot, Some(s)) => s.is_int() || s == Scalar::Index,
                        _ => true,
                    };
                    if !ok {
                        self.err(op, "operand type does not take this operator");
                    }
                    if ir.ty(ir.result(op)) != ir.ty(ir.operand(op, 0)) {
                        self.err(op, "result type differs from the operand's");
                    }
                }
            }
            OpKind::Binary(bin) => {
                if self.want_operands(op, 2) && self.want_results(op, 1) {
                    let (a, b) = (
                        self.want_scalar(op, 0, "lhs"),
                        self.want_scalar(op, 1, "rhs"),
                    );
                    if let (Some(a), Some(b)) = (a, b) {
                        if a != b {
                            self.err(op, format!("operands differ: {a} vs {b}"));
                        }
                        match bin.result(a) {
                            Some(want) if self.scalar(ir.result(op)) != Some(want) => {
                                self.err(op, format!("result must be {want}"));
                            }
                            Some(_) => {}
                            None => self.err(op, format!("{a} operands do not take this operator")),
                        }
                    }
                }
            }
            OpKind::IndexOp(iop) => {
                if self.want_operands(op, 2) && self.want_results(op, 1) {
                    self.want_type(op, 0, &Type::INDEX, "lhs");
                    self.want_type(op, 1, &Type::INDEX, "rhs");
                    let want = match iop {
                        IndexOp::CmpUlt => Scalar::Bool,
                        _ => Scalar::Index,
                    };
                    if self.scalar(ir.result(op)) != Some(want) {
                        self.err(op, format!("result must be {want}"));
                    }
                }
            }
            OpKind::Cast(want) => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    if let Some(s) = self.want_scalar(op, 0, "operand")
                        && !(s.is_numeric() && want.is_numeric())
                    {
                        self.err(op, format!("cannot convert {s} to {want}"));
                    }
                    if self.scalar(ir.result(op)) != Some(*want) {
                        self.err(op, format!("result must be {want}"));
                    }
                }
            }
            OpKind::IndexCast(want) => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    if let Some(s) = self.want_scalar(op, 0, "operand") {
                        let sized = |t: Scalar| t.is_int() || t.is_unsigned();
                        let ok = (s == Scalar::Index && sized(*want))
                            || (sized(s) && *want == Scalar::Index);
                        if !ok {
                            self.err(op, format!("index_cast does not go from {s} to {want}"));
                        }
                    }
                    if self.scalar(ir.result(op)) != Some(*want) {
                        self.err(op, format!("result must be {want}"));
                    }
                }
            }
            OpKind::Thread(_) => {
                self.want_operands(op, 0);
                if self.want_results(op, 1) && self.scalar(ir.result(op)) != Some(Scalar::Index) {
                    self.err(op, "result must be index");
                }
            }
            OpKind::ProgramId(d) | OpKind::GridDim(d) => {
                self.want_operands(op, 0);
                if *d > 2 {
                    self.err(op, "dimension past z");
                }
                if self.want_results(op, 1) && self.scalar(ir.result(op)) != Some(Scalar::Index) {
                    self.err(op, "result must be index");
                }
            }
            OpKind::Math(_) => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    let s = self.want_scalar(op, 0, "operand");
                    let r = self.scalar(ir.result(op));
                    let ok = match s {
                        Some(Scalar::Index) => r == Some(Scalar::Index),
                        Some(s) if s.is_float() => r.is_some_and(Scalar::is_float),
                        _ => false,
                    };
                    if !ok {
                        self.err(op, "takes a float or an index to its own kind");
                    }
                }
            }
            OpKind::Select => {
                if self.want_operands(op, 3) && self.want_results(op, 1) {
                    if self.want_scalar(op, 0, "condition") != Some(Scalar::Bool) {
                        self.err(op, "condition must be bool");
                    }
                    let (a, b) = (ir.ty(ir.operand(op, 1)), ir.ty(ir.operand(op, 2)));
                    if a != b || ir.ty(ir.result(op)) != a {
                        self.err(op, "both choices and the result must share a type");
                    }
                }
            }
            _ => unreachable!("{} is not a scalar op", kind.name()),
        }
    }
}
