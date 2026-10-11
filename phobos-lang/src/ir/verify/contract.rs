// The typing rules of the contractions: dots, fragment accumulators and the
// register matmul.

use super::Verifier;
use crate::ir::{Coeff, GemmType, OpId, OpKind, Space, Type};

impl Verifier<'_> {
    pub(super) fn contract(&mut self, op: OpId, kind: &OpKind) {
        let ir = self.ir;
        match kind {
            OpKind::GemmInit => {
                if self.want_operands(op, 1) {
                    self.want_scalar(op, 0, "init");
                }
                if self.want_results(op, 1) && !matches!(ir.ty(ir.result(op)), Type::Gemm(_)) {
                    self.err(op, "result must be a gemm accumulator");
                }
            }
            OpKind::GemmDot => {
                if self.want_operands(op, 3)
                    && let Some(g) = self.gemm_acc(op, 0)
                {
                    self.want_gemm_slice(op, 1, &[g.m, g.k], true);
                    self.want_gemm_slice(op, 2, &[g.k, g.n], true);
                }
                if self.want_results(op, 1) && ir.ty(ir.result(op)) != ir.ty(ir.operand(op, 0)) {
                    self.err(op, "result must be the accumulator's type");
                }
            }
            OpKind::GemmStore { alpha, beta } => {
                if self.want_operands(op, 2 + alpha.operands() + beta.operands())
                    && let Some(g) = self.gemm_acc(op, 0)
                {
                    self.want_gemm_slice(op, 1, &[g.m, g.n], false);
                    let mut i = 2;
                    for (c, what) in [(alpha, "alpha"), (beta, "beta")] {
                        if *c == Coeff::Given {
                            self.want_scalar(op, i, what);
                            i += 1;
                        }
                    }
                }
                self.want_results(op, 0);
            }
            OpKind::FragInit(_) => {
                self.want_operands(op, 0);
                if self.want_results(op, 1) && !matches!(ir.ty(ir.result(op)), Type::Frags(_)) {
                    self.err(op, "result must be fragments");
                }
            }
            OpKind::FragScale(_) | OpKind::FragDot | OpKind::FragStore => {
                let n = match kind {
                    OpKind::FragDot => 3,
                    _ => 2,
                };
                if self.want_operands(op, n) {
                    if !matches!(ir.ty(ir.operand(op, 0)), Type::Frags(_)) {
                        self.err(op, "operand 0 must be fragments");
                    }
                    self.want_tiles(op, 1..n);
                }
                if matches!(kind, OpKind::FragStore) {
                    self.want_results(op, 0);
                } else if self.want_results(op, 1)
                    && ir.ty(ir.result(op)) != ir.ty(ir.operand(op, 0))
                {
                    self.err(op, "result must be the accumulator's type");
                }
            }
            OpKind::Dot { .. } => {
                if self.want_operands(op, 2) {
                    self.want_tiles(op, 0..2);
                }
                self.want_tile_result(op);
            }
            OpKind::DotInto { .. } => {
                if self.want_operands(op, 3) {
                    self.want_tiles(op, 0..3);
                }
                self.want_results(op, 0);
            }
            OpKind::DotFused { tree, .. } => {
                let n_ops = ir.operands(op).len();
                if n_ops < 3 {
                    self.err(op, "needs two factors and a destination");
                    return;
                }
                self.want_tiles(op, 0..2);
                self.want_tiles(op, n_ops - 1..n_ops);
                let mut read = Vec::new();
                tree.operands(&mut read);
                for i in read {
                    if i == 1 || i >= n_ops - 1 {
                        self.err(op, format!("tree reads operand {i}, neither the product nor a leaf"));
                    }
                }
                self.check_tree(op, tree);
                self.want_results(op, 0);
            }
            other => self.err(op, format!("{} is not a contraction", other.name())),
        }
    }

    /// Operand `i` as the gemm accumulator it must be.
    fn gemm_acc(&mut self, op: OpId, i: usize) -> Option<GemmType> {
        match self.ir.ty(*self.ir.operands(op).get(i)?) {
            Type::Gemm(g) => Some(*g),
            other => {
                self.err(op, format!("operand {i} must be a gemm accumulator, is {other}"));
                None
            }
        }
    }

    /// Operand `i` as a `shape` tile the register matmul may index with no
    /// bounds guard. That is an unmasked tensor slice or, where the loop
    /// stages it anyway, the shared tile a masked one was materialized into.
    fn want_gemm_slice(&mut self, op: OpId, i: usize, shape: &[i64], shared_ok: bool) {
        let Some(&v) = self.ir.operands(op).get(i) else { return };
        let masked = self
            .ir
            .def_op(v)
            .is_some_and(|d| matches!(self.ir.kind(d), OpKind::Slice(s) if s.masked.iter().any(|&m| m)));
        match self.ir.ty(v) {
            Type::Tile(t) if (t.space == Space::Global && !masked) || (t.space == Space::Shared && shared_ok) => {
                if t.static_shape().as_deref() != Some(shape) {
                    self.err(op, format!("operand {i} must be a {shape:?} tile, is {}", self.ir.ty(v)));
                }
            }
            other if shared_ok => {
                self.err(op, format!("operand {i} must be an unmasked slice or a shared tile, is {other}"))
            }
            other => self.err(op, format!("operand {i} must be an unmasked global slice, is {other}")),
        }
    }
}
