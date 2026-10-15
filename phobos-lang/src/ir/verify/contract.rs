// The typing rules of the contractions: dots, fragment accumulators and the
// register matmul.

use super::Verifier;
use crate::ir::{Coeff, GemmType, OpId, OpKind, Reduce, RegLayout, Space, Type};

impl Verifier<'_> {
    pub(super) fn contract(&mut self, op: OpId, kind: &OpKind) {
        let ir = self.ir;
        match kind {
            OpKind::GemmInit => {
                if self.want_operands(op, 1) {
                    self.want_scalar(op, 0, "init");
                }
                if self.want_results(op, 1) && ir.ty(ir.result(op)).gemm().is_none() {
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
                if self.want_results(op, 1) && !matches!(ir.ty(ir.result(op)), Type::Dist(_)) {
                    self.err(op, "result must be fragments");
                }
            }
            OpKind::FragScale(_) | OpKind::FragDot { .. } | OpKind::FragDotScaled | OpKind::FragOuter { .. } | OpKind::FragStore => {
                let n = match kind {
                    OpKind::FragDot { .. } | OpKind::FragOuter { .. } => 3,
                    OpKind::FragDotScaled => 5,
                    _ => 2,
                };
                if self.want_operands(op, n) {
                    if !matches!(ir.ty(ir.operand(op, 0)), Type::Dist(_)) {
                        self.err(op, "operand 0 must be fragments");
                    }
                    // The int8 updates' operands may be generated where they
                    // are read.
                    let int8 = matches!(kind, OpKind::FragDotScaled | OpKind::FragOuter { .. });
                    for i in 1..n {
                        let generated =
                            matches!(ir.ty(ir.operand(op, i)), Type::Dist(d) if d.layout == RegLayout::Generated);
                        if !(int8 && generated) {
                            self.want_tiles(op, i..i + 1);
                        }
                    }
                }
                if matches!(kind, OpKind::FragStore) {
                    self.want_results(op, 0);
                } else if self.want_results(op, 1)
                    && ir.ty(ir.result(op)) != ir.ty(ir.operand(op, 0))
                {
                    self.err(op, "result must be the accumulator's type");
                }
            }
            OpKind::FragMap(tree) => {
                if !self.want_results(op, 1) {
                    return;
                }
                let out = ir.ty(ir.result(op));
                let Some(acc) = mma_acc(out).filter(|a| !a.imma) else {
                    self.err(op, "result must be f16-path fragments");
                    return;
                };
                let mut leaves = Vec::new();
                tree.operands(&mut leaves);
                for i in leaves {
                    let Some(&v) = ir.operands(op).get(i) else {
                        self.err(op, format!("tree reads operand {i}, which it does not have"));
                        continue;
                    };
                    let ok = match ir.ty(v) {
                        t @ Type::Dist(_) => t == out,
                        Type::Tile(t) => t.static_shape().is_some_and(|s| {
                            s.len() == 2 && (s[0] == acc.m || s[0] == 1) && (s[1] == acc.n || s[1] == 1)
                        }),
                        Type::Scalar(_) => true,
                        Type::Tensor(_) => false,
                    };
                    if !ok {
                        self.err(op, format!("operand {i} is neither the fragments' layout nor a tile broadcast to them"));
                    }
                }
            }
            OpKind::FragReduce(how) => {
                if !matches!(how, Reduce::RowSum | Reduce::RowMax) {
                    self.err(op, format!("{} is not a row reduction of fragments", how.name()));
                }
                if !self.want_operands(op, 1) || !self.want_results(op, 1) {
                    return;
                }
                let Some(acc) = mma_acc(ir.ty(ir.operand(op, 0))) else {
                    self.err(op, "operand must be fragments");
                    return;
                };
                if acc.imma || acc.wn != 1 {
                    self.err(op, "only f16-path fragments whose warps own whole rows reduce in their lanes");
                }
                let column = matches!(ir.ty(ir.result(op)), Type::Tile(t)
                    if t.space == Space::Shared && t.static_shape().as_deref() == Some(&[acc.m, 1]));
                if !column {
                    self.err(op, "result must be a shared [rows, 1] tile");
                }
            }
            OpKind::Generate => {
                self.want_operands(op, 0);
                if !self.want_results(op, 1) {
                    return;
                }
                let Type::Dist(d) = ir.ty(ir.result(op)) else {
                    self.err(op, "result must be a generated tile");
                    return;
                };
                if d.layout != RegLayout::Generated || d.shape.len() != 2 {
                    self.err(op, "result must be a generated rank-2 tile");
                }
                let block_ty = if d.elem.bytes() == Some(1) { Type::INDEX } else { Type::Scalar(d.elem) };
                let [block] = ir.blocks_of(op)[..] else {
                    self.err(op, "owns one block");
                    return;
                };
                self.want_block_args(op, block, &[Type::INDEX, Type::INDEX]);
                self.want_terminator(op, block, &OpKind::Yield, &[block_ty]);
                // Every read computes the block afresh, so it may keep
                // private scalars but write nothing anyone else sees.
                let mut writes = false;
                ir.walk(block, &mut |inner| {
                    let kind = ir.kind(inner);
                    writes |= kind.writes(ir.operands(inner).len()).iter().any(|&i| {
                        ir.ty(ir.operand(inner, i)).as_tile().is_none_or(|t| t.space != Space::Private)
                    });
                });
                if writes {
                    self.err(op, "a generated block writes memory, which it may only read");
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
            t @ Type::Dist(_) => t.gemm(),
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

/// The fragments a value is typed as, if it is.
fn mma_acc(ty: &Type) -> Option<crate::ir::MmaAcc> {
    match ty {
        Type::Dist(d) => d.mma_acc(),
        _ => None,
    }
}
