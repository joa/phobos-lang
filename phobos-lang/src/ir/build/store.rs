use anyhow::{Result, bail};

use super::{Binding, Build, DYN, Rv};
use crate::ast::{AssignOp, BinOp, Expr, ExprKind};
use crate::ir::{ElemStep, Intrinsic, Map, OpKind, RawFmt, Scalar, Tree, Type, ValueId};

impl Build {
    /// Lowers `target op= value` for a tile-shaped target.
    ///
    /// Tries a cascade of forms in order. The first that matches decides
    /// which emitter runs.
    pub(crate) fn store_tile(&mut self, target: ValueId, op: AssignOp, value: &Expr) -> Result<()> {
        // <slice or tile> = a register tile of its shape: each thread writes
        // its own elements.
        if let ExprKind::Var(n) = &value.kind
            && let Some(Binding::Regs(r)) = self.lookup(n)
            && op == AssignOp::Set
            && !self.is_masked(target)
            && matches!(self.ir.ty(r), Type::Regs(t) if t.shape == self.shape(target))
        {
            self.stmt(OpKind::RegsStore, &[r, target]);
            return Ok(());
        }
        // <tensor slice> = acc for a fragment accumulator.
        if let ExprKind::Var(n) = &value.kind
            && let Some(Binding::Frags(fa)) = self.lookup(n)
        {
            if op != AssignOp::Set {
                bail!("fragment accumulator '{n}' cannot be accumulated into a tile");
            }
            self.stmt(OpKind::FragStore, &[fa, target]);
            return Ok(());
        }

        // t = exp(t) rewrites the tile in place.
        if op == AssignOp::Set
            && let ExprKind::Call { callee, args } = &value.kind
            && callee == "exp"
            && let [Expr { kind: ExprKind::Var(n), .. }] = &args[..]
            && let Some(Binding::Tile(src)) = self.lookup(n)
            && self.is_buffer(src)
            && src == target
        {
            self.stmt(OpKind::MapInto(Map::Unary(ElemStep::Exp)), &[src, target]);
            return Ok(());
        }

        let f32_target = !self.is_masked(target) && self.elem(target) == Scalar::F32;

        // t = qmma_t(..) writes the accumulators where they are wanted.
        if op == AssignOp::Set
            && let ExprKind::Call { callee, args } = &value.kind
            && let Some(qmma @ (Intrinsic::QmmaT | Intrinsic::Q50mmaT)) = Intrinsic::from_name(callee)
            && f32_target
        {
            let mut operands = self.qmma_operands(args)?.to_vec();
            operands.push(target);
            self.stmt(OpKind::IntrinsicInto(qmma), &operands);
            return Ok(());
        }

        if op == AssignOp::Set
            && let ExprKind::Call { callee, args } = &value.kind
            && callee == "iq1s_qmma_t"
            && f32_target
        {
            let mut operands = self.iq1s_qmma_operands(args)?.to_vec();
            operands.push(target);
            self.stmt(OpKind::IntrinsicInto(Intrinsic::RawQmma(RawFmt::Iq1s)), &operands);
            return Ok(());
        }

        if op == AssignOp::Set
            && let ExprKind::Call { callee, args } = &value.kind
            && let Some(fmt) = qgemm_format(callee)
            && f32_target
        {
            let mut operands = self.qgemm_operands(fmt, callee, args)?;
            operands.push(target);
            self.stmt(OpKind::IntrinsicInto(Intrinsic::RawQgemm(fmt)), &operands);
            return Ok(());
        }

        if op == AssignOp::Set
            && let ExprKind::Call { callee, args } = &value.kind
            && callee == "iq1s_qmma_staged_t"
            && f32_target
        {
            let mut operands = self.iq1s_qmma_operands(args)?.to_vec();
            operands.push(target);
            self.stmt(
                OpKind::IntrinsicInto(Intrinsic::RawQmmaStaged(RawFmt::Iq1s)),
                &operands,
            );
            return Ok(());
        }

        if op == AssignOp::Set
            && let ExprKind::Call { callee, args } = &value.kind
            && let Some(fmt) = staged_qmma_format(callee)
            && f32_target
        {
            let mut operands = self.iq2xxs_qmma_operands(callee, args)?.to_vec();
            operands.push(target);
            self.stmt(OpKind::IntrinsicInto(Intrinsic::RawQmmaStaged(fmt)), &operands);
            return Ok(());
        }

        if let ExprKind::Call { callee, args } = &value.kind
            && (callee == "dot" || callee == "dot_t")
        {
            if self.is_masked(target) {
                bail!(
                    "writing a dot result directly into a partially out-of-bounds \
                     tensor slice is unsupported; accumulate into a tile first"
                );
            }
            let transpose = callee == "dot_t";
            let (a, b) = self.dot_operands(args)?;

            // `p = dot(p, p)` goes through a temp, since the target aliases an operand.
            if self.is_buffer(target) && (target == a || target == b) {
                self.stmt(
                    OpKind::DotInto {
                        transpose,
                        accumulate: op == AssignOp::Add,
                        aliased: true,
                    },
                    &[a, b, target],
                );
                return Ok(());
            }

            let (ash, bsh, tsh) = (self.shape(a), self.shape(b), self.shape(target));
            if transpose {
                if ash.len() != 2 || bsh.len() != 2 {
                    bail!("dot_t expects rank-2 tiles");
                }
                self.check_shapes(&[ash[1]], &[bsh[1]], "dot_t contraction dim")?;
                self.check_shapes(&[ash[0], bsh[0]], &tsh, "dot_t result")?;
            } else {
                self.check_matmul_shapes(&ash, &bsh, &tsh)?;
            }
            self.stmt(
                OpKind::DotInto {
                    transpose,
                    accumulate: op == AssignOp::Add,
                    aliased: false,
                },
                &[a, b, target],
            );
            return Ok(());
        }

        // target = i8(i32(round(t))) and its like: one sweep for the chain.
        if op == AssignOp::Set && self.store_elem_chain(target, value)? {
            return Ok(());
        }

        // Fused GEMM epilogue: target = s1 * t1 + s2 * t2 in one loop.
        if op == AssignOp::Set
            && let ExprKind::Binary {
                op: BinOp::Add,
                lhs,
                rhs,
            } = &value.kind
            && let Some((s1_expr, t1_expr)) = self.as_scale_mul(lhs)
            && let Some((s2_expr, t2_expr)) = self.as_scale_mul(rhs)
        {
            let s1 = self.emit_scalar(s1_expr)?;
            let Rv::Tile(t1) = self.emit_expr(t1_expr)? else {
                bail!("GEMM epilogue: expected tile in first operand");
            };
            let s2 = self.emit_scalar(s2_expr)?;
            let Rv::Tile(t2) = self.emit_expr(t2_expr)? else {
                bail!("GEMM epilogue: expected tile in second operand");
            };
            let tsh = self.shape(target);
            self.check_shapes(&self.shape(t1), &tsh, "GEMM epilogue lhs")?;
            self.check_shapes(&self.shape(t2), &tsh, "GEMM epilogue rhs")?;
            self.stmt(OpKind::ScaledAdd, &[s1, t1, s2, t2, target]);
            return Ok(());
        }

        // Anything else built out of arithmetic, tmax and the per-element
        // math calls: the whole tree in one sweep.
        if self.store_fused(target, op, value)? {
            return Ok(());
        }

        // t = x * y directly into the target without a temp buffer.
        if op == AssignOp::Set
            && let ExprKind::Binary { op: bop, lhs, rhs } = &value.kind
        {
            let l = self.emit_expr(lhs)?;
            return match (l, self.emit_expr(rhs)?) {
                (Rv::Tile(a), Rv::Tile(b)) => {
                    super::expr::check_tile_op(*bop, self.elem(a))?;
                    self.stmt(OpKind::MapInto(Map::Binary(*bop)), &[a, b, target]);
                    Ok(())
                }
                (Rv::Scalar(a), Rv::Scalar(b)) => {
                    let v = self.emit_binop(*bop, a, b)?;
                    let v = self.coerce(v, self.elem(target))?;
                    self.stmt(OpKind::Fill, &[v, target]);
                    Ok(())
                }
                (Rv::Tile(a), Rv::Scalar(b)) => {
                    super::expr::check_tile_op(*bop, self.elem(a))?;
                    self.stmt(
                        OpKind::MapInto(Map::Scalar {
                            op: *bop,
                            scalar_left: false,
                        }),
                        &[a, b, target],
                    );
                    Ok(())
                }
                (Rv::Scalar(a), Rv::Tile(b)) => {
                    super::expr::check_tile_op(*bop, self.elem(b))?;
                    self.stmt(
                        OpKind::MapInto(Map::Scalar {
                            op: *bop,
                            scalar_left: true,
                        }),
                        &[b, a, target],
                    );
                    Ok(())
                }
            };
        }

        match (op, self.emit_expr(value)?) {
            (AssignOp::Set, Rv::Scalar(v)) => {
                let v = self.coerce(v, self.elem(target))?;
                self.stmt(OpKind::Fill, &[v, target]);
                Ok(())
            }
            (AssignOp::Add, Rv::Scalar(v)) => {
                let map = Map::Scalar {
                    op: BinOp::Add,
                    scalar_left: false,
                };
                self.stmt(OpKind::MapInto(map), &[target, v, target]);
                Ok(())
            }
            (AssignOp::Set, Rv::Tile(src)) => {
                self.check_shapes(&self.shape(src), &self.shape(target), "tile store")?;
                if self.elem(src) != self.elem(target) {
                    self.stmt(OpKind::Convert, &[src, target]);
                } else {
                    self.stmt(OpKind::Copy { sync: true }, &[src, target]);
                }
                Ok(())
            }
            (AssignOp::Add, Rv::Tile(src)) => {
                let (sshape, tshape) = (self.shape(src), self.shape(target));
                // A source that broadcasts, a row or a column, is swept as
                // `target + src`.
                if sshape != tshape
                    && sshape.len() == tshape.len()
                    && !tshape.contains(&DYN)
                    && self.elem(src) == self.elem(target)
                    && sshape.iter().zip(&tshape).all(|(&s, &t)| s == t || s == 1)
                {
                    self.check_leaves(&[src], &tshape)?;
                    let tree = Tree::Binary(BinOp::Add, Box::new(Tree::Leaf(0)), Box::new(Tree::Leaf(1)));
                    self.stmt(OpKind::Fused(tree), &[target, src, target]);
                    return Ok(());
                }
                self.check_shapes(&sshape, &tshape, "tile accumulate")?;
                self.stmt(OpKind::Accumulate, &[src, target]);
                Ok(())
            }
        }
    }

    pub(crate) fn check_matmul_shapes(&self, a: &[i64], b: &[i64], out: &[i64]) -> Result<()> {
        if a.len() != 2 || b.len() != 2 || out.len() != 2 {
            bail!("dot expects rank-2 tiles");
        }
        self.check_shapes(&[a[1]], &[b[0]], "dot contraction dim")?;
        self.check_shapes(&[a[0], b[1]], out, "dot result")
    }
}

/// The format of a `<fmt>_qgemm_t` intrinsic name.
pub(crate) fn qgemm_format(callee: &str) -> Option<RawFmt> {
    match Intrinsic::from_name(callee)? {
        Intrinsic::RawQgemm(f) => Some(f),
        _ => None,
    }
}

/// The format of an `<fmt>_qmma_staged_t` name other than IQ1_S's, whose
/// operands differ.
pub(crate) fn staged_qmma_format(callee: &str) -> Option<RawFmt> {
    match Intrinsic::from_name(callee)? {
        Intrinsic::RawQmmaStaged(f) if f != RawFmt::Iq1s => Some(f),
        _ => None,
    }
}
