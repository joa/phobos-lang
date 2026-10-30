use anyhow::{Result, bail};

use super::{Binding, Build, Rv};
use crate::ast::{AssignOp, BinOp, Expr, ExprKind, Scalar as AstScalar, Span, Stmt, Type as AstType};
use crate::ir::{Bounds, Coeff, ForInfo, GemmType, OpKind, Scalar, Type};
use crate::shape;

fn is_f16_or_f32(s: Scalar) -> bool {
    matches!(s, Scalar::F16 | Scalar::F32)
}

/// What the match found, by reference into the statements.
struct Matched<'a> {
    acc: Scalar,
    shape: Vec<i64>,
    kk: i64,
    init: &'a Expr,
    kt: &'a str,
    start: &'a Expr,
    end: &'a Expr,
    step: Option<&'a Expr>,
    /// The k-loop's `for<unroll n>`, 1 without one.
    unroll: i64,
    a_slice: &'a Expr,
    b_slice: &'a Expr,
    epilogue: Epilogue<'a>,
    consumed: usize,
}

/// One store of the finished accumulator, `C[...] = alpha * acc [+ beta *
/// prev]`.
struct Arm<'a> {
    /// The store's statement.
    span: Span,
    target: &'a Expr,
    /// The target tensor's element type.
    elem: Scalar,
    alpha: Coeff,
    beta: Coeff,
    alpha_expr: Option<&'a Expr>,
    beta_expr: Option<&'a Expr>,
}

/// What follows the loop: one store, or a branch on a scalar between two,
/// so a kernel can say `if beta == 0.0` and leave C unread.
enum Epilogue<'a> {
    One(Arm<'a>),
    Branch { cond: &'a Expr, then: Arm<'a>, els: Arm<'a> },
}

impl<'a> Epilogue<'a> {
    fn arms(&self) -> impl Iterator<Item = &Arm<'a>> {
        match self {
            Epilogue::One(arm) => std::slice::from_ref(arm).iter().chain(None),
            Epilogue::Branch { then, els, .. } => std::slice::from_ref(then).iter().chain(Some(els)),
        }
    }
}

impl Build {
    /// Matches the register matmul and emits it. Returns how many statements
    /// it consumed.
    ///
    /// `var acc; for kt {...}; [let prev;] C[...] = epilogue` becomes a
    /// `gemm_init`, a loop, and a `gemm_store`. The loop carries the
    /// accumulator, slices the two operands, and folds them in with
    /// `gemm_dot`. Matched statements never reach the rest of the build.
    /// The prev_load's own slice is never built, since the store reads it
    /// back through its target.
    /// The `GemmStore` of one epilogue arm.
    fn gemm_store(&mut self, acc: crate::ir::ValueId, arm: &Arm<'_>) -> Result<()> {
        self.spanned(arm.span, |b| {
            let view = b.tile_operand(arm.target, "a register matmul's target")?;
            let mut operands = vec![acc, view];
            for e in [arm.alpha_expr, arm.beta_expr].into_iter().flatten() {
                let Rv::Scalar(v) = b.emit_expr(e)? else {
                    bail!("a register matmul's scaling is not a scalar");
                };
                operands.push(v);
            }
            b.stmt(
                OpKind::GemmStore {
                    alpha: arm.alpha,
                    beta: arm.beta,
                },
                &operands,
            );
            Ok(())
        })
    }

    pub(crate) fn matmul_candidate(&mut self, stmts: &[Stmt]) -> Result<Option<usize>> {
        let Some(m) = self.matmul_matches(stmts) else {
            return Ok(None);
        };
        let &[rows, cols] = &m.shape[..] else {
            bail!("a register matmul with a rank {} accumulator", m.shape.len());
        };
        let ty = Type::Dist(
            GemmType {
                acc: m.acc,
                m: rows,
                n: cols,
                k: m.kk,
            }
            .dist(),
        );

        let Rv::Scalar(init) = self.emit_expr(m.init)? else {
            bail!("a register matmul's init is not a scalar");
        };
        let seed = self.value(OpKind::GemmInit, &[init], ty.clone());

        // Each op takes the line of the statement it stands for.
        let Stmt::For {
            body: loop_body, ..
        } = &stmts[1]
        else {
            bail!("a register matmul without its loop");
        };
        let spans: Vec<Span> = loop_body.iter().map(Stmt::span).collect();
        let op = self.spanned(stmts[1].span(), |b| {
            let (lo, hi, st, iv_div) = b.loop_bounds(m.start, m.end, m.step)?;
            let body = b.ir.new_block(&[Type::INDEX, ty.clone()]);
            let iv = b.ir.args(body)[0];
            b.ir.set_name(iv, m.kt);
            let acc_arg = b.ir.args(body)[1];
            b.in_block(body, |b| {
                b.push_scope();
                b.bind(
                    m.kt,
                    Binding::Let {
                        value: iv,
                        div: iv_div,
                    },
                );
                let folded = (|| {
                    let a = b.spanned(spans[0], |b| {
                        b.tile_operand(m.a_slice, "a register matmul's a operand")
                    })?;
                    let bb = b.spanned(spans[1], |b| {
                        b.tile_operand(m.b_slice, "a register matmul's b operand")
                    })?;
                    b.spanned(spans[2], |b| {
                        Ok(b.value(OpKind::GemmDot, &[acc_arg, a, bb], ty.clone()))
                    })
                })();
                b.pop_scope();
                let folded = folded?;
                b.stmt(OpKind::Yield, &[folded]);
                Ok::<(), anyhow::Error>(())
            })?;
            Ok(b.op(
                OpKind::For(ForInfo {
                    bounds: Bounds::Dynamic,
                    ragged: false,
                    carried: 1,
                    hoisted: 0,
                    pipeline: None,
                    unroll: m.unroll,
                }),
                &[lo, hi, st, seed],
                vec![ty],
                vec![body],
            ))
        })?;
        let acc = self.ir.results(op)[0];

        match &m.epilogue {
            Epilogue::One(arm) => self.gemm_store(acc, arm)?,
            Epilogue::Branch { cond, then, els } => self.spanned(stmts[m.consumed - 1].span(), |b| {
                let c = b.emit_scalar(cond)?;
                let c = b.expect_bool(c, "if condition")?;
                let mut blocks = Vec::with_capacity(2);
                for arm in [then, els] {
                    let block = b.ir.new_block(&[]);
                    b.in_block(block, |b| {
                        b.gemm_store(acc, arm)?;
                        b.stmt(OpKind::Yield, &[]);
                        Ok::<(), anyhow::Error>(())
                    })?;
                    blocks.push(block);
                }
                b.op(OpKind::If, &[c], Vec::new(), blocks);
                Ok(())
            })?,
        }
        Ok(Some(m.consumed))
    }

    /// One epilogue store at the head of `stmts`, `[let prev;] C[...] =
    /// alpha * acc [+ beta * prev]`, and how many statements it takes.
    fn gemm_arm<'a>(&self, stmts: &'a [Stmt], acc: &str, dims: &[crate::ast::Dim]) -> Option<(Arm<'a>, usize)> {
        let (target, epilogue_val, prev_load_let, consumed) = match stmts {
            [
                Stmt::Let {
                    name: prev_load_name,
                    ty: None,
                    ..
                },
                Stmt::Assign {
                    target,
                    op: AssignOp::Set,
                    value: epi,
                    ..
                },
                ..,
            ] => (target, epi, Some(prev_load_name.as_str()), 2usize),
            [
                Stmt::Assign {
                    target,
                    op: AssignOp::Set,
                    value: epi,
                    ..
                },
                ..,
            ] => (target, epi, None, 1usize),
            _ => return None,
        };
        let ExprKind::Index { base, subs } = &target.kind else {
            return None;
        };
        let ExprKind::Var(out) = &base.kind else {
            return None;
        };
        let Some(Binding::Tensor(c)) = self.lookup(out) else {
            return None;
        };
        if !is_f16_or_f32(self.elem(c)) || subs.iter().any(|s| s.uses_name(acc)) {
            return None;
        }
        let out_shape = self.slice_static_shape(target)?;
        if out_shape != self.tile_shape(dims).ok()? || self.slice_is_partial(target) {
            return None;
        }

        // The epilogue: acc | alpha*acc | alpha*acc + beta*prev_load. A
        // coefficient is the other factor of its product, or one when the
        // side is the bare name and the epilogue has a prev_load.
        let scaled = |expr: &'a Expr, name: &str| -> Option<Option<&'a Expr>> {
            match &expr.kind {
                ExprKind::Var(n) if n.as_str() == name => Some(None),
                ExprKind::Binary {
                    op: BinOp::Mul,
                    lhs,
                    rhs,
                } if lhs.as_var() == Some(name) => Some(Some(rhs.as_ref())),
                ExprKind::Binary {
                    op: BinOp::Mul,
                    lhs,
                    rhs,
                } if rhs.as_var() == Some(name) => Some(Some(lhs.as_ref())),
                _ => None,
            }
        };
        let (alpha_expr, prev_load): (Option<&'a Expr>, Option<Option<&'a Expr>>) = match &epilogue_val.kind {
            ExprKind::Var(n) if n.as_str() == acc => (None, None),
            ExprKind::Binary { op: BinOp::Mul, .. } => (scaled(epilogue_val, acc)?, None),
            ExprKind::Binary {
                op: BinOp::Add,
                lhs,
                rhs,
            } => {
                let prev_load_n = prev_load_let?;
                let try_split = |acc_side: &'a Expr, prev_side: &'a Expr| {
                    let alpha = scaled(acc_side, acc)?;
                    let beta = scaled(prev_side, prev_load_n)?;
                    Some((alpha, Some(beta)))
                };
                try_split(lhs, rhs)
                    .or_else(|| try_split(rhs, lhs))
                    .unwrap_or((None, None))
            }
            _ => return None,
        };
        if prev_load_let.is_some() != prev_load.is_some() {
            return None;
        }
        let coeff = |e: Option<&Expr>| if e.is_some() { Coeff::Given } else { Coeff::One };
        let (alpha, beta, beta_expr) = match prev_load {
            Some(beta_expr) => (coeff(alpha_expr), coeff(beta_expr), beta_expr),
            None => (
                if alpha_expr.is_some() { Coeff::Given } else { Coeff::Absent },
                Coeff::Absent,
                None,
            ),
        };

        Some((
            Arm {
                span: stmts[consumed - 1].span(),
                target,
                elem: self.elem(c),
                alpha,
                beta,
                alpha_expr,
                beta_expr,
            },
            consumed,
        ))
    }

    /// The statements a register matmul consumes, or None.
    fn matmul_matches<'a>(&self, stmts: &'a [Stmt]) -> Option<Matched<'a>> {
        let [
            Stmt::Var {
                name: acc,
                ty: Some(AstType::Tile(acc_scalar, dims)),
                value: Some(init),
                ..
            },
            Stmt::For {
                var: kt,
                start,
                end,
                step,
                unroll,
                body,
                ..
            },
            tail @ ..,
        ] = stmts
        else {
            return None;
        };
        let unroll = match unroll {
            None => 1,
            Some(e) => self.const_fold(e).filter(|&n| n >= 1)?,
        };
        let acc_scalar = *acc_scalar;
        if !matches!(acc_scalar, AstScalar::F32 | AstScalar::F16) {
            return None;
        }
        if kt == acc || !(matches!(&init.kind, ExprKind::Float(_)) || self.const_fold(init).is_some()) {
            return None;
        }
        let (epilogue, consumed) = if let Some((arm, n)) = self.gemm_arm(tail, acc, dims) {
            (Epilogue::One(arm), 2 + n)
        } else if let [Stmt::If { cond, then, r#else: Some(els), .. }, ..] = tail
            && !cond.uses_name(acc)
            && let Some((then_arm, n)) = self.gemm_arm(then, acc, dims)
            && n == then.len()
            && let Some((else_arm, n)) = self.gemm_arm(els, acc, dims)
            && n == els.len()
        {
            (Epilogue::Branch { cond, then: then_arm, els: else_arm }, 3)
        } else {
            return None;
        };

        let staged = |s: &'a Stmt| -> Option<(&'a str, &'a Expr)> {
            let (name, None, Some(value)) = s.as_decl()? else {
                return None;
            };
            let ExprKind::Index { base, .. } = &value.kind else {
                return None;
            };
            let ExprKind::Var(src) = &base.kind else { return None };
            let Some(Binding::Tensor(t)) = self.lookup(src) else {
                return None;
            };
            (is_f16_or_f32(self.elem(t)) && !value.uses_name(acc)).then_some((name, value))
        };
        let [
            sa,
            sb,
            Stmt::Assign {
                target: Expr { kind: ExprKind::Var(t), .. },
                op: AssignOp::Add,
                value: Expr { kind: ExprKind::Call { callee, args }, .. },
                ..
            },
        ] = &body[..]
        else {
            return None;
        };
        let ((an, av), (bn, bv)) = (staged(sa)?, staged(sb)?);
        if t != acc || callee != "dot" || an == bn {
            return None;
        }
        let [Expr { kind: ExprKind::Var(d0), .. }, Expr { kind: ExprKind::Var(d1), .. }] = &args[..] else {
            return None;
        };
        let (a_slice, b_slice) = if d0 == an && d1 == bn {
            (av, bv)
        } else if d0 == bn && d1 == an {
            (bv, av)
        } else {
            return None;
        };

        let acc_shape = self.tile_shape(dims).ok()?;
        let a_shape = self.slice_static_shape(a_slice)?;
        let b_shape = self.slice_static_shape(b_slice)?;
        let (&[m, n], &[am, ak], &[bk, bn2]) = (&acc_shape[..], &a_shape[..], &b_shape[..]) else {
            return None;
        };
        if am != m || bn2 != n || ak != bk {
            return None;
        }

        if self.has_wmma() && shape::wmma_plan(m, n, ak, self.cta_threads).is_some() {
            // The tensor-core path takes f16 or f32 throughout.
        } else {
            let a_elem = self.slice_tensor_elem(a_slice);
            let b_elem = self.slice_tensor_elem(b_slice);
            if acc_scalar != AstScalar::F32
                || epilogue.arms().any(|arm| arm.elem != Scalar::F32)
                || a_elem != Some(Scalar::F32)
                || b_elem != Some(Scalar::F32)
            {
                return None;
            }
            let (tm, tn) = shape::sub_tile(m, n, self.cta_threads);
            if tm % 4 != 0 || tn % 4 != 0 {
                return None;
            }
            let (tiles_m, tiles_n) = (m / tm, n / tn);
            shape::lane_grid(tiles_m, tiles_n, tm, tn)?;
            if tiles_m * tiles_n > self.cta_threads {
                return None;
            }
        }

        let rest = &stmts[consumed..];
        if rest.iter().any(|s| s.uses_name(acc)) {
            return None;
        }
        if self.slice_is_partial_within(a_slice, &[kt.as_str()])
            || self.slice_is_partial_within(b_slice, &[kt.as_str()])
        {
            return None;
        }
        Some(Matched {
            acc: Scalar::from_ast(acc_scalar),
            shape: acc_shape,
            kk: ak,
            init,
            kt,
            start,
            end,
            step: step.as_ref(),
            unroll,
            a_slice,
            b_slice,
            epilogue,
            consumed,
        })
    }
}
