use anyhow::{Result, bail};

use super::{Binding, Build, Rv};
use crate::ast::{AssignOp, BinOp, Expr, Scalar as AstScalar, Stmt, Type as AstType};
use crate::ir::{Bounds, Coeff, ForInfo, GemmType, OpKind, Scalar, Type};

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
    a_slice: &'a Expr,
    b_slice: &'a Expr,
    target: &'a Expr,
    alpha: Coeff,
    beta: Coeff,
    alpha_expr: Option<&'a Expr>,
    beta_expr: Option<&'a Expr>,
    consumed: usize,
}

impl Build {
    /// Matches the register matmul and emits it. Returns how many statements
    /// it consumed.
    /// The register matmul: `var acc; for kt {...}; [let prev;] C[...] =
    /// epilogue` becomes a `gemm_init`, a loop carrying the accumulator
    /// whose body slices the two operands and folds them in with
    /// `gemm_dot`, and a `gemm_store`. A statement list this matches never
    /// reaches the rest of the build. The prev_load's own slice is never
    /// built: the store reads it back through its target.
    pub(crate) fn matmul_candidate(&mut self, stmts: &[Stmt]) -> Result<Option<usize>> {
        let Some(m) = self.matmul_matches(stmts) else {
            return Ok(None);
        };
        let &[rows, cols] = &m.shape[..] else {
            bail!("a register matmul with a rank {} accumulator", m.shape.len());
        };
        let ty = Type::Gemm(GemmType {
            acc: m.acc,
            m: rows,
            n: cols,
            k: m.kk,
        });

        let Rv::Scalar(init) = self.emit_expr(m.init)? else {
            bail!("a register matmul's init is not a scalar");
        };
        let seed = self.value(OpKind::GemmInit, &[init], ty.clone());

        let (lo, hi, st, iv_div) = self.loop_bounds(m.start, m.end, m.step)?;
        let body = self.ir.new_block(&[Type::INDEX, ty.clone()]);
        let iv = self.ir.args(body)[0];
        self.ir.set_name(iv, m.kt);
        let acc_arg = self.ir.args(body)[1];
        self.in_block(body, |b| {
            b.push_scope();
            b.bind(m.kt, Binding::Let { value: iv, div: iv_div });
            let folded = (|| {
                let a = b.tile_operand(m.a_slice, "a register matmul's a operand")?;
                let bb = b.tile_operand(m.b_slice, "a register matmul's b operand")?;
                Ok::<_, anyhow::Error>(b.value(OpKind::GemmDot, &[acc_arg, a, bb], ty.clone()))
            })();
            b.pop_scope();
            let folded = folded?;
            b.stmt(OpKind::Yield, &[folded]);
            Ok::<(), anyhow::Error>(())
        })?;
        let op = self.op(
            OpKind::For(ForInfo {
                bounds: Bounds::Dynamic,
                ragged: false,
                carried: 1,
                hoisted: 0,
                pipeline: None,
            }),
            &[lo, hi, st, seed],
            vec![ty],
            vec![body],
        );
        let acc = self.ir.results(op)[0];

        let view = self.tile_operand(m.target, "a register matmul's target")?;
        let mut operands = vec![acc, view];
        for e in [m.alpha_expr, m.beta_expr].into_iter().flatten() {
            let Rv::Scalar(v) = self.emit_expr(e)? else {
                bail!("a register matmul's scaling is not a scalar");
            };
            operands.push(v);
        }
        self.stmt(
            OpKind::GemmStore {
                alpha: m.alpha,
                beta: m.beta,
            },
            &operands,
        );
        Ok(Some(m.consumed))
    }

    /// The statements a register matmul consumes, or None. A port of
    /// `Codegen::matmul_candidate`.
    fn matmul_matches<'a>(&self, stmts: &'a [Stmt]) -> Option<Matched<'a>> {
        let [
            Stmt::Var {
                name: acc,
                ty: Some(AstType::Tile(acc_scalar, dims)),
                value: Some(init),
            },
            Stmt::For {
                var: kt,
                start,
                end,
                step,
                body,
            },
            tail @ ..,
        ] = stmts
        else {
            return None;
        };
        let acc_scalar = *acc_scalar;
        if !matches!(acc_scalar, AstScalar::F32 | AstScalar::F16) {
            return None;
        }
        if kt == acc || !(matches!(init, Expr::Float(_)) || self.const_fold(init).is_some()) {
            return None;
        }
        let (target, epilogue_val, prev_load_let, consumed) = match tail {
            [
                Stmt::Let {
                    name: prev_load_name,
                    ty: None,
                    value: _,
                },
                Stmt::Assign {
                    target,
                    op: AssignOp::Set,
                    value: epi,
                },
                ..,
            ] => (target, epi, Some(prev_load_name.as_str()), 4usize),
            [
                Stmt::Assign {
                    target,
                    op: AssignOp::Set,
                    value: epi,
                },
                ..,
            ] => (target, epi, None, 3usize),
            _ => return None,
        };
        let Expr::Index { base, subs } = target else {
            return None;
        };
        let Expr::Var(out) = base.as_ref() else {
            return None;
        };
        let Some(Binding::Tensor(c)) = self.lookup(out) else {
            return None;
        };
        if !is_f16_or_f32(self.elem(c)) || subs.iter().any(|s| s.uses_name(acc)) {
            return None;
        }
        let out_shape = self.slice_static_shape(target)?;
        if out_shape != self.tile_shape(dims).ok()? {
            return None;
        }

        // The epilogue: acc | alpha*acc | alpha*acc + beta*prev_load. A
        // coefficient is the other factor of its product, or one when the
        // side is the bare name and the epilogue has a prev_load.
        let scaled = |expr: &'a Expr, name: &str| -> Option<Option<&'a Expr>> {
            match expr {
                Expr::Var(n) if n.as_str() == name => Some(None),
                Expr::Binary {
                    op: BinOp::Mul,
                    lhs,
                    rhs,
                } => match (lhs.as_ref(), rhs.as_ref()) {
                    (Expr::Var(n), s) | (s, Expr::Var(n)) if n.as_str() == name => Some(Some(s)),
                    _ => None,
                },
                _ => None,
            }
        };
        let (alpha_expr, prev_load): (Option<&'a Expr>, Option<Option<&'a Expr>>) = match epilogue_val {
            Expr::Var(n) if n.as_str() == acc => (None, None),
            Expr::Binary { op: BinOp::Mul, .. } => (scaled(epilogue_val, acc)?, None),
            Expr::Binary {
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

        let staged = |s: &'a Stmt| -> Option<(&'a str, &'a Expr)> {
            let (name, None, Some(value)) = s.as_decl()? else {
                return None;
            };
            let Expr::Index { base, .. } = value else {
                return None;
            };
            let Expr::Var(src) = &**base else { return None };
            let Some(Binding::Tensor(t)) = self.lookup(src) else {
                return None;
            };
            (is_f16_or_f32(self.elem(t)) && !value.uses_name(acc)).then_some((name, value))
        };
        let [
            sa,
            sb,
            Stmt::Assign {
                target: Expr::Var(t),
                op: AssignOp::Add,
                value: Expr::Call { callee, args },
            },
        ] = &body[..]
        else {
            return None;
        };
        let ((an, av), (bn, bv)) = (staged(sa)?, staged(sb)?);
        if t != acc || callee != "dot" || an == bn {
            return None;
        }
        let [Expr::Var(d0), Expr::Var(d1)] = &args[..] else {
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

        if self.has_wmma() && self.wmma_plan(m, n, ak).is_some() {
            // The tensor-core path takes f16 or f32 throughout.
        } else {
            let a_elem = self.slice_tensor_elem(a_slice);
            let b_elem = self.slice_tensor_elem(b_slice);
            if acc_scalar != AstScalar::F32
                || self.elem(c) != Scalar::F32
                || a_elem != Some(Scalar::F32)
                || b_elem != Some(Scalar::F32)
            {
                return None;
            }
            let (tm, tn) = self.sub_tile(m, n);
            if tm % 4 != 0 || tn % 4 != 0 {
                return None;
            }
            let (tiles_m, tiles_n) = (m / tm, n / tn);
            Self::lane_grid(tiles_m, tiles_n, tm, tn)?;
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
            || self.slice_is_partial(target)
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
            a_slice,
            b_slice,
            target,
            alpha,
            beta,
            alpha_expr,
            beta_expr,
            consumed,
        })
    }
}
