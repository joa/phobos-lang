use std::collections::HashMap;

use anyhow::{Result, bail};

use super::{Binding, Build, DYN};
use crate::ast::{AssignOp, BinOp, Expr, Scalar as AstScalar, Stmt, Type as AstType};
use crate::ir::{Bounds, ForInfo, FragType, OpKind, Scalar, Type, ValueId};
use crate::shape;

/// The validated shape of a fragment-accumulator declaration.
pub(crate) struct FragAccPlan {
    name: String,
    init: f64,
    frags: FragType,
}

#[derive(Default)]
struct FragScan {
    decls: HashMap<String, (Scalar, Vec<i64>)>,
    kk: Option<i64>,
    /// Whether the updates are the int8 scaled form rather than the f16
    /// ones. A plan holds one kind.
    imma: Option<bool>,
}

fn is_f16_or_f32(s: Scalar) -> bool {
    matches!(s, Scalar::F16 | Scalar::F32)
}

/// `f32(dot_t(a, w)) * s1 * s2` as its four names.
/// `acc -= e` as the parser spells it, `acc = acc - e`: the `e`.
fn subtracted<'e>(name: &str, value: &'e Expr) -> Option<&'e Expr> {
    match value {
        Expr::Binary {
            op: BinOp::Sub,
            lhs,
            rhs,
        } if matches!(lhs.as_ref(), Expr::Var(v) if v == name) => Some(rhs),
        _ => None,
    }
}

/// The two names of `r * c`, the outer-product update's operands.
fn outer_parts(value: &Expr) -> Option<[&str; 2]> {
    let Expr::Binary {
        op: BinOp::Mul,
        lhs,
        rhs,
    } = value
    else {
        return None;
    };
    let (Expr::Var(r), Expr::Var(c)) = (lhs.as_ref(), rhs.as_ref()) else {
        return None;
    };
    Some([r, c])
}

fn scaled_dot_parts(value: &Expr) -> Option<[&str; 4]> {
    let Expr::Binary {
        op: BinOp::Mul,
        lhs,
        rhs,
    } = value
    else {
        return None;
    };
    let Expr::Var(s2) = rhs.as_ref() else { return None };
    let Expr::Binary {
        op: BinOp::Mul,
        lhs: cast,
        rhs: s1,
    } = lhs.as_ref()
    else {
        return None;
    };
    let Expr::Var(s1) = s1.as_ref() else { return None };
    let Expr::Call { callee, args } = cast.as_ref() else { return None };
    let [Expr::Call { callee: dot, args }] = &args[..] else { return None };
    if callee != "f32" || dot != "dot_t" {
        return None;
    }
    let [Expr::Var(a), Expr::Var(w)] = &args[..] else { return None };
    Some([a, w, s1, s2])
}

impl Build {
    /// Matches `var acc: tile<f32>[m, n] = <float>` whose every later use is
    /// a fragment-representable form.
    ///
    /// Each admitted form becomes one op yielding a new fragment value, which
    /// the loop carries as an iter arg. The forms are either the f16 ones
    /// (`dot` and row scaling) or the int8 scaled `dot_t`, never both.
    pub(crate) fn frag_acc_candidate(&self, stmts: &[Stmt]) -> Option<FragAccPlan> {
        let [
            Stmt::Var {
                name,
                ty: Some(AstType::Tile(AstScalar::F32, dims)),
                value: Some(Expr::Float(init)),
                ..
            },
            rest @ ..,
        ] = stmts
        else {
            return None;
        };
        let shape = self.tile_shape(dims).ok()?;
        let &[m, n] = &shape[..] else { return None };
        let mut scan = FragScan::default();
        if !self.frag_uses_ok(name, m, n, rest, &mut scan, &[]) {
            return None;
        }
        let imma = scan.imma?;
        let (wm, wn) = if imma {
            if !self.target.has_int8_mma {
                return None;
            }
            shape::imma_plan(m, n, self.cta_threads)?
        } else {
            if !self.has_mma_sync() {
                return None;
            }
            shape::wmma_plan(m, n, scan.kk?, self.cta_threads)?
        };
        Some(FragAccPlan {
            name: name.clone(),
            init: *init,
            frags: FragType { m, n, wm, wn, imma },
        })
    }

    fn frag_uses_ok(
        &self,
        name: &str,
        m: i64,
        n: i64,
        stmts: &[Stmt],
        scan: &mut FragScan,
        ivs: &[&str],
    ) -> bool {
        for stmt in stmts {
            match stmt {
                Stmt::Let { .. } | Stmt::Var { .. } => {
                    let Some((n2, ty, value)) = stmt.as_decl() else {
                        return false;
                    };
                    if n2 == name || value.is_some_and(|v| v.uses_name(name)) {
                        return false;
                    }
                    if value.is_some_and(|v| self.slice_is_partial_within(v, ivs)) {
                        return false;
                    }
                    self.frag_scan_decl(n2, ty, value, scan);
                }
                Stmt::Assign {
                    target: Expr::Var(t),
                    op,
                    value,
                    ..
                } if t == name => {
                    if !self.frag_form_ok(name, m, n, *op, value, scan) {
                        return false;
                    }
                }
                Stmt::Assign {
                    target: target @ Expr::Index { base, subs },
                    op: AssignOp::Set,
                    value: Expr::Var(v),
                    ..
                } if v == name => {
                    let Expr::Var(t) = &**base else { return false };
                    let Some(Binding::Tensor(c)) = self.lookup(t) else {
                        return false;
                    };
                    if !is_f16_or_f32(self.elem(c)) || subs.iter().any(|s| s.uses_name(name)) {
                        return false;
                    }
                    let Some(shape) = self.slice_static_shape(target) else {
                        return false;
                    };
                    if shape != [m, n] {
                        return false;
                    }
                    if self.slice_is_partial_within(target, ivs) {
                        return false;
                    }
                }
                Stmt::Assign { target, value, .. } => {
                    if target.uses_name(name) || value.uses_name(name) {
                        return false;
                    }
                }
                Stmt::For {
                    var,
                    start,
                    end,
                    step,
                    body,
                    ..
                } => {
                    if var == name
                        || start.uses_name(name)
                        || end.uses_name(name)
                        || step.as_ref().is_some_and(|e| e.uses_name(name))
                    {
                        return false;
                    }
                    let mut inner: Vec<&str> = ivs.to_vec();
                    inner.push(var.as_str());
                    if !self.frag_uses_ok(name, m, n, body, scan, &inner) {
                        return false;
                    }
                }
                s @ (Stmt::While { .. } | Stmt::If { .. }) => {
                    if s.uses_name(name) {
                        return false;
                    }
                }
                Stmt::Expr(e, _) => {
                    if e.uses_name(name) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn frag_scan_decl(&self, name: &str, ty: Option<&AstType>, value: Option<&Expr>, scan: &mut FragScan) {
        let entry = match ty {
            Some(AstType::Tile(sc, dims)) => {
                self.tile_shape(dims).ok().map(|s| (Scalar::from_ast(*sc), s))
            }
            None => value.and_then(|v| {
                // A slice of a tile declared earlier in the scanned body,
                // which is not bound yet.
                if let Expr::Index { base, subs } = v
                    && let Expr::Var(b) = base.as_ref()
                    && let Some((elem, shape)) = scan.decls.get(b)
                {
                    return Some((*elem, self.slice_sizes(shape, subs)?));
                }
                Some((self.slice_elem_of(v, true)?, self.slice_shape_of(v, true)?))
            }),
            _ => None,
        };
        if let Some(e) = entry {
            scan.decls.insert(name.to_string(), e);
        }
    }

    fn frag_operand(&self, scan: &FragScan, name: &str) -> Option<(Scalar, Vec<i64>)> {
        if let Some(e) = scan.decls.get(name) {
            return Some(e.clone());
        }
        match self.lookup(name)? {
            Binding::Tile(mv) | Binding::View(mv) if !self.shape(mv).contains(&DYN) => {
                Some((self.elem(mv), self.shape(mv)))
            }
            _ => None,
        }
    }

    /// `acc += r * c` with an f32 `[m, 1]` column and `[1, n]` row in either
    /// order, which only the int8 accumulator takes.
    fn outer_ok(&self, name: &str, m: i64, n: i64, value: &Expr, scan: &mut FragScan) -> bool {
        let Some(parts) = outer_parts(value) else { return false };
        if parts.contains(&name) || scan.imma == Some(false) {
            return false;
        }
        let masked = |x: &str| {
            !scan.decls.contains_key(x)
                && matches!(self.lookup(x), Some(Binding::Tile(v) | Binding::View(v)) if self.is_masked(v))
        };
        if parts.into_iter().any(masked) {
            return false;
        }
        let [Some((Scalar::F32, s1)), Some((Scalar::F32, s2))] = parts.map(|x| self.frag_operand(scan, x)) else {
            return false;
        };
        let (row, col) = (vec![m, 1], vec![1, n]);
        if !((s1 == row && s2 == col) || (s1 == col && s2 == row)) {
            return false;
        }
        scan.imma = Some(true);
        true
    }

    /// `acc += f32(dot_t(a, w)) * s1 * s2` with int8 `a` and `w`, and f32
    /// row and column scales in either order.
    fn scaled_dot_ok(&self, name: &str, m: i64, n: i64, value: &Expr, scan: &mut FragScan) -> bool {
        let Some(parts @ [a, w, s1, s2]) = scaled_dot_parts(value) else {
            return false;
        };
        if parts.contains(&name) || scan.imma == Some(false) {
            return false;
        }
        // The fragments are read straight from the operands, with no bounds
        // guard. A slice declared in the loop was checked with its decl.
        let masked = |x: &str| {
            !scan.decls.contains_key(x)
                && matches!(self.lookup(x), Some(Binding::Tile(v) | Binding::View(v)) if self.is_masked(v))
        };
        if parts.into_iter().any(masked) {
            return false;
        }
        let operand = |x: &str| self.frag_operand(scan, x);
        let (Some((Scalar::I8, ash)), Some((Scalar::I8, wsh))) = (operand(a), operand(w)) else {
            return false;
        };
        let (&[am, ak], &[wn, wk]) = (&ash[..], &wsh[..]) else {
            return false;
        };
        if am != m || wn != n || ak != wk || ak <= 0 || ak % 16 != 0 {
            return false;
        }
        let (Some((Scalar::F32, sh1)), Some((Scalar::F32, sh2))) = (operand(s1), operand(s2)) else {
            return false;
        };
        let (row, col) = (vec![m, 1], vec![1, n]);
        if !((sh1 == row && sh2 == col) || (sh1 == col && sh2 == row)) {
            return false;
        }
        scan.imma = Some(true);
        scan.kk.get_or_insert(ak);
        true
    }

    fn frag_form_ok(
        &self,
        name: &str,
        m: i64,
        n: i64,
        op: AssignOp,
        value: &Expr,
        scan: &mut FragScan,
    ) -> bool {
        match (op, value) {
            (
                AssignOp::Set,
                Expr::Binary {
                    op: BinOp::Mul | BinOp::Div,
                    lhs,
                    rhs,
                },
            ) => {
                let (Expr::Var(l), Expr::Var(col)) = (lhs.as_ref(), rhs.as_ref()) else {
                    return false;
                };
                if l != name || col == name {
                    return false;
                }
                let Some((elem, shape)) = self.frag_operand(scan, col) else {
                    return false;
                };
                if scan.imma == Some(true) {
                    return false;
                }
                scan.imma = Some(false);
                elem == Scalar::F32 && shape == [m, 1]
            }
            (AssignOp::Add, Expr::Binary { .. }) if outer_parts(value).is_some() => {
                self.outer_ok(name, m, n, value, scan)
            }
            (AssignOp::Set, _) if subtracted(name, value).is_some_and(|e| outer_parts(e).is_some()) => {
                self.outer_ok(name, m, n, subtracted(name, value).expect("checked by the guard"), scan)
            }
            (AssignOp::Add, Expr::Binary { .. }) => self.scaled_dot_ok(name, m, n, value, scan),
            (AssignOp::Add, Expr::Call { callee, args }) if callee == "dot" => {
                let [Expr::Var(a), Expr::Var(b)] = &args[..] else {
                    return false;
                };
                if a == name || b == name {
                    return false;
                }
                let (Some((ae, ash)), Some((be, bsh))) =
                    (self.frag_operand(scan, a), self.frag_operand(scan, b))
                else {
                    return false;
                };
                if !is_f16_or_f32(ae) || !is_f16_or_f32(be) || scan.imma == Some(true) {
                    return false;
                }
                scan.imma = Some(false);
                let (&[am, ak], &[bk, bn]) = (&ash[..], &bsh[..]) else {
                    return false;
                };
                if am != m || bn != n || ak != bk || ak % 16 != 0 {
                    return false;
                }
                scan.kk.get_or_insert(ak);
                true
            }
            _ => false,
        }
    }

    /// Seeds the fragment binding for a validated accumulator declaration.
    pub(crate) fn bind_frag_acc(&mut self, plan: &FragAccPlan) {
        let acc = self.value(OpKind::FragInit(plan.init), &[], Type::Frags(plan.frags));
        self.bind(&plan.name, Binding::Frags(acc));
    }

    /// An assignment to a fragment-bound accumulator.
    pub(crate) fn emit_frag_assign(
        &mut self,
        name: &str,
        acc: ValueId,
        op: AssignOp,
        value: &Expr,
    ) -> Result<()> {
        let ty = self.ir.ty(acc).clone();
        match (op, value) {
            (
                AssignOp::Set,
                Expr::Binary {
                    op: bop @ (BinOp::Mul | BinOp::Div),
                    rhs,
                    ..
                },
            ) => {
                let Expr::Var(col) = rhs.as_ref() else {
                    bail!("fragment scale needs a named column operand");
                };
                let Some(Binding::Tile(cmv) | Binding::View(cmv)) = self.lookup(col) else {
                    bail!("fragment scale column '{col}' is not a tile");
                };
                let next = self.value(OpKind::FragScale(*bop), &[acc, cmv], ty);
                self.update_binding(name, Binding::Frags(next));
                Ok(())
            }
            (AssignOp::Add, _) | (AssignOp::Set, _)
                if outer_parts(value).is_some() || subtracted(name, value).is_some_and(|e| outer_parts(e).is_some()) =>
            {
                let negate = op == AssignOp::Set;
                let value = if negate { subtracted(name, value).expect("checked by the guard") } else { value };
                let mut operands = vec![acc];
                for part in outer_parts(value).expect("checked by the guard") {
                    let Some(Binding::Tile(v) | Binding::View(v)) = self.lookup(part) else {
                        bail!("fragment operand '{part}' is not a tile");
                    };
                    operands.push(v);
                }
                // The column first, the row second.
                let column = |v: ValueId| matches!(self.ir.ty(v), Type::Tile(t) if t.static_shape().is_some_and(|s| s.last() == Some(&1)));
                if !column(operands[1]) {
                    operands.swap(1, 2);
                }
                let next = self.value(OpKind::FragOuter { negate }, &operands, ty);
                self.update_binding(name, Binding::Frags(next));
                Ok(())
            }
            (AssignOp::Add, Expr::Binary { .. }) => {
                let Some(parts) = scaled_dot_parts(value) else {
                    bail!("fragment accumulator '{name}' takes f32(dot_t(a, w)) * row * col");
                };
                let mut operands = vec![acc];
                for part in parts {
                    let Some(Binding::Tile(v) | Binding::View(v)) = self.lookup(part) else {
                        bail!("fragment operand '{part}' is not a tile");
                    };
                    operands.push(v);
                }
                let next = self.value(OpKind::FragDotScaled, &operands, ty);
                self.update_binding(name, Binding::Frags(next));
                Ok(())
            }
            (AssignOp::Add, Expr::Call { args, .. }) => {
                let (a, b) = self.dot_operands(args)?;
                let next = self.value(OpKind::FragDot, &[acc, a, b], ty);
                self.update_binding(name, Binding::Frags(next));
                Ok(())
            }
            _ => bail!(
                "fragment accumulator '{name}' only supports scaling, dot-accumulate, and stores"
            ),
        }
    }

    /// The fragment-bound names a loop body assigns.
    pub(crate) fn frag_carried(&self, body: &[Stmt]) -> Vec<String> {
        let mut names = Vec::new();
        self.frag_carried_into(body, &mut names);
        names
    }

    fn frag_carried_into(&self, stmts: &[Stmt], names: &mut Vec<String>) {
        for stmt in stmts {
            match stmt {
                Stmt::Assign {
                    target: Expr::Var(n),
                    ..
                } if !names.contains(n) => {
                    if matches!(self.lookup(n), Some(Binding::Frags(_) | Binding::Regs(_))) {
                        names.push(n.clone());
                    }
                }
                Stmt::For { body, .. } => self.frag_carried_into(body, names),
                _ => {}
            }
        }
    }

    /// A loop threading the fragment accumulators the body assigns as
    /// carried values.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_frag_for(
        &mut self,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        body: &[Stmt],
        names: &[String],
        hoisted: &[ValueId],
    ) -> Result<()> {
        let (lo, hi, st, iv_div) = self.loop_bounds(start, end, step)?;
        // Fragments and register tiles both ride the loop as carried values.
        let mut inits = Vec::new();
        let mut kinds = Vec::new();
        for name in names {
            match self.lookup(name) {
                Some(Binding::Frags(fa)) => inits.push(fa),
                Some(Binding::Regs(r)) => inits.push(r),
                _ => bail!("'{name}' is not register-bound"),
            }
            kinds.push(matches!(self.lookup(name), Some(Binding::Regs(_))));
        }
        let rebind = |regs: bool, v: ValueId| if regs { Binding::Regs(v) } else { Binding::Frags(v) };
        let mut arg_types = vec![Type::INDEX];
        arg_types.extend(inits.iter().map(|&v| self.ir.ty(v).clone()));
        let block = self.ir.new_block(&arg_types);
        let iv = self.ir.args(block)[0];
        self.ir.set_name(iv, var);
        let args: Vec<ValueId> = self.ir.args(block)[1..].to_vec();
        let finals = self.in_block(block, |b| {
            b.push_scope();
            b.bind(var, Binding::Let { value: iv, div: iv_div });
            for ((name, &arg), &regs) in names.iter().zip(&args).zip(&kinds) {
                b.bind(name, rebind(regs, arg));
            }
            let finals = b.emit_stmts(body).and_then(|()| {
                let mut finals = Vec::with_capacity(names.len());
                for name in names {
                    let (Some(Binding::Frags(fa)) | Some(Binding::Regs(fa))) = b.lookup(name) else {
                        bail!("'{name}' lost its register binding in the loop body");
                    };
                    finals.push(fa);
                }
                Ok(finals)
            });
            b.pop_scope();
            let finals = finals?;
            b.stmt(OpKind::Yield, &finals);
            Ok::<Vec<ValueId>, anyhow::Error>(finals)
        })?;
        let _ = finals;
        let mut operands = vec![lo, hi, st];
        operands.extend_from_slice(&inits);
        operands.extend_from_slice(hoisted);
        let types: Vec<Type> = inits.iter().map(|&v| self.ir.ty(v).clone()).collect();
        let op = self.op(
            OpKind::For(ForInfo {
                bounds: Bounds::Dynamic,
                ragged: false,
                carried: inits.len(),
                hoisted: hoisted.len(),
                pipeline: None,
            }),
            &operands,
            types,
            vec![block],
        );
        let results = self.ir.results(op).to_vec();
        for ((name, result), &regs) in names.iter().zip(results).zip(&kinds) {
            self.update_binding(name, rebind(regs, result));
        }
        Ok(())
    }
}
