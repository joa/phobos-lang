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
    decls: HashMap<String, (bool, Vec<i64>)>,
    kk: Option<i64>,
}

fn is_f16_or_f32(s: Scalar) -> bool {
    matches!(s, Scalar::F16 | Scalar::F32)
}

impl Build {
    /// Matches `var acc: tile<f32>[m, n] = <float>` whose every later use is
    /// a fragment-representable form.
    /// A fragment accumulator, if the statements declare one: each admitted
    /// form becomes one op yielding a new fragment value, which the loop
    /// carries as an iter arg.
    pub(crate) fn frag_acc_candidate(&self, stmts: &[Stmt]) -> Option<FragAccPlan> {
        let [
            Stmt::Var {
                name,
                ty: Some(AstType::Tile(AstScalar::F32, dims)),
                value: Some(Expr::Float(init)),
            },
            rest @ ..,
        ] = stmts
        else {
            return None;
        };
        if !self.has_mma_sync() {
            return None;
        }
        let shape = self.tile_shape(dims).ok()?;
        let &[m, n] = &shape[..] else { return None };
        let mut scan = FragScan::default();
        if !self.frag_uses_ok(name, m, n, rest, &mut scan, &[]) {
            return None;
        }
        let (wm, wn) = shape::wmma_plan(m, n, scan.kk?, self.cta_threads)?;
        Some(FragAccPlan {
            name: name.clone(),
            init: *init,
            frags: FragType { m, n, wm, wn },
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
                    let Some((n2, ty, Some(value))) = stmt.as_decl() else {
                        return false;
                    };
                    if n2 == name || value.uses_name(name) {
                        return false;
                    }
                    if self.slice_is_partial_within(value, ivs) {
                        return false;
                    }
                    self.frag_scan_decl(n2, ty, value, scan);
                }
                Stmt::Assign {
                    target: Expr::Var(t),
                    op,
                    value,
                } if t == name => {
                    if !self.frag_form_ok(name, m, n, *op, value, scan) {
                        return false;
                    }
                }
                Stmt::Assign {
                    target: target @ Expr::Index { base, subs },
                    op: AssignOp::Set,
                    value: Expr::Var(v),
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
                Stmt::Expr(e) => {
                    if e.uses_name(name) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn frag_scan_decl(&self, name: &str, ty: Option<&AstType>, value: &Expr, scan: &mut FragScan) {
        let entry = match ty {
            Some(AstType::Tile(sc @ (AstScalar::F16 | AstScalar::F32), dims)) => {
                self.tile_shape(dims).ok().map(|s| (*sc == AstScalar::F32, s))
            }
            None => match (self.slice_static_shape(value), self.slice_tensor_elem(value)) {
                (Some(s), Some(e)) if is_f16_or_f32(e) => Some((e == Scalar::F32, s)),
                _ => None,
            },
            _ => None,
        };
        if let Some(e) = entry {
            scan.decls.insert(name.to_string(), e);
        }
    }

    fn frag_operand(&self, scan: &FragScan, name: &str) -> Option<(bool, Vec<i64>)> {
        if let Some(e) = scan.decls.get(name) {
            return Some(e.clone());
        }
        match self.lookup(name)? {
            Binding::Tile(mv) | Binding::View(mv)
                if !self.shape(mv).contains(&DYN) && is_f16_or_f32(self.elem(mv)) =>
            {
                Some((self.elem(mv) == Scalar::F32, self.shape(mv)))
            }
            _ => None,
        }
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
                let Some((is_f32, shape)) = self.frag_operand(scan, col) else {
                    return false;
                };
                is_f32 && shape == [m, 1]
            }
            (AssignOp::Add, Expr::Call { callee, args }) if callee == "dot" => {
                let [Expr::Var(a), Expr::Var(b)] = &args[..] else {
                    return false;
                };
                if a == name || b == name {
                    return false;
                }
                let (Some((_, ash)), Some((_, bsh))) =
                    (self.frag_operand(scan, a), self.frag_operand(scan, b))
                else {
                    return false;
                };
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
                    if matches!(self.lookup(n), Some(Binding::Frags(_))) {
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
        let mut inits = Vec::new();
        for name in names {
            let Some(Binding::Frags(fa)) = self.lookup(name) else {
                bail!("'{name}' is not fragment-bound");
            };
            inits.push(fa);
        }
        let mut arg_types = vec![Type::INDEX];
        arg_types.extend(inits.iter().map(|&v| self.ir.ty(v).clone()));
        let block = self.ir.new_block(&arg_types);
        let iv = self.ir.args(block)[0];
        self.ir.set_name(iv, var);
        let args: Vec<ValueId> = self.ir.args(block)[1..].to_vec();
        let finals = self.in_block(block, |b| {
            b.push_scope();
            b.bind(var, Binding::Let { value: iv, div: iv_div });
            for (name, &arg) in names.iter().zip(&args) {
                b.bind(name, Binding::Frags(arg));
            }
            let finals = b.emit_stmts(body).and_then(|()| {
                let mut finals = Vec::with_capacity(names.len());
                for name in names {
                    let Some(Binding::Frags(fa)) = b.lookup(name) else {
                        bail!("'{name}' lost its fragment binding in the loop body");
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
        for (name, result) in names.iter().zip(results) {
            self.update_binding(name, Binding::Frags(result));
        }
        Ok(())
    }
}
