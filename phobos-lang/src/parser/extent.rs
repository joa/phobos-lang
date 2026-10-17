// `extent(x, d)` of a name whose type the source writes, a tensor parameter
// or a typed tile, resolved to the dimension the type names, once functions
// are inlined. A library loop over `extent(A, 1)` is then the loop over `K`
// the kernel would have written, which is what the code generator's pattern
// matching sees. The IR builder resolves every other `extent` from the
// values it has built.

use std::collections::{HashMap, HashSet};

use super::super::ast::{Dim, Expr, ExprKind, Kernel, Stmt, Sub, Type};

pub(super) const EXTENT: &str = "extent";

#[derive(Clone, Default)]
struct Scope {
    /// The declared dims of each name in scope with a written shape.
    dims: HashMap<String, Vec<Dim>>,
    /// Every scalar a body binds, which a dim's symbol must not mean.
    locals: HashSet<String>,
}

impl Scope {
    fn bind(&mut self, name: &str, ty: Option<&Type>) {
        match ty {
            Some(Type::Tensor(_, dims) | Type::Tile(_, dims)) => {
                self.dims.insert(name.to_string(), dims.clone());
            }
            _ => {
                self.dims.remove(name);
            }
        }
        self.locals.insert(name.to_string());
    }

    /// The dim `extent(args)` names, if the source writes it.
    fn resolve(&self, args: &[Expr]) -> Option<Expr> {
        let [x, d] = args else { return None };
        let dims = self.dims.get(x.as_var()?)?;
        let dim = dims.get(usize::try_from(d.as_int()?).ok()?)?;
        Some(match dim {
            Dim::Int(n) => Expr::int(*n),
            Dim::Sym(s) if !self.locals.contains(s) => Expr::var(s.clone()),
            Dim::Sym(_) => return None,
        })
    }
}

pub(super) fn resolve(kernel: &mut Kernel) {
    let mut scope = Scope::default();
    for p in &kernel.params {
        scope.bind(&p.name, Some(&p.ty));
    }
    block(&mut kernel.body, scope);
}

fn block(stmts: &mut [Stmt], mut scope: Scope) {
    for s in stmts {
        stmt(s, &mut scope);
    }
}

fn stmt(s: &mut Stmt, scope: &mut Scope) {
    match s {
        Stmt::Let { name, ty, value, .. } => {
            expr(value, scope);
            scope.bind(name, ty.as_ref());
        }
        Stmt::Var { name, ty, value, .. } => {
            if let Some(value) = value {
                expr(value, scope);
            }
            scope.bind(name, ty.as_ref());
        }
        Stmt::Assign { target, value, .. } => {
            expr(target, scope);
            expr(value, scope);
        }
        Stmt::For {
            var,
            start,
            end,
            step,
            unroll,
            body,
            ..
        } => {
            for e in [Some(start), Some(end), step.as_mut(), unroll.as_mut()].into_iter().flatten() {
                expr(e, scope);
            }
            let mut inner = scope.clone();
            inner.bind(var, None);
            block(body, inner);
        }
        Stmt::While { cond, body, .. } => {
            expr(cond, scope);
            block(body, scope.clone());
        }
        Stmt::If { cond, then, r#else, .. } => {
            expr(cond, scope);
            block(then, scope.clone());
            if let Some(r#else) = r#else {
                block(r#else, scope.clone());
            }
        }
        Stmt::Expr(e, _) => expr(e, scope),
    }
}

fn expr(e: &mut Expr, scope: &Scope) {
    if let ExprKind::Call { callee, args } = &e.kind
        && callee == EXTENT
        && let Some(dim) = scope.resolve(args)
    {
        *e = dim.at(e.span);
        return;
    }
    match &mut e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Var(_) => {}
        ExprKind::Unary { rhs, .. } => expr(rhs, scope),
        ExprKind::Binary { lhs, rhs, .. } => {
            expr(lhs, scope);
            expr(rhs, scope);
        }
        ExprKind::Index { base, subs } => {
            expr(base, scope);
            for sub in subs {
                match sub {
                    Sub::Point(e) | Sub::From(e) => expr(e, scope),
                    Sub::Range { start: a, end: b } | Sub::Span { start: a, len: b } => {
                        expr(a, scope);
                        expr(b, scope);
                    }
                    Sub::Full => {}
                }
            }
        }
        ExprKind::Call { args, .. } => args.iter_mut().for_each(|a| expr(a, scope)),
        ExprKind::Generate(g) => {
            let mut inner = scope.clone();
            for p in &g.params {
                inner.bind(p, None);
            }
            block(&mut g.body, inner.clone());
            for s in &g.body {
                if let Stmt::Let { name, .. } | Stmt::Var { name, .. } = s {
                    inner.bind(name, None);
                }
            }
            expr(&mut g.value, &inner);
        }
    }
}
