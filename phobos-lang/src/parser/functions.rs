// Device functions, inlined into their callers before anything past the
// parser sees a kernel.
//
// `fn f(a, b) = e` is an expression: a call is `e` with the arguments
// substituted, so it fuses like any other per-element tree. `fn f(a, b) {
// ... }` is a statement: a call splices the body in, its locals renamed so
// they cannot meet the caller's. A parameter passed a name aliases it, so a
// tile handed in can be written. Any other name in a body resolves where
// the function is called, as a macro's would.

use std::collections::HashMap;

use super::super::ast::lanes::{self, Fold};
use super::super::ast::{Expr, ExprKind, Kernel, Stmt, Sub};

/// How deep calls may nest before inlining gives up, which also catches
/// recursion.
const MAX_DEPTH: usize = 32;

pub(super) enum Body {
    Expr(Expr),
    Stmts(Vec<Stmt>),
}

pub(super) struct Function {
    pub(super) name: String,
    pub(super) params: Vec<String>,
    pub(super) body: Body,
}

/// Inlines every call to a function in `fns` into each kernel's body.
pub(super) fn inline(kernels: &mut [Kernel], fns: &[Function]) -> Result<(), String> {
    if fns.is_empty() {
        return Ok(());
    }
    let mut seen = HashMap::new();
    for f in fns {
        if seen.insert(f.name.as_str(), ()).is_some() {
            return Err(format!("function `{}` is defined twice", f.name));
        }
    }
    let mut inliner = Inliner { fns, next: 0 };
    for kernel in kernels {
        let body = std::mem::take(&mut kernel.body);
        kernel.body = inliner.stmts(body, 0).map_err(|e| format!("kernel `{}`: {e}", kernel.name))?;
    }
    Ok(())
}

struct Inliner<'a> {
    fns: &'a [Function],
    next: usize,
}

impl<'a> Inliner<'a> {
    fn find(&self, name: &str) -> Option<&'a Function> {
        self.fns.iter().find(|f| f.name == name)
    }

    fn stmts(&mut self, stmts: Vec<Stmt>, depth: usize) -> Result<Vec<Stmt>, String> {
        let mut out = Vec::new();
        for stmt in stmts {
            if let Stmt::Expr(Expr { kind: ExprKind::Call { callee, args }, .. }, _) = &stmt
                && let Some(f) = self.find(callee)
                && let Body::Stmts(_) = f.body
            {
                out.extend(self.splice(callee, args, depth)?);
                continue;
            }
            out.push(self.stmt(stmt, depth)?);
        }
        Ok(out)
    }

    fn stmt(&mut self, stmt: Stmt, depth: usize) -> Result<Stmt, String> {
        Ok(match stmt {
            Stmt::Let {
                name,
                name_span,
                ty,
                value,
                span,
            } => Stmt::Let {
                name,
                name_span,
                ty,
                value: self.expr(value, depth)?,
                span,
            },
            Stmt::Var {
                name,
                name_span,
                ty,
                value,
                span,
            } => Stmt::Var {
                name,
                name_span,
                ty,
                value: value.map(|v| self.expr(v, depth)).transpose()?,
                span,
            },
            Stmt::Assign {
                target,
                op,
                value,
                span,
            } => Stmt::Assign {
                target: self.expr(target, depth)?,
                op,
                value: self.expr(value, depth)?,
                span,
            },
            Stmt::For {
                var,
                var_span,
                start,
                end,
                step,
                unroll,
                body,
                span,
            } => Stmt::For {
                var,
                var_span,
                start: self.expr(start, depth)?,
                end: self.expr(end, depth)?,
                step: step.map(|s| self.expr(s, depth)).transpose()?,
                unroll: unroll.map(|u| self.expr(u, depth)).transpose()?,
                body: self.stmts(body, depth)?,
                span,
            },
            Stmt::While { cond, body, span } => Stmt::While {
                cond: self.expr(cond, depth)?,
                body: self.stmts(body, depth)?,
                span,
            },
            Stmt::If {
                cond,
                then,
                r#else,
                span,
            } => Stmt::If {
                cond: self.expr(cond, depth)?,
                then: self.stmts(then, depth)?,
                r#else: r#else.map(|e| self.stmts(e, depth)).transpose()?,
                span,
            },
            Stmt::Expr(e, span) => Stmt::Expr(self.expr(e, depth)?, span),
        })
    }

    /// Rewrites calls to expression functions inside `e`. An inlined body
    /// keeps the spans of the function it came from.
    fn expr(&mut self, e: Expr, depth: usize) -> Result<Expr, String> {
        let span = e.span;
        let kind = match e.kind {
            ExprKind::Call { callee, mut args } if is_reduce(&callee, &args) => {
                // The combine is named, not called: its body over the two
                // operands the butterfly hands it.
                if let Some(name) = args.last().and_then(Expr::as_var)
                    && let Some(f) = self.find(name)
                {
                    let Body::Expr(body) = &f.body else {
                        return Err(format!("`{name}` has a statement body and cannot combine a reduce"));
                    };
                    let map = bind(f, &[Expr::var(lanes::COMBINE_A), Expr::var(lanes::COMBINE_B)])?;
                    let body = subst(body, &map, &HashMap::new())?;
                    *args.last_mut().expect("a reduce has a combine") = body;
                }
                let args = args.into_iter().map(|a| self.expr(a, depth)).collect::<Result<Vec<_>, _>>()?;
                ExprKind::Call { callee, args }
            }
            ExprKind::Call { callee, args } => {
                let args = args.into_iter().map(|a| self.expr(a, depth)).collect::<Result<Vec<_>, _>>()?;
                let Some(f) = self.find(&callee) else {
                    return Ok(Expr::new(ExprKind::Call { callee, args }, span));
                };
                let Body::Expr(body) = &f.body else {
                    return Err(format!("`{callee}` has a statement body and cannot be used as a value"));
                };
                let map = bind(f, &args)?;
                let body = subst(body, &map, &HashMap::new())?;
                self.deeper(depth, &callee)?;
                return self.expr(body, depth + 1);
            }
            ExprKind::Unary { op, rhs } => ExprKind::Unary {
                op,
                rhs: Box::new(self.expr(*rhs, depth)?),
            },
            ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
                op,
                lhs: Box::new(self.expr(*lhs, depth)?),
                rhs: Box::new(self.expr(*rhs, depth)?),
            },
            ExprKind::Index { base, subs } => ExprKind::Index {
                base: Box::new(self.expr(*base, depth)?),
                subs: subs.into_iter().map(|s| self.sub(s, depth)).collect::<Result<_, _>>()?,
            },
            other => other,
        };
        Ok(Expr::new(kind, span))
    }

    fn sub(&mut self, s: Sub, depth: usize) -> Result<Sub, String> {
        Ok(match s {
            Sub::Point(e) => Sub::Point(self.expr(e, depth)?),
            Sub::From(e) => Sub::From(self.expr(e, depth)?),
            Sub::Range { start, end } => Sub::Range {
                start: self.expr(start, depth)?,
                end: self.expr(end, depth)?,
            },
            Sub::Span { start, len } => Sub::Span {
                start: self.expr(start, depth)?,
                len: self.expr(len, depth)?,
            },
            Sub::Full => Sub::Full,
        })
    }

    fn deeper(&self, depth: usize, callee: &str) -> Result<(), String> {
        if depth >= MAX_DEPTH {
            return Err(format!("calls to `{callee}` nest more than {MAX_DEPTH} deep; is it recursive?"));
        }
        Ok(())
    }

    /// The statements of a call to the statement function `callee`.
    fn splice(&mut self, callee: &str, args: &[Expr], depth: usize) -> Result<Vec<Stmt>, String> {
        self.deeper(depth, callee)?;
        let f = self.find(callee).expect("checked by the caller");
        let Body::Stmts(body) = &f.body else {
            unreachable!("checked by the caller");
        };
        let args = args.iter().cloned().map(|a| self.expr(a, depth)).collect::<Result<Vec<_>, _>>()?;
        let map = bind(f, &args)?;
        // Locals become `name#k`, a name no source can spell.
        self.next += 1;
        let mut renames = HashMap::new();
        for name in locals(body) {
            renames.insert(name.clone(), format!("{name}#{}", self.next));
        }
        let body = body
            .iter()
            .map(|s| subst_stmt(s, &map, &renames))
            .collect::<Result<Vec<_>, _>>()?;
        self.stmts(body, depth + 1)
    }
}

fn is_reduce(callee: &str, args: &[Expr]) -> bool {
    lanes::as_collective(callee, args).is_some_and(|c| c.fold == Fold::Reduce)
}

/// Each parameter of `f` against its argument.
fn bind(f: &Function, args: &[Expr]) -> Result<HashMap<String, Expr>, String> {
    if args.len() != f.params.len() {
        return Err(format!(
            "`{}` takes {} argument(s), got {}",
            f.name,
            f.params.len(),
            args.len()
        ));
    }
    Ok(f.params.iter().cloned().zip(args.iter().cloned()).collect())
}

/// Every name a body declares: its `let`s, `var`s and loop variables.
fn locals(body: &[Stmt]) -> Vec<String> {
    let mut out = Vec::new();
    for stmt in body {
        match stmt {
            Stmt::Let { name, .. } | Stmt::Var { name, .. } => out.push(name.clone()),
            Stmt::For { var, body, .. } => {
                out.push(var.clone());
                out.extend(locals(body));
            }
            Stmt::While { body, .. } => out.extend(locals(body)),
            Stmt::If { then, r#else, .. } => {
                out.extend(locals(then));
                out.extend(locals(r#else.as_deref().unwrap_or_default()));
            }
            Stmt::Assign { .. } | Stmt::Expr(..) => {}
        }
    }
    out
}

/// `e` with each parameter replaced by its argument and each local renamed.
fn subst(e: &Expr, args: &HashMap<String, Expr>, renames: &HashMap<String, String>) -> Result<Expr, String> {
    let go = |e: &Expr| subst(e, args, renames);
    let kind = match &e.kind {
        ExprKind::Var(name) => match (renames.get(name), args.get(name)) {
            (Some(local), _) => ExprKind::Var(local.clone()),
            (None, Some(arg)) => return Ok(arg.clone()),
            (None, None) => return Ok(e.clone()),
        },
        ExprKind::Unary { op, rhs } => ExprKind::Unary {
            op: *op,
            rhs: Box::new(go(rhs)?),
        },
        ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
            op: *op,
            lhs: Box::new(go(lhs)?),
            rhs: Box::new(go(rhs)?),
        },
        ExprKind::Index { base, subs } => {
            let base = go(base)?;
            if base.as_var().is_none() {
                return Err("a parameter that is indexed must be passed a name".into());
            }
            ExprKind::Index {
                base: Box::new(base),
                subs: subs.iter().map(|s| subst_sub(s, args, renames)).collect::<Result<_, _>>()?,
            }
        }
        ExprKind::Call { callee, args: call_args } => ExprKind::Call {
            callee: callee.clone(),
            args: call_args.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) => return Ok(e.clone()),
    };
    Ok(Expr::new(kind, e.span))
}

fn subst_sub(s: &Sub, args: &HashMap<String, Expr>, renames: &HashMap<String, String>) -> Result<Sub, String> {
    let go = |e: &Expr| subst(e, args, renames);
    Ok(match s {
        Sub::Point(e) => Sub::Point(go(e)?),
        Sub::From(e) => Sub::From(go(e)?),
        Sub::Range { start, end } => Sub::Range {
            start: go(start)?,
            end: go(end)?,
        },
        Sub::Span { start, len } => Sub::Span {
            start: go(start)?,
            len: go(len)?,
        },
        Sub::Full => Sub::Full,
    })
}

fn subst_stmt(stmt: &Stmt, args: &HashMap<String, Expr>, renames: &HashMap<String, String>) -> Result<Stmt, String> {
    let go = |e: &Expr| subst(e, args, renames);
    let rename = |name: &String| renames.get(name).cloned().unwrap_or_else(|| name.clone());
    let block = |b: &[Stmt]| b.iter().map(|s| subst_stmt(s, args, renames)).collect::<Result<Vec<_>, _>>();
    Ok(match stmt {
        Stmt::Let {
            name,
            name_span,
            ty,
            value,
            span,
        } => Stmt::Let {
            name: rename(name),
            name_span: *name_span,
            ty: ty.clone(),
            value: go(value)?,
            span: *span,
        },
        Stmt::Var {
            name,
            name_span,
            ty,
            value,
            span,
        } => Stmt::Var {
            name: rename(name),
            name_span: *name_span,
            ty: ty.clone(),
            value: value.as_ref().map(go).transpose()?,
            span: *span,
        },
        Stmt::Assign {
            target,
            op,
            value,
            span,
        } => {
            let target = go(target)?;
            let lvalue = match &target.kind {
                ExprKind::Var(_) => true,
                ExprKind::Index { base, .. } => matches!(&base.kind, ExprKind::Var(_)),
                _ => false,
            };
            if !lvalue {
                return Err("a parameter that is assigned must be passed a name".into());
            }
            Stmt::Assign {
                target,
                op: *op,
                value: go(value)?,
                span: *span,
            }
        }
        Stmt::For {
            var,
            var_span,
            start,
            end,
            step,
            unroll,
            body,
            span,
        } => Stmt::For {
            var: rename(var),
            var_span: *var_span,
            start: go(start)?,
            end: go(end)?,
            step: step.as_ref().map(go).transpose()?,
            unroll: unroll.as_ref().map(go).transpose()?,
            body: block(body)?,
            span: *span,
        },
        Stmt::While { cond, body, span } => Stmt::While {
            cond: go(cond)?,
            body: block(body)?,
            span: *span,
        },
        Stmt::If {
            cond,
            then,
            r#else,
            span,
        } => Stmt::If {
            cond: go(cond)?,
            then: block(then)?,
            r#else: r#else.as_deref().map(block).transpose()?,
            span: *span,
        },
        Stmt::Expr(e, span) => Stmt::Expr(go(e)?, *span),
    })
}
