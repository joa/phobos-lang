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
use super::super::ast::{Dim, Expr, ExprKind, Generated, Kernel, Span, Stmt, Sub, Type};

/// How deep calls may nest before inlining gives up, which also catches
/// recursion.
const MAX_DEPTH: usize = 32;

#[derive(Clone)]
pub(crate) enum Body {
    Expr(Expr),
    Stmts(Vec<Stmt>),
}

#[derive(Clone)]
pub(crate) struct Function {
    pub(crate) name: String,
    pub(crate) params: Vec<String>,
    pub(crate) body: Body,
    /// Its name, where it is defined.
    pub(crate) span: Span,
    /// The library file it comes from, empty for a program's own.
    pub(crate) file: String,
    /// Whether it comes from a library, whose spans point into a file of
    /// its own: an inlined body then takes the call's.
    pub(crate) prelude: bool,
}

/// Inlines every call to a function in `fns`, or else in `prelude`, into
/// each kernel's body. A program's function shadows a prelude function of
/// its name.
pub(super) fn inline(kernels: &mut [Kernel], fns: &[Function], prelude: &[Function]) -> Result<(), String> {
    let mut seen = HashMap::new();
    for f in fns {
        if seen.insert(f.name.as_str(), ()).is_some() {
            return Err(format!("function `{}` is defined twice", f.name));
        }
    }
    let fns: Vec<&Function> = fns.iter().chain(prelude.iter().filter(|f| !seen.contains_key(f.name.as_str()))).collect();
    if fns.is_empty() {
        return Ok(());
    }
    let mut inliner = Inliner { fns: &fns, next: 0 };
    for kernel in kernels {
        let body = std::mem::take(&mut kernel.body);
        kernel.body = inliner.stmts(body, 0).map_err(|e| format!("kernel `{}`: {e}", kernel.name))?;
    }
    Ok(())
}

struct Inliner<'a> {
    fns: &'a [&'a Function],
    next: usize,
}

impl<'a> Inliner<'a> {
    fn find(&self, name: &str) -> Option<&'a Function> {
        self.fns.iter().copied().find(|f| f.name == name)
    }

    fn stmts(&mut self, stmts: Vec<Stmt>, depth: usize) -> Result<Vec<Stmt>, String> {
        let mut out = Vec::new();
        for stmt in stmts {
            if let Stmt::Expr(Expr { kind: ExprKind::Call { callee, args }, span }, _) = &stmt
                && let Some(f) = self.find(callee)
                && let Body::Stmts(_) = f.body
            {
                out.extend(self.splice(callee, args, *span, depth)?);
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
                ty_span,
                value,
                span,
            } => Stmt::Let {
                name,
                name_span,
                ty,
                ty_span,
                value: self.expr(value, depth)?,
                span,
            },
            Stmt::Var {
                name,
                name_span,
                ty,
                ty_span,
                value,
                span,
            } => Stmt::Var {
                name,
                name_span,
                ty,
                ty_span,
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
                    let body = f.subst(&map, &HashMap::new(), span).expr(body)?;
                    *args.last_mut().expect("a reduce has a combine") = body;
                }
                let args = args.into_iter().map(|a| self.expr(a, depth)).collect::<Result<Vec<_>, _>>()?;
                ExprKind::Call { callee, args }
            }
            ExprKind::Call { callee, args } if callee == "generate" => {
                return self.generate(&args, span, depth);
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
                let body = f.subst(&map, &HashMap::new(), span).expr(body)?;
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

    /// `generate(f)`: `f`'s body with its locals and its two parameters
    /// renamed apart, its last statement the value.
    fn generate(&mut self, args: &[Expr], span: crate::diag::Span, depth: usize) -> Result<Expr, String> {
        let [arg] = args else {
            return Err("`generate` takes one function, as `generate(f)`".to_string());
        };
        let Some(f) = arg.as_var().and_then(|name| self.find(name)) else {
            return Err("`generate` takes the name of a function, as `generate(f)`".to_string());
        };
        let Body::Stmts(body) = &f.body else {
            return Err(format!("`{}` must have a statement body ending in its value to be generated", f.name));
        };
        let Some((Stmt::Expr(value, _), lead)) = body.split_last() else {
            return Err(format!("`{}` must end in an expression, the value it generates", f.name));
        };
        let [row, unit] = &f.params[..] else {
            return Err(format!("`{}` takes a row and a unit to be generated", f.name));
        };
        self.deeper(depth, &f.name)?;
        self.next += 1;
        let params = [format!("{row}#{}", self.next), format!("{unit}#{}", self.next)];
        let map = bind(f, &[Expr::var(params[0].clone()), Expr::var(params[1].clone())])?;
        let renames: HashMap<String, String> =
            locals(lead).into_iter().map(|name| (name.clone(), format!("{name}#{}", self.next))).collect();
        let subst = f.subst(&map, &renames, span);
        let lead = subst.block(lead)?;
        let value = subst.expr(value)?;
        let kind = ExprKind::Generate(Box::new(Generated {
            params,
            body: self.stmts(lead, depth + 1)?,
            value: self.expr(value, depth + 1)?,
        }));
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
    fn splice(&mut self, callee: &str, args: &[Expr], span: Span, depth: usize) -> Result<Vec<Stmt>, String> {
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
        let body = f.subst(&map, &renames, span).block(body)?;
        self.stmts(body, depth + 1)
    }
}

impl Function {
    /// The substitution of a call to it at `span`.
    fn subst<'a>(&self, args: &'a HashMap<String, Expr>, renames: &'a HashMap<String, String>, span: Span) -> Subst<'a> {
        Subst {
            args,
            renames,
            at: self.prelude.then_some(span),
        }
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

/// A body's substitution at one call: each parameter replaced by its
/// argument and each local renamed. With `at`, every span the body brings
/// becomes the call's, as for a prelude function, whose spans point into a
/// file other than the caller's; the arguments keep theirs.
struct Subst<'a> {
    args: &'a HashMap<String, Expr>,
    renames: &'a HashMap<String, String>,
    at: Option<Span>,
}

impl Subst<'_> {
    /// An empty span stays empty: it marks what the source does not write.
    fn span(&self, span: Span) -> Span {
        match self.at {
            Some(at) if span.lo != span.hi => at,
            _ => span,
        }
    }

    fn expr(&self, e: &Expr) -> Result<Expr, String> {
        let go = |e: &Expr| self.expr(e);
        let kind = match &e.kind {
            ExprKind::Var(name) => match (self.renames.get(name), self.args.get(name)) {
                (Some(local), _) => ExprKind::Var(local.clone()),
                (None, Some(arg)) => return Ok(arg.clone()),
                (None, None) => ExprKind::Var(name.clone()),
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
                    subs: subs.iter().map(|s| self.sub(s)).collect::<Result<_, _>>()?,
                }
            }
            ExprKind::Generate(g) => ExprKind::Generate(Box::new(Generated {
                params: g.params.clone(),
                body: self.block(&g.body)?,
                value: go(&g.value)?,
            })),
            ExprKind::Call { callee, args: call_args } => ExprKind::Call {
                callee: callee.clone(),
                args: call_args.iter().map(go).collect::<Result<_, _>>()?,
            },
            kind @ (ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_)) => kind.clone(),
        };
        Ok(Expr::new(kind, self.span(e.span)))
    }

    fn sub(&self, s: &Sub) -> Result<Sub, String> {
        let go = |e: &Expr| self.expr(e);
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

    /// A declared type, each dim that names a parameter replaced by its
    /// argument, which must be a name or a literal.
    fn ty(&self, ty: &Option<Type>) -> Result<Option<Type>, String> {
        let dims = |dims: &[Dim]| {
            dims.iter()
                .map(|d| match d {
                    Dim::Sym(name) => match self.args.get(name).map(|a| &a.kind) {
                        None => Ok(d.clone()),
                        Some(ExprKind::Var(arg)) => Ok(Dim::Sym(arg.clone())),
                        Some(ExprKind::Int(n)) => Ok(Dim::Int(*n)),
                        Some(_) => Err(format!("`{name}` sizes a tile, so it must be passed a name or a literal")),
                    },
                    Dim::Int(_) => Ok(d.clone()),
                })
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(match ty {
            Some(Type::Tile(elem, ds)) => Some(Type::Tile(*elem, dims(ds)?)),
            Some(Type::Tensor(elem, ds)) => Some(Type::Tensor(*elem, dims(ds)?)),
            other => other.clone(),
        })
    }

    fn block(&self, body: &[Stmt]) -> Result<Vec<Stmt>, String> {
        body.iter().map(|s| self.stmt(s)).collect()
    }

    fn stmt(&self, stmt: &Stmt) -> Result<Stmt, String> {
        let go = |e: &Expr| self.expr(e);
        let rename = |name: &String| self.renames.get(name).cloned().unwrap_or_else(|| name.clone());
        Ok(match stmt {
            Stmt::Let {
                name,
                name_span,
                ty,
                ty_span,
                value,
                span,
            } => Stmt::Let {
                name: rename(name),
                name_span: self.span(*name_span),
                ty: self.ty(ty)?,
                ty_span: self.span(*ty_span),
                value: go(value)?,
                span: self.span(*span),
            },
            Stmt::Var {
                name,
                name_span,
                ty,
                ty_span,
                value,
                span,
            } => Stmt::Var {
                name: rename(name),
                name_span: self.span(*name_span),
                ty: self.ty(ty)?,
                ty_span: self.span(*ty_span),
                value: value.as_ref().map(go).transpose()?,
                span: self.span(*span),
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
                    span: self.span(*span),
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
                var_span: self.span(*var_span),
                start: go(start)?,
                end: go(end)?,
                step: step.as_ref().map(go).transpose()?,
                unroll: unroll.as_ref().map(go).transpose()?,
                body: self.block(body)?,
                span: self.span(*span),
            },
            Stmt::While { cond, body, span } => Stmt::While {
                cond: go(cond)?,
                body: self.block(body)?,
                span: self.span(*span),
            },
            Stmt::If {
                cond,
                then,
                r#else,
                span,
            } => Stmt::If {
                cond: go(cond)?,
                then: self.block(then)?,
                r#else: r#else.as_deref().map(|b| self.block(b)).transpose()?,
                span: self.span(*span),
            },
            Stmt::Expr(e, span) => Stmt::Expr(go(e)?, self.span(*span)),
        })
    }
}
