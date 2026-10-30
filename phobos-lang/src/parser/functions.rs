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

use super::size;
use super::super::ast::lanes::{self, Fold};
use super::super::ast::{Dim, Expr, ExprKind, Generated, Kernel, Span, Stmt, Type};

/// How deep calls may nest before inlining gives up, which also catches
/// recursion.
const MAX_DEPTH: usize = 32;

/// The most expression nodes one inlined call may bring. A parameter used
/// twice copies its argument twice, so nested calls grow exponentially with
/// their depth, well within [`MAX_DEPTH`].
const MAX_NODES: usize = 100_000;

/// A function's body: an expression, or statements spliced in at a call.
#[derive(Clone, Debug)]
pub enum Body {
    Expr(Expr),
    Stmts(Vec<Stmt>),
}

/// A `fn` as written.
#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub params: Vec<String>,
    pub body: Body,
    /// Its name, where it is defined.
    pub span: Span,
    /// The library file it comes from, empty for a program's own.
    pub file: String,
    /// Whether it comes from a library, whose spans point into a file of
    /// its own: an inlined body then takes the call's.
    pub prelude: bool,
}

/// Inlines every call to a function in `fns`, or else in `prelude`, into
/// each kernel's body. A program's function shadows a prelude function of
/// its name.
pub(super) fn inline(kernels: &mut [Kernel], fns: &[Function], prelude: &[Function]) -> Result<(), InlineError> {
    let mut seen = HashMap::new();
    for f in fns {
        if seen.insert(f.name.as_str(), ()).is_some() {
            return Err(InlineError {
                message: format!("function `{}` is defined twice", f.name),
                span: Some((f.span, "defined again")),
            });
        }
    }
    let fns: Vec<&Function> = fns.iter().chain(prelude.iter().filter(|f| !seen.contains_key(f.name.as_str()))).collect();
    if fns.is_empty() {
        return Ok(());
    }
    let mut inliner = Inliner { fns: &fns, next: 0 };
    for kernel in kernels {
        let body = std::mem::take(&mut kernel.body);
        kernel.body = inliner.stmts(body, 0).map_err(|e| match e.span {
            Some(_) => e,
            None => format!("kernel `{}`: {}", kernel.name, e.message).into(),
        })?;
    }
    Ok(())
}

/// An inlining error, at the call it arose in, or the definition, where
/// that is known.
pub(super) struct InlineError {
    pub(super) message: String,
    pub(super) span: Option<(Span, &'static str)>,
}

impl From<String> for InlineError {
    fn from(message: String) -> Self {
        InlineError { message, span: None }
    }
}

/// An error at the call `span`.
fn at(span: Span) -> impl FnOnce(String) -> InlineError {
    move |message| InlineError {
        message,
        span: Some((span, "called here")),
    }
}

struct Inliner<'a> {
    fns: &'a [&'a Function],
    next: usize,
}

impl<'a> Inliner<'a> {
    fn find(&self, name: &str) -> Option<&'a Function> {
        self.fns.iter().copied().find(|f| f.name == name)
    }

    fn stmts(&mut self, stmts: Vec<Stmt>, depth: usize) -> Result<Vec<Stmt>, InlineError> {
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

    fn stmt(&mut self, stmt: Stmt, depth: usize) -> Result<Stmt, InlineError> {
        stmt.try_map(self, |s, e| s.expr(e, depth), |s, b| s.stmts(b, depth))
    }

    /// Rewrites calls to expression functions inside `e`. An inlined body
    /// keeps the spans of the function it came from.
    fn expr(&mut self, e: Expr, depth: usize) -> Result<Expr, InlineError> {
        let span = e.span;
        let kind = match e.kind {
            ExprKind::Call { callee, mut args } if is_reduce(&callee, &args) => {
                // The combine is named, not called: its body over the two
                // operands the butterfly hands it.
                if let Some(name) = args.last().and_then(Expr::as_var)
                    && let Some(f) = self.find(name)
                {
                    let Body::Expr(body) = &f.body else {
                        return Err(at(span)(format!("`{name}` has a statement body and cannot combine a reduce")));
                    };
                    let map = bind(f, &[Expr::var(lanes::COMBINE_A), Expr::var(lanes::COMBINE_B)]).map_err(at(span))?;
                    let body = f.subst(&map, &HashMap::new(), span).expr(body.clone()).map_err(at(span))?;
                    within_budget(&f.name, size::expr(&body)).map_err(at(span))?;
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
                    return Err(at(span)(format!("`{callee}` has a statement body and cannot be used as a value")));
                };
                let map = bind(f, &args).map_err(at(span))?;
                let body = f.subst(&map, &HashMap::new(), span).expr(body.clone()).map_err(at(span))?;
                within_budget(&callee, size::expr(&body)).map_err(at(span))?;
                self.deeper(depth, &callee).map_err(at(span))?;
                return self.expr(body, depth + 1);
            }
            other => other.try_map_children(|e| self.expr(e, depth))?,
        };
        Ok(Expr::new(kind, span))
    }

    /// `generate(f)`: `f`'s body with its locals and its two parameters
    /// renamed apart, its last statement the value.
    fn generate(&mut self, args: &[Expr], span: crate::diag::Span, depth: usize) -> Result<Expr, InlineError> {
        let [arg] = args else {
            return Err(at(span)("`generate` takes one function, as `generate(f)`".to_string()));
        };
        let Some(f) = arg.as_var().and_then(|name| self.find(name)) else {
            return Err(at(span)("`generate` takes the name of a function, as `generate(f)`".to_string()));
        };
        let Body::Stmts(body) = &f.body else {
            return Err(at(span)(format!("`{}` must have a statement body ending in its value to be generated", f.name)));
        };
        let Some((Stmt::Expr(value, _), lead)) = body.split_last() else {
            return Err(at(span)(format!("`{}` must end in an expression, the value it generates", f.name)));
        };
        let [row, unit] = &f.params[..] else {
            return Err(at(span)(format!("`{}` takes a row and a unit to be generated", f.name)));
        };
        self.deeper(depth, &f.name).map_err(at(span))?;
        self.next += 1;
        let params = [format!("{row}#{}", self.next), format!("{unit}#{}", self.next)];
        let map = bind(f, &[Expr::var(params[0].clone()), Expr::var(params[1].clone())]).map_err(at(span))?;
        let renames: HashMap<String, String> =
            locals(lead).into_iter().map(|name| (name.clone(), format!("{name}#{}", self.next))).collect();
        let subst = f.subst(&map, &renames, span);
        let lead = subst.block(lead.to_vec()).map_err(at(span))?;
        let value = subst.expr(value.clone()).map_err(at(span))?;
        within_budget(&f.name, size::block(&lead) + size::expr(&value)).map_err(at(span))?;
        let kind = ExprKind::Generate(Box::new(Generated {
            params,
            body: self.stmts(lead, depth + 1)?,
            value: self.expr(value, depth + 1)?,
        }));
        Ok(Expr::new(kind, span))
    }

    fn deeper(&self, depth: usize, callee: &str) -> Result<(), String> {
        if depth >= MAX_DEPTH {
            return Err(format!("calls to `{callee}` nest more than {MAX_DEPTH} deep; is it recursive?"));
        }
        Ok(())
    }

    /// The statements of a call to the statement function `callee`.
    fn splice(&mut self, callee: &str, args: &[Expr], span: Span, depth: usize) -> Result<Vec<Stmt>, InlineError> {
        self.deeper(depth, callee).map_err(at(span))?;
        let f = self.find(callee).expect("checked by the caller");
        let Body::Stmts(body) = &f.body else {
            unreachable!("checked by the caller");
        };
        let args = args.iter().cloned().map(|a| self.expr(a, depth)).collect::<Result<Vec<_>, _>>()?;
        let map = bind(f, &args).map_err(at(span))?;
        // Locals become `name#k`, a name no source can spell.
        self.next += 1;
        let mut renames = HashMap::new();
        for name in locals(body) {
            renames.insert(name.clone(), format!("{name}#{}", self.next));
        }
        let body = f.subst(&map, &renames, span).block(body.clone()).map_err(at(span))?;
        within_budget(callee, size::block(&body)).map_err(at(span))?;
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

/// Fails when one call's expansion has `nodes` nodes, past [`MAX_NODES`].
fn within_budget(callee: &str, nodes: usize) -> Result<(), String> {
    if nodes > MAX_NODES {
        return Err(format!("the call to `{callee}` expands to more than {MAX_NODES} nodes"));
    }
    Ok(())
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
        if let Stmt::Let { name, .. } | Stmt::Var { name, .. } | Stmt::For { var: name, .. } = stmt {
            out.push(name.clone());
        }
        for block in stmt.blocks() {
            out.extend(locals(block));
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

    fn expr(&self, e: Expr) -> Result<Expr, String> {
        let kind = match e.kind {
            ExprKind::Var(name) => match (self.renames.get(&name), self.args.get(&name)) {
                (Some(local), _) => ExprKind::Var(local.clone()),
                (None, Some(arg)) => return Ok(arg.clone()),
                (None, None) => ExprKind::Var(name),
            },
            ExprKind::Index { base, subs } => {
                let base = self.expr(*base)?;
                if base.as_var().is_none() {
                    return Err("a parameter that is indexed must be passed a name".into());
                }
                let subs = subs.into_iter().map(|s| s.try_map(|e| self.expr(e))).collect::<Result<_, _>>()?;
                ExprKind::Index { base: Box::new(base), subs }
            }
            ExprKind::Generate(g) => {
                let Generated { params, body, value } = *g;
                let (body, value) = (self.block(body)?, self.expr(value)?);
                ExprKind::Generate(Box::new(Generated { params, body, value }))
            }
            kind => kind.try_map_children(|e| self.expr(e))?,
        };
        Ok(Expr::new(kind, self.span(e.span)))
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

    fn block(&self, body: Vec<Stmt>) -> Result<Vec<Stmt>, String> {
        body.into_iter().map(|s| self.stmt(s)).collect()
    }

    fn stmt(&self, stmt: Stmt) -> Result<Stmt, String> {
        let mut stmt = match stmt {
            // The target before the value: one that is not a name fails
            // first.
            Stmt::Assign { target, op, value, span } => {
                let target = self.expr(target)?;
                let place = match &target.kind {
                    ExprKind::Var(_) => true,
                    ExprKind::Index { base, .. } => base.as_var().is_some(),
                    _ => false,
                };
                if !place {
                    return Err("a parameter that is assigned must be passed a name".into());
                }
                return Ok(Stmt::Assign { target, op, value: self.expr(value)?, span: self.span(span) });
            }
            stmt => stmt,
        };
        let rename = |name: &mut String| {
            if let Some(local) = self.renames.get(name) {
                *name = local.clone();
            }
        };
        match &mut stmt {
            Stmt::Let { name, name_span, ty, ty_span, span, .. } | Stmt::Var { name, name_span, ty, ty_span, span, .. } => {
                rename(name);
                *name_span = self.span(*name_span);
                *ty = self.ty(ty)?;
                *ty_span = self.span(*ty_span);
                *span = self.span(*span);
            }
            Stmt::For { var, var_span, span, .. } => {
                rename(var);
                *var_span = self.span(*var_span);
                *span = self.span(*span);
            }
            Stmt::Assign { span, .. } | Stmt::While { span, .. } | Stmt::If { span, .. } | Stmt::Expr(_, span) => {
                *span = self.span(*span);
            }
        }
        stmt.try_map(&mut (), |_, e| self.expr(e), |_, b| self.block(b))
    }
}
