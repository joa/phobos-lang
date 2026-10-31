// What each node of the tree holds, in source order, and the walks built
// on it. A generated tile's statements and value are a scope of their own,
// so they are not among an expression's children: a walk enters them as
// it must.

use super::*;

impl Sub {
    /// The expressions the subscript writes, in order.
    pub fn exprs(&self) -> impl Iterator<Item = &Expr> {
        let (first, second) = match self {
            Sub::Point(e) | Sub::From(e) => (Some(e), None),
            Sub::Range { start, end: other } | Sub::Span { start, len: other } => (Some(start), Some(other)),
            Sub::Full => (None, None),
        };
        first.into_iter().chain(second)
    }

    /// [`Self::exprs`], to change in place.
    pub fn exprs_mut(&mut self) -> impl Iterator<Item = &mut Expr> {
        let (first, second) = match self {
            Sub::Point(e) | Sub::From(e) => (Some(e), None),
            Sub::Range { start, end: other } | Sub::Span { start, len: other } => (Some(start), Some(other)),
            Sub::Full => (None, None),
        };
        first.into_iter().chain(second)
    }

    /// `self` with `f` applied to each expression it writes, in order.
    pub fn try_map<E>(self, mut f: impl FnMut(Expr) -> Result<Expr, E>) -> Result<Sub, E> {
        Ok(match self {
            Sub::Point(e) => Sub::Point(f(e)?),
            Sub::From(e) => Sub::From(f(e)?),
            Sub::Range { start, end } => Sub::Range { start: f(start)?, end: f(end)? },
            Sub::Span { start, len } => Sub::Span { start: f(start)?, len: f(len)? },
            Sub::Full => Sub::Full,
        })
    }

    /// Whether any expression in this subscript uses the given name.
    pub fn uses_name(&self, name: &str) -> bool {
        self.exprs().any(|e| e.uses_name(name))
    }
}

impl ExprKind {
    /// The expressions directly under this one, in order.
    pub fn children(&self) -> impl Iterator<Item = &Expr> {
        let (operands, args, subs): ([Option<&Expr>; 2], &[Expr], &[Sub]) = match self {
            ExprKind::Unary { rhs, .. } => ([Some(rhs), None], &[], &[]),
            ExprKind::Binary { lhs, rhs, .. } => ([Some(lhs), Some(rhs)], &[], &[]),
            ExprKind::Index { base, subs } => ([Some(base), None], &[], subs),
            ExprKind::Call { args, .. } => ([None, None], args, &[]),
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Var(_) | ExprKind::Generate(_) => {
                ([None, None], &[], &[])
            }
        };
        operands.into_iter().flatten().chain(args).chain(subs.iter().flat_map(Sub::exprs))
    }

    /// `self` with `f` applied to each expression directly under it, in
    /// order.
    pub fn try_map_children<E>(self, mut f: impl FnMut(Expr) -> Result<Expr, E>) -> Result<ExprKind, E> {
        Ok(match self {
            ExprKind::Unary { op, rhs } => ExprKind::Unary { op, rhs: Box::new(f(*rhs)?) },
            ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
                op,
                lhs: Box::new(f(*lhs)?),
                rhs: Box::new(f(*rhs)?),
            },
            ExprKind::Index { base, subs } => {
                let base = Box::new(f(*base)?);
                let subs = subs.into_iter().map(|s| s.try_map(&mut f)).collect::<Result<_, _>>()?;
                ExprKind::Index { base, subs }
            }
            ExprKind::Call { callee, args } => ExprKind::Call {
                callee,
                args: args.into_iter().map(&mut f).collect::<Result<_, _>>()?,
            },
            leaf @ (ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Var(_) | ExprKind::Generate(_)) => {
                leaf
            }
        })
    }
}

impl Expr {
    /// Visits this expression and its sub-expressions in pre-order.
    pub fn walk(&self, f: &mut impl FnMut(&Expr)) {
        f(self);
        if let ExprKind::Generate(g) = &self.kind {
            for stmt in &g.body {
                stmt.walk_exprs(f);
            }
            g.value.walk(f);
        }
        for child in self.kind.children() {
            child.walk(f);
        }
    }

    /// Whether this expression (or any sub-expression) reads the given name.
    pub fn uses_name(&self, name: &str) -> bool {
        let mut found = false;
        self.walk(&mut |e| {
            if let ExprKind::Var(n) = &e.kind {
                found |= n == name;
            }
        });
        found
    }
}

impl Stmt {
    /// The expressions the statement writes itself, in order, and not those
    /// of the blocks it holds.
    pub fn exprs(&self) -> impl Iterator<Item = &Expr> {
        let exprs: [Option<&Expr>; 4] = match self {
            Stmt::Let { value, .. } | Stmt::Expr(value, _) => [Some(value), None, None, None],
            Stmt::Var { value, .. } => [value.as_ref(), None, None, None],
            Stmt::Assign { target, value, .. } => [Some(target), Some(value), None, None],
            Stmt::For { start, end, step, unroll, .. } => [Some(start), Some(end), step.as_ref(), unroll.as_ref()],
            Stmt::While { cond, .. } | Stmt::If { cond, .. } => [Some(cond), None, None, None],
        };
        exprs.into_iter().flatten()
    }

    /// The blocks the statement holds, in order: a loop's body, an `if`'s
    /// branches.
    pub fn blocks(&self) -> impl Iterator<Item = &[Stmt]> {
        let blocks: [Option<&[Stmt]>; 2] = match self {
            Stmt::For { body, .. } | Stmt::While { body, .. } => [Some(body), None],
            Stmt::If { then, r#else, .. } => [Some(then), r#else.as_deref()],
            Stmt::Let { .. } | Stmt::Var { .. } | Stmt::Assign { .. } | Stmt::Expr(..) => [None, None],
        };
        blocks.into_iter().flatten()
    }

    /// `self` with `expr` applied to each expression it writes and `block`
    /// to each block it holds, in source order, each handed `cx`.
    pub fn try_map<C, E>(
        self,
        cx: &mut C,
        mut expr: impl FnMut(&mut C, Expr) -> Result<Expr, E>,
        mut block: impl FnMut(&mut C, Vec<Stmt>) -> Result<Vec<Stmt>, E>,
    ) -> Result<Stmt, E> {
        Ok(match self {
            Stmt::Let { name, name_span, ty, ty_span, value, span } => Stmt::Let {
                name,
                name_span,
                ty,
                ty_span,
                value: expr(cx, value)?,
                span,
            },
            Stmt::Var { name, name_span, ty, ty_span, value, span } => Stmt::Var {
                name,
                name_span,
                ty,
                ty_span,
                value: value.map(|v| expr(cx, v)).transpose()?,
                span,
            },
            Stmt::Assign { target, op, value, span } => Stmt::Assign {
                target: expr(cx, target)?,
                op,
                value: expr(cx, value)?,
                span,
            },
            Stmt::For { var, var_span, start, end, step, unroll, body, span } => Stmt::For {
                var,
                var_span,
                start: expr(cx, start)?,
                end: expr(cx, end)?,
                step: step.map(|e| expr(cx, e)).transpose()?,
                unroll: unroll.map(|e| expr(cx, e)).transpose()?,
                body: block(cx, body)?,
                span,
            },
            Stmt::While { cond, body, span } => Stmt::While {
                cond: expr(cx, cond)?,
                body: block(cx, body)?,
                span,
            },
            Stmt::If { cond, then, r#else, span } => Stmt::If {
                cond: expr(cx, cond)?,
                then: block(cx, then)?,
                r#else: r#else.map(|b| block(cx, b)).transpose()?,
                span,
            },
            Stmt::Expr(e, span) => Stmt::Expr(expr(cx, e)?, span),
        })
    }

    /// Whether this statement reads `name` or declares it, itself or in a
    /// block it holds.
    pub fn uses_name(&self, name: &str) -> bool {
        let declares = match self {
            Stmt::Let { name: n, .. } | Stmt::Var { name: n, .. } | Stmt::For { var: n, .. } => n == name,
            _ => false,
        };
        declares || self.exprs().any(|e| e.uses_name(name)) || self.blocks().flatten().any(|s| s.uses_name(name))
    }

    /// Visits every expression in this statement and its nested blocks, in
    /// pre-order.
    pub fn walk_exprs(&self, f: &mut impl FnMut(&Expr)) {
        for e in self.exprs() {
            e.walk(f);
        }
        for stmt in self.blocks().flatten() {
            stmt.walk_exprs(f);
        }
    }

    /// Whether this statement writes or binds any of `names`, whether as an
    /// assignment target, a let or a var.
    pub fn writes_any(&self, names: &[&str]) -> bool {
        let hits = |n: &str| names.contains(&n);
        match self {
            Stmt::Assign { target, .. } => match &target.kind {
                ExprKind::Var(n) => hits(n),
                ExprKind::Index { base, .. } => matches!(&base.kind, ExprKind::Var(n) if hits(n)),
                _ => false,
            },
            Stmt::Let { name, .. } | Stmt::Var { name, .. } => hits(name),
            _ => self.blocks().flatten().any(|s| s.writes_any(names)),
        }
    }
}
