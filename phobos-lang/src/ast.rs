use std::collections::{BTreeMap, HashMap};

pub mod ctx;
pub mod lanes;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    F16,
    BF16,
    F32,
    F64,
    I8,
    I32,
    I64,
    U8,
    U16,
    U32,
    Bool,
}

impl Scalar {
    pub const NAMES: [&str; 11] = ["f16", "bf16", "f32", "f64", "i8", "i32", "i64", "u8", "u16", "u32", "bool"];

    pub fn from_name(s: &str) -> Option<Scalar> {
        Some(match s {
            "f16" => Scalar::F16,
            "bf16" => Scalar::BF16,
            "f32" => Scalar::F32,
            "f64" => Scalar::F64,
            "i8" => Scalar::I8,
            "i32" => Scalar::I32,
            "i64" => Scalar::I64,
            "u8" => Scalar::U8,
            "u16" => Scalar::U16,
            "u32" => Scalar::U32,
            "bool" => Scalar::Bool,
            _ => return None,
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Scalar::F16 => "f16",
            Scalar::BF16 => "bf16",
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
            Scalar::I8 => "i8",
            Scalar::I32 => "i32",
            Scalar::I64 => "i64",
            Scalar::U8 => "u8",
            Scalar::U16 => "u16",
            Scalar::U32 => "u32",
            Scalar::Bool => "bool",
        }
    }
}

/// Shape dimension: a symbolic size (M, N, K, TILE_M, ...) or a literal.
#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Sym(String),
    Int(i64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Scalar(Scalar),
    Tensor(Scalar, Vec<Dim>), // global tensor with a shape
    Tile(Scalar, Vec<Dim>),   // kernel-local tile with a shape
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Ident(String),
}

#[derive(Debug, Clone)]
pub enum AttrArg {
    /// Autotune search dimension: NAME in [v0, v1, ..., vN] for those choices.
    /// NAME in [lo, hi] with 0 < lo < hi means lo, 2*lo, 4*lo, ... and hi.
    Search { name: String, choices: Vec<i64> },
    /// Keyword argument: key = value.
    KeyValue { key: String, value: Literal },
    /// Positional value or bare flag/enum: value.
    Positional(Literal),
}

pub fn search_choices(declared: &[i64]) -> Vec<i64> {
    match *declared {
        [lo, hi] if lo > 0 && hi > lo => {
            let mut out = Vec::new();
            let mut v = lo;
            while v < hi {
                out.push(v);
                v *= 2;
            }
            out.push(hi);
            out
        }
        _ => declared.to_vec(),
    }
}

#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub args: Vec<AttrArg>,
    /// The whole `@name(..)`.
    pub span: crate::diag::Span,
    /// Each of `args`'.
    pub arg_spans: Vec<crate::diag::Span>,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    /// The name's.
    pub span: crate::diag::Span,
    pub ty_span: crate::diag::Span,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub attrs: Vec<Attribute>,
    pub name: String,
    pub name_span: crate::diag::Span,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
    /// The source the spans point into.
    pub source: std::sync::Arc<crate::diag::SourceFile>,
}

pub const DEFAULT_CTA_THREADS: i64 = 256;

/// The kernel attributes the compiler reads, without their `@`.
pub const ATTRIBUTES: &[&str] = &[
    "aligned",
    "autotune",
    "cluster",
    "dynshared",
    "global",
    "grid_barriers",
    "launch",
    "padstage",
    "persistent",
    "pipeline",
    "registers",
    "shared",
    "swizzle",
    "tensorcore",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Launch {
    pub max_threads: i64,
    pub min_blocks: Option<i64>,
    pub max_nreg: Option<i64>,
}

impl Kernel {
    /// Every `@autotune` constant at its first choice, then at its value in
    /// `overrides` where that names one: the point the autotuner compiles.
    pub fn shape_env(&self, overrides: &HashMap<String, i64>) -> BTreeMap<String, i64> {
        let mut env = BTreeMap::new();
        for arg in self.attrs.iter().filter(|a| a.name == "autotune").flat_map(|a| &a.args) {
            if let AttrArg::Search { name, choices } = arg
                && let Some(&first) = choices.first()
            {
                env.insert(name.clone(), overrides.get(name).copied().unwrap_or(first));
            }
        }
        env
    }

    /// `@launch` at the autotune constants' first choices.
    pub fn launch(&self) -> Result<Option<Launch>, String> {
        self.launch_in(&self.shape_env(&HashMap::new()))
    }

    /// `@launch` with each argument that names an `@autotune` constant taken
    /// from `env`.
    pub fn launch_in(&self, env: &BTreeMap<String, i64>) -> Result<Option<Launch>, String> {
        let Some(attr) = self.attrs.iter().find(|a| a.name == "launch") else {
            return Ok(None);
        };
        let ints = attr
            .args
            .iter()
            .map(|a| match a {
                AttrArg::Positional(Literal::Int(n)) => Ok(*n),
                AttrArg::Positional(Literal::Ident(name)) => env
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("`@launch({name})`: `{name}` is not an `@autotune` constant")),
                _ => Err("@launch takes integers or `@autotune` constants: \
                    @launch(maxThreads[, minBlocks[, maxRegs]])"
                    .to_string()),
            })
            .collect::<Result<Vec<_>, String>>()?;
        let launch = match ints.as_slice() {
            [m] => Launch {
                max_threads: *m,
                min_blocks: None,
                max_nreg: None,
            },
            [m, b] => Launch {
                max_threads: *m,
                min_blocks: Some(*b),
                max_nreg: None,
            },
            [m, b, r] => Launch {
                max_threads: *m,
                min_blocks: Some(*b),
                max_nreg: Some(*r),
            },
            _ => {
                return Err("@launch takes 1 to 3 arguments: \
                    @launch(maxThreads[, minBlocks[, maxRegs]])"
                    .to_string());
            }
        };
        if launch.max_threads <= 0 || launch.max_threads % 32 != 0 {
            return Err(format!(
                "`@launch` maxThreads must be a positive multiple of 32 (got {})",
                launch.max_threads
            ));
        }
        if let Some(b) = launch.min_blocks
            && b <= 0
        {
            return Err(format!("`@launch` minBlocks must be positive (got {b})"));
        }
        // PTX .maxnreg accepts 16..=255 registers per thread
        if let Some(r) = launch.max_nreg
            && !(16..=255).contains(&r)
        {
            return Err(format!(
                "`@launch` maxRegs must be between 16 and 255 (got {r})"
            ));
        }
        Ok(Some(launch))
    }

    /// Fails on an attribute the compiler does not read, which would
    /// otherwise be silently ignored. `@cluster` is read by the cluster
    /// codegen ahead of this one.
    pub fn check_attributes(&self) -> Result<(), crate::diag::Diagnostics> {
        use crate::diag::{Diagnostic, Diagnostics};
        let list: Vec<Diagnostic> = self
            .attrs
            .iter()
            .filter(|a| !ATTRIBUTES.contains(&a.name.as_str()))
            .map(|a| {
                let d = Diagnostic::error(format!("unknown attribute `@{}`", a.name))
                    .primary(a.span, "the compiler does not read it")
                    .note(format!("the compiler reads `@{}`", ATTRIBUTES.join("`, `@")));
                match crate::diag::similar(&a.name, ATTRIBUTES.iter().copied()) {
                    Some(near) => {
                        let at = crate::diag::Span::new(a.span.lo + 1, a.span.lo + 1 + a.name.len() as u32);
                        d.suggest(format!("a similar attribute exists: `@{near}`"), at, near)
                    }
                    None => d,
                }
            })
            .collect();
        if list.is_empty() {
            Ok(())
        } else {
            Err(Diagnostics::new(self.source.clone(), list))
        }
    }

    /// The attribute `@name`, if written.
    pub fn attr(&self, name: &str) -> Option<&Attribute> {
        self.attrs.iter().find(|a| a.name == name)
    }

    /// The CTA size at the autotune constants' first choices.
    pub fn cta_threads(&self) -> Result<i64, String> {
        self.cta_threads_for(&HashMap::new())
    }

    /// The CTA size of the variant compiled with `overrides`, which a
    /// `@launch` naming an `@autotune` constant depends on.
    pub fn cta_threads_for(&self, overrides: &HashMap<String, i64>) -> Result<i64, String> {
        Ok(self
            .launch_in(&self.shape_env(overrides))?
            .map_or(DEFAULT_CTA_THREADS, |l| l.max_threads))
    }

    pub fn wants_dynamic_shared(&self) -> bool {
        self.attrs.iter().any(|a| a.name == "dynshared")
    }

    /// Whether `@padstage` was written on this kernel: pad a `var x = <tensor
    /// slice>` staging tile against shared-memory bank conflicts. See
    /// `Build::should_pad_stage` for which tiles that reaches.
    pub fn wants_padded_stage(&self) -> bool {
        self.attrs.iter().any(|a| a.name == "padstage")
    }

    /// Whether the kernel calls an intrinsic written for a module indexed at
    /// 64 bits, as its `cp.async` staging needs.
    pub fn wants_ldmatrix(&self) -> bool {
        self.calls(|callee| callee.ends_with("_qgemm_t"))
    }

    /// What `@swizzle` asks of the shared tiles the compiler would swizzle:
    /// None to decide alone, Some(true) for `@swizzle` or `@swizzle(on)`,
    /// which fails where it cannot, Some(false) for `@swizzle(off)`.
    pub fn swizzle_mode(&self) -> Result<Option<bool>, String> {
        let Some(attr) = self.attrs.iter().find(|a| a.name == "swizzle") else {
            return Ok(None);
        };
        match &attr.args[..] {
            [] => Ok(Some(true)),
            [AttrArg::Positional(Literal::Ident(v))] if v == "on" => Ok(Some(true)),
            [AttrArg::Positional(Literal::Ident(v))] if v == "off" => Ok(Some(false)),
            _ => Err("`@swizzle` takes `on` or `off`".to_string()),
        }
    }

    /// `@persistent`: None when absent, `Some(None)` written bare, and
    /// `Some(Some(dim))` with the grid's block count, a literal or an
    /// `@autotune` name, which `ctx.blocks.x` then is.
    pub fn persistent(&self) -> Result<Option<Option<Dim>>, String> {
        let Some(attr) = self.attr("persistent") else {
            return Ok(None);
        };
        match &attr.args[..] {
            [] => Ok(Some(None)),
            [AttrArg::Positional(Literal::Int(n))] if *n > 0 => Ok(Some(Some(Dim::Int(*n)))),
            [AttrArg::Positional(Literal::Ident(name))] => Ok(Some(Some(Dim::Sym(name.clone())))),
            _ => Err("`@persistent` takes the grid's block count, as `@persistent(BLOCKS)`, or nothing".to_string()),
        }
    }

    /// The tensors the body declares, `var HQ: tensor<i8>[1, FF]`: scratch
    /// the launch owns, in declaration order, with where each is declared.
    pub fn scratch(&self) -> Vec<(&str, &Type, crate::diag::Span)> {
        fn walk<'a>(stmts: &'a [Stmt], out: &mut Vec<(&'a str, &'a Type, crate::diag::Span)>) {
            for stmt in stmts {
                match stmt {
                    Stmt::Var {
                        name,
                        ty: Some(ty @ Type::Tensor(..)),
                        value: None,
                        span,
                        ..
                    } => out.push((name, ty, *span)),
                    Stmt::For { body, .. } | Stmt::While { body, .. } => walk(body, out),
                    Stmt::If { then, r#else, .. } => {
                        walk(then, out);
                        walk(r#else.as_deref().unwrap_or_default(), out);
                    }
                    _ => {}
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.body, &mut out);
        out
    }

    /// Whether the body calls `grid_barrier()` with no barrier named, which
    /// takes the one a `@persistent` launch supplies.
    pub fn implicit_grid_barrier(&self) -> bool {
        let mut found = false;
        for stmt in &self.body {
            stmt.walk_exprs(&mut |e| {
                found |= matches!(&e.kind, ExprKind::Call { callee, args } if callee == "grid_barrier" && args.is_empty());
            });
        }
        found
    }

    /// The names an attribute lists, as `@shared(a, b)` does.
    pub fn named_by(&self, attr: &str) -> Vec<String> {
        self.attrs
            .iter()
            .filter(|a| a.name == attr)
            .flat_map(|a| a.args.iter())
            .filter_map(|arg| match arg {
                AttrArg::Positional(Literal::Ident(name)) => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// `@grid_barriers(n)`: the grid barriers the kernel has, written and
    /// placed together.
    pub fn grid_barriers(&self) -> Result<Option<usize>, String> {
        let Some(attr) = self.attr("grid_barriers") else {
            return Ok(None);
        };
        match &attr.args[..] {
            [AttrArg::Positional(Literal::Int(n))] if *n >= 0 => Ok(Some(*n as usize)),
            _ => Err("`@grid_barriers` takes the kernel's barrier count, as `@grid_barriers(1)`".to_string()),
        }
    }

    /// The tiles `@registers(a, b, ..)` asks to keep in registers.
    pub fn register_tiles(&self) -> Vec<String> {
        self.attrs
            .iter()
            .filter(|a| a.name == "registers")
            .flat_map(|a| a.args.iter())
            .filter_map(|arg| match arg {
                AttrArg::Positional(Literal::Ident(name)) => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// True when the kernel reads or writes at system scope, which forms
    /// full 64-bit addresses.
    pub fn wants_system_scope(&self) -> bool {
        self.calls(|callee| matches!(callee, "load_acquire" | "store_release"))
    }

    /// Whether the body calls a built-in `pick` accepts.
    fn calls(&self, pick: impl Fn(&str) -> bool) -> bool {
        let mut found = false;
        for stmt in &self.body {
            stmt.walk_exprs(&mut |e| {
                if let ExprKind::Call { callee, .. } = &e.kind
                    && pick(callee)
                {
                    found = true;
                }
            });
        }
        found
    }

    /// True for `@tensorcore` without the `wmma` argument, which forces the
    /// legacy WMMA path.
    pub fn wants_mma_sync(&self) -> bool {
        self.attrs.iter().any(|a| {
            a.name == "tensorcore"
                && !a
                    .args
                    .iter()
                    .any(|arg| matches!(arg, AttrArg::Positional(Literal::Ident(f)) if f == "wmma"))
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignOp {
    Set,
    Add,
}

/// Where a statement sits in the source. A compound statement's span is its
/// header, up to its `{`.
pub use crate::diag::Span;

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        name_span: crate::diag::Span,
        ty: Option<Type>,
        /// Where `ty` is written; empty when it is not.
        ty_span: crate::diag::Span,
        value: Expr,
        span: Span,
    },
    Var {
        name: String,
        name_span: crate::diag::Span,
        ty: Option<Type>,
        /// Where `ty` is written; empty when it is not.
        ty_span: crate::diag::Span,
        value: Option<Expr>,
        span: Span,
    },
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
        span: Span,
    },
    For {
        var: String,
        var_span: crate::diag::Span,
        start: Expr,
        end: Expr,
        step: Option<Expr>,
        /// `for<unroll n>`: the body built `n` times a turn.
        unroll: Option<Expr>,
        body: Vec<Stmt>,
        span: Span,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    If {
        cond: Expr,
        then: Vec<Stmt>,
        r#else: Option<Vec<Stmt>>,
        span: Span,
    },
    Expr(Expr, Span),
}

/// Subscript inside A[ ... ].
#[derive(Debug, Clone)]
pub enum Sub {
    Point(Expr),                      // A[i]
    Range { start: Expr, end: Expr }, // A[start : end]  (end-exclusive)
    Span { start: Expr, len: Expr },  // A[start :+ len] (start + len)
    From(Expr),                       // A[start :]      (to the end)
    Full,                             // A[:]
}

/// An expression and the source it was parsed from. One the compiler builds
/// itself, by desugaring or rewriting, carries the span of the source it
/// stands for, or none.
#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: crate::diag::Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    Bool(bool),
    Var(String),
    Unary {
        op: UnOp,
        rhs: Box<Expr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Index {
        base: Box<Expr>,
        subs: Vec<Sub>,
    },
    Call {
        callee: String,
        args: Vec<Expr>,
    },
    /// `generate(f)`: a tile with no storage. The function inliner makes it
    /// from a statement function whose body ends in an expression.
    Generate(Box<Generated>),
}

/// A generated tile's element block at `(row, unit)`: `value` after `body`,
/// with `params` bound to the two.
#[derive(Debug, Clone)]
pub struct Generated {
    pub params: [String; 2],
    pub body: Vec<Stmt>,
    pub value: Expr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    Neg,
    Not,
    /// `~`, every bit of an integer flipped.
    BitNot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    /// `>>`, arithmetic: the in-kernel integer is signed.
    Shr,
    /// `&&` and `||` on bools. Both sides are evaluated, which is safe since
    /// an expression has no side effects.
    And,
    Or,
    /// `min(a, b)` and `max(a, b)`, which the parser turns into operators so
    /// they work wherever `+` does.
    Min,
    Max,
}

/// An expression with no source of its own.
impl From<ExprKind> for Expr {
    fn from(kind: ExprKind) -> Self {
        Expr {
            kind,
            span: crate::diag::Span::default(),
        }
    }
}

impl Sub {
    /// Whether any expression in this subscript uses the given name.
    pub fn uses_name(&self, name: &str) -> bool {
        match self {
            Sub::Point(e) => e.uses_name(name),
            Sub::Range { start, end } => start.uses_name(name) || end.uses_name(name),
            Sub::Span { start, len } => start.uses_name(name) || len.uses_name(name),
            Sub::From(start) => start.uses_name(name),
            Sub::Full => false,
        }
    }
}

impl Expr {
    pub fn new(kind: ExprKind, span: crate::diag::Span) -> Self {
        Expr { kind, span }
    }

    /// `self` placed at `span`.
    pub fn at(self, span: crate::diag::Span) -> Self {
        Expr { span, ..self }
    }

    pub fn int(n: i64) -> Self {
        ExprKind::Int(n).into()
    }

    pub fn float(x: f64) -> Self {
        ExprKind::Float(x).into()
    }

    pub fn bool(b: bool) -> Self {
        ExprKind::Bool(b).into()
    }

    pub fn var(name: impl Into<String>) -> Self {
        ExprKind::Var(name.into()).into()
    }

    pub fn unary(op: UnOp, rhs: Expr) -> Self {
        ExprKind::Unary { op, rhs: Box::new(rhs) }.into()
    }

    pub fn binary(op: BinOp, lhs: Expr, rhs: Expr) -> Self {
        ExprKind::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
        .into()
    }

    pub fn index(base: Expr, subs: Vec<Sub>) -> Self {
        ExprKind::Index {
            base: Box::new(base),
            subs,
        }
        .into()
    }

    pub fn call(callee: impl Into<String>, args: Vec<Expr>) -> Self {
        ExprKind::Call {
            callee: callee.into(),
            args,
        }
        .into()
    }

    /// The name, if this is a bare one.
    pub fn as_var(&self) -> Option<&str> {
        match &self.kind {
            ExprKind::Var(name) => Some(name),
            _ => None,
        }
    }

    /// The value, if this is an integer literal.
    pub fn as_int(&self) -> Option<i64> {
        match self.kind {
            ExprKind::Int(n) => Some(n),
            _ => None,
        }
    }

    /// Visits this expression and its sub-expressions in pre-order.
    pub fn walk(&self, f: &mut impl FnMut(&Expr)) {
        f(self);
        match &self.kind {
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Var(_) => {}
            ExprKind::Unary { rhs, .. } => rhs.walk(f),
            ExprKind::Binary { lhs, rhs, .. } => {
                lhs.walk(f);
                rhs.walk(f);
            }
            ExprKind::Index { base, subs } => {
                base.walk(f);
                for sub in subs {
                    match sub {
                        Sub::Point(e) | Sub::From(e) => e.walk(f),
                        Sub::Range { start, end } => {
                            start.walk(f);
                            end.walk(f);
                        }
                        Sub::Span { start, len } => {
                            start.walk(f);
                            len.walk(f);
                        }
                        Sub::Full => {}
                    }
                }
            }
            ExprKind::Call { args, .. } => {
                for arg in args {
                    arg.walk(f);
                }
            }
            ExprKind::Generate(g) => {
                for stmt in &g.body {
                    stmt.walk_exprs(f);
                }
                g.value.walk(f);
            }
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
    pub fn span(&self) -> Span {
        match self {
            Stmt::Let { span, .. }
            | Stmt::Var { span, .. }
            | Stmt::Assign { span, .. }
            | Stmt::For { span, .. }
            | Stmt::While { span, .. }
            | Stmt::If { span, .. }
            | Stmt::Expr(_, span) => *span,
        }
    }

    pub fn as_decl(&self) -> Option<(&str, Option<&Type>, Option<&Expr>)> {
        match self {
            Stmt::Let {
                name, ty, value, ..
            } => Some((name, ty.as_ref(), Some(value))),
            Stmt::Var {
                name, ty, value, ..
            } => Some((name, ty.as_ref(), value.as_ref())),
            _ => None,
        }
    }

    pub fn uses_name(&self, name: &str) -> bool {
        let block = |b: &[Stmt]| b.iter().any(|s| s.uses_name(name));
        match self {
            Stmt::Let { name: n, value, .. } => n == name || value.uses_name(name),
            Stmt::Var { name: n, value, .. } => {
                n == name || value.as_ref().is_some_and(|v| v.uses_name(name))
            }
            Stmt::Assign { target, value, .. } => target.uses_name(name) || value.uses_name(name),
            Stmt::For {
                var,
                start,
                end,
                step,
                body,
                ..
            } => {
                var == name
                    || start.uses_name(name)
                    || end.uses_name(name)
                    || step.as_ref().is_some_and(|e| e.uses_name(name))
                    || block(body)
            }
            Stmt::While { cond, body, .. } => cond.uses_name(name) || block(body),
            Stmt::If {
                cond, then, r#else, ..
            } => cond.uses_name(name) || block(then) || r#else.as_deref().is_some_and(block),
            Stmt::Expr(e, _) => e.uses_name(name),
        }
    }

    /// Visits every expression in this statement and its nested blocks, in
    /// pre-order.
    pub fn walk_exprs(&self, f: &mut impl FnMut(&Expr)) {
        match self {
            Stmt::Let { value, .. } => value.walk(f),
            Stmt::Var { value, .. } => {
                if let Some(value) = value {
                    value.walk(f);
                }
            }
            Stmt::Assign { target, value, .. } => {
                target.walk(f);
                value.walk(f);
            }
            Stmt::For {
                start,
                end,
                step,
                body,
                ..
            } => {
                start.walk(f);
                end.walk(f);
                if let Some(step) = step {
                    step.walk(f);
                }
                for stmt in body {
                    stmt.walk_exprs(f);
                }
            }
            Stmt::While { cond, body, .. } => {
                cond.walk(f);
                for stmt in body {
                    stmt.walk_exprs(f);
                }
            }
            Stmt::If {
                cond, then, r#else, ..
            } => {
                cond.walk(f);
                for stmt in then.iter().chain(r#else.iter().flatten()) {
                    stmt.walk_exprs(f);
                }
            }
            Stmt::Expr(e, _) => e.walk(f),
        }
    }

    /// Whether this statement writes or binds any of `names`, whether as an
    /// assignment target, a let or a var.
    pub fn writes_any(&self, names: &[&str]) -> bool {
        let hits = |n: &str| names.contains(&n);
        let block = |b: &[Stmt]| b.iter().any(|s| s.writes_any(names));
        match self {
            Stmt::Assign { target, .. } => match &target.kind {
                ExprKind::Var(n) => hits(n),
                ExprKind::Index { base, .. } => matches!(&base.kind, ExprKind::Var(n) if hits(n)),
                _ => false,
            },
            Stmt::Let { name, .. } | Stmt::Var { name, .. } => hits(name),
            Stmt::For { body, .. } | Stmt::While { body, .. } => block(body),
            Stmt::If { then, r#else, .. } => block(then) || r#else.as_deref().is_some_and(block),
            Stmt::Expr(..) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::search_choices;

    #[test]
    fn two_values_are_doubling_bounds() {
        assert_eq!(search_choices(&[16, 256]), vec![16, 32, 64, 128, 256]);
        assert_eq!(search_choices(&[4, 32]), vec![4, 8, 16, 32]);
        assert_eq!(search_choices(&[64, 100]), vec![64, 100]);
        assert_eq!(search_choices(&[64, 200]), vec![64, 128, 200]);
        assert_eq!(search_choices(&[64, 128]), vec![64, 128]);
    }

    #[test]
    fn other_counts_are_explicit_lists() {
        assert_eq!(search_choices(&[16, 32, 48]), vec![16, 32, 48]);
        assert_eq!(search_choices(&[64]), vec![64]);
        assert_eq!(search_choices(&[32, 16]), vec![32, 16]);
        assert_eq!(search_choices(&[1024, 16]), vec![1024, 16]);
    }
}
