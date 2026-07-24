#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    F16,
    F32,
    F64,
    I32,
    I64,
    Bool,
}

impl Scalar {
    pub fn from_name(s: &str) -> Option<Scalar> {
        Some(match s {
            "f16" => Scalar::F16,
            "f32" => Scalar::F32,
            "f64" => Scalar::F64,
            "i32" => Scalar::I32,
            "i64" => Scalar::I64,
            "bool" => Scalar::Bool,
            _ => return None,
        })
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
    Tile(Scalar, Vec<Dim>),   // register/accumulator tile with a shape
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
    /// Autotune search dimension: NAME in [v0, v1, ..., vN] for those choices,
    /// or NAME in [start, end] for the range start..=end when end > start.
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
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub attrs: Vec<Attribute>,
    pub name: String,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
}

pub const DEFAULT_CTA_THREADS: i64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Launch {
    pub max_threads: i64,
    pub min_blocks: Option<i64>,
    pub max_nreg: Option<i64>,
}

impl Kernel {
    pub fn launch(&self) -> Result<Option<Launch>, String> {
        let Some(attr) = self.attrs.iter().find(|a| a.name == "launch") else {
            return Ok(None);
        };
        let ints = attr
            .args
            .iter()
            .map(|a| match a {
                AttrArg::Positional(Literal::Int(n)) => Ok(*n),
                _ => Err("@launch takes integer arguments: \
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
                "@launch maxThreads must be a positive multiple of 32 (got {})",
                launch.max_threads
            ));
        }
        if let Some(b) = launch.min_blocks
            && b <= 0
        {
            return Err(format!("@launch minBlocks must be positive (got {b})"));
        }
        // PTX .maxnreg accepts 16..=255 registers per thread
        if let Some(r) = launch.max_nreg
            && !(16..=255).contains(&r)
        {
            return Err(format!(
                "@launch maxRegs must be between 16 and 255 (got {r})"
            ));
        }
        Ok(Some(launch))
    }

    pub fn cta_threads(&self) -> Result<i64, String> {
        Ok(self
            .launch()?
            .map_or(DEFAULT_CTA_THREADS, |l| l.max_threads))
    }

    // force legacy WMMA
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AssignOp {
    Set,
    Add,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<Type>,
        value: Expr,
    },
    Var {
        name: String,
        ty: Option<Type>,
        value: Expr,
    },
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    For {
        var: String,
        start: Expr,
        end: Expr,
        step: Option<Expr>,
        body: Vec<Stmt>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    If {
        cond: Expr,
        then: Vec<Stmt>,
        r#else: Option<Vec<Stmt>>,
    },
    Expr(Expr),
}

/// Subscript inside A[ ... ].
#[derive(Debug, Clone)]
pub enum Sub {
    Point(Expr),                      // A[i]
    Range { start: Expr, end: Expr }, // A[start : end]  (end-exclusive)
    Span { start: Expr, len: Expr },  // A[start :+ len] (start + len)
    Full,                             // A[:]
}

#[derive(Debug, Clone)]
pub enum Expr {
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
}

impl Sub {
    /// Whether any expression in this subscript uses the given name.
    pub fn uses_name(&self, name: &str) -> bool {
        match self {
            Sub::Point(e) => e.uses_name(name),
            Sub::Range { start, end } => start.uses_name(name) || end.uses_name(name),
            Sub::Span { start, len } => start.uses_name(name) || len.uses_name(name),
            Sub::Full => false,
        }
    }
}

impl Expr {
    /// pre-order
    pub fn walk(&self, f: &mut impl FnMut(&Expr)) {
        f(self);
        match self {
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Var(_) => {}
            Expr::Unary { rhs, .. } => rhs.walk(f),
            Expr::Binary { lhs, rhs, .. } => {
                lhs.walk(f);
                rhs.walk(f);
            }
            Expr::Index { base, subs } => {
                base.walk(f);
                for sub in subs {
                    match sub {
                        Sub::Point(e) => e.walk(f),
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
            Expr::Call { args, .. } => {
                for arg in args {
                    arg.walk(f);
                }
            }
        }
    }

    /// Whether this expression (or any sub-expression) reads the given name.
    pub fn uses_name(&self, name: &str) -> bool {
        let mut found = false;
        self.walk(&mut |e| {
            if let Expr::Var(n) = e {
                found |= n == name;
            }
        });
        found
    }
}

impl Stmt {
    /// Whether this statement uses the given name anywhere (read, write, rebinding)
    pub fn uses_name(&self, name: &str) -> bool {
        let block = |b: &[Stmt]| b.iter().any(|s| s.uses_name(name));
        match self {
            Stmt::Let { name: n, value, .. } | Stmt::Var { name: n, value, .. } => {
                n == name || value.uses_name(name)
            }
            Stmt::Assign { target, value, .. } => target.uses_name(name) || value.uses_name(name),
            Stmt::For {
                var,
                start,
                end,
                step,
                body,
            } => {
                var == name
                    || start.uses_name(name)
                    || end.uses_name(name)
                    || step.as_ref().is_some_and(|e| e.uses_name(name))
                    || block(body)
            }
            Stmt::While { cond, body } => cond.uses_name(name) || block(body),
            Stmt::If { cond, then, r#else } => {
                cond.uses_name(name) || block(then) || r#else.as_deref().is_some_and(block)
            }
            Stmt::Expr(e) => e.uses_name(name),
        }
    }

    /// pre-order over every expression in this statement and its nested blocks
    pub fn walk_exprs(&self, f: &mut impl FnMut(&Expr)) {
        match self {
            Stmt::Let { value, .. } | Stmt::Var { value, .. } => value.walk(f),
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
            Stmt::While { cond, body } => {
                cond.walk(f);
                for stmt in body {
                    stmt.walk_exprs(f);
                }
            }
            Stmt::If { cond, then, r#else } => {
                cond.walk(f);
                for stmt in then.iter().chain(r#else.iter().flatten()) {
                    stmt.walk_exprs(f);
                }
            }
            Stmt::Expr(e) => e.walk(f),
        }
    }

    /// Whether this statement writes or binds any of the given names (assignment target, let, var).
    pub fn writes_any(&self, names: &[&str]) -> bool {
        let hits = |n: &str| names.contains(&n);
        let block = |b: &[Stmt]| b.iter().any(|s| s.writes_any(names));
        match self {
            Stmt::Assign { target, .. } => match target {
                Expr::Var(n) => hits(n),
                Expr::Index { base, .. } => matches!(&**base, Expr::Var(n) if hits(n)),
                _ => false,
            },
            Stmt::Let { name, .. } | Stmt::Var { name, .. } => hits(name),
            Stmt::For { body, .. } | Stmt::While { body, .. } => block(body),
            Stmt::If { then, r#else, .. } => block(then) || r#else.as_deref().is_some_and(block),
            Stmt::Expr(_) => false,
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
