// Prints a Phobos source in its Python spelling, the inverse of the Python
// front end (`python/phobos/frontend.py`), so a `.ph` and the `.py` it
// prints compile to the same module:
//
//     ph2py FILE.ph
//     ph2py --out DIR PATH...
//
// The first prints to standard output; the second writes `DIR/<stem>.py`
// for each source, a file or every `.ph` in a directory, and prints a line
// for each it cannot spell. Phobos `/`
// becomes `//` where the kernels that use it divide integers, which a
// compile at the default target tells; a `/` no kernel builds stays `/`.
// A variable or parameter named for a Python keyword, as `from` is, takes
// a trailing `_`, which no module it compiles to sees. A source Python
// cannot spell, a kernel named for a keyword or a `fn` whose `/` divides
// integers at one call and floats at another, is an error.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::{Result, bail};
use phobos_lang::ast::lanes::{self, Group};
use phobos_lang::ast::{AssignOp, AttrArg, BinOp, Dim, Expr, ExprKind, Kernel, Literal, Stmt, Sub, Type, UnOp, ctx};
use phobos_lang::diag::{SourceFile, Span};
use phobos_lang::parser::{Body, Function};

const PY_KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
    "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield", "ph",
];

// Python's precedence, loosest first.
const OR: u8 = 1;
const AND: u8 = 2;
const NOT: u8 = 3;
const CMP: u8 = 4;
const UNARY: u8 = 11;
const ATOM: u8 = 13;

const USAGE: &str = "usage: ph2py FILE.ph
       ph2py --out DIR PATH...

Prints a Phobos source in its Python spelling, which the `phobos` package's
front end reads back to the same module.

  FILE.ph           print FILE's Python spelling to standard output
  --out DIR PATH... write DIR/<stem>.py for each source, a file or every .ph
                    in a directory, and print a line for each it cannot spell
  -V, --version     print the version and the compiler fingerprint
  -h, --help        print this";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    match args.peek().map(String::as_str) {
        Some("--help" | "-h") => {
            println!("{USAGE}");
            return Ok(());
        }
        Some("--version" | "-V") => {
            let fingerprint = phobos_kernels::COMPILER_FINGERPRINT;
            println!("ph2py {} (compiler {fingerprint})", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        _ => {}
    }
    let out = if args.peek().map(String::as_str) == Some("--out") {
        args.next();
        Some(PathBuf::from(args.next().unwrap_or_default()))
    } else {
        None
    };
    let mut files: Vec<PathBuf> = Vec::new();
    for arg in args.map(PathBuf::from) {
        if arg.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(&arg)?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "ph"))
                .collect();
            found.sort();
            files.extend(found);
        } else {
            files.push(arg);
        }
    }
    if files.is_empty() {
        bail!("{USAGE}");
    }
    let Some(dir) = out else {
        for f in &files {
            print!("{}", spell(&std::fs::read_to_string(f)?.replace('\r', ""))?);
        }
        return Ok(());
    };
    std::fs::create_dir_all(&dir)?;
    let mut failed = 0;
    for f in &files {
        let text = std::fs::read_to_string(f)?.replace('\r', "");
        match spell(&text) {
            Ok(py) => std::fs::write(dir.join(f.file_stem().unwrap()).with_extension("py"), py)?,
            Err(e) => {
                failed += 1;
                println!("{}: {e:#}", f.display());
            }
        }
    }
    println!("{} sources, {failed} not spelled", files.len());
    Ok(())
}

/// The Python module `text` spells.
fn spell(text: &str) -> Result<String> {
    let file = SourceFile::new("", text);
    let (kernels, fns) = phobos_lang::parse_items(&file)?;
    let idents = text
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect();
    NAMES.with(|names| *names.borrow_mut() = idents);
    let mut kinds: HashMap<(u32, u32), BTreeSet<bool>> = HashMap::new();
    for (gap, integer) in phobos_lang::divisions(&Default::default(), &file) {
        kinds.entry((gap.lo, gap.hi)).or_default().insert(integer);
    }
    let program: HashSet<String> = fns.iter().map(|f| f.name.clone()).collect();
    let mut p = Printer { text, kinds, program, out: String::new() };
    // The comment a source opens with says what it is, so it stays.
    for line in text.lines().take_while(|l| l.starts_with("//")) {
        let _ = writeln!(p.out, "#{}", &line[2..]);
    }
    if !p.out.is_empty() {
        p.out.push('\n');
    }
    p.out.push_str("from __future__ import annotations\n\nimport phobos.lang as ph\n");
    for f in &fns {
        p.function(f)?;
    }
    for k in &kernels {
        p.kernel(k)?;
    }
    Ok(p.out)
}

struct Printer<'a> {
    text: &'a str,
    kinds: HashMap<(u32, u32), BTreeSet<bool>>,
    program: HashSet<String>,
    out: String,
}

/// `n` as a Python name: one Python keeps for itself takes `_`s until it
/// names nothing in the source. Only a kernel's name reaches the compiled
/// module, and it never takes one.
fn name(n: &str) -> Result<String> {
    if !PY_KEYWORDS.contains(&n) {
        return Ok(n.to_string());
    }
    let taken = NAMES.with(|names| names.borrow().clone());
    let mut out = format!("{n}_");
    while taken.contains(&out) {
        out.push('_');
    }
    Ok(out)
}

thread_local! {
    /// Every identifier of the source being spelled.
    static NAMES: std::cell::RefCell<HashSet<String>> = std::cell::RefCell::new(HashSet::new());
}

fn kernel_name(n: &str) -> Result<&str> {
    if PY_KEYWORDS.contains(&n) {
        bail!("kernel `{n}` is named for a Python keyword; its Python name would rename the compiled kernel");
    }
    Ok(n)
}

fn scalar(s: phobos_lang::ast::Scalar) -> String {
    format!("ph.{}", s.name())
}

fn dims(ds: &[Dim]) -> Result<String> {
    ds.iter()
        .map(|d| match d {
            Dim::Sym(s) => name(s),
            Dim::Int(n) => Ok(n.to_string()),
        })
        .collect::<Result<Vec<_>>>()
        .map(|v| v.join(", "))
}

fn ty(t: &Type) -> Result<String> {
    Ok(match t {
        Type::Scalar(s) => scalar(*s),
        Type::Tensor(s, d) => format!("ph.tensor[{}][{}]", scalar(*s), dims(d)?),
        Type::Tile(s, d) => format!("ph.tile[{}][{}]", scalar(*s), dims(d)?),
    })
}

fn literal(l: &Literal) -> String {
    match l {
        Literal::Int(n) => n.to_string(),
        Literal::Float(x) => format!("{x:?}"),
        Literal::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Literal::Ident(s) => format!("\"{s}\""),
    }
}

impl Printer<'_> {
    fn line(&mut self, depth: usize, text: &str) {
        let _ = writeln!(self.out, "{}{text}", "    ".repeat(depth));
    }

    fn kernel(&mut self, k: &Kernel) -> Result<()> {
        self.out.push_str("\n\n@ph.kernel\n");
        for a in &k.attrs {
            let mut args = Vec::new();
            let mut keyword = false;
            for arg in &a.args {
                args.push(match arg {
                    AttrArg::Search { name: n, choices } => {
                        keyword = true;
                        let c: Vec<String> = choices.iter().map(i64::to_string).collect();
                        format!("{}=[{}]", name(n)?, c.join(", "))
                    }
                    AttrArg::KeyValue { key, value } => {
                        keyword = true;
                        format!("{}={}", name(key)?, literal(value))
                    }
                    AttrArg::Positional(l) => {
                        if keyword {
                            bail!("`@{}` has a positional argument after a keyword one", a.name);
                        }
                        literal(l)
                    }
                });
            }
            // `@x` and `@x()` parse alike; an attribute with no arguments
            // prints bare.
            if args.is_empty() {
                self.line(0, &format!("@ph.{}", a.name));
            } else {
                self.line(0, &format!("@ph.{}({})", a.name, args.join(", ")));
            }
        }
        let params = k
            .params
            .iter()
            .map(|p| Ok(format!("{}: {}", name(&p.name)?, ty(&p.ty)?)))
            .collect::<Result<Vec<_>>>()?;
        self.line(0, &format!("def {}({}):", kernel_name(&k.name)?, params.join(", ")));
        self.block(&k.body, 1)
    }

    fn function(&mut self, f: &Function) -> Result<()> {
        let mut free = BTreeSet::new();
        let mut bound: HashSet<String> = f.params.iter().cloned().collect();
        match &f.body {
            Body::Expr(e) => self.free_expr(e, &bound, &mut free),
            Body::Stmts(s) => self.free_stmts(s, &mut bound, &mut free),
        }
        self.out.push_str("\n\n");
        let params = f.params.iter().map(|p| name(p)).collect::<Result<Vec<_>>>()?;
        let free: Vec<String> = free.iter().map(|n| name(n)).collect::<Result<_>>()?;
        if free.is_empty() {
            self.line(0, "@ph.fn");
        } else {
            let list: Vec<String> = free.iter().map(|n| format!("\"{n}\"")).collect();
            let comma = if list.len() == 1 { "," } else { "" };
            self.line(0, &format!("@ph.fn(uses=({}{comma}))", list.join(", ")));
        }
        self.line(0, &format!("def {}({}):", name(&f.name)?, params.join(", ")));
        match &f.body {
            Body::Expr(e) => {
                let e = self.expr(e, 0)?;
                self.line(1, &format!("return {e}"));
                Ok(())
            }
            Body::Stmts(s) => self.block(s, 1),
        }
    }

    // The names a `fn` reads from its caller: read, not bound by it, and
    // not a function. Type dimensions count.

    fn free_expr(&self, e: &Expr, bound: &HashSet<String>, free: &mut BTreeSet<String>) {
        e.walk(&mut |e| {
            if let ExprKind::Var(n) = &e.kind
                && !n.starts_with('#')
                && !bound.contains(n)
                && !self.program.contains(n)
            {
                free.insert(n.clone());
            }
        });
    }

    fn free_type(&self, t: &Option<Type>, bound: &HashSet<String>, free: &mut BTreeSet<String>) {
        if let Some(Type::Tensor(_, ds) | Type::Tile(_, ds)) = t {
            for d in ds {
                if let Dim::Sym(n) = d
                    && !bound.contains(n)
                {
                    free.insert(n.clone());
                }
            }
        }
    }

    fn free_stmts(&self, body: &[Stmt], bound: &mut HashSet<String>, free: &mut BTreeSet<String>) {
        for s in body {
            match s {
                Stmt::Let { name, ty, value, .. } => {
                    self.free_type(ty, bound, free);
                    self.free_expr(value, bound, free);
                    bound.insert(name.clone());
                }
                Stmt::Var { name, ty, value, .. } => {
                    self.free_type(ty, bound, free);
                    if let Some(v) = value {
                        self.free_expr(v, bound, free);
                    }
                    bound.insert(name.clone());
                }
                Stmt::For { var, start, end, step, unroll, body, .. } => {
                    for e in [Some(start), Some(end), step.as_ref(), unroll.as_ref()].into_iter().flatten() {
                        self.free_expr(e, bound, free);
                    }
                    bound.insert(var.clone());
                    self.free_stmts(body, bound, free);
                }
                Stmt::While { cond, body, .. } => {
                    self.free_expr(cond, bound, free);
                    self.free_stmts(body, bound, free);
                }
                Stmt::If { cond, then, r#else, .. } => {
                    self.free_expr(cond, bound, free);
                    self.free_stmts(then, bound, free);
                    if let Some(e) = r#else {
                        self.free_stmts(e, bound, free);
                    }
                }
                Stmt::Assign { target, value, .. } => {
                    self.free_expr(target, bound, free);
                    self.free_expr(value, bound, free);
                }
                Stmt::Expr(e, _) => self.free_expr(e, bound, free),
            }
        }
    }

    // Statements.

    fn block(&mut self, body: &[Stmt], depth: usize) -> Result<()> {
        if body.is_empty() {
            self.line(depth, "pass");
        }
        for s in body {
            self.stmt(s, depth)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt, depth: usize) -> Result<()> {
        match s {
            Stmt::Let { name: n, ty: t, value, .. } => {
                let marker = match t {
                    Some(t) => format!("ph.let[{}]", ty(t)?),
                    None => "ph.let".to_string(),
                };
                let v = self.expr(value, 0)?;
                self.line(depth, &format!("{}: {marker} = {v}", name(n)?));
            }
            Stmt::Var { name: n, ty: t, value, .. } => {
                let marker = match t {
                    Some(t) => format!("ph.var[{}]", ty(t)?),
                    None => "ph.var".to_string(),
                };
                match value {
                    Some(v) => {
                        let v = self.expr(v, 0)?;
                        self.line(depth, &format!("{}: {marker} = {v}", name(n)?));
                    }
                    None => self.line(depth, &format!("{}: {marker}", name(n)?)),
                }
            }
            Stmt::Assign { target, op, value, .. } => {
                let (t, v) = (self.expr(target, 0)?, self.expr(value, 0)?);
                let sym = if *op == AssignOp::Add { "+=" } else { "=" };
                self.line(depth, &format!("{t} {sym} {v}"));
            }
            Stmt::For { var, start, end, step, unroll, body, .. } => {
                let mut range = format!("range({}, {}", self.expr(start, 0)?, self.expr(end, 0)?);
                if let Some(step) = step {
                    let _ = write!(range, ", {}", self.expr(step, 0)?);
                }
                range.push(')');
                if let Some(n) = unroll {
                    range = format!("ph.unroll({}, {range})", self.expr(n, 0)?);
                }
                self.line(depth, &format!("for {} in {range}:", name(var)?));
                self.block(body, depth + 1)?;
            }
            Stmt::While { cond, body, .. } => {
                let c = self.expr(cond, 0)?;
                self.line(depth, &format!("while {c}:"));
                self.block(body, depth + 1)?;
            }
            Stmt::If { .. } => self.branch(s, depth, "if")?,
            Stmt::Expr(e, _) => match e.as_var() {
                Some("#break") => self.line(depth, "break"),
                Some("#continue") => self.line(depth, "continue"),
                Some("#return") => self.line(depth, "return"),
                _ => {
                    let e = self.expr(e, 0)?;
                    self.line(depth, &e);
                }
            },
        }
        Ok(())
    }

    fn branch(&mut self, s: &Stmt, depth: usize, word: &str) -> Result<()> {
        let Stmt::If { cond, then, r#else, .. } = s else { unreachable!("a branch is an if") };
        let c = self.expr(cond, 0)?;
        self.line(depth, &format!("{word} {c}:"));
        self.block(then, depth + 1)?;
        match r#else.as_deref() {
            None => Ok(()),
            Some([inner @ Stmt::If { .. }]) => self.branch(inner, depth, "elif"),
            Some(body) => {
                self.line(depth, "else:");
                self.block(body, depth + 1)
            }
        }
    }

    // Expressions, parenthesized where Python would read them otherwise.

    fn expr(&self, e: &Expr, prec: u8) -> Result<String> {
        let (text, mine) = self.expr_inner(e)?;
        Ok(if mine < prec { format!("({text})") } else { text })
    }

    fn expr_inner(&self, e: &Expr) -> Result<(String, u8)> {
        Ok(match &e.kind {
            ExprKind::Int(n) => (n.to_string(), ATOM),
            ExprKind::Float(x) => (self.float(*x, e.span), ATOM),
            ExprKind::Bool(b) => (if *b { "True" } else { "False" }.to_string(), ATOM),
            ExprKind::Var(n) => (name(n)?, ATOM),
            ExprKind::Unary { op, rhs } => match op {
                UnOp::Neg => (format!("-{}", self.expr(rhs, UNARY)?), UNARY),
                UnOp::BitNot => (format!("~{}", self.expr(rhs, UNARY)?), UNARY),
                UnOp::Not => (format!("not {}", self.expr(rhs, NOT)?), NOT),
            },
            ExprKind::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs)?,
            ExprKind::Index { base, subs } => {
                let subs = subs.iter().map(|s| self.sub(s)).collect::<Result<Vec<_>>>()?;
                (format!("{}[{}]", self.expr(base, ATOM)?, subs.join(", ")), ATOM)
            }
            ExprKind::Call { callee, args } => (self.call(callee, args)?, ATOM),
            ExprKind::Generate(_) => bail!("a generated tile has no source spelling"),
        })
    }

    fn float(&self, x: f64, span: Span) -> String {
        // The literal as written, which is already a Python float literal.
        if let Some(t) = self.text.get(span.lo as usize..span.hi as usize)
            && t.parse::<f64>().is_ok()
        {
            return t.to_string();
        }
        format!("{x:?}")
    }

    fn binary(&self, op: BinOp, lhs: &Expr, rhs: &Expr) -> Result<(String, u8)> {
        if matches!(op, BinOp::Min | BinOp::Max) {
            let f = if op == BinOp::Min { "min" } else { "max" };
            return Ok((format!("ph.{f}({}, {})", self.expr(lhs, 0)?, self.expr(rhs, 0)?), ATOM));
        }
        let (sym, prec) = match op {
            BinOp::Or => ("or", OR),
            BinOp::And => ("and", AND),
            BinOp::Eq => ("==", CMP),
            BinOp::Ne => ("!=", CMP),
            BinOp::Lt => ("<", CMP),
            BinOp::Le => ("<=", CMP),
            BinOp::Gt => (">", CMP),
            BinOp::Ge => (">=", CMP),
            BinOp::BitOr => ("|", 5),
            BinOp::BitXor => ("^", 6),
            BinOp::BitAnd => ("&", 7),
            BinOp::Shl => ("<<", 8),
            BinOp::Shr => (">>", 8),
            BinOp::Add => ("+", 9),
            BinOp::Sub => ("-", 9),
            BinOp::Mul => ("*", 10),
            BinOp::Rem => ("%", 10),
            BinOp::Div => {
                let kinds = self.kinds.get(&(lhs.span.hi, rhs.span.lo));
                match kinds.map(|k| k.iter().copied().collect::<Vec<_>>()).as_deref() {
                    Some([true]) => ("//", 10),
                    Some([false]) | None => ("/", 10),
                    _ => bail!("a `/` divides integers at one call and floats at another; Python spells the two apart"),
                }
            }
            BinOp::Min | BinOp::Max => unreachable!("handled above"),
        };
        // A comparison's operands are never comparisons, which Python would
        // chain; the rest are left-associative.
        let (left, right) = if prec == CMP { (CMP + 1, CMP + 1) } else { (prec, prec + 1) };
        Ok((format!("{} {sym} {}", self.expr(lhs, left)?, self.expr(rhs, right)?), prec))
    }

    fn sub(&self, s: &Sub) -> Result<String> {
        Ok(match s {
            Sub::Point(e) => self.expr(e, 0)?,
            Sub::Range { start, end } => format!("{}:{}", self.expr(start, 0)?, self.expr(end, 0)?),
            Sub::Span { start, len } => format!("{}:+{}", self.expr(start, 0)?, self.expr(len, UNARY)?),
            Sub::From(e) => format!("{}:", self.expr(e, 0)?),
            Sub::Full => ":".to_string(),
        })
    }

    fn args(&self, args: &[Expr]) -> Result<Vec<String>> {
        args.iter().map(|a| self.expr(a, 0)).collect()
    }

    fn call(&self, callee: &str, args: &[Expr]) -> Result<String> {
        if let Some(c) = ctx::spelling(callee, args) {
            return Ok(format!("ph.{c}"));
        }
        let width = |w: &Expr| -> Result<String> {
            Ok(if w.as_int() == Some(0) { String::new() } else { format!(", width={}", self.expr(w, 0)?) })
        };
        if let Some(c) = lanes::as_collective(callee, args) {
            let over = match c.group {
                Group::Lanes => "ph.ctx.lane",
                Group::Cta => "ph.ctx.thread",
            };
            let mut positional = vec![self.expr(c.value, 0)?];
            if let Some(f) = c.combine {
                positional.push(self.expr(f, 0)?);
            }
            return Ok(format!("ph.{}({}, over={over}{})", c.fold.name(), positional.join(", "), width(c.width)?));
        }
        if let Some((mode, value, operand, w)) = lanes::as_shuffle(callee, args) {
            let word = match mode.name() {
                "from" => "from_",
                other => other,
            };
            return Ok(format!("ph.shfl({}, {word}={}{})", self.expr(value, 0)?, self.expr(operand, 0)?, width(w)?));
        }
        if callee.starts_with('#') {
            bail!("`{callee}` has no source spelling");
        }
        let prefix = if self.program.contains(callee) { "" } else { "ph." };
        Ok(format!("{prefix}{}({})", name(callee)?, self.args(args)?.join(", ")))
    }
}
