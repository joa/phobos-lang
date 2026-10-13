// `break`, `continue` and `return`, lowered to flags before anything past the
// parser sees the kernel.
//
// Control flow is CTA-uniform, so a jump is a uniform scalar flag. A jump
// sets its flag; every statement after one that may have jumped runs under
// `if !flags`; a `while` also tests the flags in its condition. A `for`
// keeps its counted form, and the alignment proofs that come with it, and
// runs its remaining iterations as no-ops.

use super::super::ast::{AssignOp, BinOp, Expr, Stmt, UnOp};

/// The placeholders the parser emits for the three jumps. A name the lexer
/// cannot produce, so no source can spell one.
pub(super) const BREAK: &str = "#break";
pub(super) const CONTINUE: &str = "#continue";
pub(super) const RETURN: &str = "#return";

/// A jump statement as the parser emits it.
pub(super) fn jump(which: &str) -> Stmt {
    Stmt::Expr(Expr::Var(which.into()))
}

fn jump_of(stmt: &Stmt) -> Option<&str> {
    match stmt {
        Stmt::Expr(Expr::Var(name)) if [BREAK, CONTINUE, RETURN].contains(&name.as_str()) => {
            Some(name.as_str())
        }
        _ => None,
    }
}

/// The flags of the innermost loop. A flag is only declared when its jump
/// is used.
struct Loop {
    brk: Option<String>,
    cont: Option<String>,
}

#[derive(Default)]
struct Desugar {
    next: usize,
    ret: Option<String>,
}

/// The jump in a function body that would leave it: a `return`, or a
/// `break` or `continue` outside the body's own loops.
pub(super) fn escaping(body: &[Stmt]) -> Option<&'static str> {
    [RETURN, BREAK, CONTINUE]
        .into_iter()
        .find(|&which| body.iter().any(|s| uses(s, which, which == RETURN)))
        .map(|which| &which[1..])
}

/// Lowers every jump in a kernel body.
pub(super) fn desugar(body: Vec<Stmt>) -> Result<Vec<Stmt>, String> {
    let any = |which| body.iter().any(|s| uses(s, which, true));
    if !any(RETURN) && !any(BREAK) && !any(CONTINUE) {
        return Ok(body);
    }
    let mut d = Desugar::default();
    let mut out = Vec::new();
    if body.iter().any(|s| uses(s, RETURN, true)) {
        let flag = d.fresh("return");
        out.push(declare(&flag));
        d.ret = Some(flag);
    }
    out.extend(d.block(body, None)?);
    Ok(out)
}

/// Whether `stmt` contains the jump `which`. With `through_loops` false, a
/// `break` or `continue` inside a nested loop belongs to that loop and does
/// not count.
fn uses(stmt: &Stmt, which: &str, through_loops: bool) -> bool {
    if jump_of(stmt) == Some(which) {
        return true;
    }
    let any = |body: &[Stmt]| body.iter().any(|s| uses(s, which, through_loops));
    match stmt {
        Stmt::If { then, r#else, .. } => any(then) || r#else.as_deref().is_some_and(any),
        Stmt::For { body, .. } | Stmt::While { body, .. } => {
            (through_loops || which == RETURN) && any(body)
        }
        _ => false,
    }
}

fn declare(flag: &str) -> Stmt {
    Stmt::Var {
        name: flag.into(),
        ty: None,
        value: Some(Expr::Bool(false)),
    }
}

fn set(flag: &str, value: bool) -> Stmt {
    Stmt::Assign {
        target: Expr::Var(flag.into()),
        op: AssignOp::Set,
        value: Expr::Bool(value),
    }
}

/// `!(a || b || ...)`.
fn none_set(flags: &[&String]) -> Expr {
    let any = flags
        .iter()
        .map(|f| Expr::Var((*f).clone()))
        .reduce(|lhs, rhs| Expr::Binary {
            op: BinOp::Or,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        })
        .expect("at least one flag");
    Expr::Unary {
        op: UnOp::Not,
        rhs: Box::new(any),
    }
}

impl Desugar {
    fn fresh(&mut self, what: &str) -> String {
        self.next += 1;
        format!("#{what}{}", self.next)
    }

    /// The flags a statement in `lp`'s body may have set.
    fn live<'a>(&'a self, lp: Option<&'a Loop>) -> Vec<&'a String> {
        let mut flags: Vec<&String> = Vec::new();
        if let Some(lp) = lp {
            flags.extend(lp.brk.iter());
            flags.extend(lp.cont.iter());
        }
        flags.extend(self.ret.iter());
        flags
    }

    /// Whether `stmt` may leave the rest of its block unrun.
    fn may_jump(stmt: &Stmt) -> bool {
        uses(stmt, RETURN, true) || uses(stmt, BREAK, false) || uses(stmt, CONTINUE, false)
    }

    fn block(&mut self, stmts: Vec<Stmt>, lp: Option<&Loop>) -> Result<Vec<Stmt>, String> {
        let mut out = Vec::new();
        let mut rest = stmts.into_iter();
        while let Some(stmt) = rest.next() {
            let jumps = Self::may_jump(&stmt);
            out.extend(self.stmt(stmt, lp)?);
            if jumps && rest.len() > 0 {
                let tail: Vec<Stmt> = rest.collect();
                let cond = none_set(&self.live(lp));
                let then = self.block(tail, lp)?;
                out.push(Stmt::If {
                    cond,
                    then,
                    r#else: None,
                });
                return Ok(out);
            }
        }
        Ok(out)
    }

    fn stmt(&mut self, stmt: Stmt, lp: Option<&Loop>) -> Result<Vec<Stmt>, String> {
        if let Some(which) = jump_of(&stmt) {
            let flag = match which {
                RETURN => self.ret.clone(),
                BREAK => lp.and_then(|l| l.brk.clone()),
                _ => lp.and_then(|l| l.cont.clone()),
            };
            let Some(flag) = flag else {
                return Err(format!("`{}` outside a loop", &which[1..]));
            };
            return Ok(vec![set(&flag, true)]);
        }
        Ok(match stmt {
            Stmt::If { cond, then, r#else } => vec![Stmt::If {
                cond,
                then: self.block(then, lp)?,
                r#else: r#else.map(|e| self.block(e, lp)).transpose()?,
            }],
            Stmt::For {
                var,
                start,
                end,
                step,
                unroll,
                body,
            } => {
                let (mut out, inner) = self.enter(&body);
                let mut body = self.loop_body(body, &inner)?;
                let stops: Vec<&String> = inner.brk.iter().chain(self.ret.iter()).collect();
                if !stops.is_empty() {
                    body = vec![Stmt::If {
                        cond: none_set(&stops),
                        then: body,
                        r#else: None,
                    }];
                }
                out.push(Stmt::For {
                    var,
                    start,
                    end,
                    step,
                    unroll,
                    body,
                });
                out
            }
            Stmt::While { cond, body } => {
                let (mut out, inner) = self.enter(&body);
                let body = self.loop_body(body, &inner)?;
                let stops: Vec<&String> = inner.brk.iter().chain(self.ret.iter()).collect();
                let cond = if stops.is_empty() {
                    cond
                } else {
                    Expr::Binary {
                        op: BinOp::And,
                        lhs: Box::new(cond),
                        rhs: Box::new(none_set(&stops)),
                    }
                };
                out.push(Stmt::While { cond, body });
                out
            }
            other => vec![other],
        })
    }

    /// Declares the flags a loop's own jumps need, ahead of the loop.
    fn enter(&mut self, body: &[Stmt]) -> (Vec<Stmt>, Loop) {
        let own = |which| body.iter().any(|s| uses(s, which, false));
        let brk = own(BREAK).then(|| self.fresh("break"));
        let cont = own(CONTINUE).then(|| self.fresh("continue"));
        let decls = brk.iter().chain(cont.iter()).map(|f| declare(f)).collect();
        (decls, Loop { brk, cont })
    }

    /// A loop body, its `continue` flag cleared at the top of each iteration.
    fn loop_body(&mut self, body: Vec<Stmt>, lp: &Loop) -> Result<Vec<Stmt>, String> {
        let mut out: Vec<Stmt> = lp.cont.iter().map(|f| set(f, false)).collect();
        out.extend(self.block(body, Some(lp))?);
        Ok(out)
    }
}
