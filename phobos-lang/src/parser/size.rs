// How many nodes a tree of the AST has, which bounds what inlining may build.

use super::super::ast::{Expr, ExprKind, Stmt, Sub};

pub(super) fn expr(e: &Expr) -> usize {
    1 + match &e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Var(_) => 0,
        ExprKind::Unary { rhs, .. } => expr(rhs),
        ExprKind::Binary { lhs, rhs, .. } => expr(lhs) + expr(rhs),
        ExprKind::Index { base, subs } => expr(base) + subs.iter().map(sub).sum::<usize>(),
        ExprKind::Call { args, .. } => args.iter().map(expr).sum(),
        ExprKind::Generate(g) => block(&g.body) + expr(&g.value),
    }
}

fn sub(s: &Sub) -> usize {
    match s {
        Sub::Point(e) | Sub::From(e) => expr(e),
        Sub::Range { start: a, end: b } | Sub::Span { start: a, len: b } => expr(a) + expr(b),
        Sub::Full => 0,
    }
}

pub(super) fn block(body: &[Stmt]) -> usize {
    body.iter().map(stmt).sum()
}

fn stmt(s: &Stmt) -> usize {
    1 + match s {
        Stmt::Let { value, .. } => expr(value),
        Stmt::Var { value, .. } => value.as_ref().map_or(0, expr),
        Stmt::Assign { target, value, .. } => expr(target) + expr(value),
        Stmt::For { start, end, step, unroll, body, .. } => {
            expr(start)
                + expr(end)
                + step.as_ref().map_or(0, expr)
                + unroll.as_ref().map_or(0, expr)
                + block(body)
        }
        Stmt::While { cond, body, .. } => expr(cond) + block(body),
        Stmt::If { cond, then, r#else, .. } => {
            expr(cond) + block(then) + r#else.as_deref().map_or(0, block)
        }
        Stmt::Expr(e, _) => expr(e),
    }
}

#[cfg(test)]
mod tests {
    fn doubling(depth: usize) -> String {
        let call = format!("{}1.0{}", "d(".repeat(depth), ")".repeat(depth));
        format!("fn d(x) = x + x\nkernel k(A: tensor<f32>[N]) {{\nA[0] = {call}\n}}")
    }

    #[test]
    fn nested_doubling_calls_are_refused_not_expanded() {
        let started = std::time::Instant::now();
        let err = format!("{:#}", crate::parse(&doubling(30)).unwrap_err());
        assert!(err.contains("expands to more than"), "{err}");
        assert!(err.contains("`d`"), "{err}");
        assert!(started.elapsed().as_secs() < 5, "took {:?}", started.elapsed());
    }

    #[test]
    fn modest_nesting_still_inlines() {
        crate::parse(&doubling(8)).unwrap();
    }
}
