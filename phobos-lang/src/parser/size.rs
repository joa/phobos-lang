// How many nodes a tree of the AST has, which bounds what inlining may build.

use super::super::ast::{Expr, ExprKind, Stmt};

pub(super) fn expr(e: &Expr) -> usize {
    let generated = match &e.kind {
        ExprKind::Generate(g) => block(&g.body) + expr(&g.value),
        _ => 0,
    };
    1 + generated + e.kind.children().map(expr).sum::<usize>()
}

pub(super) fn block(body: &[Stmt]) -> usize {
    body.iter()
        .map(|s| 1 + s.exprs().map(expr).sum::<usize>() + s.blocks().map(block).sum::<usize>())
        .sum()
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
