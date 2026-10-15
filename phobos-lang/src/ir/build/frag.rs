// The update forms an int8 fragment accumulator takes, as source spells
// them. A generated tile read only through these stays unmaterialized; the
// fragment pass decides whether the accumulator they update stays in
// registers.

use crate::ast::{BinOp, Expr, ExprKind};

/// `acc -= e` as the parser spells it, `acc = acc - e`: the `e`.
pub(super) fn subtracted<'e>(name: &str, value: &'e Expr) -> Option<&'e Expr> {
    match &value.kind {
        ExprKind::Binary {
            op: BinOp::Sub,
            lhs,
            rhs,
        } if lhs.as_var() == Some(name) => Some(rhs),
        _ => None,
    }
}

/// The two names of `r * c`, the outer-product update's operands.
pub(super) fn outer_parts(value: &Expr) -> Option<[&str; 2]> {
    let ExprKind::Binary {
        op: BinOp::Mul,
        lhs,
        rhs,
    } = &value.kind
    else {
        return None;
    };
    Some([lhs.as_var()?, rhs.as_var()?])
}

/// `f32(dot_t(a, w)) * s1 * s2` as its four names.
pub(super) fn scaled_dot_parts(value: &Expr) -> Option<[&str; 4]> {
    let ExprKind::Binary {
        op: BinOp::Mul,
        lhs,
        rhs,
    } = &value.kind
    else {
        return None;
    };
    let ExprKind::Var(s2) = &rhs.kind else { return None };
    let ExprKind::Binary {
        op: BinOp::Mul,
        lhs: cast,
        rhs: s1,
    } = &lhs.kind
    else {
        return None;
    };
    let ExprKind::Var(s1) = &s1.kind else { return None };
    let ExprKind::Call { callee, args } = &cast.kind else { return None };
    let [Expr { kind: ExprKind::Call { callee: dot, args }, .. }] = &args[..] else { return None };
    if callee != "f32" || dot != "dot_t" {
        return None;
    }
    let [Expr { kind: ExprKind::Var(a), .. }, Expr { kind: ExprKind::Var(w), .. }] = &args[..] else { return None };
    Some([a, w, s1, s2])
}
