// Literals the target cannot hold. An integer literal is an index, as wide
// as the kernel compiles its indices, and a float literal is an `f32`; one
// past either would otherwise change value silently, and its value would
// depend on the target. A 32-bit index holds any literal a 32-bit word can
// spell, a mask such as `2155905152` (0x80808080) as its bits.

use std::collections::HashSet;

use crate::ast::{Expr, ExprKind, Kernel, Span, UnOp};
use crate::diag::Diagnostic;

/// An error for each literal of `kernel`'s body that its index width or
/// `f32` cannot hold.
pub(super) fn check(kernel: &Kernel, index_bits: u32) -> Vec<Diagnostic> {
    // A negated literal is checked as the negative number it writes, so
    // the narrowest index's minimum is `-2147483648`.
    let mut negated: HashSet<Span> = HashSet::new();
    for stmt in &kernel.body {
        stmt.walk_exprs(&mut |e| {
            if let ExprKind::Unary { op: UnOp::Neg, rhs } = &e.kind
                && matches!(rhs.kind, ExprKind::Int(_))
            {
                negated.insert(rhs.span);
            }
        });
    }
    // The literal as its author wrote it, which a generated source may not.
    let text = |s: Span| {
        let s = kernel.source.locate(s);
        kernel.source.shown().text().get(s.lo as usize..s.hi as usize).unwrap_or("").to_string()
    };
    let mut out = Vec::new();
    let mut visit = |e: &Expr| {
        match e.kind {
            ExprKind::Int(n) if index_bits < 64 && e.span != Span::default() => {
                let value = if negated.contains(&e.span) { -(n as i128) } else { n as i128 };
                if i32::try_from(value).is_err() && u32::try_from(value).is_err() {
                    out.push(
                        Diagnostic::error(format!("integer literal `{}` does not fit a 32-bit index", text(e.span)))
                            .primary(e.span, format!("would be {}", value as i32))
                            .note("an integer literal is an index, 32 bits wide unless the kernel compiles at 64")
                            .note("one up to 4294967295 is the 32-bit word of its bits"),
                    );
                }
            }
            ExprKind::Float(v) if e.span != Span::default() => {
                let narrowed = v as f32;
                let what = if narrowed.is_infinite() && v.is_finite() {
                    "infinite"
                } else if narrowed == 0.0 && v != 0.0 {
                    "zero"
                } else {
                    return;
                };
                out.push(
                    Diagnostic::error(format!("float literal `{}` is {what} as an `f32`", text(e.span)))
                        .primary(e.span, "past what an `f32` holds")
                        .note("a float literal is an `f32`, whatever it meets"),
                );
            }
            _ => {}
        }
    };
    for stmt in &kernel.body {
        stmt.walk_exprs(&mut visit);
    }
    out
}
