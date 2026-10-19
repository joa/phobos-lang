// A division a front end spelled for the other kind of operand. Phobos has
// one `/`, which divides integers as integers, toward zero, and floats as
// floats; a source in another language may tell the two apart (Python's `/`
// and `//`), and only the build knows which kind it meets.

use anyhow::Result;

use super::Build;
use crate::ast::{BinOp, Expr};
use crate::diag::Spelling;
use crate::ir::AtOp;

impl Build {
    /// Fails a `/` between `lhs` and `rhs` whose spelling expects the other
    /// kind of division than the one `integer` operands get.
    pub(super) fn check_division(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr, integer: bool) -> Result<()> {
        if op != BinOp::Div {
            return Ok(());
        }
        self.report.divisions.push((crate::diag::Span::new(lhs.span.hi, rhs.span.lo), integer));
        let error = match (self.source.spelling_between(lhs.span.hi, rhs.span.lo), integer) {
            (Some(Spelling::TrueDiv), true) => AtOp::unplaced("this `/` divides two integers, as integers")
                .label("rounds toward zero, where the source's `/` gives a fraction")
                .help("write `//` to divide as integers, or convert an operand to a float type for a fraction"),
            (Some(Spelling::FloorDiv), false) => AtOp::unplaced("this `//` divides floats, which have no floor division")
                .label("a float quotient, not rounded down")
                .help("write `/` for the quotient, or `floor` of it to round down"),
            _ => return Ok(()),
        };
        self.at_span(lhs.span.to(rhs.span), |_| Err(error.into()))
    }
}
