// A bound on how deep the parser recurses. Each nested expression, unary
// operator or block is a few stack frames, so source nested far enough
// aborts the process, where a diagnostic should have been printed instead.

use super::{PResult, Parser};
use crate::diag::Diagnostic;

/// The most levels of nesting a source may have. Real kernels stay in the
/// single digits.
pub(super) const MAX_NESTING: u32 = 64;

/// The stack a parse runs on. A debug build spends tens of kilobytes a level,
/// and the thread that called may have a megabyte, so the limit above has to
/// hold on a stack of the parser's own.
const PARSE_STACK_BYTES: usize = 32 << 20;

/// Runs `parse` on a thread with [`PARSE_STACK_BYTES`] of stack.
pub(crate) fn parse_on_big_stack<T: Send>(parse: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(PARSE_STACK_BYTES)
            .spawn_scoped(scope, parse)
            .expect("spawn the parser thread")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}

pub(super) const TOO_DEEP: &str = "this is nested too deeply";

impl Parser {
    /// Runs `parse`, one level deeper. Past [`MAX_NESTING`] it fails, at the
    /// token that would open the next level, without running it.
    pub(super) fn nested<T>(&mut self, parse: impl FnOnce(&mut Self) -> PResult<T>) -> PResult<T> {
        if self.depth >= MAX_NESTING {
            return Err(Diagnostic::error(TOO_DEEP)
                .primary(self.here(), "one level too many")
                .note(format!(
                    "expressions and blocks may nest at most {MAX_NESTING} levels"
                )));
        }
        self.depth += 1;
        let parsed = parse(self);
        self.depth -= 1;
        parsed
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_NESTING, TOO_DEEP};

    fn kernel(body: &str) -> String {
        format!("kernel k(A: tensor<f32>[N]) {{\n{body}\n}}")
    }

    /// The messages the parse failed with.
    fn errors(body: &str) -> Vec<String> {
        let err = crate::parse(&kernel(body)).unwrap_err();
        format!("{err:#}").lines().map(str::to_string).collect()
    }

    fn too_deep(body: &str) {
        let lines = errors(body);
        let count = lines.iter().filter(|l| l.contains(TOO_DEEP)).count();
        assert_eq!(count, 1, "{lines:?}");
    }

    #[test]
    fn deep_parentheses_are_a_diagnostic() {
        let n = 20_000;
        too_deep(&format!("let x = {}1{}", "(".repeat(n), ")".repeat(n)));
    }

    #[test]
    fn deep_blocks_are_a_diagnostic() {
        let n = 20_000;
        too_deep(&format!("{}{}", "if true {".repeat(n), "}".repeat(n)));
    }

    #[test]
    fn deep_unary_chains_are_a_diagnostic() {
        too_deep(&format!("let x = {}1", "-".repeat(50_000)));
    }

    #[test]
    fn nesting_below_the_limit_parses() {
        let n = MAX_NESTING as usize / 2;
        let body = format!("let x = {}1{}", "(".repeat(n), ")".repeat(n));
        crate::parse(&kernel(&body)).unwrap();
        let body = format!("{}{}", "if true {".repeat(n), "}".repeat(n));
        crate::parse(&kernel(&body)).unwrap();
    }
}
