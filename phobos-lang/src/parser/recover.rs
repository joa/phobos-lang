// What the parser says when the source is not what it expected, and where it
// picks up again: a broken statement is skipped to its end, a broken item to
// the next `kernel`, `fn` or attribute, so one run reports every mistake that
// does not depend on another.

use super::Parser;
use crate::diag::{self, Diagnostic, Span};
use crate::token::Tok;

impl Parser {
    /// The current token's span. A token the source does not spell, the end
    /// of a line or of the file, is placed just past the last one consumed,
    /// where what is missing would go.
    pub(super) fn here(&self) -> Span {
        let cur = self.current();
        if cur.lo == cur.hi {
            Span::new(self.last_hi, self.last_hi)
        } else {
            cur.span()
        }
    }

    /// The current token, as a message names it.
    pub(super) fn found(&self) -> String {
        let cur = self.current();
        match cur.tok {
            Tok::Semicolon if cur.lo == cur.hi => "end of line".into(),
            ref t => t.to_string(),
        }
    }

    /// "expected `what`, found ...", pointing at the current token.
    pub(super) fn unexpected(&self, what: &str) -> Diagnostic {
        Diagnostic::error(format!("expected {what}, found {}", self.found()))
            .primary(self.here(), format!("expected {what}"))
    }

    /// A closing delimiter, the one that matches `open`.
    pub(super) fn expect_close(&mut self, close: Tok, open: Span) -> super::PResult<()> {
        if self.consume(&close) {
            return Ok(());
        }
        let d = if self.matches(&Tok::Eof) {
            Diagnostic::error("this file contains an unclosed delimiter").primary(self.here(), "")
        } else {
            self.unexpected(&close.to_string())
        };
        Err(d.secondary(open, "unclosed delimiter"))
    }

    /// Keeps `d` for the end of the parse, up to [`diag::MAX_ERRORS`].
    pub(super) fn report(&mut self, d: Diagnostic) {
        // Every enclosing block recovers and meets the same limit again.
        if d.message == super::depth::TOO_DEEP && self.errors.iter().any(|e| e.message == d.message) {
            return;
        }
        if self.errors.len() < diag::MAX_ERRORS {
            self.errors.push(d);
        }
    }

    /// Skips past the end of a statement that failed: to its terminator, or
    /// up to the `}` closing its block, which the block then takes.
    pub(super) fn sync_stmt(&mut self) {
        let mut depth = 0usize;
        loop {
            match self.peek() {
                Tok::Eof => return,
                Tok::RBrace if depth == 0 => return,
                Tok::Semicolon if depth == 0 => {
                    self.advance();
                    return;
                }
                Tok::LBrace => depth += 1,
                Tok::RBrace => depth -= 1,
                _ => {}
            }
            self.advance();
        }
    }

    /// Skips past an item that failed, which began at token `start`: to the
    /// next `kernel`, `fn` or attribute outside any braces.
    pub(super) fn sync_item(&mut self, start: usize) {
        if self.pos == start {
            self.advance();
        }
        let mut depth = 0usize;
        loop {
            match self.peek() {
                Tok::Eof => return,
                Tok::Kernel | Tok::Fn | Tok::At if depth == 0 => return,
                Tok::LBrace => depth += 1,
                Tok::RBrace => depth = depth.saturating_sub(1),
                _ => {}
            }
            self.advance();
        }
    }
}

/// `d` with a help to write the closest of `candidates` at `at`, when one is
/// close to `name`.
pub(super) fn did_you_mean<'a>(
    d: Diagnostic,
    at: Span,
    name: &str,
    candidates: impl IntoIterator<Item = &'a str>,
) -> Diagnostic {
    match diag::similar(name, candidates) {
        Some(near) => d.suggest(format!("a similar name exists: `{near}`"), at, near),
        None => d,
    }
}
