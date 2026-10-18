// A braced block of statements, the one place statements nest.

use super::{PResult, Parser};
use crate::ast::Stmt;
use crate::token::Tok;

impl Parser {
    /// `{ stmts }`. A statement that fails to parse is reported and skipped.
    pub(super) fn parse_block(&mut self) -> PResult<Vec<Stmt>> {
        self.nested(Self::block)
    }

    fn block(&mut self) -> PResult<Vec<Stmt>> {
        let open = self.here();
        self.expect(Tok::LBrace)?;
        let mut stmts = Vec::new();
        while !self.matches(&Tok::RBrace) && !self.matches(&Tok::Eof) {
            if self.consume(&Tok::Semicolon) {
                continue;
            }
            match self.parse_stmt() {
                Ok(stmt) => stmts.push(stmt),
                Err(d) => {
                    self.report(d);
                    self.sync_stmt();
                }
            }
        }
        self.expect_close(Tok::RBrace, open)?;
        Ok(stmts)
    }
}
