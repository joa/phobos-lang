// `name<config>(args)`: the collectives and the shuffle.
//
// The words inside the angle brackets are keywords only there, so `from`
// stays an ordinary name everywhere else. A name followed by `<` and one of
// them cannot start a comparison, so one token of lookahead decides it.

use super::{PResult, Parser};
use crate::ast::lanes::{self, Fold, Group, Mode};
use crate::ast::{Expr, ctx};
use crate::diag::Diagnostic;
use crate::token::Tok;

const WORDS: [&str; 6] = ["over", "width", "from", "xor", "down", "up"];

impl Parser {
    /// Whether the tokens ahead open a `<config>` list.
    pub(super) fn at_config(&self) -> bool {
        self.matches(&Tok::Lt)
            && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Ident(w)) if WORDS.contains(&w.as_str()))
    }

    /// `name<config>(args)`, the `<` next.
    pub(super) fn config_call(&mut self, name: &str) -> PResult<Expr> {
        let open = self.here();
        let at = |msg: String| Diagnostic::error(msg).primary(open, "");
        self.expect(Tok::Lt)?;
        let mut items: Vec<(String, Expr)> = Vec::new();
        loop {
            let word_at = self.here();
            let at = |msg: String| Diagnostic::error(msg).primary(word_at, "");
            let word = self.field()?;
            if !WORDS.contains(&word.as_str()) {
                return Err(at(format!("`{word}` is not a setting of `{name}<..>`")));
            }
            if items.iter().any(|(w, _)| *w == word) {
                return Err(at(format!("`{word}` is given twice")));
            }
            // A `>` closes the list, so a value stops short of comparisons
            // and shifts.
            let value = if word == "over" { self.postfix()? } else { self.term()? };
            items.push((word, value));
            if !self.consume(&Tok::Comma) {
                break;
            }
        }
        self.expect(Tok::Gt)?;
        self.expect(Tok::LParen)?;
        let paren = self.here();
        let args = if self.matches(&Tok::RParen) { Vec::new() } else { self.parse_args()? };
        self.expect_close(Tok::RParen, paren)?;

        let width = take(&mut items, "width").unwrap_or(Expr::Int(0));
        let out = if let Some(fold) = Fold::from_name(name) {
            let group = match take(&mut items, "over") {
                Some(Expr::Call { callee, .. }) if callee == ctx::LANE => Group::Lanes,
                Some(Expr::Call { callee, .. }) if callee == ctx::THREAD => Group::Cta,
                Some(_) => return Err(at(format!("`{name}<over ..>` takes `ctx.lane` or `ctx.thread`"))),
                None => return Err(at(format!("`{name}<..>` needs `over ctx.lane` or `over ctx.thread`"))),
            };
            if group == Group::Cta && !matches!(width, Expr::Int(0)) {
                return Err(at(format!("`{name}<over ctx.thread>` runs across the CTA and takes no `width`")));
            }
            let want = if fold == Fold::Reduce { 2 } else { 1 };
            if args.len() != want {
                return Err(at(format!("`{name}<..>` takes {want} argument(s), got {}", args.len())));
            }
            let mut args = args.into_iter();
            let value = args.next().expect("counted above");
            lanes::collective(fold, value, group, width, args.next())
        } else if name == "shfl" {
            let modes: Vec<Mode> = Mode::ALL.into_iter().filter(|m| items.iter().any(|(w, _)| w == m.name())).collect();
            let [mode] = modes[..] else {
                return Err(at("`shfl<..>` takes one of `from`, `xor`, `down` or `up`".into()));
            };
            let operand = take(&mut items, mode.name()).expect("found above");
            let [value] = <[Expr; 1]>::try_from(args).map_err(|a| at(format!("`shfl<..>` takes 1 argument, got {}", a.len())))?;
            lanes::shuffle(mode, value, operand, width)
        } else {
            return Err(at(format!("`{name}` takes no `<..>` settings")));
        };
        if let Some((word, _)) = items.first() {
            return Err(at(format!("`{word}` is not a setting of `{name}<..>`")));
        }
        Ok(out)
    }
}

/// Removes and returns the value given for `word`.
fn take(items: &mut Vec<(String, Expr)>, word: &str) -> Option<Expr> {
    let i = items.iter().position(|(w, _)| w == word)?;
    Some(items.remove(i).1)
}
