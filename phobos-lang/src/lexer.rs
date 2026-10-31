use super::ast::{BinOp, Scalar};
use super::diag::{Class, Diagnostic, Span};
use super::token::{Tok, Token};

pub struct Lexer<'a> {
    src: std::str::Chars<'a>,
    peek: Option<char>,
    /// The byte offset of `peek`.
    pos: u32,
}

/// The tokens one character makes, whatever follows it.
static SINGLE: [(char, Tok); 11] = [
    ('(', Tok::LParen),
    (')', Tok::RParen),
    ('{', Tok::LBrace),
    ('}', Tok::RBrace),
    ('[', Tok::LBracket),
    (']', Tok::RBracket),
    (',', Tok::Comma),
    (';', Tok::Semicolon),
    ('@', Tok::At),
    ('.', Tok::Dot),
    ('~', Tok::Tilde),
];

/// What one step of the lexer read.
enum Piece {
    Tok(Tok),
    Newline,
    Space,
    Comment,
    Bad(Diagnostic),
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut chars = input.chars();
        let peek = chars.next();
        Lexer {
            src: chars,
            peek,
            pos: 0,
        }
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek;
        self.peek = self.src.next();
        if let Some(c) = c {
            self.pos += c.len_utf8() as u32;
        }
        c
    }

    fn consume(&mut self, c: char) -> bool {
        if self.peek == Some(c) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn ends_stmt(t: &Tok) -> bool {
        matches!(
            t,
            Tok::Ident(_)
                | Tok::Int(_)
                | Tok::Float(_)
                | Tok::True
                | Tok::False
                | Tok::RParen
                | Tok::RBracket
                | Tok::RBrace
        )
    }

    /// The tokens of the source, with a terminator inserted at each newline
    /// that can end a statement, or every malformed token in it.
    pub fn tokenize(mut self) -> Result<Vec<Token>, Vec<Diagnostic>> {
        let mut out: Vec<Token> = Vec::new();
        let mut errors = Vec::new();
        loop {
            let lo = self.pos;
            let Some(piece) = self.piece() else {
                break;
            };
            match piece {
                Piece::Newline if out.last().is_some_and(|t| Self::ends_stmt(&t.tok)) => {
                    out.push(Token { tok: Tok::Semicolon, lo, hi: lo });
                }
                Piece::Newline | Piece::Space | Piece::Comment => {}
                Piece::Bad(d) => errors.push(d),
                Piece::Tok(tok) => out.push(Token { tok, lo, hi: self.pos }),
            }
        }
        if !errors.is_empty() {
            return Err(errors);
        }
        let at = self.pos;
        if out.last().is_some_and(|t| Self::ends_stmt(&t.tok)) {
            out.push(Token { tok: Tok::Semicolon, lo: at, hi: at });
        }
        out.push(Token { tok: Tok::Eof, lo: at, hi: at });
        Ok(out)
    }

    /// Reads one token, or the whitespace, comment or malformed text
    /// between tokens.
    fn piece(&mut self) -> Option<Piece> {
        let lo = self.pos;
        let c = self.peek?;
        if let Some((_, tok)) = SINGLE.iter().find(|(s, _)| *s == c) {
            self.advance();
            return Some(Piece::Tok(tok.clone()));
        }
        let tok = match c {
            '\n' => {
                self.advance();
                return Some(Piece::Newline);
            }
            c if c.is_whitespace() => {
                self.advance();
                return Some(Piece::Space);
            }
            '/' => {
                self.advance();
                if self.peek == Some('/') {
                    while self.peek.is_some() && self.peek != Some('\n') {
                        self.advance();
                    }
                    return Some(Piece::Comment);
                }
                self.op_eq(BinOp::Div, Tok::Slash)
            }
            '*' => {
                self.advance();
                self.op_eq(BinOp::Mul, Tok::Star)
            }
            '%' => {
                self.advance();
                self.op_eq(BinOp::Rem, Tok::Percent)
            }
            '&' => {
                self.advance();
                if self.consume('&') {
                    Tok::AmpAmp
                } else {
                    self.op_eq(BinOp::BitAnd, Tok::Amp)
                }
            }
            '|' => {
                self.advance();
                if self.consume('|') {
                    Tok::PipePipe
                } else {
                    self.op_eq(BinOp::BitOr, Tok::Pipe)
                }
            }
            '^' => {
                self.advance();
                self.op_eq(BinOp::BitXor, Tok::Caret)
            }
            '+' => {
                self.advance();
                if self.consume('=') {
                    Tok::PlusEq
                } else {
                    Tok::Plus
                }
            }
            '-' => {
                self.advance();
                self.op_eq(BinOp::Sub, Tok::Minus)
            }
            ':' => {
                self.advance();
                if self.consume('+') {
                    Tok::ColonPlus
                } else {
                    Tok::Colon
                }
            }
            '<' => {
                self.advance();
                if self.consume('<') {
                    self.op_eq(BinOp::Shl, Tok::Shl)
                } else if self.consume('=') {
                    Tok::Le
                } else {
                    Tok::Lt
                }
            }
            '>' => {
                self.advance();
                if self.consume('>') {
                    self.op_eq(BinOp::Shr, Tok::Shr)
                } else if self.consume('=') {
                    Tok::Ge
                } else {
                    Tok::Gt
                }
            }
            '=' => {
                self.advance();
                if self.consume('=') {
                    Tok::EqEq
                } else {
                    Tok::Eq
                }
            }
            '!' => {
                self.advance();
                if self.consume('=') {
                    Tok::NotEq
                } else {
                    Tok::Bang
                }
            }
            c if c.is_ascii_digit() => match self.number(lo) {
                Ok(tok) => tok,
                Err(d) => return Some(Piece::Bad(d)),
            },
            c if c.is_alphabetic() || c == '_' => self.ident_or_keyword(),
            other => {
                self.advance();
                return Some(Piece::Bad(
                    Diagnostic::error(format!("unexpected character `{other}`"))
                        .primary(Span::new(lo, self.pos), "not a token"),
                ));
            }
        };
        Some(Piece::Tok(tok))
    }

    /// `tok`, or its compound assignment when an `=` follows.
    fn op_eq(&mut self, op: BinOp, tok: Tok) -> Tok {
        if self.consume('=') { Tok::OpEq(op) } else { tok }
    }

    fn number(&mut self, lo: u32) -> Result<Tok, Diagnostic> {
        let mut s = String::new();
        let mut is_float = false;
        while let Some(c) = self.peek {
            if c.is_ascii_digit() {
                s.push(c);
                self.advance();
            } else if c == '.' && !is_float {
                is_float = true;
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        // An exponent, `1e-6` or `2.5E3`, makes a float of any number.
        if matches!(self.peek, Some('e' | 'E')) {
            is_float = true;
            s.push('e');
            self.advance();
            if let Some(sign @ ('+' | '-')) = self.peek {
                s.push(sign);
                self.advance();
            }
            if !self.peek.is_some_and(|c| c.is_ascii_digit()) {
                return Err(Diagnostic::error("expected at least one digit in exponent")
                    .primary(Span::new(lo, self.pos), format!("`{s}` has no digits after its `e`")));
            }
            while let Some(c) = self.peek.filter(char::is_ascii_digit) {
                s.push(c);
                self.advance();
            }
        }
        let span = Span::new(lo, self.pos);
        if is_float {
            s.parse::<f64>().map(Tok::Float).map_err(|_| {
                Diagnostic::error(format!("malformed float literal `{s}`")).primary(span, "")
            })
        } else {
            s.parse::<i64>().map(Tok::Int).map_err(|_| {
                Diagnostic::error("integer literal is too large")
                    .primary(span, "")
                    .note("an integer literal is a signed 64-bit value")
            })
        }
    }

    fn ident_or_keyword(&mut self) -> Tok {
        let mut s = String::new();
        while let Some(c) = self.peek {
            if c.is_alphanumeric() || c == '_' {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        Tok::keyword(&s).unwrap_or(Tok::Ident(s))
    }
}

/// The highlighted pieces of `text`, in order, for a diagnostic's snippet.
/// Never fails: malformed text is left plain.
pub fn classify(text: &str) -> Vec<(Span, Class)> {
    let mut lexer = Lexer::new(text);
    let mut out = Vec::new();
    let mut toks: Vec<(Tok, Span)> = Vec::new();
    loop {
        let lo = lexer.pos;
        let Some(piece) = lexer.piece() else {
            break;
        };
        let span = Span::new(lo, lexer.pos);
        match piece {
            Piece::Tok(tok) => toks.push((tok, span)),
            Piece::Comment => out.push((span, Class::Comment)),
            Piece::Newline | Piece::Space | Piece::Bad(_) => {}
        }
    }
    let mut in_ctx = false;
    for (i, (tok, span)) in toks.iter().enumerate() {
        let next = toks.get(i + 1).map(|(t, _)| t);
        let class = match tok {
            _ if tok.keyword_text().is_some() => Some(Class::Keyword),
            Tok::Int(_) | Tok::Float(_) => Some(Class::Number),
            Tok::At if matches!(next, Some(Tok::Ident(_))) => Some(Class::Attribute),
            Tok::Ident(_) if i > 0 && toks[i - 1].0 == Tok::At => Some(Class::Attribute),
            Tok::Ident(s) if s == crate::ast::ctx::NAME => {
                in_ctx = true;
                Some(Class::Ctx)
            }
            Tok::Dot if in_ctx => Some(Class::Ctx),
            Tok::Ident(_) if in_ctx && i > 0 && toks[i - 1].0 == Tok::Dot => Some(Class::Ctx),
            Tok::Ident(s) if Scalar::from_name(s).is_some() || s == "tile" || s == "tensor" => Some(Class::Type),
            Tok::Ident(_) if next == Some(&Tok::LParen) => Some(Class::Call),
            _ => None,
        };
        if !matches!(tok, Tok::Dot) && class != Some(Class::Ctx) {
            in_ctx = false;
        }
        if let Some(class) = class {
            out.push((*span, class));
        }
    }
    out.sort_by_key(|(s, _)| s.lo);
    out
}

#[cfg(test)]
mod tests;
