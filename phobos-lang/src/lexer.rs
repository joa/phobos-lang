use super::token::{Tok, Token};

pub struct Lexer<'a> {
    src: std::str::Chars<'a>,
    peek: Option<char>,
    line: u32,
    col: u32,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut chars = input.chars();
        let peek = chars.next();
        Lexer {
            src: chars,
            peek,
            line: 1,
            col: 1,
        }
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek;
        self.peek = self.src.next();
        match c {
            Some('\n') => {
                self.line += 1;
                self.col = 1;
            }
            Some(_) => self.col += 1,
            None => {}
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

    pub fn tokenize(mut self) -> Result<Vec<Token>, String> {
        let mut out: Vec<Token> = Vec::new();
        while let Some(c) = self.peek {
            let (line, col) = (self.line, self.col);
            let tok = match c {
                c if c.is_whitespace() => {
                    if c == '\n' && out.last().is_some_and(|t| Self::ends_stmt(&t.tok)) {
                        out.push(Token {
                            tok: Tok::Semicolon,
                            line,
                            col,
                        });
                    }
                    self.advance();
                    continue;
                }
                '/' => {
                    self.advance();
                    if self.peek == Some('/') {
                        while self.peek.is_some() && self.peek != Some('\n') {
                            self.advance();
                        }
                        continue;
                    }
                    Tok::Slash
                }
                '(' => {
                    self.advance();
                    Tok::LParen
                }
                ')' => {
                    self.advance();
                    Tok::RParen
                }
                '{' => {
                    self.advance();
                    Tok::LBrace
                }
                '}' => {
                    self.advance();
                    Tok::RBrace
                }
                '[' => {
                    self.advance();
                    Tok::LBracket
                }
                ']' => {
                    self.advance();
                    Tok::RBracket
                }
                ',' => {
                    self.advance();
                    Tok::Comma
                }
                ';' => {
                    self.advance();
                    Tok::Semicolon
                }
                '@' => {
                    self.advance();
                    Tok::At
                }
                '*' => {
                    self.advance();
                    Tok::Star
                }
                '%' => {
                    self.advance();
                    Tok::Percent
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
                    Tok::Minus
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
                    if self.consume('=') { Tok::Le } else { Tok::Lt }
                }
                '>' => {
                    self.advance();
                    if self.consume('=') { Tok::Ge } else { Tok::Gt }
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
                c if c.is_ascii_digit() => self.number()?,
                c if c.is_alphabetic() || c == '_' => self.ident_or_keyword(),
                other => return Err(format!("{}:{}: unexpected char '{}'", line, col, other)),
            };
            out.push(Token { tok, line, col });
        }
        if out.last().is_some_and(|t| Self::ends_stmt(&t.tok)) {
            out.push(Token {
                tok: Tok::Semicolon,
                line: self.line,
                col: self.col,
            });
        }
        out.push(Token {
            tok: Tok::Eof,
            line: self.line,
            col: self.col,
        });
        Ok(out)
    }

    fn number(&mut self) -> Result<Tok, String> {
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
        if is_float {
            s.parse::<f64>()
                .map(Tok::Float)
                .map_err(|_| format!("bad float literal '{}'", s))
        } else {
            s.parse::<i64>()
                .map(Tok::Int)
                .map_err(|_| format!("bad int literal '{}'", s))
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
        match s.as_str() {
            "kernel" => Tok::Kernel,
            "let" => Tok::Let,
            "var" => Tok::Var,
            "if" => Tok::If,
            "else" => Tok::Else,
            "for" => Tok::For,
            "in" => Tok::In,
            "while" => Tok::While,
            "true" => Tok::True,
            "false" => Tok::False,
            _ => Tok::Ident(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexes_slice_and_pluseq_and_at() {
        let toks = Lexer::new("@a acc += A[i : T]").tokenize().unwrap();
        let kinds: Vec<Tok> = toks.into_iter().map(|t| t.tok).collect();
        assert!(kinds.contains(&Tok::At));
        assert!(kinds.contains(&Tok::PlusEq));
        assert!(kinds.contains(&Tok::Colon));
    }

    #[test]
    fn lexes_colon_and_colonplus() {
        let toks = Lexer::new("@a acc += A[start : end, start :+ length]")
            .tokenize()
            .unwrap();
        let kinds: Vec<Tok> = toks.into_iter().map(|t| t.tok).collect();
        assert!(kinds.contains(&Tok::Colon));
        assert!(kinds.contains(&Tok::ColonPlus));
    }

    #[test]
    fn inserts_terminators_at_newlines() {
        // after 1, 3, and the final }, but not after the trailing +
        let toks = Lexer::new("let x = 1\nx += 2 +\n  3\n}")
            .tokenize()
            .unwrap();
        let kinds: Vec<Tok> = toks.into_iter().map(|t| t.tok).collect();
        let semis = kinds.iter().filter(|t| **t == Tok::Semicolon).count();
        assert_eq!(semis, 3);
        // let x = 1 ;
        assert_eq!(kinds[4], Tok::Semicolon);
        // 2 + continues the statement: no terminator between + and 3
        assert_eq!(kinds[8], Tok::Plus);
        assert_eq!(kinds[9], Tok::Int(3));
    }

    #[test]
    fn for_in_are_keywords() {
        let toks = Lexer::new("for i in").tokenize().unwrap();
        assert_eq!(toks[0].tok, Tok::For);
        assert_eq!(toks[1].tok, Tok::Ident("i".to_string()));
        assert_eq!(toks[2].tok, Tok::In);
    }

    fn kinds(src: &str) -> Vec<Tok> {
        Lexer::new(src)
            .tokenize()
            .unwrap()
            .into_iter()
            .map(|t| t.tok)
            .collect()
    }

    #[test]
    fn all_keywords_lex() {
        assert_eq!(
            kinds("kernel let var if else for in while true false")
                .into_iter()
                .take(10)
                .collect::<Vec<_>>(),
            vec![
                Tok::Kernel,
                Tok::Let,
                Tok::Var,
                Tok::If,
                Tok::Else,
                Tok::For,
                Tok::In,
                Tok::While,
                Tok::True,
                Tok::False,
            ]
        );
    }

    #[test]
    fn punctuation_and_brackets() {
        assert_eq!(
            kinds("( ) { } [ ] , ; @"),
            vec![
                Tok::LParen,
                Tok::RParen,
                Tok::LBrace,
                // a newline-insertion fires after } since it ends a statement
                Tok::RBrace,
                Tok::LBracket,
                Tok::RBracket,
                Tok::Comma,
                Tok::Semicolon,
                Tok::At,
                Tok::Eof,
            ]
        );
    }

    #[test]
    fn comparison_and_equality_operators() {
        assert_eq!(
            kinds("< <= > >= == != ! ="),
            vec![
                Tok::Lt,
                Tok::Le,
                Tok::Gt,
                Tok::Ge,
                Tok::EqEq,
                Tok::NotEq,
                Tok::Bang,
                Tok::Eq,
                Tok::Eof,
            ]
        );
    }

    #[test]
    fn arithmetic_operators() {
        assert_eq!(
            kinds("+ += - * / %"),
            vec![
                Tok::Plus,
                Tok::PlusEq,
                Tok::Minus,
                Tok::Star,
                Tok::Slash,
                Tok::Percent,
                Tok::Eof,
            ]
        );
    }

    #[test]
    fn line_comments_are_skipped() {
        // a comment runs to end of line and produces no tokens
        let k = kinds("let x = 1 // trailing comment\n// whole line\nlet y = 2");
        assert!(k.contains(&Tok::Let));
        assert!(k.contains(&Tok::Int(1)));
        assert!(k.contains(&Tok::Int(2)));
        // the // itself never becomes a Slash
        assert!(!k.contains(&Tok::Slash));
    }

    #[test]
    fn int_and_float_literals() {
        assert_eq!(kinds("42")[0], Tok::Int(42));
        assert_eq!(kinds("2.5")[0], Tok::Float(2.5));
        // a trailing dot still parses as a float
        assert_eq!(kinds("7.")[0], Tok::Float(7.0));
    }

    #[test]
    fn tracks_line_and_column() {
        let toks = Lexer::new("a\n  b").tokenize().unwrap();
        assert_eq!((toks[0].line, toks[0].col), (1, 1)); // a
        // toks[1] is the inserted terminator after a
        assert_eq!(toks[1].tok, Tok::Semicolon);
        // b is on line 2, column 3 (after two spaces)
        let b = toks
            .iter()
            .find(|t| t.tok == Tok::Ident("b".into()))
            .unwrap();
        assert_eq!((b.line, b.col), (2, 3));
    }

    #[test]
    fn overflowing_int_literal_is_an_error() {
        let err = Lexer::new("99999999999999999999999999")
            .tokenize()
            .unwrap_err();
        assert!(err.contains("bad int literal"), "got: {err}");
    }

    #[test]
    fn unexpected_char_is_an_error() {
        let err = Lexer::new("let x = #").tokenize().unwrap_err();
        assert!(err.contains("unexpected char"), "got: {err}");
        assert!(err.contains('#'), "got: {err}");
    }

    #[test]
    fn no_terminator_after_a_binary_operator() {
        // a newline right after an operator continues the statement
        let k = kinds("x +\n y");
        // [Ident(x), Plus, Ident(y), Semicolon(trailing), Eof]
        assert_eq!(k[1], Tok::Plus);
        assert_eq!(k[2], Tok::Ident("y".into())); // no terminator inserted after `+`
        assert_eq!(k.iter().filter(|t| **t == Tok::Semicolon).count(), 1);
    }
}
