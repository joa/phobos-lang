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
mod tests;
