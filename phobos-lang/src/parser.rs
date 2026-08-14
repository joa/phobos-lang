use super::ast::*;
use super::token::{Tok, Token};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Parser { toks, pos: 0 }
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn current(&self) -> &Token {
        &self.toks[self.pos]
    }

    fn matches(&self, t: &Tok) -> bool {
        self.peek() == t
    }

    fn advance(&mut self) -> Tok {
        let t = self.toks[self.pos].tok.clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn consume(&mut self, t: &Tok) -> bool {
        if self.matches(t) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: Tok) -> Result<(), String> {
        if self.matches(&t) {
            self.advance();
            Ok(())
        } else {
            Err(self.err_msg(format!("expected {}, found {}", t, self.peek())))
        }
    }

    /// The (line, column) of the current token.
    fn pos(&self) -> (u32, u32) {
        let cur = self.current();
        (cur.line, cur.col)
    }

    fn err_msg(&self, msg: impl std::fmt::Display) -> String {
        let (line, col) = self.pos();
        format!("{line}:{col}: {msg}")
    }

    fn ident(&mut self) -> Result<String, String> {
        let (line, col) = self.pos();
        match self.advance() {
            Tok::Ident(s) => Ok(s),
            other => Err(format!("{line}:{col}: expected identifier, found {other}")),
        }
    }

    fn int(&mut self) -> Result<i64, String> {
        let (line, col) = self.pos();
        match self.advance() {
            Tok::Int(n) => Ok(n),
            other => Err(format!("{line}:{col}: expected integer, found {other}")),
        }
    }

    fn comma_separated<T>(
        &mut self,
        mut item: impl FnMut(&mut Self) -> Result<T, String>,
    ) -> Result<Vec<T>, String> {
        let mut out = vec![item(self)?];
        while self.consume(&Tok::Comma) {
            out.push(item(self)?);
        }
        Ok(out)
    }
}

// top level

impl Parser {
    /// End a statement.
    ///
    /// Statements end at a terminator, either newline-inserted or an explicit
    /// ';', which may also be omitted before a closing } or at EOF.
    fn end_stmt(&mut self) -> Result<(), String> {
        if self.consume(&Tok::Semicolon) || self.matches(&Tok::RBrace) || self.matches(&Tok::Eof) {
            Ok(())
        } else {
            Err(self.err_msg(format!(
                "expected end of statement (newline), found {}",
                self.peek()
            )))
        }
    }

    pub fn parse_program(&mut self) -> Result<Vec<Kernel>, String> {
        let mut out = Vec::new();
        while !self.matches(&Tok::Eof) {
            // tolerate terminators between kernels (e.g. after a body's '}')
            if self.consume(&Tok::Semicolon) {
                continue;
            }
            out.push(self.parse_kernel()?);
        }
        Ok(out)
    }

    fn parse_kernel(&mut self) -> Result<Kernel, String> {
        let mut attrs = Vec::new();
        loop {
            if self.matches(&Tok::At) {
                attrs.push(self.parse_attribute()?);
            } else if !self.consume(&Tok::Semicolon) {
                // ignore semis after attributes
                break;
            }
        }

        self.expect(Tok::Kernel)?;
        let name = self.ident()?;

        self.expect(Tok::LParen)?;
        let params = if self.matches(&Tok::RParen) {
            Vec::new()
        } else {
            self.parse_params()?
        };
        self.expect(Tok::RParen)?;

        let body = self.parse_block()?;
        Ok(Kernel {
            attrs,
            name,
            params,
            body,
        })
    }

    fn parse_attribute(&mut self) -> Result<Attribute, String> {
        self.expect(Tok::At)?;

        let name = self.ident()?;

        let mut args = Vec::new();
        if self.consume(&Tok::LParen) {
            if !self.matches(&Tok::RParen) {
                args = self.comma_separated(Self::parse_attr_arg)?;
            }
            self.expect(Tok::RParen)?;
        }

        Ok(Attribute { name, args })
    }

    fn parse_attr_arg(&mut self) -> Result<AttrArg, String> {
        // an ident may begin a search dim (x in [..]), a keyword arg (x = v),
        // or a bare positional name; anything else is a positional literal
        if let Tok::Ident(_) = self.peek() {
            let name = self.ident()?;
            if self.consume(&Tok::In) {
                self.expect(Tok::LBracket)?;
                let choices = self.comma_separated(Self::int)?;
                self.expect(Tok::RBracket)?;
                Ok(AttrArg::Search { name, choices })
            } else if self.consume(&Tok::Eq) {
                Ok(AttrArg::KeyValue {
                    key: name,
                    value: self.parse_lit()?,
                })
            } else {
                Ok(AttrArg::Positional(Literal::Ident(name)))
            }
        } else {
            Ok(AttrArg::Positional(self.parse_lit()?))
        }
    }

    fn parse_lit(&mut self) -> Result<Literal, String> {
        let (line, col) = self.pos();
        match self.advance() {
            Tok::Int(n) => Ok(Literal::Int(n)),
            Tok::Float(f) => Ok(Literal::Float(f)),
            Tok::True => Ok(Literal::Bool(true)),
            Tok::False => Ok(Literal::Bool(false)),
            Tok::Ident(s) => Ok(Literal::Ident(s)),
            other => Err(format!("{line}:{col}: expected a literal, found {other}")),
        }
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, String> {
        self.comma_separated(|p| {
            let name = p.ident()?;
            p.expect(Tok::Colon)?;
            let ty = p.parse_type()?;
            Ok(Param { name, ty })
        })
    }
}

// types

impl Parser {
    fn parse_type(&mut self) -> Result<Type, String> {
        let (line, col) = self.pos();
        let name = match self.advance() {
            Tok::Ident(s) => s,
            other => return Err(format!("{line}:{col}: expected a type, found {other}")),
        };

        if let Some(s) = Scalar::from_name(&name) {
            return Ok(Type::Scalar(s));
        }

        match name.as_str() {
            "tensor" => {
                let (elem, dims) = self.parse_aggregate()?;
                Ok(Type::Tensor(elem, dims))
            }
            "tile" => {
                let (elem, dims) = self.parse_aggregate()?;
                Ok(Type::Tile(elem, dims))
            }
            other => Err(format!("{line}:{col}: unknown type '{other}'")),
        }
    }

    // '<' scalar '>' '[' dims ']'
    fn parse_aggregate(&mut self) -> Result<(Scalar, Vec<Dim>), String> {
        self.expect(Tok::Lt)?;
        let elem = self.parse_scalar()?;
        self.expect(Tok::Gt)?;
        self.expect(Tok::LBracket)?;
        let dims = self.comma_separated(Self::parse_dim)?;
        self.expect(Tok::RBracket)?;
        Ok((elem, dims))
    }

    fn parse_scalar(&mut self) -> Result<Scalar, String> {
        let (line, col) = self.pos();
        match self.advance() {
            Tok::Ident(s) => Scalar::from_name(&s).ok_or_else(|| {
                format!("{line}:{col}: expected a scalar element type, found '{s}'")
            }),
            other => Err(format!(
                "{line}:{col}: expected a scalar element type, found {other}"
            )),
        }
    }

    fn parse_dim(&mut self) -> Result<Dim, String> {
        let (line, col) = self.pos();
        match self.advance() {
            Tok::Ident(s) => Ok(Dim::Sym(s)),
            Tok::Int(n) => Ok(Dim::Int(n)),
            other => Err(format!(
                "{line}:{col}: expected a dimension (name or int), found {other}"
            )),
        }
    }
}

// statements

impl Parser {
    fn parse_block(&mut self) -> Result<Vec<Stmt>, String> {
        self.expect(Tok::LBrace)?;
        let mut stmts = Vec::new();
        while !self.matches(&Tok::RBrace) && !self.matches(&Tok::Eof) {
            if self.consume(&Tok::Semicolon) {
                continue;
            }
            stmts.push(self.parse_stmt()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        match self.peek() {
            Tok::Let => self.parse_let_or_var(false),
            Tok::Var => self.parse_let_or_var(true),
            Tok::If => self.parse_if(),
            Tok::For => self.parse_for(),
            Tok::While => self.parse_while(),
            _ => {
                let (line, col) = self.pos();
                let expr = self.parse_expr()?;
                let op = if self.consume(&Tok::Eq) {
                    Some(AssignOp::Set)
                } else if self.consume(&Tok::PlusEq) {
                    Some(AssignOp::Add)
                } else {
                    None
                };
                match op {
                    Some(op) => {
                        ensure_lvalue(&expr).map_err(|m| format!("{line}:{col}: {m}"))?;
                        let value = self.parse_expr()?;
                        self.end_stmt()?;
                        Ok(Stmt::Assign {
                            target: expr,
                            op,
                            value,
                        })
                    }
                    None => {
                        self.end_stmt()?;
                        Ok(Stmt::Expr(expr))
                    }
                }
            }
        }
    }

    fn parse_let_or_var(&mut self, is_var: bool) -> Result<Stmt, String> {
        self.advance();
        let name = self.ident()?;
        let ty = if self.consume(&Tok::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        // A `var` with a type may omit its initializer, which declares a buffer
        // without filling it. A `let` cannot: it would name nothing.
        if is_var && ty.is_some() && !self.matches(&Tok::Eq) {
            // we allow tiles without an init if a type is known
            // - uninitialized (this) : var foo: tile<f32>[D, D]
            // - initialized   (below): var bar: tile<f32>[D, D] = 0.0
            // - illegal              : var baz
            // - illegal (immutable!) : let eek: tile<f32>[D, D]
            self.end_stmt()?;
            return Ok(Stmt::Var {
                name,
                ty,
                value: None,
            });
        }
        self.expect(Tok::Eq)?;
        let value = self.parse_expr()?;
        self.end_stmt()?;
        Ok(if is_var {
            Stmt::Var {
                name,
                ty,
                value: Some(value),
            }
        } else {
            Stmt::Let { name, ty, value }
        })
    }

    fn parse_if(&mut self) -> Result<Stmt, String> {
        self.advance();
        let cond = self.parse_expr()?;
        let then = self.parse_block()?;
        let r#else = if self.consume(&Tok::Else) {
            if self.matches(&Tok::If) {
                Some(vec![self.parse_if()?])
            } else {
                Some(self.parse_block()?)
            }
        } else {
            None
        };
        Ok(Stmt::If { cond, then, r#else })
    }

    fn parse_while(&mut self) -> Result<Stmt, String> {
        self.advance();
        let cond = self.parse_expr()?;
        let body = self.parse_block()?;
        Ok(Stmt::While { cond, body })
    }

    fn parse_for(&mut self) -> Result<Stmt, String> {
        self.advance();
        let var = self.ident()?;
        self.expect(Tok::In)?;
        let (line, col) = self.pos();
        let r = self.ident()?;
        if r != "range" {
            return Err(format!(
                "{line}:{col}: expected `range(...)` in for-loop, found `{r}`"
            ));
        }
        self.expect(Tok::LParen)?;
        let start = self.parse_expr()?;
        self.expect(Tok::Comma)?;
        let end = self.parse_expr()?;
        let step = if self.consume(&Tok::Comma) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect(Tok::RParen)?;
        let body = self.parse_block()?;
        Ok(Stmt::For {
            var,
            start,
            end,
            step,
            body,
        })
    }
}

// expressions

impl Parser {
    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.equality()
    }

    fn equality(&mut self) -> Result<Expr, String> {
        let mut lhs = self.comparison()?;
        loop {
            let op = match self.peek() {
                Tok::EqEq => BinOp::Eq,
                Tok::NotEq => BinOp::Ne,
                _ => break,
            };
            self.advance();
            let rhs = self.comparison()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn comparison(&mut self) -> Result<Expr, String> {
        let mut lhs = self.term()?;
        loop {
            let op = match self.peek() {
                Tok::Lt => BinOp::Lt,
                Tok::Le => BinOp::Le,
                Tok::Gt => BinOp::Gt,
                Tok::Ge => BinOp::Ge,
                _ => break,
            };
            self.advance();
            let rhs = self.term()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn term(&mut self) -> Result<Expr, String> {
        let mut lhs = self.factor()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.factor()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn factor(&mut self) -> Result<Expr, String> {
        let mut lhs = self.unary()?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::Percent => BinOp::Rem,
                _ => break,
            };
            self.advance();
            let rhs = self.unary()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Tok::Minus => {
                self.advance();
                Ok(Expr::Unary {
                    op: UnOp::Neg,
                    rhs: Box::new(self.unary()?),
                })
            }
            Tok::Bang => {
                self.advance();
                Ok(Expr::Unary {
                    op: UnOp::Not,
                    rhs: Box::new(self.unary()?),
                })
            }
            _ => self.postfix(),
        }
    }

    fn postfix(&mut self) -> Result<Expr, String> {
        let mut expr = self.primary()?;
        loop {
            if self.consume(&Tok::LBracket) {
                let subs = self.comma_separated(Self::parse_sub)?;
                self.expect(Tok::RBracket)?;
                expr = Expr::Index {
                    base: Box::new(expr),
                    subs,
                };
            } else if self.matches(&Tok::LParen) {
                if let Expr::Var(name) = expr {
                    self.advance(); // `(`
                    let args = if self.matches(&Tok::RParen) {
                        Vec::new()
                    } else {
                        self.parse_args()?
                    };
                    self.expect(Tok::RParen)?;
                    expr = Expr::Call { callee: name, args };
                } else {
                    return Err(self.err_msg("call target must be an identifier"));
                }
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_sub(&mut self) -> Result<Sub, String> {
        if self.consume(&Tok::Colon) {
            return Ok(Sub::Full); // A[:]
        }
        let start = self.parse_expr()?;
        if self.consume(&Tok::Colon) {
            let end = self.parse_expr()?;
            Ok(Sub::Range { start, end }) // A[start : end]
        } else if self.consume(&Tok::ColonPlus) {
            let len = self.parse_expr()?;
            Ok(Sub::Span { start, len }) // A[start :+ len]
        } else {
            Ok(Sub::Point(start)) // A[index]
        }
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>, String> {
        self.comma_separated(Self::parse_expr)
    }

    fn primary(&mut self) -> Result<Expr, String> {
        let (line, col) = self.pos();
        match self.advance() {
            Tok::Int(n) => Ok(Expr::Int(n)),
            Tok::Float(f) => Ok(Expr::Float(f)),
            Tok::True => Ok(Expr::Bool(true)),
            Tok::False => Ok(Expr::Bool(false)),
            Tok::Ident(s) => Ok(Expr::Var(s)),
            Tok::LParen => {
                let expr = self.parse_expr()?;
                self.expect(Tok::RParen)?;
                Ok(expr)
            }
            other => Err(format!(
                "{line}:{col}: unexpected token in expression: {other}"
            )),
        }
    }
}

fn ensure_lvalue(e: &Expr) -> Result<(), String> {
    match e {
        Expr::Var(_) => Ok(()),
        Expr::Index { base, .. } => match base.as_ref() {
            Expr::Var(_) => Ok(()),
            _ => Err("assignment target must be a name or an indexed name".to_string()),
        },
        _ => Err("invalid assignment target".to_string()),
    }
}

#[cfg(test)]
mod tests;
