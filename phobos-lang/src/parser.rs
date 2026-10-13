use super::ast::*;
use super::token::{Tok, Token};

mod collective;
mod functions;
mod jumps;

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
            Tok::Ident(s) if s == ctx::NAME => Err(format!("{line}:{col}: `ctx` is reserved")),
            Tok::Ident(s) => Ok(s),
            other => Err(format!("{line}:{col}: expected identifier, found {other}")),
        }
    }

    /// A field name after `.`.
    fn field(&mut self) -> Result<String, String> {
        let (line, col) = self.pos();
        match self.advance() {
            Tok::Ident(s) => Ok(s),
            other => Err(format!("{line}:{col}: expected a field name, found {other}")),
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
    /// Ends a statement.
    ///
    /// A statement ends at a terminator, either newline-inserted or an
    /// explicit ';'. The terminator may be omitted before a closing } or at
    /// EOF.
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
        let mut kernels = Vec::new();
        let mut fns = Vec::new();
        while !self.matches(&Tok::Eof) {
            // tolerate terminators between kernels (e.g. after a body's '}')
            if self.consume(&Tok::Semicolon) {
                continue;
            }
            if self.matches(&Tok::Fn) {
                fns.push(self.parse_function()?);
            } else {
                kernels.push(self.parse_kernel()?);
            }
        }
        // Functions are inlined first, so a jump they splice in is lowered
        // with the caller's loops.
        functions::inline(&mut kernels, &fns)?;
        for kernel in &mut kernels {
            let body = std::mem::take(&mut kernel.body);
            kernel.body = jumps::desugar(body).map_err(|e| format!("kernel `{}`: {e}", kernel.name))?;
        }
        Ok(kernels)
    }

    /// `fn name(params) = expr` or `fn name(params) { stmts }`.
    fn parse_function(&mut self) -> Result<functions::Function, String> {
        self.expect(Tok::Fn)?;
        let name = self.ident()?;
        self.expect(Tok::LParen)?;
        let params = if self.matches(&Tok::RParen) {
            Vec::new()
        } else {
            self.comma_separated(Self::ident)?
        };
        self.expect(Tok::RParen)?;
        let body = if self.consume(&Tok::Eq) {
            let e = self.parse_expr()?;
            self.end_stmt()?;
            functions::Body::Expr(e)
        } else {
            let body = self.parse_block()?;
            if let Some(jump) = jumps::escaping(&body) {
                return Err(format!("function `{name}`: `{jump}` would leave the function body"));
            }
            functions::Body::Stmts(body)
        };
        Ok(functions::Function { name, params, body })
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
            Tok::Break | Tok::Continue | Tok::Return => {
                let which = match self.advance() {
                    Tok::Break => jumps::BREAK,
                    Tok::Continue => jumps::CONTINUE,
                    _ => jumps::RETURN,
                };
                self.end_stmt()?;
                Ok(jumps::jump(which))
            }
            _ => {
                let (line, col) = self.pos();
                let expr = self.parse_expr()?;
                // `x op= e` other than `+=` is `x = x op e`; `+=` stays an
                // accumulate, which a dot or a fragment lowers on its own.
                let (op, compound) = match self.peek().clone() {
                    Tok::Eq => (Some(AssignOp::Set), None),
                    Tok::PlusEq => (Some(AssignOp::Add), None),
                    Tok::OpEq(bop) => (Some(AssignOp::Set), Some(bop)),
                    _ => (None, None),
                };
                match op {
                    Some(op) => {
                        self.advance();
                        ensure_lvalue(&expr).map_err(|m| format!("{line}:{col}: {m}"))?;
                        let mut value = self.parse_expr()?;
                        if let Some(bop) = compound {
                            value = Expr::Binary {
                                op: bop,
                                lhs: Box::new(expr.clone()),
                                rhs: Box::new(value),
                            };
                        }
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
        // `for<unroll n>`: the one setting a loop takes.
        let unroll = if self.consume(&Tok::Lt) {
            let (line, col) = self.pos();
            let word = self.field()?;
            if word != "unroll" {
                return Err(format!("{line}:{col}: `{word}` is not a setting of `for<..>`; it takes `unroll`"));
            }
            let n = self.term()?;
            self.expect(Tok::Gt)?;
            Some(n)
        } else {
            None
        };
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
            unroll,
            body,
        })
    }
}

// expressions

impl Parser {
    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.logical_or()
    }

    /// One left-associative level of binary operators: `ops` maps a token to
    /// its operator, `next` parses the tighter level.
    fn binary_level(
        &mut self,
        ops: &[(Tok, BinOp)],
        next: fn(&mut Self) -> Result<Expr, String>,
    ) -> Result<Expr, String> {
        let mut lhs = next(self)?;
        while let Some(&(_, op)) = ops.iter().find(|(t, _)| self.matches(t)) {
            self.advance();
            let rhs = next(self)?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn logical_or(&mut self) -> Result<Expr, String> {
        self.binary_level(&[(Tok::PipePipe, BinOp::Or)], Self::logical_and)
    }

    fn logical_and(&mut self) -> Result<Expr, String> {
        self.binary_level(&[(Tok::AmpAmp, BinOp::And)], Self::equality)
    }

    fn equality(&mut self) -> Result<Expr, String> {
        self.binary_level(&[(Tok::EqEq, BinOp::Eq), (Tok::NotEq, BinOp::Ne)], Self::comparison)
    }

    fn comparison(&mut self) -> Result<Expr, String> {
        let ops = [
            (Tok::Lt, BinOp::Lt),
            (Tok::Le, BinOp::Le),
            (Tok::Gt, BinOp::Gt),
            (Tok::Ge, BinOp::Ge),
        ];
        self.binary_level(&ops, Self::bit_or)
    }

    // Bitwise operators bind tighter than comparisons, as in Rust, so
    // `x & 15 == 3` tests the masked value.
    fn bit_or(&mut self) -> Result<Expr, String> {
        self.binary_level(&[(Tok::Pipe, BinOp::BitOr)], Self::bit_xor)
    }

    fn bit_xor(&mut self) -> Result<Expr, String> {
        self.binary_level(&[(Tok::Caret, BinOp::BitXor)], Self::bit_and)
    }

    fn bit_and(&mut self) -> Result<Expr, String> {
        self.binary_level(&[(Tok::Amp, BinOp::BitAnd)], Self::shift)
    }

    fn shift(&mut self) -> Result<Expr, String> {
        self.binary_level(&[(Tok::Shl, BinOp::Shl), (Tok::Shr, BinOp::Shr)], Self::term)
    }

    fn term(&mut self) -> Result<Expr, String> {
        self.binary_level(&[(Tok::Plus, BinOp::Add), (Tok::Minus, BinOp::Sub)], Self::factor)
    }

    fn factor(&mut self) -> Result<Expr, String> {
        let ops = [
            (Tok::Star, BinOp::Mul),
            (Tok::Slash, BinOp::Div),
            (Tok::Percent, BinOp::Rem),
        ];
        self.binary_level(&ops, Self::unary)
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
            Tok::Tilde => {
                self.advance();
                Ok(Expr::Unary {
                    op: UnOp::BitNot,
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
            } else if let Expr::Var(name) = &expr
                && self.at_config()
            {
                let name = name.clone();
                expr = self.config_call(&name)?;
            } else if self.matches(&Tok::LParen) {
                if let Expr::Var(name) = expr {
                    self.advance(); // `(`
                    let args = if self.matches(&Tok::RParen) {
                        Vec::new()
                    } else {
                        self.parse_args()?
                    };
                    self.expect(Tok::RParen)?;
                    expr = if matches!(name.as_str(), "min" | "max") && args.len() == 2 {
                        let [lhs, rhs] = <[Expr; 2]>::try_from(args).expect("two arguments");
                        Expr::Binary {
                            op: if name == "min" { BinOp::Min } else { BinOp::Max },
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        }
                    } else if name == "pow" && args.len() == 2 {
                        // a^b = e^(b ln a), so it fuses like any per-element tree.
                        let [base, exponent] = <[Expr; 2]>::try_from(args).expect("two arguments");
                        let ln = Expr::Call {
                            callee: "log".into(),
                            args: vec![base],
                        };
                        Expr::Call {
                            callee: "exp".into(),
                            args: vec![Expr::Binary {
                                op: BinOp::Mul,
                                lhs: Box::new(exponent),
                                rhs: Box::new(ln),
                            }],
                        }
                    } else {
                        Expr::Call { callee: name, args }
                    };
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
            if self.matches(&Tok::RBracket) || self.matches(&Tok::Comma) {
                return Ok(Sub::Full); // A[:]
            }
            let end = self.parse_expr()?;
            return Ok(Sub::Range {
                start: Expr::Int(0),
                end,
            }); // A[: end]
        }
        let start = self.parse_expr()?;
        if self.consume(&Tok::Colon) {
            if self.matches(&Tok::RBracket) || self.matches(&Tok::Comma) {
                return Ok(Sub::From(start)); // A[start :]
            }
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
            Tok::Ident(s) if s == ctx::NAME => {
                let mut path = Vec::new();
                while self.consume(&Tok::Dot) {
                    path.push(self.field()?);
                }
                if path.is_empty() {
                    return Err(format!("{line}:{col}: `ctx` is used through a field, as `ctx.block.x`"));
                }
                ctx::field(&path).map_err(|e| format!("{line}:{col}: {e}"))
            }
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
