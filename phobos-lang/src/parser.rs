use super::ast::*;
use super::token::{Tok, Token};
use crate::diag::{self, Diagnostic};

mod collective;
mod functions;
mod jumps;
mod recover;

pub(crate) type PResult<T> = Result<T, Diagnostic>;

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
    /// Where the last token consumed other than a statement terminator ends.
    last_end: (u32, u32),
    /// The same, as a byte offset.
    last_hi: u32,
    /// What went wrong in the statements and items recovered from.
    errors: Vec<Diagnostic>,
}

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Parser {
            toks,
            pos: 0,
            last_end: (1, 1),
            last_hi: 0,
            errors: Vec::new(),
        }
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
        let cur = &self.toks[self.pos];
        let t = cur.tok.clone();
        if t != Tok::Semicolon {
            self.last_end = (cur.line, cur.end_col);
            self.last_hi = cur.hi;
        }
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

    fn expect(&mut self, t: Tok) -> PResult<()> {
        if self.consume(&t) {
            Ok(())
        } else {
            Err(self.unexpected(&t.to_string()))
        }
    }

    /// The (line, column) of the current token.
    fn pos(&self) -> (u32, u32) {
        let cur = self.current();
        (cur.line, cur.col)
    }

    /// From `start` to the end of the last token consumed.
    fn span_from(&self, (line, col): (u32, u32)) -> Span {
        let (end_line, end_col) = self.last_end;
        Span {
            line,
            col,
            end_line,
            end_col,
        }
    }

    fn ident(&mut self) -> PResult<String> {
        match self.peek().clone() {
            Tok::Ident(s) if s == ctx::NAME => Err(Diagnostic::error("`ctx` is reserved")
                .primary(self.here(), "a name of its own here")
                .note("`ctx` is the launch context, read through its fields, as `ctx.block.x`")),
            Tok::Ident(s) => {
                self.advance();
                Ok(s)
            }
            _ => Err(self.unexpected("an identifier")),
        }
    }

    /// A field name after `.`.
    fn field(&mut self) -> PResult<String> {
        match self.peek().clone() {
            Tok::Ident(s) => {
                self.advance();
                Ok(s)
            }
            _ => Err(self.unexpected("a field name")),
        }
    }

    fn int(&mut self) -> PResult<i64> {
        match *self.peek() {
            Tok::Int(n) => {
                self.advance();
                Ok(n)
            }
            _ => Err(self.unexpected("an integer")),
        }
    }

    fn comma_separated<T>(
        &mut self,
        mut item: impl FnMut(&mut Self) -> PResult<T>,
    ) -> PResult<Vec<T>> {
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
    fn end_stmt(&mut self) -> PResult<()> {
        if self.consume(&Tok::Semicolon) || self.matches(&Tok::RBrace) || self.matches(&Tok::Eof) {
            Ok(())
        } else {
            Err(Diagnostic::error(format!("expected end of statement, found {}", self.found()))
                .primary(self.here(), "expected a newline or `;` before this"))
        }
    }

    /// Every kernel in the source, or every error found in it: a statement
    /// or an item that fails to parse is reported and skipped, and parsing
    /// goes on after it.
    pub fn parse_program(&mut self) -> Result<Vec<Kernel>, Vec<Diagnostic>> {
        let mut kernels = Vec::new();
        let mut fns = Vec::new();
        while !self.matches(&Tok::Eof) && self.errors.len() < diag::MAX_ERRORS {
            // tolerate terminators between kernels (e.g. after a body's '}')
            if self.consume(&Tok::Semicolon) {
                continue;
            }
            let start = self.pos;
            let item = if self.matches(&Tok::Fn) {
                self.parse_function().map(|f| fns.push(f))
            } else {
                self.parse_kernel().map(|k| kernels.push(k))
            };
            if let Err(d) = item {
                self.report(d);
                self.sync_item(start);
            }
        }
        if !self.errors.is_empty() {
            return Err(std::mem::take(&mut self.errors));
        }
        // Functions are inlined first, so a jump they splice in is lowered
        // with the caller's loops.
        functions::inline(&mut kernels, &fns).map_err(|e| vec![Diagnostic::error(e)])?;
        for kernel in &mut kernels {
            let body = std::mem::take(&mut kernel.body);
            kernel.body = jumps::desugar(body)
                .map_err(|e| vec![Diagnostic::error(format!("kernel `{}`: {e}", kernel.name))])?;
        }
        Ok(kernels)
    }

    /// `fn name(params) = expr` or `fn name(params) { stmts }`.
    fn parse_function(&mut self) -> PResult<functions::Function> {
        self.expect(Tok::Fn)?;
        let name = self.ident()?;
        let open = self.here();
        self.expect(Tok::LParen)?;
        let params = if self.matches(&Tok::RParen) {
            Vec::new()
        } else {
            self.comma_separated(Self::ident)?
        };
        self.expect_close(Tok::RParen, open)?;
        let body = if self.consume(&Tok::Eq) {
            let e = self.parse_expr()?;
            self.end_stmt()?;
            functions::Body::Expr(e)
        } else {
            let body = self.parse_block()?;
            if let Some(jump) = jumps::escaping(&body) {
                return Err(Diagnostic::error(format!(
                    "function `{name}`: `{jump}` would leave the function body"
                )));
            }
            functions::Body::Stmts(body)
        };
        Ok(functions::Function { name, params, body })
    }

    fn parse_kernel(&mut self) -> PResult<Kernel> {
        let mut attrs = Vec::new();
        loop {
            if self.matches(&Tok::At) {
                attrs.push(self.parse_attribute()?);
            } else if !self.consume(&Tok::Semicolon) {
                // ignore semis after attributes
                break;
            }
        }

        if !self.matches(&Tok::Kernel) {
            let what = if attrs.is_empty() { "`kernel` or `fn`" } else { "`kernel` after the attributes" };
            return Err(self.unexpected(what));
        }
        self.advance();
        let name = self.ident()?;

        let open = self.here();
        self.expect(Tok::LParen)?;
        let params = if self.matches(&Tok::RParen) {
            Vec::new()
        } else {
            self.parse_params()?
        };
        self.expect_close(Tok::RParen, open)?;

        let body = self.parse_block()?;
        Ok(Kernel {
            attrs,
            name,
            params,
            body,
        })
    }

    fn parse_attribute(&mut self) -> PResult<Attribute> {
        self.expect(Tok::At)?;

        let name = self.ident()?;

        let mut args = Vec::new();
        let open = self.here();
        if self.consume(&Tok::LParen) {
            if !self.matches(&Tok::RParen) {
                args = self.comma_separated(Self::parse_attr_arg)?;
            }
            self.expect_close(Tok::RParen, open)?;
        }

        Ok(Attribute { name, args })
    }

    fn parse_attr_arg(&mut self) -> PResult<AttrArg> {
        // an ident may begin a search dim (x in [..]), a keyword arg (x = v),
        // or a bare positional name; anything else is a positional literal
        if let Tok::Ident(_) = self.peek() {
            let name = self.ident()?;
            if self.consume(&Tok::In) {
                let open = self.here();
                self.expect(Tok::LBracket)?;
                let choices = self.comma_separated(Self::int)?;
                self.expect_close(Tok::RBracket, open)?;
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

    fn parse_lit(&mut self) -> PResult<Literal> {
        let lit = match self.peek().clone() {
            Tok::Int(n) => Literal::Int(n),
            Tok::Float(f) => Literal::Float(f),
            Tok::True => Literal::Bool(true),
            Tok::False => Literal::Bool(false),
            Tok::Ident(s) => Literal::Ident(s),
            _ => return Err(self.unexpected("a literal")),
        };
        self.advance();
        Ok(lit)
    }

    fn parse_params(&mut self) -> PResult<Vec<Param>> {
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
    fn parse_type(&mut self) -> PResult<Type> {
        let at = self.here();
        let Tok::Ident(name) = self.peek().clone() else {
            return Err(self.unexpected("a type"));
        };
        self.advance();

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
            other => {
                let types = Scalar::NAMES.into_iter().chain(["tensor", "tile"]);
                Err(recover::did_you_mean(
                    Diagnostic::error(format!("unknown type `{other}`")).primary(at, "not a type"),
                    at,
                    other,
                    types,
                ))
            }
        }
    }

    // '<' scalar '>' '[' dims ']'
    fn parse_aggregate(&mut self) -> PResult<(Scalar, Vec<Dim>)> {
        self.expect(Tok::Lt)?;
        let elem = self.parse_scalar()?;
        self.expect(Tok::Gt)?;
        let open = self.here();
        self.expect(Tok::LBracket)?;
        let dims = self.comma_separated(Self::parse_dim)?;
        self.expect_close(Tok::RBracket, open)?;
        Ok((elem, dims))
    }

    fn parse_scalar(&mut self) -> PResult<Scalar> {
        let at = self.here();
        let Tok::Ident(s) = self.peek().clone() else {
            return Err(self.unexpected("a scalar element type"));
        };
        let Some(scalar) = Scalar::from_name(&s) else {
            let d = Diagnostic::error(format!("expected a scalar element type, found `{s}`"))
                .primary(at, "not a scalar type");
            return Err(recover::did_you_mean(d, at, &s, Scalar::NAMES));
        };
        self.advance();
        Ok(scalar)
    }

    fn parse_dim(&mut self) -> PResult<Dim> {
        let dim = match self.peek().clone() {
            Tok::Ident(s) => Dim::Sym(s),
            Tok::Int(n) => Dim::Int(n),
            _ => return Err(self.unexpected("a dimension, a name or an integer")),
        };
        self.advance();
        Ok(dim)
    }
}

// statements

impl Parser {
    /// `{ stmts }`. A statement that fails to parse is reported and skipped.
    fn parse_block(&mut self) -> PResult<Vec<Stmt>> {
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

    fn parse_stmt(&mut self) -> PResult<Stmt> {
        let start = self.pos();
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
                let span = self.span_from(start);
                self.end_stmt()?;
                Ok(jumps::jump(which, span))
            }
            _ => {
                let lo = self.current().lo;
                let expr = self.parse_expr()?;
                let target = diag::Span::new(lo, self.last_hi);
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
                        ensure_lvalue(&expr).map_err(|m| Diagnostic::error(m).primary(target, "cannot be assigned"))?;
                        let mut value = self.parse_expr()?;
                        if let Some(bop) = compound {
                            value = Expr::Binary {
                                op: bop,
                                lhs: Box::new(expr.clone()),
                                rhs: Box::new(value),
                            };
                        }
                        let span = self.span_from(start);
                        self.end_stmt()?;
                        Ok(Stmt::Assign {
                            target: expr,
                            op,
                            value,
                            span,
                        })
                    }
                    None => {
                        let span = self.span_from(start);
                        self.end_stmt()?;
                        Ok(Stmt::Expr(expr, span))
                    }
                }
            }
        }
    }

    fn parse_let_or_var(&mut self, is_var: bool) -> PResult<Stmt> {
        let start = self.pos();
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
            let span = self.span_from(start);
            self.end_stmt()?;
            return Ok(Stmt::Var {
                name,
                ty,
                value: None,
                span,
            });
        }
        self.expect(Tok::Eq)?;
        let value = self.parse_expr()?;
        let span = self.span_from(start);
        self.end_stmt()?;
        Ok(if is_var {
            Stmt::Var {
                name,
                ty,
                value: Some(value),
                span,
            }
        } else {
            Stmt::Let {
                name,
                ty,
                value,
                span,
            }
        })
    }

    fn parse_if(&mut self) -> PResult<Stmt> {
        let start = self.pos();
        self.advance();
        let cond = self.parse_expr()?;
        let span = self.span_from(start);
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
        Ok(Stmt::If {
            cond,
            then,
            r#else,
            span,
        })
    }

    fn parse_while(&mut self) -> PResult<Stmt> {
        let start = self.pos();
        self.advance();
        let cond = self.parse_expr()?;
        let span = self.span_from(start);
        let body = self.parse_block()?;
        Ok(Stmt::While { cond, body, span })
    }

    fn parse_for(&mut self) -> PResult<Stmt> {
        let start_pos = self.pos();
        self.advance();
        // `for<unroll n>`: the one setting a loop takes.
        let unroll = if self.consume(&Tok::Lt) {
            let at = self.here();
            let word = self.field()?;
            if word != "unroll" {
                return Err(Diagnostic::error(format!("`{word}` is not a setting of `for<..>`; it takes `unroll`"))
                    .primary(at, "unknown setting")
                    .help("unroll a loop as `for<unroll 4> i in range(0, n)`"));
            }
            let n = self.term()?;
            self.expect(Tok::Gt)?;
            Some(n)
        } else {
            None
        };
        let var = self.ident()?;
        self.expect(Tok::In)?;
        let at = self.here();
        let r = self.ident()?;
        if r != "range" {
            return Err(Diagnostic::error(format!("expected `range(...)` in for-loop, found `{r}`"))
                .primary(at, "a loop runs over a `range`")
                .help("write the bounds as `range(start, end)` or `range(start, end, step)`"));
        }
        let open = self.here();
        self.expect(Tok::LParen)?;
        let start = self.parse_expr()?;
        self.expect(Tok::Comma)?;
        let end = self.parse_expr()?;
        let step = if self.consume(&Tok::Comma) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect_close(Tok::RParen, open)?;
        let span = self.span_from(start_pos);
        let body = self.parse_block()?;
        Ok(Stmt::For {
            var,
            start,
            end,
            step,
            unroll,
            body,
            span,
        })
    }
}

// expressions

impl Parser {
    fn parse_expr(&mut self) -> PResult<Expr> {
        self.logical_or()
    }

    /// One left-associative level of binary operators: `ops` maps a token to
    /// its operator, `next` parses the tighter level.
    fn binary_level(
        &mut self,
        ops: &[(Tok, BinOp)],
        next: fn(&mut Self) -> PResult<Expr>,
    ) -> PResult<Expr> {
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

    fn logical_or(&mut self) -> PResult<Expr> {
        self.binary_level(&[(Tok::PipePipe, BinOp::Or)], Self::logical_and)
    }

    fn logical_and(&mut self) -> PResult<Expr> {
        self.binary_level(&[(Tok::AmpAmp, BinOp::And)], Self::equality)
    }

    fn equality(&mut self) -> PResult<Expr> {
        self.binary_level(&[(Tok::EqEq, BinOp::Eq), (Tok::NotEq, BinOp::Ne)], Self::comparison)
    }

    fn comparison(&mut self) -> PResult<Expr> {
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
    fn bit_or(&mut self) -> PResult<Expr> {
        self.binary_level(&[(Tok::Pipe, BinOp::BitOr)], Self::bit_xor)
    }

    fn bit_xor(&mut self) -> PResult<Expr> {
        self.binary_level(&[(Tok::Caret, BinOp::BitXor)], Self::bit_and)
    }

    fn bit_and(&mut self) -> PResult<Expr> {
        self.binary_level(&[(Tok::Amp, BinOp::BitAnd)], Self::shift)
    }

    fn shift(&mut self) -> PResult<Expr> {
        self.binary_level(&[(Tok::Shl, BinOp::Shl), (Tok::Shr, BinOp::Shr)], Self::term)
    }

    fn term(&mut self) -> PResult<Expr> {
        self.binary_level(&[(Tok::Plus, BinOp::Add), (Tok::Minus, BinOp::Sub)], Self::factor)
    }

    fn factor(&mut self) -> PResult<Expr> {
        let ops = [
            (Tok::Star, BinOp::Mul),
            (Tok::Slash, BinOp::Div),
            (Tok::Percent, BinOp::Rem),
        ];
        self.binary_level(&ops, Self::unary)
    }

    fn unary(&mut self) -> PResult<Expr> {
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

    fn postfix(&mut self) -> PResult<Expr> {
        let lo = self.current().lo;
        let mut expr = self.primary()?;
        loop {
            let open = self.here();
            if self.consume(&Tok::LBracket) {
                let subs = self.comma_separated(Self::parse_sub)?;
                self.expect_close(Tok::RBracket, open)?;
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
                    self.expect_close(Tok::RParen, open)?;
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
                    return Err(Diagnostic::error("call target must be an identifier")
                        .primary(diag::Span::new(lo, self.last_hi), "this is not a name")
                        .secondary(open, "called here"));
                }
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_sub(&mut self) -> PResult<Sub> {
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

    fn parse_args(&mut self) -> PResult<Vec<Expr>> {
        self.comma_separated(Self::parse_expr)
    }

    fn primary(&mut self) -> PResult<Expr> {
        let at = self.here();
        if !matches!(
            self.peek(),
            Tok::Int(_) | Tok::Float(_) | Tok::True | Tok::False | Tok::Ident(_) | Tok::LParen
        ) {
            return Err(Diagnostic::error(format!("expected an expression, found {}", self.found()))
                .primary(self.here(), "expected an expression"));
        }
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
                    return Err(Diagnostic::error("`ctx` is used through a field")
                        .primary(at, "")
                        .help("read a field, as `ctx.block.x` or `ctx.lane`"));
                }
                let whole = diag::Span::new(at.lo, self.last_hi);
                ctx::field(&path).map_err(|e| {
                    let d = Diagnostic::error(e).primary(whole, "");
                    match path[..] {
                        [ref field] => {
                            let field_at = diag::Span::new(whole.hi - field.len() as u32, whole.hi);
                            recover::did_you_mean(d, field_at, field, ctx::FIELDS)
                        }
                        _ => d,
                    }
                })
            }
            Tok::Ident(s) => Ok(Expr::Var(s)),
            Tok::LParen => {
                let expr = self.parse_expr()?;
                self.expect_close(Tok::RParen, at)?;
                Ok(expr)
            }
            _ => unreachable!("checked above"),
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
