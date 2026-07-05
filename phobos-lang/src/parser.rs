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
        self.expect(Tok::Eq)?;
        let value = self.parse_expr()?;
        self.end_stmt()?;
        Ok(if is_var {
            Stmt::Var { name, ty, value }
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
mod tests {
    use crate::ast::*;

    fn parse(src: &str) -> Vec<Kernel> {
        crate::parse(src).unwrap()
    }

    #[test]
    fn parses_tensor_kernel() {
        let p = parse(
            "kernel add(X: tensor<f32>[N], Y: tensor<f32>[N], Z: tensor<f32>[N]) {
                let i = program_id(0)
                Z[i] = X[i] + Y[i]
             }",
        );
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].name, "add");
        assert_eq!(p[0].params.len(), 3);
        assert!(matches!(p[0].params[0].ty, Type::Tensor(Scalar::F32, _)));
    }

    #[test]
    fn parses_attribute_slices_and_for() {
        let src = "@autotune(TILE_M in [64, 128], TILE_K in [16, 32])
            kernel mm(A: tensor<f32>[M, K], C: tensor<f32>[M, N]) {
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[0 : TILE_M, kt :+ TILE_K]
                    acc += a
                }
                C[0 : TILE_M, 0 : TILE_N] = acc
            }";
        let p = parse(src);
        assert_eq!(p[0].attrs.len(), 1);
        assert_eq!(p[0].attrs[0].name, "autotune");
        match &p[0].attrs[0].args[0] {
            AttrArg::Search { name, choices } => {
                assert_eq!(name, "TILE_M");
                assert_eq!(*choices, vec![64, 128]);
            }
            _ => panic!("expected an autotune search arg"),
        }
        // body: var, for, assign
        assert!(matches!(p[0].body[0], Stmt::Var { .. }));
        assert!(matches!(p[0].body[1], Stmt::For { .. }));
        assert!(matches!(
            p[0].body[2],
            Stmt::Assign {
                op: AssignOp::Set,
                ..
            }
        ));
    }

    #[test]
    fn parses_assign_add() {
        let p = parse("kernel k(A: tensor<f32>[N]) { var s = 0.0; s += A[0]; }");
        assert!(matches!(
            p[0].body[1],
            Stmt::Assign {
                op: AssignOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn parses_attributes() {
        let src = "@fast_math
            @launch_bounds(256, 2) @cache(read_only)
            @target(arch = sm_80)
            kernel k(A: tensor<f32>[N]) { let i = program_id(0) }";
        let p = parse(src);
        assert_eq!(p[0].attrs.len(), 4);
        assert_eq!(p[0].attrs[0].name, "fast_math");
        assert!(p[0].attrs[0].args.is_empty());
        assert!(matches!(
            p[0].attrs[1].args[0],
            AttrArg::Positional(Literal::Int(256))
        ));
        assert!(matches!(
            p[0].attrs[1].args[1],
            AttrArg::Positional(Literal::Int(2))
        ));
        assert!(
            matches!(&p[0].attrs[2].args[0], AttrArg::Positional(Literal::Ident(s)) if s.as_str() == "read_only")
        );
        assert!(
            matches!(&p[0].attrs[3].args[0], AttrArg::KeyValue { key, .. } if key.as_str() == "arch")
        );
    }

    #[test]
    fn full_range_subscript() {
        let p = parse("kernel k(A: tensor<f32>[M, N]) { let r = A[0, :]; }");
        if let Stmt::Let {
            value: Expr::Index { subs, .. },
            ..
        } = &p[0].body[0]
        {
            assert!(matches!(subs[0], Sub::Point(Expr::Int(0))));
            assert!(matches!(subs[1], Sub::Full));
        } else {
            panic!("expected indexed let");
        }
    }

    #[test]
    fn range_and_span_subscripts() {
        let p = parse("kernel k(A: tensor<f32>[M, N]) { let r = A[i : j, i :+ n]; }");
        if let Stmt::Let {
            value: Expr::Index { subs, .. },
            ..
        } = &p[0].body[0]
        {
            assert!(matches!(subs[0], Sub::Range { .. }));
            assert!(matches!(subs[1], Sub::Span { .. }));
        } else {
            panic!("expected indexed let");
        }
    }

    fn parse_err(src: &str) -> String {
        crate::parse(src).unwrap_err().to_string()
    }

    fn body(src: &str) -> Vec<Stmt> {
        parse(src).into_iter().next().unwrap().body
    }

    #[test]
    fn empty_params_and_empty_body() {
        let p = parse("kernel k() { }");
        assert_eq!(p[0].params.len(), 0);
        assert!(p[0].body.is_empty());
    }

    #[test]
    fn parses_multiple_kernels() {
        let p = parse(
            "kernel a(X: tensor<f32>[N]) { X[0] = 1.0 }
             kernel b(Y: tensor<f32>[N]) { Y[0] = 2.0 }",
        );
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].name, "a");
        assert_eq!(p[1].name, "b");
    }

    #[test]
    fn all_scalar_types_and_tile() {
        let p = parse(
            "kernel k(a: f32, b: f64, c: i32, d: i64, e: bool, t: tile<i32>[M, N],
                     g: f16, h: tensor<f16>[N]) {
                e = true
            }",
        );
        let tys: Vec<&Type> = p[0].params.iter().map(|p| &p.ty).collect();
        assert!(matches!(tys[0], Type::Scalar(Scalar::F32)));
        assert!(matches!(tys[1], Type::Scalar(Scalar::F64)));
        assert!(matches!(tys[2], Type::Scalar(Scalar::I32)));
        assert!(matches!(tys[3], Type::Scalar(Scalar::I64)));
        assert!(matches!(tys[4], Type::Scalar(Scalar::Bool)));
        assert!(matches!(tys[5], Type::Tile(Scalar::I32, _)));
        assert!(matches!(tys[6], Type::Scalar(Scalar::F16)));
        assert!(matches!(tys[7], Type::Tensor(Scalar::F16, _)));
    }

    #[test]
    fn while_loop() {
        let b = body("kernel k(A: tensor<f32>[N]) { var i = 0; while i < 4 { i = i + 1 } }");
        assert!(matches!(b[1], Stmt::While { .. }));
    }

    #[test]
    fn if_else_and_else_if_chain() {
        let b = body(
            "kernel k(a: i32) {
                if a < 0 { } else if a == 0 { } else { }
            }",
        );
        // outer if has an else holding a single nested if-statement
        if let Stmt::If {
            r#else: Some(e), ..
        } = &b[0]
        {
            assert_eq!(e.len(), 1);
            assert!(matches!(e[0], Stmt::If { .. }));
        } else {
            panic!("expected if/else-if");
        }
    }

    #[test]
    fn unary_neg_and_not() {
        let b = body("kernel k(x: i32) { let a = -x; let n = !true; }");
        assert!(matches!(
            &b[0],
            Stmt::Let {
                value: Expr::Unary { op: UnOp::Neg, .. },
                ..
            }
        ));
        assert!(matches!(
            &b[1],
            Stmt::Let {
                value: Expr::Unary { op: UnOp::Not, .. },
                ..
            }
        ));
    }

    #[test]
    fn precedence_mul_binds_tighter_than_add() {
        let b = body("kernel k(x: i32) { let a = 1 + 2 * 3; }");
        // 1 + (2 * 3): top node is Add, its rhs is a Mul
        if let Stmt::Let {
            value: Expr::Binary { op, rhs, .. },
            ..
        } = &b[0]
        {
            assert_eq!(*op, BinOp::Add);
            assert!(matches!(rhs.as_ref(), Expr::Binary { op: BinOp::Mul, .. }));
        } else {
            panic!("expected binary let");
        }
    }

    #[test]
    fn all_binary_operators_parse() {
        let b = body(
            "kernel k(x: i32) {
                let a = x + x - x * x / x % x
                let c = (x < x) == (x <= x)
                let d = (x > x) != (x >= x)
            }",
        );
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn calls_with_and_without_args() {
        let b = body("kernel k(A: tensor<f32>[N]) { let i = program_id(0); let s = dot_t(A, A); }");
        assert!(matches!(
            &b[0],
            Stmt::Let { value: Expr::Call { callee, args }, .. } if callee == "program_id" && args.len() == 1
        ));
        assert!(matches!(
            &b[1],
            Stmt::Let { value: Expr::Call { callee, args }, .. } if callee == "dot_t" && args.len() == 2
        ));
    }

    #[test]
    fn attribute_keyword_and_positional_literals() {
        let p = parse(
            "@cfg(beta = 1.5, flag = true, 2.0, false)
             kernel k(A: tensor<f32>[N]) { A[0] = 1.0 }",
        );
        let args = &p[0].attrs[0].args;
        assert!(
            matches!(&args[0], AttrArg::KeyValue { key, value: Literal::Float(_) } if key == "beta")
        );
        assert!(matches!(
            &args[1],
            AttrArg::KeyValue {
                value: Literal::Bool(true),
                ..
            }
        ));
        assert!(matches!(args[2], AttrArg::Positional(Literal::Float(_))));
        assert!(matches!(args[3], AttrArg::Positional(Literal::Bool(false))));
    }

    #[test]
    fn err_missing_paren_after_kernel_name() {
        assert!(parse_err("kernel k { }").contains("expected '('"));
    }

    #[test]
    fn err_unknown_type() {
        assert!(parse_err("kernel k(a: widget[N]) { }").contains("unknown type"));
    }

    #[test]
    fn err_bad_scalar_element_type() {
        assert!(parse_err("kernel k(a: tensor<widget>[N]) { }").contains("scalar element type"));
    }

    #[test]
    fn err_call_target_must_be_identifier() {
        let e = parse_err("kernel k(a: i32) { let x = 1(2) }");
        assert!(e.contains("call target must be an identifier"), "got: {e}");
    }

    #[test]
    fn err_invalid_assignment_target() {
        assert!(parse_err("kernel k(a: i32) { 1 = 2 }").contains("invalid assignment target"));
    }

    #[test]
    fn err_indexed_assignment_target_must_be_a_name() {
        // a doubly-indexed target's base is itself an index, not a name
        let e = parse_err("kernel k(A: tensor<f32>[N]) { A[0][1] = 1.0 }");
        assert!(e.contains("name or an indexed name"), "got: {e}");
    }

    #[test]
    fn err_for_requires_range() {
        let e = parse_err("kernel k(A: tensor<f32>[N]) { for i in foo(0, 4) { } }");
        assert!(e.contains("range"), "got: {e}");
    }

    #[test]
    fn err_unexpected_token_in_expression() {
        let e = parse_err("kernel k(a: i32) { let x = ) }");
        assert!(e.contains("unexpected token in expression"), "got: {e}");
    }

    #[test]
    fn err_expected_end_of_statement() {
        // two expressions on one line with no terminator between them
        let e = parse_err("kernel k(a: i32) { let x = 1 2 }");
        assert!(e.contains("end of statement"), "got: {e}");
    }

    #[test]
    fn error_messages_carry_line_and_column() {
        // the offending widget is on line 2
        let e = parse_err("kernel k(\n  a: widget[N]) { }");
        assert!(e.starts_with("2:"), "expected a line:col prefix, got: {e}");
    }
}
