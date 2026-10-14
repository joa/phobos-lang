use crate::ast::*;

fn parse(src: &str) -> Vec<Kernel> {
    crate::parse(src).unwrap()
}

#[test]
fn parses_tensor_kernel() {
    let p = parse(
        "kernel add(X: tensor<f32>[N], Y: tensor<f32>[N], Z: tensor<f32>[N]) {
            let i = ctx.block.x
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
        kernel k(A: tensor<f32>[N]) { let i = ctx.block.x }";
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
        value: Expr { kind: ExprKind::Index { subs, .. }, .. },
        ..
    } = &p[0].body[0]
    {
        assert!(matches!(subs[0], Sub::Point(Expr { kind: ExprKind::Int(0), .. })));
        assert!(matches!(subs[1], Sub::Full));
    } else {
        panic!("expected indexed let");
    }
}

#[test]
fn range_and_span_subscripts() {
    let p = parse("kernel k(A: tensor<f32>[M, N]) { let r = A[i : j, i :+ n]; }");
    if let Stmt::Let {
        value: Expr { kind: ExprKind::Index { subs, .. }, .. },
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
            value: Expr { kind: ExprKind::Unary { op: UnOp::Neg, .. }, .. },
            ..
        }
    ));
    assert!(matches!(
        &b[1],
        Stmt::Let {
            value: Expr { kind: ExprKind::Unary { op: UnOp::Not, .. }, .. },
            ..
        }
    ));
}

#[test]
fn precedence_mul_binds_tighter_than_add() {
    let b = body("kernel k(x: i32) { let a = 1 + 2 * 3; }");
    // 1 + (2 * 3): top node is Add, its rhs is a Mul
    if let Stmt::Let {
        value: Expr { kind: ExprKind::Binary { op, rhs, .. }, .. },
        ..
    } = &b[0]
    {
        assert_eq!(*op, BinOp::Add);
        assert!(matches!(&rhs.kind, ExprKind::Binary { op: BinOp::Mul, .. }));
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
    let b = body("kernel k(A: tensor<f32>[N]) { let i = ctx.block.x; let s = dot_t(A, A); }");
    assert!(matches!(
        &b[0],
        Stmt::Let {
            value: Expr { kind: ExprKind::Call { callee, args }, .. },
            ..
        } if callee == ctx::BLOCK && args.len() == 1
    ));
    assert!(matches!(
        &b[1],
        Stmt::Let {
            value: Expr { kind: ExprKind::Call { callee, args }, .. },
            ..
        } if callee == "dot_t" && args.len() == 2
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
    assert!(parse_err("kernel k { }").contains("expected `(`"));
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
    assert!(e.contains("expected an expression, found `)`"), "got: {e}");
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

/// The value of the first `let` in a one-statement kernel.
fn let_value(src: &str) -> Expr {
    let p = parse(&format!("kernel k(a: i32, b: i32, c: i32) {{ {src} }}"));
    match &p[0].body[0] {
        Stmt::Let { value, .. } => value.clone(),
        other => panic!("expected a let, got {other:?}"),
    }
}

/// The operator at the root of an expression and its two operands.
fn binary(e: &Expr) -> (BinOp, &Expr, &Expr) {
    match &e.kind {
        ExprKind::Binary { op, lhs, rhs } => (*op, lhs, rhs),
        other => panic!("expected a binary expression, got {other:?}"),
    }
}

#[test]
fn bitwise_binds_tighter_than_comparison_and_looser_than_arithmetic() {
    // ((a + b) << 2) & 7
    let e = let_value("let x = a + b << 2 & 7");
    let (op, lhs, rhs) = binary(&e);
    assert_eq!(op, BinOp::BitAnd);
    assert!(matches!(&rhs.kind, ExprKind::Int(7)));
    let (op, lhs, _) = binary(lhs);
    assert_eq!(op, BinOp::Shl);
    assert_eq!(binary(lhs).0, BinOp::Add);

    // (a & 15) == 3
    let e = let_value("let x = a & 15 == 3");
    let (op, lhs, _) = binary(&e);
    assert_eq!(op, BinOp::Eq);
    assert_eq!(binary(lhs).0, BinOp::BitAnd);
}

#[test]
fn bitwise_levels_order_and_then_xor_then_or() {
    // a | (b ^ (c & 1))
    let e = let_value("let x = a | b ^ c & 1");
    let (op, _, rhs) = binary(&e);
    assert_eq!(op, BinOp::BitOr);
    let (op, _, rhs) = binary(rhs);
    assert_eq!(op, BinOp::BitXor);
    assert_eq!(binary(rhs).0, BinOp::BitAnd);
}

#[test]
fn logical_and_binds_tighter_than_or() {
    // (a < b && b < c) || a == c
    let e = let_value("let x = a < b && b < c || a == c");
    let (op, lhs, rhs) = binary(&e);
    assert_eq!(op, BinOp::Or);
    assert_eq!(binary(lhs).0, BinOp::And);
    assert_eq!(binary(rhs).0, BinOp::Eq);
}

#[test]
fn bit_not_is_unary() {
    let e = let_value("let x = ~a & 3");
    let (op, lhs, _) = binary(&e);
    assert_eq!(op, BinOp::BitAnd);
    assert!(matches!(&lhs.kind, ExprKind::Unary { op: UnOp::BitNot, .. }));
}

#[test]
fn compound_assignment_becomes_a_set() {
    let p = parse("kernel k(a: i32) { var x = a\n x -= 2\n x <<= 1 }");
    for (stmt, want) in p[0].body[1..].iter().zip([BinOp::Sub, BinOp::Shl]) {
        let Stmt::Assign {
            target, op, value, ..
        } = stmt
        else {
            panic!("expected an assignment, got {stmt:?}");
        };
        assert_eq!(*op, AssignOp::Set);
        let (bop, lhs, _) = binary(value);
        assert_eq!(bop, want);
        assert_eq!((target.as_var(), lhs.as_var()), (Some("x"), Some("x")));
    }
}

#[test]
fn min_and_max_calls_become_operators() {
    let e = let_value("let x = min(a, max(b, c))");
    let (op, _, rhs) = binary(&e);
    assert_eq!(op, BinOp::Min);
    assert_eq!(binary(rhs).0, BinOp::Max);
    // Other arities stay calls, for the builder to reject.
    assert!(matches!(let_value("let x = max(a)"), Expr { kind: ExprKind::Call { .. }, .. }));
}

#[test]
fn open_ended_range_subscript() {
    let p = parse("kernel k(A: tensor<f32>[M, N]) { let r = A[2:, 0 :+ 4] }");
    let Stmt::Let {
        value: Expr { kind: ExprKind::Index { subs, .. }, .. },
        ..
    } = &p[0].body[0]
    else {
        panic!("expected an indexed let");
    };
    assert!(matches!(subs[0], Sub::From(Expr { kind: ExprKind::Int(2), .. })));
    assert!(matches!(subs[1], Sub::Span { .. }));
}

#[test]
fn range_to_subscript_starts_at_zero() {
    let p = parse("kernel k(A: tensor<f32>[M, N]) { let r = A[:4, :] }");
    let Stmt::Let {
        value: Expr { kind: ExprKind::Index { subs, .. }, .. },
        ..
    } = &p[0].body[0]
    else {
        panic!("expected an indexed let");
    };
    assert!(matches!(
        subs[0],
        Sub::Range {
            start: Expr { kind: ExprKind::Int(0), .. },
            end: Expr { kind: ExprKind::Int(4), .. }
        }
    ));
    assert!(matches!(subs[1], Sub::Full));
}

#[test]
fn break_becomes_a_flag_the_loop_tests() {
    let b = body(
        "kernel k(A: tensor<f32>[N], n: i32) {
            var i = 0
            while i < n {
                if A[i] > 0.0 {
                    break
                }
                i = i + 1
            }
        }",
    );
    // var i, var #break1, while.
    let [
        _,
        Stmt::Var { name: flag, .. },
        Stmt::While { cond, body, .. },
    ] = &b[..]
    else {
        panic!("unexpected shape: {b:?}");
    };
    assert_eq!(flag, "#break1");
    let ExprKind::Binary { op: BinOp::And, rhs, .. } = &cond.kind else {
        panic!("the condition does not test the flag: {cond:?}");
    };
    assert!(matches!(&rhs.kind, ExprKind::Unary { op: UnOp::Not, .. }));
    // The increment runs only when the `if` did not break.
    let [Stmt::If { then, .. }, Stmt::If { then: rest, r#else: None, .. }] = &body[..] else {
        panic!("unexpected loop body: {body:?}");
    };
    assert!(matches!(&then[..], [Stmt::Assign { .. }]));
    assert!(matches!(&rest[..], [Stmt::Assign { .. }]));
}

#[test]
fn continue_is_cleared_each_iteration() {
    let b = body(
        "kernel k(A: tensor<f32>[N], n: i32) {
            for i in range(0, n) {
                if A[i] < 0.0 {
                    continue
                }
                A[i] = 1.0
            }
        }",
    );
    let [Stmt::Var { name: flag, .. }, Stmt::For { body, .. }] = &b[..] else {
        panic!("unexpected shape: {b:?}");
    };
    assert_eq!(flag, "#continue1");
    let Stmt::Assign {
        value: Expr { kind: ExprKind::Bool(false), .. },
        ..
    } = &body[0]
    else {
        panic!("the flag is not cleared first: {body:?}");
    };
}

/// `span` as lines and columns of `src`, both ends.
fn lines(src: &str, span: Span) -> (u32, u32, u32, u32) {
    let file = crate::diag::SourceFile::new("", src);
    let ((line, col), (end_line, end_col)) = (file.line_col(span.lo), file.line_col(span.hi));
    (line, col, end_line, end_col)
}

#[test]
fn statements_carry_their_spans() {
    let src = "kernel k(A: tensor<f32>[N]) {
  let i = ctx.block.x // the block
  for j in range(0, N, 4) {
    A[j] = 1.0;
  }
}";
    let b = body(src);
    assert_eq!(lines(src, b[0].span()), (2, 3, 2, 22));
    // A compound statement's span is its header.
    assert_eq!(lines(src, b[1].span()), (3, 3, 3, 26));
    let Stmt::For { body: inner, .. } = &b[1] else {
        panic!("expected a for: {b:?}");
    };
    assert_eq!(lines(src, inner[0].span()), (4, 5, 4, 15));
}

#[test]
fn spans_survive_inlining_and_jumps() {
    let src = "fn bump(t) {
  t[0] = 2.0
}
kernel k(A: tensor<f32>[N]) {
  for i in range(0, N) {
    break
  }
  bump(A)
}";
    let b = body(src);
    // The break's flag is declared at its loop and set at the break.
    let [
        Stmt::Var { span: decl, .. },
        Stmt::For { body: inner, .. },
        inlined,
    ] = &b[..]
    else {
        panic!("unexpected shape: {b:?}");
    };
    assert_eq!(lines(src, *decl), (5, 3, 5, 23));
    let Stmt::If { then, .. } = &inner[0] else {
        panic!("expected the guarded body: {inner:?}");
    };
    assert_eq!(lines(src, then[0].span()), (6, 5, 6, 10));
    // A spliced statement keeps the line it was written on.
    assert_eq!(lines(src, inlined.span()), (2, 3, 2, 13));
}

/// The source a span covers.
fn spanned(src: &str, span: crate::diag::Span) -> &str {
    &src[span.lo as usize..span.hi as usize]
}

#[test]
fn expressions_span_their_source() {
    let src = "kernel k(A: tensor<f32>[N], b: f32, c: i32) {
    let x = -b + exp(A[c, 0 :+ 4]) * (c - 1)
    let y = ctx.block.x + min(c, 2)
}";
    let p = parse(src);
    let Stmt::Let { value, .. } = &p[0].body[0] else { panic!() };
    assert_eq!(spanned(src, value.span), "-b + exp(A[c, 0 :+ 4]) * (c - 1)");
    let ExprKind::Binary { lhs, rhs, .. } = &value.kind else { panic!() };
    assert_eq!(spanned(src, lhs.span), "-b");
    assert_eq!(spanned(src, rhs.span), "exp(A[c, 0 :+ 4]) * (c - 1)");
    let ExprKind::Binary { lhs: call, rhs: paren, .. } = &rhs.kind else { panic!() };
    assert_eq!(spanned(src, call.span), "exp(A[c, 0 :+ 4])");
    assert_eq!(spanned(src, paren.span), "(c - 1)");
    let ExprKind::Call { args, .. } = &call.kind else { panic!() };
    assert_eq!(spanned(src, args[0].span), "A[c, 0 :+ 4]");
    let ExprKind::Index { subs, .. } = &args[0].kind else { panic!() };
    let Sub::Span { start, len } = &subs[1] else { panic!() };
    assert_eq!((spanned(src, start.span), spanned(src, len.span)), ("0", "4"));

    let Stmt::Let { value, .. } = &p[0].body[1] else { panic!() };
    let ExprKind::Binary { lhs, rhs, .. } = &value.kind else { panic!() };
    assert_eq!(spanned(src, lhs.span), "ctx.block.x");
    // `min` becomes an operator, spanning the call it was written as.
    assert_eq!(spanned(src, rhs.span), "min(c, 2)");
}

#[test]
fn compound_assignment_spans_the_statement_it_rewrites() {
    let src = "kernel k(a: i32) {
    var x = 1
    x *= a + 2
}";
    let p = parse(src);
    let Stmt::Assign { target, value, .. } = &p[0].body[1] else { panic!() };
    assert_eq!(spanned(src, target.span), "x");
    assert_eq!(spanned(src, value.span), "x *= a + 2");
}

#[test]
fn names_attributes_and_params_carry_spans() {
    let src = "@launch(64, 2)
@autotune(T in [16, 32])
kernel copy(X: tensor<f32>[N], s: f32) {
    let i = ctx.thread
    for j in range(0, 4) {
        var t: tile<f32>[4] = 0.0
    }
}";
    let k = &parse(src)[0];
    assert_eq!(spanned(src, k.name_span), "copy");
    assert_eq!(spanned(src, k.attrs[0].span), "@launch(64, 2)");
    let args: Vec<&str> = k.attrs[0].arg_spans.iter().map(|&s| spanned(src, s)).collect();
    assert_eq!(args, ["64", "2"]);
    assert_eq!(spanned(src, k.attrs[1].arg_spans[0]), "T in [16, 32]");
    assert_eq!(spanned(src, k.params[0].span), "X");
    assert_eq!(spanned(src, k.params[0].ty_span), "tensor<f32>[N]");
    assert_eq!(spanned(src, k.params[1].ty_span), "f32");
    let Stmt::Let { name_span, .. } = &k.body[0] else { panic!() };
    assert_eq!(spanned(src, *name_span), "i");
    let Stmt::For { var_span, body, .. } = &k.body[1] else { panic!() };
    assert_eq!(spanned(src, *var_span), "j");
    let Stmt::Var { name_span, .. } = &body[0] else { panic!() };
    assert_eq!(spanned(src, *name_span), "t");
}
