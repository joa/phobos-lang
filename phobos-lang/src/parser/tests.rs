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
