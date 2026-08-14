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
