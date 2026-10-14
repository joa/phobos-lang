use super::*;

const GEMM: &str = "kernel k(A: tensor<f32>[M, K]) {
    for kt in range(0, K, 16) {
        let a = A[0 :+ 32, kt :+ 16]
        let b = B[kt :+ 32, 0 :+ 32]

        acc += dot(a, b)
    }
}
";

fn at(src: &str, needle: &str) -> Span {
    let lo = src.find(needle).expect("needle in source") as u32;
    Span::new(lo, lo + needle.len() as u32)
}

fn render(src: &str, d: Diagnostic) -> String {
    Diagnostics::new(SourceFile::new("gemm.ph", src), vec![d]).render(Colors::Off)
}

/// `expected`'s lines, each stripped up to the `|` that starts it.
fn lines(expected: &str) -> String {
    let stripped: Vec<&str> = expected
        .lines()
        .map(|l| l.trim_start().strip_prefix('|').unwrap_or(l))
        .collect();
    stripped.join("\n") + "\n"
}

#[test]
fn labels_on_several_lines_with_a_note_and_a_suggestion() {
    let d = Diagnostic::error("`dot` needs the inner dimensions to agree")
        .primary(at(GEMM, "dot(a, b)"), "[32, 16] times [32, 32]")
        .secondary(at(GEMM, "a = "), "`a` is [32, 16]")
        .secondary(at(GEMM, "b ="), "")
        .secondary(at(GEMM, "kt :+ 32"), "`b` has 32 rows from here")
        .note("`dot(a, b)` takes a as [M, K] and b as [K, N]")
        .suggest("slice `b` with the same step as `a`", at(GEMM, "kt :+ 32"), "kt :+ 16");
    assert_eq!(
        render(GEMM, d),
        lines(
            "|error: `dot` needs the inner dimensions to agree
             | --> gemm.ph:6:16
             |  |
             |3 |         let a = A[0 :+ 32, kt :+ 16]
             |  |             ---- `a` is [32, 16]
             |4 |         let b = B[kt :+ 32, 0 :+ 32]
             |  |             ---   -------- `b` has 32 rows from here
             |5 |
             |6 |         acc += dot(a, b)
             |  |                ^^^^^^^^^ [32, 16] times [32, 32]
             |  |
             |  = note: `dot(a, b)` takes a as [M, K] and b as [K, N]
             |help: slice `b` with the same step as `a`
             |  |
             |4 |         let b = B[kt :+ 16, 0 :+ 32]
             |  |                   ~~~~~~~~
             |
             |error: aborting due to 1 previous error"
        )
    );
}

#[test]
fn three_labels_on_one_line_hang_below_it() {
    let d = Diagnostic::error("three")
        .primary(at(GEMM, "a, b"), "c")
        .secondary(at(GEMM, "acc"), "a")
        .secondary(at(GEMM, "dot"), "b");
    let out = render(GEMM, d);
    let rows: Vec<&str> = out.lines().skip(3).take(5).collect();
    assert_eq!(
        rows,
        [
            "6 |         acc += dot(a, b)",
            "  |         ---    --- ^^^^ c",
            "  |         |      |",
            "  |         |      b",
            "  |         a",
        ]
    );
}

#[test]
fn a_span_over_lines_draws_a_margin() {
    let whole = at(GEMM, "for kt");
    let close = at(GEMM, "    }\n}").lo + 5;
    let d = Diagnostic::error("loop").primary(Span::new(whole.lo, close), "this loop");
    let out = render(GEMM, d);
    let rows: Vec<&str> = out.lines().skip(3).take(5).collect();
    assert_eq!(
        rows,
        [
            "2 | /     for kt in range(0, K, 16) {",
            "...",
            "7 | |     }",
            "  | |_____^ this loop",
            "",
        ]
    );

    let mid = Span::new(at(GEMM, "range").lo, at(GEMM, "let a").hi);
    let out = render(GEMM, Diagnostic::error("mid").primary(mid, "from here"));
    let rows: Vec<&str> = out.lines().skip(3).take(4).collect();
    assert_eq!(
        rows,
        [
            "2 |       for kt in range(0, K, 16) {",
            "  |  _______________^",
            "3 | |         let a = A[0 :+ 32, kt :+ 16]",
            "  | |_____________^ from here",
        ]
    );
}

#[test]
fn an_inserted_line_is_marked_and_numbered() {
    let line = at(GEMM, "        acc").lo;
    let d = Diagnostic::warning("insert")
        .primary(at(GEMM, "acc"), "")
        .suggest("add a barrier", Span::new(line, line), "        barrier()\n");
    let out = render(GEMM, d);
    assert!(
        out.contains("help: add a barrier\n  |\n6 +         barrier()\n7 |         acc += dot(a, b)\n"),
        "{out}"
    );
    assert!(out.ends_with("warning: 1 warning emitted\n"), "{out}");
}

#[test]
fn a_label_at_the_end_of_the_file_points_past_the_last_line() {
    let src = "kernel k() {\n    let x = 1";
    let end = src.len() as u32;
    let out = render(src, Diagnostic::error("unclosed").primary(Span::new(end, end), "here"));
    assert!(out.contains("2 |     let x = 1\n  |              ^ here\n"), "{out}");
}

#[test]
fn tabs_expand_and_the_underline_follows() {
    let src = "\tlet x = y\n";
    let out = render(src, Diagnostic::error("tab").primary(at(src, "y"), ""));
    assert!(out.contains("1 |     let x = y\n  |             ^\n"), "{out}");
}

#[test]
fn a_label_inside_another_keeps_its_underline() {
    let src = "var u = A[0, ctx.lane :+ 1]\n";
    let d = Diagnostic::error("nested")
        .primary(at(src, "A[0, ctx.lane :+ 1]"), "outer")
        .secondary(at(src, "ctx.lane"), "inner");
    let out = render(src, d);
    assert!(out.contains("1 | var u = A[0, ctx.lane :+ 1]\n  |         ^^^^^--------^^^^^^\n"), "{out}");
}

#[test]
fn a_long_line_is_cut_around_its_labels() {
    let src = format!("let x = {} + y + {}\n", "a + ".repeat(40), "b + ".repeat(40));
    let out = render(&src, Diagnostic::error("long").primary(at(&src, "y"), "here"));
    let rows: Vec<&str> = out.lines().skip(3).take(2).collect();
    let shown = &src[src.find('y').unwrap() - 20..src.find('y').unwrap() + 100];
    assert_eq!(rows[0], format!("1 | ...{shown}..."), "{out}");
    assert_eq!(rows[1], format!("  | {}^ here", " ".repeat(23)), "{out}");
}

#[test]
fn colors_paint_the_source_and_the_labels() {
    let d = Diagnostic::error("colored")
        .primary(at(GEMM, "dot"), "here")
        .secondary(at(GEMM, "range"), "");
    let ds = Diagnostics::new(SourceFile::new("gemm.ph", GEMM), vec![d]);
    let rgb = ds.render(Colors::Rgb);
    // `for` in the nebula, `16` in gold, `dot` called in the brand, the
    // headline in Mars.
    assert!(rgb.contains("\x1b[38;2;160;140;255mfor"), "{rgb:?}");
    assert!(rgb.contains("\x1b[38;2;255;198;107m16"), "{rgb:?}");
    assert!(rgb.contains("\x1b[38;2;106;184;255mdot"), "{rgb:?}");
    assert!(rgb.starts_with("\x1b[1m\x1b[38;2;255;138;107merror"), "{rgb:?}");
    let basic = ds.render(Colors::Basic);
    assert!(basic.contains("\x1b[95mfor") && !basic.contains("38;2;"), "{basic:?}");
    assert!(!ds.render(Colors::Off).contains('\x1b'));
}

#[test]
fn display_is_one_location_per_line() {
    let file = SourceFile::new("", GEMM);
    let ds = Diagnostics::new(
        file,
        vec![
            Diagnostic::error("first").primary(at(GEMM, "dot"), ""),
            Diagnostic::warning("second").primary(at(GEMM, "for"), ""),
            Diagnostic::error("third"),
        ],
    );
    assert_eq!(ds.to_string(), "6:16: first\n2:5: warning: second\nthird");
    let named = Diagnostics::new(SourceFile::new("k.ph", GEMM), ds.list().to_vec());
    assert!(named.to_string().starts_with("k.ph:6:16: first"));
}

#[test]
fn json_carries_spans_and_edits() {
    let d = Diagnostic::error("say \"why\"")
        .primary(at(GEMM, "dot"), "here")
        .suggest("fix", at(GEMM, "dot"), "sum");
    let json = Diagnostics::new(SourceFile::new("k.ph", GEMM), vec![d]).to_json();
    let dot = at(GEMM, "dot");
    assert!(json.contains(r#""message":"say \"why\"""#), "{json}");
    assert!(json.contains(&format!(r#""lo":{},"hi":{},"line":6,"col":16,"end_line":6,"end_col":19"#, dot.lo, dot.hi)), "{json}");
    assert!(json.contains(r#""replacement":"sum""#), "{json}");
    assert_eq!(json.lines().count(), 1);
}

#[test]
fn lines_and_columns_come_from_offsets() {
    let file = SourceFile::new("", "ab\ncd\n");
    assert_eq!(file.line_col(0), (1, 1));
    assert_eq!(file.line_col(4), (2, 2));
    // The end of a file ending in a newline is the end of its last line.
    assert_eq!(file.line_col(6), (2, 3));
    assert_eq!(file.line_count(), 2);
    assert_eq!(file.offset(2, 2), 4);
    assert_eq!(file.line_text(2), "cd");
}

#[test]
fn similar_names_are_suggested_within_a_third() {
    assert_eq!(similar("lanez", ["lane", "lanes", "warps"]), Some("lanes"));
    assert_eq!(similar("f23", ["f16", "f32", "i32"]), Some("f32"));
    assert_eq!(similar("widget", ["f32", "tile", "tensor"]), None);
    assert_eq!(similar("tile", ["tile"]), None);
}
