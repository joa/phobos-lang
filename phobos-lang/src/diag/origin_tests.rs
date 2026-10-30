// A source generated from another language's: every location it reports is
// in that source, through the span map its `SourceFile` carries.

use std::sync::Arc;

use super::{Colors, Diagnostic, Diagnostics, Origin, SourceFile, Span, Spelling};

const PHOBOS: &str = "kernel k(X: tensor<i32>[4], n: i32) {
    X[0] = n / 2
}
";

const PYTHON: &str = "@ph.kernel
def k(X: ph.tensor[ph.i32][4], n: ph.i32):
    X[0] = n // 2
";

fn at(src: &str, needle: &str) -> Span {
    let lo = src.find(needle).expect("needle in source") as u32;
    Span::new(lo, lo + needle.len() as u32)
}

/// `inner` where it starts `needle`'s first occurrence.
fn within(src: &str, needle: &str, inner: &str) -> Span {
    let lo = at(src, needle).lo + needle.find(inner).expect("inner in needle") as u32;
    Span::new(lo, lo + inner.len() as u32)
}

/// `PHOBOS` as generated from `PYTHON`: the kernel's name, the statement, its
/// operands and its operator, which Python spelled `//`.
fn generated() -> Arc<SourceFile> {
    let map = vec![
        (at(PHOBOS, "k("), at(PYTHON, "k(")),
        (at(PHOBOS, "X[0] = n / 2"), at(PYTHON, "X[0] = n // 2")),
        (at(PHOBOS, "n / 2"), at(PYTHON, "n // 2")),
        (at(PHOBOS, "/"), at(PYTHON, "//")),
    ];
    let spellings = vec![(at(PHOBOS, "/"), Spelling::FloorDiv)];
    let file = SourceFile::new("k.py", PYTHON);
    SourceFile::with_origin("k.ph", PHOBOS, Origin { file, map, spellings })
}

#[test]
fn a_span_in_a_verbatim_construct_keeps_its_place_and_any_other_takes_the_construct() {
    let f = generated();
    let k = within(PHOBOS, "k(", "k");
    assert_eq!(f.locate(k), within(PYTHON, "k(", "k"));
    // `n / 2` is not written as Python writes it, so a span in it is all of it.
    let two = within(PHOBOS, "n / 2", "2");
    assert_eq!(f.locate(two), at(PYTHON, "n // 2"));
    assert_eq!(f.locate(at(PHOBOS, "X[0]")), at(PYTHON, "X[0] = n // 2"));
    assert!(f.verbatim(k) && !f.verbatim(two));
}

#[test]
fn a_spelling_is_found_between_the_operands() {
    let f = generated();
    let (lhs, rhs) = (within(PHOBOS, "n / 2", "n").hi, within(PHOBOS, "n / 2", "2").lo);
    assert_eq!(f.spelling_between(lhs, rhs), Some(Spelling::FloorDiv));
    assert_eq!(f.spelling_between(0, lhs), None);
    assert_eq!(SourceFile::new("", PHOBOS).spelling_between(lhs, rhs), None);
}

#[test]
fn a_diagnostic_renders_where_the_source_was_written() {
    let f = generated();
    let d = Diagnostic::error("bad").primary(at(PHOBOS, "n / 2"), "here");
    let out = Diagnostics::new(f.clone(), vec![d]).render(Colors::Off);
    assert!(out.contains("--> k.py:3:12"), "{out}");
    assert!(out.contains("3 |     X[0] = n // 2"), "{out}");
    let json = Diagnostics::new(f, vec![Diagnostic::error("bad").primary(at(PHOBOS, "n / 2"), "")]).to_json();
    assert!(json.contains("\"file\":\"k.py\"") && json.contains("\"line\":3"), "{json}");
}

#[test]
fn an_edit_off_a_verbatim_construct_stays_a_help() {
    let f = generated();
    let k = within(PHOBOS, "k(", "k");
    let d = Diagnostic::error("bad").primary(k, "").suggest("rename it", k, "j");
    let kept = Diagnostics::new(f.clone(), vec![d]).render(Colors::Off);
    assert!(kept.contains("def j("), "{kept}");
    let two = within(PHOBOS, "n / 2", "2");
    let d = Diagnostic::error("bad").primary(two, "").suggest("drop it", two, "");
    let dropped = Diagnostics::new(f, vec![d]).render(Colors::Off);
    assert!(dropped.contains("= help: drop it") && !dropped.contains("~"), "{dropped}");
}

#[test]
fn a_launch_report_counts_lines_of_the_source_it_was_generated_from() {
    let phobos = "@persistent(4)\nkernel k(X: tensor<f32>[64, 64]) {\n    grid_barrier()\n}\n";
    let python = "@ph.kernel\n@ph.persistent(4)\ndef k(X: ph.tensor[ph.f32][64, 64]):\n    # a comment\n    ph.grid_barrier()\n";
    let map = vec![
        (at(phobos, "k("), at(python, "k(")),
        (at(phobos, "grid_barrier()"), at(python, "ph.grid_barrier()")),
    ];
    let file = SourceFile::with_origin("k.ph", phobos, Origin { file: SourceFile::new("k.py", python), map, spellings: vec![] });
    let report = crate::report::launch_report_source(&Default::default(), &file, "k").unwrap();
    assert_eq!(report.grid_barriers, vec![5]);
}
