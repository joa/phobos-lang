// Each `tests/ui/*.ph` fails to compile, and what the compiler prints for it,
// uncolored, is the `.stderr` beside it. `PHOBOS_BLESS=1` rewrites the
// `.stderr` files from what it prints now; review the diff.

use std::path::Path;

use phobos_base::context::Context;
use phobos_lang::diag::{self, Colors, SourceFile};

fn printed(name: &str, src: &str) -> Result<String, String> {
    let file = SourceFile::new(name, src);
    let compiled = phobos_lang::compile_source(&Context::default(), &file)
        .and_then(|out| phobos_lang::assert_pipelined(&out.pipeline_failures));
    let Err(err) = compiled else {
        return Err(format!("{name}: compiles, but a UI test must fail"));
    };
    Ok(diag::render_error(&err, Colors::Off))
}

#[test]
fn every_ui_source_prints_its_stderr() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("ui");
    let bless = phobos_base::env::flag("PHOBOS_BLESS");
    let mut sources: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "ph"))
        .collect();
    sources.sort();
    assert!(!sources.is_empty());
    let mut failures = Vec::new();
    for path in sources {
        let name = format!("tests/ui/{}", path.file_name().unwrap().to_string_lossy());
        let src = std::fs::read_to_string(&path).unwrap().replace("\r\n", "\n");
        let out = match printed(&name, &src) {
            Ok(out) => out,
            Err(e) => {
                failures.push(e);
                continue;
            }
        };
        let expected_path = path.with_extension("stderr");
        if bless {
            std::fs::write(&expected_path, &out).unwrap();
            continue;
        }
        let expected = std::fs::read_to_string(&expected_path).unwrap_or_default().replace("\r\n", "\n");
        if out != expected {
            failures.push(format!("{name}: printed\n{out}\nbut {} holds\n{expected}", expected_path.display()));
        }
    }
    assert!(failures.is_empty(), "{}\nPHOBOS_BLESS=1 rewrites the .stderr files", failures.join("\n"));
}

/// Whether `text` spells an op as the compiler's printed graph does, a
/// form the source never has.
fn names_internal_op(text: &str) -> Option<&'static str> {
    const SPELLINGS: [&str; 6] = ["fused (", "slice [", "s#", "`copy`", "`fill`", "`fused`"];
    if let Some(s) = SPELLINGS.into_iter().find(|s| text.contains(s)) {
        return Some(s);
    }
    text.as_bytes()
        .windows(2)
        .any(|w| matches!(w[0], b'#' | b'%') && w[1].is_ascii_digit())
        .then_some("#N or %N")
}

#[test]
fn no_message_names_an_internal_op() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // The messages, not the keys, which name a fence as `page#N`.
    let expected = std::fs::read_to_string(root.join("../website/src/data/expected-errors.json")).unwrap();
    let messages = expected.lines().filter_map(|l| l.split_once("\": \"").map(|(_, m)| m)).collect::<Vec<_>>();
    let mut texts = vec![("website/src/data/expected-errors.json".to_string(), messages.join("\n"))];
    for entry in std::fs::read_dir(root.join("tests").join("ui")).unwrap().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "stderr") {
            texts.push((path.display().to_string(), std::fs::read_to_string(&path).unwrap()));
        }
    }
    let found: Vec<_> = texts
        .iter()
        .flat_map(|(name, text)| text.lines().map(move |line| (name, line)))
        .filter_map(|(name, line)| names_internal_op(line).map(|s| format!("{name}: `{s}` in: {line}")))
        .collect();
    assert!(found.is_empty(), "{}", found.join("\n"));
}
