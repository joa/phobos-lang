// Each `tests/ui/*.ph` fails to compile, and what the compiler prints for it,
// uncolored, is the `.stderr` beside it. `PHOBOS_BLESS=1` rewrites the
// `.stderr` files from what it prints now; review the diff.

use std::path::Path;

use phobos_base::context::Context;
use phobos_lang::diag::{self, Colors, SourceFile};

fn printed(name: &str, src: &str) -> Result<String, String> {
    let file = SourceFile::new(name, src);
    let err = match phobos_lang::parse_source(&file) {
        Err(e) => e,
        Ok(_) => match phobos_lang::compile(&Context::default(), src) {
            Err(e) => e,
            Ok(_) => return Err(format!("{name}: compiles, but a UI test must fail")),
        },
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
