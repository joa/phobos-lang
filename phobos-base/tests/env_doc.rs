// Holds ENV.md to the tree: every `PHOBOS_*` variable the code reads has a
// row, every row is read, and a toggle is read through `phobos_base::env`
// rather than a spelling of its own. It runs under `cargo test` so neither
// can drift.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{ROOT, read, relative};

/// Set by phobos-kernels' build script and read with `env!`; never a user's.
const BUILD_TIME: &[&str] = &["PHOBOS_COMPILER_FINGERPRINT"];

/// Each `PHOBOS_*` name a string literal in Rust, or a Python `os.environ`
/// or `os.getenv` read, spells, with a file it appears in.
fn read_in_code() -> BTreeMap<String, String> {
    let mut found = BTreeMap::new();
    for file in common::files(&["rs", "py"]) {
        let text = read(&file);
        let python = file.extension().is_some_and(|e| e == "py");
        for (at, _) in text.match_indices("PHOBOS_") {
            let name: String = text[at..]
                .chars()
                .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                .collect();
            let before = &text[..at];
            let read_here = if python {
                ["environ.get(\"", "environ[\"", "getenv(\""]
                    .iter()
                    .any(|opener| before.ends_with(opener))
            } else {
                before.ends_with('"') && text[at + name.len()..].starts_with('"')
            };
            if read_here && name.len() > "PHOBOS_".len() && !BUILD_TIME.contains(&name.as_str()) {
                found.entry(name).or_insert_with(|| relative(&file));
            }
        }
    }
    found
}

/// Each variable ENV.md has a row for, with what its `takes` column says.
fn documented() -> BTreeMap<String, String> {
    read(&ROOT.join("ENV.md"))
        .lines()
        .filter_map(|line| {
            let mut cells = line.strip_prefix("| `")?.split(" | ");
            let name = cells.next()?.strip_suffix('`')?;
            Some((name.to_string(), cells.next()?.trim().to_string()))
        })
        .filter(|(name, _)| name.starts_with("PHOBOS_"))
        .collect()
}

#[test]
fn every_variable_read_is_documented_and_every_documented_one_is_read() {
    let code = read_in_code();
    let doc = documented();
    let mut wrong: Vec<String> = code
        .iter()
        .filter(|(name, _)| !doc.contains_key(*name))
        .map(|(name, file)| format!("{name} is read in {file} but has no ENV.md row"))
        .collect();
    wrong.extend(
        doc.keys()
            .filter(|name| !code.contains_key(*name))
            .map(|name| format!("{name} has an ENV.md row but nothing reads it")),
    );
    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

#[test]
fn toggles_are_read_through_the_env_helpers() {
    let toggles: BTreeSet<String> = documented()
        .into_iter()
        .filter(|(_, takes)| takes == "opt-in" || takes == "opt-out")
        .map(|(name, _)| name)
        .collect();
    let mut wrong = Vec::new();
    for file in common::files(&["rs"]) {
        let text = read(&file);
        for name in &toggles {
            for reader in ["var", "var_os"] {
                if text.contains(&format!("{reader}(\"{name}\")")) {
                    wrong.push(format!(
                        "{}: reads the toggle {name} with env::{reader}; use phobos_base::env::{{flag, flag_on, flag_set}}",
                        relative(&file)
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}
