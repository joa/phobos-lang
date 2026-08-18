// The guardrail from docs/MODULARIZE.md: a tree that was split into modules
// once grows back unless something objects, so this objects. It runs as part of
// `cargo test` rather than as a separate task because a check nobody remembers
// to run is not a guardrail.
//
// The unit is physical lines, so `wc -l` is a second opinion. Generated code is
// excluded by construction: everything the build scripts emit lands in OUT_DIR,
// which is under `target/`, which the walk skips.

use std::{
    path::{Path, PathBuf},
    sync::LazyLock,
};

/// Longest a source file may be.
const MAX_FILE_LINES: usize = 900;

/// Longest an inline `mod tests` block may be before it moves into its own file
/// beside the module it covers.
const MAX_INLINE_TEST_LINES: usize = 150;

/// The files that predate the cap, each with the length it had the day it was
/// written down. They may shrink and never grow, and an entry that falls under
/// the cap has to be deleted: that is the whole ratchet. The list only ever
/// gets shorter, and when it is empty the cap drops to 700.
const GRANDFATHERED: &[(&str, usize)] = &[
    ("phobos-gguf/src/qwen35.rs", 933),
    ("phobos-onnx/src/backend/chain.rs", 917),
    ("phobos-pod/src/engine.rs", 915),
    ("phobos-sched/src/lib.rs", 970),
];

#[test]
fn no_source_file_is_longer_than_the_cap() {
    let mut over = Vec::new();

    for file in sources() {
        let name = relative(&file);
        let lines = read(&file).lines().count();

        match GRANDFATHERED.iter().find(|(path, _)| *path == name) {
            Some(&(_, allowed)) if lines > allowed => over.push(format!(
                "{name}: {lines} lines, grandfathered at {allowed} and may only shrink"
            )),
            Some(_) => {}
            None if lines > MAX_FILE_LINES => over.push(format!(
                "{name}: {lines} lines, cap is {MAX_FILE_LINES}, split it"
            )),
            None => {}
        }
    }

    assert!(over.is_empty(), "\n{}\n", over.join("\n"));
}

#[test]
fn no_inline_test_module_is_longer_than_the_cap() {
    let mut over = Vec::new();

    for file in sources() {
        for (start_line, lines) in inline_test_modules(&read(&file)) {
            if lines > MAX_INLINE_TEST_LINES {
                let name = relative(&file);
                over.push(format!(
                    "{name}:{start_line}: mod tests is {lines} lines, cap is \
                     {MAX_INLINE_TEST_LINES}, move it to its own file"
                ));
            }
        }
    }

    assert!(over.is_empty(), "\n{}\n", over.join("\n"));
}

#[test]
fn the_grandfathered_list_holds_nothing_stale() {
    let mut stale = Vec::new();

    for &(name, allowed) in GRANDFATHERED {
        let path = ROOT.join(name);
        if !path.exists() {
            stale.push(format!("{name}: gone, drop the entry"));
            continue;
        }
        let lines = read(&path).lines().count();
        if lines <= MAX_FILE_LINES {
            stale.push(format!(
                "{name}: down to {lines} lines, under the cap, drop the entry"
            ));
        } else if lines < allowed {
            stale.push(format!(
                "{name}: down to {lines} lines from {allowed}, tighten the entry"
            ));
        }
    }

    assert!(stale.is_empty(), "\n{}\n", stale.join("\n"));
}

/// The first directory at or above this crate whose manifest declares the
/// workspace.
static ROOT: LazyLock<PathBuf> = LazyLock::new(|| {
    let mut dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.exists() && read(&manifest).contains("[workspace]") {
            return dir.to_path_buf();
        }
        dir = dir
            .parent()
            .expect("no [workspace] manifest above this crate");
    }
});

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Workspace-relative, with forward slashes, so the list above reads the same
/// on every platform.
fn relative(path: &Path) -> String {
    path.strip_prefix(ROOT.as_path())
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Every `.rs` file in the workspace, skipping build output and anything
/// hidden.
fn sources() -> Vec<PathBuf> {
    let mut found = Vec::new();
    walk(&ROOT, &mut found);
    found.sort();
    found
}

fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();

        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                walk(&path, found);
            }
        } else if path.extension().is_some_and(|e| e == "rs") {
            found.push(path);
        }
    }
}

/// The (first line, length in lines) of every inline `mod tests { .. }`.
///
/// Brace counting has to know what is code: this tree embeds kernel sources in
/// test strings, and one of them opens a brace it never closes on the Rust side.
fn inline_test_modules(src: &str) -> Vec<(usize, usize)> {
    let chars: Vec<char> = src.chars().collect();
    let mut found = Vec::new();

    let mut line = 1usize;
    let mut depth = 0usize;
    // The depth at which each currently open `mod tests` block sits.
    let mut open: Vec<(usize, usize)> = Vec::new();
    // How much of `mod tests` has been read: 0 none, 1 `mod`, 2 `mod tests`.
    let mut seen = 0u8;
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];

        if c == '\n' {
            line += 1;
            i += 1;
            continue;
        }

        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // Comments.
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && chars.get(i + 1) == Some(&'*') {
            let mut nesting = 1usize;
            i += 2;
            while i < chars.len() && nesting > 0 {
                match (chars[i], chars.get(i + 1)) {
                    ('/', Some(&'*')) => {
                        nesting += 1;
                        i += 2;
                    }
                    ('*', Some(&'/')) => {
                        nesting -= 1;
                        i += 2;
                    }
                    ('\n', _) => {
                        line += 1;
                        i += 1;
                    }
                    _ => i += 1,
                }
            }
            continue;
        }

        // Raw strings, with the `r`/`br` prefix and however many hashes.
        if let Some(len) = raw_string_len(&chars, i) {
            line += chars[i..i + len].iter().filter(|&&c| c == '\n').count();
            i += len;
            seen = 0;
            continue;
        }

        // Ordinary strings, and byte strings, which escape with a backslash.
        if c == '"' {
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                match chars[i] {
                    '\\' => i += 2,
                    '\n' => {
                        line += 1;
                        i += 1;
                    }
                    _ => i += 1,
                }
            }
            i += 1;
            seen = 0;
            continue;
        }

        // A quote is a char literal only when it closes; otherwise it is a
        // lifetime, and `&'c Block<'c>` must not swallow the code between.
        if c == '\'' {
            if let Some(len) = char_literal_len(&chars, i) {
                i += len;
                seen = 0;
                continue;
            }
            i += 1;
            continue;
        }

        if c == '_' || c.is_alphanumeric() {
            let start = i;
            while i < chars.len() && (chars[i] == '_' || chars[i].is_alphanumeric()) {
                i += 1;
            }
            let ident: String = chars[start..i].iter().collect();
            seen = match (seen, ident.as_str()) {
                (_, "mod") => 1,
                (1, "tests") => 2,
                _ => 0,
            };
            continue;
        }

        match c {
            '{' => {
                depth += 1;
                if seen == 2 {
                    open.push((depth, line));
                }
            }
            '}' => {
                if open.last().is_some_and(|&(d, _)| d == depth) {
                    let (_, start_line) = open.pop().expect("checked above");
                    found.push((start_line, line - start_line + 1));
                }
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }

        seen = 0;
        i += 1;
    }

    found
}

/// Length in chars of the raw string starting at `i`, if one does.
fn raw_string_len(chars: &[char], i: usize) -> Option<usize> {
    let mut at = i;
    if chars.get(at) == Some(&'b') {
        at += 1;
    }
    if chars.get(at) != Some(&'r') {
        return None;
    }
    // `r` after an identifier character is part of that identifier.
    if i > 0 && (chars[i - 1] == '_' || chars[i - 1].is_alphanumeric()) {
        return None;
    }
    at += 1;

    let hash_start = at;
    while chars.get(at) == Some(&'#') {
        at += 1;
    }
    let hashes = at - hash_start;
    if chars.get(at) != Some(&'"') {
        return None;
    }
    at += 1;

    let closing: Vec<char> = std::iter::once('"')
        .chain(std::iter::repeat_n('#', hashes))
        .collect();
    while at < chars.len() {
        if chars[at..].starts_with(&closing) {
            return Some(at + closing.len() - i);
        }
        at += 1;
    }
    Some(chars.len() - i)
}

/// Length in chars of the char literal starting at `i`, if one does.
fn char_literal_len(chars: &[char], i: usize) -> Option<usize> {
    match chars.get(i + 1)? {
        '\\' => {
            let mut at = i + 2;
            // Escapes run to their closing quote: `\n`, `\'`, `\u{1F600}`.
            while at < chars.len() && chars[at] != '\'' {
                at += 1;
            }
            (at < chars.len()).then_some(at + 1 - i)
        }
        _ if chars.get(i + 2) == Some(&'\'') => Some(3),
        _ => None,
    }
}

#[test]
fn brace_counting_knows_code_from_text() {
    let src = "\
#[cfg(test)]
mod tests {
    fn f<'c>(x: &'c str) -> char {
        let _ = \"kernel k() {\";
        let _ = r#\"} \" {\"#;
        let _ = '}';
        '{'
    }
}
";
    assert_eq!(inline_test_modules(src), vec![(2, 8)]);
}

#[test]
fn a_module_declaration_is_not_a_module_body() {
    let src = "#[cfg(test)]\nmod tests;\n\nfn f() {\n}\n";
    assert!(inline_test_modules(src).is_empty());
}
