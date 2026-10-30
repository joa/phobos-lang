// The workspace walk the tree-wide lints share.

use std::{
    path::{Path, PathBuf},
    sync::LazyLock,
};

/// The first directory at or above this crate whose manifest declares the
/// workspace.
pub static ROOT: LazyLock<PathBuf> = LazyLock::new(|| {
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

pub fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Workspace-relative, with forward slashes, so a lint's lists read the same
/// on every platform.
pub fn relative(path: &Path) -> String {
    path.strip_prefix(ROOT.as_path())
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Every file in the workspace with one of these extensions, skipping build
/// output, node_modules and anything hidden.
pub fn files(extensions: &[&str]) -> Vec<PathBuf> {
    let mut found = Vec::new();
    walk(&ROOT, extensions, &mut found);
    found.sort();
    found
}

fn walk(dir: &Path, extensions: &[&str], found: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();

        if path.is_dir() {
            // Only the build's own `target`: phobos-lang has a source
            // directory by that name.
            if path != ROOT.join("target") && name != "node_modules" && !name.starts_with('.') {
                walk(&path, extensions, found);
            }
        } else if path.extension().is_some_and(|e| extensions.iter().any(|x| e == *x)) {
            found.push(path);
        }
    }
}
