use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use phobos_base::context::SUPPORTED_CHIPS;
use phobos_kernels::manifest;

use crate::Args;

/// One cached kernel, found by walking the cache rather than by its key.
struct Entry {
    /// `None` for an entry from before the cache was split by chip.
    chip: Option<String>,
    name: String,
    path: PathBuf,
    size_bytes: u64,
}

/// Every entry in `root` for the chips asked for, or every entry when none
/// were, including the unsplit ones left at the top level.
fn walk(root: &Path, chips: &[String]) -> Vec<Entry> {
    let mut found = Vec::new();
    let Ok(top) = fs::read_dir(root) else { return found };
    for item in top.flatten() {
        let path = item.path();
        let file = item.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            if chips.is_empty() || chips.contains(&file) {
                files(&path, Some(&file), &mut found);
            }
        } else if chips.is_empty() {
            push(path, file, None, &mut found);
        }
    }
    found.sort_by(|a, b| (&a.chip, &a.name).cmp(&(&b.chip, &b.name)));
    found
}

fn files(dir: &Path, chip: Option<&str>, found: &mut Vec<Entry>) {
    let Ok(items) = fs::read_dir(dir) else { return };
    for item in items.flatten() {
        let file = item.file_name().to_string_lossy().into_owned();
        push(item.path(), file, chip.map(str::to_string), found);
    }
}

/// Files an entry under its kernel name, everything before the last dash in
/// `<kernel>-<hash>`. Temporary files, dot-prefixed, are no entry.
fn push(path: PathBuf, file: String, chip: Option<String>, found: &mut Vec<Entry>) {
    if file.starts_with('.') {
        return;
    }
    let Some((name, _)) = file.rsplit_once('-') else { return };
    let size_bytes = fs::metadata(&path).map_or(0, |m| m.len());
    found.push(Entry { chip, name: name.to_string(), path, size_bytes });
}

fn chip_label(chip: &Option<String>) -> &str {
    chip.as_deref().unwrap_or("unsplit")
}

pub fn list(args: &Args) -> Result<()> {
    let root = args.root()?;
    let found = walk(&root, &args.chips);
    for entry in &found {
        println!(
            "{:>9} KiB  {:<7} {}",
            entry.size_bytes / 1024,
            chip_label(&entry.chip),
            entry.name
        );
    }
    let total_bytes: u64 = found.iter().map(|e| e.size_bytes).sum();
    println!("\n{} entries, {} MiB, in {}", found.len(), total_bytes >> 20, root.display());
    Ok(())
}

pub fn clear(args: &Args) -> Result<()> {
    let root = args.root()?;
    let found = walk(&root, &args.chips);
    let mut gone = 0usize;
    let mut failed = 0usize;
    for entry in &found {
        if !args.patterns.is_empty() && !args.patterns.iter().any(|p| matches(p, &entry.name)) {
            continue;
        }
        match fs::remove_file(&entry.path) {
            Ok(()) => gone += 1,
            Err(e) => {
                eprintln!("{}: {e}", entry.path.display());
                failed += 1;
            }
        }
    }
    println!("{gone} of {} entries cleared from {}", found.len(), root.display());
    if failed > 0 {
        bail!("{failed} entries could not be removed");
    }
    Ok(())
}

/// Removes every entry the manifests do not ask for, for any supported chip,
/// so a cache warmed on top of an older one ships only what this build runs.
/// Entries of an unsupported chip and the unsplit ones go too.
pub fn prune(args: &Args) -> Result<()> {
    if args.manifests.is_empty() {
        bail!("prune needs at least one --manifest DIR");
    }
    let root = args.root()?;
    let mut keep = HashSet::new();
    for dir in &args.manifests {
        for (_, request) in manifest::read(dir)? {
            keep.extend(SUPPORTED_CHIPS.iter().map(|chip| request.entry(&root, chip)));
        }
    }
    let found = walk(&root, &[]);
    let stale: Vec<_> = found.iter().filter(|e| !keep.contains(&e.path)).collect();
    for entry in &stale {
        fs::remove_file(&entry.path)?;
    }
    println!("{} of {} entries pruned from {}", stale.len(), found.len(), root.display());
    Ok(())
}

/// `*` matches any run of characters; everything else is literal.
fn matches(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    let [first, middle @ .., last] = parts.as_slice() else {
        return pattern == name;
    };
    if name.len() < first.len() + last.len() || !name.starts_with(first) || !name.ends_with(last) {
        return false;
    }
    let mut rest = &name[first.len()..name.len() - last.len()];
    for part in middle {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::matches;

    #[test]
    fn a_pattern_without_a_star_is_the_name() {
        assert!(matches("q4k_matvec", "q4k_matvec"));
        assert!(!matches("q4k_matvec", "q4k_matvec_t"));
    }

    #[test]
    fn stars_match_any_run() {
        assert!(matches("iq*_qdot*", "iq2xs_qdot_t"));
        assert!(matches("*", "anything"));
        assert!(matches("x*yz", "xyzyz"));
        assert!(!matches("iq*_qdot*", "q8_qdot"));
        assert!(!matches("ab*ba", "aba"));
    }
}
