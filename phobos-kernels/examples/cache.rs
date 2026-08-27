//! Lists and evicts entries in the on-disk PTX cache.
//!
//! A compile costs minutes, so a benchmarking session that changed one kernel
//! wants the other entries left alone. Pin the compiler fingerprint with
//! `PHOBOS_KERNEL_CACHE_EPOCH`, evict the kernels that actually moved, and
//! every other kernel still hits.
//!
//!     cargo run -p phobos-kernels --example cache -- list
//!     cargo run -p phobos-kernels --example cache -- evict q3k_qdot_matvec
//!     cargo run -p phobos-kernels --example cache -- evict 'iq*_qdot*'
//!
//! `evict` matches against the kernel name an entry is filed under, with `*`
//! as the only wildcard. It prints what it would remove and removes it; there
//! is nothing to undo but a recompile.

use std::fs;

use phobos_kernels::util::kernel_cache_dir;

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "list".to_string());
    let patterns: Vec<String> = args.collect();

    let Some(dir) = kernel_cache_dir() else {
        eprintln!("caching is disabled (PHOBOS_KERNEL_CACHE_DIR is empty)");
        std::process::exit(1);
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        println!("{} holds no entries", dir.display());
        return;
    };

    // `<kernel>-<hash>`, so the name is everything before the last dash.
    let mut found: Vec<(String, String, u64)> = Vec::new();
    for entry in entries.flatten() {
        let file = entry.file_name().to_string_lossy().into_owned();
        if file.starts_with('.') {
            continue;
        }
        let Some((name, _)) = file.rsplit_once('-') else { continue };
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        found.push((name.to_string(), file, size));
    }
    found.sort();

    match command.as_str() {
        "list" => {
            let total: u64 = found.iter().map(|(_, _, size)| size).sum();
            for (name, file, size) in &found {
                println!("{:>9} KiB  {name}  ({file})", size / 1024);
            }
            println!("\n{} entries, {} MiB, in {}", found.len(), total / 1024 / 1024, dir.display());
        }
        "evict" => {
            if patterns.is_empty() {
                eprintln!("evict needs at least one kernel name or glob");
                std::process::exit(1);
            }
            let mut gone = 0usize;
            for (name, file, _) in &found {
                if patterns.iter().any(|p| matches(p, name)) {
                    match fs::remove_file(dir.join(file)) {
                        Ok(()) => {
                            println!("evicted {name}");
                            gone += 1;
                        }
                        Err(e) => eprintln!("{name}: {e}"),
                    }
                }
            }
            println!("\n{gone} of {} entries evicted", found.len());
        }
        other => {
            eprintln!("unknown command {other:?}; expected list or evict");
            std::process::exit(1);
        }
    }
}

/// `*` matches any run of characters; everything else is literal.
fn matches(pattern: &str, name: &str) -> bool {
    let mut at = 0usize;
    let parts: Vec<&str> = pattern.split('*').collect();
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        let Some(found) = name[at..].find(part) else { return false };
        // An anchored first part has to sit at the very start, and an anchored
        // last one has to reach the very end.
        if i == 0 && found != 0 {
            return false;
        }
        at += found + part.len();
    }
    if !pattern.ends_with('*') && let Some(last) = parts.last() && !last.is_empty() {
        return at == name.len();
    }
    true
}
