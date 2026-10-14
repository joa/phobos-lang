//! The shared-memory race check over kernel sources (`.ph`) or emitted MLIR
//! (`.mlir`, as the `emit` example prints it), files or directories of them.
//! `PHOBOS_CHIP` and `PHOBOS_INDEX_BITS` pick the target for a source, as
//! for `emit`. Exits nonzero when any CTA races or a barrier diverges.
//!
//!   cargo run -p phobos-lang --example race -- [-v] PATH...

use std::path::{Path, PathBuf};

use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};
use phobos_lang::race::{self, Config, Report};

fn main() -> anyhow::Result<()> {
    let mut verbose = false;
    let mut files = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "-v" {
            verbose = true;
        } else {
            collect(Path::new(&arg), &mut files);
        }
    }
    let base = target()?;
    let config = Config::default();
    let (mut racy, mut possible, mut stopped, mut clean, mut failed) = (0, 0, 0, 0, 0);
    for file in &files {
        let text = std::fs::read_to_string(file)?;
        let checked = if file.extension().is_some_and(|e| e == "mlir") {
            if text.starts_with("FAILED") {
                continue;
            }
            race::check_text(&text, &config)
        } else {
            race::check_source(&base, &text, &config)
        };
        let reports = match checked {
            Ok(r) => r,
            Err(e) => {
                failed += 1;
                if verbose {
                    println!("{}: {e:#}", file.display());
                }
                continue;
            }
        };
        for r in &reports {
            if !r.races.is_empty() || !r.divergent.is_empty() || !r.out_of_bounds.is_empty() {
                racy += 1;
                print(file, r, true);
            } else if r.stopped.is_some() {
                stopped += 1;
                if verbose {
                    print(file, r, false);
                }
            } else if !r.possible.is_empty() || !r.warp_races.is_empty() {
                possible += 1;
                if verbose {
                    print(file, r, false);
                }
            } else {
                clean += 1;
            }
        }
    }
    println!(
        "{} files: {racy} CTAs race, {possible} only possibly or within a warp, {stopped} stopped early, {clean} clean, {failed} not checked",
        files.len()
    );
    if racy > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn print(file: &Path, r: &Report, all: bool) {
    println!(
        "{} {} block {:?}: {} races, {} within a warp, {} possible, {} barriers",
        file.display(),
        r.kernel,
        r.block,
        r.races.len(),
        r.warp_races.len(),
        r.possible.len(),
        r.barriers
    );
    if let Some(why) = &r.stopped {
        println!("  stopped: {why}");
    }
    let show = |label: &str, races: &[race::Race]| {
        for x in races.iter().take(4) {
            println!(
                "  {label} {} byte {} ({} times)\n    t{} {} {}\n    t{} {} {}",
                x.buffer,
                x.byte,
                x.count,
                x.first.thread,
                if x.first.write { "writes" } else { "reads" },
                x.first.op,
                x.second.thread,
                if x.second.write { "writes" } else { "reads" },
                x.second.op
            );
        }
    };
    show("race", &r.races);
    if all {
        for (op, n) in &r.divergent {
            println!("  divergent barrier, {n} threads: {op}");
        }
        for o in &r.out_of_bounds {
            println!("  out of bounds: {o}");
        }
    } else {
        show("warp", &r.warp_races);
        show("possible", &r.possible);
    }
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .collect();
        entries.sort();
        for e in entries {
            collect(&e, out);
        }
    } else if path.extension().is_some_and(|e| e == "ph" || e == "mlir") {
        out.push(path.to_path_buf());
    }
}

fn target() -> anyhow::Result<Context> {
    let mut base = Context::default();
    if let Ok(chip) = std::env::var("PHOBOS_CHIP") {
        base.gpu_config = GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip));
    }
    if let Ok(bits) = std::env::var("PHOBOS_INDEX_BITS") {
        base.index_bitwidth = bits.parse()?;
    }
    Ok(base)
}
