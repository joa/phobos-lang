// Checks each kernel's launch contract against the `.entry` it compiled to,
// over every `.ph` under the given directories and every target the emit
// sweep distinguishes. `cargo test` runs the same check over the tree; this
// runs it over a corpus, as the snapshots and `PHOBOS_DUMP_DIR` dumps are.
//
//     cargo run --release -p phobos-lang --example contract_check -- [--jobs N] DIR...
//
// It compiles on every core unless `--jobs` caps the threads, each holding a
// whole module in memory. A source that fails to compile under a target is counted, not checked:
// the corpus holds sources that fail by design.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};

const TARGETS: &[(&str, u32)] = &[("sm_75", 32), ("sm_75", 64), ("sm_80", 64), ("sm_89", 64), ("sm_120", 64)];

fn main() -> anyhow::Result<()> {
    let mut sources: Vec<PathBuf> = Vec::new();
    let mut threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--jobs" {
            threads = args.next().and_then(|n| n.parse().ok()).filter(|&n| n > 0).ok_or_else(|| anyhow::anyhow!("--jobs takes a count"))?;
            continue;
        }
        collect(&PathBuf::from(arg), &mut sources)?;
    }
    sources.sort();
    anyhow::ensure!(!sources.is_empty(), "usage: contract_check DIR...");
    let jobs: Vec<_> = sources.iter().flat_map(|s| TARGETS.iter().map(move |t| (s, *t))).collect();
    let (next, kernels, failed) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let mismatches = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            std::thread::Builder::new()
                .stack_size(256 << 20)
                .spawn_scoped(scope, || {
                    while let Some(&(path, (chip, bits))) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let Ok(src) = std::fs::read_to_string(path) else {
                            failed.fetch_add(1, Ordering::Relaxed);
                            continue;
                        };
                        let ctx = Context {
                            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip.to_string())),
                            index_bitwidth: bits,
                            ..Default::default()
                        };
                        let Ok(out) = phobos_lang::compile_raw(&ctx, &src) else {
                            failed.fetch_add(1, Ordering::Relaxed);
                            continue;
                        };
                        kernels.fetch_add(out.contracts.len(), Ordering::Relaxed);
                        for c in &out.contracts {
                            for e in phobos_lang::contract::check_entry(&out.code, c) {
                                let line = format!("{} on {chip}/{bits}, {}: {e}", path.display(), c.kernel);
                                mismatches.lock().unwrap().push(line);
                            }
                        }
                    }
                })
                .expect("spawning a check thread");
        }
    });
    let mismatches = mismatches.into_inner().unwrap();
    for m in &mismatches {
        println!("{m}");
    }
    println!(
        "{} sources x {} targets: {} kernels checked, {} mismatches, {} compiles failed",
        sources.len(),
        TARGETS.len(),
        kernels.into_inner(),
        mismatches.len(),
        failed.into_inner()
    );
    anyhow::ensure!(mismatches.is_empty(), "contracts and entries disagree");
    Ok(())
}

fn collect(dir: &PathBuf, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "ph") {
            out.push(path);
        }
    }
    Ok(())
}
