use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use phobos_kernels::manifest::{self, Request};

use crate::Args;

/// Compiles every manifest request for every chip asked for into the cache,
/// skipping what is already there unless `--force`. Runs `--jobs` lowerings
/// at once, all cores by default.
pub fn run(args: &Args) -> Result<()> {
    if args.manifests.is_empty() {
        bail!("warm needs at least one --manifest DIR, recorded with PHOBOS_KERNEL_MANIFEST");
    }
    let root = args.root()?;
    let chips = args.chips_or_all();
    let ptxas = find_ptxas(args);
    match &ptxas {
        Some(path) => println!("assembling each PTX with {}", path.display()),
        None => println!("no ptxas: PTX is checked by the MLIR verifier only"),
    }

    let mut seen = HashSet::new();
    let mut requests = Vec::new();
    for dir in &args.manifests {
        for request in manifest::read(dir)? {
            if seen.insert(request.clone()) {
                requests.push(request);
            }
        }
    }
    // Longest source first, a cheap stand-in for the slowest compile, so the
    // tail of the run is not one large kernel on one core.
    requests.sort_by_key(|r| std::cmp::Reverse(r.texts.iter().map(String::len).sum::<usize>()));

    let jobs: Vec<(&Request, &str)> = requests
        .iter()
        .flat_map(|r| chips.iter().map(move |chip| (r, chip.as_str())))
        .filter(|(r, chip)| args.force || !r.is_cached(&root, chip))
        .collect();
    let skipped = requests.len() * chips.len() - jobs.len();
    println!(
        "{} requests x {} chips into {}: {} to compile, {skipped} already cached",
        requests.len(),
        chips.len(),
        root.display(),
        jobs.len(),
    );

    let workers = args
        .jobs
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .clamp(1, jobs.len().max(1));
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());
    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                while let Some(&(request, chip)) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let result = request.compile(chip).and_then(|compiled| {
                        if let Some(ptxas) = &ptxas {
                            for ptx in &compiled.ptx {
                                assemble(ptxas, chip, ptx)?;
                            }
                        }
                        compiled.store(&root)?;
                        Ok(compiled.took)
                    });
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    match result {
                        Ok(took) => println!(
                            "[{n:>5}/{}] {chip:<6} {} {:.1} s",
                            jobs.len(),
                            request.name,
                            took.as_secs_f64()
                        ),
                        Err(e) => {
                            println!("[{n:>5}/{}] {chip:<6} {} FAILED", jobs.len(), request.name);
                            failures.lock().unwrap().push(format!("{chip} {}: {e:#}", request.name));
                        }
                    }
                }
            });
        }
    });

    let failures = failures.into_inner().unwrap();
    println!(
        "\n{} compiled, {} failed, in {:.0} s",
        jobs.len() - failures.len(),
        failures.len(),
        started.elapsed().as_secs_f64()
    );
    if !failures.is_empty() {
        for failure in &failures {
            eprintln!("{failure}");
        }
        bail!("{} of {} compiles failed", failures.len(), jobs.len());
    }
    Ok(())
}

/// `--ptxas`, else `ptxas` on PATH, else the toolkit's under `CUDA_PATH`.
fn find_ptxas(args: &Args) -> Option<PathBuf> {
    if args.no_ptxas {
        return None;
    }
    if args.ptxas.is_some() {
        return args.ptxas.clone();
    }
    let runs = |path: &Path| Command::new(path).arg("--version").output().is_ok_and(|o| o.status.success());
    let on_path = PathBuf::from("ptxas");
    if runs(&on_path) {
        return Some(on_path);
    }
    let toolkit = PathBuf::from(std::env::var_os("CUDA_PATH")?).join("bin").join("ptxas");
    runs(&toolkit).then_some(toolkit)
}

/// Assembles `ptx` for `chip` and throws the object away: what a driver
/// would do with it on first load, on a host with no such card.
fn assemble(ptxas: &Path, chip: &str, ptx: &str) -> Result<()> {
    let stem = std::env::temp_dir().join(format!(
        "phobos-cache-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let (source, object) = (stem.with_extension("ptx"), stem.with_extension("cubin"));
    std::fs::write(&source, ptx).with_context(|| format!("writing {}", source.display()))?;
    let out = Command::new(ptxas)
        .args(["--gpu-name", chip, "-o"])
        .arg(&object)
        .arg(&source)
        .output();
    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&object);
    let out = out.with_context(|| format!("running {}", ptxas.display()))?;
    if !out.status.success() {
        bail!("ptxas rejected the {chip} PTX: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(())
}
