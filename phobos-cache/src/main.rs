// Fills, lists and clears the compiled-kernel cache.
//
//     phobos-cache warm --manifest DIR [--chip sm_86]... [--dir CACHE] [--jobs N]
//                       [--force] [--ptxas PATH | --no-ptxas]
//     phobos-cache list [--chip sm_86]... [--dir CACHE]
//     phobos-cache clear [--chip sm_86]... [--dir CACHE] [KERNEL...]
//
// `warm` compiles every request a manifest holds (see `PHOBOS_KERNEL_MANIFEST`)
// for each chip, with no GPU; a run on a card of that chip then starts warm.
// `--chip` defaults to every supported chip, and `--dir` to the cache a run
// reads. Each PTX is assembled with `ptxas` for its chip before it is stored,
// when one is found on PATH or under CUDA_PATH, since no card here can load
// the foreign chips' output to check it.
//
// `clear` takes kernel names with `*` as the only wildcard, and clears
// everything, including entries from before the cache was split by chip, when
// given none.

mod entries;
mod warm;

use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use phobos_base::context::SUPPORTED_CHIPS;
use phobos_kernels::util::kernel_cache_dir;

const USAGE: &str = "usage: phobos-cache <warm|list|clear> [--chip CHIP]... [--dir CACHE] \
                     [--manifest DIR]... [--jobs N] [--force] [--ptxas PATH | --no-ptxas] [KERNEL...]";

/// The command line after the subcommand: repeatable `--flag value` pairs,
/// switches, and the positionals left over.
#[derive(Default)]
struct Args {
    chips: Vec<String>,
    manifests: Vec<PathBuf>,
    dir: Option<PathBuf>,
    jobs: Option<usize>,
    force: bool,
    ptxas: Option<PathBuf>,
    no_ptxas: bool,
    patterns: Vec<String>,
}

impl Args {
    fn parse(mut raw: impl Iterator<Item = String>) -> Result<Args> {
        let mut args = Args::default();
        while let Some(arg) = raw.next() {
            let mut value = || raw.next().with_context(|| format!("{arg} needs a value"));
            match arg.as_str() {
                "--chip" => args.chips.push(value()?),
                "--manifest" => args.manifests.push(value()?.into()),
                "--dir" => args.dir = Some(value()?.into()),
                "--jobs" => args.jobs = Some(value()?.parse().context("--jobs")?),
                "--force" => args.force = true,
                "--ptxas" => args.ptxas = Some(value()?.into()),
                "--no-ptxas" => args.no_ptxas = true,
                flag if flag.starts_with("--") => bail!("unknown flag {flag}\n{USAGE}"),
                _ => args.patterns.push(arg),
            }
        }
        for chip in &args.chips {
            if !SUPPORTED_CHIPS.contains(&chip.as_str()) {
                bail!("{chip} is not a supported chip; expected one of {}", SUPPORTED_CHIPS.join(", "));
            }
        }
        Ok(args)
    }

    /// The chips asked for, or every supported one.
    fn chips_or_all(&self) -> Vec<String> {
        if self.chips.is_empty() {
            SUPPORTED_CHIPS.iter().map(|c| c.to_string()).collect()
        } else {
            self.chips.clone()
        }
    }

    fn root(&self) -> Result<PathBuf> {
        self.dir
            .clone()
            .or_else(kernel_cache_dir)
            .context("caching is disabled (PHOBOS_KERNEL_CACHE_DIR is empty) and no --dir was given")
    }
}

fn main() -> Result<()> {
    let mut raw = std::env::args().skip(1);
    let command = raw.next().unwrap_or_default();
    let args = Args::parse(raw)?;
    match command.as_str() {
        "warm" => warm::run(&args),
        "list" => entries::list(&args),
        "clear" => entries::clear(&args),
        _ => bail!("{USAGE}"),
    }
}
