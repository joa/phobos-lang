// Compiles a Phobos kernel source to PTX. No GPU.
//
//     phobos-compile FILE.ph [--chip sm_86] [--index-bits 64] [--set TILE=64]... [-o OUT.ptx]
//     phobos-compile FILE.ph --all-variants [--set TILE=64]... [-o OUT.ptx]
//
// `--chip` defaults to sm_75, which every supported card can run.
// `--index-bits` defaults to 32, and is raised to 64 for a module the runtime
// would widen. `--set` pins an `@autotune` parameter, otherwise each takes its
// first choice. `--all-variants` writes one file per point of the search
// space left after the pins, named OUT_NAME-value_NAME-value.ptx.
// `--check-races` runs the host race check (RACECHECK.md) on what it
// compiles and fails on a race, pointing at the source lines. The standard
// library is in scope unless `--no-std`; `--stdlib FILE` puts a source of
// functions in its place. A flag's value may also follow an `=`.
// `PHOBOS_PRINT_PHASES` prints the IR after each lowering phase.
// Errors go to stderr as rustc lays them out, colored on a terminal
// (`--color`), or one JSON object per line with `--error-format json`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context as _, Result, bail};
use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig, Stdlib};
use phobos_base::term::{self, ColorChoice};
use phobos_lang::ast::Kernel;
use phobos_lang::diag::{self, Diagnostics, SourceFile};
use phobos_lang::race;

const USAGE: &str = "usage: phobos-compile FILE.ph [--chip CHIP] [--index-bits 32|64] \
                     [--set NAME=VALUE]... [--all-variants] [--check-races] [-o OUT.ptx] \
                     [--stdlib FILE | --no-std] [--color auto|always|never] [--error-format human|json]";

const HELP: &str = "
  --chip CHIP              target architecture, sm_75 by default
  --index-bits 32|64       index width, raised to 64 where the runtime would
  --set NAME=VALUE         pin an @autotune parameter, which otherwise takes
                           its first choice; repeatable
  --all-variants           compile every @autotune variant left after the
                           pins, one file each, named OUT_NAME-value.ptx
                           (OUT from -o, else the source's name); prints a
                           line per file and skips a variant that fails
  --check-races            run the host race check on each kernel compiled
                           and fail on a race between warps, a barrier not
                           every thread reaches or a shared access outside
                           its buffer
  -o, --output OUT.ptx     write the PTX here instead of standard output
  --stdlib FILE            compile against the functions FILE defines in
                           place of the standard library
  --no-std                 compile with no standard library: a source calls
                           only its own functions and the built-ins
  --color WHEN             color errors: auto (on a terminal, unless
                           NO_COLOR is set), always or never
  --error-format FORMAT    human, or json for one object per error";

#[derive(Default)]
struct Report {
    color: ColorChoice,
    json: bool,
}

impl Report {
    /// Prints `found` to stderr, as an error is.
    fn print(&self, found: &Diagnostics) {
        if self.json {
            eprint!("{}", found.to_json());
        } else {
            eprint!("{}", found.render(term::stderr_colors(self.color)));
        }
    }
}

fn main() -> ExitCode {
    let mut report = Report::default();
    match run(&mut report) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if report.json {
                eprint!("{}", diag::error_json(&e));
            } else {
                eprint!("{}", diag::render_error(&e, term::stderr_colors(report.color)));
            }
            ExitCode::FAILURE
        }
    }
}

fn run(report: &mut Report) -> Result<()> {
    let mut ctx = Context {
        print_phases: phobos_base::env::flag("PHOBOS_PRINT_PHASES"),
        ..Context::default()
    };
    let mut file: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut pins: HashMap<String, i64> = HashMap::new();
    let mut all_variants = false;
    let mut check_races = false;
    let mut stdlib: Option<PathBuf> = None;
    let mut no_std = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let (arg, inline) = match arg.split_once('=') {
            Some((flag, v)) if flag.starts_with("--") => (flag.to_string(), Some(v.to_string())),
            _ => (arg, None),
        };
        let mut value = || match inline.clone() {
            Some(v) => Ok(v),
            None => args.next().with_context(|| format!("{arg} needs a value")),
        };
        match arg.as_str() {
            "--chip" => ctx.gpu_config = GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(value()?)),
            "--index-bits" => ctx.index_bitwidth = value()?.parse().context("--index-bits")?,
            "--set" => {
                let (name, v) = parse_setting(&value()?)?;
                if pins.insert(name.clone(), v).is_some() {
                    bail!("--set {name} given twice");
                }
            }
            "--all-variants" => all_variants = true,
            "--check-races" => check_races = true,
            "--stdlib" => stdlib = Some(value()?.into()),
            "--no-std" => no_std = true,
            "-o" | "--output" => out = Some(value()?.into()),
            "--color" => report.color = value()?.parse().map_err(anyhow::Error::msg)?,
            "--error-format" => {
                report.json = match value()?.as_str() {
                    "human" => false,
                    "json" => true,
                    other => bail!("--error-format takes human or json, not `{other}`"),
                }
            }
            "--version" | "-V" => {
                let fingerprint = phobos_kernels::COMPILER_FINGERPRINT;
                println!("phobos-compile {} (compiler {fingerprint})", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "-h" | "--help" => {
                println!("{USAGE}\n{HELP}");
                return Ok(());
            }
            flag if flag.starts_with('-') => bail!("unknown flag {flag}\n{USAGE}"),
            _ if file.is_none() => file = Some(arg.into()),
            _ => bail!("one source file at a time\n{USAGE}"),
        }
    }
    let file = file.context(USAGE)?;
    ctx.stdlib = match (stdlib, no_std) {
        (Some(_), true) => bail!("--stdlib and --no-std together"),
        (Some(path), false) => {
            let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            Stdlib::Custom {
                name: path.display().to_string(),
                text: text.into(),
            }
        }
        (None, true) => Stdlib::Off,
        (None, false) => Stdlib::Builtin,
    };

    let src = std::fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
    let kernels = phobos_lang::parse_source_with(&SourceFile::new(file.display().to_string(), src), &ctx.stdlib)?;

    if !all_variants {
        // Each kernel seeds its own first choices, so only the pins' names
        // need checking, not whether the kernels' spaces agree.
        let declared = kernels.iter().flat_map(phobos_lang::search_space).collect();
        phobos_lang::pin(declared, &pins)?;
        let ctx = target(&ctx, &kernels, pins.into_iter().collect());
        if check_races && let Some(found) = races(&ctx, &kernels, report)? {
            return Err(found.into());
        }
        let ptx = compile(&ctx, &kernels)?;
        match out {
            Some(out) => {
                std::fs::write(&out, ptx).with_context(|| format!("writing {}", out.display()))?
            }
            None => println!("{ptx}"),
        }
        return Ok(());
    }

    let space = phobos_lang::pin(phobos_lang::module_search_space(&kernels)?, &pins)?;
    let stem = out.unwrap_or_else(|| PathBuf::from(file.file_stem().unwrap_or(file.as_os_str())));
    let variants = phobos_base::combo::cartesian_product(&space);
    let mut failed = 0;
    let mut raced = 0;
    for variant in &variants {
        let path = variant_path(&stem, variant);
        let settings = variant
            .iter()
            .map(|(n, v)| format!("{n}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        let ctx = target(&ctx, &kernels, variant.iter().cloned().collect());
        if check_races {
            match races(&ctx, &kernels, report) {
                Ok(None) => {}
                Ok(Some(found)) => {
                    eprintln!("{settings} races:");
                    report.print(&found);
                    raced += 1;
                    continue;
                }
                Err(e) => {
                    eprintln!("skipped {settings}: {e:#}");
                    failed += 1;
                    continue;
                }
            }
        }
        let ptx = match compile(&ctx, &kernels) {
            Ok(ptx) => ptx,
            Err(e) => {
                eprintln!("skipped {settings}: {e:#}");
                failed += 1;
                continue;
            }
        };
        std::fs::write(&path, ptx).with_context(|| format!("writing {}", path.display()))?;
        println!("{} {settings}", path.display());
    }
    if raced > 0 {
        bail!("{raced} of the {} variants race", variants.len());
    }
    if failed == variants.len() {
        bail!("none of the {failed} variants compiled");
    }
    Ok(())
}

/// `ctx` with `shape_overrides` pinned, and widened as the runtime does, so
/// a kernel compiles here exactly when it compiles there.
fn target(ctx: &Context, kernels: &[Kernel], shape_overrides: HashMap<String, i64>) -> Context {
    let mut ctx = Context {
        shape_overrides,
        ..ctx.clone()
    };
    if ctx.index_bitwidth < 64 && phobos_lang::needs_wide_index(&ctx, kernels) {
        ctx.index_bitwidth = 64;
    }
    ctx
}

/// Runs the race check on `kernels`, returning what it found when that
/// includes an error. A run that stopped early alone is printed as a
/// warning.
fn races(ctx: &Context, kernels: &[Kernel], report: &Report) -> Result<Option<Diagnostics>> {
    let reports = race::check_kernels(ctx, kernels, &race::Config::default())?;
    let Some(found) = race::diagnostics(kernels, &reports) else {
        return Ok(None);
    };
    if found.errors() > 0 {
        return Ok(Some(found));
    }
    report.print(&found);
    Ok(None)
}

fn compile(ctx: &Context, kernels: &[Kernel]) -> Result<String> {
    let mut dynamic_shared = false;
    let ptx = phobos_mlir::gen_ptx(ctx, |base, context, module| {
        let out = phobos_lang::codegen::emit(base, kernels, context, module)?;
        dynamic_shared = !out.shared.is_empty();
        phobos_lang::assert_pipelined(&out.pipeline_failures)
    })?;
    // A `@dynshared` kernel's buffer is declared external only here.
    Ok(ctx.gpu_config.backend().post_process(ptx, dynamic_shared))
}

fn parse_setting(s: &str) -> Result<(String, i64)> {
    let (name, v) = s
        .split_once('=')
        .with_context(|| format!("--set {s}: expected NAME=VALUE"))?;
    let v = v
        .parse()
        .with_context(|| format!("--set {s}: the value is not an integer"))?;
    Ok((name.to_string(), v))
}

/// `stem` with each setting appended as `_NAME-value`, and the extension
/// `.ptx`. A `.ptx` already on `stem` is replaced rather than kept inside.
fn variant_path(stem: &Path, variant: &[(String, i64)]) -> PathBuf {
    let base = match stem.extension() {
        Some(ext) if ext.eq_ignore_ascii_case("ptx") => stem.with_extension(""),
        _ => stem.to_path_buf(),
    };
    let mut name = base.file_name().unwrap_or_default().to_os_string();
    for (n, v) in variant {
        name.push(format!("_{n}-{v}"));
    }
    name.push(".ptx");
    base.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_path_appends_settings_before_the_extension() {
        let v = [("TILE_M".to_string(), 64), ("U2".to_string(), 4)];
        let want = Path::new("out").join("gemm_TILE_M-64_U2-4.ptx");
        assert_eq!(variant_path(&Path::new("out").join("gemm.ptx"), &v), want);
        assert_eq!(variant_path(&Path::new("out").join("gemm"), &v), want);
        assert_eq!(variant_path(Path::new("gemm"), &[]), Path::new("gemm.ptx"));
    }

    #[test]
    fn a_dynamic_shared_kernel_declares_its_buffer_external() {
        let src = "@dynshared
            kernel k(A: tensor<f32>[64, 64], C: tensor<f32>[64, 64]) {
                var t = A[0 :+ 64, 0 :+ 64]
                C[0 :+ 64, 0 :+ 64] = t * 2.0
            }";
        let kernels = phobos_lang::parse(src).unwrap();
        let ptx = compile(&Context::default(), &kernels).unwrap();
        assert!(ptx.contains(".extern .shared .align 16 .b8 __dynamic_shmem__0[];"), "{ptx}");
    }

    #[test]
    fn parse_setting_takes_name_equals_integer() {
        assert_eq!(parse_setting("TILE=64").unwrap(), ("TILE".to_string(), 64));
        assert!(parse_setting("TILE").is_err());
        assert!(parse_setting("TILE=x").is_err());
    }
}
