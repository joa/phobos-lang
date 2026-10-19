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
// compiles and fails on a race, pointing at the source lines. With `-o`,
// each PTX file gets its launch contract beside it as JSON. A compile reads
// and writes the runtime's kernel cache, keyed as the runtime keys it, so
// either finds what the other compiled. The standard
// library is in scope unless `--no-std`; `--stdlib FILE` puts a source of
// functions in its place. A flag's value may also follow an `=`.
// `PHOBOS_PRINT_PHASES` prints the IR after each lowering phase.
// Errors go to stderr as rustc lays them out, colored on a terminal
// (`--color`), or one JSON object per line with `--error-format json`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig, Stdlib};
use phobos_base::term::{self, ColorChoice};
use phobos_lang::Contract;
use phobos_lang::ast::Kernel;
use phobos_lang::diag::{self, Diagnostic, Diagnostics, Origin, SourceFile, Span, Spelling};
use phobos_lang::race;

const USAGE: &str = "usage: phobos-compile FILE.ph [--chip CHIP | --capability CC] [--index-bits 32|64] \
                     [--set NAME=VALUE]... [--all-variants] [--check-races] [-o OUT.ptx] [--origin MAP.json] \
                     [--stdlib FILE | --no-std] [--color auto|always|never] [--error-format human|json]\n       \
                     phobos-compile --render DIAG.json [--color auto|always|never] [--error-format human|json]";

const HELP: &str = "
  --chip CHIP              target architecture, sm_75 by default
  --capability CC          target the newest supported chip a card of
                           compute capability CC runs, 86 for 8.6
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
  -o, --output OUT.ptx     write the PTX here instead of standard output,
                           and each kernel's launch contract to OUT.json
  --origin MAP.json        FILE was generated from another language's
                           source: MAP holds that source and where each
                           construct of FILE came from, and every error,
                           race and location is reported there
  --render DIAG.json       print the errors DIAG holds as this compiler
                           prints its own, and fail; for a front end's
                           errors before there is a FILE
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
    let mut origin: Option<PathBuf> = None;
    let mut render: Option<PathBuf> = None;
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
            "--capability" => {
                let capability = value()?.parse().context("--capability takes a compute capability, 86 for 8.6")?;
                let chip = phobos_base::context::best_supported_chip(capability);
                ctx.gpu_config = GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip));
            }
            "--index-bits" => ctx.index_bitwidth = value()?.parse().context("--index-bits")?,
            "--set" => {
                let (name, v) = parse_setting(&value()?)?;
                if pins.insert(name.clone(), v).is_some() {
                    bail!("--set {name} given twice");
                }
            }
            "--all-variants" => all_variants = true,
            "--check-races" => check_races = true,
            "--origin" => origin = Some(value()?.into()),
            "--render" => render = Some(value()?.into()),
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
    if let Some(path) = render {
        return Err(rendered(&path)?.into());
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
    let source = match origin {
        Some(path) => SourceFile::with_origin(file.display().to_string(), src, read_origin(&path)?),
        None => SourceFile::new(file.display().to_string(), src),
    };
    let kernels = phobos_lang::parse_source_with(&source, &ctx.stdlib)?;

    if !all_variants {
        // Each kernel seeds its own first choices, so only the pins' names
        // need checking, not whether the kernels' spaces agree.
        let declared = kernels.iter().flat_map(phobos_lang::search_space).collect();
        phobos_lang::pin(declared, &pins)?;
        let pinned = pinned(&ctx, pins.into_iter().collect());
        if check_races && let Some(found) = races(&target(&pinned, &kernels)?, &kernels, report)? {
            return Err(found.into());
        }
        let compiled = compile(&pinned, &source)?;
        match out {
            Some(out) => compiled.write(&out)?,
            None => println!("{}", compiled.ptx),
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
        let pinned = pinned(&ctx, variant.iter().cloned().collect());
        if check_races {
            match races(&target(&pinned, &kernels)?, &kernels, report) {
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
        let compiled = match compile(&pinned, &source) {
            Ok(compiled) => compiled,
            Err(e) => {
                eprintln!("skipped {settings}: {e:#}");
                failed += 1;
                continue;
            }
        };
        compiled.write(&path)?;
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

/// `ctx` with `shape_overrides` pinned: what the runtime compiles and keys
/// its cache on.
fn pinned(ctx: &Context, shape_overrides: HashMap<String, i64>) -> Context {
    Context {
        shape_overrides,
        ..ctx.clone()
    }
}

/// `ctx` widened as the compile widens it, so a kernel is race-checked as it
/// compiles.
fn target(ctx: &Context, kernels: &[Kernel]) -> Result<Context> {
    let mut ctx = ctx.clone();
    ctx.index_bitwidth = phobos_lang::index_bits(&ctx, kernels)?;
    Ok(ctx)
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

/// A module's PTX, the contract of each of its kernels, and the cache entry
/// they are filed under.
struct Compiled {
    ptx: String,
    contracts: Vec<Contract>,
    cache_key: String,
    chip: String,
}

impl Compiled {
    /// Writes the PTX to `path` and the contracts beside it, as `.json`.
    fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, &self.ptx).with_context(|| format!("writing {}", path.display()))?;
        let json = serde_json::json!({
            "compiler_fingerprint": phobos_kernels::COMPILER_FINGERPRINT,
            "cache_key": self.cache_key,
            "chip": self.chip,
            "kernels": self.contracts.iter().map(Contract::to_json).collect::<Vec<_>>(),
        });
        let json_path = path.with_extension("json");
        std::fs::write(&json_path, serde_json::to_string_pretty(&json)? + "\n")
            .with_context(|| format!("writing {}", json_path.display()))
    }
}

/// `source` compiled under `ctx` as the runtime compiles it, through the
/// runtime's cache: a hit is only emitted again, for its contracts. The
/// cache key does not cover the standard library, so a source compiled
/// against another one bypasses it.
fn compile(ctx: &Context, source: &Arc<SourceFile>) -> Result<Compiled> {
    let cached = matches!(ctx.stdlib, Stdlib::Builtin);
    let hit = if cached { phobos_kernels::cache::load(ctx, source.text()) } else { None };
    let (ptx, contracts) = match hit {
        Some((ptx, _)) => (ptx, phobos_lang::contracts(ctx, source)?),
        None => {
            let out = phobos_lang::compile_source(ctx, source)?;
            phobos_lang::assert_pipelined(&out.pipeline_failures)?;
            if cached {
                phobos_kernels::cache::store(ctx, source.text(), &out.code, &out.shared);
            }
            (out.code, out.contracts)
        }
    };
    let GpuConfig::Nvidia(nv) = &ctx.gpu_config;
    let cache_key = phobos_kernels::cache::key(ctx, source.text());
    Ok(Compiled { ptx, contracts, cache_key, chip: nv.chip().to_string() })
}

/// The source a front end generated FILE from, and its span map, as the
/// front end writes them: `{"file", "text", "map": [[lo, hi, from_lo,
/// from_hi]], "spellings": [[lo, hi, "true_div" | "floor_div"]]}`, offsets
/// in bytes.
fn read_origin(path: &Path) -> Result<Origin> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let str_of = |key: &str| json[key].as_str().with_context(|| format!("{}: no `{key}`", path.display()));
    let span = |v: &serde_json::Value, i: usize| -> Result<Span> {
        let n = |j: usize| v[j].as_u64().map(|n| n as u32).context("a span bound is not an offset");
        Ok(Span::new(n(i)?, n(i + 1)?))
    };
    let rows = |key: &str| json[key].as_array().cloned().unwrap_or_default();
    let map = rows("map").iter().map(|r| Ok((span(r, 0)?, span(r, 2)?))).collect::<Result<_>>()?;
    let spellings = rows("spellings")
        .iter()
        .map(|r| {
            let spelling = match r[2].as_str() {
                Some("true_div") => Spelling::TrueDiv,
                Some("floor_div") => Spelling::FloorDiv,
                other => bail!("{}: unknown spelling {other:?}", path.display()),
            };
            Ok((span(r, 0)?, spelling))
        })
        .collect::<Result<_>>()?;
    Ok(Origin { file: SourceFile::new(str_of("file")?, str_of("text")?), map, spellings })
}

/// The diagnostics a front end found before there was a kernel to compile,
/// `{"file", "text", "diagnostics": [{"message", "labels": [[lo, hi,
/// "text", primary]], "notes", "help"}]}`, for the error they make.
fn rendered(path: &Path) -> Result<Diagnostics> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let strings = |v: &serde_json::Value| -> Vec<String> {
        v.as_array().into_iter().flatten().filter_map(|s| s.as_str().map(str::to_string)).collect()
    };
    let list = json["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|d| {
            let mut out = Diagnostic::error(d["message"].as_str().unwrap_or_default());
            for l in d["labels"].as_array().into_iter().flatten() {
                let span = Span::new(l[0].as_u64().unwrap_or(0) as u32, l[1].as_u64().unwrap_or(0) as u32);
                let label = l[2].as_str().unwrap_or_default();
                out = if l[3].as_bool().unwrap_or(true) { out.primary(span, label) } else { out.secondary(span, label) };
            }
            for n in strings(&d["notes"]) {
                out = out.note(n);
            }
            for h in strings(&d["help"]) {
                out = out.help(h);
            }
            out
        })
        .collect();
    let file = SourceFile::new(json["file"].as_str().unwrap_or_default(), json["text"].as_str().unwrap_or_default());
    Ok(Diagnostics::new(file, list))
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
        let ctx = Context { stdlib: Stdlib::Off, ..Context::default() };
        let compiled = compile(&ctx, &SourceFile::new("", src)).unwrap();
        assert!(compiled.ptx.contains(".extern .shared .align 16 .b8 __dynamic_shmem__0[];"), "{}", compiled.ptx);
        assert_eq!(compiled.contracts[0].dynamic_shared_bytes, 64 * 64 * 4);
    }

    #[test]
    fn parse_setting_takes_name_equals_integer() {
        assert_eq!(parse_setting("TILE=64").unwrap(), ("TILE".to_string(), 64));
        assert!(parse_setting("TILE").is_err());
        assert!(parse_setting("TILE=x").is_err());
    }
}
