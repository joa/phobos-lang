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
// `PHOBOS_PRINT_PHASES` prints the IR after each lowering phase.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};
use phobos_lang::ast::Kernel;

const USAGE: &str = "usage: phobos-compile FILE.ph [--chip CHIP] [--index-bits 32|64] \
                     [--set NAME=VALUE]... [--all-variants] [-o OUT.ptx]";

const HELP: &str = "
  --chip CHIP              target architecture, sm_75 by default
  --index-bits 32|64       index width, raised to 64 where the runtime would
  --set NAME=VALUE         pin an @autotune parameter, which otherwise takes
                           its first choice; repeatable
  --all-variants           compile every @autotune variant left after the
                           pins, one file each, named OUT_NAME-value.ptx
                           (OUT from -o, else the source's name); prints a
                           line per file and skips a variant that fails
  -o, --output OUT.ptx     write the PTX here instead of standard output";

fn main() -> Result<()> {
    let mut ctx = Context {
        print_phases: phobos_base::env::flag("PHOBOS_PRINT_PHASES"),
        ..Context::default()
    };
    let mut file: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut pins: HashMap<String, i64> = HashMap::new();
    let mut all_variants = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().with_context(|| format!("{arg} needs a value"));
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
            "-o" | "--output" => out = Some(value()?.into()),
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

    let src = std::fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
    let kernels = phobos_lang::parse(&src)?;

    if !all_variants {
        // Each kernel seeds its own first choices, so only the pins' names
        // need checking, not whether the kernels' spaces agree.
        let declared = kernels.iter().flat_map(phobos_lang::search_space).collect();
        phobos_lang::pin(declared, &pins)?;
        let ptx = compile(&ctx, &kernels, pins.into_iter().collect())?;
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
    for variant in &variants {
        let path = variant_path(&stem, variant);
        let settings = variant
            .iter()
            .map(|(n, v)| format!("{n}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        let ptx = match compile(&ctx, &kernels, variant.iter().cloned().collect()) {
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
    if failed == variants.len() {
        bail!("none of the {failed} variants compiled");
    }
    Ok(())
}

fn compile(
    ctx: &Context,
    kernels: &[Kernel],
    shape_overrides: HashMap<String, i64>,
) -> Result<String> {
    let mut ctx = Context {
        shape_overrides,
        ..ctx.clone()
    };
    // Widen as the runtime does, so a kernel compiles here exactly when it
    // compiles there.
    if ctx.index_bitwidth < 64 && phobos_lang::needs_wide_index(&ctx, kernels) {
        ctx.index_bitwidth = 64;
    }
    phobos_mlir::gen_ptx(&ctx, |base, context, module| {
        phobos_lang::assert_pipelined(&phobos_lang::codegen::emit(base, kernels, context, module)?.pipeline_failures)
    })
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
    fn parse_setting_takes_name_equals_integer() {
        assert_eq!(parse_setting("TILE=64").unwrap(), ("TILE".to_string(), 64));
        assert!(parse_setting("TILE").is_err());
        assert!(parse_setting("TILE=x").is_err());
    }
}
