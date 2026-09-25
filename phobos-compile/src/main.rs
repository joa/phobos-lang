// Compiles a Phobos kernel source to PTX. No GPU.
//
//     phobos-compile FILE.ph [--chip sm_86] [--index-bits 64] [-o OUT.ptx]
//
// `--chip` defaults to sm_75, whose PTX every supported card runs; `--index-bits`
// to 32. `PHOBOS_PRINT_PHASES` prints each lowering phase's IR on the way.

use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};

const USAGE: &str = "usage: phobos-compile FILE.ph [--chip CHIP] [--index-bits 32|64] [-o OUT.ptx]";

fn main() -> Result<()> {
    let mut ctx = Context {
        print_phases: phobos_base::env::flag("PHOBOS_PRINT_PHASES"),
        ..Context::default()
    };
    let mut file: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--chip" => ctx.gpu_config = GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(value()?)),
            "--index-bits" => ctx.index_bitwidth = value()?.parse().context("--index-bits")?,
            "-o" | "--output" => out = Some(value()?.into()),
            "--version" | "-V" => {
                let fingerprint = phobos_kernels::COMPILER_FINGERPRINT;
                println!("phobos-compile {} (compiler {fingerprint})", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "-h" | "--help" => {
                println!("{USAGE}");
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
    let ptx = phobos_mlir::gen_ptx(&ctx, |base, context, module| {
        phobos_lang::codegen::emit(base, &kernels, context, module).map(|_| ())
    })?;
    match out {
        Some(out) => std::fs::write(&out, ptx).with_context(|| format!("writing {}", out.display()))?,
        None => println!("{ptx}"),
    }
    Ok(())
}
