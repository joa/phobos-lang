// The emit sweep: every `.ph` under the given directories, emitted under
// every target the codegen distinguishes, written beside a footprint table.
//
//     cargo run -p phobos-lang --example snapshot -- OUT_DIR SRC_DIR...
//
// For each source `name.ph` and each target, `OUT_DIR/name.CHIP.iBITS.mlir`
// holds the printed module, and `OUT_DIR/footprint.txt` one line per kernel
// and target with its shared-memory bytes: the dynamic peak the sideband
// reports, and the static total of its `memref.global` tiles. A source that
// fails to emit under a target gets a `.err` file instead, since a sweep
// that stops at the first failure covers nothing after it.
//
// The index width follows `compile_raw`: a kernel that wants `ldmatrix` is
// emitted at 64 bits whatever the target says, as the real compile does.

use std::{fmt::Write as _, path::Path};

use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{Location, Module, operation::OperationLike},
    utility::register_all_dialects,
};
use phobos_base::context::{Context as BaseContext, GpuConfig, NvidiaGpuConfig};

const CHIPS: &[&str] = &["sm_75", "sm_80"];
const INDEX_BITS: &[u32] = &[32, 64];

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(out) = args.next() else {
        anyhow::bail!("usage: snapshot OUT_DIR SRC_DIR...");
    };
    let out = Path::new(&out);
    std::fs::create_dir_all(out)?;

    let mut sources = Vec::new();
    for dir in args {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "ph") {
                sources.push(path);
            }
        }
    }
    sources.sort();

    let mut footprint = String::new();
    let mut failures = 0;
    for path in &sources {
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(path)?;
        for &chip in CHIPS {
            for &bits in INDEX_BITS {
                let tag = format!("{stem}.{chip}.i{bits}");
                match emit(&src, chip, bits) {
                    Ok((mlir, lines)) => {
                        std::fs::write(out.join(format!("{tag}.mlir")), mlir)?;
                        for (kernel, dynamic, r#static) in lines {
                            writeln!(footprint, "{tag} {kernel} dynamic={dynamic} static={static}")?;
                        }
                    }
                    Err(e) => {
                        failures += 1;
                        std::fs::write(out.join(format!("{tag}.err")), format!("{e:#}\n"))?;
                    }
                }
            }
        }
    }
    std::fs::write(out.join("footprint.txt"), footprint)?;
    println!(
        "{} sources x {} targets, {failures} failures",
        sources.len(),
        CHIPS.len() * INDEX_BITS.len()
    );
    Ok(())
}

type Footprint = Vec<(String, usize, usize)>;

fn emit(src: &str, chip: &str, bits: u32) -> anyhow::Result<(String, Footprint)> {
    let kernels = phobos_lang::parse(src)?;
    let mut base = BaseContext {
        gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)),
        index_bitwidth: bits,
        ..Default::default()
    };
    if kernels.iter().any(phobos_lang::ast::Kernel::wants_ldmatrix) {
        base.index_bitwidth = 64;
    }

    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    let module = Module::new(Location::unknown(&context));

    let out = phobos_lang::codegen::emit(&base, &kernels, &context, &module)?;
    if !module.as_operation().verify() {
        anyhow::bail!("emitted module failed verification");
    }
    let text = module.as_operation().to_string();

    let mut lines = Vec::new();
    for kernel in &kernels {
        let dynamic = out
            .shared
            .iter()
            .find(|(name, _)| *name == kernel.name)
            .map_or(0, |(_, bytes)| *bytes);
        let r#static = static_bytes(&text, &kernel.name);
        lines.push((kernel.name.clone(), dynamic, r#static));
    }
    Ok((text, lines))
}

/// Sum of the kernel's `memref.global` buffers, read off the printed module:
/// `memref.global "private" @__NAME_tileN : memref<AxBxT, 3> {...}` per tile
/// under the pool, or the one `@__NAME_shared : memref<Nxi8, 3>` under the plan.
fn static_bytes(text: &str, kernel: &str) -> usize {
    let prefix = format!("memref.global \"private\" @__{kernel}_");
    text.lines()
        .filter(|l| l.trim_start().starts_with(&prefix))
        .filter_map(|l| {
            let ty = l.split(" : memref<").nth(1)?;
            let inner = ty.split(['>', ',']).next()?;
            let mut elems = 1usize;
            let mut width = 0usize;
            for part in inner.split('x') {
                match part.parse::<usize>() {
                    Ok(n) => elems *= n,
                    Err(_) => width = elem_bytes(part)?,
                }
            }
            Some(elems * width)
        })
        .sum()
}

fn elem_bytes(ty: &str) -> Option<usize> {
    Some(match ty {
        "i8" => 1,
        "f16" | "bf16" | "i16" => 2,
        "f32" | "i32" => 4,
        "f64" | "i64" | "index" => 8,
        "i1" => 1,
        _ => return None,
    })
}
