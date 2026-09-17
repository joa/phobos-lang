// Prints the graph a source builds to, the fastest way to eyeball what the
// build decided: `cargo run -p phobos-lang --example ir [file]`. The same
// `PHOBOS_CHIP` and `PHOBOS_INDEX_BITS` overrides as `emit`.

use phobos_base::context::{GpuConfig, NvidiaGpuConfig};

fn main() -> anyhow::Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        anyhow::bail!("usage: ir FILE.ph");
    };
    let src = std::fs::read_to_string(path)?;
    let mut base = phobos_base::context::Context::default();
    if let Ok(chip) = std::env::var("PHOBOS_CHIP") {
        base.gpu_config = GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip));
    }
    if let Ok(bits) = std::env::var("PHOBOS_INDEX_BITS") {
        base.index_bitwidth = bits.parse()?;
    }
    let target = phobos_lang::codegen::target::build_target(&base);
    for kernel in phobos_lang::parse(&src)? {
        let (ir, report) = phobos_lang::ir::build::build(&base, target, &kernel)?;
        phobos_lang::ir::verify::verify(&ir)?;
        print!("{ir}");
        for decline in report.pipeline_declines {
            println!("// pipeline decline: {decline}");
        }
    }
    Ok(())
}
