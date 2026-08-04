use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: ptx <file.ph> [chip] [index-bitwidth]"))?;
    let chip = args.next().unwrap_or_else(|| "sm_75".to_string());
    let default_bitwidth = Context::default().index_bitwidth;
    let index_bitwidth = args.next().map_or(Ok(default_bitwidth), |s| s.parse())?;

    let ctx = Context {
        gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)),
        index_bitwidth,
        print_phases: std::env::var_os("PHOBOS_PRINT_PHASES").is_some(),
        ..Default::default()
    };
    let src = std::fs::read_to_string(path)?;
    let kernels = phobos_lang::parse(&src)?;
    let ptx = phobos_mlir::gen_ptx(&ctx, |base, context, module| {
        phobos_lang::codegen::emit(base, &kernels, context, module).map(|_| ())
    })?;
    println!("{ptx}");
    Ok(())
}
