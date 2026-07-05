use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{Location, Module, operation::OperationLike},
    utility::register_all_dialects,
};

const MATMUL: &str = "\
@autotune(TILE_M in [64, 128], TILE_N in [64, 128], TILE_K in [16, 32])
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
    let pm = program_id(0)
    let pn = program_id(1)
    var acc: tile<f32>[TILE_M, TILE_N] = 0.0
    for kt in range(0, K, TILE_K) {
        let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
        let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
        acc += dot(a, b)
    }
    C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}
";

fn main() -> anyhow::Result<()> {
    let src = match std::env::args().nth(1) {
        Some(path) => std::fs::read_to_string(path)?,
        None => MATMUL.to_string(),
    };

    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let module = Module::new(Location::unknown(&context));
    let kernels = phobos_lang::parse(&src)?;
    let base = phobos_base::context::Context::default();
    phobos_lang::codegen::emit(&base, &kernels, &context, &module)?;

    if !module.as_operation().verify() {
        anyhow::bail!("emitted module failed verification");
    }
    println!("{}", module.as_operation());
    Ok(())
}
