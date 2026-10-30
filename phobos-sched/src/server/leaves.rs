// Compiling a cluster program's leaf kernels for the nodes that run them.

use anyhow::{Result, bail};
use sha2::{Digest, Sha256};

use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig, best_supported_chip};
use phobos_base::phinfo;

/// The context leaves compile in for nodes reporting `chips`.
///
/// PTX for an older chip JITs forward, so the oldest node's chip serves them
/// all. A node that reports none, as a pod predating the report does, is
/// left out; with no chip known the default stands. Indices stay 32-bit: the
/// pod packs every size, stride and offset as a 32-bit word.
pub(super) fn leaf_context<'a>(chips: impl IntoIterator<Item = &'a str>) -> Context {
    let oldest = chips
        .into_iter()
        .filter(|chip| !chip.is_empty())
        .map(|chip| GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)).compute_capability())
        .filter(|&cc| cc > 0)
        .min();
    let mut ctx = Context::default();
    match oldest {
        Some(cc) => ctx.gpu_config = GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(best_supported_chip(cc))),
        None => phinfo!("sched: no node reported its chip, compiling leaves for {}", ctx.gpu_config.chip()),
    }
    ctx.index_bitwidth = 32;
    ctx
}

/// Each leaf's PTX for `ctx`, as (hash, entry name, ptx) by kernel id.
///
/// A leaf that needs 64-bit indices on the chip, or asks for them with
/// `@index_bits(64)`, is refused: compiled at 32 bits it would be wrong, and
/// the pod cannot launch it wider.
pub(super) fn compile_leaves(program: &phobos_cluster::ir::ClusterProgram, ctx: &Context) -> Result<Vec<(String, String, String)>> {
    program
        .leaves
        .iter()
        .map(|l| {
            let kernel = std::slice::from_ref(&l.kernel);
            if phobos_lang::index_bits(ctx, kernel)? > 32 {
                bail!(
                    "leaf '{}' needs 64-bit indices on {}, which the pod's 32-bit launch ABI does not carry",
                    l.kernel.name,
                    ctx.gpu_config.chip()
                );
            }
            let ptx = phobos_mlir::gen_ptx(ctx, |b, c, m| phobos_lang::codegen::emit(b, kernel, c, m).map(|_| ()))?;
            let hash = format!("{:x}", Sha256::digest(ptx.as_bytes()));
            Ok((hash, l.kernel.name.clone(), ptx)) // the name is the entry point
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{compile_leaves, leaf_context};

    #[test]
    fn a_leaf_that_asks_for_64_bit_indices_is_refused() {
        let src = r#"
@index_bits(64)
@cluster(TILE_M in [4096, 16384], TILE_N in [4096, 16384], TILE_K in [4096, 16384])
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
let pm = ctx.block.x
let pn = ctx.block.y
var acc: tile<f32>[TILE_M, TILE_N] = 0.0
for kt in range(0, K, TILE_K) {
    acc += dot(A[pm * TILE_M :+ TILE_M, kt :+ TILE_K], B[kt :+ TILE_K, pn * TILE_N :+ TILE_N])
}
C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}"#;
        let kernel = phobos_lang::parse(src).unwrap().remove(0);
        let program = phobos_cluster::compile(&kernel).unwrap();
        let err = compile_leaves(&program, &leaf_context([])).unwrap_err().to_string();
        assert!(err.contains("needs 64-bit indices"), "{err}");
    }

    #[test]
    fn leaves_compile_for_the_oldest_node_at_32_bits() {
        let ctx = leaf_context(["sm_89", "sm_86", "", "sm_120"]);
        assert_eq!(ctx.gpu_config.chip(), "sm_86");
        assert_eq!(ctx.index_bitwidth, 32);
        // A chip past every supported one gets the newest supported.
        assert_eq!(leaf_context(["sm_87"]).gpu_config.chip(), "sm_86");
        // Nothing reported: the default.
        assert_eq!(leaf_context(["", ""]).gpu_config.chip(), "sm_75");
    }
}
