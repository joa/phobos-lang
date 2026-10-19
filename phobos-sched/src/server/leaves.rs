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
/// A leaf that needs 64-bit indices on the chip is refused: compiled at 32
/// bits it would be wrong, and the pod cannot launch it wider.
pub(super) fn compile_leaves(program: &phobos_cluster::ir::ClusterProgram, ctx: &Context) -> Result<Vec<(String, String, String)>> {
    program
        .leaves
        .iter()
        .map(|l| {
            let kernel = std::slice::from_ref(&l.kernel);
            if phobos_lang::needs_wide_index(ctx, kernel) {
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
    use super::leaf_context;

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
