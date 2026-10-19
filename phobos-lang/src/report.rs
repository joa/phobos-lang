// What the compiler decided for a `@persistent` kernel's launch, without
// lowering it: the scratch it publishes and the barriers it has.

use anyhow::{Context as _, Result};

use crate::ir::{OpKind, Type};

/// One scratch tensor published to device memory: a parameter the launch
/// supplies, after the declared ones.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchParam {
    pub name: String,
    pub dims: Vec<i64>,
}

/// The launch a kernel takes beyond its declared parameters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LaunchReport {
    /// Published scratch, in parameter order.
    pub scratch: Vec<ScratchParam>,
    /// Whether the barrier word follows the scratch.
    pub barrier_word: bool,
    /// The source line of each grid barrier, written or placed, in order. A
    /// placed one is on the line of the statement it precedes. Lines count
    /// from 1.
    pub grid_barriers: Vec<u32>,
}

/// [`LaunchReport`] for kernel `name` of `source`, from the passes the
/// compile runs, so it matches the PTX `source` compiles to.
pub fn launch_report(base: &phobos_base::context::Context, source: &str, name: &str) -> Result<LaunchReport> {
    launch_report_source(base, &crate::diag::SourceFile::new("", source), name)
}

/// [`launch_report`] of `file`, whose grid barriers are on the lines of the
/// source it was generated from, if it was.
pub fn launch_report_source(
    base: &phobos_base::context::Context,
    file: &std::sync::Arc<crate::diag::SourceFile>,
    name: &str,
) -> Result<LaunchReport> {
    let kernels = crate::parse_source(file)?;
    let kernel = kernels.iter().find(|k| k.name == name).with_context(|| format!("no kernel `{name}`"))?;
    let target = crate::codegen::target::build_target(base);
    let (mut ir, mut report) = crate::ir::build::build(base, target, kernel)?;
    crate::ir::passes::run(&mut ir, &mut report, &target, kernel)?;
    let args = ir.args(ir.entry());
    let word = usize::from(ir.kernel.grid_barrier);
    let scratch = args[kernel.params.len()..args.len() - word]
        .iter()
        .map(|&arg| {
            let Type::Tensor(t) = ir.ty(arg) else {
                anyhow::bail!("a published scratch is not a tensor");
            };
            let dims = t.shape.iter().map(|e| e.fixed().context("a scratch of dynamic shape")).collect::<Result<_>>()?;
            Ok(ScratchParam { name: ir.name(arg).unwrap_or_default().to_string(), dims })
        })
        .collect::<Result<_>>()?;
    let grid_barriers = ir
        .all_ops()
        .into_iter()
        .filter(|&op| matches!(ir.kind(op), OpKind::GridBarrier))
        .map(|op| kernel.source.shown().line_of(kernel.source.locate(ir.span(op)).lo))
        .collect();
    Ok(LaunchReport { scratch, barrier_word: word == 1, grid_barriers })
}
