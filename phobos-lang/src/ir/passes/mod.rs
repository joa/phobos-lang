// The rewrites between the build and the verifier. The build turns source
// into ops without deciding anything that needs the whole kernel; each pass
// here reads use lists over the built graph and rewrites it in place, in the
// order `run` lists them. The lowering's recorder, planner and barrier
// elision run after all of them.

mod carry;
mod epilogue;
mod frag;
mod materialize;
mod swizzle;

use anyhow::{Result, anyhow};

use super::Ir;
use super::build::{Report, Target, locate};
use crate::ast::Kernel;

/// Runs every pass over `ir`, in order. `kernel` only places an error at the
/// attribute it concerns; no pass reads the source.
///
/// An op a pass makes stands for the one it replaces: it takes that op's
/// span (`Ir::set_span`) and its statement in `report.spans`.
pub fn run(ir: &mut Ir, report: &mut Report, target: &Target, kernel: &Kernel) -> Result<()> {
    // Fragment accumulators first: the epilogue fusion would otherwise take
    // a scaled dot's product into a shared tile.
    while let Some(acc) = frag::find(ir, target).into_iter().next() {
        carry::rewrite(ir, acc, report);
    }
    materialize::leftovers(ir, &mut report.spans);

    for (fused, dot) in epilogue::fuse_dot_epilogues(ir) {
        if let Some(&span) = report.spans.get(&dot) {
            report.spans.insert(fused, span);
        }
    }
    report.spans.retain(|&op, _| ir.is_alive(op));
    report.bounds.retain(|&op, _| ir.is_alive(op));

    // The swizzle reads which tiles the tensor-core fragments load, so it
    // runs once those reads are final.
    let at = kernel.attr("swizzle").map_or(kernel.name_span, |a| a.span);
    let mode = kernel
        .swizzle_mode()
        .map_err(|e| locate(&kernel.source, ir, anyhow!(e), at))?;
    swizzle::decide(ir, mode).map_err(|e| locate(&kernel.source, ir, e, at))
}
