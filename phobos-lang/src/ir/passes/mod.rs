// The rewrites between the build and the verifier. The build turns source
// into ops without deciding anything that needs the whole kernel; each pass
// here reads use lists over the built graph and rewrites it in place, in the
// order `run` lists them. The lowering's recorder, planner and barrier
// elision run after all of them.

mod carry;
mod epilogue;
mod frag;
mod gridbar;
mod materialize;
mod owner;
mod rows;
mod scratch;
mod swizzle;

use anyhow::{Result, anyhow};

use super::{Ir, OpKind};
use super::build::{Report, Target, locate};
use crate::ast::Kernel;

/// Where `@name` is written, or the kernel's name without it.
fn attr_span(kernel: &Kernel, name: &str) -> crate::diag::Span {
    kernel.attr(name).map_or(kernel.name_span, |a| a.span)
}

/// Runs every pass over `ir`, in order. `kernel` only places an error at the
/// attribute it concerns; no pass reads the source.
///
/// An op a pass makes stands for the one it replaces: it takes that op's
/// span (`Ir::set_span`) and its statement in `report.spans`.
pub fn run(ir: &mut Ir, report: &mut Report, target: &Target, kernel: &Kernel) -> Result<()> {
    // A persistent grid's device memory first: the barriers its blocks
    // need, and where its scratch lives. A held scratch becomes a shared
    // tile like any other for the passes after.
    let mut held = std::collections::BTreeSet::new();
    if let Some(blocks) = ir.kernel.persistent_blocks {
        let grid = super::uniform::analyze_blocks(ir);
        let owners = owner::Owners::new(ir, blocks, &grid);
        held = scratch::held(ir, &owners, &kernel.named_by("global"));
        scratch::check_shared(ir, &held, &kernel.named_by("shared"))
            .map_err(|e| locate(&kernel.source, ir, e, attr_span(kernel, "shared")))?;
        let placement = gridbar::plan(ir, &owners, &grid, &held);
        let written = ir.all_ops().into_iter().filter(|&op| matches!(ir.kind(op), OpKind::GridBarrier)).count();
        for (op, later, earlier) in &placement.before {
            phobos_base::phdebug!(
                "kernel `{}`: a grid barrier before {} orders its {} after {}",
                ir.kernel.name,
                ir.print_op(*op).trim(),
                ir.print_op(later.op).trim(),
                ir.print_op(earlier.op).trim()
            );
        }
        let total = written + placement.before.len();
        let at = attr_span(kernel, "grid_barriers");
        match kernel.grid_barriers().map_err(|e| locate(&kernel.source, ir, anyhow!(e), at))? {
            Some(n) if n != total => {
                let e = anyhow!(
                    "`@grid_barriers({n})`: the kernel has {total}, {written} written and {} placed where blocks share data",
                    placement.before.len()
                );
                return Err(locate(&kernel.source, ir, e, at));
            }
            _ => {}
        }
        gridbar::insert(ir, &placement, &mut report.spans);
    }
    scratch::place(ir, &held);

    // Fragment accumulators first: the epilogue fusion would otherwise take
    // a scaled dot's product into a shared tile.
    while let Some(acc) = frag::find(ir, target).into_iter().next() {
        carry::rewrite(ir, acc, report);
    }
    rows::promote(ir);
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

#[cfg(test)]
mod tests;
