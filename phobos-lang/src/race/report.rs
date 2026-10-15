// Race reports as diagnostics on the kernel's source, through the op
// locations codegen gives every op it emits for a source span.

use std::collections::HashSet;

use super::{Loc, Race, Report};
use crate::ast::Kernel;
use crate::diag::{Diagnostic, Diagnostics, SourceFile, Span};

/// What fails a check in `reports`, races between warps, barriers not every
/// thread reaches and accesses outside a buffer, as errors on the source of
/// the kernel each names, and a run that stopped early as a warning. A
/// finding seen in several CTAs is reported once. None when there is
/// nothing to report.
pub fn diagnostics(kernels: &[Kernel], reports: &[Report]) -> Option<Diagnostics> {
    let mut out: Option<Diagnostics> = None;
    let mut seen = HashSet::new();
    for r in reports {
        let Some(kernel) = kernels.iter().find(|k| k.name == r.kernel) else {
            continue;
        };
        let file = &kernel.source;
        let span = |loc: &Option<Loc>| loc.as_ref().map(|l| to_span(file, l));
        let mut new = |kind: u8, a: Option<Span>, b: Option<Span>| {
            seen.insert((r.kernel.clone(), kind, a, b))
        };
        let mut list = Vec::new();
        for race in &r.races {
            let (first, second) = (span(&race.first.loc), span(&race.second.loc));
            if new(0, first, second) {
                list.push(race_error(file, kernel, r, race, first, second));
            }
        }
        for d in &r.divergent {
            if new(1, span(&d.loc), None) {
                list.push(
                    Diagnostic::error("not every thread of the CTA reaches a barrier")
                        .primary(
                            span(&d.loc).unwrap_or(kernel.name_span),
                            format!("{} of {} threads reach the barrier here", d.reached, r.threads),
                        )
                        .note("every thread of a CTA has to reach each barrier, or the CTA hangs"),
                );
            }
        }
        for o in &r.out_of_bounds {
            if new(2, span(&o.loc), None) {
                list.push(
                    Diagnostic::error("a shared memory access outside its buffer")
                        .primary(
                            span(&o.loc).unwrap_or(kernel.name_span),
                            format!("reaches byte {} of its buffer", o.byte),
                        )
                        .note(seen_in(r)),
                );
            }
        }
        if let Some(why) = &r.stopped
            && new(3, span(&r.stopped_at), None)
        {
            let at = span(&r.stopped_at);
            let label = if at.is_some() { "a thread traps here" } else { "" };
            let mut d = Diagnostic::warning(format!("the race check stopped early in kernel `{}`", r.kernel))
                .primary(at.unwrap_or(kernel.name_span), label)
                .note("what runs after it is not checked");
            if at.is_none() {
                d = d.note(why.clone());
            }
            list.push(d);
        }
        if list.is_empty() {
            continue;
        }
        let found = Diagnostics::new(file.clone(), list);
        match &mut out {
            Some(all) => all.extend(found),
            None => out = Some(found),
        }
    }
    out
}

fn race_error(
    file: &SourceFile,
    kernel: &Kernel,
    r: &Report,
    race: &Race,
    first: Option<Span>,
    second: Option<Span>,
) -> Diagnostic {
    let verb = |write: bool| if write { "writes" } else { "reads" };
    let message = if race.first.write && race.second.write {
        "threads of two warps write the same shared memory with no `barrier()` between"
    } else {
        "a thread reads shared memory another warp writes, with no `barrier()` between"
    };
    let first_text = format!("thread {} {} here", race.first.thread, verb(race.first.write));
    let second_text = format!("thread {} {} here", race.second.thread, verb(race.second.write));
    let d = Diagnostic::error(message);
    let d = match (first, second) {
        (Some(a), Some(b)) if a == b => d.primary(
            b,
            format!(
                "thread {} {} and thread {} {} here",
                race.first.thread,
                verb(race.first.write),
                race.second.thread,
                verb(race.second.write)
            ),
        ),
        (Some(a), Some(b)) => d.primary(b, second_text).secondary(a, first_text),
        (None, Some(b)) => d.primary(b, second_text),
        (Some(a), None) => d.primary(a, first_text),
        (None, None) => d.primary(kernel.name_span, "in this kernel"),
    };
    let d = d.note(seen_in(r));
    match (first, second) {
        (Some(a), Some(b)) if file.line_of(a.lo) < file.line_of(b.lo) => {
            let line = file.line_of(b.lo);
            let start = file.line_start(line);
            let indent: String = file
                .line_text(line)
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .collect();
            d.suggest(
                "a barrier between the two orders them",
                Span::new(start, start),
                format!("{indent}barrier()\n"),
            )
        }
        _ => d.help(
            "put a `barrier()` after the first access and before the second, \
             which in a loop may be the next turn's",
        ),
    }
}

fn seen_in(r: &Report) -> String {
    format!("seen in CTA {:?} of kernel `{}`", r.block, r.kernel)
}

/// The span a location names; columns count characters, as the span's
/// lines and columns do.
fn to_span(file: &SourceFile, loc: &Loc) -> Span {
    Span::new(file.offset(loc.line, loc.col), file.offset(loc.end_line, loc.end_col))
}
