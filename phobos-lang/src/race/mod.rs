// A shared-memory race check on the host. It runs the MLIR a kernel was
// compiled to, so it sees the barriers exactly as emitted, elided ones
// included, for every thread of one CTA at a time, and reports two threads
// touching one byte between two barriers with at least one of them writing.
//
// It computes addresses, never data. Tensor extents and integer scalars
// take the values in [`Config`]; a load yields nothing, so an address built
// from loaded data covers its whole view (reported as a possible race), and
// a loop on loaded data ends the run as untraceable. A wait on loaded data,
// a flag or a grid barrier, is taken to end after one pass.

mod arith;
mod control;
mod exec;
mod memory;
mod program;
mod report;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use anyhow::{Context as _, Result};
use melior::ir::{BlockLike, Location, Module, RegionLike, operation::OperationLike};
use phobos_mlir::context;

use exec::{Exec, Inputs, Stop};
use memory::Level;
use program::{Program, Ty, parse_ty};

pub use report::diagnostics;

/// What a run assumes about the launch.
#[derive(Clone, Debug)]
pub struct Config {
    /// Every dynamic tensor extent.
    pub extent: i64,
    /// Every integer scalar parameter.
    pub scalar: i64,
    /// The CTAs to run, each on its own.
    pub blocks: Vec<[i64; 3]>,
    pub grid: [i64; 3],
    /// The size of dynamic shared memory, for a module that does not say.
    pub dynamic_bytes: i64,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            extent: 256,
            scalar: 4,
            blocks: vec![[0, 0, 0], [1, 1, 0]],
            grid: [2, 2, 1],
            dynamic_bytes: 100 * 1024,
        }
    }
}

/// Where an op came from, as its MLIR location says: lines and columns
/// from 1, the end exclusive.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Loc {
    pub file: String,
    pub line: u32,
    pub col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

/// One side of a race: the thread and the op.
#[derive(Clone, Debug)]
pub struct Site {
    pub thread: u32,
    pub write: bool,
    pub op: String,
    pub loc: Option<Loc>,
}

#[derive(Clone, Debug)]
pub struct Race {
    pub first: Site,
    pub second: Site,
    pub buffer: String,
    pub byte: i64,
    /// How many byte and thread pairs met at these two ops.
    pub count: usize,
}

/// One CTA of one kernel.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub kernel: String,
    pub threads: i64,
    pub block: [i64; 3],
    /// Threads of different warps.
    pub races: Vec<Race>,
    /// Lanes of one warp with no warp-wide op between them.
    pub warp_races: Vec<Race>,
    /// One side's address depended on data, so it covered its whole view.
    pub possible: Vec<Race>,
    /// Barriers some running threads did not reach.
    pub divergent: Vec<Divergent>,
    pub out_of_bounds: Vec<OutOfBounds>,
    /// Why the run ended early, if it did.
    pub stopped: Option<String>,
    /// The op it stopped at, for a trap.
    pub stopped_at: Option<Loc>,
    pub barriers: usize,
    pub bytes: u64,
    pub unfollowed: u64,
    /// Loops that waited on data, such as a flag or a grid barrier, taken
    /// to end after one pass.
    pub waits: u64,
}

/// A barrier `reached` of the CTA's threads arrived at.
#[derive(Clone, Debug)]
pub struct Divergent {
    pub op: String,
    pub reached: usize,
    pub loc: Option<Loc>,
}

/// A shared access at `byte` of `buffer`, outside it.
#[derive(Clone, Debug)]
pub struct OutOfBounds {
    pub op: String,
    pub buffer: String,
    pub byte: i64,
    pub loc: Option<Loc>,
}

impl Report {
    /// No race between warps, no divergent barrier, no access outside a
    /// buffer, and a run to the end.
    pub fn clean(&self) -> bool {
        self.races.is_empty()
            && self.divergent.is_empty()
            && self.out_of_bounds.is_empty()
            && self.stopped.is_none()
    }
}

/// Compiles `src` for `base`'s target and checks every kernel in it.
pub fn check_source(
    base: &phobos_base::context::Context,
    src: &str,
    config: &Config,
) -> Result<Vec<Report>> {
    check_kernels(base, &crate::parse(src)?, config)
}

/// Compiles `kernels` for `base`'s target and checks each.
pub fn check_kernels(
    base: &phobos_base::context::Context,
    kernels: &[crate::ast::Kernel],
    config: &Config,
) -> Result<Vec<Report>> {
    let context = context();
    let module = Module::new(Location::unknown(&context));
    let out = crate::codegen::emit(base, kernels, &context, &module)?;
    let mut config = config.clone();
    if let Some(&(_, bytes)) = out.shared.iter().max_by_key(|(_, b)| *b) {
        config.dynamic_bytes = bytes as i64;
    }
    check_module(&module, &config)
}

/// Parses printed MLIR, as the `emit` example writes it, and checks every
/// kernel in it.
pub fn check_text(text: &str, config: &Config) -> Result<Vec<Report>> {
    let context = context();
    let text = reparsable(text);
    let module = Module::parse(&context, &text).context("the MLIR did not parse")?;
    check_module(&module, config)
}

/// `text` with the one form MLIR prints but does not parse back: an async
/// copy group with no tokens, which prints as its bare name, before its
/// location if it has one. The generic form parses.
fn reparsable(text: &str) -> String {
    const BARE: &str = "= nvgpu.device_async_create_group";
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let split = line
            .find(BARE)
            .map(|at| (&line[..at], line[at + BARE.len()..].trim()));
        match split {
            Some((head, rest)) if rest.is_empty() || rest.starts_with("loc(") => {
                out.push_str(head);
                out.push_str(
                    "= \"nvgpu.device_async_create_group\"() : () -> !nvgpu.device.async.token",
                );
                if !rest.is_empty() {
                    out.push(' ');
                    out.push_str(rest);
                }
            }
            _ => out.push_str(line),
        }
        out.push('\n');
    }
    out
}

/// Checks every `gpu.func` of every `gpu.module` in `module`.
pub fn check_module(module: &Module<'_>, config: &Config) -> Result<Vec<Report>> {
    let mut reports = Vec::new();
    let mut next = module.body().first_operation();
    while let Some(gpu) = next {
        next = gpu.next_in_block();
        if gpu.name().as_string_ref().as_str()? != "gpu.module" {
            continue;
        }
        let Some(block) = gpu.region(0)?.first_block() else {
            continue;
        };
        let mut globals = HashMap::new();
        let mut funcs = Vec::new();
        let mut op = block.first_operation();
        while let Some(o) = op {
            op = o.next_in_block();
            match o.name().as_string_ref().as_str()? {
                "memref.global" => {
                    let name = o
                        .attribute("sym_name")
                        .map(|a| a.to_string())
                        .unwrap_or_default();
                    let ty = o
                        .attribute("type")
                        .map(|a| a.to_string())
                        .unwrap_or_default();
                    if let Ty::Mem(m) = parse_ty(&ty) {
                        let bytes =
                            m.shape.iter().map(|d| d.unwrap_or(0)).product::<i64>() * m.elem;
                        globals.insert(name.trim_matches('"').to_string(), bytes);
                    }
                }
                "gpu.func" => funcs.push(o),
                _ => {}
            }
        }
        for func in funcs {
            let program = Program::build(func, globals.clone())?;
            for &block in &config.blocks {
                reports.push(run(&program, config, block));
            }
        }
    }
    Ok(reports)
}

fn run(program: &Program, config: &Config, block: [i64; 3]) -> Report {
    let inputs = Inputs {
        block,
        grid: config.grid,
        extent: config.extent,
        scalar: config.scalar,
    };
    let mut exec = Exec::new(program, inputs, config.dynamic_bytes);
    let (stopped, stopped_at) = match exec.run() {
        Ok(()) => (None, None),
        Err(Stop::Trap(op)) => (
            Some(format!("trapped at {}", program.ops[op].text)),
            program.ops[op].loc.clone(),
        ),
        Err(Stop::Untraceable(why)) => (Some(format!("untraceable: {why}")), None),
    };
    let site = |t: memory::Touch| Site {
        thread: t.thread,
        write: t.write,
        op: program.ops[t.site as usize].text.clone(),
        loc: program.ops[t.site as usize].loc.clone(),
    };
    let mut report = Report {
        kernel: program.name.clone(),
        threads: program.threads,
        block,
        stopped,
        stopped_at,
        barriers: exec.mem.epochs - 1,
        bytes: exec.mem.bytes,
        unfollowed: exec.unfollowed,
        waits: exec.waits,
        divergent: exec
            .divergent
            .iter()
            .map(|&(op, reached)| Divergent {
                op: program.ops[op].text.clone(),
                reached,
                loc: program.ops[op].loc.clone(),
            })
            .collect(),
        out_of_bounds: exec
            .mem
            .out_of_bounds
            .iter()
            .map(|&(t, b, byte)| OutOfBounds {
                op: program.ops[t.site as usize].text.clone(),
                buffer: exec.mem.names[b].clone(),
                byte,
                loc: program.ops[t.site as usize].loc.clone(),
            })
            .collect(),
        ..Report::default()
    };
    for h in exec.mem.hazards() {
        let race = Race {
            first: site(h.first),
            second: site(h.second),
            buffer: exec.mem.names[h.buffer].clone(),
            byte: h.byte,
            count: h.count,
        };
        match h.level {
            Level::Cta => report.races.push(race),
            Level::Warp => report.warp_races.push(race),
            Level::Possible => report.possible.push(race),
        }
    }
    report
}
