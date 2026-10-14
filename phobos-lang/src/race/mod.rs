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

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use anyhow::{Context as _, Result};
use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{BlockLike, Location, Module, RegionLike, operation::OperationLike},
    utility::register_all_dialects,
};

use exec::{Exec, Inputs, Stop};
use memory::Level;
use program::{Program, Ty, parse_ty};

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

/// One side of a race: the thread and the op.
#[derive(Clone, Debug)]
pub struct Site {
    pub thread: u32,
    pub write: bool,
    pub op: String,
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
    /// Barriers some running threads did not reach: the op, and how many did.
    pub divergent: Vec<(String, usize)>,
    pub out_of_bounds: Vec<String>,
    /// Why the run ended early, if it did.
    pub stopped: Option<String>,
    pub barriers: usize,
    pub bytes: u64,
    pub unfollowed: u64,
    /// Loops that waited on data, such as a flag or a grid barrier, taken
    /// to end after one pass.
    pub waits: u64,
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
    let context = context();
    let module = Module::new(Location::unknown(&context));
    let kernels = crate::parse(src)?;
    let out = crate::codegen::emit(base, &kernels, &context, &module)?;
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
/// copy group with no tokens, which prints as its bare name. The generic
/// form parses.
fn reparsable(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        match line
            .trim_end()
            .strip_suffix("= nvgpu.device_async_create_group")
        {
            Some(head) => {
                out.push_str(head);
                out.push_str(
                    "= \"nvgpu.device_async_create_group\"() : () -> !nvgpu.device.async.token",
                );
            }
            None => out.push_str(line),
        }
        out.push('\n');
    }
    out
}

fn context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context
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
    let stopped = match exec.run() {
        Ok(()) => None,
        Err(Stop::Trap(op)) => Some(format!("trapped at {}", program.ops[op].text)),
        Err(Stop::Untraceable(why)) => Some(format!("untraceable: {why}")),
    };
    let site = |t: memory::Touch| Site {
        thread: t.thread,
        write: t.write,
        op: program.ops[t.site as usize].text.clone(),
    };
    let mut report = Report {
        kernel: program.name.clone(),
        threads: program.threads,
        block,
        stopped,
        barriers: exec.mem.epochs - 1,
        bytes: exec.mem.bytes,
        unfollowed: exec.unfollowed,
        waits: exec.waits,
        divergent: exec
            .divergent
            .iter()
            .map(|&(op, n)| (program.ops[op].text.clone(), n))
            .collect(),
        out_of_bounds: exec
            .mem
            .out_of_bounds
            .iter()
            .map(|(t, b, at)| {
                format!(
                    "{} at byte {at} of {}",
                    program.ops[t.site as usize].text, exec.mem.names[*b]
                )
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
