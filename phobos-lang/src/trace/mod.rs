// The kernel visualizer's exporter: one kernel compiled for one chip, and
// what the compiler decided about it, as `website/src/lib/viz/trace.ts`
// declares a trace. A pure function of its request: it reads no file and no
// environment, so the same call can later run in a browser.

mod eval;
pub mod model;
mod place;
mod view;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context as _, Result, anyhow, bail};
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module, operation::OperationLike},
    utility::register_all_dialects,
};
use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};
use sha2::{Digest, Sha256};

use crate::ast::{Dim, Type as AstType};
use crate::codegen::Planned;
use crate::ir::OpKind;
use crate::ir::uniform::{self, Kind};
pub use model::{Cta, Origin, Trace, Variant};
use model::{Role, Uniformity};

/// Grids up to this many CTAs are traced in full.
const FULL_GRID: usize = 64;

/// CTAs traced in full in a larger grid; every other one keeps its boxes.
const FULL_CTAS: usize = 4;

/// What to trace.
#[derive(Clone, Debug)]
pub struct Request {
    /// The trace's name, its file stem before the chip.
    pub name: String,
    pub origin: Origin,
    /// Repo-relative: the `.ph` file, or the docs page holding the fence.
    pub path: String,
    pub fence: Option<String>,
    /// The source as written, before a variant.
    pub text: String,
    pub variant: Option<Variant>,
    pub chip: String,
    /// `@autotune` choices; the kernel's first choice for any left out.
    pub params: BTreeMap<String, i64>,
    /// Runtime extents of the tensors' symbolic dimensions.
    pub dims: BTreeMap<String, i64>,
    pub grid: Cta,
    pub commit: String,
}

/// The hash a trace carries of its source: SHA-256 over the text with CRLF
/// as LF, then, for a variant, the variant as compact JSON.
pub fn source_sha256(text: &str, variant: Option<&Variant>) -> String {
    let mut h = Sha256::new();
    h.update(text.replace("\r\n", "\n").as_bytes());
    if let Some(v) = variant {
        h.update(
            serde_json::to_string(v)
                .expect("a variant serializes")
                .as_bytes(),
        );
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// The source with the variant applied: an attribute line added before the
/// `kernel` line, or every line holding that attribute dropped.
pub fn apply_variant(text: &str, variant: &Variant) -> Result<String> {
    let mut lines: Vec<&str> = text.lines().collect();
    if let Some(drop) = &variant.drop {
        let n = lines.len();
        lines.retain(|l| !is_attr(l, drop));
        if lines.len() == n {
            bail!("the variant drops {drop}, which the source does not carry");
        }
    }
    if let Some(add) = &variant.add {
        if lines.iter().any(|l| is_attr(l, add)) {
            bail!("the variant adds {add}, which the source already carries");
        }
        let at = lines
            .iter()
            .position(|l| l.trim_start().starts_with("kernel "))
            .ok_or_else(|| anyhow!("no `kernel` line to add {add} before"))?;
        lines.insert(at, add);
    }
    Ok(lines.join("\n") + "\n")
}

/// Whether `line` is the attribute `attr` (written with its `@`).
fn is_attr(line: &str, attr: &str) -> bool {
    let name = attr.split('(').next().unwrap_or(attr);
    line.trim_start().strip_prefix(name).is_some_and(|rest| {
        rest.is_empty() || rest.starts_with('(') || rest.starts_with(char::is_whitespace)
    })
}

/// The lines the code panel hides: `@cluster`, which only the cluster
/// scheduler reads.
fn hidden_lines(text: &str) -> Vec<u32> {
    (1..)
        .zip(text.lines())
        .filter(|(_, l)| is_attr(l, "@cluster"))
        .map(|(n, _)| n)
        .collect()
}

/// Compiles the request's kernel and traces it.
pub fn trace(req: &Request) -> Result<Trace> {
    let normalized = req.text.replace("\r\n", "\n");
    let text = match &req.variant {
        Some(v) => apply_variant(&normalized, v)?,
        None => normalized,
    };
    let kernels = crate::parse(&text)?;
    let kernel = kernels
        .first()
        .ok_or_else(|| anyhow!("the source holds no kernel"))?;
    let space: Vec<String> = crate::search_space(kernel)
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    if let Some(name) = req.params.keys().find(|n| !space.contains(n)) {
        bail!("`{name}` is no @autotune parameter of `{}`", kernel.name);
    }

    let mut base = Context {
        gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(req.chip.clone())),
        shape_overrides: req.params.iter().map(|(k, &v)| (k.clone(), v)).collect(),
        ..Context::default()
    };
    if crate::needs_wide_index(&base, &kernels) {
        base.index_bitwidth = 64;
    }

    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    let module = Module::new(Location::unknown(&context));

    let mut out: Option<Result<Trace>> = None;
    crate::codegen::emit_observed(&base, &kernels[..1], &context, &module, &mut |p| {
        out = Some(assemble(req, &text, &base, p));
    })?;
    if !module.as_operation().verify() {
        bail!("the emitted module failed verification");
    }
    out.ok_or_else(|| anyhow!("the kernel was never planned"))?
}

fn assemble(req: &Request, text: &str, base: &Context, p: &Planned<'_>) -> Result<Trace> {
    let ir = p.ir;
    let shape_env = &ir.kernel.shape_env;
    let cta_threads = ir.kernel.cta_threads;
    let warp_lanes = crate::codegen::target::build_target(base).lanes;

    // The tensor parameters, their dims resolved.
    let mut tensors = HashMap::new();
    let mut order = Vec::new();
    for (param, &arg) in p.kernel.params.iter().zip(ir.args(ir.entry())) {
        let AstType::Tensor(elem, dims) = &param.ty else {
            continue;
        };
        let shape = dims
            .iter()
            .map(|d| match d {
                Dim::Int(n) => Ok(*n),
                Dim::Sym(s) => req
                    .dims
                    .get(s)
                    .or_else(|| shape_env.get(s))
                    .copied()
                    .ok_or_else(|| anyhow!("dimension `{s}` of `{}` is not given", param.name)),
            })
            .collect::<Result<Vec<_>>>()?;
        tensors.insert(arg, (param.name.clone(), shape.clone()));
        order.push((
            param.name.clone(),
            crate::ir::Scalar::from_ast(*elem),
            shape,
        ));
    }

    let place = place::Placement::new(p, cta_threads);
    let k = eval::Kernel {
        ir,
        place: &place,
        tensors,
        grid: req.grid,
        lanes: usize::try_from(cta_threads.min(warp_lanes)).unwrap_or(1),
        warp_lanes,
        collectives: eval::collectives(ir),
    };

    let all: Vec<Cta> = (0..req.grid[2])
        .flat_map(|z| (0..req.grid[1]).flat_map(move |y| (0..req.grid[0]).map(move |x| [x, y, z])))
        .collect();
    if all.is_empty() {
        bail!("the grid {:?} has no CTA", req.grid);
    }
    let full = if all.len() <= FULL_GRID {
        all.len()
    } else {
        FULL_CTAS
    };
    let mut steps = Vec::new();
    let mut summaries = Vec::new();
    let mut roles: BTreeMap<String, (bool, bool)> = BTreeMap::new();
    let mut loops = HashMap::new();
    for (i, &cta) in all.iter().enumerate() {
        let ran = k.run(cta).with_context(|| format!("CTA {cta:?}"))?;
        if i == 0 {
            loops = ran.loops;
        }
        let ran = ran.steps;
        for s in &ran {
            for (accesses, write) in [(&s.reads, false), (&s.writes, true)] {
                for t in accesses.iter().filter_map(|a| a.tensor.as_ref()) {
                    let role = roles.entry(t.clone()).or_default();
                    if write { role.1 = true } else { role.0 = true }
                }
            }
        }
        if i < full {
            steps.extend(ran);
        } else {
            summaries.extend(ran.into_iter().filter_map(summarize));
        }
    }

    let kinds = uniform::analyze(ir);
    let ops = ir
        .all_ops()
        .into_iter()
        .map(|op| {
            let kind = match ir.results(op) {
                [] => kinds.control(ir.parent_block(op)),
                rs => rs.iter().map(|&r| kinds.of(r)).max().unwrap_or_default(),
            };
            model::Op {
                id: op.index(),
                kind: kind_name(ir.kind(op)),
                span: p.report.spans.get(&op).map(|s| model::Span {
                    line: s.line,
                    col: s.col,
                    end_line: s.end_line,
                    end_col: s.end_col,
                }),
                ir: ir.print_op(op),
                uniformity: Some(match kind {
                    Kind::Uniform => Uniformity::Uniform,
                    Kind::Warp => Uniformity::Warp,
                    Kind::Varying => Uniformity::Varying,
                }),
                instr: place.instr(op),
                parent: ir.parent_op(ir.parent_block(op)).map(|p| p.index()),
                loop_info: loops.get(&op).map(|&trips| model::LoopInfo {
                    var: match ir.kind(op) {
                        OpKind::For(_) => ir
                            .blocks_of(op)
                            .first()
                            .and_then(|&b| ir.name(ir.args(b)[0]))
                            .map(str::to_string),
                        _ => None,
                    },
                    trips,
                }),
            }
        })
        .collect();

    let tensors = order
        .into_iter()
        .map(|(name, elem, shape)| {
            let role = match roles.get(&name).copied().unwrap_or_default() {
                (_, false) => Role::In,
                (false, true) => Role::Out,
                (true, true) => Role::Inout,
            };
            model::Tensor {
                elem_bytes: elem.bytes().unwrap_or(4),
                elem: elem.mlir_name().to_string(),
                strides: dense_strides(&shape),
                shape,
                role,
                name,
            }
        })
        .collect();

    Ok(Trace {
        schema: 1,
        name: req.name.clone(),
        kernel: p.kernel.name.clone(),
        source: model::Source {
            origin: req.origin,
            path: req.path.clone(),
            fence: req.fence.clone(),
            text: text.to_string(),
            sha256: source_sha256(&req.text, req.variant.as_ref()),
            hidden_lines: hidden_lines(text),
            variant: req.variant.clone(),
        },
        compiler: model::Compiler {
            commit: req.commit.clone(),
            chip: req.chip.clone(),
            index_bits: base.index_bitwidth,
        },
        params: shape_env.clone(),
        dims: req.dims.clone(),
        launch: model::Launch {
            grid: req.grid,
            threads: cta_threads,
            lanes: warp_lanes,
            warps: cta_threads / warp_lanes.max(1),
        },
        tensors,
        ops,
        buffers: place.buffers.clone(),
        barriers: place.barriers.clone(),
        shared_peak_bytes: Some(place.peak_bytes),
        ctas: all[..full].to_vec(),
        steps,
        summaries,
    })
}

/// A step reduced to the tensor regions it touched, for a CTA not traced in
/// full.
fn summarize(s: model::Step) -> Option<model::Summary> {
    let keep = |v: Vec<model::Access>| -> Vec<model::Access> {
        v.into_iter().filter(|a| a.tensor.is_some()).collect()
    };
    let (reads, writes) = (keep(s.reads), keep(s.writes));
    (!reads.is_empty() || !writes.is_empty()).then_some(model::Summary {
        cta: s.cta,
        op: s.op,
        iter: s.iter,
        reads,
        writes,
    })
}

/// Row-major strides, in elements, of a dense tensor of `shape`.
fn dense_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// A short stable name for an op's kind.
fn kind_name(kind: &OpKind) -> &'static str {
    match kind {
        OpKind::Const(_) => "const",
        OpKind::Unary(_) => "unary",
        OpKind::Binary(_) => "binary",
        OpKind::IndexOp(_) => "index",
        OpKind::Cast(_) => "cast",
        OpKind::IndexCast(_) => "index_cast",
        OpKind::ProgramId(_) => "program_id",
        OpKind::GridDim(_) => "grid_dim",
        OpKind::Thread(_) => "thread",
        OpKind::Shuffle { .. } => "shuffle",
        OpKind::Converged(_) => "collective",
        OpKind::LaneLoad { .. } => "lane_load",
        OpKind::Dot4 { .. } | OpKind::Dp4a => "dp4a",
        OpKind::LaneStore { .. } => "lane_store",
        OpKind::F16Bits => "f16bits",
        OpKind::Word => "word",
        OpKind::BytePerm => "byteperm",
        OpKind::PopCount => "popcount",
        OpKind::Exchange => "exchange",
        OpKind::Math(_) => "math",
        OpKind::Select => "select",
        OpKind::Dim(_) => "dim",
        OpKind::Load => "load",
        OpKind::Store => "store",
        OpKind::AtomicAdd => "atomic_add",
        OpKind::AtomicCas => "atomic_cas",
        OpKind::Trap => "trap",
        OpKind::RegsFill => "regs_fill",
        OpKind::RegsFused(_) => "regs_fused",
        OpKind::RegsStore => "regs_store",
        OpKind::RegsReduce(_) => "regs_reduce",
        OpKind::LoadAcquire => "load_acquire",
        OpKind::StoreRelease => "store_release",
        OpKind::For(_) => "for",
        OpKind::While => "while",
        OpKind::If => "if",
        OpKind::Yield => "yield",
        OpKind::Condition => "condition",
        OpKind::Barrier => "barrier",
        OpKind::GridBarrier => "grid_barrier",
        OpKind::AssumeAlign => "assume_align",
        OpKind::Alloc => "alloc",
        OpKind::Slice(_) => "slice",
        OpKind::Flat => "flat",
        OpKind::Stage(_) => "stage",
        OpKind::Materialize => "materialize",
        OpKind::HoistStage => "hoist_stage",
        OpKind::Copy { .. } => "copy",
        OpKind::Convert => "convert",
        OpKind::Fill => "fill",
        OpKind::Map(_) => "map",
        OpKind::MapInto(_) => "map_into",
        OpKind::Fused(_) => "fused",
        OpKind::Chain(_) => "chain",
        OpKind::ScaledAdd => "scaled_add",
        OpKind::Reduce(_) => "reduce",
        OpKind::ReduceFused { .. } => "reduce_fused",
        OpKind::Dot { .. } => "dot",
        OpKind::DotInto { .. } => "dot_into",
        OpKind::DotFused { .. } => "dot_fused",
        OpKind::Accumulate => "accumulate",
        OpKind::Intrinsic(_) => "intrinsic",
        OpKind::IntrinsicInto(_) => "intrinsic_into",
        OpKind::FragInit(_) => "frag_init",
        OpKind::FragScale(_) => "frag_scale",
        OpKind::FragDot => "frag_dot",
        OpKind::FragStore => "frag_store",
        OpKind::GemmInit => "gemm_init",
        OpKind::GemmDot => "gemm_dot",
        OpKind::GemmStore { .. } => "gemm_store",
    }
}
