use std::collections::{HashMap, HashSet};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use melior::{
    Context,
    dialect::{arith, memref, scf},
    ir::{
        Attribute, Block, BlockLike, Identifier, Location, Module, Operation, Region, RegionLike,
        Type, Value, ValueLike,
        attribute::{DenseI32ArrayAttribute, FloatAttribute, IntegerAttribute, StringAttribute},
        operation::OperationBuilder,
        r#type::{IntegerType, MemRefType},
    },
};

use crate::ast::{BinOp, Kernel, Launch};

mod elemwise;
mod expr;
mod frag;
mod lower;
mod matmul;
mod sync;
pub mod target;
mod tile;
mod util;

pub(crate) use crate::shape::DYN;

/// A row-reduction kind (rowmax / rowsum).
#[derive(Clone, Copy)]
enum Reduce {
    Max,
    Sum,
}

/// Lanes in a warp.
const WARP: i64 = 32;

/// Values in a Q8_0 block, the granularity `qdot_t` scales at.
const Q8_BLOCK: i64 = 32;

const QDOT_LANE: i64 = 16;
const QDOT_STEP: i64 = WARP * QDOT_LANE;

const IMMA_TILE: i64 = 8;
const IMMA_K: i64 = 16;

const QMMA_TILES: i64 = 64;

const WMMA_SMEM_PAD: i64 = 8;


/// What `emit` learned across the module: the dynamic shared-memory sideband,
/// plus `(kernel name, decline reasons)` for every kernel whose `@pipeline`
/// assertion did not hold.
#[derive(Debug)]
pub struct EmitOutput {
    pub shared: Vec<(String, usize)>,
    pub pipeline_failures: Vec<(String, Vec<String>)>,
}

pub fn emit<'c>(
    base: &phobos_base::context::Context,
    kernels: &[Kernel],
    context: &'c Context,
    module: &Module<'c>,
) -> Result<EmitOutput> {
    let loc = Location::unknown(context);
    let gpu_block = Block::new(&[]);
    let mut shared = Vec::new();
    let mut pipeline_failures = Vec::new();

    for kernel in kernels {
        let mut cg = Codegen::new(base, context, kernel)?;
        let func = {
            let target = target::build_target(base);
            let (ir, report) = crate::ir::build::build(base, target, kernel)?;
            #[cfg(debug_assertions)]
            crate::ir::verify::verify(&ir)?;

            // A first emission, thrown away, records every buffer the
            // emitters allocate; the plan places them and the second
            // emission hands out its offsets. See `lower::plan`.
            let mut recorder = Codegen::new(base, context, kernel)?;
            recorder.policy = SharedPolicy::Record;
            recorder.pipeline_declines = report.pipeline_declines.clone();
            drop(recorder.emit_kernel_ir(&ir)?);
            let plan = lower::plan::Plan::compute(&ir, &recorder.trace);
            let elision = lower::membar::Elision::decide(&ir, &recorder.trace, &plan);

            cg.elided = elision.skip;
            cg.policy = SharedPolicy::Replay(plan);
            cg.pipeline_declines = report.pipeline_declines;
            let func = cg.emit_kernel_ir(&ir)?;
            cg.finish_shared_buffer()?;
            func
        };
        if cg.dynamic_shared {
            let peak = match &cg.policy {
                SharedPolicy::Replay(plan) => plan.peak,
                _ => cg.shared_bytes_peak,
            };
            shared.push((kernel.name.clone(), peak as usize));
        }
        if cg.pipeline_assert && !cg.pipelined_any {
            pipeline_failures.push((kernel.name.clone(), cg.pipeline_declines));
        }

        // shared-memory tile buffers are module-level globals
        for global in cg.shared_globals.drain(..) {
            gpu_block.append_operation(global);
        }
        gpu_block.append_operation(func);
    }

    let gpu_region = Region::new();
    gpu_region.append_block(gpu_block);

    let gpu_module = OperationBuilder::new("gpu.module", loc)
        .add_attributes(&[(
            Identifier::new(context, "sym_name"),
            StringAttribute::new(context, "kernels").into(),
        )])
        .add_regions([gpu_region])
        .build()?;

    module.body().append_operation(gpu_module);

    Ok(EmitOutput {
        shared,
        pipeline_failures,
    })
}

/// XOR column swizzle for mma.sync staging buffers.
/// col' = col ^ (((row >> shift) & ((1 << bits) - 1)) << elem_log)
#[derive(Clone, Copy)]
struct Swizzle {
    bits: u32,
    shift: u32,
    elem_log: u32,
}

/// A memref-typed value: a tensor parameter, a slice of one, or a tile buffer.
#[derive(Clone)]
struct MemVal<'c> {
    mem: Value<'c, 'c>,
    elem: Type<'c>,
    /// Static dims, with [`DYN`] for dynamic ones.
    shape: Vec<i64>,
    /// None means contiguous, otherwise the padded stride; see [`Codegen::alloc_tile_padded`].
    row_stride: Option<i64>,
    /// Proven divisor, in elements, of the base offset and of every stride
    /// above the innermost. The row-pitch ABI promises 4 elements on its own
    /// (16 bytes of f32, 8 of f16); `@aligned` can raise it further. Zero
    /// means no offset at all, so any width goes.
    align_div: i64,

    /// XOR column swizzle for ldmatrix staging buffers; every access, staging
    /// store and load alike, must permute the column index. See [`Swizzle`].
    swizzle: Option<Swizzle>,

    /// The memref.global symbol backing a whole tile buffer (None for subviews
    /// and tensor params); what [`Codegen::release`] returns to the pool.
    global: Option<String>,

    /// Whether the bytes are in shared memory rather than global.
    /// Unlike [`Self::global`] this survives a subview.
    shared: bool,

    /// A fresh unnamed temp whose buffer may be released once all reads of it
    /// are emitted; a named buffer is never pooled.
    owned: bool,

    /// Per-dimension bounds mask for a slice that may reach past the source
    /// extent. Some((offset, extent)) lets a load zero-fill and a store skip
    /// where offset + local index >= extent; None means provably in bounds.
    /// Empty for tile buffers and whole slices.
    mask: Vec<Option<(Value<'c, 'c>, Value<'c, 'c>)>>,
}

impl<'c> MemVal<'c> {
    /// Whether any dimension carries a bounds mask (a partial tile).
    fn is_masked(&self) -> bool {
        self.mask.iter().any(Option::is_some)
    }

    /// Whether a `width`-element vector access stays on a legal boundary. A
    /// zero divisor is an unoffset base, which every width clears.
    fn vectorizes(&self, width: i64) -> bool {
        self.align_div % width == 0
    }
}

/// A tile accumulator living in per-lane mma.sync D fragments instead of
/// shared memory (the flash-attention acc): each warp of the wm x wn grid
/// carries fm*fnn*2 vector<2x2xf32> fragments of its [m, n] slice, and the
/// values ride enclosing loops as scf.for iter_args. See codegen/frag.rs.
#[derive(Clone)]
struct FragAcc<'c> {
    frags: Vec<Value<'c, 'c>>,
    m: i64,
    n: i64,
    wm: i64,
    wn: i64,
}

impl FragAcc<'_> {
    /// The warp's fragment-grid extents (fm, fnn).
    fn warp_frags(&self) -> (i64, i64) {
        ((self.m / 16) / self.wm, (self.n / 16) / self.wn)
    }
}

struct Codegen<'c> {
    /// The chip this kernel is emitted for. Every construct the portable
    /// dialects cannot express goes through here, see [`target::Isa`].
    isa: Box<dyn target::Isa>,
    ctx: &'c Context,
    loc: Location<'c>,
    index_t: Type<'c>,
    f16_t: Type<'c>,
    bf16_t: Type<'c>,
    f32_t: Type<'c>,
    f64_t: Type<'c>,
    i8_t: Type<'c>,
    i32_t: Type<'c>,
    i64_t: Type<'c>,
    bool_t: Type<'c>,
    kernel_name: String,
    shared_globals: Vec<Operation<'c>>, // shared-memory tiles (memref.global)
    tile_count: usize,
    // Released tile buffers by (element type, physical shape), reused so a
    // temp doesn't grow the CTA's static shared footprint and cap occupancy.
    tile_pool: HashMap<(String, Vec<i64>), Vec<String>>,
    /// Tile buffers a view still aliases, which never go back to the pool.
    /// See [`Codegen::tile_flat`].
    aliased: HashSet<String>,
    /// Tiles live in one dynamic allocation rather than a global apiece, sized
    /// at launch. See [`Kernel::wants_dynamic_shared`].
    dynamic_shared: bool,
    /// Byte offset of each named tile within that allocation, and how far it
    /// reaches: what the host has to pass at launch.
    tile_offsets: HashMap<String, i64>,
    shared_bytes: i64,
    /// High-water mark of `shared_bytes`: what the host must reserve, since a
    /// barrier-separated kernel's phases can each ask for less than the peak.
    shared_bytes_peak: i64,
    /// Dynamic tiles allocated and not yet released. At zero the next
    /// allocation restarts at offset 0, so a barrier-separated kernel's
    /// phases share the space instead of summing it.
    dynamic_live: i64,
    // Loop-invariant dot operands staged into shared f16 in a loop's preheader,
    // one frame per active for loop: (source view's memref value, staged buffer).
    hoisted_stages: Vec<Vec<(Value<'c, 'c>, MemVal<'c>)>>,
    // Induction variable of the ragged remainder chunk, if any; a slice offset
    // by it is guarded against the runtime dim. None in the trimmed main loop.
    // Induction variables of the enclosing trimmed main loops: their trip
    // count is rounded to whole chunks, so an offset slice needs no mask.
    /// Whether `@pipeline` was written on this kernel. The generic loop path
    /// auto-attempts every eligible loop regardless; this only gates the
    /// fused-GEMM backend's double-buffering (see [`Self::staging_pairs`]).
    pipeline_assert: bool,
    /// Whether some loop in the kernel being emitted did pipeline, through
    /// either mechanism `pipeline_assert` gates.
    pipelined_any: bool,
    /// Why each loop declined, collected while `pipeline_assert` is set so a
    /// failed assertion can say more than that it failed.
    pipeline_declines: Vec<String>,
    tensorcore: bool, // whether to use tensor cores (fp16 inputs)
    mma_sync: bool,   // whether to use mma.sync, disable with @tensorcore(wmma)
    launch: Option<Launch>,
    cta_threads: i64,
    /// What each graph value became, one layer per instantiated body; see
    /// `codegen/lower`.
    lowered: Vec<HashMap<crate::ir::ValueId, lower::Lowered<'c>>>,
    /// Where tile buffers get their bytes; see [`SharedPolicy`].
    policy: SharedPolicy,
    /// What a recording emission saw; see `lower::plan`.
    trace: lower::plan::Trace,
    /// Allocations handed out so far by a replaying emission.
    replay_next: usize,
    /// Ops whose trailing barrier the membar pass elided, with how many
    /// barriers each emits; see `lower::membar`.
    elided: std::collections::BTreeMap<crate::ir::OpId, usize>,
    /// While emitting an elided op: which barrier call to skip, and how
    /// many have been made.
    skip_barrier: Option<usize>,
    barrier_calls: usize,
}

/// How a tile buffer gets its bytes.
enum SharedPolicy {
    /// A free list per shape, reused last-in first-out, and what a
    /// recording emission runs on top of.
    Pool,
    /// The pool, with every allocation and release recorded for the plan.
    Record,
    /// A planned offset per allocation, in recording order, in one byte
    /// buffer the kernel owns.
    Replay(lower::plan::Plan),
}

/// Widens a value's borrow to the context lifetime: every block here is
/// appended to a region the module owns, so the value outlives the borrow.
fn detach<'c>(value: Value<'c, '_>) -> Value<'c, 'c> {
    unsafe { Value::from_raw(value.to_raw()) }
}

impl<'c> Codegen<'c> {
    fn new(
        base: &phobos_base::context::Context,
        ctx: &'c Context,
        kernel: &Kernel,
    ) -> Result<Self> {
        let launch = kernel.launch().map_err(|e| anyhow!(e))?;
        let cta_threads = launch.map_or(crate::ast::DEFAULT_CTA_THREADS, |l| l.max_threads);

        Ok(Codegen {
            isa: target::isa_for(base),
            ctx,
            loc: Location::unknown(ctx),
            index_t: Type::index(ctx),
            f16_t: Type::float16(ctx),
            bf16_t: Type::bfloat16(ctx),
            f32_t: Type::float32(ctx),
            f64_t: Type::float64(ctx),
            i8_t: IntegerType::new(ctx, 8).into(),
            i32_t: IntegerType::new(ctx, 32).into(),
            i64_t: IntegerType::new(ctx, 64).into(),
            bool_t: IntegerType::new(ctx, 1).into(),
            kernel_name: kernel.name.clone(),
            shared_globals: Vec::new(),
            tile_count: 0,
            tile_pool: HashMap::new(),
            aliased: HashSet::new(),
            dynamic_shared: kernel.wants_dynamic_shared(),
            tile_offsets: HashMap::new(),
            shared_bytes: 0,
            shared_bytes_peak: 0,
            dynamic_live: 0,
            hoisted_stages: Vec::new(),
            pipeline_assert: kernel.attrs.iter().any(|a| a.name == "pipeline"),
            pipelined_any: false,
            pipeline_declines: Vec::new(),
            tensorcore: kernel.attrs.iter().any(|a| a.name == "tensorcore"),
            mma_sync: kernel.wants_mma_sync(),
            launch,
            cta_threads,
            lowered: Vec::new(),
            policy: SharedPolicy::Pool,
            trace: lower::plan::Trace::default(),
            replay_next: 0,
            elided: std::collections::BTreeMap::new(),
            skip_barrier: None,
            barrier_calls: 0,
        })
    }

    /// How many staging buffers per operand the fused-GEMM backend allocates:
    /// two to double-buffer the k loop, one otherwise. Unlike the generic loop
    /// path, this backend has no legality or budget check of its own, so it
    /// stays gated on `@pipeline` rather than auto-attempted.
    fn staging_pairs(&self) -> usize {
        if self.pipeline_assert { 2 } else { 1 }
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    let (a, b) = (a.abs(), b.abs());
    if b == 0 { a } else { gcd(b, a % b) }
}

/// row-major strides for a shape ([`DYN`] propagates outward).
fn row_major_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = if shape[i + 1] == DYN || strides[i + 1] == DYN {
            DYN
        } else {
            strides[i + 1] * shape[i + 1]
        };
    }
    strides
}

fn int_list<T: ToString>(values: &[T]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn fmt_dim(d: i64) -> String {
    if d == DYN {
        "?".to_string()
    } else {
        d.to_string()
    }
}

fn fmt_shape(shape: &[i64]) -> String {
    shape
        .iter()
        .map(|&d| fmt_dim(d))
        .collect::<Vec<_>>()
        .join("x")
}

#[cfg(test)]
mod tests;
