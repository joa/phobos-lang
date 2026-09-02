use std::collections::{HashMap, HashSet};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use melior::{
    Context,
    dialect::{arith, memref, scf},
    ir::{
        Attribute, Block, BlockLike, Identifier, Location, Module, Operation, Region, RegionLike,
        Type, Value, ValueLike,
        attribute::{
            DenseI32ArrayAttribute, FloatAttribute, IntegerAttribute, StringAttribute,
            TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::{FunctionType, IntegerType, MemRefType},
    },
};

use crate::ast::{
    AssignOp, AttrArg, BinOp, Dim, Expr, Kernel, Launch, Literal, Scalar, Stmt, Sub,
    Type as AstType, UnOp,
};

mod call;
mod elemwise;
mod expr;
mod frag;
mod hoist;
mod kernel;
mod matmul;
mod pipeline;
mod stmt;
mod store;
mod sync;
mod target;
mod tile;
mod util;

use tile::{QFormat, QgFormat};

/// MLIR's ShapedType::kDynamic
const DYN: i64 = i64::MIN;

/// A row-reduction kind (rowmax / rowsum).
#[derive(Clone, Copy)]
enum Reduce {
    Max,
    Sum,
}

/// Lanes in a warp.
const WARP: i64 = 32;

/// Values in a Q8_0 block.
/// This is the granularity `qdot_t` scales at.
const Q8_BLOCK: i64 = 32;

const QDOT_LANE: i64 = 16;
const QDOT_STEP: i64 = WARP * QDOT_LANE;

const IMMA_TILE: i64 = 8;
const IMMA_K: i64 = 16;

const QMMA_TILES: i64 = 64;

const WMMA_SMEM_PAD: i64 = 8;

/// Shared-memory bank period on every architecture this compiler targets: 32
/// banks, 4 bytes each. See [`Codegen::should_pad_stage`].
const SHARED_BANK_BYTES: i64 = 128;

/// What `emit` learned across the module: the dynamic shared-memory sideband,
/// plus `(kernel name, decline reasons)` for every kernel whose `@pipeline`
/// assertion did not hold. `phobos_lang::compile_shared` turns a non-empty
/// `pipeline_failures` into an error.
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
        let func = cg.emit_kernel(kernel)?;
        if cg.dynamic_shared {
            shared.push((kernel.name.clone(), cg.shared_bytes_peak as usize));
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

/// XOR column swizzle for mma.sync staging buffers
///
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
    /// None means contiguous, otherwise the padded stride. See
    /// [`Codegen::alloc_tile_padded`]
    row_stride: Option<i64>,
    /// Proven divisor, in elements, of the base offset and of every stride
    /// above the innermost. The row-pitch ABI promises 4 elements on its own
    /// (16 bytes of f32, 8 of f16); `@aligned` can raise it further. Zero
    /// means no offset at all, so any width goes.
    align_div: i64,

    /// XOR column swizzle for ldmatrix staging buffers. Every access, staging
    /// store and ldmatrix load alike, must permute the column index. See
    /// [`Swizzle`]
    swizzle: Option<Swizzle>,

    /// The memref.global symbol backing a whole tile buffer (None for subviews
    /// and tensor params); what [`Codegen::release`] returns to the pool.
    global: Option<String>,

    /// Whether the bytes are in shared memory rather than global.
    /// Unlike [`Self::global`] this survives a subview.
    shared: bool,

    /// A fresh unnamed temp whose buffer may be released back to the pool once
    /// all reads of it have been emitted. [`Codegen::bind`] clears this, so
    /// named buffers are never pooled.
    owned: bool,

    /// Per-dimension bounds mask for a slice that may reach past the source
    /// extent. Some((offset, extent)) lets a load zero-fill and a store skip
    /// where offset + local index >= extent; None means provably in bounds.
    /// Empty for tile buffers and whole slices, see [`Codegen::emit_subview`].
    mask: Vec<Option<(Value<'c, 'c>, Value<'c, 'c>)>>,

    /// Known divisor of each dynamic extent, 1 when nothing is known. Only
    /// `@aligned` raises it, for tensor params only: the host's promise that
    /// the extent is a whole number of tiles. See [`Codegen::dyn_in_bounds`].
    dim_div: Vec<i64>,
}

impl<'c> MemVal<'c> {
    /// Whether any dimension carries a bounds mask (a partial tile).
    fn is_masked(&self) -> bool {
        self.mask.iter().any(Option::is_some)
    }

    /// The promised divisor of dim `d`, 1 when nothing was promised.
    fn div_of(&self, d: usize) -> i64 {
        self.dim_div.get(d).copied().unwrap_or(1)
    }

    /// Whether a `width`-element vector access stays on a legal boundary. A
    /// zero divisor is an unoffset base, which every width clears.
    fn vectorizes(&self, width: i64) -> bool {
        self.align_div % width == 0
    }
}

/// The result of evaluating an expression.
enum Rv<'c> {
    Scalar(Value<'c, 'c>),
    Tile(MemVal<'c>),
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

/// An optional scalar coefficient in a GEMM epilogue (None means 1.0/identity)
type Coeff<'a> = Option<&'a Expr>;

/// What a name in scope resolves to.
#[derive(Clone)]
enum Binding<'c> {
    /// An immutable scalar: let bindings, scalar params, loop ivs.
    /// div is the largest known divisor of the value (see [`Codegen::expr_div`]).
    Let { value: Value<'c, 'c>, div: i64 },
    /// A mutable scalar (var): a rank-0 memref holding the current value.
    Var { slot: Value<'c, 'c>, elem: Type<'c> },
    /// A tensor parameter (identity layout, global memory, sliceable).
    Tensor(MemVal<'c>),
    /// A read-only tile (let-bound slice or tile expression).
    View(MemVal<'c>),
    /// A writable tile (var-bound buffer).
    Tile(MemVal<'c>),
    /// A fragment-resident accumulator (never in shared memory).
    Frags(FragAcc<'c>),
}

struct Codegen<'c> {
    /// The chip this kernel is emitted for. Every construct the portable
    /// dialects cannot express goes through here; see [`target::Isa`]. It is
    /// the only thing the host's config survives as: the autotune choices are
    /// resolved into `shape_env` at construction and nothing else is read.
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
    shape_env: HashMap<String, i64>, // autotune search dims, seeded with each dim's first choice.
    scopes: Vec<HashMap<String, Binding<'c>>>, // symbol table
    kernel_name: String,
    shared_globals: Vec<Operation<'c>>, // shared-memory tiles (memref.global)
    tile_count: usize,
    // Released tile buffers by (element type, physical shape), reused so temps
    // don't each grow the CTA's static shared footprint, which caps occupancy.
    // See Codegen::release.
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
    /// High-water mark of `shared_bytes`, what the host must actually reserve.
    /// A barrier-separated kernel can let `shared_bytes` fall back to zero
    /// between phases (see `dynamic_live`), but the allocation still has to
    /// cover whichever phase asked for the most.
    shared_bytes_peak: i64,
    /// Count of dynamic tiles allocated and not yet released. Reaching zero
    /// between two phases of a barrier-separated kernel lets the next shape
    /// mint restart the allocation at offset 0 instead of growing to fit both
    /// phases at once. See `alloc_tile_shaped` and `release`.
    dynamic_live: i64,
    // Loop-invariant dot operands staged into shared f16 in a loop's preheader,
    // one frame per active for loop: (source view's memref value, staged
    // buffer). See codegen/hoist.rs.
    hoisted_stages: Vec<Vec<(Value<'c, 'c>, MemVal<'c>)>>,
    // Induction variable of the ragged remainder chunk being emitted, if any.
    // emit_subview guards a slice offset by it against the runtime dim; the
    // trimmed main loop leaves it None and keeps the unmasked fast paths.
    ragged_iv: Option<String>,
    // Induction variables of the enclosing trimmed main loops. Their trip count
    // was rounded down to whole chunks, so a slice offset by one of them is in
    // bounds by construction and needs no mask.
    trimmed_ivs: Vec<String>,
    /// Whether `@pipeline` was written on this kernel. The generic loop path
    /// auto-attempts every eligible loop regardless; this still gates the
    /// fused-GEMM backend's double-buffering (see [`Self::staging_pairs`]) and
    /// is checked kernel-wide as an assertion via `pipelined_any` in `emit`.
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
    /// Whether `@padstage` was written on this kernel; see
    /// [`Codegen::should_pad_stage`], its only reader.
    pad_stage: bool,
}

/// Widens a value's borrow to the context lifetime. Values borrow the block they
/// were created in, but every block here is appended to a region the module
/// transitively owns, so the MlirValue stays valid for the whole build; only the
/// borrow is too conservative.
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

        let mut shape_env = HashMap::new();
        for attr in &kernel.attrs {
            if attr.name == "autotune" {
                for arg in &attr.args {
                    if let AttrArg::Search { name, choices } = arg
                        && let Some(&first) = choices.first()
                    {
                        shape_env.insert(name.clone(), first);
                    }
                }
            }
        }

        // the autotuner pins specific choices via the phobos context.
        // only declared search dims may be overridden.
        for (name, value) in &base.shape_overrides {
            if shape_env.contains_key(name) {
                shape_env.insert(name.clone(), *value);
            }
        }

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
            shape_env,
            scopes: Vec::new(),
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
            ragged_iv: None,
            trimmed_ivs: Vec::new(),
            pipeline_assert: kernel.attrs.iter().any(|a| a.name == "pipeline"),
            pipelined_any: false,
            pipeline_declines: Vec::new(),
            tensorcore: kernel.attrs.iter().any(|a| a.name == "tensorcore"),
            mma_sync: kernel.wants_mma_sync(),
            launch,
            cta_threads,
            pad_stage: kernel.wants_padded_stage(),
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

fn broadcast_shape(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    if a.len() != b.len() {
        return None;
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| match (x, y) {
            _ if x == y => Some(x),
            (1, _) => Some(y),
            (_, 1) => Some(x),
            (DYN, _) => Some(y),
            (_, DYN) => Some(x),
            _ => None,
        })
        .collect()
}

/// Whether a slice dimension provably never reaches past the source extent,
/// so it needs no bounds mask. `size` is the slice's static extent, `off_div`
/// the largest known divisor of the slice offset (see [`Codegen::expr_div`]).
/// A dynamic source extent is handled one level up, by
/// [`Codegen::emit_split_for`] trimming the loop to provably-whole chunks.
fn dim_in_bounds(extent: i64, size: i64, off_div: i64) -> bool {
    if extent == DYN {
        return true;
    }
    if size == DYN {
        return false;
    }
    extent % size == 0 && off_div % size == 0
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
