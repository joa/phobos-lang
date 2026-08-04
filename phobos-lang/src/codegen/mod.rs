use std::collections::HashMap;

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

mod expr;
mod frag;
mod hoist;
mod kernel;
mod matmul;
mod pipeline;
mod stmt;
mod tile;
mod util;

/// MLIR's ShapedType::kDynamic
const DYN: i64 = i64::MIN;

/// A row-reduction kind (rowmax / rowsum).
#[derive(Clone, Copy)]
enum Reduce {
    Max,
    Sum,
}

/// Address space of tensor parameters -> GPU global memory.
const MEM_GLOBAL: i64 = 1;

/// Address space of tile buffers -> GPU shared memory (one per CTA).
const MEM_SHARED: i64 = 3;

/// Address space of dynamic tile buffers.
const MEM_SHARED_SYM: &str = "#gpu.address_space<workgroup>";

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

const WMMA_A: &str = "!gpu.mma_matrix<16x16xf16, \"AOp\">";
const WMMA_B: &str = "!gpu.mma_matrix<16x16xf16, \"BOp\">";
const WMMA_C: &str = "!gpu.mma_matrix<16x16xf32, \"COp\">";

const WMMA_SMEM_PAD: i64 = 8;

pub fn emit<'c>(
    base: &phobos_base::context::Context,
    kernels: &[Kernel],
    context: &'c Context,
    module: &Module<'c>,
) -> Result<Vec<(String, usize)>> {
    let loc = Location::unknown(context);
    let gpu_block = Block::new(&[]);
    let mut shared = Vec::new();

    for kernel in kernels {
        let mut cg = Codegen::new(base, context, kernel)?;
        let func = cg.emit_kernel(kernel)?;
        if cg.dynamic_shared {
            shared.push((kernel.name.clone(), cg.shared_bytes as usize));
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

    Ok(shared)
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
    /// None means contiguous, otherwise the stride when the buffer is padded.
    ///
    /// See [`Codegen::alloc_tile_padded`]
    row_stride: Option<i64>,
    /// Base is 16-byte aligned (tile buffers and subviews have unproven dynamic offsets; tensor params unknown strides).
    ///
    /// Required for vectorization.
    aligned: bool,

    /// XOR column swizzle for ldmatrix staging buffers.
    ///
    /// Every access (staging store and ldmatrix load) must permute the column
    /// index.
    ///
    /// See [`Swizzle`]
    swizzle: Option<Swizzle>,

    /// The memref.global symbol backing a whole tile buffer (None for
    /// subviews and tensor params); what [`Codegen::release`] returns to
    /// the buffer pool.
    global: Option<String>,

    /// A fresh unnamed temp whose buffer may be released back to the pool
    /// once all reads of it have been emitted. [`Codegen::bind`] clears
    /// this, so named buffers are never pooled.
    owned: bool,

    /// Per-dimension bounds mask for a tensor slice that may reach past the
    /// source extent (a partially out-of-bounds tile). Some((offset, extent))
    /// records the slice's global offset and the source dim extent so a load
    /// can zero-fill and a store can skip the elements where offset + local
    /// index >= extent; None means the dim is provably in bounds. Empty for
    /// tile buffers and fully in-bounds slices (see [`Codegen::emit_subview`]).
    mask: Vec<Option<(Value<'c, 'c>, Value<'c, 'c>)>>,

    /// Known divisor of each dynamic extent, 1 when nothing is known. Only a
    /// kernel's `@aligned` attribute raises it, and only for tensor params: it
    /// is the host's promise that the extent is a whole number of tiles, which
    /// is what lets a program-id-offset slice skip its bounds mask. Empty means
    /// nothing is known about any dim. See [`Codegen::dyn_in_bounds`].
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

struct Codegen<'p, 'c> {
    base: &'p phobos_base::context::Context,
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
    // Released tile buffers by (element type, physical shape), reused by
    // later allocations so temps don't each grow the CTA's static shared
    // footprint (which caps occupancy). See Codegen::release.
    tile_pool: HashMap<(String, Vec<i64>), Vec<String>>,
    /// Tiles live in one dynamic allocation rather than a global apiece, sized
    /// at launch. See [`Kernel::wants_dynamic_shared`].
    dynamic_shared: bool,
    /// Byte offset of each named tile within that allocation, and how far it
    /// reaches: what the host has to pass at launch.
    tile_offsets: HashMap<String, i64>,
    shared_bytes: i64,
    // Loop-invariant dot operands staged into shared f16 in a loop's
    // preheader, one frame per active for loop: (source view's memref
    // value, staged buffer). The dot staging sites consult this instead of
    // re-staging per iteration. See codegen/hoist.rs.
    hoisted_stages: Vec<Vec<(Value<'c, 'c>, MemVal<'c>)>>,
    // Induction variable of the ragged remainder chunk currently being
    // emitted, if any. A slice offset by this variable can run past a dynamic
    // tensor extent, so emit_subview guards it against the runtime dim. The
    // trimmed main loop leaves it None and keeps the unmasked fast paths.
    // See Codegen::emit_split_for.
    ragged_iv: Option<String>,
    // Induction variables of the trimmed main loops enclosing the code being
    // emitted. Their trip count was rounded down to whole chunks, so a slice
    // offset by one of them is in bounds of a dynamic extent by construction
    // and needs no mask. See Codegen::emit_split_for.
    trimmed_ivs: Vec<String>,
    pipeline: bool,   // whether to double-buffer staged tiles in for loops
    tensorcore: bool, // whether to use tensor cores (fp16 inputs)
    mma_sync: bool,   // whether to use mma.sync, disable with @tensorcore(wmma)
    launch: Option<Launch>,
    cta_threads: i64,
}

/// Widens a value's borrow to the context lifetime. Values borrow the block
/// they were created in, but every block here is appended to a region owned
/// (transitively) by the module, so the underlying MlirValue stays valid
/// for the whole build; only the borrow is too conservative.
fn detach<'c>(value: Value<'c, '_>) -> Value<'c, 'c> {
    unsafe { Value::from_raw(value.to_raw()) }
}

impl<'p, 'c> Codegen<'p, 'c> {
    fn new(
        base: &'p phobos_base::context::Context,
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
            base,
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
            dynamic_shared: kernel.wants_dynamic_shared(),
            tile_offsets: HashMap::new(),
            shared_bytes: 0,
            hoisted_stages: Vec::new(),
            ragged_iv: None,
            trimmed_ivs: Vec::new(),
            pipeline: kernel.attrs.iter().any(|a| a.name == "pipeline"),
            tensorcore: kernel.attrs.iter().any(|a| a.name == "tensorcore"),
            mma_sync: kernel.wants_mma_sync(),
            launch,
            cta_threads,
        })
    }

    pub(super) fn cp_async(&self) -> bool {
        self.base.gpu_config.supports_cp_async() && self.base.index_bitwidth == 64
    }

    pub(super) fn wmma(&self) -> bool {
        self.tensorcore && self.base.gpu_config.compute_capability() >= 70
    }

    pub(super) fn mma_sync(&self) -> bool {
        self.mma_sync
            && self.base.gpu_config.mma_sync_k().is_some()
            && self.base.index_bitwidth == 64
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

fn mult4(divisor: i64) -> bool {
    divisor % 4 == 0
}

/// Whether a slice dimension provably never reaches past the source extent,
/// so it needs no bounds mask. `size` is the slice's static extent along the
/// dim, `off_div` the largest known divisor of the slice offset (see
/// [`Codegen::expr_div`]). Masking kicks in for a statically known extent that
/// a static, aligned tile cannot tile evenly.
///
/// A dynamic source extent has no static proof either way, so it is handled
/// one level up: [`Codegen::emit_split_for`] trims the loop to the chunks that
/// are provably whole (which reach here and stay unmasked, keeping the vector,
/// WMMA and cp.async fast paths) and replays the remainder under a runtime
/// mask against the tensor's own `memref.dim`.
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
mod tests {
    use melior::{
        Context,
        dialect::DialectRegistry,
        ir::{Location, Module, operation::OperationLike},
        utility::register_all_dialects,
    };

    fn emit_mlir(src: &str) -> String {
        emit_mlir_on(src, "sm_75")
    }

    fn emit_mlir_on(src: &str, chip: &str) -> String {
        use phobos_base::context::{Context as BaseContext, GpuConfig, NvidiaGpuConfig};
        emit_mlir_base(
            src,
            &BaseContext {
                gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)),
                ..Default::default()
            },
        )
    }

    /// Like [`emit_mlir_on`] but with 64-bit index lowering, which the
    /// @tensorcore mma.sync path requires (see Codegen::mma_sync).
    fn emit_mlir_sync(src: &str, chip: &str) -> String {
        use phobos_base::context::{Context as BaseContext, GpuConfig, NvidiaGpuConfig};
        emit_mlir_base(
            src,
            &BaseContext {
                gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)),
                index_bitwidth: 64,
                ..Default::default()
            },
        )
    }

    fn emit_mlir_base(src: &str, base: &phobos_base::context::Context) -> String {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let module = Module::new(Location::unknown(&context));
        let kernels = crate::parse(src).unwrap();
        super::emit(base, &kernels, &context, &module).unwrap();

        let text = module.as_operation().to_string();
        assert!(module.as_operation().verify(), "invalid module:\n{text}");
        text
    }

    /// emit_mlir already verifies; this just makes the intent of a test that
    /// only cares about validity readable.
    fn module_verifies(mlir: &str) -> bool {
        !mlir.is_empty()
    }

    fn assert_contains(mlir: &str, needles: &[&str]) {
        for needle in needles {
            assert!(mlir.contains(needle), "missing `{needle}` in:\n{mlir}");
        }
    }

    #[test]
    fn add_kernel_lowers_to_gpu_func() {
        let mlir = emit_mlir(
            "kernel add(X: tensor<f32>[N], Y: tensor<f32>[N], Z: tensor<f32>[N], n: i32) {
                let i = program_id(0)
                if i < n {
                    Z[i] = X[i] + Y[i]
                }
            }",
        );
        assert_contains(
            &mlir,
            &[
                "gpu.module",
                "gpu.func @add",
                "gpu.block_id",
                "arith.cmpi slt",
                "scf.if",
                "memref.load",
                "memref.store",
                "memref<?xf32, 1>",
            ],
        );
    }

    #[test]
    fn constant_range_uses_affine_dynamic_uses_scf() {
        let mlir = emit_mlir(
            "@autotune(TILE in [16, 32])
            kernel k(A: tensor<f32>[N], B: tensor<f32>[N], n: i32) {
                for j in range(0, TILE) {
                    B[j] += A[j]
                }
                for j in range(0, n) {
                    B[j] = A[j]
                }
            }",
        );
        assert_contains(&mlir, &["affine.for", "scf.for"]);
        // @autotune's first choice seeds the bound.
        assert!(mlir.contains("to 16"), "expected `to 16` bound in:\n{mlir}");
    }

    #[test]
    fn vars_and_while_lower_to_alloca_and_scf_while() {
        let mlir = emit_mlir(
            "kernel k(A: tensor<f32>[N]) {
                var s = 0.0
                var i = 0
                while i < 10 {
                    s += A[i]
                    i = i + 1
                }
            }",
        );
        assert_contains(&mlir, &["memref.alloca() : memref<f32>", "scf.while"]);
    }

    #[test]
    fn static_dims_from_literals_and_autotune() {
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64, 128])
            kernel k(A: tensor<f32>[TILE_M, 8], B: tensor<f32>[M, N]) {
                A[0, 0] = 1.0
            }",
        );
        assert_contains(&mlir, &["memref<64x8xf32, 1>", "memref<?x?xf32, 1>"]);
    }

    #[test]
    fn matmul_kernel_lowers_to_subviews_and_distributed_loops() {
        // The SPEC example, verbatim.
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64, 128], TILE_N in [64, 128], TILE_K in [16, 32])
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert_contains(
            &mlir,
            &[
                // staging buffers in shared memory: a is staged k-major
                // (transposed: 16x64, not 64x16), b is 16x64 anyway
                "memref.global \"private\" @__matmul_tile0 : memref<16x64xf32, 3>",
                "memref.global \"private\" @__matmul_tile1 : memref<16x64xf32, 3>",
                "memref.get_global @__matmul_tile0",
                // K is dynamic, so the kt loop is scf
                "memref.dim",
                // tile loads are subviews with static sizes and dynamic offsets
                "memref.subview",
                "memref<64x16xf32, strided<[?, 1], offset: ?>, 1>",
                // staging is distributed across the CTA's threads, with
                // barriers publishing it
                "gpu.thread_id",
                "gpu.block_dim",
                "gpu.barrier",
                // the lane's accumulator vector rides the kt loop as an
                // iter_arg, fed by k-chunk contractions (whose outer-product
                // lowering ends in single-rounding vector FMAs -> PTX fma.rn)
                "iter_args",
                "vector.contract",
                // fragment loads and the epilogue store are 128-bit vectors
                "vector.load",
                "vector.store",
            ],
        );
        // The acc tile itself never exists in shared memory.
        assert!(
            !mlir.contains("memref<64x64xf32, 3>"),
            "unexpected shared acc buffer in:\n{mlir}"
        );
    }

    #[test]
    fn matmul_accumulates_in_registers() {
        // Same kernel as above: the canonical pattern fuses, so the lane's
        // 4x4 accumulator vector rides the kt loop, surplus warps clamp
        // onto the last warp tile, and the epilogue writes registers
        // straight to the C subview.
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert_contains(
            &mlir,
            &["arith.minsi", "vector<4x4xf32>", "vector.contract"],
        );
        assert!(
            !mlir.contains("memref<64x64xf32, 3>"),
            "unexpected shared acc buffer in:\n{mlir}"
        );
    }

    #[test]
    fn register_fusion_bails_when_acc_outlives_store() {
        // acc is read again after the epilogue store -> no fusion; the
        // shared-accumulator path runs instead (also contraction-based,
        // but acc lives in shared memory and round-trips per kt).
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert_contains(&mlir, &["memref<64x64xf32, 3>", "vector.contract"]);
        assert!(
            !mlir.contains("math.fma"),
            "scalar FMAs left on the unfused path:\n{mlir}"
        );
    }

    #[test]
    fn matmul_is_warp_tiled() {
        // 64x64 output, 4x4 sub-tiles -> 16x16 sub-tile grid; the 4x8 lane
        // grid wins (16x32 warp tiles, most square -> minimal shared traffic),
        // giving (16/4)*(16/8) = 8 warp tiles.
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert_contains(
            &mlir,
            &[
                // tid decomposes against the warp size into warp id and
                // lane (unsigned: non-negative, pow2 -> shift/mask)
                "arith.constant 32 : index",
                "arith.divui",
                "arith.remui",
                // warps stride over the 8 warp tiles
                "arith.constant 8 : index",
                // lane offsets scale by the 16x32 warp-tile extents
                "arith.constant 16 : index",
            ],
        );
    }

    #[test]
    fn large_tiles_widen_register_blocking() {
        let src = |tile_m: &str| {
            format!(
                "@autotune(TILE_M in [{tile_m}], TILE_N in [64], TILE_K in [16])
                @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
                kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {{
                    let pm = program_id(0)
                    let pn = program_id(1)
                    var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                    for kt in range(0, K, TILE_K) {{
                        let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                        let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                        acc += dot(a, b)
                    }}
                    C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
                }}"
            )
        };
        // 128x64: a 16x16 grid of 8x4 sub-tiles (256, one per thread of a
        // 256-thread CTA) -> the lane accumulator is an 8x4 vector.
        let mlir = emit_mlir(&src("128"));
        assert_contains(&mlir, &["vector<8x4xf32>"]);
        // 64x64 can't afford TM=8 (would leave only 128 sub-tiles): 4x4.
        let mlir = emit_mlir(&src("64"));
        assert_contains(&mlir, &["vector<4x4xf32>"]);
        assert!(
            !mlir.contains("vector<8x4xf32>"),
            "unexpected TM=8 upgrade in:\n{mlir}"
        );
    }

    #[test]
    fn pipeline_double_buffers_staged_slices() {
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @pipeline
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert_contains(
            &mlir,
            &[
                // two shared buffers per staged tile; the fused matmul has
                // no acc buffer and stages a k-major, so all four are 16x64
                // (allocation order: a0, b0, a1, b1)
                "@__matmul_tile0 : memref<16x64xf32, 3>",
                "@__matmul_tile1 : memref<16x64xf32, 3>",
                "@__matmul_tile2 : memref<16x64xf32, 3>",
                "@__matmul_tile3 : memref<16x64xf32, 3>",
                // guarded prefetches and the unrolled half-B guard
                "scf.if",
            ],
        );
        // Buffers are referenced statically (unroll-by-2), never selected.
        assert!(
            !mlir.contains("arith.select"),
            "unexpected dynamic buffer select in:\n{mlir}"
        );
    }

    #[test]
    fn shape_overrides_pin_autotune_choices() {
        let src = "@autotune(TILE in [16, 32])
            kernel k(A: tensor<f32>[N], B: tensor<f32>[N]) {
                for j in range(0, TILE) {
                    B[j] = A[j]
                }
            }";
        // Default: the first choice seeds the shape env.
        assert!(emit_mlir(src).contains("to 16"));
        // The autotuner pins a choice through the base context.
        let base = phobos_base::context::Context {
            shape_overrides: [("TILE".to_string(), 32)].into(),
            ..Default::default()
        };
        assert!(emit_mlir_base(src, &base).contains("to 32"));
    }

    #[test]
    fn pipeline_uses_cp_async_on_supporting_targets() {
        let src = "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @pipeline
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }";
        // cp.async additionally requires 64-bit index lowering (upstream's
        // convert-nvgpu-to-nvvm hardcodes a 64-bit converter).
        use phobos_base::context::{Context as BaseContext, GpuConfig, NvidiaGpuConfig};
        let base = BaseContext {
            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_80")),
            index_bitwidth: 64,
            ..Default::default()
        };
        let mlir = emit_mlir_base(src, &base);
        assert_contains(
            &mlir,
            &[
                // prefetches are 16B cp.async transfers, L1-bypassed
                "nvgpu.device_async_copy",
                "bypassL1",
                // one group per stage, waited on before the closing barrier
                "nvgpu.device_async_create_group",
                "nvgpu.device_async_wait",
            ],
        );
        // Without the capability the same kernel uses plain vector copies.
        let plain = emit_mlir(src);
        assert!(
            !plain.contains("nvgpu."),
            "cp.async leaked into a non-sm_80 target:\n{plain}"
        );
        // Under the default 32-bit index ABI, sm_80 must also stay plain.
        let narrow = emit_mlir_on(src, "sm_80");
        assert!(
            !narrow.contains("nvgpu."),
            "cp.async leaked into a 32-bit-index module:\n{narrow}"
        );
    }

    #[test]
    fn tensorcore_matmul_uses_wmma() {
        // At the 32-bit-index default emit_mlir uses, @tensorcore falls back
        // to the legacy WMMA path (mma.sync needs 64-bit index).
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @tensorcore
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert_contains(
            &mlir,
            &[
                // operands staged into shared as f16 (a m-major), with one
                // truncf rounding per element; inner dims bank-conflict padded
                // by 8 (16 -> 24, 64 -> 72)
                "memref<64x24xf16, 3>",
                "memref<16x72xf16, 3>",
                "arith.truncf",
                // warp-collective fragment loads and m16n16k16 computes,
                // f32 accumulator fragments riding the kt loop
                "gpu.subgroup_mma_load_matrix",
                "!gpu.mma_matrix<16x16xf16, \"AOp\">",
                "!gpu.mma_matrix<16x16xf16, \"BOp\">",
                "gpu.subgroup_mma_compute",
                "!gpu.mma_matrix<16x16xf32, \"COp\">",
                "iter_args",
                // fragment loads stride the padded rows: lead = inner + 8
                "leadDimension = 24 : index",
                "leadDimension = 72 : index",
                // the epilogue drains through the per-warp f32 slabs
                // (8 warps x 16 rows), not straight to C
                "gpu.subgroup_mma_store_matrix",
                "memref<128x16xf32, 3>",
            ],
        );
        // The tensor cores replace the vector-FMA MAC grid entirely.
        assert!(
            !mlir.contains("vector.contract"),
            "vector MACs left on the tensor-core path:\n{mlir}"
        );
    }

    #[test]
    fn tensorcore_uses_mma_sync_on_sm75() {
        // Bare @tensorcore defaults to the mma.sync path (at 64-bit index).
        let mlir = emit_mlir_sync(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @tensorcore
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
            "sm_75",
        );
        assert_contains(
            &mlir,
            &[
                // f16 staging, but UNPADDED (no +8): the legacy pad fought bank
                // conflicts ldmatrix instead defeats with a swizzle.
                "memref<64x16xf16, 3>",
                "memref<16x64xf16, 3>",
                "arith.truncf",
                // ldmatrix loads the per-lane fragments: A as two 8x8 tiles
                // (m16k8), B as one transposed 8x8 (k8n8).
                "nvgpu.ldmatrix",
                "numTiles = 2 : i32",
                "numTiles = 1 : i32",
                "transpose = true",
                "transpose = false",
                // the m16n8k8 Turing shape, f16 operands, f32 accumulate
                "nvgpu.mma.sync",
                "mmaShape = [16, 8, 8]",
                "vector<2x2xf16>",
                "vector<1x2xf16>",
                "-> vector<2x2xf32>",
                // the XOR column swizzle (zero-cost bank-conflict avoidance)
                // riding the staging store and the ldmatrix load
                "arith.xori",
                // the epilogue still drains through the per-warp f32 slab
                "memref<128x16xf32, 3>",
            ],
        );
        // The legacy WMMA ops are gone, and so is the padded staging.
        assert!(
            !mlir.contains("subgroup_mma") && !mlir.contains("mma_matrix"),
            "legacy WMMA ops left on the mma.sync path:\n{mlir}"
        );
        assert!(
            !mlir.contains("memref<64x24xf16"),
            "padded staging left on the mma.sync path:\n{mlir}"
        );
    }

    #[test]
    fn tensorcore_sync_widens_k_to_16_on_sm80() {
        // Ampere+ has the m16n8k16 shape, halving the k-steps vs Turing's k8.
        let mlir = emit_mlir_sync(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @tensorcore(sync)
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
            "sm_80",
        );
        assert_contains(
            &mlir,
            &[
                "mmaShape = [16, 8, 16]",
                // wider k: A spans four 8x8 tiles, B spans two.
                "numTiles = 4 : i32",
                "numTiles = 2 : i32",
                "vector<4x2xf16>",
                "vector<2x2xf16>",
            ],
        );
    }

    #[test]
    fn tensorcore_sync_f16_accumulator() {
        // The gemm_fp16.ph shape: f16 inputs and an f16 accumulator, which the
        // mma.sync path carries as a vector<2x2xf16> C/D fragment.
        let mlir = emit_mlir_sync(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @tensorcore(sync)
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f16>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
            "sm_75",
        );
        assert_contains(
            &mlir,
            &[
                "nvgpu.mma.sync",
                "mmaShape = [16, 8, 8]",
                // f16 accumulator C/D fragment and the matching f16 slab.
                "-> vector<2x2xf16>",
                "memref<128x16xf16, 3>",
                // the coalesced f16 epilogue drain: 4xf16 widen / round / store
                // (f16 inputs stage without truncf, so these are the drain).
                "vector<4xf16>",
                "arith.extf",
                "arith.truncf",
            ],
        );
    }

    #[test]
    fn tensorcore_wmma_optout_forces_legacy() {
        // @tensorcore defaults to mma.sync, but @tensorcore(wmma) forces the
        // legacy warp-collective WMMA back on even at 64-bit index, where
        // mma.sync would otherwise be selected.
        let mlir = emit_mlir_sync(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @tensorcore(wmma)
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
            "sm_75",
        );
        assert!(
            mlir.contains("subgroup_mma") && !mlir.contains("nvgpu.mma.sync"),
            "@tensorcore(wmma) should force the legacy WMMA path:\n{mlir}"
        );
    }

    #[test]
    fn tensorcore_falls_back_to_wmma_without_wide_index() {
        // The mma.sync path needs 64-bit index lowering (nvgpu memref
        // descriptors are pointer-width); at the 32-bit default the default
        // mma.sync selection is ignored and codegen stays on WMMA rather than
        // emit casts that won't reconcile. emit_mlir uses the 32-bit default.
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @tensorcore
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f16>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert!(
            mlir.contains("subgroup_mma") && !mlir.contains("nvgpu."),
            "sync at 32-bit index should fall back to WMMA:\n{mlir}"
        );
    }

    #[test]
    fn tensorcore_pipelines_f16_staging() {
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [32])
            @pipeline
            @tensorcore
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert_contains(
            &mlir,
            &[
                // two f16 buffers per staged tile (order: a0, b0, a1, b1),
                // inner dims bank-conflict padded by 8 (32 -> 40, 64 -> 72)
                "@__matmul_tile0 : memref<64x40xf16, 3>",
                "@__matmul_tile1 : memref<32x72xf16, 3>",
                "@__matmul_tile2 : memref<64x40xf16, 3>",
                "@__matmul_tile3 : memref<32x72xf16, 3>",
                // guarded prefetches / the half-B guard, fragments threading
                // through as scf.if results
                "scf.if",
                "gpu.subgroup_mma_compute",
            ],
        );
        // No cp.async here: these inputs are f32, and the f32 -> f16 staging
        // round-down can't be a raw cp.async byte transfer (f16 inputs can;
        // see tensorcore_f16_inputs_pipeline_with_cp_async).
        assert!(
            !mlir.contains("nvgpu."),
            "cp.async leaked into the f32 -> f16 staging:\n{mlir}"
        );
    }

    #[test]
    fn tensorcore_drops_padding_only_when_maxregs_admits_a_cta() {
        // A large pipelined f16 tile (256x128x16): padded staging exceeds half
        // of sm_75's 64 KB carveout, so the pad blocks a second resident CTA.
        let body = "@autotune(TILE_M in [256], TILE_N in [128], TILE_K in [16])
            @LAUNCH
            @pipeline
            @tensorcore
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f16>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }";

        // Without a register cap, ptxas would keep one CTA on the register
        // limit anyway, so the pad stays (dropping it would only cost the
        // bank-conflict mitigation): buffers keep their padded width.
        let padded = emit_mlir(&body.replace("@LAUNCH", "@launch(256, 2)"));
        assert_contains(&padded, &["memref<256x24xf16, 3>", "memref<16x136xf16, 3>"]);

        // Hard-capping registers at 128 lets two CTAs fit by registers, so the
        // shared pad is now what blocks the second CTA -- it is dropped and the
        // staging stays at its logical width (lead = inner dim).
        let unpadded = emit_mlir(&body.replace("@LAUNCH", "@launch(256, 2, 128)"));
        assert_contains(
            &unpadded,
            &[
                "memref<256x16xf16, 3>",
                "memref<16x128xf16, 3>",
                "leadDimension = 16 : index",
                "leadDimension = 128 : index",
            ],
        );
        assert!(
            !unpadded.contains("memref<256x24xf16, 3>"),
            "padding survived despite the register cap admitting a second CTA:\n{unpadded}"
        );
    }

    #[test]
    fn tensorcore_f16_inputs_pipeline_with_cp_async() {
        // f16 operands stage into the WMMA fragments as a straight byte copy,
        // so the pipelined prefetch can lower to cp.async. The wmma opt-out
        // keeps this on the legacy path; at sm_80 + 64-bit index bare
        // @tensorcore would select mma.sync instead.
        let src = "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [32])
            @pipeline
            @tensorcore(wmma)
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f16>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }";
        // cp.async needs sm_80+ and 64-bit index lowering, as for the f32 path.
        use phobos_base::context::{Context as BaseContext, GpuConfig, NvidiaGpuConfig};
        let base = BaseContext {
            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_80")),
            index_bitwidth: 64,
            ..Default::default()
        };
        let mlir = emit_mlir_base(src, &base);
        assert_contains(
            &mlir,
            &[
                "nvgpu.device_async_copy",
                "nvgpu.device_async_create_group",
                "nvgpu.device_async_wait",
                "gpu.subgroup_mma_compute",
            ],
        );
        // The f16 transfers are 8 bytes (4xf16), below cp.async.cg's 16-byte
        // requirement, so unlike the f32 register path they must not bypass L1.
        assert!(
            !mlir.contains("bypassL1"),
            "8-byte f16 cp.async must not set bypassL1:\n{mlir}"
        );
        // Capability-gated: sm_75 and the 32-bit-index ABI stay on plain copies.
        assert!(
            !emit_mlir(src).contains("nvgpu."),
            "cp.async leaked into a non-sm_80 target"
        );
        assert!(
            !emit_mlir_on(src, "sm_80").contains("nvgpu."),
            "cp.async leaked into a 32-bit-index module"
        );
    }

    #[test]
    fn tensorcore_f16_pipeline_register_stages_on_sm75() {
        // Without cp.async (sm_75), the f16-input WMMA pipeline register-stages:
        // the next tile's global loads are hoisted into registers and held
        // across the WMMA compute, with the shared store deferred past it, so
        // the global latency overlaps the tensor-core math instead of stalling
        // the store. The load is unconditional with a clamped index (the
        // arith.subi/minsi below), the store guarded. (The load-before-
        // compute ordering is verified in the emitted PTX; here we lock the
        // path is taken and stays cp.async-free.)
        let src = "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @pipeline
            @tensorcore
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f16>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }";
        let mlir = emit_mlir(src); // sm_75 (no cp.async)
        assert_contains(
            &mlir,
            &[
                // unconditional global loads (from the strided slice), held in
                // registers; the in-bounds clamp distinguishes this from the
                // synchronous path (which has no arith.subi)
                "vector.load",
                "arith.subi",
                "arith.minsi",
                // the deferred shared store sits under the prefetch guard
                "scf.if",
                "vector.store",
                "gpu.subgroup_mma_compute",
            ],
        );
        assert!(
            !mlir.contains("nvgpu."),
            "cp.async leaked into the sm_75 register-staged path:\n{mlir}"
        );
        // The synchronous fallback applies when the staging tile does not divide
        // evenly across the CTA: a 16x16 A-tile is only 256 elements (< one
        // 4-wide vector per thread for the 256-thread CTA), so no arith.subi
        // clamp is emitted.
        let small = src.replace("TILE_M in [64]", "TILE_M in [16]");
        let small = small.replace("TILE_N in [64]", "TILE_N in [128]");
        let plain = emit_mlir(&small);
        assert!(
            !plain.contains("arith.subi"),
            "register-staging fired on an indivisible staging tile:\n{plain}"
        );
    }

    #[test]
    fn tensorcore_bails_to_vector_path_when_shape_does_not_fragment() {
        // TILE_K = 8 is not a multiple of 16 -> no WMMA; the regular
        // register-accumulator vector path must run instead.
        let src = "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [8])
            @tensorcore
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }";
        let mlir = emit_mlir(src);
        assert_contains(&mlir, &["vector.contract"]);
        assert!(
            !mlir.contains("subgroup_mma"),
            "WMMA emitted for a non-fragmenting shape:\n{mlir}"
        );
        // Pre-Volta chips have no tensor cores: @tensorcore is ignored.
        let old = emit_mlir_on(&src.replace("TILE_K in [8]", "TILE_K in [16]"), "sm_60");
        assert_contains(&old, &["vector.contract"]);
        assert!(
            !old.contains("subgroup_mma"),
            "WMMA emitted for a pre-sm_70 chip:\n{old}"
        );
    }

    #[test]
    fn elementwise_tile_accumulate_is_distributed() {
        let mlir = emit_mlir(
            "@autotune(T in [8, 16])
            kernel k(A: tensor<f32>[N]) {
                var acc: tile<f32>[T] = 0.0
                let a = A[0 :+ T]
                acc += a
            }",
        );
        assert_contains(
            &mlir,
            &[
                "memref<8xf32, 3>",
                "memref.get_global",
                "memref.subview",
                "gpu.thread_id",
                "gpu.block_dim",
                "scf.for",
                "arith.addf",
                // aligned operands, 8 % 4 == 0 -> vectorized accumulate
                "vector.load",
                "vector.store",
            ],
        );
    }

    #[test]
    fn fused_slice_binary_has_no_temp_buffer() {
        let mlir = emit_mlir(
            "@autotune(BLOCK in [1024])
            @aligned(N = BLOCK)
            kernel add(a: tensor<f32>[N], b: tensor<f32>[N], c: tensor<f32>[N]) {
                let base = program_id(0) * BLOCK
                c[base :+ BLOCK] = a[base :+ BLOCK] + b[base :+ BLOCK]
            }",
        );
        // c[slice] = a[slice] + b[slice] writes the target subview directly.
        assert!(
            !mlir.contains("memref.alloca"),
            "unexpected temp in:\n{mlir}"
        );
        assert!(!mlir.contains("memref.copy"), "unexpected copy in:\n{mlir}");
        assert_contains(
            &mlir,
            &[
                "gpu.thread_id",
                "gpu.block_dim",
                "scf.for",
                "arith.addf",
                // slice offsets are provably 16B-aligned (base = pid*1024),
                // so the whole add is 128-bit vectorized
                "vector.load",
                "vector.store",
                "alignment = 16",
            ],
        );
    }

    #[test]
    fn range_and_full_slices() {
        let mlir = emit_mlir(
            "kernel k(A: tensor<f32>[M, N], i: i32, j: i32) {
                let t = A[i : j, :]
                A[0 :+ 4, :] = 0.0
            }",
        );
        assert_contains(
            &mlir,
            &[
                // i:j -> dynamic size (subi), : on dynamic N -> memref.dim
                "arith.subi",
                "memref.dim",
                "memref.subview",
                // scalar store into a slice is a distributed fill
                "gpu.thread_id",
                "memref.store",
                "memref<4x?xf32, strided<[?, 1], offset: ?>, 1>",
            ],
        );
    }

    fn emit_err(src: &str) -> String {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        let module = Module::new(Location::unknown(&context));
        let kernels = crate::parse(src).unwrap();
        super::emit(
            &phobos_base::context::Context::default(),
            &kernels,
            &context,
            &module,
        )
        .expect_err("expected codegen to fail")
        .to_string()
    }

    #[test]
    fn launch_emits_nvvm_bounds() {
        let mlir = emit_mlir(
            "@launch(128, 2)
            kernel k(A: tensor<f32>[N]) { let i = program_id(0) }",
        );
        assert_contains(
            &mlir,
            &["nvvm.maxntid = array<i32: 128>", "nvvm.minctasm = 2"],
        );
    }

    #[test]
    fn launch_min_blocks_optional() {
        let mlir = emit_mlir(
            "@launch(64)
            kernel k(A: tensor<f32>[N]) { let i = program_id(0) }",
        );
        assert_contains(&mlir, &["nvvm.maxntid = array<i32: 64>"]);
        assert!(
            !mlir.contains("nvvm.minctasm"),
            "unexpected minctasm without minBlocks:\n{mlir}"
        );
    }

    #[test]
    fn launch_max_regs_emits_nvvm_maxnreg() {
        // The third @launch arg hard-caps registers per thread (PTX .maxnreg),
        // the lever for forcing occupancy when .minnctapersm is only advisory.
        let mlir = emit_mlir(
            "@launch(256, 2, 128)
            kernel k(A: tensor<f32>[N]) { let i = program_id(0) }",
        );
        assert_contains(
            &mlir,
            &[
                "nvvm.maxntid = array<i32: 256>",
                "nvvm.minctasm = 2",
                "nvvm.maxnreg = 128",
            ],
        );
    }

    #[test]
    fn launch_max_regs_must_be_in_range() {
        let err = emit_err(
            "@launch(256, 2, 8)
            kernel k(A: tensor<f32>[N]) { let i = program_id(0) }",
        );
        assert!(
            err.contains("between 16 and 255"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn no_launch_emits_no_bounds() {
        let mlir = emit_mlir("kernel k(A: tensor<f32>[N]) { let i = program_id(0) }");
        assert!(
            !mlir.contains("nvvm.maxntid")
                && !mlir.contains("nvvm.minctasm")
                && !mlir.contains("nvvm.maxnreg"),
            "unexpected launch bounds on a kernel without @launch:\n{mlir}"
        );
    }

    #[test]
    fn launch_max_threads_must_be_warp_multiple() {
        let err = emit_err(
            "@launch(100)
            kernel k(A: tensor<f32>[N]) { let i = program_id(0) }",
        );
        assert!(err.contains("multiple of 32"), "unexpected error: {err}");
    }

    #[test]
    fn flash_attention_lowers_softmax_builtins() {
        // The SPEC's online-softmax kernel exercises dot_t, exp, rowmax,
        // rowsum, tmax, broadcast subtract/divide, and tile-scalar scaling.
        let mlir = emit_mlir(
            "@autotune(D in [64], BR in [32], BC in [32])
            @aligned(Nq = BR, Nk = BC)
            kernel flash_attention(Q: tensor<f32>[Nq, D], K: tensor<f32>[Nk, D],
                                   V: tensor<f32>[Nk, D], O: tensor<f32>[Nq, D], scale: f32) {
                let pid = program_id(0)
                let row = pid * BR
                let q = Q[row :+ BR, :]
                var acc: tile<f32>[BR, D] = 0.0
                var m: tile<f32>[BR, 1] = -300000000.0
                var l: tile<f32>[BR, 1] = 0.0
                for kt in range(0, Nk, BC) {
                    let k = K[kt :+ BC, :]
                    let v = V[kt :+ BC, :]
                    var s: tile<f32>[BR, BC] = dot_t(q, k)
                    s = s * scale
                    var mnew: tile<f32>[BR, 1] = rowmax(s)
                    mnew = tmax(m, mnew)
                    var p: tile<f32>[BR, BC] = exp(s - mnew)
                    var corr: tile<f32>[BR, 1] = exp(m - mnew)
                    l = l * corr
                    l += rowsum(p)
                    acc = acc * corr
                    acc += dot(p, v)
                    m = mnew
                }
                acc = acc / l
                O[row :+ BR, :] = acc
            }",
        );
        assert_contains(
            &mlir,
            &[
                "gpu.func @flash_attention",
                "vector.contract",    // dot / dot_t
                "ex2.approx.ftz.f32", // exp lowers to the PTX ex2 intrinsic
                "arith.cmpf ogt",     // rowmax / tmax reductions
                "arith.divf",         // the broadcast normalize divide
                "arith.subf",         // broadcast subtraction inside exp
            ],
        );
    }

    #[test]
    fn layernorm_lowers_sqrt_to_ptx_intrinsic() {
        // A LayerNorm-shaped body: mean and variance via rowsum, an
        // inverse-stddev via sqrt, and broadcast center/scale. sqrt must lower
        // to the PTX sqrt.approx intrinsic (like exp -> ex2.approx).
        let mlir = emit_mlir(
            "@launch(128)
            @autotune(BR in [32], W in [64])
            kernel layernorm(X: tensor<f32>[N, W], Y: tensor<f32>[N, W]) {
                let row = program_id(0) * BR
                var x: tile<f32>[BR, W] = 0.0
                x += X[row :+ BR, :]
                var s: tile<f32>[BR, 1] = rowsum(x)
                var mu: tile<f32>[BR, 1] = s / 64.0
                var xc: tile<f32>[BR, W] = x - mu
                var v: tile<f32>[BR, 1] = rowsum(xc * xc)
                var sd: tile<f32>[BR, 1] = sqrt(v / 64.0 + 0.00001)
                Y[row :+ BR, :] = xc / sd
            }",
        );
        assert_contains(&mlir, &["sqrt.approx.f32", "arith.divf", "arith.subf"]);
    }

    #[test]
    fn log_lowers_to_the_ptx_base_two_intrinsic() {
        // softplus is the reason log exists: the GatedDeltaNet decay is
        // exp(rate * log(1 + exp(x))), so the gates cannot be computed on the
        // device without it. The hardware primitive is base two, so a natural
        // log is lg2 with the change of base folded in.
        let mlir = emit_mlir(
            "@launch(256)
            @autotune(TILE in [64])
            kernel softplus(X: tensor<f32>[M, N], Y: tensor<f32>[M, N]) {
                let p = program_id(0)
                var x = X[0 :+ 1, p * TILE :+ TILE]
                Y[0 :+ 1, p * TILE :+ TILE] = log(1.0 + exp(x))
            }",
        );
        assert_contains(
            &mlir,
            &["lg2.approx.ftz.f32", "ex2.approx.ftz.f32", "arith.mulf"],
        );
    }

    #[test]
    fn unary_minus_negates_a_tile() {
        // `-t` on a tile lowers through the scalar-broadcast path as `0 - t`,
        // so a sigmoid written the obvious way compiles.
        let mlir = emit_mlir(
            "@launch(256)
            @autotune(TILE in [64])
            kernel neg(X: tensor<f32>[M, N], Y: tensor<f32>[M, N]) {
                let p = program_id(0)
                var x = X[0 :+ 1, p * TILE :+ TILE]
                Y[0 :+ 1, p * TILE :+ TILE] = x / (1.0 + exp(-x))
            }",
        );
        assert_contains(&mlir, &["arith.subf", "arith.divf", "ex2.approx"]);
    }

    #[test]
    fn narrow_types_convert_on_load_and_store() {
        // An i8 tensor sign-extends into f32 and a bf16 result rounds back
        // down: the two halves of a dequantizing weight load.
        let mlir = emit_mlir(
            "@launch(256)
            kernel narrow(W: tensor<i8>[M, N], S: tensor<f32>[M, N], C: tensor<bf16>[M, N]) {
                let p = program_id(0)
                let w = W[0 :+ 32, p * 32 :+ 32]
                let s = S[0 :+ 32, p * 32 :+ 32]
                var out: tile<f32>[32, 32] = f32(w) * s
                C[0 :+ 32, p * 32 :+ 32] = bf16(out)
            }",
        );
        assert_contains(
            &mlir,
            &["memref<?x?xi8, 1>", "arith.sitofp", "arith.truncf", "bf16"],
        );
    }

    #[test]
    fn f16_and_bf16_operands_meet_at_f32() {
        // Neither 16-bit float contains the other, so mixing them widens to
        // f32 rather than picking a side. Two extf, no truncf.
        let mlir = emit_mlir(
            "@launch(256)
            kernel meet(A: tensor<f16>[M, N], B: tensor<bf16>[M, N], C: tensor<f32>[M, N]) {
                let p = program_id(0)
                let a = A[0 :+ 32, p * 32 :+ 32]
                let b = B[0 :+ 32, p * 32 :+ 32]
                C[0 :+ 32, p * 32 :+ 32] = a + b
            }",
        );
        assert_contains(&mlir, &["arith.extf", "f16 to f32", "bf16 to f32"]);
        assert!(
            !mlir.contains("arith.truncf"),
            "the join is f32, so nothing should round down:\n{mlir}"
        );
    }

    #[test]
    fn bf16_is_emulated_below_ampere_and_native_from_ampere() {
        // The type is available on every target; only the instruction count
        // changes. sm_75 has no bf16 unit, so NVPTX emits the shift and
        // round-to-nearest-even sequence, while sm_80 has cvt.rn.bf16.f32.
        let src = "@launch(256)
            kernel round(A: tensor<f32>[M, N], C: tensor<bf16>[M, N]) {
                let p = program_id(0)
                var a = A[0 :+ 32, p * 32 :+ 32]
                C[0 :+ 32, p * 32 :+ 32] = a * 2.0
            }";
        for chip in ["sm_75", "sm_80"] {
            assert_contains(&emit_mlir_on(src, chip), &["arith.truncf", "bf16"]);
        }
        // Capability, not availability: both compile, and phobos-base is what
        // codegen consults when it has to choose an instruction itself.
        use phobos_base::context::{GpuConfig, NvidiaGpuConfig};
        assert!(!GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_75")).supports_bf16_native());
        assert!(GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_80")).supports_bf16_native());
    }

    const I8_DOT_T: &str = "@launch(256)
            @autotune(TN in [32])
            @aligned(K = 32, N = TN)
            kernel i8dot(A: tensor<i8>[M, K], W: tensor<i8>[N, K], C: tensor<i32>[M, N]) {
                let pn = program_id(0)
                let a = A[0 :+ 1, 0 :+ 32]
                let w = W[pn * TN :+ TN, 0 :+ 32]
                C[0 :+ 1, pn * TN :+ TN] = dot_t(a, w)
            }";

    #[test]
    fn i8_dot_t_uses_the_hardware_four_way_dot() {
        // dp4a needs four contiguous bytes from each operand, which is why this
        // lands on dot_t: it contracts the last axis of both, so both walk
        // memory contiguously. One vector<4xi8> load per operand per step.
        let mlir = emit_mlir(I8_DOT_T);
        assert_contains(
            &mlir,
            &[
                "nvvm.dot.accumulate.4way",
                "vector<4xi8>",
                "<signed>",
                "memref<?x?xi8, 1>",
            ],
        );
    }

    #[test]
    fn i8_dot_t_falls_back_below_pascal() {
        // dp4a arrived with Pascal. Older targets still compile, on the generic
        // integer path.
        let mlir = emit_mlir_on(I8_DOT_T, "sm_50");
        assert!(
            !mlir.contains("nvvm.dot.accumulate.4way"),
            "sm_50 has no dp4a:\n{mlir}"
        );
        assert_contains(&mlir, &["arith.muli", "arith.addi"]);

        use phobos_base::context::{GpuConfig, NvidiaGpuConfig};
        assert!(!GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_50")).supports_dp4a());
        assert!(GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_61")).supports_dp4a());
    }

    const I8_DOT_T_TILED: &str = "@launch(256)
            @autotune(TM in [16], TN in [32])
            @aligned(M = TM, K = 32, N = TN)
            kernel i8mma(A: tensor<i8>[M, K], W: tensor<i8>[N, K], C: tensor<i32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                let a = A[pm * TM :+ TM, 0 :+ 32]
                let w = W[pn * TN :+ TN, 0 :+ 32]
                C[pm * TM :+ TM, pn * TN :+ TN] = dot_t(a, w)
            }";

    #[test]
    fn i8_dot_t_uses_the_integer_tensor_cores() {
        // With whole 8x8 output tiles the contraction goes to mma.sync instead
        // of dp4a: same four bytes per operand per lane, sixteen times the
        // products per issue. The m8n8k16 fragments are vector<1x4xi8> operands
        // into a vector<1x2xi32> accumulator.
        let mlir = emit_mlir(I8_DOT_T_TILED);
        assert_contains(
            &mlir,
            &[
                "nvgpu.mma.sync",
                "mmaShape = [8, 8, 16]",
                "vector<1x4xi8>",
                "vector<1x2xi32>",
            ],
        );
        assert!(
            !mlir.contains("nvvm.dot.accumulate.4way"),
            "dp4a left on the tensor-core path:\n{mlir}"
        );
    }

    #[test]
    fn a_single_row_i8_dot_t_stays_on_dp4a() {
        // The tensor core's smallest output tile is 8 rows, so decoding one
        // token would do eight times the arithmetic to keep one row of it.
        let mlir = emit_mlir(I8_DOT_T);
        assert!(
            !mlir.contains("nvgpu.mma.sync"),
            "one row does not fill an m8 tile:\n{mlir}"
        );
    }

    #[test]
    fn i8_dot_t_falls_back_below_turing() {
        // The integer tensor cores arrived with Turing; Pascal and Volta still
        // compile, on dp4a.
        let mlir = emit_mlir_on(I8_DOT_T_TILED, "sm_70");
        assert!(
            !mlir.contains("nvgpu.mma.sync"),
            "sm_70 has no integer tensor cores:\n{mlir}"
        );
        assert_contains(&mlir, &["nvvm.dot.accumulate.4way"]);

        use phobos_base::context::{GpuConfig, NvidiaGpuConfig};
        assert!(!GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_70")).supports_int8_mma());
        assert!(GpuConfig::Nvidia(NvidiaGpuConfig::with_chip("sm_75")).supports_int8_mma());
    }

    const Q8_QMMA: &str = "            @launch(256)
            @autotune(TM in [64], TN in [64])
            @aligned(M = TM, N = TN, K = 32)
            kernel qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                        W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
                        C: tensor<f32>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                C[pm * TM :+ TM, pn * TN :+ TN] = qmma_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                         W[pn * TN :+ TN, :], WS[:, pn * TN :+ TN])
            }";

    #[test]
    fn qmma_t_keeps_its_accumulators_in_registers() {
        // Written as a tile-language loop the block scales have to be applied
        // every 32 elements of k, which puts the accumulator in shared memory
        // and stages both operands there. Folding them into the operation
        // leaves the accumulators as loop-carried f32 values and the operands
        // as plain loads.
        let mlir = emit_mlir(Q8_QMMA);
        assert_contains(
            &mlir,
            &[
                "nvgpu.mma.sync",
                "mmaShape = [8, 8, 16]",
                "vector<1x4xi8>",
                "vector<1x2xi32>",
                // The accumulator becomes an f32 by landing in the mantissa of
                // 1.5 * 2^23, not by a quarter-rate conversion instruction.
                "arith.bitcast",
                "arith.constant 0x4B400000 : f32",
            ],
        );
        assert!(
            !mlir.contains("arith.sitofp"),
            "the block accumulator still converts the slow way in:
{mlir}"
        );
        // Eight tiles a warp, two f32 accumulators each, carried across k.
        assert_contains(&mlir, &["iter_args"]);
        let carried = mlir
            .matches("%cst = arith.constant 0.000000e+00 : f32")
            .count();
        assert!(
            carried > 0,
            "no f32 accumulator initializer in:
{mlir}"
        );
    }

    #[test]
    fn qmma_t_needs_whole_tensor_core_tiles() {
        // The integer tensor core issues 8x8 outputs, so a tile that is not a
        // multiple of eight both ways has no fragment layout to sit in.
        let src = Q8_QMMA.replace("TN in [64]", "TN in [12]");
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        let module = Module::new(Location::unknown(&context));
        let kernels = crate::parse(&src).unwrap();
        let base = phobos_base::context::Context::default();
        let err = super::emit(&base, &kernels, &context, &module).unwrap_err();
        assert!(
            err.to_string().contains("multiple of 8"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_dead_named_tile_returns_its_buffer_to_the_pool() {
        // A named tile used to hold its shared buffer for the whole kernel,
        // because bind cannot see where the name stops being read. A block's
        // own declarations are the case where it can: the last statement
        // mentioning the name ends its life, so the next allocation in the
        // block reuses the buffer instead of minting another global.
        let mlir = emit_mlir(
            "@launch(256)
            kernel chain(X: tensor<f32>[R, N], O: tensor<f32>[R, N]) {
                var a: tile<f32>[16, 64] = X[0 :+ 16, 0 :+ 64]
                var b: tile<f32>[16, 64] = a + 1.0
                var c: tile<f32>[16, 64] = b * 2.0
                var d: tile<f32>[16, 64] = c - 3.0
                O[0 :+ 16, 0 :+ 64] = d
            }",
        );
        let buffers = mlir.matches("memref.global").count();
        assert!(
            buffers <= 2,
            "four dead-on-arrival tiles want at most two buffers, got {buffers}:
{mlir}"
        );
    }

    #[test]
    fn a_tile_read_after_a_loop_keeps_its_buffer() {
        // The last mention is what ends a name's life, and a nested body is
        // part of the statement that contains it: `keep` is read inside the
        // loop and again after it, so neither point may release it.
        let mlir = emit_mlir(
            "@launch(256)
            kernel later(X: tensor<f32>[R, N], O: tensor<f32>[R, N]) {
                var keep: tile<f32>[16, 64] = X[0 :+ 16, 0 :+ 64]
                var acc: tile<f32>[16, 64] = 0.0
                for i in range(0, 4, 1) {
                    acc = acc + keep
                }
                O[0 :+ 16, 0 :+ 64] = acc + keep
            }",
        );
        assert!(module_verifies(&mlir), "{mlir}");
    }

    #[test]
    fn a_matmul_into_one_of_its_own_operands_uses_a_temp() {
        // Every matmul path writes the target as it goes, so an operand that
        // is the target would be read after it had been partly overwritten.
        // Squaring a matrix in place is the shape this takes in practice, and
        // it produced a plausible wrong answer rather than a failure: the
        // first four rows of a 16-row triangular inverse were right and the
        // rest were not.
        let mlir = emit_mlir(
            "@launch(256)
            kernel square(X: tensor<f32>[R, N], O: tensor<f32>[R, N]) {
                var p: tile<f32>[16, 16] = X[0 :+ 16, 0 :+ 16]
                p = dot(p, p)
                O[0 :+ 16, 0 :+ 16] = p
            }",
        );
        // The temp is the tell: two 16x16 buffers rather than one.
        let buffers = mlir.matches("memref<16x16xf32, 3>").count();
        assert!(
            buffers >= 2,
            "in-place square wants a temp, got {buffers} buffers:
{mlir}"
        );
    }

    #[test]
    fn i8_contraction_accumulates_in_i32() {
        // A dot product of bytes overflows i8 almost immediately, so the
        // accumulator widens even when the result type is not written down.
        let mlir = emit_mlir(
            "@launch(256)
            @aligned(K = 32, N = 32)
            kernel acc(A: tensor<i8>[M, K], W: tensor<i8>[N, K], C: tensor<f32>[M, N]) {
                let a = A[0 :+ 1, 0 :+ 32]
                let w = W[0 :+ 32, 0 :+ 32]
                C[0 :+ 1, 0 :+ 32] = f32(dot_t(a, w))
            }",
        );
        assert_contains(&mlir, &["i32", "arith.sitofp"]);
    }

    #[test]
    fn a_ragged_contraction_stays_off_the_dp4a_path() {
        // 30 bytes is not a whole number of four-byte groups; the generic path
        // handles the remainder correctly and dp4a would not.
        let mlir = emit_mlir(
            "@launch(256)
            @aligned(K = 30, N = 32)
            kernel ragged(A: tensor<i8>[M, K], W: tensor<i8>[N, K], C: tensor<i32>[M, N]) {
                let a = A[0 :+ 1, 0 :+ 30]
                let w = W[0 :+ 32, 0 :+ 30]
                C[0 :+ 1, 0 :+ 32] = dot_t(a, w)
            }",
        );
        assert!(
            !mlir.contains("nvvm.dot.accumulate.4way"),
            "30 is not a multiple of 4:\n{mlir}"
        );
    }

    #[test]
    fn converting_to_a_non_numeric_type_is_an_error() {
        let err = emit_err(
            "kernel bad(A: tensor<f32>[M, N], C: tensor<f32>[M, N]) {
                let a = A[0 :+ 32, 0 :+ 32]
                C[0 :+ 32, 0 :+ 32] = bool(a)
            }",
        );
        assert!(
            err.contains("unknown function 'bool'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rowreduce_cooperates_via_warp_shuffles() {
        // 128 threads over 32 rows leaves four lanes per row: each lane
        // folds a strided quarter of the columns in a register and a
        // gpu.shuffle xor butterfly combines the partials, instead of one
        // thread sweeping all 64 columns while three quarters of the CTA
        // idles.
        let mlir = emit_mlir(
            "@launch(128)
            @autotune(BR in [32], BC in [64])
            kernel rmax(A: tensor<f32>[N, BC], R: tensor<f32>[N, 1]) {
                let pid = program_id(0)
                let row = pid * BR
                var t: tile<f32>[BR, BC] = 0.0
                t += A[row :+ BR, :]
                var r: tile<f32>[BR, 1] = rowmax(t)
                R[row :+ BR, :] = r
            }",
        );
        assert_contains(&mlir, &["gpu.shuffle", "arith.cmpf ogt"]);
    }

    #[test]
    fn rowreduce_serial_without_spare_threads() {
        // One thread per row (128 rows, 128 threads) leaves no lanes to
        // cooperate, so the reduction stays the serial per-row sweep.
        let mlir = emit_mlir(
            "@launch(128)
            @autotune(BR in [128], BC in [64])
            kernel rsum(A: tensor<f32>[N, BC], R: tensor<f32>[N, 1]) {
                let pid = program_id(0)
                let row = pid * BR
                var t: tile<f32>[BR, BC] = 0.0
                t += A[row :+ BR, :]
                var r: tile<f32>[BR, 1] = rowsum(t)
                R[row :+ BR, :] = r
            }",
        );
        assert!(
            !mlir.contains("gpu.shuffle"),
            "warp shuffles emitted with no spare lanes per row:\n{mlir}"
        );
    }

    #[test]
    fn flash_accumulator_rides_in_fragments() {
        // The canonical fp16 flash kernel (examples/flash_attention_fp16.ph):
        // acc's every use is fragment-representable (scale by a column, +=
        // dot, final store), so it never exists in shared memory. Its
        // per-lane vector<2x2xf32> fragments ride the kt loop as scf.for
        // iter_args, the scale/normalize passes become register math, and
        // the epilogue scatters straight to O. Together with buffer pooling,
        // the fused in-place s = exp(s - mnew), and temp adoption, the
        // kernel carries 9 shared globals / ~15KB (down from 18 / 40KB), so
        // four CTAs fit an sm_75 SM instead of one.
        let mlir = emit_mlir_sync(
            "@autotune(D in [64], BR in [32], BC in [32])
            @tensorcore
            @launch(128)
            @aligned(Nq = BR, Nk = BC)
            kernel flash_attention(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D],
                                   V: tensor<f16>[Nk, D], O: tensor<f16>[Nq, D], scale: f32) {
                let pid = program_id(0)
                let row = pid * BR
                let q = Q[row :+ BR, :]
                var acc: tile<f32>[BR, D] = 0.0
                var m: tile<f32>[BR, 1] = -65504.0
                var l: tile<f32>[BR, 1] = 0.0
                for kt in range(0, Nk, BC) {
                    let k = K[kt :+ BC, :]
                    let v = V[kt :+ BC, :]
                    var s: tile<f32>[BR, BC] = dot_t(q, k)
                    s = s * scale
                    var mnew: tile<f32>[BR, 1] = rowmax(s)
                    mnew = tmax(m, mnew)
                    s = exp(s - mnew)
                    var corr: tile<f32>[BR, 1] = exp(m - mnew)
                    l = l * corr
                    l += rowsum(s)
                    acc = acc * corr
                    acc += dot(s, v)
                    m = mnew
                }
                acc = acc / l
                O[row :+ BR, :] = acc
            }",
            "sm_75",
        );
        assert_contains(
            &mlir,
            &[
                "nvgpu.mma.sync",
                "nvgpu.ldmatrix",
                // fragments thread the kt loop as iter_args
                "iter_args",
                "vector<2x2xf32>",
            ],
        );
        // acc never materializes in shared memory.
        assert!(
            !mlir.contains("memref<32x64xf32, 3>"),
            "fragment accumulator materialized in shared memory:\n{mlir}"
        );
        let globals = mlir.matches("memref.global").count();
        assert!(
            globals <= 9,
            "expected at most 9 pooled shared buffers, got {globals}:\n{mlir}"
        );
        // One [BR, BC] f32 buffer serves scores and probabilities: the
        // fused exp(s - mnew) sweep rewrites it in place.
        let s_bufs = mlir
            .lines()
            .filter(|l| l.contains("memref.global") && l.contains("memref<32x32xf32, 3>"))
            .count();
        assert_eq!(
            s_bufs, 1,
            "expected the score tile to stay a single in-place buffer:\n{mlir}"
        );
    }

    /// Splits emitted IR at the flash kt loop (the only loop bounded by a
    /// dynamic %dim) into (preheader, body) for staging-placement asserts.
    ///
    /// A ragged-split loop is trimmed to whole chunks first, so it is bounded
    /// by that arithmetic rather than by %dim directly; the frag-carried path
    /// does not split and still reads `to %dim`. Take whichever comes first.
    fn split_at_kt_loop(mlir: &str) -> (&str, &str) {
        let pos = [" = arith.subi %dim", " to %dim"]
            .iter()
            .filter_map(|pat| mlir.find(pat))
            .min()
            .expect("no dynamically-bounded loop in module");
        mlir.split_at(pos)
    }

    #[test]
    fn flash_q_staging_hoists_to_the_loop_preheader() {
        // q in dot_t(q, k) is a let-bound Q slice defined outside the kt
        // loop and nothing in the body stores to global memory, so its
        // global-to-shared f16 staging copy runs once in the preheader
        // instead of every iteration. q's subview is the first in the
        // kernel, so its loads print as "%subview[" exactly.
        let src = "@autotune(D in [64], BR in [32], BC in [32])
            @tensorcore
            @launch(128)
            @aligned(Nq = BR, Nk = BC)
            kernel flash_attention(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D],
                                   V: tensor<f16>[Nk, D], O: tensor<f16>[Nq, D], scale: f32) {
                let pid = program_id(0)
                let row = pid * BR
                let q = Q[row :+ BR, :]
                var acc: tile<f32>[BR, D] = 0.0
                var m: tile<f32>[BR, 1] = -65504.0
                var l: tile<f32>[BR, 1] = 0.0
                for kt in range(0, Nk, BC) {
                    let k = K[kt :+ BC, :]
                    let v = V[kt :+ BC, :]
                    var s: tile<f32>[BR, BC] = dot_t(q, k)
                    s = s * scale
                    var mnew: tile<f32>[BR, 1] = rowmax(s)
                    mnew = tmax(m, mnew)
                    s = exp(s - mnew)
                    var corr: tile<f32>[BR, 1] = exp(m - mnew)
                    l = l * corr
                    l += rowsum(s)
                    acc = acc * corr
                    acc += dot(s, v)
                    m = mnew
                }
                acc = acc / l
                O[row :+ BR, :] = acc
            }";
        // Both tensor-core paths hoist: mma.sync (frag-carried kt loop) and
        // the legacy WMMA fallback (plain kt loop).
        for mlir in [emit_mlir_sync(src, "sm_75"), emit_mlir(src)] {
            let (preheader, body) = split_at_kt_loop(&mlir);
            assert!(
                preheader.contains("load %subview["),
                "q staging not hoisted to the preheader:\n{mlir}"
            );
            assert!(
                !body.contains("load %subview["),
                "q still re-staged inside the kt loop:\n{mlir}"
            );
        }
    }

    #[test]
    fn dot_staging_stays_in_loop_when_body_stores_global() {
        // The body stores s to O each iteration: a staged copy of q could
        // not see global writes, so the hoist must stand down and q stages
        // inside the loop as before.
        let mlir = emit_mlir_sync(
            "@autotune(D in [64], BR in [32], BC in [32])
            @tensorcore
            @launch(128)
            @aligned(Nq = BR, Nk = BC)
            kernel qk(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D], O: tensor<f32>[Nq, BC]) {
                let pid = program_id(0)
                let row = pid * BR
                let q = Q[row :+ BR, :]
                for kt in range(0, Nk, BC) {
                    let k = K[kt :+ BC, :]
                    var s: tile<f32>[BR, BC] = dot_t(q, k)
                    O[row :+ BR, :] = s
                }
            }",
            "sm_75",
        );
        let (preheader, body) = split_at_kt_loop(&mlir);
        assert!(
            !preheader.contains("load %subview["),
            "q staging hoisted past a global store:\n{mlir}"
        );
        assert!(
            body.contains("load %subview["),
            "q staging missing from the loop body:\n{mlir}"
        );
    }

    #[test]
    fn frag_acc_falls_back_on_unsanctioned_reads() {
        // rowsum(o) reads the accumulator outside the fragment-representable
        // forms, so the candidate must reject it and o stays a shared tile.
        let mlir = emit_mlir_sync(
            "@autotune(D in [64], BR in [64], BC in [64])
            @tensorcore
            @launch(256)
            @aligned(Nq = BR, Nk = BC)
            kernel pv(P: tensor<f16>[Nq, Nk], V: tensor<f16>[Nk, D],
                      O: tensor<f32>[Nq, D], R: tensor<f32>[Nq, 1]) {
                let pid = program_id(0)
                let row = pid * BR
                var o: tile<f32>[BR, D] = 0.0
                let p = P[row :+ BR, 0 :+ BC]
                let v = V[0 :+ BC, :]
                o += dot(p, v)
                var r: tile<f32>[BR, 1] = rowsum(o)
                O[row :+ BR, :] = o
                R[row :+ BR, :] = r
            }",
            "sm_75",
        );
        assert_contains(&mlir, &["memref<64x64xf32, 3>", "nvgpu.mma.sync"]);
    }

    #[test]
    fn tensorcore_dot_t_loads_transposed_b() {
        // dot_t (Q @ K.T) on the tensor cores stages both operands in
        // their natural [rows, D] layout and loads the B (K) fragment
        // column-major (transpose), so no transposing staging pass is
        // needed.
        let mlir = emit_mlir(
            "@autotune(D in [64], BR in [64], BC in [64])
            @tensorcore
            @launch(256)
            @aligned(Nq = BR, Nk = BC)
            kernel qk(Q: tensor<f32>[Nq, D], K: tensor<f32>[Nk, D], S: tensor<f32>[Nq, Nk]) {
                let pid = program_id(0)
                let row = pid * BR
                let q = Q[row :+ BR, :]
                let k = K[0 :+ BC, :]
                var s: tile<f32>[BR, BC] = dot_t(q, k)
                S[row :+ BR, 0 :+ BC] = s
            }",
        );
        assert_contains(
            &mlir,
            &[
                // both operands staged f16 in their natural [rows, D] layout
                "memref<64x64xf16, 3>",
                "arith.truncf",
                // warp-collective fragment loads + computes, the B load
                // column-major (transpose) for the Q @ K.T contraction
                "gpu.subgroup_mma_load_matrix",
                "transpose",
                "gpu.subgroup_mma_compute",
                "gpu.subgroup_mma_store_matrix",
            ],
        );
        assert!(
            !mlir.contains("vector.contract"),
            "vector MACs left on the tensor-core dot_t path:\n{mlir}"
        );
    }

    #[test]
    fn tensorcore_f16_dot_stages_vectorized() {
        // With f16 operands the WMMA staging is a plain copy (no truncf), and
        // it vectorizes as 4xf16 (8-byte) accesses: the row-pitch ABI's
        // multiple-of-4-elements guarantee is exactly 8B for f16. Without this
        // the f16 staging would be scalar, costing it the 4x it loses to the
        // f32 path's 4xf32 staging.
        let mlir = emit_mlir(
            "@autotune(D in [64], BR in [64], BC in [64])
            @tensorcore
            @launch(256)
            @aligned(Nq = BR, Nk = BC)
            kernel qk(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D], S: tensor<f32>[Nq, Nk]) {
                let pid = program_id(0)
                let row = pid * BR
                let q = Q[row :+ BR, :]
                let k = K[0 :+ BC, :]
                var s: tile<f32>[BR, BC] = dot_t(q, k)
                S[row :+ BR, 0 :+ BC] = s
            }",
        );
        assert_contains(
            &mlir,
            &[
                "memref<64x64xf16, 3>",
                // 8-byte 4xf16 staging loads and stores, not scalar
                "vector.load",
                "vector.store",
                "vector<4xf16>",
                "alignment = 8",
                "gpu.subgroup_mma_compute",
            ],
        );
        // f16 in, f16 staged: no rounding conversion on the staging path.
        assert!(
            !mlir.contains("arith.truncf"),
            "unexpected truncf staging f16 operands:\n{mlir}"
        );
    }

    #[test]
    fn tensorcore_dot_uses_wmma() {
        // The plain dot (P @ V) path runs on the tensor cores with a
        // row-major (non-transposed) B load.
        let mlir = emit_mlir(
            "@autotune(D in [64], BR in [64], BC in [64])
            @tensorcore
            @launch(256)
            @aligned(Nq = BR, Nk = BC)
            kernel pv(P: tensor<f32>[Nq, Nk], V: tensor<f32>[Nk, D], O: tensor<f32>[Nq, D]) {
                let pid = program_id(0)
                let row = pid * BR
                let p = P[row :+ BR, 0 :+ BC]
                let v = V[0 :+ BC, :]
                var o: tile<f32>[BR, D] = dot(p, v)
                O[row :+ BR, :] = o
            }",
        );
        assert_contains(
            &mlir,
            &[
                "memref<64x64xf16, 3>",
                "gpu.subgroup_mma_load_matrix",
                "gpu.subgroup_mma_compute",
                "gpu.subgroup_mma_store_matrix",
            ],
        );
        // NN contraction: the B fragment stays row-major, no transpose.
        assert!(
            !mlir.contains("transpose"),
            "unexpected transposed load on the NN dot path:\n{mlir}"
        );
        assert!(
            !mlir.contains("vector.contract"),
            "vector MACs left on the tensor-core dot path:\n{mlir}"
        );
    }

    #[test]
    fn tensorcore_dot_t_uses_mma_sync() {
        // At 64-bit index dot_t (Q @ K.T) takes the default mma.sync path:
        // ldmatrix + nvgpu.mma.sync over swizzled f16 staging, no WMMA.
        let mlir = emit_mlir_sync(
            "@autotune(D in [64], BR in [64], BC in [64])
            @tensorcore
            @launch(256)
            @aligned(Nq = BR, Nk = BC)
            kernel qk(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D], S: tensor<f32>[Nq, Nk]) {
                let pid = program_id(0)
                let row = pid * BR
                let q = Q[row :+ BR, :]
                let k = K[0 :+ BC, :]
                var s: tile<f32>[BR, BC] = dot_t(q, k)
                S[row :+ BR, 0 :+ BC] = s
            }",
            "sm_75",
        );
        assert_contains(
            &mlir,
            &[
                // unpadded, XOR-swizzled f16 staging read by ldmatrix
                "memref<64x64xf16, 3>",
                "arith.xori",
                "nvgpu.ldmatrix",
                "nvgpu.mma.sync",
                "mmaShape = [16, 8, 8]",
                // the m16n8 f32 accumulator scattered to the shared output tile
                "vector<2x2xf32>",
            ],
        );
        assert!(
            !mlir.contains("subgroup_mma") && !mlir.contains("mma_matrix"),
            "legacy WMMA ops left on the mma.sync dot_t path:\n{mlir}"
        );
    }

    #[test]
    fn tensorcore_dot_accumulate_rides_in_fragments() {
        // o += dot(p, v) with an accumulator whose every use is fragment-
        // representable never materializes o in shared memory: the per-lane
        // mma.sync D fragments seed the MAC directly and the epilogue
        // scatters them straight to O. The NN B operand is still read
        // transposed (k-major staging).
        let mlir = emit_mlir_sync(
            "@autotune(D in [64], BR in [64], BC in [64])
            @tensorcore
            @launch(256)
            @aligned(Nq = BR, Nk = BC)
            kernel pv(P: tensor<f16>[Nq, Nk], V: tensor<f16>[Nk, D], O: tensor<f32>[Nq, D]) {
                let pid = program_id(0)
                let row = pid * BR
                var o: tile<f32>[BR, D] = 0.0
                let p = P[row :+ BR, 0 :+ BC]
                let v = V[0 :+ BC, :]
                o += dot(p, v)
                O[row :+ BR, :] = o
            }",
            "sm_75",
        );
        assert_contains(
            &mlir,
            &[
                "nvgpu.ldmatrix",
                "nvgpu.mma.sync",
                // NN dot reads the k-major B staging transposed
                "transpose = true",
            ],
        );
        assert!(
            !mlir.contains("memref<64x64xf32, 3>"),
            "fragment accumulator materialized in shared memory:\n{mlir}"
        );
        assert!(
            !mlir.contains("subgroup_mma"),
            "legacy WMMA ops left on the mma.sync dot accumulate path:\n{mlir}"
        );
    }

    #[test]
    fn dot_falls_back_to_vector_without_tensorcore() {
        // Same shapes, but no @tensorcore: the generic tile-dot (vector)
        // path must run, never WMMA.
        let mlir = emit_mlir(
            "@autotune(D in [64], BR in [64], BC in [64])
            @launch(256)
            @aligned(Nq = BR, Nk = BC)
            kernel qk(Q: tensor<f32>[Nq, D], K: tensor<f32>[Nk, D], S: tensor<f32>[Nq, Nk]) {
                let pid = program_id(0)
                let row = pid * BR
                let q = Q[row :+ BR, :]
                let k = K[0 :+ BC, :]
                var s: tile<f32>[BR, BC] = dot_t(q, k)
                S[row :+ BR, 0 :+ BC] = s
            }",
        );
        assert!(
            !mlir.contains("subgroup_mma"),
            "WMMA emitted for dot_t without @tensorcore:\n{mlir}"
        );
    }

    #[test]
    fn scalar_ops_cover_arith_and_comparisons() {
        // every scalar operator across float, integer, and bool operands,
        // plus unary neg/not and an f32/f64 mixed-width promotion.
        let mlir = emit_mlir(
            "kernel k(out: tensor<f32>[N], a: f32, b: f64) {
                var f = a * a + a - a / a
                f = -f
                let wide = a + b
                let lt = a < a
                let cmp = (a == a) != (a <= a)
                var i = 0
                i = i * 2 + 1 - i / 1
                let ic = i < 4
                let bn = !lt
                out[0] = f
            }",
        );
        assert_contains(
            &mlir,
            &[
                "arith.mulf",
                "arith.addf",
                "arith.subf",
                "arith.divf",
                "arith.negf", // unary neg on a float
                "arith.cmpf", // float comparisons
                "arith.extf", // f32 -> f64 widening in a + b
                "arith.muli", // integer arithmetic on index
                "arith.cmpi", // integer comparison
                "arith.xori", // !lt on a bool
            ],
        );
    }

    #[test]
    fn unknown_builtin_is_an_error() {
        let err = emit_err("kernel k(A: tensor<f32>[N]) { let x = wobble(A) }");
        assert!(err.contains("unknown function"), "got: {err}");
    }

    #[test]
    fn program_id_dimension_must_be_literal_0_to_2() {
        let err = emit_err("kernel k(A: tensor<f32>[N]) { let x = program_id(3) }");
        assert!(err.contains("program_id"), "got: {err}");
    }

    #[test]
    fn unknown_identifier_is_an_error() {
        let err = emit_err("kernel k(A: tensor<f32>[N]) { A[0] = nope }");
        assert!(err.contains("unknown identifier"), "got: {err}");
    }

    #[test]
    fn tensor_used_as_value_is_an_error() {
        let err = emit_err("kernel k(A: tensor<f32>[N]) { let x = A }");
        assert!(err.contains("index or slice it"), "got: {err}");
    }

    #[test]
    fn generic_pipeline_double_buffers_a_non_matmul_loop() {
        // a @pipeline loop that stages a slice but is not the fused matmul
        // template, so it exercises the generic software-pipelining path
        // (double buffers + guarded prefetch) rather than the GEMM backend.
        let mlir = emit_mlir(
            "@pipeline
            @autotune(T in [16])
            kernel stage(A: tensor<f32>[M, K], C: tensor<f32>[M, K]) {
                let pm = program_id(0)
                var acc: tile<f32>[T, T] = 0.0
                for kt in range(0, K, T) {
                    var a = A[pm * T :+ T, kt :+ T]
                    acc += a
                }
                C[pm * T :+ T, 0 :+ T] = acc
            }",
        );
        assert_contains(
            &mlir,
            &[
                "gpu.func @stage",
                // two shared staging buffers for the single staged slice
                "@__stage_tile0 : memref<16x16xf32, 3>",
                "@__stage_tile1 : memref<16x16xf32, 3>",
                "scf.if",      // the guarded prefetch of the next tile
                "gpu.barrier", // publish/consume barriers around staging
            ],
        );
    }

    #[test]
    fn f16_tensors_lower_to_f16_memrefs() {
        // f16 tensor params and tile buffers carry the f16 element type, and
        // the 0.0 f32 literal seed is rounded down on store.
        let mlir = emit_mlir(
            "@autotune(T in [8])
            kernel k(A: tensor<f16>[N]) {
                var acc: tile<f16>[T] = 0.0
                let a = A[0 :+ T]
                acc += a
            }",
        );
        assert_contains(
            &mlir,
            &[
                "memref<?xf16, 1>", // the f16 tensor param
                "memref<8xf16, 3>", // the f16 shared tile buffer
                "arith.truncf",     // the f32 literal rounded to f16
                "arith.addf",       // the elementwise f16 accumulate
            ],
        );
        // f16 rows aren't 16B-aligned under the multiple-of-4 ABI, so the
        // elementwise op stays scalar (no 128-bit f32 vectors).
        assert!(
            !mlir.contains("vector<4xf32>"),
            "unexpected f32 vectorization of an f16 tile:\n{mlir}"
        );
    }

    #[test]
    fn f16_scalar_arithmetic_widens_to_f32() {
        // Mixing an f16 operand with an f32 one widens to f32 (arith.extf).
        let mlir = emit_mlir(
            "kernel k(out: tensor<f32>[N], a: f16, b: f32) {
                out[0] = a + b
            }",
        );
        assert_contains(&mlir, &["arith.extf", "arith.addf"]);
    }

    #[test]
    fn f16_matmul_runs_on_tensor_cores() {
        // f16 inputs and output, f32 accumulation: the operands stage into
        // the WMMA fragments verbatim (no rounding), and the f32 result is
        // rounded back to f16 in the epilogue.
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @tensorcore
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert_contains(
            &mlir,
            &[
                // f16 operand tensors and f16 shared staging (a m-major),
                // inner dims bank-conflict padded by 8 (16 -> 24, 64 -> 72)
                "memref<?x?xf16, 1>",
                "memref<64x24xf16, 3>",
                "memref<16x72xf16, 3>",
                "gpu.subgroup_mma_compute",
                "!gpu.mma_matrix<16x16xf32, \"COp\">",
                // the f32 accumulator is rounded to the f16 output
                "arith.truncf",
            ],
        );
        assert!(
            !mlir.contains("vector.contract"),
            "vector MACs left on the f16 tensor-core path:\n{mlir}"
        );
    }

    #[test]
    fn f16_matmul_accumulates_in_f16_on_tensor_cores() {
        // An f16 accumulator runs the WMMA in the m16n16k16 f16.f16 mode:
        // f16 COp fragments and an f16 drain slab, no f32 anywhere in the MAC.
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @tensorcore
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f16>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert_contains(
            &mlir,
            &[
                "gpu.subgroup_mma_compute",
                "!gpu.mma_matrix<16x16xf16, \"COp\">", // f16 accumulator fragments
                "memref<128x16xf16, 3>",               // the f16 drain slab
            ],
        );
        // No f32 accumulator fragments or slab on the f16-accumulate path.
        assert!(
            !mlir.contains("\"COp\">") || !mlir.contains("mma_matrix<16x16xf32, \"COp\">"),
            "unexpected f32 accumulator fragment in f16-accumulate matmul:\n{mlir}"
        );
        assert!(
            !mlir.contains("memref<128x16xf32, 3>"),
            "unexpected f32 drain slab in f16-accumulate matmul:\n{mlir}"
        );
    }

    #[test]
    fn f16_matmul_without_tensorcore_uses_f16_vector_contract() {
        // No @tensorcore and an f16 accumulator: the generic register matmul
        // contracts in f16 (no fusion, no WMMA).
        let mlir = emit_mlir(
            "@autotune(TILE_M in [64], TILE_N in [64], TILE_K in [16])
            @aligned(M = TILE_M, N = TILE_N, K = TILE_K)
            kernel matmul(A: tensor<f16>[M, K], B: tensor<f16>[K, N], C: tensor<f16>[M, N]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f16>[TILE_M, TILE_N] = 0.0
                for kt in range(0, K, TILE_K) {
                    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert_contains(&mlir, &["vector<4x4xf16>", "vector.contract"]);
        assert!(
            !mlir.contains("subgroup_mma"),
            "WMMA emitted for an f16 matmul without @tensorcore:\n{mlir}"
        );
    }

    #[test]
    fn f16_flash_attention_runs_on_tensor_cores() {
        // f16 Q/K/V/O with an f32 online-softmax state: both matmuls run on
        // the tensor cores (f16 operands, f32 accumulate), the softmax math
        // stays f32, and the result rounds back to f16 on the store.
        let mlir = emit_mlir(
            "@autotune(D in [64], BR in [64], BC in [64])
            @tensorcore
            @pipeline
            @launch(256)
            @aligned(Nq = BR, Nk = BC)
            kernel flash_attention(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D],
                                   V: tensor<f16>[Nk, D], O: tensor<f16>[Nq, D], scale: f32) {
                let pid = program_id(0)
                let row = pid * BR
                let q = Q[row :+ BR, :]
                var acc: tile<f32>[BR, D] = 0.0
                var m: tile<f32>[BR, 1] = -65504.0
                var l: tile<f32>[BR, 1] = 0.0
                for kt in range(0, Nk, BC) {
                    let k = K[kt :+ BC, :]
                    let v = V[kt :+ BC, :]
                    var s: tile<f32>[BR, BC] = dot_t(q, k)
                    s = s * scale
                    var mnew: tile<f32>[BR, 1] = rowmax(s)
                    mnew = tmax(m, mnew)
                    var p: tile<f32>[BR, BC] = exp(s - mnew)
                    var corr: tile<f32>[BR, 1] = exp(m - mnew)
                    l = l * corr
                    l += rowsum(p)
                    acc = acc * corr
                    acc += dot(p, v)
                    m = mnew
                }
                acc = acc / l
                O[row :+ BR, :] = acc
            }",
        );
        assert_contains(
            &mlir,
            &[
                "gpu.func @flash_attention",
                "memref<?x64xf16, 1>",      // f16 Q/K/V/O tensors
                "gpu.subgroup_mma_compute", // dot_t and dot on the cores
                "ex2.approx.ftz.f32",       // f32 softmax exp
                "arith.truncf",             // f32 acc rounded to the f16 O
            ],
        );
        assert!(
            !mlir.contains("vector.contract"),
            "vector MACs left on the f16 tensor-core attention path:\n{mlir}"
        );
    }

    #[test]
    fn f16_flash_attention_without_tensorcore_widens_to_f32() {
        // Same kernel, no @tensorcore and a non-fragmenting tile: the mixed
        // f16-input/f32-accumulate dots fall back to the vector path, widening
        // each f16 operand to f32 (arith.extf) on load.
        let mlir = emit_mlir(
            "@autotune(D in [8], BR in [8], BC in [8])
            @aligned(Nq = BR, Nk = BC)
            kernel flash_attention(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D],
                                   V: tensor<f16>[Nk, D], O: tensor<f16>[Nq, D], scale: f32) {
                let pid = program_id(0)
                let row = pid * BR
                let q = Q[row :+ BR, :]
                var acc: tile<f32>[BR, D] = 0.0
                var m: tile<f32>[BR, 1] = -65504.0
                var l: tile<f32>[BR, 1] = 0.0
                for kt in range(0, Nk, BC) {
                    let k = K[kt :+ BC, :]
                    let v = V[kt :+ BC, :]
                    var s: tile<f32>[BR, BC] = dot_t(q, k)
                    s = s * scale
                    var mnew: tile<f32>[BR, 1] = rowmax(s)
                    mnew = tmax(m, mnew)
                    var p: tile<f32>[BR, BC] = exp(s - mnew)
                    var corr: tile<f32>[BR, 1] = exp(m - mnew)
                    l = l * corr
                    l += rowsum(p)
                    acc = acc * corr
                    acc += dot(p, v)
                    m = mnew
                }
                acc = acc / l
                O[row :+ BR, :] = acc
            }",
        );
        assert_contains(
            &mlir,
            &[
                "vector.contract", // generic mixed-precision dots
                "arith.extf",      // f16 operands widened to the f32 accumulator
                "arith.truncf",    // f32 result rounded to the f16 output tensor
            ],
        );
        assert!(
            !mlir.contains("subgroup_mma"),
            "WMMA emitted without @tensorcore:\n{mlir}"
        );
    }

    #[test]
    fn partial_static_slice_is_masked() {
        // A 32-wide tile does not tile a 100-element tensor evenly, so the
        // last tile runs past the end. The read stages through a zero-filled
        // buffer (select) and the write skips the out-of-bounds lanes (scf.if
        // guarded by offset + index < extent).
        let mlir = emit_mlir(
            "kernel copy(A: tensor<f32>[100], B: tensor<f32>[100]) {
                let p = program_id(0)
                B[p * 32 :+ 32] = A[p * 32 :+ 32]
            }",
        );
        assert_contains(
            &mlir,
            &[
                "memref.subview",
                "arith.cmpi ult", // offset + index < extent
                "arith.select",   // out-of-bounds reads fold to zero
                "scf.if",         // the store runs only in bounds
            ],
        );
    }

    #[test]
    fn a_program_id_slice_of_a_dynamic_extent_is_masked() {
        // The extent is a runtime value, so nothing inside the kernel bounds
        // the grid: the last program id can address a tile that runs off the
        // end. Left unmasked this writes through the end of a row and into the
        // next one, which is a whole-tensor corruption rather than a bad tail.
        let mlir = emit_mlir(
            "kernel copy(A: tensor<f32>[M, N], B: tensor<f32>[M, N]) {
                let p = program_id(0)
                B[0 :+ 1, p * 32 :+ 32] = A[0 :+ 1, p * 32 :+ 32]
            }",
        );
        assert_contains(
            &mlir,
            &[
                "memref.dim",     // the extent is only known at runtime
                "arith.cmpi ult", // offset + index < extent
                "arith.select",   // out-of-bounds reads fold to zero
                "scf.if",         // the store runs only in bounds
            ],
        );
    }

    #[test]
    fn an_aligned_declaration_drops_the_dynamic_bounds_mask() {
        // @aligned is the host promising the extent is a whole number of tiles,
        // which is what a general GEMM needs to keep the register-blocked and
        // tensor-core drains: those have no per-element store guard, so without
        // the promise they decline and the kernel falls back.
        let mlir = emit_mlir(
            "@aligned(N = 32)
            kernel copy(A: tensor<f32>[M, N], B: tensor<f32>[M, N]) {
                let p = program_id(0)
                B[0 :+ 1, p * 32 :+ 32] = A[0 :+ 1, p * 32 :+ 32]
            }",
        );
        assert!(
            !mlir.contains("scf.if"),
            "unexpected bounds guard for a declared-aligned extent:
{mlir}"
        );
    }

    #[test]
    fn an_aligned_declaration_must_cover_the_tile() {
        // Promising a coarser tiling than the slice takes proves nothing: 32
        // does not divide a multiple of 24, so the mask stays.
        let mlir = emit_mlir(
            "@aligned(N = 24)
            kernel copy(A: tensor<f32>[M, N], B: tensor<f32>[M, N]) {
                let p = program_id(0)
                B[0 :+ 1, p * 32 :+ 32] = A[0 :+ 1, p * 32 :+ 32]
            }",
        );
        assert_contains(&mlir, &["arith.cmpi ult", "scf.if"]);
    }

    #[test]
    fn an_unknown_aligned_constant_is_rejected() {
        let err = emit_err(
            "@aligned(N = TILE)
            kernel copy(A: tensor<f32>[M, N]) {
                let p = program_id(0)
                A[0 :+ 1, p * 32 :+ 32] = A[0 :+ 1, p * 32 :+ 32]
            }",
        );
        assert!(err.contains("unknown constant"), "unexpected error: {err}");
    }

    #[test]
    fn aligned_static_slice_is_not_masked() {
        // A 32-wide tile tiles a 128-element tensor evenly, so no lane ever
        // leaves the tensor: no bounds mask, and the copy still vectorizes.
        let mlir = emit_mlir(
            "kernel copy(A: tensor<f32>[128], B: tensor<f32>[128]) {
                let p = program_id(0)
                B[p * 32 :+ 32] = A[p * 32 :+ 32]
            }",
        );
        assert_contains(&mlir, &["memref.subview", "vector<4xf32>"]);
        assert!(
            !mlir.contains("scf.if") && !mlir.contains("arith.select"),
            "unexpected bounds mask for an evenly tiled tensor:\n{mlir}"
        );
    }

    #[test]
    fn partial_matmul_epilogue_is_masked() {
        // N = 100 is not a multiple of TILE_N = 32, so the fused register
        // matmul declines (its blocking has no bounds guard) and the generic
        // tiled path runs with a masked epilogue store into C.
        let mlir = emit_mlir(
            "@autotune(TILE_M in [32], TILE_N in [32], TILE_K in [32])
            kernel matmul(A: tensor<f32>[96, 64], B: tensor<f32>[64, 100], C: tensor<f32>[96, 100]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var acc: tile<f32>[TILE_M, TILE_N] = 0.0
                for kt in range(0, 64, TILE_K) {
                    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
                    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
                    acc += dot(a, b)
                }
                C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
            }",
        );
        assert_contains(&mlir, &["memref.subview", "arith.cmpi ult", "scf.if"]);
    }

    #[test]
    fn dot_directly_into_partial_slice_is_rejected() {
        // A dot result written straight into a partially out-of-bounds slice
        // has no accumulator tile to carry the bounds guard, so it is a
        // compile error rather than an out-of-bounds store.
        let err = emit_err(
            "kernel matmul(A: tensor<f32>[100, 64], B: tensor<f32>[64, 100], C: tensor<f32>[100, 100]) {
                let pm = program_id(0)
                let pn = program_id(1)
                var a = A[pm * 32 :+ 32, 0 :+ 64]
                var b = B[0 :+ 64, pn * 32 :+ 32]
                C[pm * 32 :+ 32, pn * 32 :+ 32] = dot(a, b)
            }",
        );
        assert!(
            err.contains("partially out-of-bounds"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn dynamic_extent_loop_splits_off_a_masked_remainder() {
        // N is dynamic, so nothing statically proves C tiles it evenly. The
        // loop is trimmed to (N / C) * C and the ragged remainder replays the
        // body once, guarded against the tensor's runtime memref.dim.
        let mlir = emit_mlir(
            "@autotune(D in [32], C in [32])
            kernel scale(X: tensor<f32>[N, D], O: tensor<f32>[N, D]) {
                for c in range(0, N, C) {
                    var t: tile<f32>[C, D] = X[c :+ C, :]
                    t = t * 2.0
                    O[c :+ C, :] = t
                }
            }",
        );
        assert_contains(
            &mlir,
            &[
                "memref.dim",     // the runtime extent
                "arith.divui",    // (N - 0) / C whole chunks
                "arith.cmpi ult", // offset + index < N inside the remainder
                "arith.select",   // out-of-bounds reads fold to zero
                "scf.if",         // the remainder runs only when N is ragged
            ],
        );
    }

    #[test]
    fn static_extent_loop_does_not_split() {
        // A static extent the tile divides evenly is provably whole, so the
        // loop keeps its single unguarded form.
        let mlir = emit_mlir(
            "@autotune(D in [32], C in [32])
            kernel scale(X: tensor<f32>[128, D], O: tensor<f32>[128, D]) {
                for c in range(0, 128, C) {
                    var t: tile<f32>[C, D] = X[c :+ C, :]
                    t = t * 2.0
                    O[c :+ C, :] = t
                }
            }",
        );
        assert!(
            !mlir.contains("arith.cmpi ult") && !mlir.contains("memref.dim"),
            "unexpected ragged split for an evenly tiled static extent:\n{mlir}"
        );
    }

    #[test]
    fn split_main_loop_keeps_wmma_and_needs_no_mask() {
        // The trimmed main loop's slices are in bounds by construction, so it
        // keeps the tensor-core path and takes no bounds mask; only the
        // remainder pays for the guard.
        let mlir = emit_mlir(
            "@autotune(D in [64], BR in [32], BC in [32])
            @tensorcore
            @launch(128)
            @aligned(Nq = BR, Nk = BC)
            kernel qk(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D], O: tensor<f32>[Nq, BC]) {
                let pid = program_id(0)
                let q = Q[pid * BR :+ BR, :]
                var acc: tile<f32>[BR, BC] = 0.0
                for kt in range(0, Nk, BC) {
                    let k = K[kt :+ BC, :]
                    acc += dot_t(q, k)
                }
                O[pid * BR :+ BR, :] = acc
            }",
        );

        let (_, loops) = split_at_kt_loop(&mlir);
        let at_remainder = loops
            .find("\n      scf.if ")
            .expect("no ragged remainder in module");
        let (main, remainder) = loops.split_at(at_remainder);
        // The remainder's `trim < extent` guard is emitted just before the
        // scf.if, so drop it before asserting the main loop carries no mask.
        let main = main.rfind("arith.cmpi ult").map_or(main, |at| &main[..at]);

        assert!(
            main.contains("gpu.subgroup_mma_compute"),
            "the trimmed main loop lost WMMA:\n{mlir}"
        );
        assert!(
            !main.contains("arith.cmpi ult"),
            "the trimmed main loop should need no bounds mask:\n{mlir}"
        );
        assert!(
            remainder.contains("arith.cmpi ult"),
            "the ragged remainder is unguarded:\n{mlir}"
        );
    }

    #[test]
    fn cumsum_tril_transpose_lower_and_verify() {
        // The linear-attention primitives: cumsum scans the sequence axis,
        // tril masks the strict upper triangle (a compare plus select), and
        // transpose mirrors a rank-2 tile so a contraction can run over the
        // leading axis.
        let mlir = emit_mlir(
            "@autotune(D in [32], C in [32])
            kernel prim(G: tensor<f32>[N, 1], X: tensor<f32>[N, D], O: tensor<f32>[N, D]) {
                let c = program_id(0)
                let g = G[c :+ C, :]
                var b: tile<f32>[C, 1] = cumsum(g)
                let x = X[c :+ C, :]
                var xb: tile<f32>[C, D] = x * b
                var p: tile<f32>[C, C] = dot_t(xb, xb)
                p = tril(p)
                var xt = transpose(xb)               // [D, C]
                var kv: tile<f32>[C, C] = dot(xb, xt) // [C,D] @ [D,C] -> [C,C]
                var o: tile<f32>[C, D] = dot(p, x)
                O[c :+ C, :] = o
            }",
        );
        assert_contains(
            &mlir,
            &[
                "gpu.func @prim",
                "arith.cmpi sle",  // tril's j <= i predicate
                "arith.select",    // tril keeps or zeroes each element
                "vector.contract", // the dot / dot_t matmuls
            ],
        );
    }

    #[test]
    fn kda_chunkwise_gated_linear_attention_lowers() {
        // The KDA backbone (examples/kda_fp32.ph): chunkwise gated linear
        // attention carrying an [D, D] recurrent state, exercising cumsum
        // (the gate), tril (causal mask), transpose (K^T V), exp, and the
        // intra/inter dot products.
        let mlir = emit_mlir(
            "@autotune(D in [64], C in [32, 128])
            kernel kda(Q: tensor<f32>[N, D], K: tensor<f32>[N, D], V: tensor<f32>[N, D],
                       G: tensor<f32>[N, 1], O: tensor<f32>[N, D], scale: f32) {
                var S: tile<f32>[D, D] = 0.0
                for c in range(0, N, C) {
                    let q = Q[c :+ C, :]
                    let k = K[c :+ C, :]
                    let v = V[c :+ C, :]
                    let g = G[c :+ C, :]
                    var b: tile<f32>[C, 1] = cumsum(g)
                    var db = exp(b)
                    var negb = b * -1.0
                    var dbi = exp(negb)
                    var qd: tile<f32>[C, D] = q * db
                    qd = qd * scale
                    var kd: tile<f32>[C, D] = k * dbi
                    var p: tile<f32>[C, C] = dot_t(qd, kd)
                    p = tril(p)
                    var o: tile<f32>[C, D] = dot(p, v)
                    o += dot(qd, S)
                    O[c :+ C, :] = o
                    var gt = transpose(g)
                    var total: tile<f32>[1, 1] = rowsum(gt)
                    var kfin = k * exp(total - b)
                    var kt = transpose(kfin)
                    var kv: tile<f32>[D, D] = dot(kt, v)
                    S = S * exp(total) + kv
                }
            }",
        );
        assert_contains(
            &mlir,
            &[
                "gpu.func @kda",
                "ex2.approx.ftz.f32", // exp on the gates
                "arith.select",       // tril causal mask
                "vector.contract",    // the chunk matmuls
            ],
        );
    }
}
