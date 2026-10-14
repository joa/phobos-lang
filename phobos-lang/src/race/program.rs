// One kernel's emitted MLIR, flattened into what the checker runs: every op
// with its operands and results as dense indices, its regions inlined, and
// the facts the interpreter needs (element sizes, static layouts, segment
// sizes) parsed once from the printed types and attributes.

use std::collections::HashMap;

use anyhow::{Result, bail};
use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, OperationRef, RegionLike, Value, ValueLike};

/// A value's index in [`Program::values`].
pub(super) type Vid = usize;

pub(super) struct Program {
    pub name: String,
    pub threads: i64,
    pub params: Vec<(Vid, Ty)>,
    /// For a parameter's dimension, the divisor its extent must be a
    /// multiple of: the remainder an `@aligned` check at entry traps on.
    pub aligned: HashMap<(Vid, usize), i64>,
    pub body: Region,
    pub ops: Vec<Op>,
    pub types: Vec<Ty>,
    /// Every shared-memory global of the module, with its size in bytes.
    pub globals: HashMap<String, i64>,
}

#[derive(Default)]
pub(super) struct Region {
    pub args: Vec<Vid>,
    pub ops: Vec<usize>,
    /// The terminator: a yield, a condition or a return.
    pub term: Option<usize>,
}

pub(super) struct Op {
    pub kind: Kind,
    pub operands: Vec<Vid>,
    pub results: Vec<Vid>,
    pub regions: Vec<Region>,
    /// The op's name and operand types, for a report.
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Space {
    Global,
    Shared,
    Private,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct MemTy {
    pub space: Space,
    pub shape: Vec<Option<i64>>,
    pub elem: i64,
    /// From a `strided<..>` layout; row-major from the shape otherwise.
    pub strides: Option<Vec<Option<i64>>>,
    pub offset: Option<Option<i64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Ty {
    Int(u32),
    Mem(MemTy),
    /// A vector, with its size in bytes.
    Vector(i64),
    /// A warp's matrix fragment: rows, columns, element bytes.
    Frag(i64, i64, i64),
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Arith {
    Add,
    Sub,
    Mul,
    DivU,
    DivS,
    RemU,
    RemS,
    And,
    Or,
    Xor,
    Shl,
    ShrU,
    ShrS,
    MinS,
    MaxS,
    MinU,
    MaxU,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Kind {
    Const(Option<i64>),
    Arith(Arith),
    /// `arith.cmpi`, by its predicate's number.
    Cmp(i64),
    Select,
    /// An integer conversion to the result's width; `true` zero-extends.
    Convert(bool),
    Thread,
    BlockDim,
    Block(usize),
    Grid(usize),
    Lane,
    Dim,
    /// A byte shift into a raw buffer.
    View,
    /// Static offsets, sizes and strides, `None` where an operand gives it.
    Subview(Vec<Option<i64>>, Vec<Option<i64>>, Vec<Option<i64>>),
    Reinterpret(Vec<Option<i64>>, Vec<Option<i64>>, Vec<Option<i64>>),
    /// A result that is its first operand: an alignment promise, a cast.
    Alias,
    Global(String),
    DynShared,
    Alloca,
    Load,
    Store,
    VLoad,
    VStore,
    LdMatrix(i64),
    /// Operand segments (dst, dst indices, src, src indices) and the
    /// element count copied.
    AsyncCopy([usize; 4], i64),
    AsyncGroup,
    AsyncWait(i64),
    MmaLoad(i64),
    MmaStore(i64),
    Barrier,
    /// An op every lane of a warp takes together, which orders the warp.
    WarpSync,
    For,
    AffineFor(i64, i64, i64),
    If,
    While,
    Yield,
    Condition,
    Return,
    Trap,
    Opaque,
}

impl Program {
    /// The kernel `func`, a `gpu.func`, of a module whose shared globals are
    /// `globals`.
    pub(super) fn build(
        func: OperationRef<'_, '_>,
        globals: HashMap<String, i64>,
    ) -> Result<Program> {
        let name = attr(&func, "sym_name").trim_matches('"').to_string();
        let threads = ints(&attr(&func, "nvvm.maxntid"))
            .first()
            .copied()
            .unwrap_or(crate::ast::DEFAULT_CTA_THREADS);
        let mut b = Builder {
            ids: HashMap::new(),
            types: Vec::new(),
            ops: Vec::new(),
        };
        let body = b.region(func.region(0)?)?;
        let params: Vec<(Vid, Ty)> = body.args.iter().map(|&v| (v, b.types[v].clone())).collect();
        let aligned = aligned(&b.ops, &params);
        Ok(Program {
            name,
            threads,
            params,
            aligned,
            body,
            ops: b.ops,
            types: b.types,
            globals,
        })
    }
}

/// The divisors entry checks take a parameter's dimensions by: a `memref.dim`
/// of the parameter, or of an alias of it, at a constant index, then a
/// remainder by a constant.
fn aligned(ops: &[Op], params: &[(Vid, Ty)]) -> HashMap<(Vid, usize), i64> {
    let mut root: HashMap<Vid, Vid> = params.iter().map(|&(v, _)| (v, v)).collect();
    let mut consts = HashMap::new();
    let mut dims = HashMap::new();
    let mut out = HashMap::new();
    for op in ops {
        match &op.kind {
            Kind::Const(Some(c)) => {
                consts.insert(op.results[0], *c);
            }
            Kind::Alias => {
                if let Some(&r) = root.get(&op.operands[0]) {
                    root.insert(op.results[0], r);
                }
            }
            Kind::Dim => {
                if let (Some(&r), Some(&d)) =
                    (root.get(&op.operands[0]), consts.get(&op.operands[1]))
                {
                    dims.insert(op.results[0], (r, d as usize));
                }
            }
            Kind::Arith(Arith::RemS | Arith::RemU) => {
                if let (Some(&dim), Some(&c)) =
                    (dims.get(&op.operands[0]), consts.get(&op.operands[1]))
                    && c > 0
                {
                    let lcm = out.get(&dim).map_or(c, |&e| lcm(e, c));
                    out.insert(dim, lcm);
                }
            }
            _ => {}
        }
    }
    out
}

fn lcm(a: i64, b: i64) -> i64 {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
}

struct Builder {
    ids: HashMap<usize, Vid>,
    types: Vec<Ty>,
    ops: Vec<Op>,
}

impl Builder {
    fn value(&mut self, v: Value<'_, '_>) -> Vid {
        let key = v.to_raw().ptr as usize;
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.types.len();
        self.types.push(parse_ty(&v.r#type().to_string()));
        self.ids.insert(key, id);
        id
    }

    fn region(&mut self, region: melior::ir::RegionRef<'_, '_>) -> Result<Region> {
        let Some(block) = region.first_block() else {
            return Ok(Region::default());
        };
        let mut out = Region::default();
        for i in 0..block.argument_count() {
            let arg = block.argument(i)?;
            out.args.push(self.value(arg.into()));
        }
        let mut next = block.first_operation();
        while let Some(op) = next {
            let index = self.op(op)?;
            if matches!(
                self.ops[index].kind,
                Kind::Yield | Kind::Condition | Kind::Return
            ) {
                out.term = Some(index);
            } else {
                out.ops.push(index);
            }
            next = op.next_in_block();
        }
        Ok(out)
    }

    fn op(&mut self, op: OperationRef<'_, '_>) -> Result<usize> {
        let name = op.name().as_string_ref().as_str()?.to_string();
        let operands = op.operands().map(|v| self.value(v)).collect::<Vec<_>>();
        let results = op
            .results()
            .map(|r| self.value(r.into()))
            .collect::<Vec<_>>();
        let kind = kind(&name, &op, &operands, &self.types)?;
        let mut regions = Vec::new();
        // An atomic's body runs on the device's own terms; nothing in it
        // touches shared memory.
        if name != "memref.generic_atomic_rmw" {
            for r in op.regions() {
                regions.push(self.region(r)?);
            }
        }
        // The name and operand types. Printing the op itself names every
        // value of its function, for every op, which costs the square of
        // the function's size.
        let types: Vec<String> = op.operands().map(|v| v.r#type().to_string()).collect();
        let text = if types.is_empty() {
            name.clone()
        } else {
            format!("{name} ({})", types.join(", "))
        };
        self.ops.push(Op {
            kind,
            operands,
            results,
            regions,
            text,
        });
        Ok(self.ops.len() - 1)
    }
}

fn kind(name: &str, op: &OperationRef<'_, '_>, operands: &[Vid], types: &[Ty]) -> Result<Kind> {
    use Arith::*;
    Ok(match name {
        "arith.constant" => Kind::Const(constant(&attr(op, "value"))),
        "arith.addi" => Kind::Arith(Add),
        "arith.subi" => Kind::Arith(Sub),
        "arith.muli" => Kind::Arith(Mul),
        "arith.divui" => Kind::Arith(DivU),
        "arith.divsi" => Kind::Arith(DivS),
        "arith.remui" => Kind::Arith(RemU),
        "arith.remsi" => Kind::Arith(RemS),
        "arith.andi" => Kind::Arith(And),
        "arith.ori" => Kind::Arith(Or),
        "arith.xori" => Kind::Arith(Xor),
        "arith.shli" => Kind::Arith(Shl),
        "arith.shrui" => Kind::Arith(ShrU),
        "arith.shrsi" => Kind::Arith(ShrS),
        "arith.minsi" => Kind::Arith(MinS),
        "arith.maxsi" => Kind::Arith(MaxS),
        "arith.minui" => Kind::Arith(MinU),
        "arith.maxui" => Kind::Arith(MaxU),
        "arith.cmpi" => Kind::Cmp(constant(&attr(op, "predicate")).unwrap_or(-1)),
        "arith.select" => Kind::Select,
        "arith.index_cast" | "arith.extsi" | "arith.trunci" => Kind::Convert(false),
        "arith.index_castui" | "arith.extui" => Kind::Convert(true),
        "gpu.thread_id" => Kind::Thread,
        "gpu.block_dim" => Kind::BlockDim,
        "gpu.block_id" => Kind::Block(dimension(&attr(op, "dimension"))),
        "gpu.grid_dim" => Kind::Grid(dimension(&attr(op, "dimension"))),
        "gpu.lane_id" => Kind::Lane,
        "memref.dim" => Kind::Dim,
        "memref.view" => Kind::View,
        "memref.subview" => Kind::Subview(
            dynamic(&attr(op, "static_offsets")),
            dynamic(&attr(op, "static_sizes")),
            dynamic(&attr(op, "static_strides")),
        ),
        "memref.reinterpret_cast" => Kind::Reinterpret(
            dynamic(&attr(op, "static_offsets")),
            dynamic(&attr(op, "static_sizes")),
            dynamic(&attr(op, "static_strides")),
        ),
        "memref.assume_alignment" | "memref.cast" => Kind::Alias,
        "memref.get_global" => Kind::Global(attr(op, "name").trim_start_matches('@').to_string()),
        "gpu.dynamic_shared_memory" => Kind::DynShared,
        "memref.alloca" => Kind::Alloca,
        "memref.load" => Kind::Load,
        "memref.store" => Kind::Store,
        "vector.load" => Kind::VLoad,
        "vector.store" => Kind::VStore,
        "nvgpu.ldmatrix" => Kind::LdMatrix(constant(&attr(op, "numTiles")).unwrap_or(1)),
        "nvgpu.device_async_copy" => {
            let seg = ints(&attr(op, "operandSegmentSizes"));
            if seg.len() < 4 {
                bail!("device_async_copy without operand segments");
            }
            let s = [
                seg[0] as usize,
                seg[1] as usize,
                seg[2] as usize,
                seg[3] as usize,
            ];
            Kind::AsyncCopy(s, constant(&attr(op, "dstElements")).unwrap_or(0))
        }
        "nvgpu.device_async_create_group" => Kind::AsyncGroup,
        "nvgpu.device_async_wait" => Kind::AsyncWait(constant(&attr(op, "numGroups")).unwrap_or(0)),
        "gpu.subgroup_mma_load_matrix" => {
            Kind::MmaLoad(constant(&attr(op, "leadDimension")).unwrap_or(0))
        }
        "gpu.subgroup_mma_store_matrix" => {
            Kind::MmaStore(constant(&attr(op, "leadDimension")).unwrap_or(0))
        }
        "gpu.barrier" => Kind::Barrier,
        "gpu.shuffle" | "nvgpu.mma.sync" | "gpu.subgroup_mma_compute" => Kind::WarpSync,
        "scf.for" => Kind::For,
        "affine.for" => {
            let lo = affine_constant(&attr(op, "lowerBoundMap"));
            let hi = affine_constant(&attr(op, "upperBoundMap"));
            let step = constant(&attr(op, "step")).unwrap_or(1);
            match (lo, hi) {
                (Some(lo), Some(hi)) if operands.is_empty() => Kind::AffineFor(lo, hi, step),
                _ => bail!(
                    "affine.for with bounds the checker does not read: {}",
                    attr(op, "upperBoundMap")
                ),
            }
        }
        "scf.if" => Kind::If,
        "scf.while" => Kind::While,
        "scf.yield" | "affine.yield" => Kind::Yield,
        "scf.condition" => Kind::Condition,
        "gpu.return" => Kind::Return,
        "llvm.intr.trap" => Kind::Trap,
        _ => {
            let _ = types;
            Kind::Opaque
        }
    })
}

/// An attribute as printed, or empty when the op has none by that name.
fn attr(op: &OperationRef<'_, '_>, name: &str) -> String {
    op.attribute(name)
        .map(|a| a.to_string())
        .unwrap_or_default()
}

/// The integer of a printed attribute such as `64 : index`, `-1 : i32` or
/// `true`; `None` for a float or anything else.
fn constant(s: &str) -> Option<i64> {
    let head = s.split(" : ").next()?.trim();
    match head {
        "true" => Some(1),
        "false" => Some(0),
        _ => head.parse().ok(),
    }
}

/// The integers of a printed `array<i64: ..>` or `array<i32: ..>`.
fn ints(s: &str) -> Vec<i64> {
    let Some(body) = s.split_once(':').map(|(_, b)| b.trim_end_matches('>')) else {
        return Vec::new();
    };
    body.split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

/// A static offsets, sizes or strides array, with MLIR's dynamic sentinel
/// as `None`.
fn dynamic(s: &str) -> Vec<Option<i64>> {
    ints(s)
        .into_iter()
        .map(|v| (v != i64::MIN).then_some(v))
        .collect()
}

/// The axis of a printed `#gpu<dim x>`.
fn dimension(s: &str) -> usize {
    match s.trim_end_matches('>').chars().last() {
        Some('y') => 1,
        Some('z') => 2,
        _ => 0,
    }
}

/// The value of an affine map with no inputs, such as `affine_map<() -> (3)>`.
fn affine_constant(s: &str) -> Option<i64> {
    let inner = s.strip_prefix("affine_map<() -> (")?.strip_suffix(")>")?;
    inner.trim().parse().ok()
}

/// Splits on commas outside of brackets.
fn top_level(s: &str) -> Vec<&str> {
    let (mut depth, mut start, mut out) = (0i32, 0, Vec::new());
    for (i, c) in s.char_indices() {
        match c {
            '<' | '[' | '(' => depth += 1,
            '>' | ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[start..].trim());
    out
}

fn scalar_bytes(s: &str) -> i64 {
    match s {
        "i1" | "i8" => 1,
        "f16" | "bf16" | "i16" => 2,
        "f32" | "i32" => 4,
        _ => 8,
    }
}

/// `4xf32` inside `vector<..>`, in bytes.
fn vector_bytes(s: &str) -> i64 {
    let parts: Vec<&str> = s.split('x').collect();
    let elem = scalar_bytes(parts.last().copied().unwrap_or(""));
    parts[..parts.len() - 1]
        .iter()
        .filter_map(|d| d.parse::<i64>().ok())
        .product::<i64>()
        * elem
}

pub(super) fn parse_ty(s: &str) -> Ty {
    let s = s.trim();
    if s == "index" {
        return Ty::Int(64);
    }
    if let Some(bits) = s.strip_prefix('i').and_then(|b| b.parse().ok()) {
        return Ty::Int(bits);
    }
    if let Some(inner) = s.strip_prefix("vector<").and_then(|v| v.strip_suffix('>')) {
        return Ty::Vector(vector_bytes(inner));
    }
    if let Some(inner) = s
        .strip_prefix("!gpu.mma_matrix<")
        .and_then(|v| v.strip_suffix('>'))
    {
        let shape = top_level(inner)[0];
        let parts: Vec<&str> = shape.split('x').collect();
        if parts.len() == 3 {
            let (r, c) = (
                parts[0].parse().unwrap_or(16),
                parts[1].parse().unwrap_or(16),
            );
            return Ty::Frag(r, c, scalar_bytes(parts[2]));
        }
        return Ty::Other;
    }
    let Some(inner) = s.strip_prefix("memref<").and_then(|v| v.strip_suffix('>')) else {
        return Ty::Other;
    };
    let parts = top_level(inner);
    let mut dims: Vec<&str> = parts[0].split('x').collect();
    let elem_name = dims.pop().unwrap_or("");
    let elem = if let Some(v) = elem_name.strip_prefix("vector<") {
        vector_bytes(v.trim_end_matches('>'))
    } else {
        scalar_bytes(elem_name)
    };
    let shape = dims.iter().map(|d| d.parse().ok()).collect();
    // With no address space, a buffer of the thread's own.
    let mut mem = MemTy {
        space: Space::Private,
        shape,
        elem,
        strides: None,
        offset: None,
    };
    for p in &parts[1..] {
        if let Some(layout) = p.strip_prefix("strided<") {
            let (strides, offset) = layout
                .trim_end_matches('>')
                .split_once("offset:")
                .unwrap_or((layout, "0"));
            let strides = strides
                .trim()
                .trim_end_matches(',')
                .trim_start_matches('[')
                .trim_end_matches(']');
            mem.strides = Some(strides.split(',').map(|x| x.trim().parse().ok()).collect());
            mem.offset = Some(offset.trim().parse().ok());
        } else if *p == "3" || p.contains("workgroup") {
            mem.space = Space::Shared;
        } else if *p == "1" || p.contains("global") {
            mem.space = Space::Global;
        }
    }
    Ty::Mem(mem)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_read_as_printed() {
        let Ty::Mem(m) = parse_ty("memref<1x64xf32, strided<[64, 1], offset: ?>, 1>") else {
            panic!()
        };
        assert_eq!((m.space, m.elem), (Space::Global, 4));
        assert_eq!(m.strides, Some(vec![Some(64), Some(1)]));
        assert_eq!(m.offset, Some(None));
        let Ty::Mem(m) = parse_ty("memref<?xi8, #gpu.address_space<workgroup>>") else {
            panic!()
        };
        assert_eq!((m.space, m.shape.clone()), (Space::Shared, vec![None]));
        assert_eq!(parse_ty("vector<8xf16>"), Ty::Vector(16));
        assert_eq!(
            parse_ty("!gpu.mma_matrix<16x16xf16, \"AOp\">"),
            Ty::Frag(16, 16, 2)
        );
        let Ty::Mem(m) = parse_ty("memref<index>") else {
            panic!()
        };
        assert_eq!((m.space, m.elem), (Space::Private, 8));
    }

    #[test]
    fn attributes_read_as_printed() {
        assert_eq!(constant("64 : index"), Some(64));
        assert_eq!(constant("true"), Some(1));
        assert_eq!(constant("1.000000e+00 : f32"), None);
        assert_eq!(
            dynamic("array<i64: 0, -9223372036854775808>"),
            vec![Some(0), None]
        );
        assert_eq!(affine_constant("affine_map<() -> (3)>"), Some(3));
        assert_eq!(dimension("#gpu<dim y>"), 1);
    }
}
