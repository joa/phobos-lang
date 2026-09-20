mod call;
mod elemwise;
mod expr;
mod frag;
mod hoist;
mod matmul;
mod pipeline;
mod slice;
mod stmt;
mod store;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, anyhow, bail};

use super::{
    Builder, Extent, Ir, KernelInfo, Layout, OpId, OpKind, Scalar, Space, TensorType, TileType,
    Type, ValueId,
};
use crate::ast::{AttrArg, Dim, Kernel, Literal as AstLiteral, Type as AstType};

pub(crate) use crate::shape::DYN;

/// What the build asks the chip. The `@tensorcore` attributes are folded
/// in by [`Build::has_wmma`] and [`Build::has_mma_sync`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub has_cp_async: bool,
    pub has_wmma: bool,
    pub has_mma_sync: bool,
    pub mma_sync_k: Option<i64>,
    pub has_dp4a: bool,
    pub has_int8_mma: bool,
}

/// What the build learned for the emitter and the caller to report: the
/// loop shapes the pipeline declined, before any budget was counted.
#[derive(Debug, Default)]
pub struct Report {
    pub pipeline_declines: Vec<String>,
}

/// What a name in scope resolves to.
#[derive(Clone, Debug)]
pub(crate) enum Binding {
    /// An immutable scalar; `div` is its largest known divisor.
    Let { value: ValueId, div: i64 },
    /// A mutable scalar: the private slot holding it.
    Var { slot: ValueId, elem: Scalar },
    Tensor(ValueId),
    /// A read-only tile.
    View(ValueId),
    /// A writable tile buffer.
    Tile(ValueId),
    Frags(ValueId),
}

/// The result of an expression.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Rv {
    Scalar(ValueId),
    Tile(ValueId),
}

pub(crate) struct Build {
    pub(crate) ir: Ir,
    pub(crate) target: Target,
    /// The block ops are appended to.
    pub(crate) block: super::BlockId,
    scopes: Vec<HashMap<String, Binding>>,
    pub(crate) shape_env: BTreeMap<String, i64>,
    pub(crate) report: Report,
    /// Per-dimension divisors of a slice's extents, what `@aligned`
    /// promised carried through `Full` subscripts; only the prescans read
    /// it, so it stays beside the build rather than on the type.
    view_divs: HashMap<ValueId, Vec<i64>>,
    /// Loop-invariant dot operands staged in an enclosing loop's preheader,
    /// one frame per open loop: `(source view, staged buffer)`.
    pub(crate) hoisted: Vec<Vec<(ValueId, ValueId)>>,
    /// Induction variable of the ragged remainder being built, if any.
    pub(crate) ragged_iv: Option<String>,
    /// Induction variables of the enclosing trimmed main loops.
    pub(crate) trimmed_ivs: Vec<String>,
    pub(crate) tensorcore: bool,
    pub(crate) mma_sync: bool,
    pub(crate) pad_stage: bool,
    pub(crate) pipeline_assert: bool,
    pub(crate) cta_threads: i64,
}

/// Builds one kernel's graph. Names are resolved here and nowhere else,
/// and every builtin becomes one op.
///
/// What the build cannot decide it records for the emitter: whether a loop
/// shaped for double buffering fits its budget depends on what the pool
/// holds when the loop is reached, so the `For` carries the candidate and
/// the emitter answers.
pub fn build(
    base: &phobos_base::context::Context,
    target: Target,
    kernel: &Kernel,
) -> Result<(Ir, Report)> {
    let launch = kernel.launch().map_err(|e| anyhow!(e))?;
    let cta_threads = launch.map_or(crate::ast::DEFAULT_CTA_THREADS, |l| l.max_threads);

    let mut shape_env = BTreeMap::new();
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
    for (name, value) in &base.shape_overrides {
        if shape_env.contains_key(name) {
            shape_env.insert(name.clone(), *value);
        }
    }

    let info = KernelInfo {
        name: kernel.name.clone(),
        launch,
        cta_threads,
        dynamic_shared: kernel.wants_dynamic_shared(),
        pad_stage: kernel.wants_padded_stage(),
        tensorcore: kernel.attrs.iter().any(|a| a.name == "tensorcore"),
        mma_sync: kernel.wants_mma_sync(),
        pipeline_assert: kernel.attrs.iter().any(|a| a.name == "pipeline"),
        shape_env: shape_env.clone(),
    };

    let mut b = Build {
        ir: Ir::new(info.clone(), &[]),
        target,
        block: super::BlockId::from_index(0),
        scopes: Vec::new(),
        shape_env,
        report: Report::default(),
        view_divs: HashMap::new(),
        hoisted: Vec::new(),
        ragged_iv: None,
        trimmed_ivs: Vec::new(),
        tensorcore: info.tensorcore,
        mma_sync: info.mma_sync,
        pad_stage: info.pad_stage,
        pipeline_assert: info.pipeline_assert,
        cta_threads,
    };
    let params = kernel
        .params
        .iter()
        .map(|p| b.param_type(kernel, &p.ty))
        .collect::<Result<Vec<_>>>()?;
    b.ir = Ir::new(info, &params);
    b.block = b.ir.entry();
    b.emit_kernel(kernel)?;
    Ok((b.ir, b.report))
}

impl Build {
    /// Parameters bound, symbolic tensor
    /// dims bound to their runtime extents, then the body.
    fn emit_kernel(&mut self, kernel: &Kernel) -> Result<()> {
        self.scopes.push(HashMap::new());
        let entry = self.ir.entry();
        for (i, param) in kernel.params.iter().enumerate() {
            let arg = self.ir.args(entry)[i];
            self.ir.set_name(arg, &param.name);
            let binding = match &param.ty {
                // integer params join the index-type world on entry
                AstType::Scalar(crate::ast::Scalar::I32 | crate::ast::Scalar::I64) => {
                    let value = self.value(OpKind::IndexCast(Scalar::Index), &[arg], Type::INDEX);
                    Binding::Let { value, div: 1 }
                }
                AstType::Scalar(_) => Binding::Let { value: arg, div: 1 },
                // narrow-element tensor base pointers are assumed 16-byte
                // aligned (the host allocator returns aligned buffers), so
                // their rows can vectorize.
                AstType::Tensor(
                    crate::ast::Scalar::F32
                    | crate::ast::Scalar::F16
                    | crate::ast::Scalar::BF16
                    | crate::ast::Scalar::I8,
                    _,
                ) => {
                    let ty = self.ir.ty(arg).clone();
                    Binding::Tensor(self.value(OpKind::AssumeAlign, &[arg], ty))
                }
                AstType::Tensor(..) => Binding::Tensor(arg),
                AstType::Tile(..) => bail!(
                    "kernel '{}': tile-typed parameters are not supported (use tensor instead)",
                    kernel.name
                ),
            };
            self.bind(&param.name, binding);

            if let AstType::Tensor(_, dims) = &param.ty {
                for (d, dim) in dims.iter().enumerate() {
                    if let Dim::Sym(name) = dim
                        && !self.shape_env.contains_key(name)
                        && self.lookup(name).is_none()
                    {
                        let size = self.value(OpKind::Dim(d), &[arg], Type::INDEX);
                        self.ir.set_name(size, name);
                        // Dynamic dims are assumed multiples of 4 elements, the
                        // 16-byte row-pitch ABI.
                        self.bind(name, Binding::Let { value: size, div: 4 });
                    }
                }
            }
        }
        self.emit_stmts(&kernel.body)?;
        self.scopes.pop();
        Ok(())
    }

    fn param_type(&self, kernel: &Kernel, ty: &AstType) -> Result<Type> {
        Ok(match ty {
            AstType::Scalar(s) => Type::Scalar(Scalar::from_ast(*s)),
            AstType::Tensor(s, dims) => Type::Tensor(TensorType {
                elem: Scalar::from_ast(*s),
                shape: self.tensor_shape(dims).iter().map(|&d| extent(d)).collect(),
                div: self.declared_divs(kernel, dims)?,
            }),
            AstType::Tile(..) => bail!(
                "kernel '{}': tile-typed parameters are not supported (use tensor instead)",
                kernel.name
            ),
        })
    }

    /// Per-dimension divisors promised by `@aligned(NAME = tile)`.
    fn declared_divs(&self, kernel: &Kernel, dims: &[Dim]) -> Result<Vec<i64>> {
        let Some(attr) = kernel.attrs.iter().find(|a| a.name == "aligned") else {
            return Ok(vec![1; dims.len()]);
        };
        dims.iter()
            .map(|dim| {
                let Dim::Sym(name) = dim else {
                    return Ok(1);
                };
                let Some(AttrArg::KeyValue { value, .. }) = attr.args.iter().find(|a| match a {
                    AttrArg::KeyValue { key, .. } => key == name,
                    _ => false,
                }) else {
                    return Ok(1);
                };
                let div = match value {
                    AstLiteral::Int(n) => *n,
                    AstLiteral::Ident(sym) => *self
                        .shape_env
                        .get(sym)
                        .ok_or_else(|| anyhow!("@aligned({name} = {sym}): unknown constant"))?,
                    other => bail!("@aligned({name}): expected an integer, found {other:?}"),
                };
                anyhow::ensure!(div > 0, "@aligned({name}): the divisor must be positive");
                Ok(div)
            })
            .collect()
    }

    // Op creation at the current block.

    pub(crate) fn value(&mut self, kind: OpKind, operands: &[ValueId], ty: Type) -> ValueId {
        let block = self.block;
        Builder::at_end(&mut self.ir, block).value(kind, operands, ty)
    }

    pub(crate) fn stmt(&mut self, kind: OpKind, operands: &[ValueId]) -> OpId {
        let block = self.block;
        Builder::at_end(&mut self.ir, block).stmt(kind, operands)
    }

    pub(crate) fn op(
        &mut self,
        kind: OpKind,
        operands: &[ValueId],
        result_types: Vec<Type>,
        blocks: Vec<super::BlockId>,
    ) -> OpId {
        let block = self.block;
        Builder::at_end(&mut self.ir, block).op(kind, operands, result_types, blocks)
    }

    /// Runs `f` with ops going into `block`, then returns to the current one.
    pub(crate) fn in_block<T>(&mut self, block: super::BlockId, f: impl FnOnce(&mut Self) -> T) -> T {
        let outer = std::mem::replace(&mut self.block, block);
        let out = f(self);
        self.block = outer;
        out
    }

    pub(crate) fn const_index(&mut self, n: i64) -> ValueId {
        self.value(OpKind::Const(super::Literal::Int(n)), &[], Type::INDEX)
    }

    pub(crate) fn const_f32(&mut self, v: f64) -> ValueId {
        self.value(OpKind::Const(super::Literal::Float(v)), &[], Type::Scalar(Scalar::F32))
    }

    pub(crate) fn const_bool(&mut self, v: bool) -> ValueId {
        self.value(OpKind::Const(super::Literal::Bool(v)), &[], Type::BOOL)
    }

    // Scopes.

    pub(crate) fn lookup(&self, name: &str) -> Option<Binding> {
        self.scopes.iter().rev().find_map(|s| s.get(name).cloned())
    }

    /// Binds a name. A named tile is never an owned temp, which the graph
    /// records as the value's name (see `Ir::name`).
    pub(crate) fn bind(&mut self, name: &str, binding: Binding) {
        match &binding {
            Binding::View(v) | Binding::Tile(v) | Binding::Frags(v) | Binding::Tensor(v) => {
                if self.ir.name(*v).is_none() {
                    self.ir.set_name(*v, name);
                }
            }
            Binding::Let { value, .. } => {
                if self.ir.name(*value).is_none() {
                    self.ir.set_name(*value, name);
                }
            }
            Binding::Var { slot, .. } => {
                if self.ir.name(*slot).is_none() {
                    self.ir.set_name(*slot, name);
                }
            }
        }
        self.scopes
            .last_mut()
            .expect("a scope is always open while building")
            .insert(name.to_string(), binding);
    }

    /// Replaces the binding of an already-bound name in the innermost scope
    /// containing it.
    pub(crate) fn update_binding(&mut self, name: &str, binding: Binding) {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.get_mut(name) {
                *slot = binding;
                return;
            }
        }
        panic!("update_binding of unbound name '{name}'");
    }

    pub(crate) fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub(crate) fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    // Facts about values.

    pub(crate) fn tile(&self, v: ValueId) -> &TileType {
        match self.ir.ty(v) {
            Type::Tile(t) => t,
            other => panic!("{v} is {other}, not a tile"),
        }
    }

    /// The shape with [`DYN`] for dynamic extents, of a tile or tensor.
    pub(crate) fn shape(&self, v: ValueId) -> Vec<i64> {
        self.ir
            .ty(v)
            .shape()
            .unwrap_or_else(|| panic!("{v} has no shape"))
            .iter()
            .map(|e| e.fixed().unwrap_or(DYN))
            .collect()
    }

    pub(crate) fn elem(&self, v: ValueId) -> Scalar {
        self.ir
            .ty(v)
            .elem()
            .unwrap_or_else(|| panic!("{v} has no element type"))
    }

    pub(crate) fn scalar_of(&self, v: ValueId) -> Scalar {
        self.ir
            .ty(v)
            .as_scalar()
            .unwrap_or_else(|| panic!("{v} is not a scalar"))
    }

    /// Whether a value is a slice that may reach past its source.
    pub(crate) fn is_masked(&self, v: ValueId) -> bool {
        match self.ir.def_op(v).map(|op| self.ir.kind(op)) {
            Some(OpKind::Slice(s)) => s.masked.iter().any(|&m| m),
            _ => false,
        }
    }

    /// Whether a value is a whole buffer of its own rather than a view or a
    /// tensor: what `MemVal::global.is_some()` said.
    pub(crate) fn is_buffer(&self, v: ValueId) -> bool {
        self.ir
            .def_op(v)
            .is_some_and(|op| self.ir.kind(op).makes_buffer())
    }

    /// Whether a value is an unnamed temp whose buffer the consuming op may
    /// release: what `MemVal::owned` said.
    pub(crate) fn owned(&self, v: ValueId) -> bool {
        self.is_buffer(v) && self.ir.name(v).is_none()
    }

    pub(crate) fn div_of(&self, v: ValueId, d: usize) -> i64 {
        match self.ir.ty(v) {
            Type::Tensor(t) => t.div.get(d).copied().unwrap_or(1),
            _ => self.view_divs.get(&v).and_then(|ds| ds.get(d)).copied().unwrap_or(1),
        }
    }

    pub(crate) fn set_view_divs(&mut self, v: ValueId, divs: Vec<i64>) {
        self.view_divs.insert(v, divs);
    }

    // Types.

    /// Literal and `@autotune` dims become static; the rest is dynamic.
    pub(crate) fn tensor_shape(&self, dims: &[Dim]) -> Vec<i64> {
        dims.iter()
            .map(|d| match d {
                Dim::Int(n) => *n,
                Dim::Sym(name) => self.shape_env.get(name).copied().unwrap_or(DYN),
            })
            .collect()
    }

    /// A tile's shape must be fully static.
    pub(crate) fn tile_shape(&self, dims: &[Dim]) -> Result<Vec<i64>> {
        dims.iter()
            .map(|d| match d {
                Dim::Int(n) => Ok(*n),
                Dim::Sym(name) => self.shape_env.get(name).copied().ok_or_else(|| {
                    anyhow!("tile dim '{name}' must be a constant (literal or @autotune symbol)")
                }),
            })
            .collect()
    }

    /// A contiguous shared tile type of a static shape.
    pub(crate) fn shared_ty(&self, elem: Scalar, shape: &[i64]) -> Type {
        Type::Tile(TileType {
            elem,
            shape: shape.iter().map(|&d| extent(d)).collect(),
            layout: Layout {
                row_stride: None,
                swizzle: None,
                align_div: alloc_align_div(shape),
            },
            space: Space::Shared,
        })
    }

    /// A padded staging tile: the row pitch widened by the WMMA pad.
    pub(crate) fn padded_ty(&self, elem: Scalar, shape: &[i64]) -> Type {
        let mut phys = shape.to_vec();
        let last = phys.len() - 1;
        phys[last] += WMMA_SMEM_PAD;
        Type::Tile(TileType {
            elem,
            shape: shape.iter().map(|&d| extent(d)).collect(),
            layout: Layout {
                row_stride: Some(phys[last]),
                swizzle: None,
                align_div: alloc_align_div(&phys),
            },
            space: Space::Shared,
        })
    }

    /// The type a fresh buffer of `src`'s shape and element type takes.
    pub(crate) fn like(&self, src: ValueId) -> Type {
        let shape = self.shape(src);
        self.shared_ty(self.elem(src), &shape)
    }

    // What the chip can do, with what the kernel asked for folded in.

    pub(crate) fn has_wmma(&self) -> bool {
        self.tensorcore && self.target.has_wmma
    }

    pub(crate) fn has_mma_sync(&self) -> bool {
        self.mma_sync && self.target.has_mma_sync
    }
}

/// The WMMA staging pad, in elements; kept in step with
/// `codegen::WMMA_SMEM_PAD`.
pub(crate) const WMMA_SMEM_PAD: i64 = 8;

pub(crate) fn extent(d: i64) -> Extent {
    if d == DYN { Extent::Dyn } else { Extent::Fixed(d) }
}

/// The alignment `alloc_tile_shaped` proves for a fresh buffer: the gcd of
/// its row-major strides above the innermost.
pub(crate) fn alloc_align_div(shape: &[i64]) -> i64 {
    row_major_strides(shape)[..shape.len() - 1]
        .iter()
        .fold(0i64, |acc, &s| gcd(acc, s.abs().max(1)))
}

pub(crate) fn gcd(a: i64, b: i64) -> i64 {
    let (a, b) = (a.abs(), b.abs());
    if b == 0 { a } else { gcd(b, a % b) }
}

pub(crate) fn broadcast_shape(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
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

/// Whether a slice dimension provably never reaches past the source extent.
pub(crate) fn dim_in_bounds(extent: i64, size: i64, off_div: i64) -> bool {
    if extent == DYN {
        return true;
    }
    if size == DYN {
        return false;
    }
    extent % size == 0 && off_div % size == 0
}

/// Row-major strides for a shape ([`DYN`] propagates outward).
pub(crate) fn row_major_strides(shape: &[i64]) -> Vec<i64> {
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

pub(crate) fn fmt_dim(d: i64) -> String {
    if d == DYN { "?".to_string() } else { d.to_string() }
}

pub(crate) fn fmt_shape(shape: &[i64]) -> String {
    shape.iter().map(|&d| fmt_dim(d)).collect::<Vec<_>>().join("x")
}
