mod call;
mod elemwise;
mod epilogue;
mod expr;
mod frag;
mod hoist;
mod lane_sum;
mod lanes;
mod matmul;
mod names;
mod pipeline;
mod regs;
mod slice;
mod stmt;
mod store;
mod unroll;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};

use super::{
    Builder, Extent, Ir, KernelInfo, Layout, OpId, OpKind, Scalar, Space, TensorType, TileType,
    Type, ValueId,
};
use crate::ast::{AttrArg, BinOp, Dim, Kernel, Literal as AstLiteral, Span, Stmt, Type as AstType};
use crate::diag::{Diagnostic, Diagnostics, SourceFile};

pub(crate) use super::at_op::locate;
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
    /// The most shared memory a block may opt into.
    pub smem_per_block: i64,
    /// Lanes in a warp.
    pub lanes: i64,
}

/// What the build learned for the emitter and the caller to report: the
/// loop shapes the pipeline declined, before any budget was counted, and the
/// statement each op came from, which the visualizer highlights. An op a
/// pass made after the build has no statement; [`Ir::span`] is the innermost
/// source an op stands for.
#[derive(Debug, Default)]
pub struct Report {
    pub pipeline_declines: Vec<String>,
    pub spans: BTreeMap<OpId, Span>,
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
    /// A tile held in registers; see `regs.rs`.
    Regs(ValueId),
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
    /// Per-dimension divisors of a slice's extents, carried from `@aligned`
    /// through `Full` subscripts. Only the prescans read it, so it lives
    /// here rather than on the type.
    view_divs: HashMap<ValueId, Vec<i64>>,
    /// Loop-invariant dot operands staged in an enclosing loop's preheader,
    /// one frame per open loop: `(source view, staged buffer)`.
    pub(crate) hoisted: Vec<Vec<(ValueId, ValueId)>>,
    /// Induction variable of the ragged remainder being built, if any.
    pub(crate) ragged_iv: Option<String>,
    /// Induction variables of the enclosing trimmed main loops.
    pub(crate) trimmed_ivs: Vec<String>,
    /// Loop variables a prescan reasons about before their loop binds them,
    /// with the divisor every value they take has.
    pub(crate) pending_ivs: Vec<(String, i64)>,
    pub(crate) tensorcore: bool,
    pub(crate) mma_sync: bool,
    pub(crate) pad_stage: bool,
    pub(crate) pipeline_assert: bool,
    pub(crate) cta_threads: i64,
    /// The tiles `@registers` names, and those declared so far.
    pub(crate) registers: Vec<String>,
    pub(crate) regs_declared: Vec<String>,
    /// Set while planning a register tile's update, the one tree a register
    /// tile may be a leaf of.
    pub(crate) fusing_regs: bool,
    /// The kernel's source, which errors are placed in.
    pub(crate) source: Arc<SourceFile>,
    /// What went wrong in the statements recovered from.
    errors: Vec<Diagnostic>,
    /// Names whose declaration failed; see [`Poisoned`].
    poisoned: HashSet<String>,
}

/// A name used whose declaration already failed. It is reported once, at
/// the declaration, so a use of it says nothing more.
#[derive(Debug)]
pub(crate) struct Poisoned;

impl std::fmt::Display for Poisoned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a name whose declaration failed")
    }
}

impl std::error::Error for Poisoned {}

/// What a statement that fails may leave behind, put back before the next.
pub(crate) struct Saved {
    block: super::BlockId,
    scopes: usize,
    hoisted: usize,
    ragged_iv: Option<String>,
    trimmed_ivs: usize,
    pending_ivs: usize,
    fusing_regs: bool,
}

/// Builds one kernel's graph. Names are resolved here and nowhere else,
/// and every builtin becomes one op.
///
/// What the build cannot decide it records for the emitter. Whether a
/// double-buffered loop fits its budget depends on what the pool holds when
/// the loop is reached, so the `For` carries the candidate and the emitter
/// decides.
pub fn build(
    base: &phobos_base::context::Context,
    target: Target,
    kernel: &Kernel,
) -> Result<(Ir, Report)> {
    let empty = Ir::new(KernelInfo::default(), &[]);
    let attr_span = |name: &str| kernel.attr(name).map_or(kernel.name_span, |a| a.span);
    let launch = kernel
        .launch()
        .map_err(|e| locate(&kernel.source, &empty, anyhow!(e), attr_span("launch")))?;
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
        pending_ivs: Vec::new(),
        tensorcore: info.tensorcore,
        mma_sync: info.mma_sync,
        pad_stage: info.pad_stage,
        pipeline_assert: info.pipeline_assert,
        cta_threads,
        registers: kernel.register_tiles(),
        regs_declared: Vec::new(),
        fusing_regs: false,
        source: kernel.source.clone(),
        errors: Vec::new(),
        poisoned: HashSet::new(),
    };
    let params = kernel
        .params
        .iter()
        .map(|p| {
            b.param_type(kernel, &p.ty)
                .map_err(|e| locate(&kernel.source, &empty, e, p.ty_span))
        })
        .collect::<Result<Vec<_>>>()?;
    b.ir = Ir::new(info, &params);
    b.block = b.ir.entry();
    let built = b
        .emit_kernel(kernel)
        .map_err(|e| locate(&kernel.source, &b.ir, e, kernel.name_span));
    if let Err(e) = built {
        b.report_error(e)?;
    }
    if !b.errors.is_empty() {
        return Err(Diagnostics::new(b.source.clone(), std::mem::take(&mut b.errors)).into());
    }
    super::uniform::check(&b.ir, &super::uniform::analyze(&b.ir))
        .map_err(|e| locate(&kernel.source, &b.ir, e, kernel.name_span))?;
    for (fused, dot) in epilogue::fuse_dot_epilogues(&mut b.ir) {
        if let Some(&span) = b.report.spans.get(&dot) {
            b.report.spans.insert(fused, span);
        }
    }
    b.report.spans.retain(|&op, _| b.ir.is_alive(op));
    let swizzle = kernel
        .swizzle_mode()
        .map_err(|e| locate(&kernel.source, &b.ir, anyhow!(e), attr_span("swizzle")))?;
    super::swizzle::decide(&mut b.ir, swizzle)
        .map_err(|e| locate(&kernel.source, &b.ir, e, attr_span("swizzle")))?;
    Ok((b.ir, b.report))
}

impl Build {
    /// Binds the parameters and the symbolic tensor dims, then builds the
    /// body.
    fn emit_kernel(&mut self, kernel: &Kernel) -> Result<()> {
        self.scopes.push(HashMap::new());
        let entry = self.ir.entry();
        for (i, param) in kernel.params.iter().enumerate() {
            let arg = self.ir.args(entry)[i];
            self.ir.set_name(arg, &param.name);
            let binding = match &param.ty {
                // Integer params convert to index on entry.
                AstType::Scalar(crate::ast::Scalar::I32 | crate::ast::Scalar::I64) => {
                    let value = self.value(OpKind::IndexCast(Scalar::Index), &[arg], Type::INDEX);
                    Binding::Let { value, div: 1 }
                }
                AstType::Scalar(_) => Binding::Let { value: arg, div: 1 },
                // These tensors' base pointers are assumed 16-byte aligned,
                // as the host allocator guarantees, so their rows can
                // vectorize.
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
                AstType::Tile(..) => bail!("a kernel parameter cannot be a tile; declare it as a `tensor`"),
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
        self.check_aligned(kernel)?;
        self.emit_stmts(&kernel.body)?;
        // A declaration that failed was reported where it was written.
        let missing = |n: &&String| !self.regs_declared.contains(n) && !self.poisoned.contains(*n);
        if let Some(name) = self.registers.iter().find(missing) {
            bail!("`@registers({name})` names no `var {name}: tile<..>[..]` of the kernel");
        }
        self.scopes.pop();
        Ok(())
    }

    /// Traps at entry when a launch breaks an `@aligned(DIM = tile)` promise
    /// on a dynamic dimension, before anything is written past a tensor.
    fn check_aligned(&mut self, kernel: &Kernel) -> Result<()> {
        for (name, div) in self.aligned_promises(kernel)? {
            let Some(Binding::Let { value: size, .. }) = self.lookup(&name) else {
                continue;
            };
            if div <= 1 {
                continue;
            }
            let div = self.const_index(div);
            let rem = self.value(OpKind::Binary(BinOp::Rem), &[size, div], Type::INDEX);
            let zero = self.const_index(0);
            let broken = self.value(OpKind::Binary(BinOp::Ne), &[rem, zero], Type::Scalar(Scalar::Bool));
            let then = self.ir.new_block(&[]);
            self.in_block(then, |b| {
                b.stmt(OpKind::Trap, &[]);
                b.stmt(OpKind::Yield, &[]);
            });
            self.op(OpKind::If, &[broken], Vec::new(), vec![then]);
        }
        Ok(())
    }

    /// Every `@aligned(NAME = tile)` pair, the tile resolved to its value.
    fn aligned_promises(&self, kernel: &Kernel) -> Result<Vec<(String, i64)>> {
        let Some(attr) = kernel.attrs.iter().find(|a| a.name == "aligned") else {
            return Ok(Vec::new());
        };
        attr.args
            .iter()
            .filter_map(|a| match a {
                AttrArg::KeyValue { key, value } => Some((key, value)),
                _ => None,
            })
            .map(|(name, value)| {
                let div = match value {
                    AstLiteral::Int(n) => *n,
                    AstLiteral::Ident(sym) => *self
                        .shape_env
                        .get(sym)
                        .ok_or_else(|| anyhow!("`@aligned({name} = {sym})`: unknown constant"))?,
                    other => bail!("`@aligned({name})`: expected an integer, found {other:?}"),
                };
                anyhow::ensure!(div > 0, "`@aligned({name})`: the divisor must be positive");
                Ok((name.clone(), div))
            })
            .collect()
    }

    fn param_type(&self, kernel: &Kernel, ty: &AstType) -> Result<Type> {
        Ok(match ty {
            AstType::Scalar(s) => Type::Scalar(Scalar::from_ast(*s)),
            AstType::Tensor(s, dims) => Type::Tensor(TensorType {
                elem: Scalar::from_ast(*s),
                shape: self.tensor_shape(dims).iter().map(|&d| extent(d)).collect(),
                div: self.declared_divs(kernel, dims)?,
            }),
            AstType::Tile(..) => bail!("a kernel parameter cannot be a tile; declare it as a `tensor`"),
        })
    }

    /// Per-dimension divisors promised by `@aligned(NAME = tile)`.
    fn declared_divs(&self, kernel: &Kernel, dims: &[Dim]) -> Result<Vec<i64>> {
        let promises = self.aligned_promises(kernel)?;
        Ok(dims
            .iter()
            .map(|dim| match dim {
                Dim::Sym(name) => promises.iter().find(|(n, _)| n == name).map_or(1, |&(_, div)| div),
                Dim::Int(_) => 1,
            })
            .collect())
    }

    /// Runs `f` for the statement at `span`: every op it made that no
    /// nested statement claimed comes from it, and an error it returns is
    /// placed there unless something inside placed it first.
    pub(crate) fn spanned<T>(
        &mut self,
        span: Span,
        f: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let first = self.ir.op_count();
        let out = self.at_span(span, f);
        if span != Span::default() {
            for op in (first..self.ir.op_count()).map(OpId::from_index) {
                self.report.spans.entry(op).or_insert(span);
            }
        }
        out
    }

    /// Runs `f` for the source at `span`, the span the ops it makes take and
    /// where an error it returns is placed. An empty span changes neither.
    pub(crate) fn at_span<T>(&mut self, span: Span, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if span == Span::default() {
            return f(self);
        }
        let outer = self.ir.set_span(span);
        let out = f(self);
        self.ir.set_span(outer);
        out.map_err(|e| locate(&self.source, &self.ir, e, span))
    }

    /// The error for a name nothing in scope binds: [`Poisoned`] when its
    /// declaration failed.
    pub(crate) fn unknown(&self, name: &str) -> anyhow::Error {
        if self.poisoned.contains(name) {
            return Poisoned.into();
        }
        let names = self.scopes.iter().flat_map(|s| s.keys()).map(String::as_str);
        let near = crate::diag::similar(name, names);
        super::AtOp::unplaced(format!("unknown identifier `{name}`"))
            .label("not found in this scope")
            .instead(name, near)
            .into()
    }

    /// The state a failing statement may leave half changed.
    pub(crate) fn save(&self) -> Saved {
        Saved {
            block: self.block,
            scopes: self.scopes.len(),
            hoisted: self.hoisted.len(),
            ragged_iv: self.ragged_iv.clone(),
            trimmed_ivs: self.trimmed_ivs.len(),
            pending_ivs: self.pending_ivs.len(),
            fusing_regs: self.fusing_regs,
        }
    }

    fn restore(&mut self, saved: Saved) {
        self.block = saved.block;
        self.scopes.truncate(saved.scopes);
        self.hoisted.truncate(saved.hoisted);
        self.ragged_iv = saved.ragged_iv;
        self.trimmed_ivs.truncate(saved.trimmed_ivs);
        self.pending_ivs.truncate(saved.pending_ivs);
        self.fusing_regs = saved.fusing_regs;
    }

    /// Keeps a placed error for the end of the build, up to
    /// [`crate::diag::MAX_ERRORS`]; one with no place goes on up.
    fn report_error(&mut self, e: anyhow::Error) -> Result<()> {
        if e.is::<Poisoned>() {
            return Ok(());
        }
        let ds = e.downcast::<Diagnostics>()?;
        let room = crate::diag::MAX_ERRORS.saturating_sub(self.errors.len());
        self.errors.extend(ds.list().iter().take(room).cloned());
        Ok(())
    }

    /// Reports why `stmt` failed and puts the build back as it was before
    /// it, so the next statement builds; the names it declared are poisoned.
    pub(crate) fn recover(&mut self, e: anyhow::Error, stmt: &Stmt, saved: Saved) -> Result<()> {
        self.restore(saved);
        if let Stmt::Let { name, .. } | Stmt::Var { name, .. } = stmt {
            self.poisoned.insert(name.clone());
        }
        self.report_error(e)
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
            Binding::View(v) | Binding::Tile(v) | Binding::Frags(v) | Binding::Regs(v) | Binding::Tensor(v) => {
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
    /// tensor. The codegen counterpart is `MemVal::global.is_some()`.
    pub(crate) fn is_buffer(&self, v: ValueId) -> bool {
        self.ir
            .def_op(v)
            .is_some_and(|op| self.ir.kind(op).makes_buffer())
    }

    /// Whether a value is an unnamed temp whose buffer the consuming op may
    /// release. The codegen counterpart is `MemVal::owned`.
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
                    anyhow!("tile dim `{name}` must be a constant (literal or `@autotune` symbol)")
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

    /// A padded staging tile: the row pitch widened by
    /// [`crate::shape::stage_pad`].
    pub(crate) fn padded_ty(&self, elem: Scalar, shape: &[i64]) -> Type {
        let mut phys = shape.to_vec();
        let last = phys.len() - 1;
        phys[last] += crate::shape::stage_pad(elem.bytes().unwrap_or(4));
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

/// Whether a slice dimension provably never reaches past the source extent:
/// from the offset's divisor, or from the offset itself when it is the
/// constant `start`.
pub(crate) fn dim_in_bounds(extent: i64, size: i64, off_div: i64, start: Option<i64>) -> bool {
    if extent == DYN {
        return true;
    }
    if size == DYN {
        return false;
    }
    if start.is_some_and(|s| s >= 0 && s + size <= extent) {
        return true;
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
