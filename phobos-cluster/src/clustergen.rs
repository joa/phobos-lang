use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};
use phobos_lang::ast::{
    AssignOp, AttrArg, BinOp, Dim, Expr, Kernel, Scalar, Stmt, Sub, Type as AstType,
};

use crate::ir::{
    ClusterProgram, ClusterStmt, Coord, GridAxis, LeafKernel, ScalarDecl, SearchDim, SuperTile,
    TensorDecl,
};
use crate::tile::{AccessMode, DataType};

/// Compile a @cluster kernel to the parametric cluster IR.
pub fn compile(kernel: &Kernel) -> Result<ClusterProgram> {
    Analyzer::new(kernel)?.run()
}

fn data_type(s: Scalar) -> DataType {
    match s {
        Scalar::F16 => DataType::F16,
        Scalar::F32 => DataType::F32,
        Scalar::F64 => DataType::F64,
        Scalar::I32 => DataType::I32,
        Scalar::I64 => DataType::I64,
        Scalar::Bool => DataType::Bool,
    }
}

/// Cluster scale scalar value.
#[derive(Clone, Debug)]
enum ScalarValue {
    /// program_id(i)
    Pid(usize),
    /// program_id(i) * SUPER
    PidSuper(usize, String),
}

/// What a name binds to at cluster scale.
#[derive(Clone, Debug)]
enum Binding {
    Scalar(ScalarValue),
    /// Supertile view (let a = A[..]).
    Ref(SuperTile),
    /// The accumulator tile (var acc: tile<..>[..] = 0.0).
    Scratch,
    /// Cluster-loop iv; payload is the step's super sym.
    LoopVar(String),
    /// A leaf-internal (device-scale) loop iv -> used only by the single-leaf path.
    /// Payload is the dim the loop uses; a slice offset by it covers
    /// the whole axis, so the cluster never tiles that axis (see [`Coord::Full`]).
    DeviceLoop(String),
}

enum Statement {
    /// Placeholder at the scratch declaration; becomes the init compute.
    InitPlaceholder,
    Compute(Pending),
    Loop {
        var: String,
        dim: Dim,
        super_sym: String,
        body: Vec<Statement>,
    },
}

struct Pending {
    /// Reads collected from the value expression.
    ///
    /// Key is tensor index.
    reads: Vec<(usize, SuperTile)>,
    /// Direct-compute write target; None when the target is the scratch.
    target: Option<(usize, SuperTile, AccessMode)>,
    uses_scratch: bool,
}

struct Scratch {
    init: Expr,
}

/// The chain's C[<grid slice>] = <epilogue> store.
struct Define {
    tensor: usize,
    coords: Vec<Coord>,
    /// Super sym per axis (for the synthesized init leaf's slice).
    supers: Vec<String>,
    /// Top-level statement index, for the step leaf's store rewrite.
    stmt_idx: usize,
    /// The GEMM epilogue when the store is alpha*acc + beta*c_old rather than
    /// a bare = acc. None keeps the plain chain (zero-init, += acc step).
    epilogue: Option<Epilogue>,
}

/// A GEMM-shaped accumulator epilogue: C[..] = [alpha *] acc [+ [beta *] c_old]
/// where c_old is a prior load of the same output supertile.
///
/// Copy-elision keeps the accumulator on C's buffer, so the epilogue folds:
/// the step leaf accumulates alpha*acc (stored as alpha*acc + c_old, which
/// stays on the fused register-accumulator path with an implicit beta of 1),
/// and the init leaf seeds C with beta*c_old instead of zero-filling. The
/// running sum then lands on beta*C_orig + alpha*sum(dot) = alpha*acc + beta*C_orig.
struct Epilogue {
    /// The accumulator scratch var name.
    acc: String,
    /// Coefficient on acc (None is identity 1.0), applied per step.
    alpha: Option<Expr>,
    /// The prior-C term: Some((beta, c_old)) when the epilogue reads C back.
    /// beta is the coefficient (None is identity 1.0); c_old names the load.
    prev: Option<(Option<Expr>, String)>,
}

/// How to seed the accumulator output before the chain runs.
struct InitInfo {
    /// No init compute at all (beta is identity, so C keeps its original value).
    skip: bool,
    /// C's access mode in the init compute (Write to zero-fill, RMW to scale C).
    c_mode: AccessMode,
    /// Scalar indices the init compute carries (the beta coefficient's params).
    scalars: Vec<usize>,
}

struct Analyzer<'a> {
    kernel: &'a Kernel,
    super_dims: Vec<SearchDim>,
    super_set: HashSet<String>,
    tensors: Vec<TensorDecl>,
    tindex: HashMap<String, usize>,
    scalars: Vec<ScalarDecl>,
    symbols: HashMap<String, Binding>,
    /// Per-tensor, per-axis supertile sym discovered from slice extents.
    tensor_syms: Vec<Vec<Option<String>>>,
    /// Grid axes discovered from slice uses, indexed by pid.
    grid: Vec<Option<GridAxis>>,
    scratch: Option<Scratch>,
    define: Option<Define>,
    n_computes: usize,
}

impl<'a> Analyzer<'a> {
    fn new(kernel: &'a Kernel) -> Result<Self> {
        let mut super_dims = Vec::new();
        let mut super_set = HashSet::new();
        for attr in kernel.attrs.iter().filter(|a| a.name == "cluster") {
            for arg in &attr.args {
                match arg {
                    AttrArg::Search { name, choices } => {
                        super_set.insert(name.clone());
                        super_dims.push(SearchDim {
                            name: name.clone(),
                            choices: phobos_lang::ast::search_choices(choices),
                        });
                    }
                    _ => bail!("@cluster takes only search dims (`NAME in [..]`)"),
                }
            }
        }
        if super_dims.is_empty() {
            bail!("kernel '{}' has no @cluster search dims", kernel.name);
        }

        let autotune: HashSet<&str> = kernel
            .attrs
            .iter()
            .filter(|a| a.name == "autotune")
            .flat_map(|a| a.args.iter())
            .filter_map(|arg| match arg {
                AttrArg::Search { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        for d in &super_dims {
            if !autotune.contains(d.name.as_str()) {
                bail!(
                    "@cluster dim '{}' must also be an @autotune dim so the \
                     leaf kernel can bind it at device scale",
                    d.name
                );
            }
        }

        let mut tensors = Vec::new();
        let mut tindex = HashMap::new();
        let mut tsyms: Vec<Vec<Option<String>>> = Vec::new();
        let mut scalars = Vec::new();
        for (pos, p) in kernel.params.iter().enumerate() {
            match &p.ty {
                AstType::Tensor(scalar, dims) => {
                    tindex.insert(p.name.clone(), tensors.len());
                    tsyms.push(vec![None; dims.len()]);
                    tensors.push(TensorDecl {
                        name: p.name.clone(),
                        data_type: data_type(*scalar),
                        dims: dims.clone(),
                        super_syms: Vec::new(), // filled in finalize from tsyms
                        mode: AccessMode::Read, // outputs patched in finalize
                    });
                }
                AstType::Scalar(s) => scalars.push(ScalarDecl {
                    name: p.name.clone(),
                    data_type: data_type(*s),
                    param_pos: pos,
                }),
                AstType::Tile(..) => {
                    bail!(
                        "tile parameter '{}' is not supported under @cluster",
                        p.name
                    )
                }
            }
        }

        Ok(Analyzer {
            kernel,
            super_dims,
            super_set,
            tensors,
            tindex,
            scalars,
            symbols: HashMap::new(),
            tensor_syms: tsyms,
            grid: Vec::new(),
            scratch: None,
            define: None,
            n_computes: 0,
        })
    }

    fn run(self) -> Result<ClusterProgram> {
        if has_cluster_loop(&self.kernel.body, &self.super_set) {
            self.run_accumulator()
        } else {
            self.run_single_leaf()
        }
    }

    fn run_accumulator(mut self) -> Result<ClusterProgram> {
        let mut body = Vec::new();
        for (idx, stmt) in self.kernel.body.iter().enumerate() {
            if let Some(b) = self.stmt(stmt, idx, false)? {
                body.push(b);
            }
        }
        self.finalize(body)
    }

    fn run_single_leaf(mut self) -> Result<ClusterProgram> {
        let mut refs: HashMap<usize, SuperTile> = HashMap::new();
        let mut reads: HashSet<usize> = HashSet::new();
        let mut writes: HashSet<usize> = HashSet::new();
        self.walk_single(&self.kernel.body, &mut refs, &mut reads, &mut writes)?;
        self.finalize_single(refs, reads, writes)
    }

    fn walk_single(
        &mut self,
        stmts: &'a [Stmt],
        refs: &mut HashMap<usize, SuperTile>,
        reads: &mut HashSet<usize>,
        writes: &mut HashSet<usize>,
    ) -> Result<()> {
        for stmt in stmts {
            match stmt {
                Stmt::Let { name, ty, value } | Stmt::Var { name, ty, value } => {
                    self.scan_reads(value, refs, reads)?;
                    if matches!(ty, Some(AstType::Tile(..))) {
                        continue;
                    }
                    if let Some(sv) = self.classify_scalar(value)? {
                        self.symbols.insert(name.clone(), Binding::Scalar(sv));
                    }
                }
                Stmt::Assign { target, op, value } => {
                    self.scan_reads(value, refs, reads)?;
                    match target {
                        Expr::Index { base, subs } => {
                            let Expr::Var(t) = base.as_ref() else {
                                bail!("invalid assignment target under @cluster");
                            };
                            let Some(&ti) = self.tindex.get(t) else {
                                bail!("'{t}' is not a tensor parameter");
                            };
                            let r = self.classify_ref_leaf(ti, subs)?;
                            self.unify_ref(refs, ti, r)?;
                            writes.insert(ti);
                            if *op == AssignOp::Add {
                                reads.insert(ti);
                            }
                        }
                        // assignment to a tile var (running state) is leaf-internal
                        Expr::Var(_) => {}
                        _ => bail!("invalid assignment target under @cluster"),
                    }
                }
                Stmt::For {
                    var,
                    start,
                    end,
                    step,
                    body,
                } => {
                    if !matches!(start, Expr::Int(0)) {
                        bail!("device-level loops must start at 0");
                    }
                    if let Some(Expr::Var(s)) = step
                        && self.super_set.contains(s)
                    {
                        unreachable!("a cluster loop must route to the accumulator path");
                    }
                    let Expr::Var(d) = end else {
                        bail!(
                            "device-level loop bound must be a symbolic dim so its \
                             slices span a whole supertile axis"
                        );
                    };
                    self.symbols
                        .insert(var.clone(), Binding::DeviceLoop(d.clone()));
                    self.walk_single(body, refs, reads, writes)?;
                    self.symbols.remove(var);
                }
                Stmt::While { cond, body } => {
                    self.scan_reads(cond, refs, reads)?;
                    self.walk_single(body, refs, reads, writes)?;
                }
                Stmt::If { cond, then, r#else } => {
                    self.scan_reads(cond, refs, reads)?;
                    self.walk_single(then, refs, reads, writes)?;
                    if let Some(e) = r#else {
                        self.walk_single(e, refs, reads, writes)?;
                    }
                }
                Stmt::Expr(e) => self.scan_reads(e, refs, reads)?,
            }
        }
        Ok(())
    }

    /// Record every tensor-parameter read reachable from e.
    ///
    /// Tile-var reads are NOT tensor accesses, so they are ignored;
    /// that access was already captured where the tile var was bound.
    fn scan_reads(
        &mut self,
        e: &Expr,
        refs: &mut HashMap<usize, SuperTile>,
        reads: &mut HashSet<usize>,
    ) -> Result<()> {
        match e {
            Expr::Index { base, subs } => {
                if let Expr::Var(t) = base.as_ref()
                    && let Some(&ti) = self.tindex.get(t)
                {
                    let r = self.classify_ref_leaf(ti, subs)?;
                    self.unify_ref(refs, ti, r)?;
                    reads.insert(ti);
                } else {
                    self.scan_reads(base, refs, reads)?;
                    for s in subs {
                        match s {
                            Sub::Point(x) => self.scan_reads(x, refs, reads)?,
                            Sub::Range { start, end } => {
                                self.scan_reads(start, refs, reads)?;
                                self.scan_reads(end, refs, reads)?;
                            }
                            Sub::Span { start, len } => {
                                self.scan_reads(start, refs, reads)?;
                                self.scan_reads(len, refs, reads)?;
                            }
                            Sub::Full => {}
                        }
                    }
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                self.scan_reads(lhs, refs, reads)?;
                self.scan_reads(rhs, refs, reads)?;
            }
            Expr::Unary { rhs, .. } => self.scan_reads(rhs, refs, reads)?,
            Expr::Call { args, .. } => {
                for a in args {
                    self.scan_reads(a, refs, reads)?;
                }
            }
            Expr::Var(_) | Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) => {}
        }
        Ok(())
    }

    fn unify_ref(
        &self,
        refs: &mut HashMap<usize, SuperTile>,
        ti: usize,
        r: SuperTile,
    ) -> Result<()> {
        match refs.get(&ti) {
            None => {
                refs.insert(ti, r);
            }
            Some(prev) if *prev != r => bail!(
                "tensor '{}' is accessed at two different supertile coordinates in \
                 one leaf (cross-supertile access needs halo exchange, unsupported)",
                self.tensors[ti].name
            ),
            Some(_) => {}
        }
        Ok(())
    }

    fn classify_ref_leaf(&mut self, tensor: usize, subs: &[Sub]) -> Result<SuperTile> {
        let rank = self.tensors[tensor].dims.len();
        if subs.len() != rank {
            bail!(
                "slice of '{}' has {} subscripts, tensor has rank {rank}",
                self.tensors[tensor].name,
                subs.len()
            );
        }
        let mut coords = Vec::new();
        for (axis, sub) in subs.iter().enumerate() {
            let dim = self.tensors[tensor].dims[axis].clone();
            match sub {
                Sub::Full => {
                    let sym = self.axis_dim_sym(tensor, axis, &dim)?;
                    self.set_tensor_sym(tensor, axis, &sym)?;
                    coords.push(Coord::Full);
                }
                Sub::Span { start, len } => {
                    if let Some(ScalarValue::PidSuper(i, s)) = self.classify_scalar(start)? {
                        let Expr::Var(ls) = len else {
                            bail!("grid slice extent must be the @cluster dim '{s}'");
                        };
                        if ls != &s {
                            bail!("slice offset is scaled by '{s}' but its extent is '{ls}'");
                        }
                        self.set_tensor_sym(tensor, axis, &s)?;
                        self.note_grid(i, tensor, axis, &s)?;
                        coords.push(Coord::Grid(i));
                        continue;
                    }
                    if let Expr::Var(v) = start
                        && let Some(Binding::DeviceLoop(swept)) = self.symbols.get(v).cloned()
                    {
                        let sym = self.axis_dim_sym(tensor, axis, &dim)?;
                        if swept != sym {
                            bail!(
                                "device loop '{v}' sweeps '{swept}' but '{}' axis {axis} is \
                                 '{sym}', so the slice is not a whole supertile",
                                self.tensors[tensor].name
                            );
                        }
                        self.set_tensor_sym(tensor, axis, &sym)?;
                        coords.push(Coord::Full);
                        continue;
                    }
                    bail!(
                        "slice offset of '{}' (axis {axis}) is not supertile-aligned: expected \
                         `program_id(i) * SUPER`, a `:` full slice, or a device-loop var",
                        self.tensors[tensor].name
                    );
                }
                Sub::Point(_) | Sub::Range { .. } => bail!(
                    "only `start :+ len` spans and `:` full slices are supported under \
                     @cluster (slice of '{}')",
                    self.tensors[tensor].name
                ),
            }
        }
        Ok(SuperTile { tensor, coords })
    }

    fn axis_dim_sym(&self, tensor: usize, axis: usize, dim: &Dim) -> Result<String> {
        match dim {
            Dim::Sym(s) => Ok(s.clone()),
            Dim::Int(_) => bail!(
                "tensor '{}' axis {axis} has a literal size; a `:` or device-loop slice \
                 needs a symbolic dim to name the supertile",
                self.tensors[tensor].name
            ),
        }
    }

    fn finalize_grid(&mut self) -> Result<Vec<GridAxis>> {
        let mut grid = Vec::new();
        for (i, g) in self.grid.drain(..).enumerate() {
            match g {
                Some(g) => grid.push(g),
                None => bail!("program_id({i}) is never used to address a supertile"),
            }
        }
        if grid.is_empty() {
            bail!("kernel addresses no supertiles via program_id");
        }
        Ok(grid)
    }

    fn finalize_super_syms(&mut self) -> Result<()> {
        for (t, syms) in self.tensor_syms.iter().enumerate() {
            for (axis, s) in syms.iter().enumerate() {
                match s {
                    Some(s) => self.tensors[t].super_syms.push(s.clone()),
                    None => bail!(
                        "tensor '{}' axis {axis} is never sliced, so its supertile \
                         shape is unknown",
                        self.tensors[t].name
                    ),
                }
            }
        }
        Ok(())
    }

    fn finalize_single(
        mut self,
        refs: HashMap<usize, SuperTile>,
        reads: HashSet<usize>,
        writes: HashSet<usize>,
    ) -> Result<ClusterProgram> {
        let grid = self.finalize_grid()?;

        for (i, t) in self.tensors.iter().enumerate() {
            if !refs.contains_key(&i) {
                bail!(
                    "tensor '{}' is never accessed; every kernel parameter must map to a \
                     supertile",
                    t.name
                );
            }
        }
        self.finalize_super_syms()?;

        let mode_of = |ti: usize| match (reads.contains(&ti), writes.contains(&ti)) {
            (true, true) => AccessMode::RMW,
            (false, true) => AccessMode::Write,
            _ => AccessMode::Read,
        };
        for i in 0..self.tensors.len() {
            self.tensors[i].mode = mode_of(i);
        }

        // the single leaf is the whole kernel, reinterpreted at device scale
        let mut leaf = self.kernel.clone();
        leaf.attrs.retain(|a| a.name != "cluster");
        let modes = self
            .kernel
            .params
            .iter()
            .map(|p| match &p.ty {
                AstType::Tensor(..) => mode_of(self.tindex[&p.name]),
                _ => AccessMode::Read,
            })
            .collect();

        let args = (0..self.tensors.len())
            .map(|i| (refs[&i].clone(), mode_of(i)))
            .collect();
        let scalars = (0..self.scalars.len()).collect();
        let body = vec![ClusterStmt::Compute {
            leaf: 0,
            args,
            scalars,
        }];

        Ok(ClusterProgram {
            name: self.kernel.name.clone(),
            super_dims: self.super_dims,
            tensors: self.tensors,
            scalars: self.scalars,
            leaves: vec![LeafKernel {
                kernel: leaf,
                modes,
            }],
            grid,
            body,
        })
    }

    fn stmt(&mut self, stmt: &Stmt, idx: usize, in_loop: bool) -> Result<Option<Statement>> {
        match stmt {
            // tile declaration: the accumulator
            Stmt::Var {
                name,
                ty: Some(AstType::Tile(_, dims)),
                value,
            } => {
                if in_loop {
                    bail!("tile declarations inside a cluster loop are not supported");
                }
                if self.scratch.is_some() {
                    bail!("only one accumulator tile is supported under @cluster for now");
                }
                for d in dims {
                    let Dim::Sym(s) = d else {
                        bail!("accumulator tile dims must be @cluster dims, got a literal");
                    };
                    if !self.super_set.contains(s) {
                        bail!("accumulator tile dim '{s}' is not a @cluster dim");
                    }
                }
                if !matches!(value, Expr::Int(_) | Expr::Float(_)) {
                    bail!("accumulator tile must be initialized with a literal");
                }
                self.scratch = Some(Scratch {
                    init: value.clone(),
                });
                self.symbols.insert(name.clone(), Binding::Scratch);
                Ok(Some(Statement::InitPlaceholder))
            }
            Stmt::Let {
                ty: Some(AstType::Tile(..)),
                ..
            } => bail!("accumulator tiles must be declared with `var`, not `let`"),

            // scalar / ref bindings
            Stmt::Let { name, ty, value } | Stmt::Var { name, ty, value } => {
                if let Some(t) = ty
                    && !matches!(t, AstType::Scalar(_))
                {
                    bail!("unsupported declaration type for '{name}' under @cluster");
                }
                if let Some(sv) = self.classify_scalar(value)? {
                    self.symbols.insert(name.clone(), Binding::Scalar(sv));
                    return Ok(None);
                }
                if let Expr::Index { base, subs } = value
                    && let Expr::Var(t) = base.as_ref()
                    && let Some(&ti) = self.tindex.get(t)
                {
                    let r = self.classify_ref(ti, subs)?;
                    self.symbols.insert(name.clone(), Binding::Ref(r));
                    return Ok(None);
                }
                bail!(
                    "`{name} = ...` is not interpretable at cluster scale \
                     (expected program_id arithmetic or a supertile slice)"
                );
            }

            Stmt::For {
                var,
                start,
                end,
                step,
                body,
            } => {
                if in_loop {
                    bail!("nested cluster-level loops are not supported yet");
                }
                if !matches!(start, Expr::Int(0)) {
                    bail!("cluster-level loops must start at 0");
                }
                let Some(Expr::Var(s)) = step else {
                    bail!("cluster-level loops must step by a @cluster dim");
                };
                if !self.super_set.contains(s) {
                    bail!("loop step '{s}' is not a @cluster dim");
                }
                let dim = match end {
                    Expr::Var(d) => Dim::Sym(d.clone()),
                    Expr::Int(n) => Dim::Int(*n),
                    _ => bail!("cluster-level loop bound must be a symbolic dim or literal"),
                };
                self.symbols
                    .insert(var.clone(), Binding::LoopVar(s.clone()));
                let mut inner = Vec::new();
                for st in body {
                    if let Some(b) = self.stmt(st, idx, true)? {
                        inner.push(b);
                    }
                }
                Ok(Some(Statement::Loop {
                    var: var.clone(),
                    dim,
                    super_sym: s.clone(),
                    body: inner,
                }))
            }

            Stmt::Assign { target, op, value } => self.assign(target, *op, value, idx, in_loop),

            Stmt::While { .. } | Stmt::If { .. } | Stmt::Expr(_) => {
                bail!("`while`/`if`/expression statements are not supported under @cluster yet")
            }
        }
    }

    fn assign(
        &mut self,
        target: &Expr,
        op: AssignOp,
        value: &Expr,
        idx: usize,
        in_loop: bool,
    ) -> Result<Option<Statement>> {
        match target {
            // accumulator update: the rmw chain step
            Expr::Var(n) if matches!(self.symbols.get(n), Some(Binding::Scratch)) => {
                if op != AssignOp::Add {
                    bail!("the accumulator only supports `+=` updates under @cluster");
                }
                let mut reads = Vec::new();
                self.collect_reads(value, &mut reads)?;
                self.n_computes += 1;
                Ok(Some(Statement::Compute(Pending {
                    reads,
                    target: None,
                    uses_scratch: true,
                })))
            }
            Expr::Var(n) => bail!("'{n}' is not assignable at cluster scale"),

            Expr::Index { base, subs } => {
                let Expr::Var(t) = base.as_ref() else {
                    bail!("invalid assignment target under @cluster");
                };
                let Some(&ti) = self.tindex.get(t) else {
                    bail!("'{t}' is not a tensor parameter");
                };

                // the chain's final store: C[<grid slice>] = acc
                if op == AssignOp::Set
                    && let Expr::Var(v) = value
                    && matches!(self.symbols.get(v), Some(Binding::Scratch))
                {
                    if in_loop {
                        bail!("the accumulator must be stored outside the cluster loop");
                    }
                    if self.define.is_some() {
                        bail!("the accumulator is stored more than once");
                    }
                    let r = self.classify_ref(ti, subs)?;
                    let mut supers = Vec::new();
                    for (sub, coord) in subs.iter().zip(&r.coords) {
                        if !matches!(coord, Coord::Grid(_)) {
                            bail!(
                                "output supertile coordinates must be grid vars \
                                 (loop-varying accumulator stores are not supported)"
                            );
                        }
                        let Sub::Span {
                            len: Expr::Var(s), ..
                        } = sub
                        else {
                            unreachable!("classify_ref accepts only `:+ SYM` spans");
                        };
                        supers.push(s.clone());
                    }
                    self.tensors[ti].mode = AccessMode::Write; // init leaf overwrites
                    self.define = Some(Define {
                        tensor: ti,
                        coords: r.coords,
                        supers,
                        stmt_idx: idx,
                        epilogue: None,
                    });
                    return Ok(None);
                }

                // the chain's GEMM epilogue: C[<grid slice>] = f(acc, C_old, scalars)
                if op == AssignOp::Set && self.uses_scratch(value) {
                    if in_loop {
                        bail!("the accumulator must be stored outside the cluster loop");
                    }
                    if self.define.is_some() {
                        bail!("the accumulator is stored more than once");
                    }
                    let r = self.classify_ref(ti, subs)?;
                    let epilogue = self.parse_epilogue(value, &r)?;
                    let mut supers = Vec::new();
                    for (sub, coord) in subs.iter().zip(&r.coords) {
                        if !matches!(coord, Coord::Grid(_)) {
                            bail!(
                                "output supertile coordinates must be grid vars \
                                 (loop-varying accumulator stores are not supported)"
                            );
                        }
                        let Sub::Span {
                            len: Expr::Var(s), ..
                        } = sub
                        else {
                            unreachable!("classify_ref accepts only `:+ SYM` spans");
                        };
                        supers.push(s.clone());
                    }
                    // C is read back (beta*c_old) so it must be loaded, not zeroed.
                    self.tensors[ti].mode = match &epilogue.prev {
                        Some(_) => AccessMode::RMW,
                        None => AccessMode::Write,
                    };
                    self.define = Some(Define {
                        tensor: ti,
                        coords: r.coords,
                        supers,
                        stmt_idx: idx,
                        epilogue: Some(epilogue),
                    });
                    return Ok(None);
                }

                // direct compute: Z[$slice] = expr or Z[$slice] += expr
                let r = self.classify_ref(ti, subs)?;
                let mode = match op {
                    AssignOp::Set => AccessMode::Write,
                    AssignOp::Add => AccessMode::RMW,
                };
                self.tensors[ti].mode = mode;
                let mut reads = Vec::new();
                self.collect_reads(value, &mut reads)?;
                self.n_computes += 1;
                Ok(Some(Statement::Compute(Pending {
                    reads,
                    target: Some((ti, r, mode)),
                    uses_scratch: false,
                })))
            }
            _ => bail!("invalid assignment target under @cluster"),
        }
    }

    fn classify_scalar(&self, e: &Expr) -> Result<Option<ScalarValue>> {
        match e {
            Expr::Call { callee, args } if callee == "program_id" => {
                let [Expr::Int(i)] = args.as_slice() else {
                    bail!("program_id takes one literal axis");
                };
                Ok(Some(ScalarValue::Pid(*i as usize)))
            }
            Expr::Var(n) => Ok(match self.symbols.get(n) {
                Some(Binding::Scalar(sv)) => Some(sv.clone()),
                _ => None,
            }),
            Expr::Binary {
                op: BinOp::Mul,
                lhs,
                rhs,
            } => {
                // pid * SUPER or SUPER * pid
                for (a, b) in [(lhs, rhs), (rhs, lhs)] {
                    if let Expr::Var(s) = b.as_ref()
                        && self.super_set.contains(s)
                        && let Some(ScalarValue::Pid(i)) = self.classify_scalar(a)?
                    {
                        return Ok(Some(ScalarValue::PidSuper(i, s.clone())));
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn classify_ref(&mut self, tensor: usize, subs: &[Sub]) -> Result<SuperTile> {
        let rank = self.tensors[tensor].dims.len();
        if subs.len() != rank {
            bail!(
                "slice of '{}' has {} subscripts, tensor has rank {rank}",
                self.tensors[tensor].name,
                subs.len()
            );
        }
        let mut coords = Vec::new();
        for (axis, sub) in subs.iter().enumerate() {
            let Sub::Span { start, len } = sub else {
                bail!(
                    "only `start :+ SUPER` spans are supported under @cluster \
                     (slice of '{}')",
                    self.tensors[tensor].name
                );
            };
            let Expr::Var(s) = len else {
                bail!("slice extent must be a @cluster dim");
            };
            if !self.super_set.contains(s) {
                bail!("slice extent '{s}' is not a @cluster dim");
            }
            self.set_tensor_sym(tensor, axis, s)?;

            if let Some(ScalarValue::PidSuper(i, s2)) = self.classify_scalar(start)? {
                if s2 != *s {
                    bail!("slice offset is scaled by '{s2}' but its extent is '{s}'");
                }
                self.note_grid(i, tensor, axis, s)?;
                coords.push(Coord::Grid(i));
                continue;
            }
            if let Expr::Var(v) = start
                && let Some(Binding::LoopVar(step)) = self.symbols.get(v)
            {
                if step != s {
                    bail!(
                        "loop '{v}' steps by '{step}' but the slice extent is '{s}', \
                         so offsets would not be supertile-aligned"
                    );
                }
                coords.push(Coord::Loop(v.clone()));
                continue;
            }
            bail!(
                "slice offset of '{}' (axis {axis}) is not supertile-aligned: \
                 expected `program_id(i) * {s}` or a cluster-loop var stepping by {s}",
                self.tensors[tensor].name
            );
        }
        Ok(SuperTile { tensor, coords })
    }

    fn set_tensor_sym(&mut self, tensor: usize, axis: usize, sym: &str) -> Result<()> {
        match &self.tensor_syms[tensor][axis] {
            None => self.tensor_syms[tensor][axis] = Some(sym.to_string()),
            Some(prev) if prev != sym => bail!(
                "tensor '{}' axis {axis} is sliced with conflicting supertile \
                 dims '{prev}' and '{sym}'",
                self.tensors[tensor].name
            ),
            Some(_) => {}
        }
        Ok(())
    }

    fn note_grid(&mut self, pid: usize, tensor: usize, axis: usize, sym: &str) -> Result<()> {
        let dim = self.tensors[tensor].dims[axis].clone();
        if self.grid.len() <= pid {
            self.grid.resize_with(pid + 1, || None);
        }
        match &self.grid[pid] {
            None => {
                self.grid[pid] = Some(GridAxis {
                    pid,
                    dim,
                    super_sym: sym.to_string(),
                });
                Ok(())
            }
            Some(g) => {
                if g.super_sym != sym || g.dim != dim {
                    bail!(
                        "program_id({pid}) is used with conflicting supertile shapes: \
                         {:?}/{} vs {:?}/{}",
                        g.dim,
                        g.super_sym,
                        dim,
                        sym
                    );
                }
                Ok(())
            }
        }
    }

    fn collect_reads(&mut self, e: &Expr, out: &mut Vec<(usize, SuperTile)>) -> Result<()> {
        match e {
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) => Ok(()),
            Expr::Var(n) => match self.symbols.get(n).cloned() {
                Some(Binding::Ref(r)) => {
                    out.push((r.tensor, r));
                    Ok(())
                }
                Some(Binding::Scratch) => bail!(
                    "the accumulator can only be updated with `+=` and stored once \
                     ('{n}' read inside an expression)"
                ),
                Some(Binding::Scalar(_))
                | Some(Binding::LoopVar(_))
                | Some(Binding::DeviceLoop(_)) => {
                    bail!("grid/loop scalar '{n}' cannot appear inside a supertile computation")
                }
                None => bail!("unknown identifier '{n}' in a supertile computation"),
            },
            Expr::Index { base, subs } => {
                let Expr::Var(t) = base.as_ref() else {
                    bail!("unsupported indexing in a supertile computation");
                };
                let Some(&ti) = self.tindex.get(t) else {
                    bail!("'{t}' is not a tensor parameter");
                };
                let r = self.classify_ref(ti, subs)?;
                out.push((ti, r));
                Ok(())
            }
            Expr::Binary { lhs, rhs, .. } => {
                self.collect_reads(lhs, out)?;
                self.collect_reads(rhs, out)
            }
            Expr::Unary { rhs, .. } => self.collect_reads(rhs, out),
            Expr::Call { callee, args } if callee == "dot" => {
                for a in args {
                    self.collect_reads(a, out)?;
                }
                Ok(())
            }
            Expr::Call { callee, .. } => {
                bail!("call to '{callee}' is not supported in a supertile computation")
            }
        }
    }

    /// Whether e reads the accumulator scratch tile anywhere.
    fn uses_scratch(&self, e: &Expr) -> bool {
        match e {
            Expr::Var(n) => matches!(self.symbols.get(n), Some(Binding::Scratch)),
            Expr::Binary { lhs, rhs, .. } => self.uses_scratch(lhs) || self.uses_scratch(rhs),
            Expr::Unary { rhs, .. } => self.uses_scratch(rhs),
            Expr::Call { args, .. } => args.iter().any(|a| self.uses_scratch(a)),
            _ => false,
        }
    }

    /// Parse an accumulator epilogue [alpha *] acc [+ [beta *] c_old], where
    /// c_old is a prior load of the output supertile target. The store's
    /// scalars stay out of the cluster IR: they only surface in the leaves.
    fn parse_epilogue(&self, value: &Expr, target: &SuperTile) -> Result<Epilogue> {
        if let Expr::Binary {
            op: BinOp::Add,
            lhs,
            rhs,
        } = value
        {
            // one side carries acc, the other the prior-C load
            let (acc_side, prev_side) = if self.uses_scratch(lhs) {
                (lhs.as_ref(), rhs.as_ref())
            } else {
                (rhs.as_ref(), lhs.as_ref())
            };
            let (alpha, acc) = self.peel_acc(acc_side)?;
            let (beta, c_old) = self.peel_prev(prev_side, target)?;
            return Ok(Epilogue {
                acc,
                alpha,
                prev: Some((beta, c_old)),
            });
        }
        let (alpha, acc) = self.peel_acc(value)?;
        Ok(Epilogue {
            acc,
            alpha,
            prev: None,
        })
    }

    /// Peel acc or SCALAR * acc into (coefficient, acc name).
    fn peel_acc(&self, e: &Expr) -> Result<(Option<Expr>, String)> {
        if let Expr::Var(n) = e
            && matches!(self.symbols.get(n), Some(Binding::Scratch))
        {
            return Ok((None, n.clone()));
        }
        if let Expr::Binary {
            op: BinOp::Mul,
            lhs,
            rhs,
        } = e
        {
            for (atom, coeff) in [(lhs, rhs), (rhs, lhs)] {
                if let Expr::Var(n) = atom.as_ref()
                    && matches!(self.symbols.get(n), Some(Binding::Scratch))
                {
                    self.check_invariant(coeff)?;
                    return Ok((Some((**coeff).clone()), n.clone()));
                }
            }
        }
        bail!("the accumulator term of a GEMM epilogue must be `acc` or `SCALAR * acc`");
    }

    /// Peel c_old or SCALAR * c_old into (coefficient, c_old name), checking
    /// that c_old is a prior load of the same supertile the store targets.
    fn peel_prev(&self, e: &Expr, target: &SuperTile) -> Result<(Option<Expr>, String)> {
        let (coeff, name) = match e {
            Expr::Var(n) => (None, n.clone()),
            Expr::Binary {
                op: BinOp::Mul,
                lhs,
                rhs,
            } => {
                let mut found = None;
                for (atom, coeff) in [(lhs, rhs), (rhs, lhs)] {
                    if let Expr::Var(n) = atom.as_ref()
                        && matches!(self.symbols.get(n), Some(Binding::Ref(_)))
                    {
                        self.check_invariant(coeff)?;
                        found = Some((Some((**coeff).clone()), n.clone()));
                        break;
                    }
                }
                found.ok_or_else(|| {
                    anyhow::anyhow!(
                        "the second term of a GEMM epilogue must be `c_old` or `SCALAR * c_old`"
                    )
                })?
            }
            _ => bail!("the second term of a GEMM epilogue must be `c_old` or `SCALAR * c_old`"),
        };
        match self.symbols.get(&name) {
            Some(Binding::Ref(r)) if r == target => Ok((coeff, name)),
            Some(Binding::Ref(_)) => bail!(
                "GEMM epilogue reads '{name}' at a different supertile than it writes \
                 (cross-supertile access needs halo exchange, unsupported)"
            ),
            _ => bail!("GEMM epilogue term '{name}' is not a prior load of the output supertile"),
        }
    }

    /// Reject epilogue coefficients that are not loop-invariant scalar arithmetic
    /// (each alpha*acc step and the one-shot beta*c_old init need it constant).
    fn check_invariant(&self, e: &Expr) -> Result<()> {
        let mut idxs = Vec::new();
        self.collect_scalar_refs(e, &mut idxs)
    }

    /// Collect the scalar-parameter indices an epilogue coefficient references,
    /// in first-appearance order, rejecting any non-scalar term.
    fn collect_scalar_refs(&self, e: &Expr, out: &mut Vec<usize>) -> Result<()> {
        match e {
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) => Ok(()),
            Expr::Var(n) => match self.scalars.iter().position(|s| &s.name == n) {
                Some(i) => {
                    if !out.contains(&i) {
                        out.push(i);
                    }
                    Ok(())
                }
                None => bail!("epilogue coefficient '{n}' is not a scalar parameter"),
            },
            Expr::Binary { lhs, rhs, .. } => {
                self.collect_scalar_refs(lhs, out)?;
                self.collect_scalar_refs(rhs, out)
            }
            Expr::Unary { rhs, .. } => self.collect_scalar_refs(rhs, out),
            _ => bail!("epilogue coefficients must be arithmetic over scalar parameters"),
        }
    }

    fn finalize(mut self, body: Vec<Statement>) -> Result<ClusterProgram> {
        let grid = self.finalize_grid()?;

        if self.n_computes == 0 {
            bail!("kernel performs no supertile computation");
        }
        if self.n_computes > 1 {
            bail!(
                "kernels with multiple compute statements are not supported under \
                 @cluster yet (found {})",
                self.n_computes
            );
        }
        if self.scratch.is_some() && self.define.is_none() {
            bail!("the accumulator tile is never stored to an output tensor");
        }

        self.finalize_super_syms()?;

        // The step compute carries only the kernel's own scalars; the init leaf
        // may append its own (beta) decls below, so snapshot the count first.
        let step_scalars: Vec<usize> = (0..self.scalars.len()).collect();

        let scratch = self.scratch.take();
        let define = self.define.take();

        // step leaf: the kernel itself, reinterpreted at device scale
        let mut step = self.kernel.clone();
        step.attrs.retain(|a| a.name != "cluster");
        if let Some(d) = &define {
            rewrite_step_store(&mut step, d);
        }
        let out_tensor = define
            .as_ref()
            .map(|d| (d.tensor, AccessMode::RMW))
            .or_else(|| find_target(&body));
        let step_modes = (0..self.tensors.len())
            .map(|i| match out_tensor {
                Some((t, m)) if t == i => m,
                _ => AccessMode::Read,
            })
            .collect();
        let mut leaves = vec![LeafKernel {
            kernel: step,
            modes: step_modes,
        }];

        // init leaf: how the output supertile is seeded before the chain runs
        let mut init = InitInfo {
            skip: false,
            c_mode: AccessMode::Write,
            scalars: Vec::new(),
        };
        if let (Some(scratch), Some(d)) = (&scratch, &define) {
            match &d.epilogue {
                // plain = acc or = alpha*acc: zero-fill (the scratch literal)
                None
                | Some(Epilogue { prev: None, .. })
                | Some(Epilogue {
                    prev: Some((None, _)),
                    ..
                }) => match &d.epilogue {
                    // beta identity (+ c_old): C keeps its original value, no init
                    Some(Epilogue {
                        prev: Some((None, _)),
                        ..
                    }) => {
                        init.skip = true;
                        init.c_mode = AccessMode::RMW;
                    }
                    _ => leaves.push(LeafKernel {
                        kernel: self.zero_init_leaf(scratch, d),
                        modes: vec![AccessMode::Write],
                    }),
                },
                // + beta*c_old: seed C with beta*C_orig (reads C back)
                Some(Epilogue {
                    prev: Some((Some(beta), c_old)),
                    ..
                }) => {
                    let (kernel, modes, scalars) = self.beta_init_leaf(d, beta, c_old);
                    leaves.push(LeafKernel { kernel, modes });
                    init.c_mode = AccessMode::RMW;
                    init.scalars = scalars;
                }
            }
        }

        let body = lower_body(body, &self.tensors, define.as_ref(), &step_scalars, &init)?;

        Ok(ClusterProgram {
            name: self.kernel.name.clone(),
            super_dims: self.super_dims,
            tensors: self.tensors,
            scalars: self.scalars,
            leaves,
            grid,
            body,
        })
    }

    /// The grid slice [p0*S0 :+ S0, ..] of the output supertile, plus the
    /// let p{pid} = program_id(pid) bindings its offsets need.
    fn output_slice(&self, d: &Define) -> (Vec<Stmt>, Vec<Sub>) {
        let mut lets = Vec::new();
        let mut declared = HashSet::new();
        let mut subs = Vec::new();
        for (coord, sym) in d.coords.iter().zip(&d.supers) {
            let Coord::Grid(pid) = coord else {
                unreachable!("define coords are validated as grid vars");
            };
            let p = format!("p{pid}");
            if declared.insert(*pid) {
                lets.push(Stmt::Let {
                    name: p.clone(),
                    ty: None,
                    value: Expr::Call {
                        callee: "program_id".into(),
                        args: vec![Expr::Int(*pid as i64)],
                    },
                });
            }
            subs.push(Sub::Span {
                start: Expr::Binary {
                    op: BinOp::Mul,
                    lhs: Box::new(Expr::Var(p)),
                    rhs: Box::new(Expr::Var(sym.clone())),
                },
                len: Expr::Var(sym.clone()),
            });
        }
        (lets, subs)
    }

    fn init_attrs(&self) -> Vec<phobos_lang::ast::Attribute> {
        self.kernel
            .attrs
            .iter()
            .filter(|a| a.name == "autotune" || a.name == "launch")
            .cloned()
            .collect()
    }

    /// Synthesize the plain chain's init leaf:
    /// kernel {name}_init(C: ..) { let p0 = program_id(0); ..; C[p0*S :+ S, ..] = <init> }
    fn zero_init_leaf(&self, scratch: &Scratch, d: &Define) -> Kernel {
        let (mut body, subs) = self.output_slice(d);
        body.push(Stmt::Assign {
            target: Expr::Index {
                base: Box::new(Expr::Var(self.tensors[d.tensor].name.clone())),
                subs,
            },
            op: AssignOp::Set,
            value: scratch.init.clone(),
        });
        Kernel {
            attrs: self.init_attrs(),
            name: format!("{}_init", self.kernel.name),
            params: vec![self.kernel.params[d.tensor].clone()],
            body,
        }
    }

    /// Synthesize the GEMM chain's init leaf, which seeds C with beta*c_old
    /// (folding the epilogue's prior-C term out of the per-step accumulation):
    /// kernel {name}_init(C: .., <beta scalars>) { ..; let c_old = C[..]; C[..] = beta*c_old }
    ///
    /// The leaf's scalar params get fresh decls at local positions (the pod
    /// marshals a leaf's args by dense parameter index), returned as the init
    /// compute's scalar list.
    fn beta_init_leaf(
        &mut self,
        d: &Define,
        beta: &Expr,
        c_old: &str,
    ) -> (Kernel, Vec<AccessMode>, Vec<usize>) {
        let mut orig_idxs = Vec::new();
        self.collect_scalar_refs(beta, &mut orig_idxs)
            .expect("epilogue coefficients were validated during parse");

        let cname = self.tensors[d.tensor].name.clone();
        let mut params = vec![self.kernel.params[d.tensor].clone()];
        let mut modes = vec![AccessMode::RMW];
        let mut scalars = Vec::new();
        for oi in orig_idxs {
            let (name, data_type, orig_pos) = {
                let s = &self.scalars[oi];
                (s.name.clone(), s.data_type, s.param_pos)
            };
            let param_pos = params.len();
            params.push(self.kernel.params[orig_pos].clone());
            modes.push(AccessMode::Read);
            scalars.push(self.scalars.len());
            self.scalars.push(ScalarDecl {
                name,
                data_type,
                param_pos,
            });
        }

        let (mut body, subs) = self.output_slice(d);
        body.push(Stmt::Let {
            name: c_old.to_string(),
            ty: None,
            value: Expr::Index {
                base: Box::new(Expr::Var(cname.clone())),
                subs: subs.clone(),
            },
        });
        body.push(Stmt::Assign {
            target: Expr::Index {
                base: Box::new(Expr::Var(cname)),
                subs,
            },
            op: AssignOp::Set,
            value: Expr::Binary {
                op: BinOp::Mul,
                lhs: Box::new(beta.clone()),
                rhs: Box::new(Expr::Var(c_old.to_string())),
            },
        });

        let kernel = Kernel {
            attrs: self.init_attrs(),
            name: format!("{}_init", self.kernel.name),
            params,
            body,
        };
        (kernel, modes, scalars)
    }
}

/// Rewrite the step leaf's store so each launch contributes one k-chunk.
///
/// A plain C[..] = acc flips to += acc. A GEMM epilogue keeps the fused
/// register-accumulator store shape C[..] = alpha*acc + c_old (an implicit
/// beta of 1); the real beta is applied once by the init leaf. alpha*acc
/// with no prior-C term accumulates as += alpha*acc.
fn rewrite_step_store(step: &mut Kernel, d: &Define) {
    let Stmt::Assign { op, value, .. } = &mut step.body[d.stmt_idx] else {
        unreachable!("define index points at the accumulator store");
    };
    let Some(epi) = &d.epilogue else {
        *op = AssignOp::Add;
        return;
    };
    let acc_term = match &epi.alpha {
        Some(a) => Expr::Binary {
            op: BinOp::Mul,
            lhs: Box::new(a.clone()),
            rhs: Box::new(Expr::Var(epi.acc.clone())),
        },
        None => Expr::Var(epi.acc.clone()),
    };
    match &epi.prev {
        Some((_, c_old)) => {
            *op = AssignOp::Set;
            *value = Expr::Binary {
                op: BinOp::Add,
                lhs: Box::new(acc_term),
                rhs: Box::new(Expr::Var(c_old.clone())),
            };
        }
        None => {
            *op = AssignOp::Add;
            *value = acc_term;
        }
    }
}

fn has_cluster_loop(body: &[Stmt], super_set: &HashSet<String>) -> bool {
    body.iter().any(|s| match s {
        Stmt::For { step, body, .. } => {
            matches!(step, Some(Expr::Var(v)) if super_set.contains(v))
                || has_cluster_loop(body, super_set)
        }
        Stmt::While { body, .. } => has_cluster_loop(body, super_set),
        Stmt::If { then, r#else, .. } => {
            has_cluster_loop(then, super_set)
                || r#else
                    .as_deref()
                    .is_some_and(|e| has_cluster_loop(e, super_set))
        }
        _ => false,
    })
}

fn find_target(body: &[Statement]) -> Option<(usize, AccessMode)> {
    body.iter().find_map(|b| match b {
        Statement::Compute(p) => p.target.as_ref().map(|(t, _, m)| (*t, *m)),
        Statement::Loop { body, .. } => find_target(body),
        Statement::InitPlaceholder => None,
    })
}

fn lower_body(
    body: Vec<Statement>,
    tensors: &[TensorDecl],
    define: Option<&Define>,
    step_scalars: &[usize],
    init: &InitInfo,
) -> Result<Vec<ClusterStmt>> {
    let mut out = Vec::new();
    for b in body {
        match b {
            Statement::InitPlaceholder => {
                if init.skip {
                    continue; // beta identity: C keeps its original value
                }
                let d = define.expect("placeholder implies a define (validated)");
                out.push(ClusterStmt::Compute {
                    leaf: 1,
                    args: vec![(
                        SuperTile {
                            tensor: d.tensor,
                            coords: d.coords.clone(),
                        },
                        init.c_mode,
                    )],
                    scalars: init.scalars.clone(),
                });
            }
            Statement::Compute(p) => out.push(finalize_compute(p, tensors, define, step_scalars)?),
            Statement::Loop {
                var,
                dim,
                super_sym,
                body,
            } => out.push(ClusterStmt::Loop {
                var,
                dim,
                super_sym,
                body: lower_body(body, tensors, define, step_scalars, init)?,
            }),
        }
    }
    Ok(out)
}

fn finalize_compute(
    p: Pending,
    tensors: &[TensorDecl],
    define: Option<&Define>,
    step_scalars: &[usize],
) -> Result<ClusterStmt> {
    let mut args = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        // every read of this tensor in the compute must hit the same supertile
        let mut reads = p.reads.iter().filter(|(ti, _)| *ti == i).map(|(_, r)| r);
        let read = reads.next();
        if let Some(first) = read
            && reads.any(|r| r != first)
        {
            bail!(
                "tensor '{}' is read at multiple supertile coordinates in one \
                 compute (cross-supertile access needs halo exchange, unsupported)",
                t.name
            );
        }

        if let Some((ti, r, mode)) = &p.target
            && *ti == i
        {
            match read {
                Some(rr) if rr != r => bail!(
                    "tensor '{}' is read and written at different supertile \
                     coordinates in one compute",
                    t.name
                ),
                // read+write of the same supertile is read-modify-write
                Some(_) => args.push((r.clone(), AccessMode::RMW)),
                None => args.push((r.clone(), *mode)),
            }
            continue;
        }
        if p.uses_scratch
            && let Some(d) = define
            && d.tensor == i
        {
            // the accumulator, copy-elided onto the output supertile
            args.push((
                SuperTile {
                    tensor: i,
                    coords: d.coords.clone(),
                },
                AccessMode::RMW,
            ));
            continue;
        }
        match read {
            Some(r) => args.push((r.clone(), AccessMode::Read)),
            None => bail!(
                "tensor '{}' is not used in the compute; every kernel parameter \
                 must map to exactly one supertile per compute",
                t.name
            ),
        }
    }
    Ok(ClusterStmt::Compute {
        leaf: 0,
        args,
        scalars: step_scalars.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use phobos_lang::ast::{AssignOp, BinOp, Dim, Expr, Kernel, Stmt};

    use super::compile;
    use crate::ir::{ClusterStmt, Coord, SuperTile};
    use crate::tile::AccessMode;

    const MATMUL: &str = r#"
@cluster(TILE_M in [4096, 16384], TILE_N in [4096, 16384], TILE_K in [4096, 16384])
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
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
}"#;

    const ADD: &str = r#"
@cluster(BLOCK in [1048576, 16777216])
@autotune(BLOCK in [16, 4096])
kernel add(a: tensor<f32>[N], b: tensor<f32>[N], c: tensor<f32>[N]) {
    let base = program_id(0) * BLOCK
    c[base :+ BLOCK] = a[base :+ BLOCK] + b[base :+ BLOCK]
}"#;

    fn first(src: &str) -> Kernel {
        phobos_lang::parse(src).unwrap().remove(0)
    }

    /// Mirrors phobos-lang's emit_mlir test harness: device-compile a leaf
    /// and run the MLIR verifier on the emitted module.
    fn verify_leaf(k: &Kernel) -> String {
        use melior::{
            Context,
            dialect::DialectRegistry,
            ir::{Location, Module, operation::OperationLike},
            utility::register_all_dialects,
        };
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        let module = Module::new(Location::unknown(&context));
        let base = phobos_base::context::Context::default();
        phobos_lang::codegen::emit(&base, std::slice::from_ref(k), &context, &module).unwrap();
        let text = module.as_operation().to_string();
        assert!(module.as_operation().verify(), "invalid module:\n{text}");
        text
    }

    #[test]
    fn matmul_cluster_ir() {
        let p = compile(&first(MATMUL)).unwrap();

        assert_eq!(p.name, "matmul");
        assert_eq!(p.super_dims.len(), 3);
        // two-value search dims expand to doubling choices
        assert_eq!(p.super_dims[0].choices, vec![4096, 8192, 16384]);

        // grid: (M / TILE_M, N / TILE_N)
        assert_eq!(p.grid.len(), 2);
        assert_eq!(p.grid[0].dim, Dim::Sym("M".into()));
        assert_eq!(p.grid[0].super_sym, "TILE_M");
        assert_eq!(p.grid[1].dim, Dim::Sym("N".into()));
        assert_eq!(p.grid[1].super_sym, "TILE_N");

        // tensors: A read, B read, C pure output (init leaf overwrites)
        let modes: Vec<_> = p.tensors.iter().map(|t| t.mode).collect();
        assert_eq!(
            modes,
            vec![AccessMode::Read, AccessMode::Read, AccessMode::Write]
        );
        assert_eq!(p.tensors[0].super_syms, vec!["TILE_M", "TILE_K"]);
        assert_eq!(p.tensors[1].super_syms, vec!["TILE_K", "TILE_N"]);
        assert_eq!(p.tensors[2].super_syms, vec!["TILE_M", "TILE_N"]);

        // body: init compute, then the kt chain
        assert_eq!(p.body.len(), 2);
        let ClusterStmt::Compute { leaf: 1, args, .. } = &p.body[0] else {
            panic!("expected init compute, got {:?}", p.body[0]);
        };
        assert_eq!(
            args[0],
            (
                SuperTile {
                    tensor: 2,
                    coords: vec![Coord::Grid(0), Coord::Grid(1)]
                },
                AccessMode::Write
            )
        );
        let ClusterStmt::Loop {
            var,
            dim,
            super_sym,
            body,
        } = &p.body[1]
        else {
            panic!("expected cluster loop, got {:?}", p.body[1]);
        };
        assert_eq!(var, "kt");
        assert_eq!(*dim, Dim::Sym("K".into()));
        assert_eq!(super_sym, "TILE_K");

        // chain step: A(pm, kt) read, B(kt, pn) read, C(pm, pn) rmw (acc elided)
        let ClusterStmt::Compute { leaf: 0, args, .. } = &body[0] else {
            panic!("expected step compute, got {:?}", body[0]);
        };
        assert_eq!(
            args[0],
            (
                SuperTile {
                    tensor: 0,
                    coords: vec![Coord::Grid(0), Coord::Loop("kt".into())]
                },
                AccessMode::Read
            )
        );
        assert_eq!(
            args[1],
            (
                SuperTile {
                    tensor: 1,
                    coords: vec![Coord::Loop("kt".into()), Coord::Grid(1)]
                },
                AccessMode::Read
            )
        );
        assert_eq!(
            args[2],
            (
                SuperTile {
                    tensor: 2,
                    coords: vec![Coord::Grid(0), Coord::Grid(1)]
                },
                AccessMode::RMW
            )
        );
    }

    #[test]
    fn matmul_leaves() {
        let p = compile(&first(MATMUL)).unwrap();
        assert_eq!(p.leaves.len(), 2);

        // step leaf: original kernel with the final store flipped to +=
        let step = &p.leaves[0];
        assert_eq!(step.kernel.name, "matmul");
        assert!(step.kernel.attrs.iter().all(|a| a.name != "cluster"));
        let Some(Stmt::Assign { op, .. }) = step.kernel.body.last() else {
            panic!("step leaf does not end in the accumulator store");
        };
        assert_eq!(*op, AssignOp::Add, "chain step must accumulate into C");
        assert_eq!(
            step.modes,
            vec![AccessMode::Read, AccessMode::Read, AccessMode::RMW]
        );

        // init leaf: zero-fill of the output supertile
        let init = &p.leaves[1];
        assert_eq!(init.kernel.name, "matmul_init");
        assert_eq!(init.kernel.params.len(), 1);
        assert_eq!(init.kernel.params[0].name, "C");
        assert_eq!(init.modes, vec![AccessMode::Write]);
    }

    #[test]
    fn matmul_leaves_compile_at_device_scale() {
        let p = compile(&first(MATMUL)).unwrap();
        let step = verify_leaf(&p.leaves[0].kernel);
        assert!(step.contains("gpu.func"), "step leaf missing gpu.func");
        let init = verify_leaf(&p.leaves[1].kernel);
        assert!(init.contains("gpu.func"), "init leaf missing gpu.func");
    }

    /// A full GEMM: C = alpha*acc + beta*C_old. The accumulator epilogue reads
    /// C back, so copy-elision folds beta into the init leaf (C = beta*C_old,
    /// an rmw) and keeps alpha on the step chain (C = alpha*acc + c_old).
    const GEMM: &str = r#"
@cluster(TILE_M in [4096, 16384], TILE_N in [4096, 16384], TILE_K in [4096, 16384])
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N], alpha: f32, beta: f32) {
    let pm = program_id(0)
    let pn = program_id(1)
    var acc: tile<f32>[TILE_M, TILE_N] = 0.0
    for kt in range(0, K, TILE_K) {
        let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
        let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
        acc += dot(a, b)
    }
    let c_old = C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N]
    C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = alpha * acc + beta * c_old
}"#;

    #[test]
    fn gemm_cluster_ir() {
        let p = compile(&first(GEMM)).unwrap();

        // C is now read-modify-write: the init leaf seeds it with beta*C_old,
        // so its original value must be loaded rather than zeroed.
        let modes: Vec<_> = p.tensors.iter().map(|t| t.mode).collect();
        assert_eq!(
            modes,
            vec![AccessMode::Read, AccessMode::Read, AccessMode::RMW]
        );

        // both scalars stay out of the cluster IR's dataflow; the step compute
        // carries the kernel's own [alpha, beta], the init carries its local beta.
        assert_eq!(p.scalars[0].name, "alpha");
        assert_eq!(p.scalars[1].name, "beta");

        // body: init (beta*C_old) reads C rmw and carries a beta scalar
        let ClusterStmt::Compute {
            leaf: 1,
            args,
            scalars,
        } = &p.body[0]
        else {
            panic!("expected init compute, got {:?}", p.body[0]);
        };
        assert_eq!(
            args[0],
            (
                SuperTile {
                    tensor: 2,
                    coords: vec![Coord::Grid(0), Coord::Grid(1)]
                },
                AccessMode::RMW
            )
        );
        assert_eq!(scalars.len(), 1, "init leaf carries its beta coefficient");
        assert_eq!(p.scalars[scalars[0]].name, "beta");
        assert_eq!(
            p.scalars[scalars[0]].param_pos, 1,
            "beta sits at the init leaf's local position (after C)"
        );

        // the chain step still targets C(pm,pn) rmw (acc copy-elided) and carries
        // both kernel scalars
        let ClusterStmt::Loop { body, .. } = &p.body[1] else {
            panic!("expected the k-loop, got {:?}", p.body[1]);
        };
        let ClusterStmt::Compute {
            leaf: 0,
            args,
            scalars,
        } = &body[0]
        else {
            panic!("expected step compute, got {:?}", body[0]);
        };
        assert_eq!(scalars, &vec![0, 1]);
        assert_eq!(args[2].1, AccessMode::RMW);
    }

    #[test]
    fn gemm_leaves() {
        let p = compile(&first(GEMM)).unwrap();
        assert_eq!(p.leaves.len(), 2);

        // step leaf: the store keeps the fused epilogue shape alpha*acc + c_old
        // (an implicit beta of 1); the real beta is applied once by the init.
        let step = &p.leaves[0];
        let Some(Stmt::Assign {
            op: AssignOp::Set,
            value:
                Expr::Binary {
                    op: BinOp::Add,
                    lhs,
                    rhs,
                },
            ..
        }) = step.kernel.body.last()
        else {
            panic!("step leaf must end in `C[..] = alpha*acc + c_old`");
        };
        assert!(
            matches!(lhs.as_ref(), Expr::Binary { op: BinOp::Mul, .. }),
            "step store's acc term should be alpha*acc"
        );
        assert!(
            matches!(rhs.as_ref(), Expr::Var(n) if n == "c_old"),
            "step store's prior-C term should be c_old with an implicit beta of 1"
        );

        // init leaf: C = beta*c_old, an rmw over C plus the beta scalar param
        let init = &p.leaves[1];
        assert_eq!(init.kernel.name, "matmul_init");
        assert_eq!(init.kernel.params.len(), 2);
        assert_eq!(init.kernel.params[0].name, "C");
        assert_eq!(init.kernel.params[1].name, "beta");
        assert_eq!(init.modes, vec![AccessMode::RMW, AccessMode::Read]);
    }

    #[test]
    fn gemm_leaves_compile_at_device_scale() {
        let p = compile(&first(GEMM)).unwrap();
        let step = verify_leaf(&p.leaves[0].kernel);
        assert!(step.contains("gpu.func"), "step leaf missing gpu.func");
        let init = verify_leaf(&p.leaves[1].kernel);
        assert!(init.contains("gpu.func"), "init leaf missing gpu.func");
    }

    #[test]
    fn add_cluster_ir() {
        let p = compile(&first(ADD)).unwrap();
        assert_eq!(p.grid.len(), 1);
        assert_eq!(p.grid[0].dim, Dim::Sym("N".into()));
        assert_eq!(p.grid[0].super_sym, "BLOCK");
        assert_eq!(p.leaves.len(), 1, "elementwise kernels need no init leaf");
        assert_eq!(p.body.len(), 1);
        let ClusterStmt::Compute { leaf: 0, args, .. } = &p.body[0] else {
            panic!("expected one compute, got {:?}", p.body[0]);
        };
        let modes: Vec<_> = args.iter().map(|(_, m)| *m).collect();
        assert_eq!(
            modes,
            vec![AccessMode::Read, AccessMode::Read, AccessMode::Write]
        );
        for (i, (r, _)) in args.iter().enumerate() {
            assert_eq!(
                *r,
                SuperTile {
                    tensor: i,
                    coords: vec![Coord::Grid(0)]
                }
            );
        }
        verify_leaf(&p.leaves[0].kernel);
    }

    /// A flash-attention-shaped kernel: grid over query blocks, a device-scale
    /// key loop kept inside the leaf, full : slices over the head dim, running
    /// tile state, and a scalar parameter.
    const FLASH: &str = r#"
@cluster(BR in [1024, 4096])
@autotune(D in [64], BR in [32, 128], BC in [32, 128])
kernel attn(Q: tensor<f32>[Nq, D],
            K: tensor<f32>[Nk, D],
            V: tensor<f32>[Nk, D],
            O: tensor<f32>[Nq, D],
            scale: f32) {
    let pid = program_id(0)
    let row = pid * BR
    let q = Q[row :+ BR, :]
    var acc: tile<f32>[BR, D] = 0.0
    var l: tile<f32>[BR, 1] = 0.0
    for kt in range(0, Nk, BC) {
        let k = K[kt :+ BC, :]
        let v = V[kt :+ BC, :]
        var s: tile<f32>[BR, BC] = dot_t(q, k)
        s = s * scale
        var p: tile<f32>[BR, BC] = exp(s)
        l += rowsum(p)
        acc += dot(p, v)
    }
    acc = acc / l
    O[row :+ BR, :] = acc
}"#;

    #[test]
    fn accepts_scalar_params() {
        // matmul + an (unused) alpha scalar still clusters; the scalar is
        // recorded with its parameter position and the step compute carries it.
        let src = MATMUL.replace("C: tensor<f32>[M, N])", "C: tensor<f32>[M, N], alpha: f32)");
        let p = compile(&first(&src)).unwrap();
        assert_eq!(p.scalars.len(), 1);
        assert_eq!(p.scalars[0].name, "alpha");
        assert_eq!(p.scalars[0].data_type, crate::tile::DataType::F32);
        assert_eq!(p.scalars[0].param_pos, 3);
        // the step compute (leaf 0) inside the k-loop passes the scalar
        let ClusterStmt::Loop { body, .. } = &p.body[1] else {
            panic!("expected the k-loop");
        };
        let ClusterStmt::Compute {
            leaf: 0, scalars, ..
        } = &body[0]
        else {
            panic!("expected the step compute");
        };
        assert_eq!(scalars, &vec![0]);
    }

    #[test]
    fn flash_single_leaf() {
        let p = compile(&first(FLASH)).unwrap();

        // one grid axis over query blocks, supertiled by BR
        assert_eq!(p.grid.len(), 1);
        assert_eq!(p.grid[0].dim, Dim::Sym("Nq".into()));
        assert_eq!(p.grid[0].super_sym, "BR");

        // the whole kernel is a single leaf (no cluster loop, no init leaf)
        assert_eq!(p.leaves.len(), 1);
        assert_eq!(p.leaves[0].kernel.name, "attn");

        // scalar recorded at its parameter position (after the four tensors)
        assert_eq!(p.scalars.len(), 1);
        assert_eq!(p.scalars[0].name, "scale");
        assert_eq!(p.scalars[0].param_pos, 4);

        // Q/O are query-tiled (grid, full head dim); K/V are read whole
        assert_eq!(p.tensors[0].super_syms, vec!["BR", "D"]); // Q
        assert_eq!(p.tensors[1].super_syms, vec!["Nk", "D"]); // K
        assert_eq!(p.tensors[2].super_syms, vec!["Nk", "D"]); // V
        assert_eq!(p.tensors[3].super_syms, vec!["BR", "D"]); // O
        let modes: Vec<_> = p.tensors.iter().map(|t| t.mode).collect();
        assert_eq!(
            modes,
            vec![
                AccessMode::Read,
                AccessMode::Read,
                AccessMode::Read,
                AccessMode::Write
            ]
        );

        // exactly one compute: Q(p0,:), K(:,:), V(:,:) read, O(p0,:) written,
        // carrying the scalar
        assert_eq!(p.body.len(), 1);
        let ClusterStmt::Compute {
            leaf: 0,
            args,
            scalars,
        } = &p.body[0]
        else {
            panic!("expected one leaf compute, got {:?}", p.body[0]);
        };
        assert_eq!(scalars, &vec![0]);
        assert_eq!(
            args[0],
            (
                SuperTile {
                    tensor: 0,
                    coords: vec![Coord::Grid(0), Coord::Full]
                },
                AccessMode::Read
            )
        );
        assert_eq!(
            args[1],
            (
                SuperTile {
                    tensor: 1,
                    coords: vec![Coord::Full, Coord::Full]
                },
                AccessMode::Read
            )
        );
        assert_eq!(
            args[3],
            (
                SuperTile {
                    tensor: 3,
                    coords: vec![Coord::Grid(0), Coord::Full]
                },
                AccessMode::Write
            )
        );

        // the leaf device-compiles (the f32 scalar lands as a value param)
        let text = verify_leaf(&p.leaves[0].kernel);
        assert!(text.contains("f32"), "leaf missing the scalar param");
    }

    #[test]
    fn rejects_misaligned_offsets() {
        let src = MATMUL.replace("A[pm * TILE_M :+ TILE_M", "A[pm * TILE_M + 1 :+ TILE_M");
        let err = compile(&first(&src)).unwrap_err().to_string();
        assert!(err.contains("not supertile-aligned"), "got: {err}");
    }

    #[test]
    fn rejects_extent_step_mismatch() {
        // slice along K uses TILE_M as extent while the loop steps by TILE_K
        let src = MATMUL.replace("kt :+ TILE_K]", "kt :+ TILE_M]");
        let err = compile(&first(&src)).unwrap_err().to_string();
        assert!(err.contains("steps by"), "got: {err}");
    }

    #[test]
    fn rejects_cluster_dim_without_autotune() {
        let src = MATMUL.replace(
            "@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])",
            "@autotune(TILE_M in [32, 256], TILE_N in [32, 256])",
        );
        let err = compile(&first(&src)).unwrap_err().to_string();
        assert!(err.contains("must also be an @autotune dim"), "got: {err}");
    }

    #[test]
    fn rejects_multiple_cluster_computes() {
        // Two rmw chains over the same cluster loop is still unsupported on the
        // accumulator path.
        let src = r#"
@cluster(TILE_M in [4096, 16384], TILE_N in [4096, 16384], TILE_K in [4096, 16384])
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel two(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N], D: tensor<f32>[M, N]) {
    let pm = program_id(0)
    let pn = program_id(1)
    var acc: tile<f32>[TILE_M, TILE_N] = 0.0
    for kt in range(0, K, TILE_K) {
        let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
        let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
        acc += dot(a, b)
        D[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] += dot(a, b)
    }
    C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}"#;
        let err = compile(&first(src)).unwrap_err().to_string();
        assert!(err.contains("multiple compute statements"), "got: {err}");
    }
}
