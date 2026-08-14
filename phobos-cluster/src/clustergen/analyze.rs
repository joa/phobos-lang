// Walking a kernel: what each statement and assignment contributes to
// the cluster program being built.

use super::*;

impl<'a> Analyzer<'a> {
    pub(super) fn new(kernel: &'a Kernel) -> Result<Self> {
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

    pub(super) fn run(self) -> Result<ClusterProgram> {
        if has_cluster_loop(&self.kernel.body, &self.super_set) {
            self.run_accumulator()
        } else {
            self.run_single_leaf()
        }
    }

    pub(super) fn run_accumulator(mut self) -> Result<ClusterProgram> {
        let mut body = Vec::new();
        for (idx, stmt) in self.kernel.body.iter().enumerate() {
            if let Some(b) = self.stmt(stmt, idx, false)? {
                body.push(b);
            }
        }
        self.finalize(body)
    }

    pub(super) fn run_single_leaf(mut self) -> Result<ClusterProgram> {
        let mut refs: HashMap<usize, SuperTile> = HashMap::new();
        let mut reads: HashSet<usize> = HashSet::new();
        let mut writes: HashSet<usize> = HashSet::new();
        self.walk_single(&self.kernel.body, &mut refs, &mut reads, &mut writes)?;
        self.finalize_single(refs, reads, writes)
    }

    pub(super) fn walk_single(
        &mut self,
        stmts: &'a [Stmt],
        refs: &mut HashMap<usize, SuperTile>,
        reads: &mut HashSet<usize>,
        writes: &mut HashSet<usize>,
    ) -> Result<()> {
        for stmt in stmts {
            match stmt {
                Stmt::Let { .. } | Stmt::Var { .. } => {
                    // An unfilled tile buffer reads nothing and is not a scalar.
                    let Some((name, ty, Some(value))) = stmt.as_decl() else {
                        continue;
                    };
                    self.scan_reads(value, refs, reads)?;
                    if matches!(ty, Some(AstType::Tile(..))) {
                        continue;
                    }
                    if let Some(sv) = self.classify_scalar(value)? {
                        self.symbols.insert(name.to_string(), Binding::Scalar(sv));
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
    pub(super) fn scan_reads(
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

    pub(super) fn stmt(
        &mut self,
        stmt: &Stmt,
        idx: usize,
        in_loop: bool,
    ) -> Result<Option<Statement>> {
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
                let Some(init @ (Expr::Int(_) | Expr::Float(_))) = value else {
                    bail!("accumulator tile must be initialized with a literal");
                };
                self.scratch = Some(Scratch { init: init.clone() });
                self.symbols.insert(name.clone(), Binding::Scratch);
                Ok(Some(Statement::InitPlaceholder))
            }
            Stmt::Let {
                ty: Some(AstType::Tile(..)),
                ..
            } => bail!("accumulator tiles must be declared with `var`, not `let`"),

            // scalar / ref bindings
            Stmt::Let { .. } | Stmt::Var { .. } => {
                let Some((name, ty, Some(value))) = stmt.as_decl() else {
                    bail!("a declaration under @cluster needs an initializer");
                };
                if let Some(t) = ty
                    && !matches!(t, AstType::Scalar(_))
                {
                    bail!("unsupported declaration type for '{name}' under @cluster");
                }
                if let Some(sv) = self.classify_scalar(value)? {
                    self.symbols.insert(name.to_string(), Binding::Scalar(sv));
                    return Ok(None);
                }
                if let Expr::Index { base, subs } = value
                    && let Expr::Var(t) = base.as_ref()
                    && let Some(&ti) = self.tindex.get(t)
                {
                    let r = self.classify_ref(ti, subs)?;
                    self.symbols.insert(name.to_string(), Binding::Ref(r));
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

    pub(super) fn assign(
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
}
