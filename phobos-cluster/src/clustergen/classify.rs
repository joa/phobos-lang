// Classifies tensor references: which supertile each names, its grid axis, and scalar invariance.

use super::*;

impl<'a> Analyzer<'a> {
    pub(super) fn unify_ref(
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

    pub(super) fn classify_ref_leaf(&mut self, tensor: usize, subs: &[Sub]) -> Result<SuperTile> {
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
                        let ExprKind::Var(ls) = &len.kind else {
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
                    if let ExprKind::Var(v) = &start.kind
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
                         `ctx.block.<axis> * SUPER`, a `:` full slice, or a device-loop var",
                        self.tensors[tensor].name
                    );
                }
                Sub::Point(_) | Sub::Range { .. } | Sub::From(_) => bail!(
                    "only `start :+ len` spans and `:` full slices are supported under \
                     @cluster (slice of '{}')",
                    self.tensors[tensor].name
                ),
            }
        }
        Ok(SuperTile { tensor, coords })
    }

    pub(super) fn classify_ref(&mut self, tensor: usize, subs: &[Sub]) -> Result<SuperTile> {
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
            let ExprKind::Var(s) = &len.kind else {
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
            if let ExprKind::Var(v) = &start.kind
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
                 expected `ctx.block.<axis> * {s}` or a cluster-loop var stepping by {s}",
                self.tensors[tensor].name
            );
        }
        Ok(SuperTile { tensor, coords })
    }

    pub(super) fn classify_scalar(&self, e: &Expr) -> Result<Option<ScalarValue>> {
        match &e.kind {
            ExprKind::Call { callee, args } if callee == phobos_lang::ast::ctx::BLOCK => {
                let [Expr { kind: ExprKind::Int(i), .. }] = args.as_slice() else {
                    unreachable!("the parser builds ctx.block with one literal axis");
                };
                Ok(Some(ScalarValue::Pid(*i as usize)))
            }
            ExprKind::Var(n) => Ok(match self.symbols.get(n) {
                Some(Binding::Scalar(sv)) => Some(sv.clone()),
                _ => None,
            }),
            ExprKind::Binary {
                op: BinOp::Mul,
                lhs,
                rhs,
            } => {
                // `pid * SUPER` or `SUPER * pid`
                for (a, b) in [(lhs, rhs), (rhs, lhs)] {
                    if let ExprKind::Var(s) = &b.kind
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

    pub(super) fn axis_dim_sym(&self, tensor: usize, axis: usize, dim: &Dim) -> Result<String> {
        match dim {
            Dim::Sym(s) => Ok(s.clone()),
            Dim::Int(_) => bail!(
                "tensor '{}' axis {axis} has a literal size; a `:` or device-loop slice \
                 needs a symbolic dim to name the supertile",
                self.tensors[tensor].name
            ),
        }
    }

    pub(super) fn set_tensor_sym(&mut self, tensor: usize, axis: usize, sym: &str) -> Result<()> {
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

    pub(super) fn note_grid(
        &mut self,
        pid: usize,
        tensor: usize,
        axis: usize,
        sym: &str,
    ) -> Result<()> {
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
                        "`ctx.block` axis {pid} is used with conflicting supertile shapes: \
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

    pub(super) fn collect_reads(
        &mut self,
        e: &Expr,
        out: &mut Vec<(usize, SuperTile)>,
    ) -> Result<()> {
        match &e.kind {
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) => Ok(()),
            ExprKind::Var(n) => match self.symbols.get(n).cloned() {
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
            ExprKind::Index { base, subs } => {
                let ExprKind::Var(t) = &base.kind else {
                    bail!("unsupported indexing in a supertile computation");
                };
                let Some(&ti) = self.tindex.get(t) else {
                    bail!("'{t}' is not a tensor parameter");
                };
                let r = self.classify_ref(ti, subs)?;
                out.push((ti, r));
                Ok(())
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                self.collect_reads(lhs, out)?;
                self.collect_reads(rhs, out)
            }
            ExprKind::Unary { rhs, .. } => self.collect_reads(rhs, out),
            ExprKind::Call { callee, args } if callee == "dot" => {
                for a in args {
                    self.collect_reads(a, out)?;
                }
                Ok(())
            }
            ExprKind::Call { callee, .. } => {
                bail!("call to '{callee}' is not supported in a supertile computation")
            }
            ExprKind::Generate(_) => bail!("a generated tile is not supported in a supertile computation"),
        }
    }

    /// Whether `e` reads the accumulator scratch tile anywhere.
    pub(super) fn uses_scratch(&self, e: &Expr) -> bool {
        let mut found = false;
        e.walk(&mut |e| {
            if let ExprKind::Var(n) = &e.kind {
                found |= matches!(self.symbols.get(n), Some(Binding::Scratch));
            }
        });
        found
    }

    /// Collect the scalar-parameter indices an epilogue coefficient references,
    /// in first-appearance order. Any non-scalar term is an error.
    pub(super) fn collect_scalar_refs(&self, e: &Expr, out: &mut Vec<usize>) -> Result<()> {
        match &e.kind {
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) => Ok(()),
            ExprKind::Var(n) => match self.scalars.iter().position(|s| &s.name == n) {
                Some(i) => {
                    if !out.contains(&i) {
                        out.push(i);
                    }
                    Ok(())
                }
                None => bail!("epilogue coefficient '{n}' is not a scalar parameter"),
            },
            ExprKind::Binary { lhs, rhs, .. } => {
                self.collect_scalar_refs(lhs, out)?;
                self.collect_scalar_refs(rhs, out)
            }
            ExprKind::Unary { rhs, .. } => self.collect_scalar_refs(rhs, out),
            _ => bail!("epilogue coefficients must be arithmetic over scalar parameters"),
        }
    }

    /// Reject epilogue coefficients that are not scalar-parameter arithmetic.
    /// They must be loop-invariant: alpha applies every step, beta once at init.
    pub(super) fn check_invariant(&self, e: &Expr) -> Result<()> {
        let mut idxs = Vec::new();
        self.collect_scalar_refs(e, &mut idxs)
    }
}
