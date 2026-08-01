use super::*;

/// The validated shape of a fragment-accumulator declaration (see
/// [`Codegen::frag_acc_candidate`]).
pub(super) struct FragAccPlan {
    pub(super) name: String,
    pub(super) init: f64,
    pub(super) m: i64,
    pub(super) n: i64,
    pub(super) wm: i64,
    pub(super) wn: i64,
}

/// Shapes the candidate walk has resolved so far: declared tiles and
/// let-bound static tensor slices, by name, as (is_f32, shape). Only f16/f32
/// entries are recorded, since those are all the fragment ops can take.
#[derive(Default)]
struct FragScan {
    decls: HashMap<String, (bool, Vec<i64>)>,
    kk: Option<i64>,
}

impl<'p, 'c> Codegen<'p, 'c> {
    /// Matches var acc: tile<f32>[m, n] = <float> whose every later use is
    /// one of the fragment-representable forms:
    ///
    ///   acc = acc * <[m, 1] f32 tile>    (also /)
    ///   acc += dot(<[m, k] tile>, <[k, n] tile or tensor slice>)
    ///   <[m, n] float tensor slice> = acc
    ///
    /// with mutations only at statement level of the surrounding blocks and
    /// enclosing for loops (threaded as iter_args by emit_frag_for; never
    /// inside if/while, whose regions cannot carry them). Such an accumulator
    /// never exists in shared memory: it lives in per-lane mma.sync D
    /// fragments, which kills both its shared footprint (occupancy) and its
    /// per-iteration shared round-trips (the += seed/scatter and the scale
    /// pass). Returns None to fall back to the regular shared-tile path.
    pub(super) fn frag_acc_candidate(&self, stmts: &[Stmt]) -> Option<FragAccPlan> {
        let [
            Stmt::Var {
                name,
                ty: Some(AstType::Tile(Scalar::F32, dims)),
                value: Expr::Float(init),
            },
            rest @ ..,
        ] = stmts
        else {
            return None;
        };
        if !self.mma_sync() {
            return None;
        }
        let shape = self.tile_shape(dims).ok()?;
        let &[m, n] = &shape[..] else { return None };

        let mut scan = FragScan::default();
        if !self.frag_uses_ok(name, m, n, rest, &mut scan) {
            return None;
        }

        // At least one dot pins the contraction extent the plan gate needs;
        // an accumulator that is never dotted has no business in fragments.
        let (wm, wn) = self.wmma_plan(m, n, scan.kk?)?;

        Some(FragAccPlan {
            name: name.clone(),
            init: *init,
            m,
            n,
            wm,
            wn,
        })
    }

    /// Walks statements checking every appearance of name is a sanctioned
    /// fragment form, collecting resolvable shapes along the way.
    fn frag_uses_ok(
        &self,
        name: &str,
        m: i64,
        n: i64,
        stmts: &[Stmt],
        scan: &mut FragScan,
    ) -> bool {
        for stmt in stmts {
            match stmt {
                Stmt::Let {
                    name: n2,
                    ty,
                    value,
                }
                | Stmt::Var {
                    name: n2,
                    ty,
                    value,
                } => {
                    // A redeclaration would shadow the accumulator mid-scan.
                    if n2 == name || value.uses_name(name) {
                        return false;
                    }

                    if self.slice_is_partial(value) {
                        return false;
                    }

                    self.frag_scan_decl(n2, ty.as_ref(), value, scan);
                }
                Stmt::Assign {
                    target: Expr::Var(t),
                    op,
                    value,
                } if t == name => {
                    if !self.frag_form_ok(name, m, n, *op, value, scan) {
                        return false;
                    }
                }
                // <tensor slice> = acc: the fragment scatter store.
                Stmt::Assign {
                    target: target @ Expr::Index { base, subs },
                    op: AssignOp::Set,
                    value: Expr::Var(v),
                } if v == name => {
                    let Expr::Var(t) = &**base else { return false };
                    let Some(Binding::Tensor(c)) = self.lookup(t) else {
                        return false;
                    };

                    if !self.is_f16_or_f32(c.elem) || subs.iter().any(|s| s.uses_name(name)) {
                        return false;
                    }

                    let Some(shape) = self.slice_static_shape(target) else {
                        return false;
                    };
                    if shape != [m, n] {
                        return false;
                    }

                    if self.slice_is_partial(target) {
                        return false;
                    }
                }
                Stmt::Assign { target, value, .. } => {
                    if target.uses_name(name) || value.uses_name(name) {
                        return false;
                    }
                }
                Stmt::For {
                    var,
                    start,
                    end,
                    step,
                    body,
                } => {
                    if var == name
                        || start.uses_name(name)
                        || end.uses_name(name)
                        || step.as_ref().is_some_and(|e| e.uses_name(name))
                    {
                        return false;
                    }
                    if !self.frag_uses_ok(name, m, n, body, scan) {
                        return false;
                    }
                }
                // if/while regions cannot thread iter_args, so the
                // accumulator must not appear inside them at all.
                s @ (Stmt::While { .. } | Stmt::If { .. }) => {
                    if s.uses_name(name) {
                        return false;
                    }
                }
                Stmt::Expr(e) => {
                    if e.uses_name(name) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Records a declared tile or let-bound static tensor slice the later
    /// form checks can resolve shapes against.
    fn frag_scan_decl(&self, name: &str, ty: Option<&AstType>, value: &Expr, scan: &mut FragScan) {
        let entry = match ty {
            Some(AstType::Tile(sc @ (Scalar::F16 | Scalar::F32), dims)) => {
                self.tile_shape(dims).ok().map(|s| (*sc == Scalar::F32, s))
            }
            None => match (
                self.slice_static_shape(value),
                self.slice_tensor_elem(value),
            ) {
                (Some(s), Some(e)) if self.is_f16_or_f32(e) => Some((e == self.f32_t, s)),
                _ => None,
            },
            _ => None,
        };
        if let Some(e) = entry {
            scan.decls.insert(name.to_string(), e);
        }
    }

    /// A named operand's element class and static shape: a decl the walk has
    /// seen, or a tile binding that predates the accumulator.
    fn frag_operand(&self, scan: &FragScan, name: &str) -> Option<(bool, Vec<i64>)> {
        if let Some(e) = scan.decls.get(name) {
            return Some(e.clone());
        }
        match self.lookup(name)? {
            Binding::Tile(mv) | Binding::View(mv)
                if !mv.shape.contains(&DYN) && self.is_f16_or_f32(mv.elem) =>
            {
                Some((mv.elem == self.f32_t, mv.shape))
            }
            _ => None,
        }
    }

    /// Checks one assignment to the accumulator against the sanctioned forms.
    fn frag_form_ok(
        &self,
        name: &str,
        m: i64,
        n: i64,
        op: AssignOp,
        value: &Expr,
        scan: &mut FragScan,
    ) -> bool {
        match (op, value) {
            // acc = acc {*, /} <[m, 1] f32 column>
            (
                AssignOp::Set,
                Expr::Binary {
                    op: BinOp::Mul | BinOp::Div,
                    lhs,
                    rhs,
                },
            ) => {
                let (Expr::Var(l), Expr::Var(col)) = (lhs.as_ref(), rhs.as_ref()) else {
                    return false;
                };
                if l != name || col == name {
                    return false;
                }
                let Some((is_f32, shape)) = self.frag_operand(scan, col) else {
                    return false;
                };
                is_f32 && shape == [m, 1]
            }
            // acc += dot(<[m, k]>, <[k, n]>) with f16/f32 operands
            (AssignOp::Add, Expr::Call { callee, args }) if callee == "dot" => {
                let [Expr::Var(a), Expr::Var(b)] = &args[..] else {
                    return false;
                };
                if a == name || b == name {
                    return false;
                }
                let (Some((_, ash)), Some((_, bsh))) =
                    (self.frag_operand(scan, a), self.frag_operand(scan, b))
                else {
                    return false;
                };
                let (&[am, ak], &[bk, bn]) = (&ash[..], &bsh[..]) else {
                    return false;
                };
                if am != m || bn != n || ak != bk || ak % 16 != 0 {
                    return false;
                }
                scan.kk.get_or_insert(ak);
                true
            }
            _ => false,
        }
    }
}

// emission
impl<'p, 'c> Codegen<'p, 'c> {
    /// Seeds the fragment binding for a validated accumulator declaration.
    pub(super) fn bind_frag_acc(&mut self, block: &Block<'c>, plan: &FragAccPlan) -> Result<()> {
        let acc_t = Type::vector(&[2, 2], self.f32_t);
        let init = self.push(
            block,
            arith::constant(
                self.ctx,
                FloatAttribute::new(self.ctx, self.f32_t, plan.init).into(),
                self.loc,
            ),
        )?;
        let seed = self.vec_broadcast(block, init, acc_t)?;
        let fa = FragAcc {
            frags: Vec::new(),
            m: plan.m,
            n: plan.n,
            wm: plan.wm,
            wn: plan.wn,
        };
        let (fm, fnn) = fa.warp_frags();
        let frags = vec![seed; (fm * fnn * 2) as usize];
        self.bind(&plan.name, Binding::Frags(FragAcc { frags, ..fa }));
        Ok(())
    }

    /// Emits an assignment to a fragment-bound accumulator. The candidate
    /// walk admitted only the scale and dot-accumulate forms.
    pub(super) fn emit_frag_assign(
        &mut self,
        block: &Block<'c>,
        name: &str,
        fa: &FragAcc<'c>,
        op: AssignOp,
        value: &Expr,
    ) -> Result<()> {
        match (op, value) {
            (
                AssignOp::Set,
                Expr::Binary {
                    op: bop @ (BinOp::Mul | BinOp::Div),
                    rhs,
                    ..
                },
            ) => {
                let Expr::Var(col) = rhs.as_ref() else {
                    bail!("fragment scale needs a named column operand");
                };
                let Some(Binding::Tile(cmv) | Binding::View(cmv)) = self.lookup(col) else {
                    bail!("fragment scale column '{col}' is not a tile");
                };
                self.frag_scale(block, name, fa, *bop, &cmv)
            }
            (AssignOp::Add, Expr::Call { args, .. }) => {
                let (a, b) = self.dot_operands(block, args)?;
                self.frag_dot(block, name, fa, &a, &b)?;
                self.release(&a);
                self.release(&b);
                Ok(())
            }
            _ => bail!(
                "fragment accumulator '{name}' only supports scaling, dot-accumulate, and stores"
            ),
        }
    }

    /// acc = acc {*, /} col for a [m, 1] f32 column vector in shared memory:
    /// each lane rescales its fragment elements by col[row], the rows coming
    /// from the same per-lane walk the scatter store uses. Pure register
    /// math plus broadcast column reads; no barrier (col is only read).
    fn frag_scale(
        &mut self,
        block: &Block<'c>,
        name: &str,
        fa: &FragAcc<'c>,
        op: BinOp,
        col: &MemVal<'c>,
    ) -> Result<()> {
        let (fm, fnn) = fa.warp_frags();
        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        let zero = self.const_index(block, 0)?;
        let mut frags = fa.frags.clone();
        self.for_each_dfrag(block, (fm, fnn), m0, n0, |cg, i, elems| {
            for ([di, dj], [row, _]) in elems {
                let c = cg.load_as(block, col.mem, &[*row, zero], cg.f32_t)?;
                let e = cg.vec_extract(block, frags[i], &[*di, *dj], cg.f32_t)?;
                let r = cg.push(block, cg.elem_arith(op, cg.f32_t, e, c)?)?;
                frags[i] = cg.vec_insert(block, r, frags[i], &[*di, *dj])?;
            }
            Ok(())
        })?;
        self.update_binding(
            name,
            Binding::Frags(FragAcc {
                frags,
                ..fa.clone()
            }),
        );
        Ok(())
    }

    /// acc += dot(a, b) folded straight into the lane's D fragments: the same
    /// swizzled f16 staging and mma.sync MAC as mma_sync_dot, but the seeds
    /// come from the binding and the finals go back into it, so the running
    /// sum never round-trips shared memory.
    fn frag_dot(
        &mut self,
        block: &Block<'c>,
        name: &str,
        fa: &FragAcc<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
    ) -> Result<()> {
        let kk = a.shape[1];
        self.check_shapes(&a.shape, &[fa.m, kk], "fragment dot lhs")?;
        self.check_shapes(&b.shape, &[kk, fa.n], "fragment dot rhs")?;
        let (fm, fnn) = fa.warp_frags();

        let (a_buf, a_hoisted) = self.dot_stage(block, a, &[fa.m, kk], true)?;
        let (b_buf, b_hoisted) = self.dot_stage(block, b, &b.shape.clone(), true)?;
        self.barrier(block)?;

        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        let finals = self.mma_sync_mac(
            block,
            &a_buf,
            &b_buf,
            (kk, fm, fnn),
            m0,
            n0,
            &fa.frags,
            false,
        )?;

        // No shared output to publish, but the barrier still orders the
        // ldmatrix reads before the released staging is reused. Hoisted
        // buffers outlive the loop.
        self.barrier(block)?;
        if !a_hoisted {
            self.release(&a_buf);
        }
        if !b_hoisted {
            self.release(&b_buf);
        }

        self.update_binding(
            name,
            Binding::Frags(FragAcc {
                frags: finals,
                ..fa.clone()
            }),
        );
        Ok(())
    }

    /// <tensor slice> = acc: scatters each lane's fragment elements straight
    /// to the target, rounding f32 to f16 when the tensor is f16. The
    /// epilogue analogue of mma_sync_store_frags, minus the shared hop.
    pub(super) fn frag_store(
        &mut self,
        block: &Block<'c>,
        target: &MemVal<'c>,
        fa: &FragAcc<'c>,
    ) -> Result<()> {
        self.check_shapes(&[fa.m, fa.n], &target.shape, "fragment store")?;
        let (fm, fnn) = fa.warp_frags();
        let (.., m0, n0) = self.warp_block_origin(block, fa.wm, fa.wn, fm * 16, fnn * 16)?;
        self.for_each_dfrag(block, (fm, fnn), m0, n0, |cg, i, elems| {
            for ([di, dj], addr) in elems {
                let e = cg.vec_extract(block, fa.frags[i], &[*di, *dj], cg.f32_t)?;
                let e = cg.coerce(block, e, target.elem)?;
                block.append_operation(memref::store(e, target.mem, addr, cg.loc));
            }
            Ok(())
        })
    }

    /// The fragment-bound names a loop body assigns: their values must ride
    /// the loop as iter_args (see emit_frag_for). The candidate walk rejected
    /// mutations inside if/while, so only for bodies recurse.
    pub(super) fn frag_carried(&self, body: &[Stmt]) -> Vec<String> {
        let mut names = Vec::new();
        self.frag_carried_into(body, &mut names);
        names
    }

    fn frag_carried_into(&self, stmts: &[Stmt], names: &mut Vec<String>) {
        for stmt in stmts {
            match stmt {
                Stmt::Assign {
                    target: Expr::Var(n),
                    ..
                } if !names.contains(n) => {
                    if matches!(self.lookup(n), Some(Binding::Frags(_))) {
                        names.push(n.clone());
                    }
                }
                Stmt::For { body, .. } => self.frag_carried_into(body, names),
                _ => {}
            }
        }
    }

    /// scf.for threading the fragment accumulators the body assigns as
    /// iter_args: fragments are SSA values, unlike the memref-backed tiles,
    /// so loop-carried updates must ride the loop op. The body scope shadows
    /// each name onto the block arguments and the outer bindings pick up the
    /// loop results.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_frag_for(
        &mut self,
        block: &Block<'c>,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        body: &[Stmt],
        names: &[String],
    ) -> Result<()> {
        let (lo, hi, st, iv_div) = self.loop_bounds(block, start, end, step)?;

        let mut accs = Vec::with_capacity(names.len());
        let mut inits = Vec::new();
        for name in names {
            let Some(Binding::Frags(fa)) = self.lookup(name) else {
                bail!("'{name}' is not fragment-bound");
            };
            inits.extend(fa.frags.iter().copied());
            accs.push(fa);
        }

        let mut args = vec![(self.index_t, self.loc)];
        args.extend(inits.iter().map(|v| (v.r#type(), self.loc)));
        let body_block = Block::new(&args);
        let iv = detach(body_block.argument(0)?.into());

        self.scopes.push(HashMap::new());
        self.bind(
            var,
            Binding::Let {
                value: iv,
                div: iv_div,
            },
        );
        let mut off = 1;
        for (name, fa) in names.iter().zip(&accs) {
            let mut frags = Vec::with_capacity(fa.frags.len());
            for k in 0..fa.frags.len() {
                frags.push(detach(body_block.argument(off + k)?.into()));
            }
            off += fa.frags.len();
            self.bind(
                name,
                Binding::Frags(FragAcc {
                    frags,
                    ..fa.clone()
                }),
            );
        }

        // Collect the loop-carried finals before the scope pops.
        let finals = self.emit_stmts(&body_block, body).and_then(|()| {
            let mut finals = Vec::with_capacity(inits.len());
            for name in names {
                let Some(Binding::Frags(fa)) = self.lookup(name) else {
                    bail!("'{name}' lost its fragment binding in the loop body");
                };
                finals.extend(fa.frags);
            }
            Ok(finals)
        });
        self.scopes.pop();
        let finals = finals?;
        body_block.append_operation(scf::r#yield(&finals, self.loc));

        let region = Region::new();
        region.append_block(body_block);
        let types: Vec<Type<'c>> = inits.iter().map(|v| v.r#type()).collect();
        let mut operands = vec![lo, hi, st];
        operands.extend_from_slice(&inits);
        let op = block.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&operands)
                .add_results(&types)
                .add_regions([region])
                .build()?,
        );

        let mut off = 0;
        for (name, fa) in names.iter().zip(&accs) {
            let mut frags = Vec::with_capacity(fa.frags.len());
            for k in 0..fa.frags.len() {
                frags.push(detach(op.result(off + k)?.into()));
            }
            off += fa.frags.len();
            self.update_binding(
                name,
                Binding::Frags(FragAcc {
                    frags,
                    ..fa.clone()
                }),
            );
        }
        Ok(())
    }
}
