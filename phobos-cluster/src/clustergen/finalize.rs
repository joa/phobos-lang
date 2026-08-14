// Turning the analysis into a program: grid axes, super symbols, the
// init leaves and the output slices.

use super::*;

impl<'a> Analyzer<'a> {
    pub(super) fn finalize_grid(&mut self) -> Result<Vec<GridAxis>> {
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

    pub(super) fn finalize_super_syms(&mut self) -> Result<()> {
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

    pub(super) fn finalize_single(
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

    pub(super) fn finalize(mut self, body: Vec<Statement>) -> Result<ClusterProgram> {
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
    pub(super) fn output_slice(&self, d: &Define) -> (Vec<Stmt>, Vec<Sub>) {
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

    pub(super) fn init_attrs(&self) -> Vec<phobos_lang::ast::Attribute> {
        self.kernel
            .attrs
            .iter()
            .filter(|a| a.name == "autotune" || a.name == "launch")
            .cloned()
            .collect()
    }

    /// Synthesize the plain chain's init leaf:
    /// kernel {name}_init(C: ..) { let p0 = program_id(0); ..; C[p0*S :+ S, ..] = <init> }
    pub(super) fn zero_init_leaf(&self, scratch: &Scratch, d: &Define) -> Kernel {
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
    pub(super) fn beta_init_leaf(
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
