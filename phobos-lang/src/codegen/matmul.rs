use super::*;

/// Width of the vector<Nxf16> global loads used when staging f16 operands
const HALF_VEC: i64 = 4;

/// The matched register-accumulator GEMM pattern.
///
/// See also [`Codegen::matmul_candidate`].
///
/// ```plain
/// var acc: tile<f32>[M, N] = <scalar>
///
/// for kt in range(lo, hi, st) {
///     var a = A[<static f32 slice>] // let works too
///     var b = B[<static f32 slice>]
///     acc += dot(a, b)
/// }
///
/// // Optional GEMM epilogue (alpha/beta scaling with prev_load load):
/// [let prev_load = C[<slice>]] // same slice as store target
/// C[<slice>] = [alpha *] acc [+ beta * prev_load] // acc's last use
/// ```
pub struct MatmulFusion<'a> {
    pub dims: &'a [Dim],
    /// Element type of the accumulator
    pub acc_scalar: Scalar,
    pub init: &'a Expr,
    pub kt: &'a str,
    pub start: &'a Expr,
    pub end: &'a Expr,
    pub step: Option<&'a Expr>,
    pub a_slice: &'a Expr,
    pub b_slice: &'a Expr,
    pub out: &'a str,
    pub out_subs: &'a [Sub],
    /// Number of statements this fusion consumes (3 minimum, 4 with prev_load)
    pub consumed: usize,
    /// Coefficient for acc in the epilogue; None is identity (1.0)
    pub alpha: Option<&'a Expr>,
    /// Previous-C load for the GEMM epilogue; None = pure accumulation
    pub prev_load: Option<GemmPrevLoad<'a>>,
}

/// The prev_load term in a GEMM epilogue: beta*prev_load loaded from global C.
pub struct GemmPrevLoad<'a> {
    /// Coefficient for prev_load; None is identity (1.0)
    pub beta: Option<&'a Expr>,
}

/// How an epilogue drain moves half a slab row to C.
#[derive(Clone, Copy)]
enum DrainMode {
    /// 16-byte 4xf32 vectors: f32 slab to an aligned f32 C.
    VecF32,
    /// 8-byte 4xf16 vectors, widened to f32 for the scaling and rounded back:
    /// f16 slab to an aligned f16 C. Coalesced, where the scalar path only
    /// writes 2 bytes per sector.
    VecF16,
    /// Element-wise: widen to f32, scale, round to C's element type (a no-op
    /// when both are f32).
    Scalar,
}

/// The loop-invariant state of a tensor-core epilogue drain. Both the WMMA
/// and the mma.sync epilogues park each finished 16x16 tile in the warp's
/// shared slab, then the lanes copy it out to C between barriers: half a slab
/// row per lane (row = lane / 2, column = lane % 2 * 8), as two 4-vectors.
struct SlabDrain<'c> {
    slab: MemVal<'c>,
    view: MemVal<'c>,
    /// The lane's slab row (its warp's tile origin plus lrow).
    srow: Value<'c, 'c>,
    /// The lane's row and column offsets within a 16x16 tile.
    lrow: Value<'c, 'c>,
    lcol: Value<'c, 'c>,
    /// Precomputed GEMM scaling (see [`Codegen::epilogue_scaling`]).
    alpha: Option<Value<'c, 'c>>,
    beta: Option<Value<'c, 'c>>,
    mode: DrainMode,
}

// Fused register-accumulator matmul
impl<'p, 'c> Codegen<'p, 'c> {
    pub(super) fn matmul_candidate<'a>(&self, stmts: &'a [Stmt]) -> Option<MatmulFusion<'a>> {
        // match: var acc, for kt, [optional let c_old,] store.
        let [
            Stmt::Var {
                name: acc,
                ty: Some(AstType::Tile(acc_scalar, dims)),
                value: init,
            },
            Stmt::For {
                var: kt,
                start,
                end,
                step,
                body,
            },
            tail @ ..,
        ] = stmts
        else {
            return None;
        };

        // f32 accumulation is the default; f16 accumulation is only valid on
        // the tensor cores, where it is a native WMMA mode.
        //
        // TODO(joa): must be architecture dependent
        let acc_scalar = *acc_scalar;
        if !matches!(acc_scalar, Scalar::F32 | Scalar::F16) {
            return None;
        }

        if kt == acc || !(matches!(init, Expr::Float(_)) || self.const_fold(init).is_some()) {
            return None;
        }

        // detect the optional [let prev_load = C[<slice>]] before final store
        let (target, epilogue_val, prev_load_let, consumed) = match tail {
            // 4-statement GEMM: let prev_load = C[...]; C[...] = f(acc, prev_load)
            [
                Stmt::Let {
                    name: prev_load_name,
                    ty: None,
                    value: _,
                },
                Stmt::Assign {
                    target,
                    op: AssignOp::Set,
                    value: epi,
                },
                ..,
            ] => (target, epi, Some(prev_load_name.as_str()), 4usize),
            // 3-statement plain / alpha-only: C[...] = [alpha *] acc
            [
                Stmt::Assign {
                    target,
                    op: AssignOp::Set,
                    value: epi,
                },
                ..,
            ] => (target, epi, None, 3usize),
            _ => return None,
        };

        // epilogue target: a slice of an f32 tensor, not mentioning acc. The slice's shape must be statically known and equal to acc.
        let Expr::Index { base, subs } = target else {
            return None;
        };

        let Expr::Var(out) = base.as_ref() else {
            return None;
        };
        let Some(Binding::Tensor(c)) = self.lookup(out) else {
            return None;
        };

        if !self.is_f16_or_f32(c.elem) || subs.iter().any(|s| s.uses_name(acc)) {
            // TODO(joa): see above, we bail if it's not f16/f16
            return None;
        }

        let out_shape = self.slice_static_shape(target)?;
        if out_shape != self.tile_shape(dims).ok()? {
            return None;
        }

        // parse the epilogue expression: acc | alpha*acc | alpha*acc+beta*prev_load
        // returns (alpha, prev_load) where prev_load = (beta, prev_load_name)
        let (alpha, prev_load) = {
            let scaled = |expr: &'a Expr, name: &str| -> Option<Coeff<'a>> {
                match expr {
                    Expr::Var(n) if n.as_str() == name => Some(None),
                    Expr::Binary {
                        op: BinOp::Mul,
                        lhs,
                        rhs,
                    } => match (lhs.as_ref(), rhs.as_ref()) {
                        (Expr::Var(n), s) | (s, Expr::Var(n)) if n.as_str() == name => {
                            Some(Some(s))
                        }
                        _ => None,
                    },
                    _ => None,
                }
            };

            match epilogue_val {
                // plain acc
                Expr::Var(n) if n.as_str() == acc => (None::<&Expr>, None),
                // alpha * acc or acc * alpha
                Expr::Binary { op: BinOp::Mul, .. } => (scaled(epilogue_val, acc)?, None),
                // alpha_term + beta_term (full GEMM)
                Expr::Binary {
                    op: BinOp::Add,
                    lhs,
                    rhs,
                } => {
                    let prev_load_n = prev_load_let?;
                    // try x+y and y+x
                    let try_split = |acc_side: &'a Expr, prev_load_side: &'a Expr|
                        -> Option<(Coeff<'a>, Option<(Coeff<'a>, &'a str)>)>
                    {
                        let alpha = scaled(acc_side, acc)?;
                        let beta = scaled(prev_load_side, prev_load_n)?;
                        Some((alpha, Some((beta, prev_load_n))))
                    };
                    try_split(lhs, rhs)
                        .or_else(|| try_split(rhs, lhs))
                        .unwrap_or((None, None))
                }
                _ => return None,
            }
        };

        // prev_load let statement must be used if the epilogue references it
        if prev_load_let.is_some() != prev_load.is_some() {
            return None;
        }

        // loop body: exactly two tensor slices, then one accumulating dot of exactly those two names
        let staged = |s: &'a Stmt| -> Option<(&'a str, &'a Expr)> {
            let (Stmt::Let {
                name,
                ty: None,
                value,
            }
            | Stmt::Var {
                name,
                ty: None,
                value,
            }) = s
            else {
                return None;
            };
            let Expr::Index { base, .. } = value else {
                return None;
            };
            let Expr::Var(src) = &**base else { return None };
            let Some(Binding::Tensor(t)) = self.lookup(src) else {
                return None;
            };
            // TODO(joa): f16 operands are only valid on the tensor-core path
            (self.is_f16_or_f32(t.elem) && !value.uses_name(acc)).then_some((name.as_str(), value))
        };

        let [
            sa,
            sb,
            Stmt::Assign {
                target: Expr::Var(t),
                op: AssignOp::Add,
                value: Expr::Call { callee, args },
            },
        ] = &body[..]
        else {
            return None;
        };

        let ((an, av), (bn, bv)) = (staged(sa)?, staged(sb)?);
        if t != acc || callee != "dot" || an == bn {
            return None;
        }

        let [Expr::Var(d0), Expr::Var(d1)] = &args[..] else {
            return None;
        };

        let (a_slice, b_slice) = if d0 == an && d1 == bn {
            (av, bv)
        } else if d0 == bn && d1 == an {
            (bv, av)
        } else {
            return None;
        };

        // shapes must be statically known and the same
        let acc_shape = self.tile_shape(dims).ok()?;
        let a_shape = self.slice_static_shape(a_slice)?;
        let b_shape = self.slice_static_shape(b_slice)?;
        let (&[m, n], &[am, ak], &[bk, bn2]) = (&acc_shape[..], &a_shape[..], &b_shape[..]) else {
            return None;
        };

        if am != m || bn2 != n || ak != bk {
            return None;
        }

        if self.wmma() && self.wmma_plan(m, n, ak).is_some() {
            // tensor-core path: f16 or f32 operands, accumulator, and C.
        } else {
            // vector path: f32 throughout (an f16 accumulator needs the tensor cores; otherwise fall back to the generic tile matmul).
            let a_elem = self.slice_tensor_elem(a_slice);
            let b_elem = self.slice_tensor_elem(b_slice);

            if acc_scalar != Scalar::F32
                || c.elem != self.f32_t
                || a_elem != Some(self.f32_t)
                || b_elem != Some(self.f32_t)
            {
                return None;
            }

            let (tm, tn) = self.sub_tile(m, n);
            if tm % 4 != 0 || tn % 4 != 0 {
                return None;
            }

            let (tiles_m, tiles_n) = (m / tm, n / tn);

            Self::lane_grid(tiles_m, tiles_n, tm, tn)?;

            if tiles_m * tiles_n > self.cta_threads {
                return None;
            }
        }

        // acc must die at the epilogue store
        let rest = &stmts[consumed..];
        if rest.iter().any(|s| s.uses_name(acc)) {
            return None;
        }

        // operand slices sit inside the k loop, which is not emitted yet.
        if self.slice_is_partial_within(a_slice, &[kt.as_str()])
            || self.slice_is_partial_within(b_slice, &[kt.as_str()])
            || self.slice_is_partial(target)
        {
            return None;
        }

        Some(MatmulFusion {
            dims,
            acc_scalar,
            init,
            kt,
            start,
            end,
            step: step.as_ref(),
            a_slice,
            b_slice,
            out,
            out_subs: subs,
            consumed,
            alpha,
            prev_load: prev_load.map(|(beta, _name)| GemmPrevLoad { beta }),
        })
    }

    /// Precomputes the alpha and beta operands of a GEMM epilogue
    /// (alpha*acc [+ beta*prev_load]), broadcast to vec_t when the store is
    /// aligned. Returns (None, None) for a plain accumulation.
    pub(super) fn epilogue_scaling(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        vec_t: Type<'c>,
        aligned: bool,
    ) -> Result<(Option<Value<'c, 'c>>, Option<Value<'c, 'c>>)> {
        let one = Expr::Float(1.0);
        let prep = |cg: &mut Self, e: &Expr| -> Result<Value<'c, 'c>> {
            let v = cg.emit_scalar(block, e)?;
            let v = cg.coerce(block, v, cg.f32_t)?; // TODO(joa): always f32 currently
            if aligned {
                cg.vec_broadcast(block, v, vec_t)
            } else {
                Ok(v)
            }
        };

        match (&p.prev_load, p.alpha) {
            (Some(epi), _) => Ok((
                Some(prep(self, p.alpha.unwrap_or(&one))?),
                Some(prep(self, epi.beta.unwrap_or(&one))?),
            )),
            (None, Some(a)) => Ok((Some(prep(self, a)?), None)),
            (None, None) => Ok((None, None)),
        }
    }

    /// Applies precomputed GEMM scaling to one accumulator value, producing
    /// alpha*acc + beta*prev_load, or alpha*acc, or just acc. prev_load (the
    /// prior C value) is only loaded when beta is present. Works the same on
    /// scalars and vectors since the arith ops are element-wise.
    pub(super) fn apply_scaling(
        &mut self,
        block: &Block<'c>,
        acc: Value<'c, 'c>,
        alpha: Option<Value<'c, 'c>>,
        beta: Option<Value<'c, 'c>>,
        load_prev: impl FnOnce(&mut Self) -> Result<Value<'c, 'c>>,
    ) -> Result<Value<'c, 'c>> {
        let f32 = self.f32_t;
        match (alpha, beta) {
            (Some(a), Some(b)) => {
                let prev = load_prev(self)?;
                let ar = self.push(block, self.elem_arith(BinOp::Mul, f32, a, acc)?)?;
                let br = self.push(block, self.elem_arith(BinOp::Mul, f32, b, prev)?)?;
                self.push(block, self.elem_arith(BinOp::Add, f32, ar, br)?)
            }
            (Some(a), None) => self.push(block, self.elem_arith(BinOp::Mul, f32, a, acc)?),
            _ => Ok(acc),
        }
    }

    /// Runs the fused matmul's k-loop. Both the vector and WMMA paths use it.
    ///
    /// With @pipeline on we double-buffer by unrolling two iterations at a
    /// time. The first half computes from one buffer pair while prefetching
    /// the next iteration into the other; the second half does the same for
    /// the odd iteration (its accumulators pass through a CTA-uniform scf.if).
    /// Without @pipeline we just stage, barrier, accumulate, barrier in place.
    ///
    /// Returns the finished accumulators. The closures: stage prefetches one
    /// iteration into a pair, half emits one pipelined half, mac accumulates
    /// one resident pair.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul_kloop(
        &mut self,
        block: &Block<'c>,
        bounds: (Value<'c, 'c>, Value<'c, 'c>, Value<'c, 'c>),
        regs: &[Value<'c, 'c>],
        a_bufs: &[MemVal<'c>],
        b_bufs: &[MemVal<'c>],
        stage: impl Fn(&mut Self, &Block<'c>, Value<'c, 'c>, &MemVal<'c>, &MemVal<'c>) -> Result<()>,
        half: impl Fn(
            &mut Self,
            &Block<'c>,
            Value<'c, 'c>,
            (&MemVal<'c>, &MemVal<'c>),
            (&MemVal<'c>, &MemVal<'c>),
            &[Value<'c, 'c>],
        ) -> Result<Vec<Value<'c, 'c>>>,
        mac: impl Fn(
            &mut Self,
            &Block<'c>,
            &MemVal<'c>,
            &MemVal<'c>,
            &[Value<'c, 'c>],
        ) -> Result<Vec<Value<'c, 'c>>>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let (lo, hi, st) = bounds;

        if !self.pipeline {
            return self.carry_loop(block, lo, hi, st, regs, |cg, body, kt, accs| {
                stage(cg, body, kt, &a_bufs[0], &b_bufs[0])?;

                // publish the staging, accumulate, then retire this
                // iteration's reads before the next one overwrites.
                cg.barrier(body)?;
                let next = mac(cg, body, &a_bufs[0], &b_bufs[0], accs)?;
                cg.barrier(body)?;
                Ok(next)
            });
        }

        // prologue: stage lo into the first pair, then unroll by two
        stage(self, block, lo, &a_bufs[0], &b_bufs[0])?;
        self.barrier(block)?;
        let two = self.const_index(block, 2)?;
        let st2 = self.muli(block, st, two)?;
        self.carry_loop(block, lo, hi, st2, regs, |cg, body, kt, accs| {
            let next = cg.addi(body, kt, st)?;
            let half_a = half(
                cg,
                body,
                next,
                (&a_bufs[0], &b_bufs[0]),
                (&a_bufs[1], &b_bufs[1]),
                accs,
            )?;

            // half_b: when iteration kt + st exists.
            let have_b = cg.push(
                body,
                arith::cmpi(cg.ctx, arith::CmpiPredicate::Slt, next, hi, cg.loc),
            )?;
            let then_block = Block::new(&[]);
            let next2 = cg.addi(&then_block, next, st)?;
            let half_b = half(
                cg,
                &then_block,
                next2,
                (&a_bufs[1], &b_bufs[1]),
                (&a_bufs[0], &b_bufs[0]),
                &half_a,
            )?;

            then_block.append_operation(scf::r#yield(&half_b, cg.loc));
            let then_region = Region::new();
            then_region.append_block(then_block);

            let else_block = Block::new(&[]);
            else_block.append_operation(scf::r#yield(&half_a, cg.loc));
            let else_region = Region::new();
            else_region.append_block(else_block);

            let types: Vec<Type<'c>> = half_a.iter().map(|v| v.r#type()).collect();
            let op =
                body.append_operation(scf::r#if(have_b, &types, then_region, else_region, cg.loc));

            (0..half_a.len())
                .map(|i| Ok(detach(op.result(i)?.into())))
                .collect()
        })
    }

    /// The warp's block origin (m0, n0) on a wm x wn warp grid of bm x bn
    /// element blocks, with surplus warps clamped onto the last block.
    /// Returns (tid, w, wt, m0, n0) so callers can derive lane coordinates.
    pub(super) fn warp_block_origin(
        &self,
        block: &Block<'c>,
        wm: i64,
        wn: i64,
        bm: i64,
        bn: i64,
    ) -> Result<(
        Value<'c, 'c>,
        Value<'c, 'c>,
        Value<'c, 'c>,
        Value<'c, 'c>,
        Value<'c, 'c>,
    )> {
        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
        let w = self.const_index(block, 32)?;
        let warp_id = self.divui(block, tid, w)?;
        let wmax = self.const_index(block, wm * wn - 1)?;
        let wt = self.minsi(block, warp_id, wmax)?;
        let wn_v = self.const_index(block, wn)?;
        let q = self.divui(block, wt, wn_v)?;
        let r = self.remui(block, wt, wn_v)?;
        let bm_v = self.const_index(block, bm)?;
        let bn_v = self.const_index(block, bn)?;
        let m0 = self.muli(block, q, bm_v)?;
        let n0 = self.muli(block, r, bn_v)?;

        Ok((tid, w, wt, m0, n0))
    }

    /// The (row, col) origin of 16x16 fragment (fi, fj) within the warp's
    /// block at (m0, n0).
    fn frag_origin(
        &self,
        block: &Block<'c>,
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        fi: i64,
        fj: i64,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let c_fi = self.const_index(block, fi * 16)?;
        let c_fj = self.const_index(block, fj * 16)?;
        Ok((self.addi(block, m0, c_fi)?, self.addi(block, n0, c_fj)?))
    }

    /// The fusion's static GEMM extents: the accumulator shape [m, n] and the
    /// k extent of the staged a-slice.
    fn fusion_dims(&mut self, p: &MatmulFusion<'_>) -> Result<(Vec<i64>, i64)> {
        let shape = self.tile_shape(p.dims)?;
        let kk = self
            .slice_static_shape(p.a_slice)
            .ok_or_else(|| anyhow!("matmul fusion without a static a-slice"))?[1];
        Ok((shape, kk))
    }

    pub(super) fn epilogue_view(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        shape: &[i64],
    ) -> Result<MemVal<'c>> {
        let Some(Binding::Tensor(cmv)) = self.lookup(p.out) else {
            bail!("matmul fusion target '{}' is not a tensor", p.out);
        };

        let view = self.emit_subview(block, &cmv, p.out_subs)?;

        self.check_shapes(&view.shape, shape, "matmul epilogue store")?;

        // matmul_candidate declines partial output tiles, so the register/WMMA
        // drains (which have no per-element bounds guard) never reach here.
        if view.is_masked() {
            bail!("internal: register matmul epilogue reached a partial output tile");
        }

        Ok(view)
    }

    /// Picks the f16 staging strategy for a tensor-core k-loop. Returns
    /// (stage_async, reg_stage): the first is true when cp.async is available
    /// for f16 operands, the second when register staging works (the tile has
    /// to divide evenly by the CTA).
    pub(super) fn staging_mode(
        &self,
        p: &MatmulFusion<'_>,
        m: i64,
        kk: i64,
        n: i64,
    ) -> (bool, bool) {
        let both_f16 = self.slice_tensor_elem(p.a_slice) == Some(self.f16_t)
            && self.slice_tensor_elem(p.b_slice) == Some(self.f16_t);
        let stage_async = self.cp_async() && both_f16; // TODO(joa): probably too conservative
        let reg_stage = !self.cp_async()
            && both_f16
            && self.reg_stage_divides(m, kk)
            && self.reg_stage_divides(kk, n);
        (stage_async, reg_stage)
    }

    pub(super) fn alloc_staging_pairs(
        &mut self,
        pairs: usize,
        mut make: impl FnMut(&mut Self) -> Result<(MemVal<'c>, MemVal<'c>)>,
    ) -> Result<(Vec<MemVal<'c>>, Vec<MemVal<'c>>)> {
        let mut a_bufs = Vec::with_capacity(pairs);
        let mut b_bufs = Vec::with_capacity(pairs);
        for _ in 0..pairs {
            let (a, b) = make(self)?;
            a_bufs.push(a);
            b_bufs.push(b);
        }
        Ok((a_bufs, b_bufs))
    }

    pub(super) fn wmma_const_frag(
        &self,
        block: &Block<'c>,
        init: Value<'c, 'c>,
        c_frag_t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("gpu.subgroup_mma_constant_matrix", self.loc)
                .add_operands(&[init])
                .add_results(&[c_frag_t])
                .build()?,
        )
    }

    pub(super) fn emit_register_matmul(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
    ) -> Result<()> {
        let (shape, kk) = self.fusion_dims(p)?;
        let (m, n) = (shape[0], shape[1]);

        if self.wmma() && self.wmma_plan(m, n, kk).is_some() {
            return if self.mma_sync() {
                self.emit_mma_sync_matmul(block, p)
            } else {
                self.emit_wmma_matmul(block, p)
            };
        }

        let (tm, tn) = self.sub_tile(m, n);
        let (tiles_m, tiles_n) = (m / tm, n / tn);
        let (lm, ln) = Self::lane_grid(tiles_m, tiles_n, tm, tn)
            .ok_or_else(|| anyhow!("matmul fusion without a lane grid"))?;

        let init = self.emit_scalar(block, p.init)?;
        let init = self.coerce(block, init, self.f32_t)?;

        // the lane's whole accumulator is one TMxTN vector
        let acc_t = Type::vector(&[tm as u64, tn as u64], self.f32_t);
        let regs = vec![self.vec_broadcast(block, init, acc_t)?];

        let (lo, hi, st, iv_div) = self.loop_bounds(block, p.start, p.end, p.step)?;

        // The lane's sub-tile origin: the warp's block origin (surplus warps
        // clamped onto the last block, as in tile_matmul) plus the lane's
        // position on the lm x ln lane grid of tm x tn sub-tiles.
        let (tid, w, _, wm0, wn0) =
            self.warp_block_origin(block, tiles_m / lm, tiles_n / ln, lm * tm, ln * tn)?;
        let lane = self.remui(block, tid, w)?;
        let ln_v = self.const_index(block, ln)?;
        let lane_m = self.divui(block, lane, ln_v)?;
        let lane_n = self.remui(block, lane, ln_v)?;
        let tm_v = self.const_index(block, tm)?;
        let tn_v = self.const_index(block, tn)?;
        let off_m = self.muli(block, lane_m, tm_v)?;
        let off_n = self.muli(block, lane_n, tn_v)?;
        let m0 = self.addi(block, wm0, off_m)?;
        let n0 = self.addi(block, wn0, off_n)?;

        // staging buffers; a k-major (kk x m).
        let pairs = if self.pipeline { 2 } else { 1 };
        let (a_bufs, b_bufs) = self.alloc_staging_pairs(pairs, |cg| {
            Ok((
                cg.alloc_tile_shaped(block, cg.f32_t, &[kk, m])?,
                cg.alloc_tile_shaped(block, cg.f32_t, &[kk, n])?,
            ))
        })?;

        let dims = (kk, tm, tn);
        let finals = self.matmul_kloop(
            block,
            (lo, hi, st),
            &regs,
            &a_bufs,
            &b_bufs,
            |cg, body, kt, a, b| cg.fused_stage(body, p, kt, iv_div, a, b, false),
            |cg, body, piv, cur, dst, accs| {
                cg.fused_half(body, p, iv_div, piv, hi, cur, dst, dims, m0, n0, accs)
            },
            |cg, body, a, b, accs| cg.register_mac(body, a, b, dims, m0, n0, accs),
        )?;

        // epilogue: each lane writes its finished sub-tile straight to C,
        // optionally applying alpha*acc + beta*prev_load
        let view = self.epilogue_view(block, p, &shape)?;
        let row_t = Type::vector(&[tn as u64], self.f32_t);

        // pre-compute alpha/beta broadcasts once
        let (alpha_row, beta_row) = self.epilogue_scaling(block, p, row_t, view.aligned)?;

        for i in 0..tm {
            let ci = self.const_index(block, i)?;
            let mi = self.addi(block, m0, ci)?;
            let row = self.vec_extract(block, finals[0], &[i], row_t)?;

            if view.aligned {
                let out_row = self.apply_scaling(block, row, alpha_row, beta_row, |cg| {
                    cg.vec_load(block, view.mem, &[mi, n0], row_t)
                })?;

                self.vec_store(block, out_row, view.mem, &[mi, n0])?;
            } else {
                for j in 0..tn {
                    let cj = self.const_index(block, j)?;
                    let nj = self.addi(block, n0, cj)?;
                    let e = self.vec_extract(block, row, &[j], self.f32_t)?;
                    let out_e = self.apply_scaling(block, e, alpha_row, beta_row, |cg| {
                        cg.push(block, memref::load(view.mem, &[mi, nj], cg.loc))
                    })?;

                    block.append_operation(memref::store(out_e, view.mem, &[mi, nj], self.loc));
                }
            }
        }
        Ok(())
    }

    /// Evaluates one staged operand slice with the fusion's k-loop iv bound
    /// to kt.
    fn emit_kt_slice(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        slice: &Expr,
        kt: Value<'c, 'c>,
        iv_div: i64,
    ) -> Result<MemVal<'c>> {
        self.scopes.push(HashMap::new());
        self.bind(
            p.kt,
            Binding::Let {
                value: kt,
                div: iv_div,
            },
        );
        let src = self.emit_expr(block, slice);
        self.scopes.pop();

        match src? {
            Rv::Tile(mv) => Ok(mv),
            Rv::Scalar(_) => bail!("staged value must be a tensor slice"),
        }
    }

    /// Stages one iteration's a (k-major) and b tiles into shared, without a
    /// barrier. With async_copy the transfers are cp.async and the caller owns
    /// the group and wait.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fused_stage(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        kt: Value<'c, 'c>,
        iv_div: i64,
        a_buf: &MemVal<'c>,
        b_buf: &MemVal<'c>,
        async_copy: bool,
    ) -> Result<()> {
        let a_src = self.emit_kt_slice(block, p, p.a_slice, kt, iv_div)?;
        let b_src = self.emit_kt_slice(block, p, p.b_slice, kt, iv_div)?;

        self.tile_copy_transposed(block, &a_src, a_buf, async_copy)?;
        self.tile_copy(block, &b_src, b_buf, false, async_copy)
    }

    /// One unrolled half of a pipelined loop: a guarded, barrier-free prefetch
    /// of iteration prefetch_iv, accumulation from the resident buffers via
    /// mac, then a closing wait and barrier. The barrier publishes the prefetch
    /// and retires the resident reads before the next half overwrites them.
    fn pipelined_half(
        &mut self,
        block: &Block<'c>,
        prefetch_iv: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        use_async: bool,
        stage: impl FnOnce(&mut Self, &Block<'c>) -> Result<()>,
        mac: impl FnOnce(&mut Self) -> Result<Vec<Value<'c, 'c>>>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        self.guarded_prefetch(block, prefetch_iv, hi, stage)?;

        // cp.async group outside the guard (an empty group is a no-op wait;
        // tokens can't cross scf.if regions).
        let group = if use_async {
            Some(self.async_create_group(block)?)
        } else {
            None
        };

        let next = mac(self)?;

        if let Some(group) = group {
            self.async_wait(block, group)?;
        }
        self.barrier(block)?;
        Ok(next)
    }

    /// The vector path's pipelined half: prefetch into dst, register-MAC from
    /// cur (see [`Self::pipelined_half`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fused_half(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        iv_div: i64,
        prefetch_iv: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        cur: (&MemVal<'c>, &MemVal<'c>),
        dst: (&MemVal<'c>, &MemVal<'c>),
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let use_async = self.cp_async();
        self.pipelined_half(
            block,
            prefetch_iv,
            hi,
            use_async,
            |cg, then| cg.fused_stage(then, p, prefetch_iv, iv_div, dst.0, dst.1, use_async),
            |cg| cg.register_mac(block, cur.0, cur.1, dims, m0, n0, accs),
        )
    }

    /// The fused k-loop, done as a vector contraction. The lane's accumulator
    /// rides the loop as one vector<TMxTNxf32> iter_arg. Each iteration grabs a
    /// CxTM and a CxTN chunk and folds them in with a single vector.contract
    /// over k. C is the largest of 4, 2, 1 that divides TILE_K; a_t and b_sh
    /// are both k-major, so the chunk rows are contiguous loads.
    ///
    /// Doing C k-steps per iteration cuts the loop overhead by C. The contract
    /// lowers to broadcast + vector.fma rank-1 updates, which on NVPTX is the
    /// same fma.rn stream as the scalar form, just without the extract/insert
    /// noise in the IR.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn register_mac(
        &mut self,
        block: &Block<'c>,
        a_t: &MemVal<'c>,
        b_sh: &MemVal<'c>,
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let (kk, tm, tn) = dims;
        let chunk = [4, 2, 1]
            .into_iter()
            .find(|c| kk % c == 0)
            .expect("1 divides everything");

        let a_row_t = Type::vector(&[tm as u64], self.f32_t);
        let b_row_t = Type::vector(&[tn as u64], self.f32_t);
        let lhs_t = Type::vector(&[chunk as u64, tm as u64], self.f32_t);
        let rhs_t = Type::vector(&[chunk as u64, tn as u64], self.f32_t);
        let lo = self.const_index(block, 0)?;
        let hi = self.const_index(block, kk)?;
        let st = self.const_index(block, chunk)?;

        self.carry_loop(block, lo, hi, st, accs, |cg, lblk, k, accs| {
            let zero = cg.zero_scalar(lblk, cg.f32_t)?;
            let mut lhs = cg.vec_broadcast(lblk, zero, lhs_t)?;
            let mut rhs = cg.vec_broadcast(lblk, zero, rhs_t)?;

            for j in 0..chunk {
                let c = cg.const_index(lblk, j)?;
                let kj = cg.addi(lblk, k, c)?;

                let a_row = cg.vec_load(lblk, a_t.mem, &[kj, m0], a_row_t)?;
                lhs = cg.vec_insert(lblk, a_row, lhs, &[j])?;

                let b_row = cg.vec_load(lblk, b_sh.mem, &[kj, n0], b_row_t)?;
                rhs = cg.vec_insert(lblk, b_row, rhs, &[j])?;
            }

            Ok(vec![cg.vec_contract(lblk, lhs, rhs, accs[0], true)?])
        })
    }
}

// @tensorcore

impl<'p, 'c> Codegen<'p, 'c> {
    /// Warp grid (wm x wn, with wm*wn the launch ABI's warp count) laid over
    /// the matmul's 16x16-fragment grid. Returns None when the shape doesn't
    /// split into whole fragments owned by whole warps. We pick the
    /// factorization that loads the fewest fragments per k-step (fm + fn for
    /// fm*fn computes), breaking ties toward wider warp blocks since those give
    /// contiguous b reads and epilogue stores.
    pub(super) fn wmma_plan(&self, m: i64, n: i64, kk: i64) -> Option<(i64, i64)> {
        if m == DYN || n == DYN || kk == DYN || m % 16 != 0 || n % 16 != 0 || kk % 16 != 0 {
            return None;
        }

        let warps = self.cta_threads / 32;
        let (gm, gn) = (m / 16, n / 16);

        (1..=warps)
            .filter(|wm| warps % wm == 0)
            .map(|wm| (wm, warps / wm))
            .filter(|&(wm, wn)| gm % wm == 0 && gn % wn == 0)
            .min_by_key(|&(wm, wn)| (gm / wm + gn / wn, gm / wm))
    }

    /// Whether to pad the WMMA staging buffers.
    ///
    /// More padding means a larger CTA footprint, so fewer CTAs fit per SM. We
    /// skip it when kernel registers (@launch) are the limiting factor instead.
    fn wmma_should_pad(&self, m: i64, kk: i64, n: i64, acc_elem: Type<'c>, pairs: i64) -> bool {
        // TODO(joa): @autotune this
        let Some(maxnreg) = self.launch.and_then(|l| l.max_nreg) else {
            return true;
        };

        let cfg = &self.base.gpu_config;
        let sm_bytes = cfg.smem_per_sm() as i64;
        let warps = self.cta_threads / 32;
        let warp_blocks = (cfg.max_warps_per_sm() as i64 / warps).max(1);
        let reg_blocks = (cfg.regs_per_sm() as i64 / (self.cta_threads * maxnreg)).max(1);
        let acc_bytes = if acc_elem == self.f16_t { 2 } else { 4 }; // TODO(joa): support for other data types
        let slab = warps * 16 * 16 * acc_bytes;

        // f16 staging, both operands, doubled under @pipeline; plus the slab
        let smem = |ak: i64, bn: i64| pairs * (m * ak + kk * bn) * 2 + slab;
        let blocks = |bytes: i64| (sm_bytes / bytes).min(warp_blocks).min(reg_blocks);

        blocks(smem(kk + WMMA_SMEM_PAD, n + WMMA_SMEM_PAD)) >= blocks(smem(kk, n))
    }

    pub(super) fn wmma_c_type(&self, elem: Type<'c>) -> Result<Type<'c>> {
        if elem == self.f16_t {
            self.parse_type("!gpu.mma_matrix<16x16xf16, \"COp\">")
        } else if elem == self.f32_t {
            self.parse_type(WMMA_C)
        } else {
            bail!("WMMA accumulation needs an f16 or f32 type, got {elem}")
        }
    }

    /// The lane's read slice of its warp's 16x16 slab tile (see
    /// [`SlabDrain`]). Returns (srow, lrow, lcol) for the warp tile at slab
    /// row slab0.
    fn slab_lane_slice(
        &self,
        block: &Block<'c>,
        slab0: Value<'c, 'c>,
        lane: Value<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>, Value<'c, 'c>)> {
        let two = self.const_index(block, 2)?;
        let lrow = self.divui(block, lane, two)?;
        let lhalf = self.remui(block, lane, two)?;
        let eight = self.const_index(block, 8)?;
        let lcol = self.muli(block, lhalf, eight)?;
        let srow = self.addi(block, slab0, lrow)?;
        Ok((srow, lrow, lcol))
    }

    /// Copies the warp's slab out to C for fragment (fi, fj), applying
    /// alpha*acc + beta*prev_load. The caller owns the surrounding barriers.
    fn drain_slab_tile(
        &mut self,
        block: &Block<'c>,
        d: &SlabDrain<'c>,
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        fi: i64,
        fj: i64,
    ) -> Result<()> {
        let (mb, nb) = self.frag_origin(block, m0, n0, fi, fj)?;
        let mi = self.addi(block, mb, d.lrow)?;
        let nbl = self.addi(block, nb, d.lcol)?;
        let row_t = Type::vector(&[4], self.f32_t);
        let row_f16_t = Type::vector(&[4], self.f16_t);

        for h in 0..2 {
            let c_h = self.const_index(block, h * 4)?;
            let sc = self.addi(block, d.lcol, c_h)?;
            let nj = self.addi(block, nbl, c_h)?;

            match d.mode {
                DrainMode::VecF32 => {
                    let v = self.vec_load(block, d.slab.mem, &[d.srow, sc], row_t)?;
                    let out_v = self.apply_scaling(block, v, d.alpha, d.beta, |cg| {
                        cg.vec_load(block, d.view.mem, &[mi, nj], row_t)
                    })?;

                    self.vec_store(block, out_v, d.view.mem, &[mi, nj])?;
                }
                DrainMode::VecF16 => {
                    let raw = self.vec_load_al(block, d.slab.mem, &[d.srow, sc], row_f16_t, 8)?;
                    let acc_v = self.vec_extf(block, raw, row_t)?;
                    let out_v = self.apply_scaling(block, acc_v, d.alpha, d.beta, |cg| {
                        let c = cg.vec_load_al(block, d.view.mem, &[mi, nj], row_f16_t, 8)?;
                        cg.vec_extf(block, c, row_t)
                    })?;
                    let out_v = self.vec_truncf(block, out_v, row_f16_t)?;

                    self.vec_store_al(block, out_v, d.view.mem, &[mi, nj], 8)?;
                }
                DrainMode::Scalar => {
                    for e in 0..4 {
                        let c_e = self.const_index(block, e)?;
                        let sce = self.addi(block, sc, c_e)?;
                        let ne = self.addi(block, nj, c_e)?;
                        let x = self.load_as(block, d.slab.mem, &[d.srow, sce], self.f32_t)?;
                        let out_e = self.apply_scaling(block, x, d.alpha, d.beta, |cg| {
                            cg.load_as(block, d.view.mem, &[mi, ne], cg.f32_t)
                        })?;
                        let out_e = self.coerce(block, out_e, d.view.elem)?;

                        block.append_operation(memref::store(
                            out_e,
                            d.view.mem,
                            &[mi, ne],
                            self.loc,
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Drives the tensor-core k-loop shared by the WMMA and mma.sync paths:
    /// the loop bounds, the warp's fragment-block origin (surplus warps clamp
    /// onto the last block), the staging pairs (a as [m, kk], b as [kk, n],
    /// via alloc, double-buffered under @pipeline), and [`Self::matmul_kloop`]
    /// with WMMA staging and the tensor-core MAC. Returns the finished
    /// accumulators and the origin (as [`Self::warp_block_origin`]) for the
    /// epilogue.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn tc_matmul_kloop(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        (m, n, kk): (i64, i64, i64),
        (wm, wn): (i64, i64),
        regs: &[Value<'c, 'c>],
        mma_sync: bool,
        alloc: impl Fn(&mut Self, &Block<'c>, &[i64]) -> Result<MemVal<'c>>,
    ) -> Result<(
        Vec<Value<'c, 'c>>,
        (
            Value<'c, 'c>,
            Value<'c, 'c>,
            Value<'c, 'c>,
            Value<'c, 'c>,
            Value<'c, 'c>,
        ),
    )> {
        let (fm, fnn) = ((m / 16) / wm, (n / 16) / wn);
        let dims = (kk, fm, fnn);

        let (lo, hi, st, iv_div) = self.loop_bounds(block, p.start, p.end, p.step)?;
        let origin = self.warp_block_origin(block, wm, wn, fm * 16, fnn * 16)?;
        let (_, _, _, m0, n0) = origin;

        let pairs = if self.pipeline { 2 } else { 1 };
        let (a_bufs, b_bufs) = self.alloc_staging_pairs(pairs, |cg| {
            Ok((alloc(cg, block, &[m, kk])?, alloc(cg, block, &[kk, n])?))
        })?;

        let (stage_async, reg_stage) = self.staging_mode(p, m, kk, n);

        let finals = self.matmul_kloop(
            block,
            (lo, hi, st),
            regs,
            &a_bufs,
            &b_bufs,
            |cg, body, kt, a, b| cg.wmma_stage(body, p, kt, iv_div, a, b, false),
            |cg, body, piv, cur, dst, accs| {
                cg.wmma_half(
                    body,
                    p,
                    iv_div,
                    piv,
                    hi,
                    st,
                    cur,
                    dst,
                    dims,
                    m0,
                    n0,
                    accs,
                    stage_async,
                    reg_stage,
                    mma_sync,
                )
            },
            |cg, body, a, b, accs| cg.tc_mac(body, a, b, dims, m0, n0, accs, false, mma_sync),
        )?;

        Ok((finals, origin))
    }

    pub(super) fn emit_wmma_matmul(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
    ) -> Result<()> {
        let (shape, kk) = self.fusion_dims(p)?;
        let (m, n) = (shape[0], shape[1]);
        let (wm, wn) = self
            .wmma_plan(m, n, kk)
            .ok_or_else(|| anyhow!("wmma matmul without a warp grid"))?;

        // fragments per warp: fmxfn block of the 16x16-fragment grid
        let (fm, fnn) = ((m / 16) / wm, (n / 16) / wn);

        // the warp's accumulators, seeded with the init scalar rounded to the accumulator type
        let acc_elem = self.scalar_type(p.acc_scalar);
        let init = self.emit_scalar(block, p.init)?;
        let init = self.coerce(block, init, acc_elem)?;
        let c_frag_t = self.wmma_c_type(acc_elem)?;
        let mut regs = Vec::with_capacity((fm * fnn) as usize);

        for _ in 0..fm * fnn {
            regs.push(self.wmma_const_frag(block, init, c_frag_t)?);
        }

        // f16 staging, padded against bank conflicts when the CTA budget allows
        let pairs = if self.pipeline { 2 } else { 1 };
        let pad = self.wmma_should_pad(m, kk, n, acc_elem, pairs as i64);
        let alloc = if pad {
            Self::alloc_tile_padded
        } else {
            Self::alloc_tile_shaped
        };

        let (finals, (tid, w, wt, m0, n0)) = self.tc_matmul_kloop(
            block,
            p,
            (m, n, kk),
            (wm, wn),
            &regs,
            false,
            |cg, blk, shape| alloc(cg, blk, cg.f16_t, shape),
        )?;

        // epilogue: each warp drains its fragments through its 16x16 slab,
        // optionally applying alpha*acc + beta*prev_load
        let view = self.epilogue_view(block, p, &shape)?;

        // The slab matches the accumulator fragments' element type; the f32
        // vector drain only applies when both the slab and C are f32.
        let slab = self.alloc_tile_shaped(block, acc_elem, &[(self.cta_threads / 32) * 16, 16])?;
        let slab_f32 = acc_elem == self.f32_t;
        let sixteen = self.const_index(block, 16)?;
        let slab0 = self.muli(block, wt, sixteen)?;
        let zero = self.const_index(block, 0)?;

        let lane = self.remui(block, tid, w)?;
        let (srow, lrow, lcol) = self.slab_lane_slice(block, slab0, lane)?;
        let row_t = Type::vector(&[4], self.f32_t);

        // The 128-bit vector drain needs an f32 slab and a 16B-aligned f32 C
        // row (aligned on its own is element-agnostic: an f16 C is 8B-aligned
        // but takes the scalar, rounding store).
        let vec_drain = slab_f32 && view.elem == self.f32_t && view.aligned;

        // pre-compute alpha/beta broadcasts once
        let (alpha, beta) = self.epilogue_scaling(block, p, row_t, vec_drain)?;
        let drain = SlabDrain {
            slab: slab.clone(),
            view,
            srow,
            lrow,
            lcol,
            alpha,
            beta,
            mode: if vec_drain {
                DrainMode::VecF32
            } else {
                DrainMode::Scalar
            },
        };

        for fi in 0..fm {
            for fj in 0..fnn {
                let frag = finals[(fi * fnn + fj) as usize];

                block.append_operation(
                    OperationBuilder::new("gpu.subgroup_mma_store_matrix", self.loc)
                        .add_operands(&[frag, slab.mem, slab0, zero])
                        .add_attributes(&[(
                            self.id("leadDimension"),
                            IntegerAttribute::new(self.index_t, 16).into(),
                        )])
                        .build()?,
                );

                // publish the slab and read it out before the next iteration overwrites it
                self.barrier(block)?;
                self.drain_slab_tile(block, &drain, m0, n0, fi, fj)?;
                self.barrier(block)?;
            }
        }
        Ok(())
    }

    pub(super) fn emit_mma_sync_matmul(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
    ) -> Result<()> {
        let (shape, kk) = self.fusion_dims(p)?;
        let (m, n) = (shape[0], shape[1]);
        let (wm, wn) = self
            .wmma_plan(m, n, kk)
            .ok_or_else(|| anyhow!("mma.sync matmul without a warp grid"))?;
        let (fm, fnn) = ((m / 16) / wm, (n / 16) / wn);

        // the warp's accumulators: one m16n8 vector<2x2x{acc}> per (fi, fj, n8),
        // initialized with the scalar rounded to the accumulator type
        let acc_elem = self.scalar_type(p.acc_scalar);
        let init = self.emit_scalar(block, p.init)?;
        let init = self.coerce(block, init, acc_elem)?;
        let acc_t = Type::vector(&[2, 2], acc_elem);
        let seed = self.vec_broadcast(block, init, acc_t)?;
        let regs = vec![seed; (fm * fnn * 2) as usize];

        // f16 staging, XOR-swizzled (not padded): the ldmatrix reads stride
        // consecutive rows, which alias shared banks unpadded, so the column
        // is permuted per row to spread them, at zero extra SM. The store and
        // load MUST apply the same swizzle.
        let (finals, (tid, _, wt, m0, n0)) = self.tc_matmul_kloop(
            block,
            p,
            (m, n, kk),
            (wm, wn),
            &regs,
            true,
            |cg, blk, shape| cg.alloc_tile_swizzled(blk, cg.f16_t, shape),
        )?;

        self.mma_sync_epilogue(
            block,
            p,
            &shape,
            finals,
            (fm, fnn),
            acc_elem,
            tid,
            wt,
            m0,
            n0,
        )
    }

    /// Drains the mma.sync accumulators to C. Each lane's m16n8 D fragment
    /// (vector<2x2x{acc}>) maps to known (row, col) pairs of the 16x8 tile
    /// (row = laneId/4 [+8], col = 2*(laneId%4) [+1]). The warp scatters them
    /// into its private 16x16 shared slab, then the lanes copy slab to C between
    /// barriers, applying alpha*acc + beta*prev_load. The drain matches the WMMA
    /// epilogue (lane = half a slab row, two 4-vectors) so the f32 fast path and
    /// the f16 rounding path can be shared.
    #[allow(clippy::too_many_arguments)]
    fn mma_sync_epilogue(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        shape: &[i64],
        finals: Vec<Value<'c, 'c>>,
        warp_frags: (i64, i64),
        acc_elem: Type<'c>,
        tid: Value<'c, 'c>,
        wt: Value<'c, 'c>,
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
    ) -> Result<()> {
        let (fm, fnn) = warp_frags;
        let view = self.epilogue_view(block, p, shape)?;

        let slab = self.alloc_tile_shaped(block, acc_elem, &[(self.cta_threads / 32) * 16, 16])?;
        let slab_f32 = acc_elem == self.f32_t;
        let w = self.const_index(block, 32)?;
        let sixteen = self.const_index(block, 16)?;
        let slab0 = self.muli(block, wt, sixteen)?;

        // The lane's D-fragment coordinates within an m16n8 tile.
        let lane = self.remui(block, tid, w)?;
        let eight = self.const_index(block, 8)?;
        let (gid, dcol) = self.mma_sync_dfrag_base(block, lane)?;
        let srow_d = self.addi(block, slab0, gid)?;

        let (srow, lrow, lcol) = self.slab_lane_slice(block, slab0, lane)?;
        let row_t = Type::vector(&[4], self.f32_t);

        // Coalesced 4-wide drains: an f32 slab straight to an f32 C, or an f16
        // slab through an f32 scale to an f16 C (the gemm_fp16.ph case). Both
        // need an aligned (multiple-of-4 row pitch) output, otherwise we drain
        // scalar. The scaling vectors are f32 either way (the f16 path scales
        // after extf).
        let vec_f32 = slab_f32 && view.elem == self.f32_t && view.aligned;
        let vec_f16 = !slab_f32 && view.elem == self.f16_t && view.aligned;
        let (alpha, beta) = self.epilogue_scaling(block, p, row_t, vec_f32 || vec_f16)?;
        let drain = SlabDrain {
            slab: slab.clone(),
            view,
            srow,
            lrow,
            lcol,
            alpha,
            beta,
            mode: if vec_f32 {
                DrainMode::VecF32
            } else if vec_f16 {
                DrainMode::VecF16
            } else {
                DrainMode::Scalar
            },
        };

        for fi in 0..fm {
            for fj in 0..fnn {
                // Scatter both n8 D fragments into the warp's 16x16 slab.
                for nn in 0..2 {
                    let frag = finals[((fi * fnn + fj) * 2 + nn) as usize];
                    let ncol = self.const_index(block, nn * 8)?;
                    let scol0 = self.addi(block, ncol, dcol)?;
                    for di in 0..2 {
                        let rbase = if di == 0 {
                            srow_d
                        } else {
                            self.addi(block, srow_d, eight)?
                        };
                        for dj in 0..2 {
                            let e = self.vec_extract(block, frag, &[di, dj], acc_elem)?;
                            let cj = self.const_index(block, dj)?;
                            let sc = self.addi(block, scol0, cj)?;
                            block.append_operation(memref::store(
                                e,
                                slab.mem,
                                &[rbase, sc],
                                self.loc,
                            ));
                        }
                    }
                }

                // publish the scatter, drain it to C, and retire the reads
                // before the next fragment overwrites the slab
                self.barrier(block)?;
                self.drain_slab_tile(block, &drain, m0, n0, fi, fj)?;
                self.barrier(block)?;
            }
        }

        Ok(())
    }

    /// The lane's m16n8 D-fragment base within an mma.sync tile. The row is
    /// laneId / 4 (the second 8-row half adds 8) and the column is
    /// 2 * (laneId % 4) (the second element of a pair adds 1).
    fn mma_sync_dfrag_base(
        &self,
        block: &Block<'c>,
        lane: Value<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let four = self.const_index(block, 4)?;
        let two = self.const_index(block, 2)?;
        let gid = self.divui(block, lane, four)?;
        let tig = self.remui(block, lane, four)?;
        let dcol = self.muli(block, tig, two)?;

        Ok((gid, dcol))
    }

    /// The mma.sync counterpart of the legacy [`Self::wmma_dot`] body (taken
    /// when mma_sync() holds): swizzled f16 staging, per-lane vector<2x2xf32>
    /// accumulators folded with nvgpu.mma.sync, and a direct scatter of the D
    /// fragments to the shared out tile. += seeds the accumulators from out
    /// (folding the running sum into the MAC), the same as the WMMA path.
    #[allow(clippy::too_many_arguments)]
    fn mma_sync_dot(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        transpose_b: bool,
        accumulate: bool,
        wm: i64,
        wn: i64,
    ) -> Result<bool> {
        let (m, n) = (out.shape[0], out.shape[1]);
        let kk = a.shape[1];
        let (fm, fnn) = ((m / 16) / wm, (n / 16) / wn);
        let dims = (kk, fm, fnn);

        // Swizzled (not padded) f16 staging, matching each operand's natural
        // layout: a as [m, k], b as [k, n] (NN) or [n, k] (NT). The ldmatrix
        // reads stride consecutive rows, which alias shared banks when unpadded.
        // The swizzle permutes the column per row, the same way on store and
        // load, so it spreads the banks out for free.
        let (a_buf, a_hoisted) = self.dot_stage(block, a, &[m, kk], true)?;
        let (b_buf, b_hoisted) = self.dot_stage(block, b, &b.shape.clone(), true)?;
        self.barrier(block)?;

        // The warp's fragment-block origin; surplus warps clamp onto the last.
        let (_, _, _, m0, n0) = self.warp_block_origin(block, wm, wn, fm * 16, fnn * 16)?;

        // Accumulators: the running out D fragments for +=, else zero.
        let acc_t = Type::vector(&[2, 2], self.f32_t);
        let regs = if accumulate {
            self.mma_sync_load_frags(block, out, (fm, fnn), m0, n0)?
        } else {
            let zero = self.zero_scalar(block, self.f32_t)?;
            let seed = self.vec_broadcast(block, zero, acc_t)?;

            vec![seed; (fm * fnn * 2) as usize]
        };

        let finals = self.mma_sync_mac(block, &a_buf, &b_buf, dims, m0, n0, &regs, transpose_b)?;

        self.mma_sync_store_frags(block, out, &finals, (fm, fnn), m0, n0)?;
        self.barrier(block)?;

        // The staging is dead past the MAC; the closing barrier orders its
        // reads before any pooled reuse. Hoisted buffers outlive the loop.
        if !a_hoisted {
            self.release(&a_buf);
        }
        if !b_hoisted {
            self.release(&b_buf);
        }

        Ok(true)
    }

    /// Walks the warp's m16n8 D fragments over the tile at (m0, n0). For each
    /// (fi, fj, n8) fragment it computes the lane's four element addresses
    /// (see [`Self::mma_sync_dfrag_base`]) and yields the fragment's register
    /// index with each element's (di, dj) position and [row, col] address.
    /// The scatter and its inverse gather share this walk, so their
    /// addressing cannot drift apart.
    pub(super) fn for_each_dfrag(
        &mut self,
        block: &Block<'c>,
        warp_frags: (i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        mut f: impl FnMut(&mut Self, usize, &[([i64; 2], [Value<'c, 'c>; 2])]) -> Result<()>,
    ) -> Result<()> {
        let (fm, fnn) = warp_frags;
        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
        let w = self.const_index(block, 32)?;
        let eight = self.const_index(block, 8)?;
        let lane = self.remui(block, tid, w)?;
        let (gid, dcol) = self.mma_sync_dfrag_base(block, lane)?;

        for fi in 0..fm {
            for fj in 0..fnn {
                let (mb, nb) = self.frag_origin(block, m0, n0, fi, fj)?;
                let mrow0 = self.addi(block, mb, gid)?;

                for nn in 0..2 {
                    let ncol = self.const_index(block, nn * 8)?;
                    let ncb = self.addi(block, nb, ncol)?;
                    let ncb = self.addi(block, ncb, dcol)?;
                    let mut elems = Vec::with_capacity(4);

                    for di in 0..2 {
                        let mrow = if di == 0 {
                            mrow0
                        } else {
                            self.addi(block, mrow0, eight)?
                        };

                        for dj in 0..2 {
                            let cj = self.const_index(block, dj)?;
                            let col = self.addi(block, ncb, cj)?;
                            elems.push(([di, dj], [mrow, col]));
                        }
                    }

                    f(self, ((fi * fnn + fj) * 2 + nn) as usize, &elems)?;
                }
            }
        }

        Ok(())
    }

    /// Scatters each lane's m16n8 D fragment (vector<2x2xf32>) straight to its
    /// (row, col) spot in the shared out tile. This is the dot's barrier-free
    /// publish, the mma.sync take on a straight-to-shared
    /// subgroup_mma_store_matrix. The caller barriers afterward.
    fn mma_sync_store_frags(
        &mut self,
        block: &Block<'c>,
        dst: &MemVal<'c>,
        finals: &[Value<'c, 'c>],
        warp_frags: (i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
    ) -> Result<()> {
        self.for_each_dfrag(block, warp_frags, m0, n0, |cg, i, elems| {
            for ([di, dj], addr) in elems {
                let e = cg.vec_extract(block, finals[i], &[*di, *dj], cg.f32_t)?;
                block.append_operation(memref::store(e, dst.mem, addr, cg.loc));
            }
            Ok(())
        })
    }

    /// Seeds the per-lane D fragments from the shared out tile (the += running
    /// sum), the inverse of [`Self::mma_sync_store_frags`]: each lane reads its
    /// four (row, col) elements per m16n8 tile into a vector<2x2xf32>.
    fn mma_sync_load_frags(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        warp_frags: (i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let acc_t = Type::vector(&[2, 2], self.f32_t);
        let mut regs = Vec::with_capacity((warp_frags.0 * warp_frags.1 * 2) as usize);

        self.for_each_dfrag(block, warp_frags, m0, n0, |cg, _, elems| {
            let zero = cg.zero_scalar(block, cg.f32_t)?;
            let mut frag = cg.vec_broadcast(block, zero, acc_t)?;

            for ([di, dj], addr) in elems {
                let x = cg.load_as(block, src.mem, addr, cg.f32_t)?;
                frag = cg.vec_insert(block, x, frag, &[*di, *dj])?;
            }

            regs.push(frag);
            Ok(())
        })?;

        Ok(regs)
    }

    /// Tile-by-tile matmul on the tensor cores: out = a @ b (NN), or
    /// out = a @ b.T (NT, transpose_b), contracting the last dim of both
    /// operands. f16 inputs, f32 accumulate; the f32 result goes straight into
    /// the (shared) out tile. Returns false without emitting anything when
    /// @tensorcore is off or the shapes don't split into whole 16x16 tiles
    /// owned by whole warps, so the caller can fall back to the vector path.
    ///
    /// With accumulate (out += a @ b) the accumulator fragments start from out
    /// instead of zero, folding the running sum into the tensor-core MAC. This
    /// relies on the launch matching cta_threads (no surplus warps), which the
    /// warp grid (wm * wn == warps) and the launch ABI guarantee; a surplus
    /// warp would double-count.
    pub(super) fn wmma_dot(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        transpose_b: bool,
        accumulate: bool,
    ) -> Result<bool> {
        let (m, n) = (out.shape[0], out.shape[1]);
        let kk = a.shape[1];
        let Some((wm, wn)) = self.wmma().then(|| self.wmma_plan(m, n, kk)).flatten() else {
            return Ok(false);
        };

        // The tensor cores accumulate in f32: the output (and so the running
        // sum for +=) must be f32. Operands may be f16 or f32; f32 is
        // rounded to f16 when staged. Anything else stays on the vector path.
        if out.elem != self.f32_t || !self.is_f16_or_f32(a.elem) || !self.is_f16_or_f32(b.elem) {
            return Ok(false);
        }
        let (fm, fnn) = ((m / 16) / wm, (n / 16) / wn);
        let dims = (kk, fm, fnn);

        // thread-level mma.sync + ldmatrix
        if self.mma_sync() {
            return self.mma_sync_dot(block, a, b, out, transpose_b, accumulate, wm, wn);
        }

        // f16 staging buffers, mirroring each operand's natural layout: a as
        // [m, k], b as [k, n] (NN) or [n, k] (NT). No transposing copy; the
        // NT B fragment is transposed in the wmma load instead.
        let (a_buf, a_hoisted) = self.dot_stage(block, a, &[m, kk], false)?;
        let (b_buf, b_hoisted) = self.dot_stage(block, b, &b.shape.clone(), false)?;
        self.barrier(block)?;

        // The warp's fragment-block origin; surplus warps clamp onto the
        // last block and recompute it (identical writes, benign).
        let (_, _, _, m0, n0) = self.warp_block_origin(block, wm, wn, fm * 16, fnn * 16)?;

        // Accumulators: the running out fragments for +=, else zero.
        let c_frag_t = self.parse_type(WMMA_C)?;
        let mut regs = Vec::with_capacity((fm * fnn) as usize);

        if accumulate {
            for fi in 0..fm {
                for fj in 0..fnn {
                    let (mb, nb) = self.frag_origin(block, m0, n0, fi, fj)?;
                    regs.push(self.wmma_load(block, out, &[mb, nb], c_frag_t, false)?);
                }
            }
        } else {
            let zero = self.zero_scalar(block, self.f32_t)?;

            for _ in 0..fm * fnn {
                regs.push(self.wmma_const_frag(block, zero, c_frag_t)?);
            }
        }
        let finals = self.wmma_mac(block, &a_buf, &b_buf, dims, m0, n0, &regs, transpose_b)?;

        // Each warp stores its fragments straight to its disjoint slice of
        // the shared output (lead dimension = the tile's row stride). A
        // closing barrier publishes them before downstream reads.
        for fi in 0..fm {
            for fj in 0..fnn {
                let frag = finals[(fi * fnn + fj) as usize];
                let (mb, nb) = self.frag_origin(block, m0, n0, fi, fj)?;
                block.append_operation(
                    OperationBuilder::new("gpu.subgroup_mma_store_matrix", self.loc)
                        .add_operands(&[frag, out.mem, mb, nb])
                        .add_attributes(&[(
                            self.id("leadDimension"),
                            IntegerAttribute::new(self.index_t, n).into(),
                        )])
                        .build()?,
                );
            }
        }
        self.barrier(block)?;

        // The staging is dead past the MAC; the closing barrier orders its
        // reads before any pooled reuse. Hoisted buffers outlive the loop.
        if !a_hoisted {
            self.release(&a_buf);
        }
        if !b_hoisted {
            self.release(&b_buf);
        }
        Ok(true)
    }

    /// Returns the staged f16 shared buffer for one tile-dot operand: the
    /// preheader copy when an enclosing loop hoisted this operand (see
    /// codegen/hoist.rs), else a fresh pooled buffer staged here without a
    /// barrier. The flag is true for the hoisted case, where the caller
    /// must skip the release; the loop epilogue owns that buffer.
    pub(super) fn dot_stage(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        shape: &[i64],
        swizzled: bool,
    ) -> Result<(MemVal<'c>, bool)> {
        if let Some(buf) = self.hoisted_stage(src) {
            return Ok((buf, true));
        }
        let buf = if swizzled {
            self.alloc_tile_swizzled(block, self.f16_t, shape)?
        } else {
            self.alloc_tile_shaped(block, self.f16_t, shape)?
        };
        self.stage_to_f16(block, src, &buf, false)?;
        Ok((buf, false))
    }

    /// Stages one iteration's a and b slices into shared as f16, without a
    /// barrier (the caller owns synchronization).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn wmma_stage(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        kt: Value<'c, 'c>,
        iv_div: i64,
        a_buf: &MemVal<'c>,
        b_buf: &MemVal<'c>,
        async_copy: bool,
    ) -> Result<()> {
        let a_src = self.emit_kt_slice(block, p, p.a_slice, kt, iv_div)?;
        let b_src = self.emit_kt_slice(block, p, p.b_slice, kt, iv_div)?;

        self.stage_to_f16(block, &a_src, a_buf, async_copy)?;
        self.stage_to_f16(block, &b_src, b_buf, async_copy)
    }

    /// The tensor-core pipelined half: prefetch into dst, tensor-core MAC from
    /// cur (see [`Self::pipelined_half`]). reg_stage selects the sm_75
    /// register-staged variant (see [`Self::wmma_half_reg_staged`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn wmma_half(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        iv_div: i64,
        prefetch_iv: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        st: Value<'c, 'c>,
        cur: (&MemVal<'c>, &MemVal<'c>),
        dst: (&MemVal<'c>, &MemVal<'c>),
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
        async_copy: bool,
        reg_stage: bool,
        mma_sync: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        if reg_stage {
            return self.wmma_half_reg_staged(
                block,
                p,
                iv_div,
                prefetch_iv,
                hi,
                st,
                cur,
                dst,
                dims,
                m0,
                n0,
                accs,
                mma_sync,
            );
        }

        self.pipelined_half(
            block,
            prefetch_iv,
            hi,
            async_copy,
            |cg, then| cg.wmma_stage(then, p, prefetch_iv, iv_div, dst.0, dst.1, async_copy),
            |cg| cg.tc_mac(block, cur.0, cur.1, dims, m0, n0, accs, false, mma_sync),
        )
    }

    pub(super) fn reg_stage_divides(&self, rows: i64, cols: i64) -> bool {
        let lane_elems = self.cta_threads * HALF_VEC;
        cols % HALF_VEC == 0 && rows * cols >= lane_elems && (rows * cols) % lane_elems == 0
    }

    /// The sm_75 register-staged pipeline half. The synchronous path goes
    /// load -> shared -> barrier -> compute, so the shared store stalls on the
    /// global load before any math runs. This one issues the next tile's global
    /// loads into registers, runs the current tile's WMMA compute while they're
    /// in flight, then stores the registers to shared under the same prefetch
    /// guard. The loads are unconditional (so they overlap the compute) with a
    /// clamped, in-bounds index, and the guarded store skips the redundant final
    /// read. Assumes the launch block is cta_threads (the WMMA launch ABI).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn wmma_half_reg_staged(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        iv_div: i64,
        prefetch_iv: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        st: Value<'c, 'c>,
        cur: (&MemVal<'c>, &MemVal<'c>),
        dst: (&MemVal<'c>, &MemVal<'c>),
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
        mma_sync: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        // Clamp the prefetch to the last valid tile start: the load runs even on
        // the final iteration (no "next" tile), where it harmlessly re-reads the
        // last tile since the guarded store below skips it.
        let last = self.subi(block, hi, st)?;
        let safe = self.minsi(block, prefetch_iv, last)?;

        // Issue the next tile's global loads into registers (held across compute).
        let a_loaded = self.wmma_load_operand(block, p, p.a_slice, safe, iv_div, dst.0)?;
        let b_loaded = self.wmma_load_operand(block, p, p.b_slice, safe, iv_div, dst.1)?;

        // Compute the current tile while the loads are in flight.
        let next = self.tc_mac(block, cur.0, cur.1, dims, m0, n0, accs, false, mma_sync)?;

        // Commit the prefetched tile to shared (only when it was real), then a
        // closing barrier publishes it and retires the reads cur will reuse.
        self.guarded_prefetch(block, prefetch_iv, hi, |cg, then| {
            cg.wmma_store_operand(then, &a_loaded, dst.0)?;
            cg.wmma_store_operand(then, &b_loaded, dst.1)
        })?;

        self.barrier(block)?;

        Ok(next)
    }

    /// Unrolled per-thread vector<[`HALF_VEC`]xf16> loads of one staged operand
    /// from global into registers (no shared store). Returns each loaded vector
    /// with its [row, col] destination index, for [`Self::wmma_store_operand`]
    /// to write after the compute. The per-thread count is static (the caller
    /// gates on [`Self::reg_stage_divides`]) and the CTA stride is cta_threads.
    fn wmma_load_operand(
        &mut self,
        block: &Block<'c>,
        p: &MatmulFusion<'_>,
        slice: &Expr,
        kt: Value<'c, 'c>,
        iv_div: i64,
        dst: &MemVal<'c>,
    ) -> Result<Vec<(Value<'c, 'c>, [Value<'c, 'c>; 2])>> {
        let src = self.emit_kt_slice(block, p, slice, kt, iv_div)?;
        let (rows, cols) = (dst.shape[0], dst.shape[1]);
        let inner = cols / HALF_VEC; // HALF_VEC-wide vectors per row
        let per = (rows * inner) / self.cta_threads;
        let vec_t = Type::vector(&[HALF_VEC as u64], self.f16_t);
        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
        let inner_v = self.const_index(block, inner)?;
        let vec_w = self.const_index(block, HALF_VEC)?;
        let mut loaded = Vec::with_capacity(per as usize);

        for i in 0..per {
            // linear vector index tid + i*cta_threads row-major
            let off = self.const_index(block, i * self.cta_threads)?;
            let lin = self.addi(block, tid, off)?;
            let r = self.divui(block, lin, inner_v)?;
            let c4 = self.remui(block, lin, inner_v)?;
            let c = self.muli(block, c4, vec_w)?;
            // 8-byte f16 vector load from the global slice
            let v = self.vec_load_al(block, src.mem, &[r, c], vec_t, 8)?;
            loaded.push((v, [r, c]));
        }

        Ok(loaded)
    }

    fn wmma_store_operand(
        &mut self,
        block: &Block<'c>,
        loaded: &[(Value<'c, 'c>, [Value<'c, 'c>; 2])],
        dst: &MemVal<'c>,
    ) -> Result<()> {
        for (v, idx) in loaded {
            let didx = self.swizzled_index(block, dst, idx)?;
            self.vec_store_al(block, *v, dst.mem, &didx, 8)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn tc_mac(
        &mut self,
        block: &Block<'c>,
        a_buf: &MemVal<'c>,
        b_buf: &MemVal<'c>,
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
        transpose_b: bool,
        mma_sync: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        if mma_sync {
            self.mma_sync_mac(block, a_buf, b_buf, dims, m0, n0, accs, transpose_b)
        } else {
            self.wmma_mac(block, a_buf, b_buf, dims, m0, n0, accs, transpose_b)
        }
    }

    /// One k-tile of mma.sync computes from the staged f16 buffers. The legacy
    /// WMMA MAC works in m16n16k16 fragments, but the hardware mma.sync shape is
    /// m16n8kK (K = 8 on Turing, 16 on Ampere+), so each logical 16x16 fragment
    /// splits into two n-sub-tiles of 8, and the warp owns fm * fnn * 2
    /// vector<2x2x{acc}> accumulators (one per (fi, fj, n8)). Each kK-deep step
    /// ldmatrix-loads the warp's A and B fragments and folds them in with
    /// nvgpu.mma.sync. With transpose_b the B operand is staged [n, k] (the
    /// dot_t layout) and read without the transpose flip.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mma_sync_mac(
        &mut self,
        block: &Block<'c>,
        a_buf: &MemVal<'c>,
        b_buf: &MemVal<'c>,
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
        transpose_b: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let (kk, fm, fnn) = dims;
        let mma_k = self
            .base
            .gpu_config
            .mma_sync_k()
            .ok_or_else(|| anyhow!("mma.sync needs a chip with a native shape (sm_75+)"))?
            as i64;
        // Per-lane fragment register counts: A spans m16 x kK as 8x8 tiles, B
        // spans kK x n8, both two f16 per tile. C/D is the m16n8 accumulator.
        let a_tiles = 2 * (mma_k / 8);
        let b_tiles = mma_k / 8;
        let a_frag_t = Type::vector(&[a_tiles as u64, 2], self.f16_t);
        let b_frag_t = Type::vector(&[b_tiles as u64, 2], self.f16_t);
        let acc_t = accs
            .first()
            .map(|v| v.r#type())
            .ok_or_else(|| anyhow!("mma_sync_mac needs at least one accumulator"))?;
        // The lane index spreads the ldmatrix reads across the warp.
        let tid = self.gpu_index(block, "gpu.thread_id", "x")?;
        let w = self.const_index(block, 32)?;
        let lane = self.remui(block, tid, w)?;
        let mut accs = accs.to_vec();

        for ks in 0..kk / mma_k {
            let kbase = self.const_index(block, ks * mma_k)?;

            // A fragments: one m16 x kK block per fi.
            let mut a_frags = Vec::with_capacity(fm as usize);
            for fi in 0..fm {
                let c = self.const_index(block, fi * 16)?;
                let mi = self.addi(block, m0, c)?;
                a_frags.push(self.ldmatrix_frag(
                    block,
                    a_buf,
                    &[mi, kbase],
                    a_tiles,
                    false,
                    a_frag_t,
                    lane,
                )?);
            }

            // B fragments: one kK x n8 block per (fj, n8). NN reads b[k, n]
            // transposed (k-major buffer, col-major operand); NT reads the
            // [n, k] buffer straight.
            let mut b_frags = Vec::with_capacity((fnn * 2) as usize);
            for fj in 0..fnn {
                for nn in 0..2 {
                    let c = self.const_index(block, fj * 16 + nn * 8)?;
                    let nj = self.addi(block, n0, c)?;
                    let (idx, trans) = if transpose_b {
                        ([nj, kbase], false)
                    } else {
                        ([kbase, nj], true)
                    };
                    b_frags.push(
                        self.ldmatrix_frag(block, b_buf, &idx, b_tiles, trans, b_frag_t, lane)?,
                    );
                }
            }

            for fi in 0..fm {
                for fj in 0..fnn {
                    for nn in 0..2 {
                        let i = ((fi * fnn + fj) * 2 + nn) as usize;
                        let shape = self.parse_attr(&format!("[16, 8, {mma_k}]"))?;
                        accs[i] = self.push(
                            block,
                            OperationBuilder::new("nvgpu.mma.sync", self.loc)
                                .add_operands(&[
                                    a_frags[fi as usize],
                                    b_frags[(fj * 2 + nn) as usize],
                                    accs[i],
                                ])
                                .add_attributes(&[(self.id("mmaShape"), shape)])
                                .add_results(&[acc_t])
                                .build()?,
                        )?;
                    }
                }
            }
        }
        Ok(accs)
    }

    /// nvgpu.ldmatrix buf[indices] {numTiles, transpose}: a warp-collective load
    /// of num_tiles 8x8 f16 fragments into per-lane registers, in the mma.sync
    /// operand layout. indices is the warp-tile origin (row, col). The lowering
    /// just does a plain strided-element-pointer on them and doesn't spread the
    /// work across the warp's lanes, so we fold the per-lane address offset in
    /// here (each ldmatrix lane holds the start address of one 8-element row).
    ///
    /// The offset depends on how the operand's tiles are laid out. Non-transpose
    /// A is 16 rows by num_tiles/2 8-wide k-tiles, so the row is lane % 16 and
    /// the k-column is (lane / 16) % (num_tiles/2) * 8. Transpose B is num_tiles
    /// 8x8 tiles stacked along k, so the row is lane % (8 * num_tiles) and the
    /// column is just the warp-tile column. On sm_75 these collapse to A
    /// lane % 16 / col 0 and B lane % 8. Since the offset rides the index, any
    /// swizzle (a later phase) has to ride the staging store, not this load.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ldmatrix_frag(
        &self,
        block: &Block<'c>,
        buf: &MemVal<'c>,
        indices: &[Value<'c, 'c>],
        num_tiles: i64,
        transpose: bool,
        frag_t: Type<'c>,
        lane: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        // Fold the per-lane (row, col) offset into the warp-tile origin.
        // Lanes past the consumed address count (8 per tile) still compute an
        // address the hardware dereferences even though the value is unused,
        // so the row modulus clamps them into the block: an operand in the
        // last rows of the final shared buffer would otherwise read past the
        // CTA's shared window and fault (observed on sm_75 with ldmatrix.x1
        // once buffer pooling shrank the window).
        let (row_off, col_off) = if transpose {
            let rows = self.const_index(block, 8 * num_tiles)?;
            (self.remui(block, lane, rows)?, None)
        } else {
            let rows = self.const_index(block, (8 * num_tiles).min(16))?;
            let row = self.remui(block, lane, rows)?;
            let r16 = self.const_index(block, 16)?;
            let col_tiles = self.const_index(block, (num_tiles / 2).max(1))?;
            let g = self.divui(block, lane, r16)?;
            let gt = self.remui(block, g, col_tiles)?;
            let eight = self.const_index(block, 8)?;
            (row, Some(self.muli(block, gt, eight)?))
        };

        let row = self.addi(block, indices[0], row_off)?;
        let col = match col_off {
            Some(c) => self.addi(block, indices[1], c)?,
            None => indices[1],
        };

        // Read back the swizzled column the staging store wrote (identity when
        // the buffer is unswizzled).
        let col = self.swizzle_col(block, buf, row, col)?;

        let operands = vec![buf.mem, row, col];
        let trans = self.parse_attr(if transpose { "true" } else { "false" })?;

        self.push(
            block,
            OperationBuilder::new("nvgpu.ldmatrix", self.loc)
                .add_operands(&operands)
                .add_attributes(&[
                    (self.id("transpose"), trans),
                    (
                        self.id("numTiles"),
                        IntegerAttribute::new(self.i32_t, num_tiles).into(),
                    ),
                ])
                .add_results(&[frag_t])
                .build()?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn wmma_mac(
        &mut self,
        block: &Block<'c>,
        a_buf: &MemVal<'c>,
        b_buf: &MemVal<'c>,
        dims: (i64, i64, i64),
        m0: Value<'c, 'c>,
        n0: Value<'c, 'c>,
        accs: &[Value<'c, 'c>],
        transpose_b: bool,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let (kk, fm, fnn) = dims;
        let a_frag_t = self.parse_type(WMMA_A)?;
        let b_frag_t = self.parse_type(WMMA_B)?;
        let c_frag_t = accs
            .first()
            .map(|v| v.r#type())
            .ok_or_else(|| anyhow!("wmma_mac needs at least one accumulator fragment"))?;
        let mut accs = accs.to_vec();

        for ks in 0..kk / 16 {
            let k_v = self.const_index(block, ks * 16)?;
            let mut a_frags = Vec::with_capacity(fm as usize);

            for fi in 0..fm {
                let c = self.const_index(block, fi * 16)?;
                let mi = self.addi(block, m0, c)?;
                a_frags.push(self.wmma_load(block, a_buf, &[mi, k_v], a_frag_t, false)?);
            }

            let mut b_frags = Vec::with_capacity(fnn as usize);
            for fj in 0..fnn {
                let c = self.const_index(block, fj * 16)?;
                let nj = self.addi(block, n0, c)?;
                // NN reads b[k, n]; NT reads the same logical fragment out of
                // the [n, k] buffer at [n, k], transposed in the load.
                let idx = if transpose_b { [nj, k_v] } else { [k_v, nj] };
                b_frags.push(self.wmma_load(block, b_buf, &idx, b_frag_t, transpose_b)?);
            }

            for fi in 0..fm {
                for fj in 0..fnn {
                    let i = (fi * fnn + fj) as usize;

                    accs[i] = self.push(
                        block,
                        OperationBuilder::new("gpu.subgroup_mma_compute", self.loc)
                            .add_operands(&[a_frags[fi as usize], b_frags[fj as usize], accs[i]])
                            .add_results(&[c_frag_t])
                            .build()?,
                    )?;
                }
            }
        }

        Ok(accs)
    }

    /// gpu.subgroup_mma_load_matrix buf[indices], a warp-collective fragment
    /// load. The lead dimension is the buffer's (static) row stride. transpose
    /// flips the fragment to column-major (a .col wmma load), reading a logically
    /// transposed tile out of a row-major buffer without a separate staging pass.
    pub(super) fn wmma_load(
        &self,
        block: &Block<'c>,
        buf: &MemVal<'c>,
        indices: &[Value<'c, 'c>],
        frag_t: Type<'c>,
        transpose: bool,
    ) -> Result<Value<'c, 'c>> {
        let mut operands = vec![buf.mem];

        operands.extend_from_slice(indices);

        // The lead dimension is the buffer's physical row stride, which exceeds
        // the logical inner extent when the tile is bank-conflict padded.
        let lead = buf.row_stride.unwrap_or(buf.shape[1]);
        let mut attrs = vec![(
            self.id("leadDimension"),
            IntegerAttribute::new(self.index_t, lead).into(),
        )];

        if transpose {
            attrs.push((self.id("transpose"), Attribute::unit(self.ctx)));
        }

        self.push(
            block,
            OperationBuilder::new("gpu.subgroup_mma_load_matrix", self.loc)
                .add_operands(&operands)
                .add_attributes(&attrs)
                .add_results(&[frag_t])
                .build()?,
        )
    }

    /// Stages an operand into an f16 shared buffer for WMMA. An f32 source is
    /// rounded down ([`Self::tile_copy_f16`]), an f16 source is copied as-is,
    /// and anything else is rejected. async_copy only applies to the straight
    /// f16 copy; the f32 round-down can't be a raw cp.async byte transfer. Never
    /// emits a barrier; the caller owns synchronization.
    pub(super) fn stage_to_f16(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        dst: &MemVal<'c>,
        async_copy: bool,
    ) -> Result<()> {
        if dst.elem != self.f16_t {
            bail!("WMMA staging destination must be f16");
        }

        if src.elem == self.f32_t {
            self.tile_copy_f16(block, src, dst)
        } else if src.elem == self.f16_t {
            self.tile_copy(block, src, dst, false, async_copy)
        } else {
            bail!("WMMA operands must be f16 or f32, got {}", src.elem)
        }
    }

    /// dst[...] = f16(src[...]): stages an f32 slice into an f16 shared buffer,
    /// rounding each element once (arith.truncf). Vectorized as 4xf32 loads /
    /// 4xf16 (8-byte) stores when the source rows are provably aligned. Never
    /// emits a barrier; the caller owns sync.
    pub(super) fn tile_copy_f16(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
        dst: &MemVal<'c>,
    ) -> Result<()> {
        if src.elem != self.f32_t || dst.elem != self.f16_t {
            bail!("f16 staging needs an f32 source and an f16 destination");
        }

        let last = *dst.shape.last().expect("tile values are not rank-0");
        let width = if src.aligned && last != DYN && last % 4 == 0 {
            4
        } else {
            1
        };

        let v32_t = Type::vector(&[4], self.f32_t);
        let v16_t = Type::vector(&[4], self.f16_t);

        self.distribute(block, dst, width, false, |cg, blk, idx| {
            // Read the (unswizzled) source, round, store to the swizzled column.
            let didx = cg.swizzled_index(blk, dst, idx)?;

            if width > 1 {
                let v = cg.vec_load(blk, src.mem, idx, v32_t)?;
                let h = cg.truncf(blk, v, v16_t)?;
                cg.vec_store_al(blk, h, dst.mem, &didx, 8)?;
            } else {
                let e = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
                let h = cg.truncf(blk, e, cg.f16_t)?;
                blk.append_operation(memref::store(h, dst.mem, &didx, cg.loc));
            }

            Ok(())
        })
    }

    pub(super) fn truncf(
        &self,
        block: &Block<'c>,
        value: Value<'c, 'c>,
        t: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        self.push(
            block,
            OperationBuilder::new("arith.truncf", self.loc)
                .add_operands(&[value])
                .add_results(&[t])
                .build()?,
        )
    }
}
