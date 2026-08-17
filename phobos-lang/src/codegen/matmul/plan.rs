// Recognising a matmul in a statement list, and the shape and
// epilogue decisions that follow from it.

use super::*;

impl<'c> Codegen<'c> {
    pub(in crate::codegen) fn matmul_candidate<'a>(
        &self,
        stmts: &'a [Stmt],
    ) -> Option<MatmulFusion<'a>> {
        // match: var acc, for kt, [optional let c_old,] store.
        let [
            Stmt::Var {
                name: acc,
                ty: Some(AstType::Tile(acc_scalar, dims)),
                value: Some(init),
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

        // The epilogue target: a slice of an f32 tensor, not mentioning acc,
        // whose shape is statically known and equal to acc's.
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

        // The loop body: exactly two tensor slices, then one accumulating dot of
        // exactly those two names.
        let staged = |s: &'a Stmt| -> Option<(&'a str, &'a Expr)> {
            let (name, None, Some(value)) = s.as_decl()? else {
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
            (self.is_f16_or_f32(t.elem) && !value.uses_name(acc)).then_some((name, value))
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

        if self.has_wmma() && self.wmma_plan(m, n, ak).is_some() {
            // The tensor-core path: f16 or f32 operands, accumulator and C.
        } else {
            // The vector path is f32 throughout; an f16 accumulator needs the
            // tensor cores, so anything else falls back to the generic matmul.
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

    /// The fusion's static GEMM extents: the accumulator shape [m, n] and the
    /// k extent of the staged a-slice.
    pub(super) fn fusion_dims(&mut self, p: &MatmulFusion<'_>) -> Result<(Vec<i64>, i64)> {
        let shape = self.tile_shape(p.dims)?;
        let kk = self
            .slice_static_shape(p.a_slice)
            .ok_or_else(|| anyhow!("matmul fusion without a static a-slice"))?[1];
        Ok((shape, kk))
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
    pub(in crate::codegen) fn matmul_kloop(
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

        if !self.pipeline_assert {
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

    /// Precomputes the alpha and beta operands of a GEMM epilogue
    /// (alpha*acc [+ beta*prev_load]), broadcast to vec_t when the store is
    /// aligned. Returns (None, None) for a plain accumulation.
    pub(in crate::codegen) fn epilogue_scaling(
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
    pub(in crate::codegen) fn apply_scaling(
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

    pub(in crate::codegen) fn epilogue_view(
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

    /// The warp's block origin (m0, n0) on a wm x wn warp grid of bm x bn
    /// element blocks, with surplus warps clamped onto the last block.
    /// Returns (tid, w, wt, m0, n0) so callers can derive lane coordinates.
    pub(in crate::codegen) fn warp_block_origin(
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
        let tid = self.thread_id(block)?;
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
    pub(super) fn frag_origin(
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
}
