use anyhow::{Result, anyhow, bail, ensure};

use super::{Binding, Build, DYN, Rv, broadcast_shape};
use crate::ast::{Expr, Scalar as AstScalar};
use crate::ir::{ElemStep, Intrinsic, Map, OpKind, RawFmt, Reduce, Scalar, Type, ValueId};

impl Build {
    /// Each arm evaluates its operands in source order and produces one op
    /// whose result type is the buffer the emitter allocates.
    pub(crate) fn emit_call(&mut self, callee: &str, args: &[Expr]) -> Result<Rv> {
        match callee {
            "program_id" => {
                let [Expr::Int(d @ 0..=2)] = args else {
                    bail!("program_id expects one literal dimension argument 0..=2");
                };
                Ok(Rv::Scalar(self.value(OpKind::ProgramId(*d as u8), &[], Type::INDEX)))
            }
            "atomic_add" => self.emit_atomic_add(args),
            "rms_norm_q_t" => self.emit_rms_norm_q(args),
            "grid_barrier" => self.emit_grid_barrier(args),
            "warp_partial" => self.emit_warp_partial(args),
            "dot" => {
                let (a, b) = self.dot_operands(args)?;
                let (ash, bsh) = (self.shape(a), self.shape(b));
                let shape = [ash[0], bsh[1]];
                if shape.contains(&DYN) {
                    bail!("dot result shape must be static; assign to a tile-typed var instead");
                }
                let acc_elem = super::expr::accumulator_elem(self.elem(a), self.elem(b))?;
                self.check_matmul_shapes(&ash, &bsh, &shape)?;
                let ty = self.shared_ty(acc_elem, &shape);
                Ok(Rv::Tile(self.value(OpKind::Dot { transpose: false }, &[a, b], ty)))
            }
            "dot_t" => {
                let (a, b) = self.dot_operands(args)?;
                let (ash, bsh) = (self.shape(a), self.shape(b));
                if ash.len() != 2 || bsh.len() != 2 {
                    bail!("dot_t expects rank-2 tiles");
                }
                self.check_shapes(&[ash[1]], &[bsh[1]], "dot_t contraction dim")?;
                let shape = [ash[0], bsh[0]];
                if shape.contains(&DYN) {
                    bail!("dot_t result shape must be static");
                }
                let acc_elem = super::expr::accumulator_elem(self.elem(a), self.elem(b))?;
                let ty = self.shared_ty(acc_elem, &shape);
                Ok(Rv::Tile(self.value(OpKind::Dot { transpose: true }, &[a, b], ty)))
            }
            "qdot_t" => {
                let [a, asc, w, wsc] = args else {
                    bail!("qdot_t expects (a, a_scales, w, w_scales)");
                };
                let a = self.tile_operand(a, "qdot_t")?;
                let asc = self.tile_operand(asc, "qdot_t")?;
                let w = self.tile_operand(w, "qdot_t")?;
                let wsc = self.tile_operand(wsc, "qdot_t")?;
                let shape = [self.shape(a)[0], self.shape(w)[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::QdotT), &[a, asc, w, wsc], ty)))
            }
            "qmma_t" => {
                let [a, asc, w, wsc] = self.qmma_operands(args)?;
                let shape = [self.shape(a)[0], self.shape(w)[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::QmmaT), &[a, asc, w, wsc], ty)))
            }
            _ if matches!(Intrinsic::from_name(callee), Some(Intrinsic::RawQdotI8(_))) => {
                let Some(Intrinsic::RawQdotI8(fmt)) = Intrinsic::from_name(callee) else {
                    unreachable!("matched above");
                };
                let want = 4 + qg_tables(fmt);
                if args.len() != want {
                    bail!("{callee} expects {want} operands: (aq, asc, qb, d, tables..)");
                }
                let tiles = args
                    .iter()
                    .map(|e| self.tile_operand(e, callee))
                    .collect::<Result<Vec<_>>>()?;
                let shape = [1, self.shape(tiles[2])[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::RawQdotI8(fmt)), &tiles, ty)))
            }
            "iq1s_qdot_t" => self.raw_qdot(callee, RawFmt::Iq1s, args, "(a, qb, d, grid)", 4),
            "iq1s_qmma_t" => {
                let [a, asc, qb, d, grid] = self.iq1s_qmma_operands(args)?;
                let shape = [self.shape(a)[0], self.shape(qb)[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                Ok(Rv::Tile(self.value(
                    OpKind::Intrinsic(Intrinsic::RawQmma(RawFmt::Iq1s)),
                    &[a, asc, qb, d, grid],
                    ty,
                )))
            }
            "iq1s_qmma_staged_t" => {
                let [a, asc, qb, d, grid] = self.iq1s_qmma_operands(args)?;
                let shape = [self.shape(a)[0], self.shape(qb)[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                Ok(Rv::Tile(self.value(
                    OpKind::Intrinsic(Intrinsic::RawQmmaStaged(RawFmt::Iq1s)),
                    &[a, asc, qb, d, grid],
                    ty,
                )))
            }
            _ if super::store::qgemm_format(callee).is_some() => {
                let fmt = super::store::qgemm_format(callee).expect("matched above");
                let tiles = self.qgemm_operands(fmt, callee, args)?;
                let shape = [self.shape(tiles[0])[0], self.shape(tiles[2])[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::RawQgemm(fmt)), &tiles, ty)))
            }
            _ if super::store::staged_qmma_format(callee).is_some() => {
                let fmt = super::store::staged_qmma_format(callee).expect("matched above");
                let ops = self.iq2xxs_qmma_operands(callee, args)?;
                let shape = [self.shape(ops[0])[0], self.shape(ops[2])[0]];
                let ty = self.shared_ty(Scalar::F32, &shape);
                Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::RawQmmaStaged(fmt)), &ops, ty)))
            }
            "iq1m_qdot_t" => self.raw_qdot(callee, RawFmt::Iq1m, args, "(a, qb, d, grid)", 4),
            "iq2xxs_qdot_t" => self.raw_qdot(callee, RawFmt::Iq2xxs, args, "(a, qb, d, grid, signs)", 5),
            "iq2s_qdot_t" => self.raw_qdot(callee, RawFmt::Iq2s, args, "(a, qb, d, grid, signs)", 5),
            "iq2xs_qdot_t" => self.raw_qdot(callee, RawFmt::Iq2xs, args, "(a, qb, d, grid, signs)", 5),
            "iq3xxs_qdot_t" => self.raw_qdot(callee, RawFmt::Iq3xxs, args, "(a, qb, d, grid, signs)", 5),
            "iq3s_qdot_t" => self.raw_qdot(callee, RawFmt::Iq3s, args, "(a, qb, d, grid, signs)", 5),
            "iq4xs_qdot_t" => self.raw_qdot(callee, RawFmt::Iq4xs, args, "(a, qb, d, codebook)", 4),
            "q2k_qdot_t" => self.raw_qdot(callee, RawFmt::Q2k, args, "(a, qb, d, dmin)", 4),
            "q3k_qdot_t" => self.raw_qdot(callee, RawFmt::Q3k, args, "(a, qb, d)", 3),
            "gather" => {
                let [table, idx] = args else {
                    bail!("gather expects (table, idx)");
                };
                let table = self.tile_operand(table, "gather")?;
                let idx = self.tile_operand(idx, "gather")?;
                let shape = self.shape(idx);
                let ty = self.shared_ty(self.elem(table), &shape);
                Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::Gather), &[table, idx], ty)))
            }
            "flat" => {
                let [arg] = args else {
                    bail!("flat expects one tile argument");
                };
                let Rv::Tile(t) = self.emit_expr(arg)? else {
                    bail!("flat expects a tile argument");
                };
                Ok(Rv::Tile(self.tile_flat(t)?))
            }
            "exp" | "log" | "round" | "sqrt" | "tanh" => {
                let [arg] = args else {
                    bail!("{callee} expects one tile argument");
                };
                let Rv::Tile(t) = self.emit_expr(arg)? else {
                    bail!("{callee} expects a tile argument");
                };
                let step = ElemStep::from_callee(callee).expect("a math call");
                // An owned temp is rewritten in place; the emitter decides,
                // and the type is the source's either way.
                let shape = self.shape(t);
                if !self.owned(t) && shape.contains(&DYN) {
                    bail!("{callee} needs a static tile shape");
                }
                let ty = self.like(t);
                Ok(Rv::Tile(self.value(OpKind::Map(Map::Unary(step)), &[t], ty)))
            }
            "tmax" => {
                let [x, y] = args else {
                    bail!("tmax expects two tile arguments");
                };
                let (Rv::Tile(a), Rv::Tile(b)) = (self.emit_expr(x)?, self.emit_expr(y)?) else {
                    bail!("tmax expects tile arguments");
                };
                if self.elem(a) != self.elem(b) {
                    bail!("tmax operands must share an element type");
                }
                let shape = broadcast_shape(&self.shape(a), &self.shape(b))
                    .ok_or_else(|| anyhow!("tmax operands are not broadcast-compatible"))?;
                let ty = self.shared_ty(self.elem(a), &shape);
                Ok(Rv::Tile(self.value(OpKind::Map(Map::Max), &[a, b], ty)))
            }
            "argsel" => self.emit_argsel(args),
            "rowmax" | "rowsum" => {
                let how = if callee == "rowmax" { Reduce::RowMax } else { Reduce::RowSum };
                let t = self.reduce_arg(args, callee)?;
                let shape = self.shape(t);
                if shape.len() != 2 {
                    bail!("row reduction expects a rank-2 tile");
                }
                if shape.contains(&DYN) {
                    bail!("row reduction needs a static tile shape");
                }
                if !self.elem(t).is_float() {
                    bail!("row reduction needs a float element type");
                }
                let ty = self.shared_ty(self.elem(t), &[shape[0], 1]);
                Ok(Rv::Tile(self.value(OpKind::Reduce(how), &[t], ty)))
            }
            "cumsum" => {
                let t = self.reduce_arg(args, "cumsum")?;
                let shape = self.shape(t);
                if shape.len() != 2 {
                    bail!("cumsum expects a rank-2 tile");
                }
                if shape.contains(&DYN) {
                    bail!("cumsum needs a static tile shape");
                }
                if !self.elem(t).is_float() {
                    bail!("cumsum needs a float element type");
                }
                let ty = self.like(t);
                Ok(Rv::Tile(self.value(OpKind::Reduce(Reduce::CumSum), &[t], ty)))
            }
            "tril" => {
                let t = self.reduce_arg(args, "tril")?;
                let shape = self.shape(t);
                if shape.len() != 2 {
                    bail!("tril expects a rank-2 tile");
                }
                if shape.contains(&DYN) {
                    bail!("tril needs a static tile shape");
                }
                if !self.elem(t).is_float() {
                    bail!("tril needs a float element type");
                }
                let ty = self.like(t);
                Ok(Rv::Tile(self.value(OpKind::Reduce(Reduce::Tril), &[t], ty)))
            }
            "transpose" => {
                let t = self.reduce_arg(args, "transpose")?;
                let shape = self.shape(t);
                if shape.len() != 2 {
                    bail!("transpose expects a rank-2 tile");
                }
                if shape.contains(&DYN) {
                    bail!("transpose needs a static tile shape");
                }
                let ty = self.shared_ty(self.elem(t), &[shape[1], shape[0]]);
                Ok(Rv::Tile(self.value(OpKind::Reduce(Reduce::Transpose), &[t], ty)))
            }
            other if AstScalar::from_name(other).is_some_and(|s| s != AstScalar::Bool) => {
                let want = Scalar::from_ast(AstScalar::from_name(other).expect("matched above"));
                let [arg] = args else {
                    bail!("{other} expects one argument to convert");
                };
                match self.emit_expr(arg)? {
                    Rv::Tile(t) => Ok(Rv::Tile(self.tile_cast(t, want)?)),
                    Rv::Scalar(v) => Ok(Rv::Scalar(self.numeric_cast(v, want))),
                }
            }
            other if super::store::qdecode_format(other).is_some() => bail!(
                "{other} writes a tensor slice: use it as the whole right-hand side of an assignment"
            ),
            other => bail!("unknown function '{other}'"),
        }
    }

    /// A `<fmt>_qdot_t` call: `n` tile operands, a `[1, qb rows]` result.
    fn raw_qdot(&mut self, callee: &str, fmt: RawFmt, args: &[Expr], sig: &str, n: usize) -> Result<Rv> {
        if args.len() != n {
            bail!("{callee} expects {sig}");
        }
        let tiles = args
            .iter()
            .map(|e| self.tile_operand(e, callee))
            .collect::<Result<Vec<_>>>()?;
        let shape = [1, self.shape(tiles[1])[0]];
        let ty = self.shared_ty(Scalar::F32, &shape);
        Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::RawQdot(fmt)), &tiles, ty)))
    }

    /// An operand that has to be a tile: `"{what} expects tile operands"`.
    pub(crate) fn tile_operand(&mut self, e: &Expr, what: &str) -> Result<ValueId> {
        match self.emit_expr(e)? {
            Rv::Tile(t) => Ok(t),
            Rv::Scalar(_) => bail!("{what} expects tile operands"),
        }
    }

    pub(crate) fn dot_operands(&mut self, args: &[Expr]) -> Result<(ValueId, ValueId)> {
        let [lhs, rhs] = args else {
            bail!("dot expects two tile arguments");
        };
        let Rv::Tile(a) = self.emit_expr(lhs)? else {
            bail!("dot expects tile operands");
        };
        let Rv::Tile(b) = self.emit_expr(rhs)? else {
            bail!("dot expects tile operands");
        };
        Ok((a, b))
    }

    pub(crate) fn reduce_arg(&mut self, args: &[Expr], name: &str) -> Result<ValueId> {
        let [arg] = args else {
            bail!("{name} expects one tile argument");
        };
        let Rv::Tile(t) = self.emit_expr(arg)? else {
            bail!("{name} expects a tile argument");
        };
        Ok(t)
    }

    pub(crate) fn qmma_operands(&mut self, args: &[Expr]) -> Result<[ValueId; 4]> {
        let [a, asc, w, wsc] = args else {
            bail!("qmma_t expects (a, a_scales, w, w_scales)");
        };
        let a = self.tile_operand(a, "qmma_t")?;
        let asc = self.tile_operand(asc, "qmma_t")?;
        let w = self.tile_operand(w, "qmma_t")?;
        let wsc = self.tile_operand(wsc, "qmma_t")?;
        Ok([a, asc, w, wsc])
    }

    pub(crate) fn iq1s_qmma_operands(&mut self, args: &[Expr]) -> Result<[ValueId; 5]> {
        let [a, asc, qb, d, grid] = args else {
            bail!("iq1s_qmma_t expects (a, a_scales, qb, d, grid)");
        };
        let a = self.tile_operand(a, "iq1s_qmma_t")?;
        let asc = self.tile_operand(asc, "iq1s_qmma_t")?;
        let qb = self.tile_operand(qb, "iq1s_qmma_t")?;
        let d = self.tile_operand(d, "iq1s_qmma_t")?;
        let grid = self.tile_operand(grid, "iq1s_qmma_t")?;
        Ok([a, asc, qb, d, grid])
    }

    pub(crate) fn qgemm_operands(&mut self, fmt: RawFmt, callee: &str, args: &[Expr]) -> Result<Vec<ValueId>> {
        let want = 4 + qg_tables(fmt);
        if args.len() != want {
            bail!("{callee} expects {want} operands: (a, a_scales, qb, d, tables..)");
        }
        args.iter().map(|e| self.tile_operand(e, callee)).collect()
    }

    /// `(a, a_scales, qb, d, grid, signs)` for the staged IQ2/IQ3 projections.
    pub(crate) fn iq2xxs_qmma_operands(&mut self, callee: &str, args: &[Expr]) -> Result<[ValueId; 6]> {
        let _ = callee;
        let [a, asc, qb, d, grid, signs] = args else {
            bail!("iq2xxs_qmma_t expects (a, a_scales, qb, d, grid, signs)");
        };
        let a = self.tile_operand(a, "iq2xxs_qmma_t")?;
        let asc = self.tile_operand(asc, "iq2xxs_qmma_t")?;
        let qb = self.tile_operand(qb, "iq2xxs_qmma_t")?;
        let d = self.tile_operand(d, "iq2xxs_qmma_t")?;
        let grid = self.tile_operand(grid, "iq2xxs_qmma_t")?;
        let signs = self.tile_operand(signs, "iq2xxs_qmma_t")?;
        Ok([a, asc, qb, d, grid, signs])
    }

    /// `(qb, d, tables..)` for a `<fmt>_qdecode_t` call.
    pub(crate) fn qdecode_operands(&mut self, fmt: RawFmt, callee: &str, args: &[Expr]) -> Result<Vec<ValueId>> {
        let want = 2 + qdecode_tables(fmt);
        if args.len() != want {
            bail!("{callee} expects {want} tile operands");
        }
        args.iter().map(|e| self.tile_operand(e, callee)).collect()
    }

    fn emit_argsel(&mut self, args: &[Expr]) -> Result<Rv> {
        let [va, vb, ia, ib] = args else {
            bail!("argsel expects four tile arguments (value, value, index, index)");
        };
        let (Rv::Tile(va), Rv::Tile(vb), Rv::Tile(ia), Rv::Tile(ib)) = (
            self.emit_expr(va)?,
            self.emit_expr(vb)?,
            self.emit_expr(ia)?,
            self.emit_expr(ib)?,
        ) else {
            bail!("argsel expects tile arguments");
        };
        let vshape = broadcast_shape(&self.shape(va), &self.shape(vb))
            .ok_or_else(|| anyhow!("argsel value operands are not broadcast-compatible"))?;
        let ishape = broadcast_shape(&self.shape(ia), &self.shape(ib))
            .ok_or_else(|| anyhow!("argsel index operands are not broadcast-compatible"))?;
        ensure!(
            self.elem(va) == self.elem(vb) && self.elem(ia) == self.elem(ib) && vshape == ishape,
            "argsel: value and index operands must each share an element type, and both \
             pairs must broadcast to the same shape"
        );
        let ty = self.shared_ty(self.elem(ia), &ishape);
        Ok(Rv::Tile(self.value(OpKind::Intrinsic(Intrinsic::ArgSel), &[va, vb, ia, ib], ty)))
    }

    /// `flat(t)`: a tile viewed as one row.
    fn tile_flat(&mut self, src: ValueId) -> Result<ValueId> {
        let shape = self.shape(src);
        if shape.len() != 2 {
            bail!("flat expects a rank-2 tile");
        }
        if !self.is_buffer(src) {
            bail!("flat expects a declared tile, not a slice of one");
        }
        let t = self.tile(src).clone();
        if t.layout.row_stride.is_some() || t.layout.swizzle.is_some() {
            bail!("a padded or swizzled staging tile has no flat view");
        }
        let [rows, cols] = [shape[0], shape[1]];
        if rows == DYN || cols == DYN {
            bail!("flat expects a static tile shape");
        }
        let len = rows * cols;
        let ty = Type::Tile(crate::ir::TileType {
            elem: t.elem,
            shape: vec![crate::ir::Extent::Fixed(1), crate::ir::Extent::Fixed(len)],
            layout: crate::ir::Layout {
                row_stride: None,
                swizzle: None,
                align_div: t.layout.align_div,
            },
            space: t.space,
        });
        Ok(self.value(OpKind::Flat, &[src], ty))
    }

    fn emit_atomic_add(&mut self, args: &[Expr]) -> Result<Rv> {
        let [t, i, v] = args else {
            bail!("atomic_add expects (tensor, index, value)");
        };
        let mem = self.barrier_tensor(t, "atomic_add")?;
        let idx = self.emit_index(i, "atomic_add index")?;
        let val = self.emit_scalar(v)?;
        let val = match self.scalar_of(val) {
            Scalar::I32 => val,
            Scalar::Index | Scalar::I8 | Scalar::I64 => {
                self.value(OpKind::IndexCast(Scalar::I32), &[val], Type::Scalar(Scalar::I32))
            }
            t => bail!("atomic_add value must be an integer, got {}", t.mlir_name()),
        };
        let old = self.value(OpKind::AtomicAdd, &[mem, idx, val], Type::Scalar(Scalar::I32));
        Ok(Rv::Scalar(old))
    }

    /// The atomic-state operand: a named `i32` tensor parameter.
    fn barrier_tensor(&self, e: &Expr, what: &str) -> Result<ValueId> {
        let Expr::Var(name) = e else {
            bail!("{what} expects a named i32 tensor parameter");
        };
        let Some(Binding::Tensor(mem)) = self.lookup(name) else {
            bail!("{what} expects a tensor parameter, but '{name}' is not one");
        };
        if self.elem(mem) != Scalar::I32 {
            bail!(
                "{what} expects an i32 tensor, but '{name}' holds {}",
                self.elem(mem).mlir_name()
            );
        }
        Ok(mem)
    }

    fn emit_grid_barrier(&mut self, args: &[Expr]) -> Result<Rv> {
        let [bar] = args else {
            bail!("grid_barrier expects one argument, the barrier tensor");
        };
        let mem = self.barrier_tensor(bar, "grid_barrier")?;
        Ok(Rv::Scalar(self.value(OpKind::GridBarrier, &[mem], Type::INDEX)))
    }

    /// `rms_norm_q_t(x, gain, eps, [out,] q, scales)`. Yields the zero the
    /// source sees; the emitter checks the shapes.
    fn emit_rms_norm_q(&mut self, args: &[Expr]) -> Result<Rv> {
        let (x, g, eps, o, q, s) = match args {
            [x, g, eps, o, q, s] => (x, g, eps, Some(o), q, s),
            [x, g, eps, q, s] => (x, g, eps, None, q, s),
            _ => bail!("rms_norm_q_t expects (x, gain, eps, [out,] q, scales)"),
        };
        let tile = |cg: &mut Self, e: &Expr, what: &str| match cg.emit_expr(e)? {
            Rv::Tile(t) => Ok(t),
            Rv::Scalar(_) => bail!("rms_norm_q_t {what} must be a tile"),
        };
        let x = tile(self, x, "x")?;
        let g = tile(self, g, "gain")?;
        let eps = self.emit_scalar(eps)?;
        let o = o.map(|o| tile(self, o, "out")).transpose()?;
        let q = tile(self, q, "q")?;
        let s = tile(self, s, "scales")?;
        let mut operands = vec![x, g, eps];
        operands.extend(o);
        operands.push(q);
        operands.push(s);
        let inv = self.value(
            OpKind::Intrinsic(Intrinsic::RmsNormQ),
            &operands,
            Type::Scalar(Scalar::F32),
        );
        Ok(Rv::Scalar(inv))
    }

    /// `warp_partial(q, K, V, lo, hi, col, WM, WL, WACC, scale)`: the tiles
    /// and tensors by name, then the scalars in the order the emitter
    /// evaluated them. Operands: the six buffers, then `scale, lo, hi, col`.
    fn emit_warp_partial(&mut self, args: &[Expr]) -> Result<Rv> {
        let [q, k, v, lo, hi, col, wm, wl, wacc, scale] = args else {
            bail!("warp_partial expects (q, K, V, lo, hi, col, WM, WL, WACC, scale)");
        };
        let q = self.named_tile(q, "warp_partial q")?;
        let k = self.named_tensor(k, "warp_partial K")?;
        let v = self.named_tensor(v, "warp_partial V")?;
        let wm = self.named_tile(wm, "warp_partial WM")?;
        let wl = self.named_tile(wl, "warp_partial WL")?;
        let wacc = self.named_tile(wacc, "warp_partial WACC")?;
        let scale = self.emit_scalar(scale)?;
        let lo = self.emit_index(lo, "warp_partial lo")?;
        let hi = self.emit_index(hi, "warp_partial hi")?;
        let col = self.emit_index(col, "warp_partial col")?;
        let zero = self.value(
            OpKind::Intrinsic(Intrinsic::WarpPartial),
            &[q, k, v, wm, wl, wacc, scale, lo, hi, col],
            Type::INDEX,
        );
        Ok(Rv::Scalar(zero))
    }

    fn named_tile(&self, e: &Expr, what: &str) -> Result<ValueId> {
        let Expr::Var(name) = e else {
            bail!("{what} expects a named tile variable");
        };
        match self.lookup(name) {
            Some(Binding::Tile(t) | Binding::View(t)) => Ok(t),
            _ => bail!("{what}: '{name}' is not a tile"),
        }
    }

    fn named_tensor(&self, e: &Expr, what: &str) -> Result<ValueId> {
        let Expr::Var(name) = e else {
            bail!("{what} expects a named tensor parameter");
        };
        match self.lookup(name) {
            Some(Binding::Tensor(t)) => Ok(t),
            _ => bail!("{what}: '{name}' is not a tensor parameter"),
        }
    }
}

/// Table operands after `(a, a_scales, qb, d)` of the grouped raw formats.
pub(crate) fn qg_tables(fmt: RawFmt) -> usize {
    match fmt {
        RawFmt::Iq1s | RawFmt::Iq1m => 1,
        RawFmt::Iq2xxs | RawFmt::Iq2xs | RawFmt::Iq2s | RawFmt::Iq3xxs | RawFmt::Iq3s => 2,
        RawFmt::Iq4xs | RawFmt::Q2k | RawFmt::Q3k | RawFmt::Q4k | RawFmt::Q5k | RawFmt::Q6k => 0,
        RawFmt::Ptq1 => 0,
    }
}

/// Table operands after `(qb, d)` of a `<fmt>_qdecode_t`.
pub(crate) fn qdecode_tables(fmt: RawFmt) -> usize {
    match fmt {
        RawFmt::Iq1s | RawFmt::Iq1m => 1,
        _ => 2,
    }
}
