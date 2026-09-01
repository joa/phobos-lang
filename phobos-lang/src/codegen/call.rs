use super::*;

impl<'c> Codegen<'c> {
    pub(super) fn qmma_operands(
        &mut self,
        block: &Block<'c>,
        args: &[Expr],
    ) -> Result<[MemVal<'c>; 4]> {
        let [a, asc, w, wsc] = args else {
            bail!("qmma_t expects (a, a_scales, w, w_scales)");
        };
        let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
            Rv::Tile(t) => Ok(t),
            Rv::Scalar(_) => bail!("qmma_t expects tile operands"),
        };
        let a = operand(self, a)?;
        let asc = operand(self, asc)?;
        let w = operand(self, w)?;
        let wsc = operand(self, wsc)?;
        Ok([a, asc, w, wsc])
    }

    /// `(a, a_scales, qb, d, grid)` for an `iq1s_qmma_t` call.
    pub(super) fn iq1s_qmma_operands(
        &mut self,
        block: &Block<'c>,
        args: &[Expr],
    ) -> Result<[MemVal<'c>; 5]> {
        let [a, asc, qb, d, grid] = args else {
            bail!("iq1s_qmma_t expects (a, a_scales, qb, d, grid)");
        };
        let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
            Rv::Tile(t) => Ok(t),
            Rv::Scalar(_) => bail!("iq1s_qmma_t expects tile operands"),
        };
        let a = operand(self, a)?;
        let asc = operand(self, asc)?;
        let qb = operand(self, qb)?;
        let d = operand(self, d)?;
        let grid = operand(self, grid)?;
        Ok([a, asc, qb, d, grid])
    }

    /// `(qb, d, tables..)` for a `<fmt>_qdecode_t` call.
    pub(super) fn qdecode_operands(
        &mut self,
        block: &Block<'c>,
        fmt: QFormat,
        args: &[Expr],
    ) -> Result<Vec<MemVal<'c>>> {
        let want = 2 + fmt.tables();
        if args.len() != want {
            bail!("{} expects {want} tile operands", fmt.intrinsic());
        }
        let mut tiles = Vec::with_capacity(want);
        for arg in args {
            match self.emit_expr(block, arg)? {
                Rv::Tile(t) => tiles.push(t),
                Rv::Scalar(_) => bail!("{} expects tile operands", fmt.intrinsic()),
            }
        }
        Ok(tiles)
    }

    pub(super) fn emit_call(
        &mut self,
        block: &Block<'c>,
        callee: &str,
        args: &[Expr],
    ) -> Result<Rv<'c>> {
        match callee {
            "program_id" => {
                let [Expr::Int(d @ 0..=2)] = args else {
                    bail!("program_id expects one literal dimension argument 0..=2");
                };
                let dim = ["x", "y", "z"][*d as usize];
                Ok(Rv::Scalar(self.block_id(block, dim)?))
            }
            // Grid-wide synchronization, spanning several stages of a pass; see codegen/sync.rs.
            "atomic_add" => self.emit_atomic_add(block, args),
            "grid_barrier" => self.emit_grid_barrier(block, args),
            "warp_partial" => self.emit_warp_partial(block, args),
            // dot outside an assignment materializes into a fresh buffer;
            // acc += dot(a, b) is handled in store_tile.
            "dot" => {
                let (a, b) = self.dot_operands(block, args)?;
                let shape = [a.shape[0], b.shape[1]];
                if shape.contains(&DYN) {
                    bail!("dot result shape must be static; assign to a tile-typed var instead");
                }
                let acc_elem = self.accumulator_elem(a.elem, b.elem)?;
                let out = self.alloc_tile_shaped(block, acc_elem, &shape)?;
                self.check_matmul_shapes(&a, &b, &out)?;
                if !self.wmma_dot(block, &a, &b, &out, false, false)? {
                    let zero = self.zero_scalar(block, out.elem)?;
                    self.tile_fill(block, zero, &out)?;
                    self.tile_matmul(block, &a, &b, &out)?;
                }
                self.release(&a);
                self.release(&b);
                Ok(Rv::Tile(out))
            }
            // dot_t(a, b) = a * b^T, materialized into a fresh buffer; no
            // transposed analogue of acc += dot(a, b) exists.
            "dot_t" => {
                let (a, b) = self.dot_operands(block, args)?;
                if a.shape.len() != 2 || b.shape.len() != 2 {
                    bail!("dot_t expects rank-2 tiles");
                }
                self.check_shapes(&[a.shape[1]], &[b.shape[1]], "dot_t contraction dim")?;
                let shape = [a.shape[0], b.shape[0]];
                if shape.contains(&DYN) {
                    bail!("dot_t result shape must be static");
                }
                let acc_elem = self.accumulator_elem(a.elem, b.elem)?;
                let out = self.alloc_tile_shaped(block, acc_elem, &shape)?;
                if !self.wmma_dot(block, &a, &b, &out, true, false)? {
                    self.tile_matmul_t(block, &a, &b, &out)?;
                }
                self.release(&a);
                self.release(&b);
                Ok(Rv::Tile(out))
            }
            // qdot_t(a, a_scales, w, w_scales): the Q8_0 contraction with its
            // block scales folded in
            "qdot_t" => {
                let [a, asc, w, wsc] = args else {
                    bail!("qdot_t expects (a, a_scales, w, w_scales)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("qdot_t expects tile operands"),
                };
                let (a, asc) = (operand(self, a)?, operand(self, asc)?);
                let (w, wsc) = (operand(self, w)?, operand(self, wsc)?);
                let out = self.tile_qdot_t(block, &a, &asc, &w, &wsc)?;
                for t in [&a, &asc, &w, &wsc] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // qmma_t(a, a_scales, w, w_scales): the Q8_0 contraction batched
            // over rows, on the integer tensor cores. The weight scales are
            // [block, out] here where qdot_t wants [out, block].
            "qmma_t" => {
                let [a, asc, w, wsc] = self.qmma_operands(block, args)?;
                let out = self.tile_qmma_t(block, &a, &asc, &w, &wsc)?;
                for t in [&a, &asc, &w, &wsc] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // iq1s_qdot_i8_t(aq, asc, qb, d, grid): the same contraction
            // against an int8-quantized activation, in dp4a.
            // iq2xxs_qdot_i8_t(aq, asc, qb, d, grid, signs): as above, for
            // the format whose decode is a signed table lookup.
            "iq2s_qdot_i8_t" => {
                let [aq, asc, qb, d, grid, signs] = args else {
                    bail!("iq2s_qdot_i8_t expects (aq, asc, qb, d, grid, signs)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq2s_qdot_i8_t expects tile operands"),
                };
                let (aq, asc) = (operand(self, aq)?, operand(self, asc)?);
                let (qb, d) = (operand(self, qb)?, operand(self, d)?);
                let (grid, signs) = (operand(self, grid)?, operand(self, signs)?);
                let out = self.tile_iq2s_qdot_i8_t(block, &aq, &asc, &qb, &d, &grid, &signs)?;
                for t in [&aq, &asc, &qb, &d, &grid, &signs] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            "iq2xs_qdot_i8_t" => {
                let [aq, asc, qb, d, grid, signs] = args else {
                    bail!("iq2xs_qdot_i8_t expects (aq, asc, qb, d, grid, signs)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq2xs_qdot_i8_t expects tile operands"),
                };
                let (aq, asc) = (operand(self, aq)?, operand(self, asc)?);
                let (qb, d) = (operand(self, qb)?, operand(self, d)?);
                let (grid, signs) = (operand(self, grid)?, operand(self, signs)?);
                let out = self.tile_iq2xs_qdot_i8_t(block, &aq, &asc, &qb, &d, &grid, &signs)?;
                for t in [&aq, &asc, &qb, &d, &grid, &signs] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            "iq1m_qdot_i8_t" => {
                let [aq, asc, qb, d, grid] = args else {
                    bail!("iq1m_qdot_i8_t expects (aq, asc, qb, d, grid)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq1m_qdot_i8_t expects tile operands"),
                };
                let (aq, asc) = (operand(self, aq)?, operand(self, asc)?);
                let (qb, d) = (operand(self, qb)?, operand(self, d)?);
                let grid = operand(self, grid)?;
                let out = self.tile_iq1m_qdot_i8_t(block, &aq, &asc, &qb, &d, &grid)?;
                for t in [&aq, &asc, &qb, &d, &grid] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            "iq3xxs_qdot_i8_t" => {
                let [aq, asc, qb, d, grid, signs] = args else {
                    bail!("iq3xxs_qdot_i8_t expects (aq, asc, qb, d, grid, signs)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq3xxs_qdot_i8_t expects tile operands"),
                };
                let (aq, asc) = (operand(self, aq)?, operand(self, asc)?);
                let (qb, d) = (operand(self, qb)?, operand(self, d)?);
                let (grid, signs) = (operand(self, grid)?, operand(self, signs)?);
                let out = self.tile_iq3xxs_qdot_i8_t(block, &aq, &asc, &qb, &d, &grid, &signs)?;
                for t in [&aq, &asc, &qb, &d, &grid, &signs] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            "iq3s_qdot_i8_t" => {
                let [aq, asc, qb, d, grid, signs] = args else {
                    bail!("iq3s_qdot_i8_t expects (aq, asc, qb, d, grid, signs)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq3s_qdot_i8_t expects tile operands"),
                };
                let (aq, asc) = (operand(self, aq)?, operand(self, asc)?);
                let (qb, d) = (operand(self, qb)?, operand(self, d)?);
                let (grid, signs) = (operand(self, grid)?, operand(self, signs)?);
                let out = self.tile_iq3s_qdot_i8_t(block, &aq, &asc, &qb, &d, &grid, &signs)?;
                for t in [&aq, &asc, &qb, &d, &grid, &signs] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            "iq2xxs_qdot_i8_t" => {
                let [aq, asc, qb, d, grid, signs] = args else {
                    bail!("iq2xxs_qdot_i8_t expects (aq, asc, qb, d, grid, signs)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq2xxs_qdot_i8_t expects tile operands"),
                };
                let (aq, asc) = (operand(self, aq)?, operand(self, asc)?);
                let (qb, d) = (operand(self, qb)?, operand(self, d)?);
                let (grid, signs) = (operand(self, grid)?, operand(self, signs)?);
                let out = self.tile_iq2xxs_qdot_i8_t(block, &aq, &asc, &qb, &d, &grid, &signs)?;
                for t in [&aq, &asc, &qb, &d, &grid, &signs] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            "iq1s_qdot_i8_t" => {
                let [aq, asc, qb, d, grid] = args else {
                    bail!("iq1s_qdot_i8_t expects (aq, asc, qb, d, grid)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq1s_qdot_i8_t expects tile operands"),
                };
                let (aq, asc) = (operand(self, aq)?, operand(self, asc)?);
                let (qb, d) = (operand(self, qb)?, operand(self, d)?);
                let grid = operand(self, grid)?;
                let out = self.tile_iq1s_qdot_i8_t(block, &aq, &asc, &qb, &d, &grid)?;
                for t in [&aq, &asc, &qb, &d, &grid] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // iq1s_qdot_t(a, qb, d, grid): IQ1_S's matvec contraction with
            // the grid-table decode folded in, `qdot_t`'s warp-owns-an-output
            // shape without `qdot_t`'s hardware instruction.
            "iq1s_qdot_t" => {
                let [a, qb, d, grid] = args else {
                    bail!("iq1s_qdot_t expects (a, qb, d, grid)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq1s_qdot_t expects tile operands"),
                };
                let (a, qb) = (operand(self, a)?, operand(self, qb)?);
                let (d, grid) = (operand(self, d)?, operand(self, grid)?);
                let out = self.tile_iq1s_qdot_t(block, &a, &qb, &d, &grid)?;
                for t in [&a, &qb, &d, &grid] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // iq1s_qmma_t(a, a_scales, qb, d, grid): the same decode as
            // `iq1s_qdot_t`, contracted on the integer tensor cores over a
            // batch of rows instead. What a prompt pass runs, so that the
            // expanded weight is never written at all.
            "iq1s_qmma_t" => {
                let [a, asc, qb, d, grid] = self.iq1s_qmma_operands(block, args)?;
                let out = self.tile_iq1s_qmma_t(block, &a, &asc, &qb, &d, &grid)?;
                for t in [&a, &asc, &qb, &d, &grid] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // iq1s_qmma_staged_t(..): the same projection with the decoded
            // weight staged through shared memory. Only expressible where one
            // warp owns one patch; see `iq1s_qmma_staged_into`.
            "iq1s_qmma_staged_t" => {
                let [a, asc, qb, d, grid] = self.iq1s_qmma_operands(block, args)?;
                let out = self.alloc_tile_shaped(block, self.f32_t, &[a.shape[0], qb.shape[0]])?;
                self.iq1s_qmma_staged_into(block, &a, &asc, &qb, &d, &grid, &out)?;
                for t in [&a, &asc, &qb, &d, &grid] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // iq1m_qdot_t(a, qb, d, grid): IQ1_M's matvec contraction with
            // the decode folded in, IQ1_S's shared grid but its own
            // per-group scale pairs.
            "iq1m_qdot_t" => {
                let [a, qb, d, grid] = args else {
                    bail!("iq1m_qdot_t expects (a, qb, d, grid)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq1m_qdot_t expects tile operands"),
                };
                let (a, qb) = (operand(self, a)?, operand(self, qb)?);
                let (d, grid) = (operand(self, d)?, operand(self, grid)?);
                let out = self.tile_iq1m_qdot_t(block, &a, &qb, &d, &grid)?;
                for t in [&a, &qb, &d, &grid] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // iq2xxs_qdot_t(a, qb, d, grid, signs): IQ2_XXS's matvec
            // contraction with its magnitude-grid and sign-table lookups
            // folded in, `iq1s_qdot_t`'s shape with a second gather.
            "iq2xxs_qdot_t" => {
                let [a, qb, d, grid, signs] = args else {
                    bail!("iq2xxs_qdot_t expects (a, qb, d, grid, signs)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq2xxs_qdot_t expects tile operands"),
                };
                let (a, qb) = (operand(self, a)?, operand(self, qb)?);
                let (d, grid) = (operand(self, d)?, operand(self, grid)?);
                let signs = operand(self, signs)?;
                let out = self.tile_iq2xxs_qdot_t(block, &a, &qb, &d, &grid, &signs)?;
                for t in [&a, &qb, &d, &grid, &signs] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // iq2s_qdot_t(a, qb, d, grid, signs): IQ2_S's matvec contraction
            // with its magnitude-grid and sign-table lookups folded in.
            "iq2s_qdot_t" => {
                let [a, qb, d, grid, signs] = args else {
                    bail!("iq2s_qdot_t expects (a, qb, d, grid, signs)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq2s_qdot_t expects tile operands"),
                };
                let (a, qb) = (operand(self, a)?, operand(self, qb)?);
                let (d, grid) = (operand(self, d)?, operand(self, grid)?);
                let signs = operand(self, signs)?;
                let out = self.tile_iq2s_qdot_t(block, &a, &qb, &d, &grid, &signs)?;
                for t in [&a, &qb, &d, &grid, &signs] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // iq2xs_qdot_t(a, qb, d, grid, signs): IQ2_XS's matvec
            // contraction with its magnitude-grid and sign-table lookups
            // folded in.
            "iq2xs_qdot_t" => {
                let [a, qb, d, grid, signs] = args else {
                    bail!("iq2xs_qdot_t expects (a, qb, d, grid, signs)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq2xs_qdot_t expects tile operands"),
                };
                let (a, qb) = (operand(self, a)?, operand(self, qb)?);
                let (d, grid) = (operand(self, d)?, operand(self, grid)?);
                let signs = operand(self, signs)?;
                let out = self.tile_iq2xs_qdot_t(block, &a, &qb, &d, &grid, &signs)?;
                for t in [&a, &qb, &d, &grid, &signs] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // iq3xxs_qdot_t(a, qb, d, grid, signs): IQ3_XXS's matvec
            // contraction with its two four-wide grid lookups and sign-table
            // lookup folded in.
            "iq3xxs_qdot_t" => {
                let [a, qb, d, grid, signs] = args else {
                    bail!("iq3xxs_qdot_t expects (a, qb, d, grid, signs)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq3xxs_qdot_t expects tile operands"),
                };
                let (a, qb) = (operand(self, a)?, operand(self, qb)?);
                let (d, grid) = (operand(self, d)?, operand(self, grid)?);
                let signs = operand(self, signs)?;
                let out = self.tile_iq3xxs_qdot_t(block, &a, &qb, &d, &grid, &signs)?;
                for t in [&a, &qb, &d, &grid, &signs] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // iq3s_qdot_t(a, qb, d, grid, signs): IQ3_S's matvec contraction
            // with its two four-wide grid lookups and sign-table lookup
            // folded in.
            "iq3s_qdot_t" => {
                let [a, qb, d, grid, signs] = args else {
                    bail!("iq3s_qdot_t expects (a, qb, d, grid, signs)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq3s_qdot_t expects tile operands"),
                };
                let (a, qb) = (operand(self, a)?, operand(self, qb)?);
                let (d, grid) = (operand(self, d)?, operand(self, grid)?);
                let signs = operand(self, signs)?;
                let out = self.tile_iq3s_qdot_t(block, &a, &qb, &d, &grid, &signs)?;
                for t in [&a, &qb, &d, &grid, &signs] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // iq4xs_qdot_t(a, qb, d, codebook): IQ4_XS's matvec contraction
            // with its fixed 16-entry codebook lookup folded in.
            "iq4xs_qdot_t" => {
                let [a, qb, d, codebook] = args else {
                    bail!("iq4xs_qdot_t expects (a, qb, d, codebook)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("iq4xs_qdot_t expects tile operands"),
                };
                let (a, qb) = (operand(self, a)?, operand(self, qb)?);
                let (d, codebook) = (operand(self, d)?, operand(self, codebook)?);
                let out = self.tile_iq4xs_qdot_t(block, &a, &qb, &d, &codebook)?;
                for t in [&a, &qb, &d, &codebook] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // q2k_qdot_t(a, qb, d, dmin): Q2_K's matvec contraction with its
            // static-offset decode folded in.
            "q2k_qdot_t" => {
                let [a, qb, d, dmin] = args else {
                    bail!("q2k_qdot_t expects (a, qb, d, dmin)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("q2k_qdot_t expects tile operands"),
                };
                let (a, qb) = (operand(self, a)?, operand(self, qb)?);
                let (d, dmin) = (operand(self, d)?, operand(self, dmin)?);
                let out = self.tile_q2k_qdot_t(block, &a, &qb, &d, &dmin)?;
                for t in [&a, &qb, &d, &dmin] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // q3k_qdot_t(a, qb, d): Q3_K's matvec contraction with its
            // static-offset decode folded in.
            "q3k_qdot_t" => {
                let [a, qb, d] = args else {
                    bail!("q3k_qdot_t expects (a, qb, d)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("q3k_qdot_t expects tile operands"),
                };
                let (a, qb) = (operand(self, a)?, operand(self, qb)?);
                let d = operand(self, d)?;
                let out = self.tile_q3k_qdot_t(block, &a, &qb, &d)?;
                for t in [&a, &qb, &d] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // gather(TABLE, IDX): a per-element table lookup, out[i] =
            // TABLE[IDX[i]].
            "gather" => {
                let [table, idx] = args else {
                    bail!("gather expects (table, idx)");
                };
                let operand = |cg: &mut Self, e: &Expr| match cg.emit_expr(block, e)? {
                    Rv::Tile(t) => Ok(t),
                    Rv::Scalar(_) => bail!("gather expects tile operands"),
                };
                let (table, idx) = (operand(self, table)?, operand(self, idx)?);
                let out = self.tile_gather(block, &table, &idx)?;
                for t in [&table, &idx] {
                    self.release(t);
                }
                Ok(Rv::Tile(out))
            }
            // flat(t): a tile viewed as one row. A view, not a copy, so the
            // result is read-only and the buffer stays the declaration's.
            "flat" => {
                let [arg] = args else {
                    bail!("flat expects one tile argument");
                };
                let Rv::Tile(t) = self.emit_expr(block, arg)? else {
                    bail!("flat expects a tile argument");
                };
                Ok(Rv::Tile(self.tile_flat(block, &t)?))
            }
            // element-wise unary math over a tile.
            "exp" | "log" | "round" | "sqrt" | "tanh" => {
                let [arg] = args else {
                    bail!("{callee} expects one tile argument");
                };
                let Rv::Tile(t) = self.emit_expr(block, arg)? else {
                    bail!("{callee} expects a tile argument");
                };
                let out = match callee {
                    "exp" => self.tile_exp(block, &t)?,
                    "log" => self.tile_log(block, &t)?,
                    "round" => self.tile_round(block, &t)?,
                    "sqrt" => self.tile_sqrt(block, &t)?,
                    _ => self.tile_tanh(block, &t)?,
                };
                Ok(Rv::Tile(out))
            }
            // tmax(a, b): element-wise maximum (broadcasting).
            "tmax" => {
                let [x, y] = args else {
                    bail!("tmax expects two tile arguments");
                };
                let (Rv::Tile(a), Rv::Tile(b)) =
                    (self.emit_expr(block, x)?, self.emit_expr(block, y)?)
                else {
                    bail!("tmax expects tile arguments");
                };
                if a.elem != b.elem {
                    bail!("tmax operands must share an element type");
                }
                let shape = broadcast_shape(&a.shape, &b.shape)
                    .ok_or_else(|| anyhow!("tmax operands are not broadcast-compatible"))?;
                let out = self.alloc_tile_shaped(block, a.elem, &shape)?;
                self.tile_max_bc(block, &a, &b, &out)?;
                self.release(&a);
                self.release(&b);
                Ok(Rv::Tile(out))
            }
            "argsel" => self.emit_argsel(block, args), // a tmax-shaped fold's index side
            // rowmax(t) / rowsum(t): reduce a rank-2 tile over its last
            // column dim, producing a [rows, 1] column vector.
            "rowmax" | "rowsum" => {
                let how = if callee == "rowmax" {
                    Reduce::Max
                } else {
                    Reduce::Sum
                };
                let t = self.reduce_arg(block, args, callee)?;
                let out = self.tile_rowreduce(block, &t, how)?;
                self.release(&t);
                Ok(Rv::Tile(out))
            }
            // cumsum(t): inclusive prefix sum down the rows (the sequence axis).
            "cumsum" => {
                let t = self.reduce_arg(block, args, "cumsum")?;
                let out = self.tile_cumsum(block, &t)?;
                self.release(&t);
                Ok(Rv::Tile(out))
            }
            // tril(t): causal lower-triangular mask. May rewrite t in place,
            // so t is not released here.
            "tril" => {
                let t = self.reduce_arg(block, args, "tril")?;
                let out = self.tile_tril(block, &t)?;
                Ok(Rv::Tile(out))
            }
            // transpose(t): rank-2 tile transpose.
            "transpose" => {
                let t = self.reduce_arg(block, args, "transpose")?;
                let out = self.tile_transpose(block, &t)?;
                self.release(&t);
                Ok(Rv::Tile(out))
            }
            // f32(x), bf16(x), i8(x), ...: convert a tile or a scalar to the
            // named element type.
            other if Scalar::from_name(other).is_some_and(|s| s != Scalar::Bool) => {
                let want = self.scalar_type(Scalar::from_name(other).expect("matched above"));
                let [arg] = args else {
                    bail!("{other} expects one argument to convert");
                };
                match self.emit_expr(block, arg)? {
                    Rv::Tile(t) => {
                        let out = self.tile_cast(block, &t, want)?;
                        self.release(&t);
                        Ok(Rv::Tile(out))
                    }
                    Rv::Scalar(v) => Ok(Rv::Scalar(self.numeric_cast(block, v, want)?)),
                }
            }
            // <fmt>_qdecode_t(qb, d, tables..): a raw format expanded into
            // the [K, N] scratch a batched matmul reads. It writes the
            // destination itself (see `Codegen::store_tile`), so it is only
            // ever the whole right-hand side of a store.
            other if QFormat::from_intrinsic(other).is_some() => bail!(
                "{other} writes a tensor slice: use it as the whole right-hand side of an assignment"
            ),
            other => bail!("unknown function '{other}'"),
        }
    }

    pub(super) fn dot_operands(
        &mut self,
        block: &Block<'c>,
        args: &[Expr],
    ) -> Result<(MemVal<'c>, MemVal<'c>)> {
        let [lhs, rhs] = args else {
            bail!("dot expects two tile arguments");
        };
        let Rv::Tile(a) = self.emit_expr(block, lhs)? else {
            bail!("dot expects tile operands");
        };
        let Rv::Tile(b) = self.emit_expr(block, rhs)? else {
            bail!("dot expects tile operands");
        };
        Ok((a, b))
    }

    pub(super) fn reduce_arg(
        &mut self,
        block: &Block<'c>,
        args: &[Expr],
        name: &str,
    ) -> Result<MemVal<'c>> {
        let [arg] = args else {
            bail!("{name} expects one tile argument");
        };
        let Rv::Tile(t) = self.emit_expr(block, arg)? else {
            bail!("{name} expects a tile argument");
        };
        Ok(t)
    }
}
