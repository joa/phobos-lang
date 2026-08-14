// Tile contractions and the shape math that plans them, plus
// cumsum, tril and transpose.

use super::*;

impl<'c> Codegen<'c> {
    /// out[i, j] = sum_k(a[i, k] * b[j, k]), the transposed matmul behind
    /// `dot_t` (contracts the last dim of both operands).
    ///
    /// The int8 operands take the tensor cores first and `dp4a` second. The
    /// generic fallback is a plain thread-per-output scalar reduction: the
    /// heavily-tuned [`Self::tile_matmul`] has no transposed-b variant, and the
    /// attention S = Q @ K.T tile is small relative to the kernel's other
    /// costs.
    pub(in crate::codegen) fn tile_matmul_t(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        self.check_matmul_elems(a, b, out, "dot_t")?;
        let kd = a.shape[1];
        if kd == DYN {
            bail!("dot_t needs a static contraction dim");
        }

        if self.tile_matmul_t_imma(block, a, b, out, kd)?
            || self.tile_matmul_t_dp4a(block, a, b, out, kd)?
        {
            return Ok(());
        }

        let elem = out.elem;
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let (i, j) = (idx[0], idx[1]);
            let slot_t = MemRefType::new(elem, &[], None, None);
            let slot = cg.push(blk, memref::alloca(cg.ctx, slot_t, &[], &[], None, cg.loc))?;
            let zero = cg.zero_scalar(blk, elem)?;
            blk.append_operation(memref::store(zero, slot, &[], cg.loc));

            let lo = cg.const_index(blk, 0)?;
            let hi = cg.const_index(blk, kd)?;
            let st = cg.const_index(blk, 1)?;
            let kb = Block::new(&[(cg.index_t, cg.loc)]);
            let k = detach(kb.argument(0)?.into());
            // operands widen to the accumulator type (f16 inputs, f32 acc).
            let va = cg.load_as(&kb, a.mem, &[i, k], elem)?;
            let vb = cg.load_as(&kb, b.mem, &[j, k], elem)?;
            let cur = cg.push(&kb, memref::load(slot, &[], cg.loc))?;
            let nv = cg.elem_mac(&kb, elem, va, vb, cur)?;
            kb.append_operation(memref::store(nv, slot, &[], cg.loc));
            kb.append_operation(scf::r#yield(&[], cg.loc));
            let region = Region::new();
            region.append_block(kb);
            blk.append_operation(scf::r#for(lo, hi, st, region, cg.loc));

            let fin = cg.push(blk, memref::load(slot, &[], cg.loc))?;
            blk.append_operation(memref::store(fin, out.mem, idx, cg.loc));
            Ok(())
        })
    }

    /// out[i, j] = sum_{r <= i} src[r, j]: an inclusive prefix sum down the
    /// rows (the sequence axis) of a rank-2 tile, the running gate cumulant
    /// gated linear attention needs. One thread owns each column and sweeps
    /// its rows in order, carrying the partial in a register (an scf.for
    /// iter_arg); the sequential row dependence keeps this off the warp
    /// path.
    pub(in crate::codegen) fn tile_cumsum(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        if src.shape.len() != 2 {
            bail!("cumsum expects a rank-2 tile");
        }

        let (rows, cols) = (src.shape[0], src.shape[1]);
        if rows == DYN || cols == DYN {
            bail!("cumsum needs a static tile shape");
        }

        let elem = src.elem;
        if !self.is_float(elem) {
            bail!("cumsum needs a float element type");
        }

        let out = self.alloc_tile_shaped(block, elem, &[rows, cols])?;

        let total = self.const_index(block, cols)?;
        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;

        let body = Block::new(&[(self.index_t, self.loc)]);
        let j = detach(body.argument(0)?.into());

        let init = self.zero_scalar(&body, elem)?;
        let lo = self.const_index(&body, 0)?;
        let hi = self.const_index(&body, rows)?;
        let st = self.const_index(&body, 1)?;
        let rb = Block::new(&[(self.index_t, self.loc), (elem, self.loc)]);
        let i = detach(rb.argument(0)?.into());
        let acc = detach(rb.argument(1)?.into());
        let v = self.push(&rb, memref::load(src.mem, &[i, j], self.loc))?;
        let nacc = self.push(&rb, arith::addf(acc, v, self.loc))?;

        rb.append_operation(memref::store(nacc, out.mem, &[i, j], self.loc));
        rb.append_operation(scf::r#yield(&[nacc], self.loc));

        let rr = Region::new();
        rr.append_block(rb);

        body.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&[lo, hi, st, init])
                .add_results(&[elem])
                .add_regions([rr])
                .build()?,
        );

        body.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body);

        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));
        self.barrier(block)?;

        Ok(out)
    }

    /// out[i, j] = src[i, j] when j <= i, else 0: the causal (lower-
    /// triangular) mask for intra-chunk attention. Rewrites src in place
    /// when it owns an unswizzled buffer (each thread reads and writes one
    /// element), otherwise writes a fresh tile.
    pub(in crate::codegen) fn tile_tril(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        if src.shape.len() != 2 {
            bail!("tril expects a rank-2 tile");
        }

        if src.shape.contains(&DYN) {
            bail!("tril needs a static tile shape");
        }

        if !self.is_float(src.elem) {
            bail!("tril needs a float element type");
        }

        let out = if src.owned && src.swizzle.is_none() {
            src.clone()
        } else {
            self.alloc_tile_shaped(block, src.elem, &src.shape)?
        };

        let zero = self.zero_scalar(block, src.elem)?;

        self.distribute(block, &out, 1, true, |cg, blk, idx| {
            let (i, j) = (idx[0], idx[1]);
            let keep = cg.push(
                blk,
                arith::cmpi(cg.ctx, arith::CmpiPredicate::Sle, j, i, cg.loc),
            )?;

            let v = cg.push(blk, memref::load(src.mem, idx, cg.loc))?;
            let r = cg.push(
                blk,
                OperationBuilder::new("arith.select", cg.loc)
                    .add_operands(&[keep, v, zero])
                    .add_results(&[src.elem])
                    .build()?,
            )?;

            blk.append_operation(memref::store(r, out.mem, idx, cg.loc));

            Ok(())
        })?;

        Ok(out)
    }

    /// out[i, j] = src[j, i]: the rank-2 tile transpose. Each output element
    /// is owned by one thread that reads the mirrored source element. Needed
    /// to contract over the sequence axis (the K.T @ V state update) since
    /// dot / dot_t only contract the last axes.
    pub(in crate::codegen) fn tile_transpose(
        &mut self,
        block: &Block<'c>,
        src: &MemVal<'c>,
    ) -> Result<MemVal<'c>> {
        if src.shape.len() != 2 {
            bail!("transpose expects a rank-2 tile");
        }

        let (rows, cols) = (src.shape[0], src.shape[1]);
        if rows == DYN || cols == DYN {
            bail!("transpose needs a static tile shape");
        }

        let out = self.alloc_tile_shaped(block, src.elem, &[cols, rows])?;

        self.distribute(block, &out, 1, true, |cg, blk, idx| {
            let (i, j) = (idx[0], idx[1]);
            let v = cg.push(blk, memref::load(src.mem, &[j, i], cg.loc))?;
            blk.append_operation(memref::store(v, out.mem, idx, cg.loc));
            Ok(())
        })?;

        Ok(out)
    }

    /// out[m, n] += sum_k(a[m, k] * b[k, n]): accumulates into out.
    ///
    /// Register-blocked when out has a static shape: the CTA's threads stride over
    /// TMxTN sub-tiles of the output ([`Self::sub_tile`]), carrying the
    /// accumulator through the k-loop as one vector<TMxTN> iter_arg, fed by
    /// vector.contract over k-chunks. That loads TM + TN operand elements per
    /// k-step for TM*TN MACs, instead of two loads per MAC in the element-wise
    /// scheme.
    ///
    /// Warp-tiled when a factorization of the 32 lanes divides the sub-tile grid
    /// ([`Self::lane_grid`]): warps stride over WMxWN warp tiles (WM = lm*TM,
    /// WN = ln*TN) with lanes laid out lmxln row-major inside, so the warp's
    /// per-k-step shared reads collapse to WM + WN distinct elements. Lanes in a
    /// row broadcast the same a fragment, lanes in a column the same b fragment,
    /// and the b row segments the lanes read are contiguous (conflict-free). The
    /// flat fallback scatters the warp across a thin full-width strip and reads up
    /// to twice as much.
    pub(in crate::codegen) fn tile_matmul(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        self.check_matmul_elems(a, b, out, "dot")?;
        let k_extent = if a.shape[1] == DYN {
            let pos = self.const_index(block, 1)?;
            self.push(block, memref::dim(a.mem, pos, self.loc))?
        } else {
            self.const_index(block, a.shape[1])?
        };
        let (m, n) = (out.shape[0], out.shape[1]);
        if m == DYN || n == DYN {
            return self.tile_matmul_dynamic(block, a, b, out, k_extent);
        }
        let elem = out.elem;
        let (tm, tn) = self.sub_tile(m, n);
        let (tiles_m, tiles_n) = (m / tm, n / tn);
        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;

        // Warp decomposition, hoisted out of the warp-tile loop: which warp
        // this thread belongs to, how many warps the CTA has (the launch ABI
        // requires a multiple of 32 threads), and the lane's element offset
        // within its warp tile. Unsigned div/rem (non-negative operands) by
        // constants strength-reduce to shift/mask.
        let warp = match Self::lane_grid(tiles_m, tiles_n, tm, tn) {
            Some((lm, ln)) => {
                let w = self.const_index(block, 32)?;
                let warp_id = self.divui(block, tid, w)?;
                let lane = self.remui(block, tid, w)?;
                let nwarps = self.divui(block, bdim, w)?;
                let ln_v = self.const_index(block, ln)?;
                let lane_m = self.divui(block, lane, ln_v)?;
                let lane_n = self.remui(block, lane, ln_v)?;
                let tm_v = self.const_index(block, tm)?;
                let tn_v = self.const_index(block, tn)?;
                let off_m = self.muli(block, lane_m, tm_v)?;
                let off_n = self.muli(block, lane_n, tn_v)?;
                Some((lm, ln, warp_id, nwarps, off_m, off_n))
            }
            None => None,
        };
        let (lo, total, step) = match warp {
            Some((lm, ln, warp_id, nwarps, ..)) => {
                let total = self.const_index(block, (tiles_m / lm) * (tiles_n / ln))?;
                (warp_id, total, nwarps)
            }
            None => (tid, self.const_index(block, tiles_m * tiles_n)?, bdim),
        };

        let body = Block::new(&[(self.index_t, self.loc)]);
        let st = detach(body.argument(0)?.into());

        let (m0, n0) = if let Some((lm, ln, _, _, off_m, off_n)) = warp {
            // Warp-tile origin: wt -> (wt / wtiles_n * WM, wt % wtiles_n * WN),
            // plus the lane's offset within the warp tile.
            let wtiles_n_v = self.const_index(&body, tiles_n / ln)?;
            let q = self.divui(&body, st, wtiles_n_v)?;
            let r = self.remui(&body, st, wtiles_n_v)?;
            let wm_v = self.const_index(&body, lm * tm)?;
            let wn_v = self.const_index(&body, ln * tn)?;
            let wm0 = self.muli(&body, q, wm_v)?;
            let wn0 = self.muli(&body, r, wn_v)?;
            (self.addi(&body, wm0, off_m)?, self.addi(&body, wn0, off_n)?)
        } else {
            // Flat sub-tile origin: st -> (st / tiles_n * TM, st % tiles_n * TN).
            // tiles_n is a constant, so the div/rem strength-reduce.
            let tiles_n_v = self.const_index(&body, tiles_n)?;
            let q = self.divui(&body, st, tiles_n_v)?;
            let r = self.remui(&body, st, tiles_n_v)?;
            let tm_v = self.const_index(&body, tm)?;
            let tn_v = self.const_index(&body, tn)?;
            (self.muli(&body, q, tm_v)?, self.muli(&body, r, tn_v)?)
        };
        let mut ms = Vec::with_capacity(tm as usize);
        for i in 0..tm {
            let c = self.const_index(&body, i)?;
            ms.push(self.addi(&body, m0, c)?);
        }
        let mut ns = Vec::with_capacity(tn as usize);
        for j in 0..tn {
            let c = self.const_index(&body, j)?;
            ns.push(self.addi(&body, n0, c)?);
        }

        // Row segments become single vector accesses when the buffer's base
        // and row starts are provably 16-byte aligned (MemVal::aligned; n0 is
        // a multiple of tn, k chunks are multiples of 4). A vector access per
        // thread is also free of shared-bank conflicts, where stride-4 scalar
        // accesses conflict 4-way. Unvectorizable operands get assembled
        // element-wise instead; the MAC grid is a vector.contract either way
        // (see register_mac for the scheme; here a is m-major, so lhs chunks
        // are (m, k) and the contract's lhs transpose folds at constant
        // positions).
        let kk = a.shape[1];
        let chunk = if kk == DYN {
            1
        } else {
            [4, 2, 1]
                .into_iter()
                .find(|c| kk % c == 0)
                .expect("1 divides everything")
        };
        let vec_a = chunk == 4 && elem == self.f32_t && a.vectorizes(4) && a.elem == elem;
        let vec_b = tn % 4 == 0 && elem == self.f32_t && b.vectorizes(4) && b.elem == elem;
        let vec_out = tn % 4 == 0 && elem == self.f32_t && out.vectorizes(4);
        let acc_t = Type::vector(&[tm as u64, tn as u64], elem);
        let row_t = Type::vector(&[tn as u64], elem);
        let a_row_t = Type::vector(&[chunk as u64], elem);
        let lhs_t = Type::vector(&[tm as u64, chunk as u64], elem);
        let rhs_t = Type::vector(&[chunk as u64, tn as u64], elem);

        // The accumulator starts from the current output values (the +=),
        // assembled into one TMxTN vector (zero seeds are fully overwritten
        // and fold away in lowering).
        let zero = self.zero_scalar(&body, elem)?;
        let mut acc = self.vec_broadcast(&body, zero, acc_t)?;
        for (i, mi) in ms.iter().enumerate() {
            if vec_out {
                let v = self.vec_load(&body, out.mem, &[*mi, n0], row_t)?;
                acc = self.vec_insert(&body, v, acc, &[i as i64])?;
            } else {
                for (j, nj) in ns.iter().enumerate() {
                    let e = self.push(&body, memref::load(out.mem, &[*mi, *nj], self.loc))?;
                    acc = self.vec_insert(&body, e, acc, &[i as i64, j as i64])?;
                }
            }
        }

        let k_lo = self.const_index(&body, 0)?;
        let k_st = self.const_index(&body, chunk)?;
        let finals =
            self.carry_loop(&body, k_lo, k_extent, k_st, &[acc], |cg, lblk, k, accs| {
                let zero = cg.zero_scalar(lblk, elem)?;
                let mut lhs = cg.vec_broadcast(lblk, zero, lhs_t)?;
                for (i, mi) in ms.iter().enumerate() {
                    if vec_a {
                        let v = cg.vec_load(lblk, a.mem, &[*mi, k], a_row_t)?;
                        lhs = cg.vec_insert(lblk, v, lhs, &[i as i64])?;
                    } else {
                        for j in 0..chunk {
                            let c = cg.const_index(lblk, j)?;
                            let kj = cg.addi(lblk, k, c)?;
                            let e = cg.load_as(lblk, a.mem, &[*mi, kj], elem)?;
                            lhs = cg.vec_insert(lblk, e, lhs, &[i as i64, j])?;
                        }
                    }
                }
                let mut rhs = cg.vec_broadcast(lblk, zero, rhs_t)?;
                for j in 0..chunk {
                    let c = cg.const_index(lblk, j)?;
                    let kj = cg.addi(lblk, k, c)?;
                    if vec_b {
                        let v = cg.vec_load(lblk, b.mem, &[kj, n0], row_t)?;
                        rhs = cg.vec_insert(lblk, v, rhs, &[j])?;
                    } else {
                        for (l, nl) in ns.iter().enumerate() {
                            let e = cg.load_as(lblk, b.mem, &[kj, *nl], elem)?;
                            rhs = cg.vec_insert(lblk, e, rhs, &[j, l as i64])?;
                        }
                    }
                }
                Ok(vec![cg.vec_contract(lblk, lhs, rhs, accs[0], false)?])
            })?;

        for (i, mi) in ms.iter().enumerate() {
            let row = self.vec_extract(&body, finals[0], &[i as i64], row_t)?;
            if vec_out {
                self.vec_store(&body, row, out.mem, &[*mi, n0])?;
            } else {
                for (j, nj) in ns.iter().enumerate() {
                    let e = self.vec_extract(&body, row, &[j as i64], elem)?;
                    body.append_operation(memref::store(e, out.mem, &[*mi, *nj], self.loc));
                }
            }
        }
        body.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(lo, total, step, region, self.loc));
        self.barrier(block)?;
        Ok(())
    }

    /// Largest register sub-tile extent that divides d.
    pub(in crate::codegen) fn sub_extent(d: i64) -> i64 {
        [4, 2].into_iter().find(|c| d % c == 0).unwrap_or(1)
    }

    /// Register sub-tile extents (TM, TN) for an mxn matmul output: the
    /// largest of 8x8 (4 MACs per shared element loaded) and 8x4 (2.67)
    /// whose sub-tile grid keeps at least one sub-tile per CTA thread (the
    /// @launch thread count); bigger lane tiles below that would idle
    /// threads instead of adding work per lane, else the legacy <=4 extents
    /// (2.0). (Every shape that selects 8x8 needs an m*n >= 128x128 output,
    /// whose shared acc tile exceeds the 48KB CTA budget on the unfused path
    /// -- those configs only ever launch through the register-accumulator
    /// fusion.)
    pub(in crate::codegen) fn sub_tile(&self, m: i64, n: i64) -> (i64, i64) {
        for (tm, tn) in [(8, 8), (8, 4)] {
            if m % tm == 0 && n % tn == 0 && (m / tm) * (n / tn) >= self.cta_threads {
                return (tm, tn);
            }
        }
        (Self::sub_extent(m), Self::sub_extent(n))
    }

    /// Lane grid (lm x ln, lm*ln = 32) for warp tiling: each warp owns an
    /// (lm*TM)x(ln*TN) warp tile with lanes laid out row-major inside it.
    /// Picks the factorization minimizing the warp's distinct shared reads
    /// per k-step (WM + WN, for WM*WN MACs), i.e. the most square warp tile.
    /// Ties break toward wider WN so the warp's b reads span one contiguous
    /// row segment. None when no factorization divides the sub-tile grid, in
    /// which case the flat per-thread distribution is used instead.
    pub(in crate::codegen) fn lane_grid(
        tiles_m: i64,
        tiles_n: i64,
        tm: i64,
        tn: i64,
    ) -> Option<(i64, i64)> {
        [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]
            .into_iter()
            .filter(|&(lm, ln)| tiles_m % lm == 0 && tiles_n % ln == 0)
            .min_by_key(|&(lm, ln)| (lm * tm + ln * tn, lm))
    }

    /// Element-wise matmul fallback for dynamically-shaped outputs.
    pub(in crate::codegen) fn tile_matmul_dynamic(
        &mut self,
        block: &Block<'c>,
        a: &MemVal<'c>,
        b: &MemVal<'c>,
        out: &MemVal<'c>,
        k_extent: Value<'c, 'c>,
    ) -> Result<()> {
        self.distribute(block, out, 1, true, |cg, blk, idx| {
            let (m, n) = (idx[0], idx[1]);
            let init = cg.push(blk, memref::load(out.mem, &[m, n], cg.loc))?;
            let sums = cg.reduce_loop_multi(blk, k_extent, &[init], |cg, lblk, k, accs| {
                let x = cg.load_as(lblk, a.mem, &[m, k], out.elem)?;
                let y = cg.load_as(lblk, b.mem, &[k, n], out.elem)?;
                Ok(vec![cg.elem_mac(lblk, out.elem, x, y, accs[0])?])
            })?;
            blk.append_operation(memref::store(sums[0], out.mem, &[m, n], cg.loc));
            Ok(())
        })
    }

    /// scf.for k = 0..ub carrying inits as iter_args; returns the finals.
    pub(in crate::codegen) fn reduce_loop_multi(
        &mut self,
        block: &Block<'c>,
        ub: Value<'c, 'c>,
        inits: &[Value<'c, 'c>],
        body: impl FnOnce(
            &mut Self,
            &Block<'c>,
            Value<'c, 'c>,
            &[Value<'c, 'c>],
        ) -> Result<Vec<Value<'c, 'c>>>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let zero = self.const_index(block, 0)?;
        let one = self.const_index(block, 1)?;
        self.carry_loop(block, zero, ub, one, inits, body)
    }

    /// scf.for iv = lo to hi step st carrying inits as iter_args.
    pub(in crate::codegen) fn carry_loop(
        &mut self,
        block: &Block<'c>,
        lo: Value<'c, 'c>,
        hi: Value<'c, 'c>,
        st: Value<'c, 'c>,
        inits: &[Value<'c, 'c>],
        body: impl FnOnce(
            &mut Self,
            &Block<'c>,
            Value<'c, 'c>,
            &[Value<'c, 'c>],
        ) -> Result<Vec<Value<'c, 'c>>>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let types: Vec<Type<'c>> = inits.iter().map(|v| v.r#type()).collect();
        let mut block_args = vec![(self.index_t, self.loc)];
        block_args.extend(types.iter().map(|&t| (t, self.loc)));
        let body_block = Block::new(&block_args);
        let iv = detach(body_block.argument(0)?.into());
        let mut accs = Vec::with_capacity(inits.len());
        for i in 0..inits.len() {
            accs.push(detach(body_block.argument(i + 1)?.into()));
        }
        let next = body(self, &body_block, iv, &accs)?;
        body_block.append_operation(scf::r#yield(&next, self.loc));

        let region = Region::new();
        region.append_block(body_block);
        let mut operands = vec![lo, hi, st];
        operands.extend_from_slice(inits);
        let op = block.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&operands)
                .add_results(&types)
                .add_regions([region])
                .build()?,
        );
        let mut finals = Vec::with_capacity(inits.len());
        for i in 0..inits.len() {
            finals.push(detach(op.result(i)?.into()));
        }
        Ok(finals)
    }

    /// One multiply-accumulate (acc + a*b) on the element type. Floats use
    /// math.fma, a single rounding that lowers to PTX fma.rn, because a
    /// separate mul/add pair emits explicitly-rounded mul.rn/add.rn, which
    /// ptxas is not allowed to contract into an FMA: the matmul would spend
    /// two instructions per MAC and halve its FLOP ceiling.
    pub(in crate::codegen) fn elem_mac(
        &mut self,
        block: &Block<'c>,
        elem: Type<'c>,
        a: Value<'c, 'c>,
        b: Value<'c, 'c>,
        acc: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        if self.is_float(elem) {
            self.push(
                block,
                OperationBuilder::new("math.fma", self.loc)
                    .add_operands(&[a, b, acc])
                    .add_results(&[elem])
                    .build()?,
            )
        } else {
            let prod = self.elem_arith(BinOp::Mul, elem, a, b)?;
            let prod = self.push(block, prod)?;
            let sum = self.elem_arith(BinOp::Add, elem, acc, prod)?;
            self.push(block, sum)
        }
    }

    /// The arith op for a tile loop body, on the element type.
    pub(in crate::codegen) fn elem_arith(
        &self,
        op: BinOp,
        elem: Type<'c>,
        a: Value<'c, '_>,
        b: Value<'c, '_>,
    ) -> Result<Operation<'c>> {
        let loc = self.loc;
        Ok(if self.is_float(elem) {
            match op {
                BinOp::Add => arith::addf(a, b, loc),
                BinOp::Sub => arith::subf(a, b, loc),
                BinOp::Mul => arith::mulf(a, b, loc),
                BinOp::Div => arith::divf(a, b, loc),
                BinOp::Rem => arith::remf(a, b, loc),
                _ => bail!("operator not supported for tile operands"),
            }
        } else if self.is_int(elem) {
            match op {
                BinOp::Add => arith::addi(a, b, loc),
                BinOp::Sub => arith::subi(a, b, loc),
                BinOp::Mul => arith::muli(a, b, loc),
                BinOp::Div => arith::divsi(a, b, loc),
                BinOp::Rem => arith::remsi(a, b, loc),
                _ => bail!("operator not supported for tile operands"),
            }
        } else {
            bail!("tile ops need a numeric element type, got {elem}")
        })
    }
}
