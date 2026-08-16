// A warp-partitioned online-softmax accumulation, the register-resident
// counterpart to `attention_split_src`'s old one-chain-per-block loop.
//
// `distribute`'s CTA-collective tile ops (`dot_t`, `rowmax`, `exp`, `dot`, ...)
// all end in a CTA-wide `gpu.barrier`, which every one of the block's 256
// threads must reach the same number of times. That is fine when the whole
// block marches through one shared loop together, which is what every other
// kernel in this file does -- but it is exactly what stands in the way of
// giving the block's eight warps eight *independent* key sub-ranges: warps
// with different trip counts would call that barrier a different number of
// times, which hangs rather than merely slows down.
//
// `warp_partial` sidesteps the barrier question instead of solving it: the
// whole per-key computation lives in registers, and the only cross-lane
// communication is `gpu.shuffle`, which synchronizes exactly the 32 lanes of
// the issuing warp and nothing more. A warp's own loop trip count is a
// function of its own warp id and is uniform across its 32 lanes (every lane
// of one warp computes the same `warp_id`), so the shuffles inside it are
// always issued in convergence; different warps finishing at different
// iteration counts is ordinary, unproblematic SIMT divergence, the same kind
// `distribute` already relies on when a tile is smaller than the CTA. The one
// barrier this function does emit sits after every warp's loop has ended,
// reached by all 256 threads exactly once regardless of how many iterations
// their own warp took -- safe by the same "same call count for every thread"
// rule the rest of this compiler's barriers already follow.
//
// The head dimension is spread across a warp's 32 lanes (`D / 32` elements
// each), so one key's QK dot product is a short per-lane multiply-accumulate
// followed by a five-step xor-shuffle butterfly, not the single thread's
// `D`-deep serial reduction `dot_t` uses today. The accumulator rides the
// same split: each lane only ever holds its own slice of `acc[i, :]`, so the
// final store back to the caller's `WACC` tile is one lane-local write with
// no shuffle at all. `m`/`l` are scalars every lane ends up holding an
// identical copy of (the butterfly is an all-reduce, not a reduce-to-lane-0),
// so only lane 0 commits them, to avoid every lane redundantly storing the
// same word.
//
// This is deliberately its own primitive rather than a warp-scoped mode of
// `dot_t`/`rowmax`/`dot`: those are used by every other kernel in the tree,
// and giving them a second, thread-range-virtualized code path would risk
// every one of them for the sake of this one caller. `warp_partial` touches
// nothing outside this file and `expr.rs`'s dispatch table.

use super::*;

impl<'c> Codegen<'c> {
    /// `warp_partial(q, K, V, lo, hi, col, WM, WL, WACC, scale)`.
    ///
    /// `q` is the caller's already-staged `[QG, D]` query tile. `K`/`V` are
    /// the raw `[NK, KW]` f16 cache tensors; `col` is the byte-free column
    /// offset of this call's key head within them (`h / G * D`, already
    /// computed by the caller exactly as the old design's `dot_t` calls
    /// wanted it). `lo`/`hi` bound the *block's* key range, same as before;
    /// this function further divides `[lo, hi)` into one piece per warp.
    ///
    /// `WM`/`WL` are `[QG, W]` and `WACC` is `[QG * W, D]`, all freshly
    /// `var`-declared by the caller (`W` is a new `@autotune` symbol, not
    /// derived from anything here) -- this function only ever writes warp
    /// `w`'s own column/row range of them, so no two warps ever address the
    /// same byte, and a plain CTA barrier at the end (not a per-iteration
    /// one) is enough to publish the result to whatever combine step reads
    /// these tiles next.
    pub(in crate::codegen) fn emit_warp_partial(
        &mut self,
        block: &Block<'c>,
        args: &[Expr],
    ) -> Result<Rv<'c>> {
        let [q, k, v, lo, hi, col, wm, wl, wacc, scale] = args else {
            bail!("warp_partial expects (q, K, V, lo, hi, col, WM, WL, WACC, scale)");
        };

        let q_mv = self.named_tile(q, "warp_partial q")?;
        let k_mv = self.named_tensor(k, "warp_partial K")?;
        let v_mv = self.named_tensor(v, "warp_partial V")?;
        let wm_mv = self.named_tile(wm, "warp_partial WM")?;
        let wl_mv = self.named_tile(wl, "warp_partial WL")?;
        let wacc_mv = self.named_tile(wacc, "warp_partial WACC")?;

        if q_mv.is_masked() {
            bail!("warp_partial needs an unmasked q tile");
        }
        let qg = q_mv.shape[0];
        let d = q_mv.shape[1];
        if d % WARP != 0 {
            bail!("warp_partial needs a head dim divisible by {WARP}, got {d}");
        }
        let dpl = d / WARP;
        if wm_mv.shape != [qg, wm_mv.shape[1]] || wl_mv.shape != wm_mv.shape {
            bail!("warp_partial WM/WL must both be [QG, W]");
        }
        let wct = wm_mv.shape[1];
        if wacc_mv.shape != [qg * wct, d] {
            bail!("warp_partial WACC must be [QG * W, D]");
        }

        let scale_v = self.emit_scalar(block, scale)?;
        let lo_v = self.emit_index(block, lo, "warp_partial lo")?;
        let hi_v = self.emit_index(block, hi, "warp_partial hi")?;
        let col_v = self.emit_index(block, col, "warp_partial col")?;

        let tid = self.thread_id(block)?;
        let warp_w = self.const_index(block, WARP)?;
        let warp_id = self.divui(block, tid, warp_w)?;
        let lane = self.remui(block, tid, warp_w)?;

        // This warp's own slice of [lo, hi): ceil-divide the block's range
        // into `wct` pieces and clamp both ends to hi, so a warp past the end
        // (the range does not evenly divide `wct`) simply runs zero
        // iterations rather than needing a separate guard.
        let wct_v = self.const_index(block, wct)?;
        let span = self.subi(block, hi_v, lo_v)?;
        let one = self.const_index(block, 1)?;
        let span_ceil = self.addi(block, span, self.subi(block, wct_v, one)?)?;
        let share = self.divui(block, span_ceil, wct_v)?;
        let w_off = self.muli(block, warp_id, share)?;
        let glo = self.minsi(block, self.addi(block, lo_v, w_off)?, hi_v)?;
        let ghi = self.minsi(block, self.addi(block, glo, share)?, hi_v)?;

        let dpl_v = self.const_index(block, dpl)?;
        let lane_off = self.muli(block, lane, dpl_v)?;

        let neg_inf = self.push(
            block,
            arith::constant(
                self.ctx,
                FloatAttribute::new(self.ctx, self.f32_t, -3.0e38).into(),
                self.loc,
            ),
        )?;
        let zero_f = self.zero_scalar(block, self.f32_t)?;

        // iter_args, one row of QG at a time: m, l, then dpl accumulator
        // elements (the lane's own slice of acc[i, :]).
        let per_row = 2 + dpl as usize;
        let mut inits = Vec::with_capacity(qg as usize * per_row);
        for _ in 0..qg {
            inits.push(neg_inf);
            inits.push(zero_f);
            for _ in 0..dpl {
                inits.push(zero_f);
            }
        }

        let finals = self.carry_loop(block, glo, ghi, one, &inits, |cg, lblk, kt, accs| {
            let k_col = cg.addi(lblk, col_v, lane_off)?;
            // This warp's own dpl-wide slice of k[kt, :] and v[kt, :],
            // widened to f32 once and reused across every query row.
            let mut k_vals = Vec::with_capacity(dpl as usize);
            let mut v_vals = Vec::with_capacity(dpl as usize);
            for j in 0..dpl {
                let jc = cg.const_index(lblk, j)?;
                let kc = cg.addi(lblk, k_col, jc)?;
                k_vals.push(cg.load_as(lblk, k_mv.mem, &[kt, kc], cg.f32_t)?);
                v_vals.push(cg.load_as(lblk, v_mv.mem, &[kt, kc], cg.f32_t)?);
            }

            let mut next = Vec::with_capacity(accs.len());
            for i in 0..qg {
                let base = i as usize * per_row;
                let m_old = accs[base];
                let l_old = accs[base + 1];
                let acc_old = &accs[base + 2..base + 2 + dpl as usize];

                let i_idx = cg.const_index(lblk, i)?;
                let mut partial = zero_f;
                for (j, &k_val) in k_vals.iter().enumerate() {
                    let jc = cg.const_index(lblk, j as i64)?;
                    let d_idx = cg.addi(lblk, lane_off, jc)?;
                    let q_val = cg.push(lblk, memref::load(q_mv.mem, &[i_idx, d_idx], cg.loc))?;
                    partial = cg.elem_mac(lblk, cg.f32_t, q_val, k_val, partial)?;
                }
                // Warp all-reduce: every lane ends this loop holding the same
                // full dot product, not just lane 0.
                let mut s = partial;
                let mut mask = WARP / 2;
                while mask >= 1 {
                    let other = cg.shfl_xor_f32(lblk, s, mask)?;
                    s = cg.push(lblk, arith::addf(s, other, cg.loc))?;
                    mask /= 2;
                }
                s = cg.push(lblk, arith::mulf(s, scale_v, cg.loc))?;

                let new_m = cg.fmax(lblk, m_old, s)?;
                let m_diff = cg.push(lblk, arith::subf(m_old, new_m, cg.loc))?;
                let corr = cg.approx_exp(lblk, m_diff)?;
                let s_diff = cg.push(lblk, arith::subf(s, new_m, cg.loc))?;
                let p = cg.approx_exp(lblk, s_diff)?;
                let new_l = cg.elem_mac(lblk, cg.f32_t, l_old, corr, p)?;

                next.push(new_m);
                next.push(new_l);
                for (&v_val, &a_old) in v_vals.iter().zip(acc_old) {
                    let pv = cg.push(lblk, arith::mulf(p, v_val, cg.loc))?;
                    next.push(cg.elem_mac(lblk, cg.f32_t, a_old, corr, pv)?);
                }
            }
            Ok(next)
        })?;

        let lane_zero = self.const_index(block, 0)?;
        let is_lead = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, lane_zero, self.loc),
        )?;

        for i in 0..qg {
            let base = i as usize * per_row;
            let (m_f, l_f) = (finals[base], finals[base + 1]);
            let i_idx = self.const_index(block, i)?;

            let lead = Block::new(&[]);
            lead.append_operation(memref::store(m_f, wm_mv.mem, &[i_idx, warp_id], self.loc));
            lead.append_operation(memref::store(l_f, wl_mv.mem, &[i_idx, warp_id], self.loc));
            lead.append_operation(scf::r#yield(&[], self.loc));
            let lead_region = Region::new();
            lead_region.append_block(lead);
            block.append_operation(scf::r#if(
                is_lead,
                &[],
                lead_region,
                Region::new(),
                self.loc,
            ));

            let row_base = self.const_index(block, i * wct)?;
            let row = self.addi(block, row_base, warp_id)?;
            for j in 0..dpl as usize {
                let jc = self.const_index(block, j as i64)?;
                let col_idx = self.addi(block, lane_off, jc)?;
                let val = finals[base + 2 + j];
                block.append_operation(memref::store(val, wacc_mv.mem, &[row, col_idx], self.loc));
            }
        }

        self.barrier(block)?;
        Ok(Rv::Scalar(self.const_index(block, 0)?))
    }

    /// Resolves a named, already-bound tile (`var`-declared): the arguments
    /// this call and [`Self::named_tensor`] accept are always a bare
    /// identifier, the same restriction `atomic_add`/`grid_barrier` place on
    /// their own state operands, and for the same reason -- these read the
    /// buffer's own memref directly rather than going through the general
    /// slice/subview machinery, so there is no offset to fold in.
    fn named_tile(&self, e: &Expr, what: &str) -> Result<MemVal<'c>> {
        let Expr::Var(name) = e else {
            bail!("{what} expects a named tile variable");
        };
        match self.lookup(name) {
            Some(Binding::Tile(t) | Binding::View(t)) => Ok(t),
            _ => bail!("{what}: '{name}' is not a tile"),
        }
    }

    fn named_tensor(&self, e: &Expr, what: &str) -> Result<MemVal<'c>> {
        let Expr::Var(name) = e else {
            bail!("{what} expects a named tensor parameter");
        };
        match self.lookup(name) {
            Some(Binding::Tensor(t)) => Ok(t),
            _ => bail!("{what}: '{name}' is not a tensor parameter"),
        }
    }
}
