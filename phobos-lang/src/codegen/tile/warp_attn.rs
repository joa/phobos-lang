// A warp-partitioned online-softmax accumulation: each warp of the CTA owns
// an independent key sub-range.
//
// This bypasses `distribute`: its CTA-collective ops end in a CTA-wide
// `gpu.barrier` that every thread must reach the same number of times, and
// these warps take different trip counts. Per-key state stays in registers
// instead, reduced only with `gpu.shuffle`, which is scoped to the issuing
// warp's 32 lanes; the one barrier sits after every warp's own loop.
//
// The head dimension splits across a warp's 32 lanes (`D / 32` each): a
// key's QK dot is a per-lane multiply-accumulate plus an xor-shuffle
// butterfly, the accumulator keeps the same split so each lane's `WACC`
// store is lane-local, and `m`/`l` are all-reduced so only lane 0 commits
// them.
//
// It is its own primitive rather than a warp-scoped mode of
// `dot_t`/`rowmax`/`dot` so a thread-range-virtualized path through those
// shared kernels never risks the other callers.

use super::*;

impl<'c> Codegen<'c> {

    /// The warp's partial attention over resolved operands: the tiles
    /// `q, K, V, WM, WL, WACC` and the scalars `scale, lo, hi, col`.
    pub(in crate::codegen) fn warp_partial_raw(
        &mut self,
        block: &Block<'c>,
        tiles: [&MemVal<'c>; 6],
        scalars: [Value<'c, 'c>; 4],
    ) -> Result<()> {
        let [q_mv, k_mv, v_mv, wm_mv, wl_mv, wacc_mv] = tiles;
        let [scale_v, lo_v, hi_v, col_v] = scalars;
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
        if wct != self.cta_threads / WARP {
            bail!(
                "warp_partial's WM/WL/WACC are sized for {wct} warps, but this kernel launches \
                 {} ({}-thread CTA / {WARP}); every launched warp writes its own row, so the two \
                 must match",
                self.cta_threads / WARP,
                self.cta_threads
            );
        }

        let tid = self.thread_id(block)?;
        let warp_w = self.const_index(block, WARP)?;
        let warp_id = self.divui(block, tid, warp_w)?;
        let lane = self.remui(block, tid, warp_w)?;

        // This warp's own slice of [lo, hi): ceil-divide the block's range
        // into `wct` pieces and clamp both ends to hi, so a warp past the
        // end runs zero iterations instead of needing a separate guard.
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
        let acc_len = qg as usize * per_row;
        let mut inits = Vec::with_capacity(acc_len);
        for _ in 0..qg {
            inits.push(neg_inf);
            inits.extend(std::iter::repeat_n(zero_f, per_row - 1));
        }

        // Widest f16 vector load a lane's dpl-wide slice divides evenly into,
        // falling back to scalar when none does. Every such load is
        // `2 * dpl`-byte aligned, since `col` is a multiple of `D = 32 * dpl`,
        // `lane * dpl` is a multiple of `dpl`, and the row pitch `KW` is a
        // multiple of `D` per the kernel's own `@aligned(KW = D)`.
        let vw = [8, 4, 2, 1].into_iter().find(|w| dpl % w == 0).unwrap_or(1);
        let k16_vec_t = Type::vector(&[vw as u64], self.f16_t);
        let f32_vec_t = Type::vector(&[vw as u64], self.f32_t);
        let vec_align = vw * 2; // f16 element size in bytes

        // Software-pipelined K/V load: iteration kt issues kt+1's load before
        // consuming its own carried-in values, overlapping load latency with
        // the QK dot, shuffle, and softmax update. Only the raw f16 vector
        // loop-carries, not the widened f32 slice, to halve the carried
        // registers.
        //
        // A load's address clamps to `ghi - 1`, never `kt + 1` directly: a
        // warp's range can be empty (`glo == ghi`), and even a non-empty
        // range's last iteration has no `kt + 1` to load. The clamp is in
        // bounds either way since `ghi <= hi <= NK`, and an empty range never
        // runs the body that would read it.
        let ghi_m1 = self.subi(block, ghi, one)?;
        let first_row = self.minsi(block, glo, ghi_m1)?;

        // Shared by the prologue load (`self`/`block`) and the in-loop
        // prefetch (`cg`/`lblk`), hence the explicit receiver parameters.
        let load_raw = |cg: &mut Self,
                        blk: &Block<'c>,
                        row: Value<'c, 'c>|
         -> Result<(Vec<Value<'c, 'c>>, Vec<Value<'c, 'c>>)> {
            let k_col = cg.addi(blk, col_v, lane_off)?;
            let mut k_raw = Vec::with_capacity((dpl / vw) as usize);
            let mut v_raw = Vec::with_capacity((dpl / vw) as usize);
            let mut off = 0;
            while off < dpl {
                let jc = cg.const_index(blk, off)?;
                let kc = cg.addi(blk, k_col, jc)?;
                if vw > 1 {
                    k_raw.push(cg.vec_load_al(blk, k_mv.mem, &[row, kc], k16_vec_t, vec_align)?);
                    v_raw.push(cg.vec_load_al(blk, v_mv.mem, &[row, kc], k16_vec_t, vec_align)?);
                } else {
                    k_raw.push(cg.load_as(blk, k_mv.mem, &[row, kc], cg.f32_t)?);
                    v_raw.push(cg.load_as(blk, v_mv.mem, &[row, kc], cg.f32_t)?);
                }
                off += vw;
            }
            Ok((k_raw, v_raw))
        };

        let (k_raw0, v_raw0) = load_raw(self, block, first_row)?;
        let chunks = k_raw0.len();
        inits.extend_from_slice(&k_raw0);
        inits.extend_from_slice(&v_raw0);

        let finals_all = self.carry_loop(block, glo, ghi, one, &inits, |cg, lblk, kt, accs| {
            let row_accs = &accs[..acc_len];
            let cur_k = &accs[acc_len..acc_len + chunks];
            let cur_v = &accs[acc_len + chunks..acc_len + 2 * chunks];

            // Issue next iteration's load before this iteration touches the
            // carried-in "current" values below.
            let kt_plus1 = cg.addi(lblk, kt, one)?;
            let next_row = cg.minsi(lblk, kt_plus1, ghi_m1)?;
            let (next_k, next_v) = load_raw(cg, lblk, next_row)?;

            // Widen this iteration's carried-in raw K/V once, here (a no-op
            // copy in the scalar fallback, where `load_raw` already
            // produced f32).
            let mut k_vals = Vec::with_capacity(dpl as usize);
            let mut v_vals = Vec::with_capacity(dpl as usize);
            if vw > 1 {
                for &raw_k in cur_k {
                    let fk = cg.vec_extf(lblk, raw_k, f32_vec_t)?;
                    for e in 0..vw {
                        k_vals.push(cg.vec_extract(lblk, fk, &[e], cg.f32_t)?);
                    }
                }
                for &raw_v in cur_v {
                    let fv = cg.vec_extf(lblk, raw_v, f32_vec_t)?;
                    for e in 0..vw {
                        v_vals.push(cg.vec_extract(lblk, fv, &[e], cg.f32_t)?);
                    }
                }
            } else {
                k_vals.extend_from_slice(cur_k);
                v_vals.extend_from_slice(cur_v);
            }

            let mut next = Vec::with_capacity(accs.len());
            for i in 0..qg {
                let base = i as usize * per_row;
                let m_old = row_accs[base];
                let l_old = row_accs[base + 1];
                let acc_old = &row_accs[base + 2..base + 2 + dpl as usize];

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
            next.extend_from_slice(&next_k);
            next.extend_from_slice(&next_v);
            Ok(next)
        })?;
        let finals = &finals_all[..acc_len];

        let lane_zero = self.const_index(block, 0)?;
        let is_lead = self.push(
            block,
            arith::cmpi(
                self.ctx,
                arith::CmpiPredicate::Eq,
                lane,
                lane_zero,
                self.loc,
            ),
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

        self.barrier(block)
    }

}
