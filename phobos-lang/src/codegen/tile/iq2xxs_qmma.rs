// IQ2_XXS on the integer tensor cores, staged through shared memory.
//
// The same shape as `iq1s_qmma.rs`, and it fits for the same reason: the format
// decodes to a magnitude times a sign, both already i8, so a weight is an exact
// `i8` in -43..43 and needs no correction term. It is the second largest item
// in a prompt pass after IQ1_S, 24.0% of one against IQ1_S's 25.0%, and the
// expansion it replaces pays the same 11x in bytes.
//
// Only the staged form exists here. IQ1_S has both because the register one
// came first; there is no reason to write a second register form now that the
// staged one measures faster on the same shapes.

use super::iq2xxs::{IQ2XXS_BLOCK_BYTES, IQ2XXS_LANE};
use super::*;

impl<'c> Codegen<'c> {
    /// out[i, j] = sum_b (sum_{k in group b} a[i, k] * w[j, k]) with `w`
    /// decoded from IQ2_XXS: the batched contraction, both table lookups and
    /// both scales as one operation.
    ///
    /// Needs one patch a warp, for the reason
    /// [`Self::iq1s_qmma_staged_into`] gives.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn iq2xxs_qmma_staged_into(
        &mut self,
        block: &Block<'c>,
        aq: &MemVal<'c>,
        asc: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        grid: &MemVal<'c>,
        signs: &MemVal<'c>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        for (v, what) in [
            (aq, "iq2xxs_qmma_t a"),
            (asc, "iq2xxs_qmma_t a scales"),
            (qb, "iq2xxs_qmma_t qb"),
            (d, "iq2xxs_qmma_t d"),
            (grid, "iq2xxs_qmma_t grid"),
            (signs, "iq2xxs_qmma_t signs"),
        ] {
            if v.shape.len() != 2 {
                bail!("{what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{what} must be a fully in-bounds slice");
            }
        }
        if aq.elem != self.i8_t || qb.elem != self.i8_t {
            bail!("iq2xxs_qmma_t contracts an int8 activation against IQ2_XXS's raw bytes");
        }
        if d.elem != self.f16_t {
            bail!("iq2xxs_qmma_t's block scale must be f16");
        }
        if out.elem != self.f32_t || out.is_masked() {
            bail!("iq2xxs_qmma_t needs a fully in-bounds f32 destination");
        }
        if !self.has_int8_mma() {
            bail!("iq2xxs_qmma_t needs the integer tensor cores (sm_75 or later)");
        }
        if self.cta_threads % WARP != 0 {
            bail!("iq2xxs_qmma_t needs a CTA that is a whole number of warps");
        }
        let (md, nd) = (aq.shape[0], qb.shape[0]);
        if md == DYN || nd == DYN {
            bail!("iq2xxs_qmma_t needs a static output shape");
        }
        self.check_shapes(&[md, nd], &out.shape, "iq2xxs_qmma_t destination")?;
        if md % IMMA_TILE != 0 || nd % IMMA_TILE != 0 {
            bail!("iq2xxs_qmma_t needs an output tile that is a multiple of {IMMA_TILE} both ways");
        }
        if aq.shape[1] != DYN && aq.shape[1] % 256 != 0 {
            bail!("iq2xxs_qmma_t needs a whole number of 256-element blocks");
        }

        let (rt, ct) = (md / IMMA_TILE, nd / IMMA_TILE);
        let (rm, rn) = self.qmma_patch(rt, ct);
        let patches = (rt / rm) * (ct / rn);
        let warps = self.cta_threads / WARP;
        if patches != warps {
            bail!("staged iq2xxs_qmma_t needs one patch a warp, got {patches} for {warps} warps");
        }
        let entries = nd * (Q8_BLOCK / IQ2XXS_LANE);
        if entries % self.cta_threads != 0 {
            bail!("staged iq2xxs_qmma_t needs the CTA to divide the {entries} lanes it stages");
        }

        let (i32_t, f32_t) = (self.i32_t, self.f32_t);
        let vec4_i8 = Type::vector(&[4], self.i8_t);
        let frag_t = Type::vector(&[1, 4], self.i8_t);
        let acc_t = Type::vector(&[1, 2], i32_t);
        let stage = self.alloc_tile_shaped(block, self.i8_t, &[nd, Q8_BLOCK])?;

        let one = self.const_index(block, 1)?;
        let kd = if aq.shape[1] == DYN {
            self.push(block, memref::dim(aq.mem, one, self.loc))?
        } else {
            self.const_index(block, aq.shape[1])?
        };

        let tid = self.thread_id(block)?;
        let warp_w = self.const_index(block, WARP)?;
        let warp = self.divui(block, tid, warp_w)?;
        let lane = self.remui(block, tid, warp_w)?;
        let four = self.const_index(block, 4)?;
        let two = self.const_index(block, 2)?;
        let quad = self.divui(block, lane, four)?;
        let in_quad = self.remui(block, lane, four)?;
        let k_off = self.muli(block, in_quad, four)?;
        let d_col = self.muli(block, in_quad, two)?;

        let across = self.const_index(block, ct / rn)?;
        let ui = self.divui(block, warp, across)?;
        let uj = self.remui(block, warp, across)?;
        let patch_m = self.const_index(block, rm * IMMA_TILE)?;
        let patch_n = self.const_index(block, rn * IMMA_TILE)?;
        let i0 = self.muli(block, ui, patch_m)?;
        let j0 = self.muli(block, uj, patch_n)?;
        let row_base = self.addi(block, i0, quad)?;
        let col_base = self.addi(block, j0, d_col)?;
        let frag_base = self.addi(block, j0, quad)?;

        let mut a_rows = Vec::with_capacity(rm as usize);
        for r in 0..rm {
            let off = self.const_index(block, r * IMMA_TILE)?;
            a_rows.push(self.addi(block, row_base, off)?);
        }
        let (mut w_rows, mut out_cols) = (Vec::new(), Vec::new());
        for c in 0..rn {
            let off = self.const_index(block, c * IMMA_TILE)?;
            w_rows.push(self.addi(block, frag_base, off)?);
            out_cols.push(self.addi(block, col_base, off)?);
        }
        let per_thread = entries / self.cta_threads;
        let mut mine = Vec::with_capacity(per_thread as usize);
        for e in 0..per_thread {
            let off = self.const_index(block, e * self.cta_threads)?;
            mine.push(self.addi(block, tid, off)?);
        }

        let lanes = (rm * rn * 2) as usize;
        let mut args = vec![(self.index_t, self.loc)];
        args.extend(std::iter::repeat_n((f32_t, self.loc), lanes));
        let kb = Block::new(&args);
        let k = detach(kb.argument(0)?.into());
        let mut accs = Vec::with_capacity(lanes);
        for slot in 0..lanes {
            accs.push(detach(kb.argument(slot + 1)?.into()));
        }

        let blk_w = self.const_index(&kb, Q8_BLOCK)?;
        let b = self.divui(&kb, k, blk_w)?;
        let groups = self.const_index(&kb, 256 / Q8_BLOCK)?;
        let blk = self.divui(&kb, b, groups)?;
        let ib = self.remui(&kb, b, groups)?;
        let blk_bytes = self.const_index(&kb, IQ2XXS_BLOCK_BYTES)?;
        let blk_off = self.muli(&kb, blk, blk_bytes)?;
        let tables = QTables { grid, signs };
        let eight = self.const_index(&kb, IQ2XXS_LANE)?;
        let four_ib = self.muli(&kb, ib, four)?;

        // Stage the block, one format lane a thread a turn. `iq2xxs_lane` maps
        // a lane index to the geometry, and the group's four lanes are
        // `4 * ib + l`, so the same helper the matvec uses serves here.
        for entry in &mine {
            let j = self.divui(&kb, *entry, four)?;
            let l = self.remui(&kb, *entry, four)?;
            let fmt_lane = self.addi(&kb, four_ib, l)?;
            let geom = self.iq2xxs_lane(&kb, fmt_lane)?;
            let at = BlockAt {
                j,
                blk,
                off: blk_off,
            };
            let dec = self.iq2xxs_block(&kb, &geom, qb, d, &tables, &at)?;
            // A magnitude times its sign is the weight, and both are already
            // i8: -43..43, which is why this format reaches the tensor cores
            // without a correction term.
            let v = self.push(&kb, arith::muli(dec.grid_v, dec.signs_v, self.loc))?;
            let dst = self.muli(&kb, l, eight)?;
            self.vec_store_al(&kb, v, stage.mem, &[j, dst], 8)?;
        }
        self.barrier(&kb)?;

        let k_from = self.addi(&kb, k, k_off)?;
        let halves = Q8_BLOCK / IMMA_K;
        let (mut k_cols, mut stage_cols) = (Vec::new(), Vec::new());
        for h in 0..halves {
            let off = self.const_index(&kb, h * IMMA_K)?;
            k_cols.push(self.addi(&kb, k_from, off)?);
            stage_cols.push(self.addi(&kb, k_off, off)?);
        }

        let mut a_frags = Vec::new();
        for row in &a_rows {
            for col in &k_cols {
                let v = self.vec_load_al(&kb, aq.mem, &[*row, *col], vec4_i8, 4)?;
                a_frags.push(self.vec_shape_cast(&kb, v, frag_t)?);
            }
        }
        let mut w_frags = Vec::new();
        for row in &w_rows {
            for col in &stage_cols {
                let v = self.vec_load_al(&kb, stage.mem, &[*row, *col], vec4_i8, 4)?;
                w_frags.push(self.vec_shape_cast(&kb, v, frag_t)?);
            }
        }

        let zero_i = self.zero_scalar(&kb, i32_t)?;
        let empty = self.vec_broadcast(&kb, zero_i, acc_t)?;
        let shape = self.mma_shape(IMMA_TILE, IMMA_TILE, IMMA_K)?;

        // The scale belongs to an output column, which is not the column whose
        // fragment this lane staged, so it takes its own.
        let mut w_scales = Vec::with_capacity(rn as usize * 2);
        for out_col in &out_cols {
            for dj in 0..2 {
                let off = self.const_index(&kb, dj)?;
                let col = self.addi(&kb, *out_col, off)?;
                let at = BlockAt {
                    j: col,
                    blk,
                    off: blk_off,
                };
                w_scales.push(self.iq2xxs_db(&kb, qb, d, &at, ib)?);
            }
        }

        let mut next = Vec::with_capacity(lanes);
        for r in 0..rm as usize {
            let sa = self.push(&kb, memref::load(asc.mem, &[a_rows[r], b], self.loc))?;
            for c in 0..rn as usize {
                let mut sum = empty;
                for h in 0..halves as usize {
                    sum = self.mma_sync(
                        &kb,
                        a_frags[r * halves as usize + h],
                        w_frags[c * halves as usize + h],
                        sum,
                        shape,
                        acc_t,
                    )?;
                }
                for dj in 0..2 {
                    let sw = w_scales[c * 2 + dj as usize];
                    let scale = self.push(&kb, arith::mulf(sa, sw, self.loc))?;
                    let raw = self.vec_extract(&kb, sum, &[0, dj], i32_t)?;
                    let as_f = self.small_int_to_f32(&kb, raw)?;
                    let slot = (r * rn as usize + c) * 2 + dj as usize;
                    next.push(self.elem_mac(&kb, f32_t, as_f, scale, accs[slot])?);
                }
            }
        }
        self.barrier(&kb)?;
        kb.append_operation(scf::r#yield(&next, self.loc));

        let zero_k = self.const_index(block, 0)?;
        let step = self.const_index(block, Q8_BLOCK)?;
        let init = self.zero_scalar(block, f32_t)?;
        let mut operands = vec![zero_k, kd, step];
        operands.extend(std::iter::repeat_n(init, lanes));
        let kr = Region::new();
        kr.append_block(kb);
        let loop_op = block.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&operands)
                .add_results(&vec![f32_t; lanes])
                .add_regions([kr])
                .build()?,
        );

        for (r, row) in a_rows.iter().enumerate() {
            for (c, out_col) in out_cols.iter().enumerate() {
                for dj in 0..2 {
                    let off = self.const_index(block, dj)?;
                    let col = self.addi(block, *out_col, off)?;
                    let slot = (r * rn as usize + c) * 2 + dj as usize;
                    let value = detach(loop_op.result(slot)?.into());
                    block.append_operation(memref::store(value, out.mem, &[*row, col], self.loc));
                }
            }
        }
        self.barrier(block)?;
        Ok(())
    }
}
