// The IQ2 and IQ3 formats on the integer tensor cores, with the decoded
// weight staged through shared memory.
//
// One macro covers every format that decodes to `scale * magnitude * sign`
// with both planes already i8. The weight is then an exact i8, and the
// contraction needs no correction term.
//
// IQ2_S and IQ2_XS carry two scales per 32-element block: the low nibble for
// lanes 0 and 1, the high one for lanes 2 and 3. Using one scale for them is
// a silent correctness bug. With `split`, each k step's two `mma` halves get
// their own accumulator and scale, which doubles the epilogue.

use super::iq2s::{IQ2S_BLOCK_BYTES, IQ2S_LANE};
use super::iq3s::{IQ3S_BLOCK_BYTES, IQ3S_LANE};
use super::iq3xxs::{IQ3XXS_BLOCK_BYTES, IQ3XXS_LANE};
use super::iq2xs::{IQ2XS_BLOCK_BYTES, IQ2XS_LANE};
use super::iq2xxs::{IQ2XXS_BLOCK_BYTES, IQ2XXS_LANE};
use super::*;

/// The lane's eight magnitudes: one eight-wide grid entry for IQ2, or two
/// four-wide ones joined for IQ3.
macro_rules! qmma_grid {
    ($self:ident, $kb:ident, $dec:ident, $g:ident) => {
        $dec.$g
    };
    ($self:ident, $kb:ident, $dec:ident, $g1:ident, $g2:ident) => {{
        let want = Type::vector(&[8], $self.i8_t);
        $self.vec_shuffle(&$kb, $dec.$g1, $dec.$g2, &[0, 1, 2, 3, 4, 5, 6, 7], want)?
    }};
}

macro_rules! signed_qmma {
    ($fn:ident, $label:literal, $lane_fn:ident, $block_fn:ident,
     $bytes:ident, $lane_w:ident, $scale:ident, $split:expr, $($grid_f:ident),+) => {
        impl<'c> Codegen<'c> {
            /// out[i, j] = sum_b (sum_{k in group b} a[i, k] * w[j, k]) with
            /// `w` decoded from the format. The contraction, both table
            /// lookups and both scales are one operation. The decoded weight
            /// is staged through shared memory, so the CTA decodes each column
            /// once.
            ///
            /// Needs one patch per warp, as
            /// [`Codegen::iq1s_qmma_staged_into`] does.
            #[allow(clippy::too_many_arguments)]
            pub(in crate::codegen) fn $fn(
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
                    (aq, concat!($label, " a")),
                    (asc, concat!($label, " a scales")),
                    (qb, concat!($label, " qb")),
                    (d, concat!($label, " d")),
                    (grid, concat!($label, " grid")),
                    (signs, concat!($label, " signs")),
                ] {
                    if v.shape.len() != 2 {
                        bail!("{what} must be a rank-2 tile");
                    }
                    if v.is_masked() {
                        bail!("{what} must be a fully in-bounds slice");
                    }
                }
                if aq.elem != self.i8_t || qb.elem != self.i8_t {
                    bail!(concat!(
                        $label,
                        " contracts an int8 activation against the format's raw bytes"
                    ));
                }
                if d.elem != self.f16_t {
                    bail!(concat!($label, "'s block scale must be f16"));
                }
                if out.elem != self.f32_t || out.is_masked() {
                    bail!(concat!($label, " needs a fully in-bounds f32 destination"));
                }
                if !self.has_int8_mma() {
                    bail!(concat!(
                        $label,
                        " needs the integer tensor cores (sm_75 or later)"
                    ));
                }
                if self.cta_threads % WARP != 0 {
                    bail!(concat!(
                        $label,
                        " needs a CTA that is a whole number of warps"
                    ));
                }
                let (md, nd) = (aq.shape[0], qb.shape[0]);
                if md == DYN || nd == DYN {
                    bail!(concat!($label, " needs a static output shape"));
                }
                self.check_shapes(&[md, nd], &out.shape, concat!($label, " destination"))?;
                if md % IMMA_TILE != 0 || nd % IMMA_TILE != 0 {
                    bail!(
                        "{} needs an output tile that is a multiple of {IMMA_TILE} both ways",
                        $label
                    );
                }
                if aq.shape[1] != DYN && aq.shape[1] % 256 != 0 {
                    bail!(concat!(
                        $label,
                        " needs a whole number of 256-element blocks"
                    ));
                }

                let (rt, ct) = (md / IMMA_TILE, nd / IMMA_TILE);
                let (rm, rn) = self.qmma_patch(rt, ct);
                let patches = (rt / rm) * (ct / rn);
                let warps = self.cta_threads / WARP;
                if patches != warps {
                    bail!(
                        "{} needs one patch a warp, got {patches} for {warps}",
                        $label
                    );
                }
                let entries = nd * (Q8_BLOCK / $lane_w);
                if entries % self.cta_threads != 0 {
                    bail!(
                        "{} needs the CTA to divide its {entries} staging lanes",
                        $label
                    );
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
                let blk_bytes = self.const_index(&kb, $bytes)?;
                let tables = QTables { grid, signs };
                let eight = self.const_index(&kb, $lane_w)?;
                let four_ib = self.muli(&kb, ib, four)?;

                // Stage the block, one format lane per thread per turn. A
                // group's four lanes are `4 * ib + l`, so the matvec's lane
                // helper gives their geometry.
                for entry in &mine {
                    let j = self.divui(&kb, *entry, four)?;
                    let l = self.remui(&kb, *entry, four)?;
                    let fmt_lane = self.addi(&kb, four_ib, l)?;
                    let geom = self.$lane_fn(&kb, fmt_lane)?;
                    let at = self.raw_block_at(&kb, j, blk, blk_bytes)?;
                    let dec = self.$block_fn(&kb, &geom, qb, d, &tables, &at)?;
                    // The weight is magnitude times sign, both already i8.
                    let mag = qmma_grid!(self, kb, dec, $($grid_f),+);
                    let v = self.push(&kb, arith::muli(mag, dec.signs_v, self.loc))?;
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

                // The scale belongs to an output column, not the column whose
                // fragment this lane staged, so the lane decodes that column's
                // block itself. Its unused grid and sign loads are dead code
                // that ptxas drops.
                //
                // `split` means the format has one scale per 16 elements
                // rather than per 32. A k step is two mma operations split on
                // that boundary, so a split format keeps one scale per half.
                let split = $split;
                let per_col = if split { halves } else { 1 };
                let mut w_scales = Vec::with_capacity(rn as usize * 2 * per_col as usize);
                for out_col in &out_cols {
                    for dj in 0..2 {
                        let off = self.const_index(&kb, dj)?;
                        let col = self.addi(&kb, *out_col, off)?;
                        for h in 0..per_col {
                            // Lane 0 gives the low scale and lane 2 the high
                            // one, matching the decode's `l < 2` select.
                            let l = self.const_index(&kb, 2 * h)?;
                            let fmt_lane = self.addi(&kb, four_ib, l)?;
                            let geom = self.$lane_fn(&kb, fmt_lane)?;
                            let at = self.raw_block_at(&kb, col, blk, blk_bytes)?;
                            let dec = self.$block_fn(&kb, &geom, qb, d, &tables, &at)?;
                            w_scales.push(dec.$scale);
                        }
                    }
                }

                let mut next = Vec::with_capacity(lanes);
                for r in 0..rm as usize {
                    let sa = self.push(&kb, memref::load(asc.mem, &[a_rows[r], b], self.loc))?;
                    for c in 0..rn as usize {
                        // One accumulator per half when the scale changes
                        // there, one for the pair otherwise. Chaining the two
                        // mma operations is only sound when they share a
                        // scale.
                        let mut sums = Vec::with_capacity(per_col as usize);
                        if split {
                            for h in 0..halves as usize {
                                sums.push(self.mma_sync(
                                    &kb,
                                    a_frags[r * halves as usize + h],
                                    w_frags[c * halves as usize + h],
                                    empty,
                                    shape,
                                    acc_t,
                                )?);
                            }
                        } else {
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
                            sums.push(sum);
                        }
                        for dj in 0..2 {
                            let slot = (r * rn as usize + c) * 2 + dj as usize;
                            let mut acc = accs[slot];
                            for (h, sum) in sums.iter().enumerate() {
                                let sw = w_scales[(c * 2 + dj as usize) * per_col as usize + h];
                                let scale = self.push(&kb, arith::mulf(sa, sw, self.loc))?;
                                let raw = self.vec_extract(&kb, *sum, &[0, dj], i32_t)?;
                                let as_f = self.small_int_to_f32(&kb, raw)?;
                                acc = self.elem_mac(&kb, f32_t, as_f, scale, acc)?;
                            }
                            next.push(acc);
                        }
                    }
                }
                // The next turn of the loop overwrites what this one staged.
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
                            block.append_operation(memref::store(
                                value,
                                out.mem,
                                &[*row, col],
                                self.loc,
                            ));
                        }
                    }
                }
                self.barrier(block)?;
                Ok(())
            }
        }
    };
}

signed_qmma!(
    iq2xxs_qmma_staged_into,
    "iq2xxs_qmma_t",
    iq2xxs_lane,
    iq2xxs_block,
    IQ2XXS_BLOCK_BYTES,
    IQ2XXS_LANE,
    db,
    false,
    grid_v
);
signed_qmma!(
    iq2s_qmma_staged_into,
    "iq2s_qmma_t",
    iq2s_lane,
    iq2s_block,
    IQ2S_BLOCK_BYTES,
    IQ2S_LANE,
    dl,
    true,
    grid_v
);
signed_qmma!(
    iq2xs_qmma_staged_into,
    "iq2xs_qmma_t",
    iq2xs_lane,
    iq2xs_block,
    IQ2XS_BLOCK_BYTES,
    IQ2XS_LANE,
    dl,
    true,
    grid_v
);
// IQ3's grid entry is four bytes, so a lane joins two of them. IQ3_XXS has
// one scale per 32 elements and IQ3_S one per 64. Either is constant across a
// k step, which is all the unsplit arm needs.
signed_qmma!(
    iq3xxs_qmma_staged_into,
    "iq3xxs_qmma_t",
    iq3xxs_lane,
    iq3xxs_block,
    IQ3XXS_BLOCK_BYTES,
    IQ3XXS_LANE,
    db,
    false,
    g1_v,
    g2_v
);
signed_qmma!(
    iq3s_qmma_staged_into,
    "iq3s_qmma_t",
    iq3s_lane,
    iq3s_block,
    IQ3S_BLOCK_BYTES,
    IQ3S_LANE,
    db,
    false,
    g1_v,
    g2_v
);
