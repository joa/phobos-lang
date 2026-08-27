// IQ2_XXS, IQ2_S and IQ2_XS against an int8 activation, contracted in
// `dp4a`. All three decode to `scale * mag_i * sign_i` with both planes
// already i8; they differ only in a block stride, a scale field name and
// which lane and block helpers they call, so a macro generates them.

use super::*;
use super::iq2s::{IQ2S_BLOCK_BYTES, IQ2S_LANE};
use super::iq2xs::{IQ2XS_BLOCK_BYTES, IQ2XS_LANE};
use super::iq2xxs::{IQ2XXS_BLOCK_BYTES, IQ2XXS_LANE};

macro_rules! signed_table_i8 {
    ($fn:ident, $label:literal, $lane_fn:ident, $block_fn:ident,
     $bytes:ident, $lane_w:ident, $scale:ident) => {
        impl<'c> Codegen<'c> {
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
        ) -> Result<MemVal<'c>> {
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
                bail!(concat!($label, " contracts int8 activations against the format's raw bytes"));
            }
            if asc.elem != self.f32_t {
                bail!(concat!($label, "'s activation scale must be f32"));
            }
            if d.elem != self.f16_t {
                bail!(concat!($label, "'s block scale must be f16"));
            }
            if !self.has_dp4a() {
                bail!(concat!($label, " needs dp4a; this target has none"));
            }
            if self.cta_threads % WARP != 0 {
                bail!(concat!($label, " needs a CTA that is a whole number of warps"));
            }
            let cols = qb.shape[0];
            if cols == DYN {
                bail!(concat!($label, " needs a static output width"));
            }
            // Derived, not chosen: a fixed count against a mismatched tile
            // leaves part of the CTA idle.
            let per_warp = cols * WARP / self.cta_threads;
            if per_warp < 1 || cols % per_warp != 0 {
                bail!(concat!($label, " needs a tile that fills the CTA a whole number of times"));
            }
            if aq.shape[1] != DYN && aq.shape[1] % 256 != 0 {
                bail!(concat!($label, " needs a whole number of 256-element blocks"));
            }
            self.check_shapes(&[cols], &[d.shape[0]], concat!($label, " d rows"))?;

            let out = self.alloc_tile_shaped(block, self.f32_t, &[1, cols])?;
            let (i8_t, i32_t, f32_t) = (self.i8_t, self.i32_t, self.f32_t);

            // Both tables into shared memory; see `stage_table`.
            let staged_grid = self.stage_table(block, grid)?;
            let staged_signs = self.stage_table(block, signs)?;
            let grid = staged_grid.as_ref().unwrap_or(grid);
            let signs = staged_signs.as_ref().unwrap_or(signs);

            let warp_w = self.const_index(block, WARP)?;
            // One work item a column group.
            let total = self.const_index(block, cols / per_warp * WARP)?;
            let tid = self.thread_id(block)?;
            let bdim = self.block_dim(block)?;

            let body = Block::new(&[(self.index_t, self.loc)]);
            let li = detach(body.argument(0)?.into());
            let slot = self.divui(&body, li, warp_w)?;
            let lane = self.remui(&body, li, warp_w)?;
            let width = self.const_index(&body, per_warp)?;
            let base_col = self.muli(&body, slot, width)?;
            let mut js = Vec::with_capacity(per_warp as usize);
            for c in 0..per_warp {
                let off = self.const_index(&body, c)?;
                js.push(self.addi(&body, base_col, off)?);
            }

            let geom = self.$lane_fn(&body, lane)?;
            let zero_idx = self.const_index(&body, 0)?;

            let step = self.const_index(&body, 256)?;
            let kd = if aq.shape[1] == DYN {
                let one = self.const_index(&body, 1)?;
                self.push(&body, memref::dim(aq.mem, one, self.loc))?
            } else {
                self.const_index(&body, aq.shape[1])?
            };
            let init = self.zero_scalar(&body, f32_t)?;
            let blk_bytes = self.const_index(&body, $bytes)?;

            let mut kb_args = vec![(self.index_t, self.loc)];
            kb_args.extend(std::iter::repeat_n((f32_t, self.loc), per_warp as usize));
            let kb = Block::new(&kb_args);
            let kbase = detach(kb.argument(0)?.into());
            let mut carries = Vec::with_capacity(per_warp as usize);
            for c in 0..per_warp as usize {
                carries.push(detach(kb.argument(c + 1)?.into()));
            }
            let blk = self.divui(&kb, kbase, step)?;
            let blk_off = self.muli(&kb, blk, blk_bytes)?;

            // One gather chain a column, all issued before any is consumed.
            let tables = QTables { grid, signs };
            let mut decs = Vec::with_capacity(per_warp as usize);
            for &j in &js {
                let at = BlockAt { j, blk, off: blk_off };
                decs.push(self.$block_fn(&kb, &geom, qb, d, &tables, &at)?);
            }

            // `k_lane_off` is a multiple of eight, so this load is aligned and
            // the eight elements share one activation scale.
            let k_off = self.addi(&kb, kbase, geom.k_lane_off)?;
            let byte_vec_t = Type::vector(&[$lane_w as u64], i8_t);
            let aq_v = self.vec_load_al(&kb, aq.mem, &[zero_idx, k_off], byte_vec_t, 8)?;
            let group = self.divui(&kb, k_off, self.const_index(&kb, ACT_SCALE_BLOCK)?)?;
            let sa = self.push(&kb, memref::load(asc.mem, &[zero_idx, group], self.loc))?;

            let pair_t = Type::vector(&[2, $lane_w as u64 / 2], i8_t);
            let quad_t = Type::vector(&[$lane_w as u64 / 2], i8_t);
            let aq_pair = self.vec_shape_cast(&kb, aq_v, pair_t)?;

            // Signs arrive as a 0/-1 mask, so applying them is bitwise: split
            // the magnitudes into signed halves and subtract one dot from the
            // other. Multiplying by +/-1 instead has no hardware form, and
            // LLVM scalarizes it into a byte-at-a-time unpack.
            let all_ones = self.push(
                &kb,
                arith::constant(self.ctx, IntegerAttribute::new(i8_t, -1).into(), self.loc),
            )?;
            let ones_v = self.vec_broadcast(&kb, all_ones, byte_vec_t)?;

            let mut partials = Vec::with_capacity(per_warp as usize);
            for (dec, &carry) in decs.iter().zip(&carries) {
                let keep = self.push(&kb, arith::xori(dec.signs_v, ones_v, self.loc))?;
                let pos = self.push(&kb, arith::andi(dec.grid_v, keep, self.loc))?;
                let neg = self.push(&kb, arith::andi(dec.grid_v, dec.signs_v, self.loc))?;
                let pos_pair = self.vec_shape_cast(&kb, pos, pair_t)?;
                let neg_pair = self.vec_shape_cast(&kb, neg, pair_t)?;
                let mut dot_p = self.zero_scalar(&kb, i32_t)?;
                let mut dot_n = self.zero_scalar(&kb, i32_t)?;
                for half in 0..2i64 {
                    let q = self.vec_extract(&kb, aq_pair, &[half], quad_t)?;
                    let p = self.vec_extract(&kb, pos_pair, &[half], quad_t)?;
                    let n = self.vec_extract(&kb, neg_pair, &[half], quad_t)?;
                    dot_p = self.dot4_accumulate(&kb, p, q, dot_p)?;
                    dot_n = self.dot4_accumulate(&kb, n, q, dot_n)?;
                }
                let dot = self.push(&kb, arith::subi(dot_p, dot_n, self.loc))?;
                let dot_f = self.numeric_cast(&kb, dot, f32_t)?;
                let scale = self.push(&kb, arith::mulf(dec.$scale, sa, self.loc))?;
                let contribution = self.push(&kb, arith::mulf(scale, dot_f, self.loc))?;
                partials.push(self.push(&kb, arith::addf(carry, contribution, self.loc))?);
            }
            kb.append_operation(scf::r#yield(&partials, self.loc));

            let mut loop_operands = vec![zero_idx, kd, step];
            loop_operands.extend(std::iter::repeat_n(init, per_warp as usize));
            let kr = Region::new();
            kr.append_block(kb);
            let loop_op = body.append_operation(
                OperationBuilder::new("scf.for", self.loc)
                    .add_operands(&loop_operands)
                    .add_results(&vec![f32_t; per_warp as usize])
                    .add_regions([kr])
                    .build()?,
            );

            let mut accs = Vec::with_capacity(per_warp as usize);
            for slot in 0..per_warp as usize {
                let mut acc: Value<'c, 'c> = detach(loop_op.result(slot)?.into());
                let mut mask = WARP / 2;
                while mask >= 1 {
                    let other = self.shfl_xor_f32(&body, acc, mask)?;
                    acc = self.push(&body, arith::addf(acc, other, self.loc))?;
                    mask /= 2;
                }
                accs.push(acc);
            }

            let is_lead = self.push(
                &body,
                arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, zero_idx, self.loc),
            )?;
            let store = Block::new(&[]);
            for (acc, &j) in accs.iter().zip(&js) {
                store.append_operation(memref::store(*acc, out.mem, &[zero_idx, j], self.loc));
            }
            store.append_operation(scf::r#yield(&[], self.loc));
            let sr = Region::new();
            sr.append_block(store);
            body.append_operation(scf::r#if(is_lead, &[], sr, Region::new(), self.loc));
            body.append_operation(scf::r#yield(&[], self.loc));

            let region = Region::new();
            region.append_block(body);
            block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));
            self.barrier(block)?;
            Ok(out)
        }
    }
    };
}

signed_table_i8!(
    tile_iq2xxs_qdot_i8_t, "iq2xxs_qdot_i8_t", iq2xxs_lane, iq2xxs_block,
    IQ2XXS_BLOCK_BYTES, IQ2XXS_LANE, db
);
signed_table_i8!(
    tile_iq2s_qdot_i8_t, "iq2s_qdot_i8_t", iq2s_lane, iq2s_block,
    IQ2S_BLOCK_BYTES, IQ2S_LANE, dl
);
signed_table_i8!(
    tile_iq2xs_qdot_i8_t, "iq2xs_qdot_i8_t", iq2xs_lane, iq2xs_block,
    IQ2XS_BLOCK_BYTES, IQ2XS_LANE, dl
);
