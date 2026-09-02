// `<fmt>_qdot_i8_t`: a raw format's single-row contraction against an
// int8 activation in `dp4a`. Four lanes decode a column, a quarter of each
// 256-element block apiece, and a warp covers eight columns; the block
// bytes ride a two-deep register pipeline and the activations come from
// L1 at decode time. The decode itself is `qgemm_fmt.rs`'s.

use super::qgemm_fmt::{IQ1_LUT4_MINUS, IQ1_LUT4_PLUS};
use super::*;

/// Octets of a block a lane decodes.
const LANE_OCTETS: usize = 8;
const LANES_PER_COL: i64 = 4;
const COLS_PER_WARP: i64 = WARP / LANES_PER_COL;
/// Blocks the register pipeline holds ahead of the decode.
const DEPTH: usize = 2;
/// Blocks past the pipeline that are prefetched into L2.
const PREFETCH_BLOCKS: i64 = 1;

/// One load of a lane's quarter: byte offset into the block, and width.
struct Piece<'c> {
    off: Value<'c, 'c>,
    width: i64,
}

impl<'c> Codegen<'c> {
    /// out[0, j] = sum_k a[0, k] * w[j, k] with `w` decoded from `fmt`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn tile_qdot_i8_reg_t(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        aq: &MemVal<'c>,
        asc: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        tables: &[MemVal<'c>],
    ) -> Result<MemVal<'c>> {
        let label = fmt.qdot_i8_intrinsic();
        for (v, what) in [(aq, "a"), (asc, "a scales"), (qb, "qb"), (d, "d")]
            .into_iter()
            .chain(tables.iter().map(|t| (t, "table")))
        {
            if v.shape.len() != 2 {
                bail!("{label} {what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{label} {what} must be a fully in-bounds slice");
            }
        }
        if aq.elem != self.i8_t || qb.elem != self.i8_t || tables.iter().any(|t| t.elem != self.i8_t)
        {
            bail!("{label} contracts int8 activations against the format's raw bytes");
        }
        if asc.elem != self.f32_t {
            bail!("{label}'s activation scale must be f32");
        }
        if d.elem != self.f16_t {
            bail!("{label}'s block scale must be f16");
        }
        if !self.has_dp4a() {
            bail!("{label} needs dp4a; this target has none");
        }
        if self.cta_threads % WARP != 0 {
            bail!("{label} needs a CTA that is a whole number of warps");
        }
        let cols = qb.shape[0];
        if cols == DYN {
            bail!("{label} needs a static output width");
        }
        // A tile narrower than the CTA leaves its trailing warps idle.
        if cols % COLS_PER_WARP != 0 {
            bail!("{label} needs a tile of whole groups of {COLS_PER_WARP} columns");
        }
        if aq.shape[1] != DYN && aq.shape[1] % 256 != 0 {
            bail!("{label} needs a whole number of 256-element blocks");
        }
        self.check_shapes(&[cols], &[d.shape[0]], "qdot_i8 d rows")?;
        let table_bytes = fmt.qdot_tables();
        if tables.len() != table_bytes.len()
            || tables.iter().zip(table_bytes).any(|(t, &want)| t.shape[1] != want)
        {
            bail!("{label} wants tables of {table_bytes:?} bytes");
        }

        let out = self.alloc_tile_shaped(block, self.f32_t, &[1, cols])?;
        let (i8_t, f32_t) = (self.i8_t, self.f32_t);
        let mut tabs = Vec::with_capacity(tables.len());
        for (table, &bytes) in tables.iter().zip(table_bytes) {
            let tile = self.alloc_tile_shaped(block, i8_t, &[1, bytes])?;
            self.stage_bytes(block, table, &tile, bytes)?;
            tabs.push(tile);
        }
        self.barrier(block)?;

        let warp_w = self.const_index(block, WARP)?;
        let total = self.const_index(block, cols / COLS_PER_WARP * WARP)?;
        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;

        // One turn per warp per group of eight columns.
        let body = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body.argument(0)?.into());
        let slot = self.divui(&body, li, warp_w)?;
        let lane = self.remui(&body, li, warp_w)?;
        let lanes_per_col = self.const_index(&body, LANES_PER_COL)?;
        let cols_per_warp = self.const_index(&body, COLS_PER_WARP)?;
        let base_col = self.muli(&body, slot, cols_per_warp)?;
        let in_warp_col = self.divui(&body, lane, lanes_per_col)?;
        let j = self.addi(&body, base_col, in_warp_col)?;
        let quarter = self.remui(&body, lane, lanes_per_col)?;
        let pieces = self.qr_pieces(&body, fmt, quarter)?;
        let sixty_four = self.const_index(&body, 64)?;
        let k_lane_off = self.muli(&body, quarter, sixty_four)?;
        let sixteen = self.const_index(&body, 16)?;
        let pf_lane_off = self.muli(&body, quarter, sixteen)?;

        let step = self.const_index(&body, 256)?;
        let zero_idx = self.const_index(&body, 0)?;
        let kd = if aq.shape[1] == DYN {
            let one = self.const_index(&body, 1)?;
            self.push(&body, memref::dim(aq.mem, one, self.loc))?
        } else {
            self.const_index(&body, aq.shape[1])?
        };
        let init = self.zero_scalar(&body, f32_t)?;
        let blk_bytes = self.const_index(&body, fmt.block_bytes())?;
        let nb = self.divui(&body, kd, step)?;
        let one_k = self.const_index(&body, 1)?;
        let last_blk = self.subi(&body, nb, one_k)?;

        // Blocks 0..DEPTH are fetched before the loop; each turn decodes the
        // oldest and fetches the block DEPTH ahead. The loop stops DEPTH
        // blocks short, and the blocks still in the pipeline are decoded
        // after it, each gated on existing (the fetch clamps to the last).
        let mut stages = Vec::with_capacity(DEPTH);
        for stage in 0..DEPTH {
            let blk = self.const_index(&body, stage as i64)?;
            let blk = self.push(&body, arith::minui(blk, last_blk, self.loc))?;
            stages.push(self.qr_fetch(&body, &pieces, j, qb, d, blk, blk_bytes)?);
        }
        let per_stage = stages[0].len();

        let mut kb_args = vec![(self.index_t, self.loc), (f32_t, self.loc)];
        for stage in &stages {
            kb_args.extend(stage.iter().map(|v| (v.r#type(), self.loc)));
        }
        let kb = Block::new(&kb_args);
        let kbase = detach(kb.argument(0)?.into());
        let carry = detach(kb.argument(1)?.into());
        let stage_regs: Vec<Vec<Value<'c, 'c>>> = (0..DEPTH)
            .map(|stage| {
                (0..per_stage)
                    .map(|i| Ok(detach(kb.argument(2 + stage * per_stage + i)?.into())))
                    .collect::<Result<_>>()
            })
            .collect::<Result<_>>()?;

        // The block DEPTH ahead, unclamped so the addresses stay affine.
        let blk = self.divui(&kb, kbase, step)?;
        let depth_k = self.const_index(&kb, DEPTH as i64)?;
        let ahead_blk = self.addi(&kb, blk, depth_k)?;
        // L2 prefetch of the lane's column, PREFETCH_BLOCKS further on.
        let ahead = self.const_index(&kb, PREFETCH_BLOCKS)?;
        let pf_blk = self.addi(&kb, ahead_blk, ahead)?;
        let pf_blk = self.push(&kb, arith::minui(pf_blk, last_blk, self.loc))?;
        let pf = self.raw_block_at(&kb, j, pf_blk, blk_bytes)?;
        let pf_at = self.addi(&kb, pf.off, pf_lane_off)?;
        self.prefetch_read(&kb, qb, &[pf.j, pf_at])?;

        let k_off = self.addi(&kb, kbase, k_lane_off)?;
        let acc = self.qr_block(&kb, fmt, &tabs, &stage_regs[0], aq, asc, k_off, carry)?;
        let mut yields = vec![acc];
        for stage in &stage_regs[1..] {
            yields.extend(stage.iter().copied());
        }
        yields.extend(self.qr_fetch(&kb, &pieces, j, qb, d, ahead_blk, blk_bytes)?);
        kb.append_operation(scf::r#yield(&yields, self.loc));

        // `kd - DEPTH * 256`, clamped at zero for a row shorter than that.
        let span_body = self.const_index(&body, DEPTH as i64 * 256)?;
        let last_k = self.subi(&body, kd, span_body)?;
        let last_k = self.push(&body, arith::maxsi(last_k, zero_idx, self.loc))?;
        let mut loop_operands = vec![zero_idx, last_k, step, init];
        for stage in &stages {
            loop_operands.extend(stage.iter().copied());
        }
        let mut results = vec![f32_t];
        for stage in &stages {
            results.extend(stage.iter().map(|v| v.r#type()));
        }
        let kr = Region::new();
        kr.append_block(kb);
        let loop_op = body.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&loop_operands)
                .add_results(&results)
                .add_regions([kr])
                .build()?,
        );

        // Stage `s` holds block `nb - DEPTH + s`, present when `nb + s >= DEPTH`.
        let mut acc: Value<'c, 'c> = detach(loop_op.result(0)?.into());
        for stage in 0..DEPTH {
            let regs: Vec<Value<'c, 'c>> = (0..per_stage)
                .map(|i| Ok(detach(loop_op.result(1 + stage * per_stage + i)?.into())))
                .collect::<Result<_>>()?;
            let stage_k = self.const_index(&body, stage as i64)?;
            let reach = self.addi(&body, nb, stage_k)?;
            let depth_body = self.const_index(&body, DEPTH as i64)?;
            let exists = self.push(
                &body,
                arith::cmpi(self.ctx, arith::CmpiPredicate::Uge, reach, depth_body, self.loc),
            )?;
            let blk_s = self.subi(&body, reach, depth_body)?;
            let blk_s = self.push(&body, arith::maxsi(blk_s, zero_idx, self.loc))?;
            let kbase_s = self.muli(&body, blk_s, step)?;
            let k_off = self.addi(&body, kbase_s, k_lane_off)?;
            let with = self.qr_block(&body, fmt, &tabs, &regs, aq, asc, k_off, acc)?;
            acc = self.select(&body, exists, with, acc)?;
        }

        // Sum the column's four quarters.
        for mask in [1, 2] {
            let other = self.shfl_xor_f32(&body, acc, mask)?;
            acc = self.push(&body, arith::addf(acc, other, self.loc))?;
        }
        let is_lead = self.push(
            &body,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, quarter, zero_idx, self.loc),
        )?;
        let store = Block::new(&[]);
        store.append_operation(memref::store(acc, out.mem, &[zero_idx, j], self.loc));
        store.append_operation(scf::r#yield(&[], self.loc));
        let sr = Region::new();
        sr.append_block(store);
        body.append_operation(scf::r#if(is_lead, &[], sr, Region::new(), self.loc));
        body.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));
        self.barrier(block)?;
        for t in &tabs {
            self.release(t);
        }
        Ok(out)
    }

    /// The loads of a lane's quarter of a block, by format. Layouts are in
    /// `qgemm_fmt.rs`.
    fn qr_pieces(&mut self, body: &Block<'c>, fmt: QgFormat, quarter: Value<'c, 'c>) -> Result<Vec<Piece<'c>>> {
        // `base + mul * quarter`.
        let at = |cg: &mut Self, base: i64, mul: i64| -> Result<Value<'c, 'c>> {
            let mul = cg.const_index(body, mul)?;
            let off = cg.muli(body, quarter, mul)?;
            let base = cg.const_index(body, base)?;
            cg.addi(body, off, base)
        };
        let piece = |off, width| Piece { off, width };
        Ok(match fmt {
            QgFormat::Iq1s => vec![piece(at(self, 0, 8)?, 8), piece(at(self, 32, 4)?, 4)],
            QgFormat::Iq1m => vec![
                piece(at(self, 0, 8)?, 8),
                piece(at(self, 32, 4)?, 4),
                piece(at(self, 48, 2)?, 2),
            ],
            QgFormat::Iq2xxs => vec![piece(at(self, 0, 16)?, 16)],
            // The 72-byte block is only eight-aligned.
            QgFormat::Iq2xs => vec![
                piece(at(self, 0, 16)?, 8),
                piece(at(self, 8, 16)?, 8),
                piece(at(self, 64, 2)?, 2),
            ],
            QgFormat::Iq2s => vec![
                piece(at(self, 0, 8)?, 8),
                piece(at(self, 32, 8)?, 8),
                piece(at(self, 64, 2)?, 2),
                piece(at(self, 72, 2)?, 2),
            ],
            QgFormat::Iq3xxs => vec![piece(at(self, 0, 16)?, 16), piece(at(self, 64, 8)?, 8)],
            // The 108-byte block is only four-aligned.
            QgFormat::Iq3s => vec![
                piece(at(self, 0, 16)?, 4),
                piece(at(self, 4, 16)?, 4),
                piece(at(self, 8, 16)?, 4),
                piece(at(self, 12, 16)?, 4),
                piece(at(self, 64, 2)?, 2),
                piece(at(self, 72, 8)?, 4),
                piece(at(self, 76, 8)?, 4),
                piece(at(self, 104, 1)?, 1),
            ],
        })
    }

    /// The registers of block `blk`: the lane's pieces as words, then the
    /// column's `d`.
    #[allow(clippy::too_many_arguments)]
    fn qr_fetch(
        &mut self,
        block: &Block<'c>,
        pieces: &[Piece<'c>],
        j: Value<'c, 'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        blk: Value<'c, 'c>,
        blk_bytes: Value<'c, 'c>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let i16_t: Type<'c> = IntegerType::new(self.ctx, 16).into();
        let i32_t = self.i32_t;
        let mut regs = Vec::with_capacity(12);
        let at = self.raw_block_at(block, j, blk, blk_bytes)?;
        let (j, blk_off, d_row, d_col) = (at.j, at.off, at.d_row, at.d_col);
        for piece in pieces {
            let at = self.addi(block, blk_off, piece.off)?;
            match piece.width {
                1 => {
                    let b = self.push(block, memref::load(qb.mem, &[j, at], self.loc))?;
                    regs.push(self.extui(block, b, i32_t)?);
                }
                2 => {
                    let v = self.vec_load_al(block, qb.mem, &[j, at], Type::vector(&[2], self.i8_t), 2)?;
                    let v = self.vec_bitcast(block, v, Type::vector(&[1], i16_t))?;
                    let v = self.vec_extract(block, v, &[0], i16_t)?;
                    regs.push(self.extui(block, v, i32_t)?);
                }
                width => {
                    let words = width / 4;
                    let v = self.vec_load_al(block, qb.mem, &[j, at], Type::vector(&[width as u64], self.i8_t), width)?;
                    let v = self.vec_bitcast(block, v, Type::vector(&[words as u64], i32_t))?;
                    for w in 0..words {
                        regs.push(self.vec_extract(block, v, &[w], i32_t)?);
                    }
                }
            }
        }
        regs.push(self.push(block, memref::load(d.mem, &[d_row, d_col], self.loc))?);
        Ok(regs)
    }

    /// The lane's 64 activations from `k_off` as `dp4a` words, and their
    /// two scales.
    fn qr_act(
        &mut self,
        block: &Block<'c>,
        aq: &MemVal<'c>,
        asc: &MemVal<'c>,
        k_off: Value<'c, 'c>,
    ) -> Result<(Vec<Value<'c, 'c>>, [Value<'c, 'c>; 2])> {
        let zero_row = self.const_index(block, 0)?;
        let sixteen = self.const_index(block, 16)?;
        let bytes16 = Type::vector(&[16], self.i8_t);
        let quad_t = Type::vector(&[4], self.i8_t);
        let quads_t = Type::vector(&[4, 4], self.i8_t);
        let mut act = Vec::with_capacity(16);
        let mut at = k_off;
        for _ in 0..4 {
            let v = self.vec_load_al(block, aq.mem, &[zero_row, at], bytes16, 16)?;
            let v = self.vec_shape_cast(block, v, quads_t)?;
            for w in 0..4 {
                act.push(self.vec_extract(block, v, &[w], quad_t)?);
            }
            at = self.addi(block, at, sixteen)?;
        }
        let group = self.divui(block, k_off, self.const_index(block, ACT_SCALE_BLOCK)?)?;
        let scales = self.vec_load_al(block, asc.mem, &[zero_row, group], Type::vector(&[2], self.f32_t), 8)?;
        let sa = [
            self.vec_extract(block, scales, &[0], self.f32_t)?,
            self.vec_extract(block, scales, &[1], self.f32_t)?,
        ];
        Ok((act, sa))
    }

    /// `carry` plus the block in `regs` (as [`Self::qr_fetch`] laid it out)
    /// against the activations from `k_off`.
    #[allow(clippy::too_many_arguments)]
    fn qr_block(
        &mut self,
        kb: &Block<'c>,
        fmt: QgFormat,
        tabs: &[MemVal<'c>],
        regs: &[Value<'c, 'c>],
        aq: &MemVal<'c>,
        asc: &MemVal<'c>,
        k_off: Value<'c, 'c>,
        carry: Value<'c, 'c>,
    ) -> Result<Value<'c, 'c>> {
        let (i8_t, i32_t, f32_t) = (self.i8_t, self.i32_t, self.f32_t);
        let words = &regs[..regs.len() - 1];
        let dv = regs[regs.len() - 1];
        let (act, sa) = self.qr_act(kb, aq, asc, k_off)?;

        let quad_t = Type::vector(&[4], i8_t);
        let one_i32 = Type::vector(&[1], i32_t);
        let as_bytes = |cg: &mut Self, w: Value<'c, 'c>| -> Result<Value<'c, 'c>> {
            let v = cg.vec_broadcast(kb, w, one_i32)?;
            cg.vec_bitcast(kb, v, quad_t)
        };

        // A run's weight: d, the activation scale, the format's factor.
        let dv = self.numeric_cast(kb, dv, f32_t)?;
        let factor = fmt.scale_factor();
        let mut run_weight = Vec::with_capacity(2);
        for &s in &sa {
            let s = if factor == 1.0 {
                s
            } else {
                let f = self.const_f32(kb, factor)?;
                self.push(kb, arith::mulf(s, f, self.loc))?
            };
            run_weight.push(self.push(kb, arith::mulf(dv, s, self.loc))?);
        }

        let octets = self.qr_octets(kb, fmt, tabs, words)?;
        let odds = self.qr_odds(kb, fmt, words)?;
        let run_octets = LANE_OCTETS / odds.len();
        let exact = fmt.max_run_dot(run_octets as i64) < (1 << 22);
        let zero_i = self.zero_scalar(kb, i32_t)?;
        let mut acc = carry;
        for (run, &odd) in odds.iter().enumerate() {
            let mut dot = zero_i;
            for o in run * run_octets..(run + 1) * run_octets {
                let (w0, w1) = octets[o];
                let w0 = as_bytes(self, w0)?;
                let w1 = as_bytes(self, w1)?;
                dot = self.dot4_accumulate(kb, w0, act[2 * o], dot)?;
                dot = self.dot4_accumulate(kb, w1, act[2 * o + 1], dot)?;
            }
            let dot = self.push(kb, arith::muli(dot, odd, self.loc))?;
            let dot_f = if exact {
                self.small_int_to_f32(kb, dot)?
            } else {
                self.push(kb, arith::sitofp(dot, f32_t, self.loc))?
            };
            let weight = run_weight[run * run_octets * 8 / 32];
            acc = self.elem_mac(kb, f32_t, dot_f, weight, acc)?;
        }
        Ok(acc)
    }

    /// The lane's eight octets as `dp4a` word pairs, from the piece words.
    fn qr_octets(
        &mut self,
        kb: &Block<'c>,
        fmt: QgFormat,
        tabs: &[MemVal<'c>],
        w: &[Value<'c, 'c>],
    ) -> Result<Vec<(Value<'c, 'c>, Value<'c, 'c>)>> {
        let byte = |cg: &mut Self, word: Value<'c, 'c>, k: usize| cg.qg_bits(kb, word, 8 * k as i64, 0xFF);
        let bits = |cg: &mut Self, word: Value<'c, 'c>, shift: usize, mask: i64| cg.qg_bits(kb, word, shift as i64, mask);
        let mut out = Vec::with_capacity(LANE_OCTETS);
        match fmt {
            QgFormat::Iq1s => {
                let (qs, qh) = (&w[0..2], w[2]);
                let luts = [self.qr_lut4(kb, qh, 15)?, self.qr_lut4(kb, qh, 31)?];
                for o in 0..LANE_OCTETS {
                    let (s, i) = (o / 4, o % 4);
                    let lo = byte(self, qs[s], i)?;
                    let hi = bits(self, qh, 16 * s + 3 * i, 7)?;
                    let idx = self.qg_join(kb, lo, hi, 8)?;
                    out.push(self.qg_ternary4(kb, &tabs[0], idx, luts[s])?);
                }
            }
            QgFormat::Iq1m => {
                let (qs, qh) = (&w[0..2], w[2]);
                for o in 0..LANE_OCTETS {
                    let lo = byte(self, qs[o / 4], o % 4)?;
                    let hi = bits(self, qh, 4 * o, 7)?;
                    let idx = self.qg_join(kb, lo, hi, 8)?;
                    let lut = self.qr_lut4(kb, qh, 4 * o + 3)?;
                    out.push(self.qg_ternary4(kb, &tabs[0], idx, lut)?);
                }
            }
            QgFormat::Iq2xxs => {
                for o in 0..LANE_OCTETS {
                    let (s, i) = (o / 4, o % 4);
                    let idx = byte(self, w[2 * s], i)?;
                    let sidx = bits(self, w[2 * s + 1], 7 * i, 127)?;
                    out.push(self.qr_signed(kb, tabs, idx, sidx, true)?);
                }
            }
            QgFormat::Iq2xs => {
                for o in 0..LANE_OCTETS {
                    let u = bits(self, w[o / 2], 16 * (o % 2), 0xFFFF)?;
                    let idx = self.qg_bits(kb, u, 0, 511)?;
                    let sidx = self.qg_bits(kb, u, 9, 127)?;
                    out.push(self.qr_signed(kb, tabs, idx, sidx, true)?);
                }
            }
            QgFormat::Iq2s => {
                let (qs, signs, qh) = (&w[0..2], &w[2..4], w[4]);
                for o in 0..LANE_OCTETS {
                    let lo = byte(self, qs[o / 4], o % 4)?;
                    let hi = bits(self, qh, 2 * o, 3)?;
                    let idx = self.qg_join(kb, lo, hi, 8)?;
                    let sidx = byte(self, signs[o / 4], o % 4)?;
                    out.push(self.qr_signed(kb, tabs, idx, sidx, false)?);
                }
            }
            QgFormat::Iq3xxs => {
                let (qs, aux) = (&w[0..4], &w[4..6]);
                for o in 0..LANE_OCTETS {
                    let (s, i) = (o / 4, o % 4);
                    let b1 = byte(self, qs[o / 2], 2 * (o % 2))?;
                    let b2 = byte(self, qs[o / 2], 2 * (o % 2) + 1)?;
                    let sidx = bits(self, aux[s], 7 * i, 127)?;
                    out.push(self.qr_signed_pair(kb, tabs, b1, b2, sidx, true)?);
                }
            }
            QgFormat::Iq3s => {
                let (qs, qh, signs) = (&w[0..4], w[4], &w[5..7]);
                for o in 0..LANE_OCTETS {
                    let b1 = byte(self, qs[o / 2], 2 * (o % 2))?;
                    let b2 = byte(self, qs[o / 2], 2 * (o % 2) + 1)?;
                    let h1 = bits(self, qh, 2 * o, 1)?;
                    let h2 = bits(self, qh, 2 * o + 1, 1)?;
                    let g1 = self.qg_join(kb, b1, h1, 8)?;
                    let g2 = self.qg_join(kb, b2, h2, 8)?;
                    let sidx = byte(self, signs[o / 4], o % 4)?;
                    out.push(self.qr_signed_pair(kb, tabs, g1, g2, sidx, false)?);
                }
            }
        }
        Ok(out)
    }

    /// `2 n + 1` for each scale run of the quarter: two of 32 or four of 16.
    fn qr_odds(&mut self, kb: &Block<'c>, fmt: QgFormat, w: &[Value<'c, 'c>]) -> Result<Vec<Value<'c, 'c>>> {
        // `2 n + 1` for the `bits`-wide `n` at bit `shift`: read one bit low.
        let odd_at = |cg: &mut Self, word: Value<'c, 'c>, shift: usize, bits: i64| -> Result<Value<'c, 'c>> {
            let even = cg.qg_bits(kb, word, shift as i64 - 1, ((1 << bits) - 1) << 1)?;
            let one = cg.const_i32(kb, 1)?;
            cg.push(kb, arith::ori(even, one, cg.loc))
        };
        let odd_of = |cg: &mut Self, word: Value<'c, 'c>, shift: usize, bits: i64| -> Result<Value<'c, 'c>> {
            let n = cg.qg_bits(kb, word, shift as i64, (1 << bits) - 1)?;
            let two = cg.const_i32(kb, 2)?;
            let one = cg.const_i32(kb, 1)?;
            let n2 = cg.push(kb, arith::muli(n, two, cg.loc))?;
            cg.push(kb, arith::addi(n2, one, cg.loc))
        };
        Ok(match fmt {
            QgFormat::Iq1s => vec![odd_at(self, w[2], 12, 3)?, odd_at(self, w[2], 28, 3)?],
            QgFormat::Iq1m => (0..4).map(|g| odd_of(self, w[3], 3 * g, 3)).collect::<Result<_>>()?,
            QgFormat::Iq2xxs => vec![odd_at(self, w[1], 28, 4)?, odd_at(self, w[3], 28, 4)?],
            QgFormat::Iq2xs => (0..4).map(|g| odd_of(self, w[4], 4 * g, 4)).collect::<Result<_>>()?,
            QgFormat::Iq2s => (0..4).map(|g| odd_of(self, w[5], 4 * g, 4)).collect::<Result<_>>()?,
            QgFormat::Iq3xxs => vec![odd_at(self, w[4], 28, 4)?, odd_at(self, w[5], 28, 4)?],
            QgFormat::Iq3s => vec![odd_of(self, w[7], 0, 4)?, odd_of(self, w[7], 4, 4)?],
        })
    }

    /// An IQ2 octet: the grid entry `idx` under the signs `sidx`, whose
    /// eighth is the parity of the rest when `parity`.
    fn qr_signed(
        &mut self,
        kb: &Block<'c>,
        tabs: &[MemVal<'c>],
        idx: Value<'c, 'c>,
        sidx: Value<'c, 'c>,
        parity: bool,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let mags = self.qg_entry(kb, &tabs[0], idx, 8)?;
        let masks = self.qr_sign_masks(kb, sidx, parity)?;
        Ok((self.qg_negate(kb, mags[0], masks[0])?, self.qg_negate(kb, mags[1], masks[1])?))
    }

    /// An IQ3 octet: two four-byte grid entries, signed alike.
    fn qr_signed_pair(
        &mut self,
        kb: &Block<'c>,
        tabs: &[MemVal<'c>],
        g1: Value<'c, 'c>,
        g2: Value<'c, 'c>,
        sidx: Value<'c, 'c>,
        parity: bool,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let m0 = self.qg_entry(kb, &tabs[0], g1, 4)?[0];
        let m1 = self.qg_entry(kb, &tabs[0], g2, 4)?[0];
        let masks = self.qr_sign_masks(kb, sidx, parity)?;
        Ok((self.qg_negate(kb, m0, masks[0])?, self.qg_negate(kb, m1, masks[1])?))
    }

    /// The two 0/-1 byte-mask words of eight sign bits. With `parity` the
    /// eighth bit is the odd parity of the seven given. `v * 0x204081`
    /// lays four copies of a nibble seven bits apart; `0x01010101` keeps
    /// one bit of each.
    fn qr_sign_masks(&mut self, kb: &Block<'c>, sidx: Value<'c, 'c>, parity: bool) -> Result<[Value<'c, 'c>; 2]> {
        let bits = if parity {
            let ones = self.push(kb, OperationBuilder::new("math.ctpop", self.loc).add_operands(&[sidx]).add_results(&[self.i32_t]).build()?)?;
            let one = self.const_i32(kb, 1)?;
            let seven = self.const_i32(kb, 7)?;
            let odd = self.push(kb, arith::andi(ones, one, self.loc))?;
            let eighth = self.push(kb, arith::shli(odd, seven, self.loc))?;
            self.push(kb, arith::ori(sidx, eighth, self.loc))?
        } else {
            sidx
        };
        let spread = self.const_i32(kb, 0x0020_4081)?;
        let pick = self.const_i32(kb, 0x0101_0101)?;
        let fill = self.const_i32(kb, 0xFF)?;
        let mut masks = Vec::with_capacity(2);
        for half in 0..2 {
            let nibble = self.qg_bits(kb, bits, 4 * half, 0xF)?;
            let laid = self.push(kb, arith::muli(nibble, spread, self.loc))?;
            let picked = self.push(kb, arith::andi(laid, pick, self.loc))?;
            masks.push(self.push(kb, arith::muli(picked, fill, self.loc))?);
        }
        Ok([masks[0], masks[1]])
    }

    /// The `8 g +- 1` byte table, minus when bit `bit` of `w` is set.
    fn qr_lut4(&self, block: &Block<'c>, w: Value<'c, 'c>, bit: usize) -> Result<Value<'c, 'c>> {
        let zero = self.const_i32(block, 0)?;
        let mask = self.const_i32(block, 1 << bit)?;
        let bit = self.push(block, arith::andi(w, mask, self.loc))?;
        let is_neg = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Ne, bit, zero, self.loc),
        )?;
        let plus = self.const_i32(block, IQ1_LUT4_PLUS)?;
        let minus = self.const_i32(block, IQ1_LUT4_MINUS)?;
        self.select(block, is_neg, minus, plus)
    }
}

impl QgFormat {
    /// The power of two in the block scale beside its odd `2 n + 1`.
    fn scale_factor(self) -> f64 {
        match self {
            QgFormat::Iq1s | QgFormat::Iq1m | QgFormat::Iq2xxs | QgFormat::Iq2xs | QgFormat::Iq2s => 0.125,
            QgFormat::Iq3xxs => 0.25,
            QgFormat::Iq3s => 1.0,
        }
    }

    /// The largest int8 magnitude the grid holds.
    fn max_magnitude(self) -> i64 {
        match self {
            QgFormat::Iq1s | QgFormat::Iq1m => 9,
            QgFormat::Iq2xxs | QgFormat::Iq2xs | QgFormat::Iq2s => 43,
            QgFormat::Iq3xxs => 62,
            QgFormat::Iq3s => 15,
        }
    }

    /// The largest a run of `octets` octets' dot product times its odd
    /// factor can be.
    fn max_run_dot(self, octets: i64) -> i64 {
        let odd_max = match self {
            QgFormat::Iq1s | QgFormat::Iq1m => 15,
            _ => 31,
        };
        octets * 8 * 127 * self.max_magnitude() * odd_max
    }
}
