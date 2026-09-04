// The `<fmt>_qgemm_t` family: a prompt projection against a raw quantized
// weight, both operands staged through shared memory. Per 128 elements of
// `k` the CTA copies its 128 activation rows in with 16-byte loads, decodes
// its 64 weight columns into an int8 tile beside them, writes the column
// scales, and contracts both with `ldmatrix` and `mma.m8n8k16`; the next
// tile's global reads are issued before the current one is contracted.
// What a format reads and how it decodes lives in `qgemm_fmt.rs`. The
// epilogue is `iq1s_qmma_t`'s in the same order, so the two agree bit for
// bit (`examples/qmma_probe`).

use super::kquant::KQ_GROUP;
use super::*;

pub(in crate::codegen) use super::qgemm_fmt::QgFormat;

/// Rows of the output tile.
pub(in crate::codegen) const QGEMM_TM: i64 = 128;

/// Columns of the output tile.
pub(in crate::codegen) const QGEMM_TN: i64 = 64;

/// Elements of `k` staged at once, half a 256-element block.
pub(in crate::codegen) const QGEMM_KT: i64 = 128;

/// Threads the kernel is written for: eight warps, four down and two
/// across, each holding a 32 x 32 patch.
pub(in crate::codegen) const QGEMM_THREADS: i64 = 256;

/// Bytes of a 16-element `k` chunk, what one `ldmatrix` row carries.
pub(super) const CHUNK_BYTES: i64 = 16;

/// Tiles of 8 x 8 a warp's patch spans, each way.
const PATCH: i64 = 4;

/// The loop-invariant geometry of one thread.
pub(super) struct Lanes<'c> {
    /// The warp's patch origin and the lane's place in a fragment.
    i0: Value<'c, 'c>,
    j0: Value<'c, 'c>,
    quad: Value<'c, 'c>,
    in_quad: Value<'c, 'c>,
    /// The `ldmatrix` row this lane addresses in each operand's tile, and
    /// the swizzle its chunk index is permuted by.
    ldm_a_row: Value<'c, 'c>,
    ldm_w_row: Value<'c, 'c>,
    ldm_xor: Value<'c, 'c>,
    /// Activation staging: the row and chunk of each of the thread's four
    /// 16-byte pieces, and the permuted column each lands at.
    a_rows: Vec<Value<'c, 'c>>,
    a_chunks: Vec<Value<'c, 'c>>,
    a_dst: Vec<Value<'c, 'c>>,
    /// Activation scale staging: the row and the first of the two scales.
    sa_row: Value<'c, 'c>,
    sa_col: Value<'c, 'c>,
    /// The weight column the thread decodes and its 32-element group within
    /// the stage, plus `j % 8`, the swizzle phase of that row.
    pub(super) j: Value<'c, 'c>,
    pub(super) g: Value<'c, 'c>,
    j_xor: Value<'c, 'c>,
    /// The row of the grouped weight the column's bytes are read from,
    /// `8 (j / 8)`, and `j % 8`. See [`RAW_GROUP`].
    pub(super) qb_row: Value<'c, 'c>,
    in_group: Value<'c, 'c>,
}

/// Where a tile's bytes sit in the weight: the 256-element block, and the
/// thread's group within it.
pub(super) struct TileAt<'c> {
    /// The byte offset of the block in the group's row, and the entry of
    /// its scale in the plane's: both `8 blk + j % 8` blocks in.
    pub(super) blk_off: Value<'c, 'c>,
    pub(super) d_col: Value<'c, 'c>,
    /// The thread's group, 0 to 7, within the block.
    pub(super) ib: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {
    /// out[i, j] = sum_k a[i, k] * w[j, k] with `w` decoded from `fmt`, both
    /// operands staged through shared memory. See the module comment.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::codegen) fn qgemm_into(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        aq: &MemVal<'c>,
        asc: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        tables: &[MemVal<'c>],
        out: &MemVal<'c>,
    ) -> Result<()> {
        let name = fmt.intrinsic();
        for (v, what) in [(aq, "a"), (asc, "a scales"), (qb, "qb"), (d, "d")]
            .into_iter()
            .chain(tables.iter().map(|t| (t, "table")))
        {
            if v.shape.len() != 2 {
                bail!("{name} {what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("{name} {what} must be a fully in-bounds slice");
            }
            if v.swizzle.is_some() {
                bail!("{name} {what} must not be swizzled");
            }
        }
        if aq.elem != self.i8_t || qb.elem != self.i8_t || tables.iter().any(|t| t.elem != self.i8_t)
        {
            bail!("{name} contracts an int8 activation against the format's raw bytes");
        }
        if asc.elem != self.f32_t {
            bail!("{name}'s activation scale must be f32");
        }
        if d.elem != self.f16_t {
            bail!("{name}'s block scale must be f16");
        }
        if out.elem != self.f32_t || out.is_masked() {
            bail!("{name} needs a fully in-bounds f32 destination");
        }
        if !self.has_int8_mma() {
            bail!("{name} needs the integer tensor cores (sm_75 or later)");
        }
        if self.cta_threads != QGEMM_THREADS {
            bail!("{name} is written for a CTA of {QGEMM_THREADS} threads");
        }
        let (md, nd) = (aq.shape[0], qb.shape[0]);
        if md != QGEMM_TM || nd != QGEMM_TN {
            bail!("{name} takes a {QGEMM_TM} x {QGEMM_TN} output tile, got {md} x {nd}");
        }
        self.check_shapes(&[md, nd], &out.shape, "qgemm destination")?;
        self.check_shapes(&[md], &[asc.shape[0]], "qgemm a scale rows")?;
        self.check_shapes(&[nd], &[d.shape[0]], "qgemm d rows")?;
        let table_bytes = fmt.tables();
        if tables.len() != table_bytes.len()
            || tables.iter().zip(table_bytes).any(|(t, &want)| t.shape[1] != want)
        {
            bail!("{name} wants tables of {table_bytes:?} bytes");
        }
        if aq.shape[1] != DYN && aq.shape[1] % 256 != 0 {
            bail!("{name} needs a whole number of 256-element blocks");
        }

        let (i8_t, f32_t) = (self.i8_t, self.f32_t);
        let scales_per_tile = QGEMM_KT / fmt.scale_run();

        // The stage: both operand tiles, the scales of each, and the tables.
        let a_st = self.alloc_tile_shaped(block, i8_t, &[QGEMM_TM, QGEMM_KT])?;
        let w_st = self.alloc_tile_shaped(block, i8_t, &[QGEMM_TN, QGEMM_KT])?;
        let sa_st = self.alloc_tile_shaped(block, f32_t, &[QGEMM_TM, QGEMM_KT / Q8_BLOCK])?;
        let sw_st = self.alloc_tile_shaped(block, f32_t, &[scales_per_tile, QGEMM_TN])?;
        // A format with a minimum: `dmin * m` a column a group beside the
        // scales, and the activation's sum a row a group, both filled at
        // stage time (see `kquant.rs`).
        let mw_st = match fmt.has_min() {
            true => Some(self.alloc_tile_shaped(block, f32_t, &[scales_per_tile, QGEMM_TN])?),
            false => None,
        };
        let as_st = match fmt.has_min() {
            true => Some(self.alloc_tile_shaped(block, self.i32_t, &[QGEMM_TM, QGEMM_KT / KQ_GROUP])?),
            false => None,
        };
        let mut tabs = Vec::with_capacity(tables.len());
        for (table, &bytes) in tables.iter().zip(table_bytes) {
            let tile = self.alloc_tile_shaped(block, i8_t, &[1, bytes])?;
            self.stage_bytes(block, table, &tile, bytes)?;
            tabs.push(tile);
        }
        // The first decode below reads entries other threads wrote.
        self.barrier(block)?;

        let one = self.const_index(block, 1)?;
        let kd = if aq.shape[1] == DYN {
            self.push(block, memref::dim(aq.mem, one, self.loc))?
        } else {
            self.const_index(block, aq.shape[1])?
        };
        let kt_w = self.const_index(block, QGEMM_KT)?;
        let nk = self.divui(block, kd, kt_w)?;

        let lanes = self.qgemm_lanes(block)?;

        // Stage the first tile before the loop; the loop stages the next one
        // at the end of every turn.
        let zero_k = self.const_index(block, 0)?;
        let first = self.qgemm_load(block, fmt, &lanes, aq, asc, qb, d, zero_k)?;
        let planes = (mw_st.as_ref(), as_st.as_ref());
        self.qgemm_store(block, fmt, &lanes, &first, &a_st, &w_st, &sa_st, &sw_st, planes, &tabs)?;
        self.barrier(block)?;

        let acc_count = (PATCH * PATCH * 2) as usize;
        let mut args = vec![(self.index_t, self.loc)];
        args.extend(std::iter::repeat_n((f32_t, self.loc), acc_count));
        let kb = Block::new(&args);
        let kt = detach(kb.argument(0)?.into());
        let accs: Vec<Value<'c, 'c>> = (0..acc_count)
            .map(|slot| Ok(detach(kb.argument(slot + 1)?.into())))
            .collect::<Result<_>>()?;

        // The next tile's global reads go out first, so they are in flight
        // under the whole of this tile's contraction. The last turn re-reads
        // its own tile, which costs one stage and needs no branch.
        let last = self.push(&kb, arith::subi(nk, one, self.loc))?;
        let after = self.addi(&kb, kt, one)?;
        let next = self.push(&kb, arith::minui(after, last, self.loc))?;
        let fetched = self.qgemm_load(&kb, fmt, &lanes, aq, asc, qb, d, next)?;

        let accs = self.qgemm_contract(&kb, fmt, &lanes, &a_st, &w_st, &sa_st, &sw_st, planes, &accs)?;

        // Every warp is done with the stage before anyone overwrites it, and
        // the new stage is complete before anyone reads it.
        self.barrier(&kb)?;
        self.qgemm_store(&kb, fmt, &lanes, &fetched, &a_st, &w_st, &sa_st, &sw_st, planes, &tabs)?;
        self.barrier(&kb)?;
        kb.append_operation(scf::r#yield(&accs, self.loc));

        let init = self.zero_scalar(block, f32_t)?;
        let mut operands = vec![zero_k, nk, one];
        operands.extend(std::iter::repeat_n(init, acc_count));
        let kr = Region::new();
        kr.append_block(kb);
        let loop_op = block.append_operation(
            OperationBuilder::new("scf.for", self.loc)
                .add_operands(&operands)
                .add_results(&vec![f32_t; acc_count])
                .add_regions([kr])
                .build()?,
        );

        // The accumulators land straight in the output rows they belong to.
        let (rows, cols) = self.qgemm_patch_coords(block, &lanes)?;
        for (r, row) in rows.iter().enumerate() {
            for (c, col) in cols.iter().enumerate() {
                for dj in 0..2usize {
                    let off = self.const_index(block, dj as i64)?;
                    let at = self.addi(block, *col, off)?;
                    let slot = (r * PATCH as usize + c) * 2 + dj;
                    let value = detach(loop_op.result(slot)?.into());
                    block.append_operation(memref::store(value, out.mem, &[*row, at], self.loc));
                }
            }
        }
        self.barrier(block)?;
        Ok(())
    }

    /// Copy a `[1, bytes]` table into shared memory, 16 bytes a thread a turn.
    pub(super) fn stage_bytes(
        &mut self,
        block: &Block<'c>,
        from: &MemVal<'c>,
        to: &MemVal<'c>,
        bytes: i64,
    ) -> Result<()> {
        if bytes % CHUNK_BYTES != 0 {
            bail!("a staged table is a whole number of {CHUNK_BYTES} bytes");
        }
        let vec16 = Type::vector(&[CHUNK_BYTES as u64], self.i8_t);
        let tid = self.thread_id(block)?;
        let sixteen = self.const_index(block, CHUNK_BYTES)?;
        let zero = self.const_index(block, 0)?;
        let from_at = self.muli(block, tid, sixteen)?;
        let to_at = self.const_index(block, bytes)?;
        let step = self.const_index(block, CHUNK_BYTES * QGEMM_THREADS)?;
        let body = Block::new(&[(self.index_t, self.loc)]);
        let at = detach(body.argument(0)?.into());
        let v = self.vec_load_al(&body, from.mem, &[zero, at], vec16, CHUNK_BYTES)?;
        self.vec_store_al(&body, v, to.mem, &[zero, at], CHUNK_BYTES)?;
        body.append_operation(scf::r#yield(&[], self.loc));
        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(from_at, to_at, step, region, self.loc));
        Ok(())
    }

    /// The thread's place in everything: its fragment lane, its patch, and
    /// which pieces of each stage it copies or decodes.
    fn qgemm_lanes(&mut self, block: &Block<'c>) -> Result<Lanes<'c>> {
        let tid = self.thread_id(block)?;
        let c = |cg: &mut Self, v: i64| cg.const_index(block, v);
        let warp_w = c(self, WARP)?;
        let warp = self.divui(block, tid, warp_w)?;
        let lane = self.remui(block, tid, warp_w)?;
        let four = c(self, 4)?;
        let two = c(self, 2)?;
        let eight = c(self, 8)?;
        let quad = self.divui(block, lane, four)?;
        let in_quad = self.remui(block, lane, four)?;

        // Warps 0..7 as four down and two across, 32 rows or columns apiece.
        let patch_w = c(self, PATCH * IMMA_TILE)?;
        let wr = self.divui(block, warp, two)?;
        let wc = self.remui(block, warp, two)?;
        let i0 = self.muli(block, wr, patch_w)?;
        let j0 = self.muli(block, wc, patch_w)?;

        // ldmatrix.x4: lanes 8t..8t+7 address the eight rows of tile t, so a
        // lane's row is its index within the patch, and the chunk it reads is
        // permuted by that row's place in the bank period.
        let ldm_a_row = self.addi(block, i0, lane)?;
        let ldm_w_row = self.addi(block, j0, lane)?;
        let ldm_xor = self.remui(block, lane, eight)?;

        // Activation staging: piece q = tid + 256 i is row q / 8, chunk q % 8.
        let (mut a_rows, mut a_chunks, mut a_dst) = (Vec::new(), Vec::new(), Vec::new());
        let chunk_w = c(self, CHUNK_BYTES)?;
        for i in 0..(QGEMM_TM * QGEMM_KT / CHUNK_BYTES / QGEMM_THREADS) {
            let off = c(self, i * QGEMM_THREADS)?;
            let q = self.addi(block, tid, off)?;
            let row = self.divui(block, q, eight)?;
            let chunk = self.remui(block, q, eight)?;
            let phase = self.remui(block, row, eight)?;
            let phys = self.push(block, arith::xori(chunk, phase, self.loc))?;
            a_dst.push(self.muli(block, phys, chunk_w)?);
            a_rows.push(row);
            a_chunks.push(self.muli(block, chunk, chunk_w)?);
        }
        let sa_row = self.divui(block, tid, two)?;
        let sa_half = self.remui(block, tid, two)?;
        let sa_col = self.muli(block, sa_half, two)?;

        // Weight staging: four threads a column, one 32-element group each.
        let j = self.divui(block, tid, four)?;
        let g = self.remui(block, tid, four)?;
        let j_xor = self.remui(block, j, eight)?;
        let group_w = c(self, RAW_GROUP)?;
        let group = self.divui(block, j, group_w)?;
        let qb_row = self.muli(block, group, group_w)?;
        let in_group = self.remui(block, j, group_w)?;

        Ok(Lanes {
            i0,
            j0,
            quad,
            in_quad,
            ldm_a_row,
            ldm_w_row,
            ldm_xor,
            a_rows,
            a_chunks,
            a_dst,
            sa_row,
            sa_col,
            j,
            g,
            j_xor,
            qb_row,
            in_group,
        })
    }

    /// The rows of A and the even output columns this lane's accumulators
    /// cover: `i0 + 8r + quad` and `j0 + 8c + 2 in_quad`.
    fn qgemm_patch_coords(
        &mut self,
        block: &Block<'c>,
        lanes: &Lanes<'c>,
    ) -> Result<(Vec<Value<'c, 'c>>, Vec<Value<'c, 'c>>)> {
        let two = self.const_index(block, 2)?;
        let row_base = self.addi(block, lanes.i0, lanes.quad)?;
        let d_col = self.muli(block, lanes.in_quad, two)?;
        let col_base = self.addi(block, lanes.j0, d_col)?;
        let (mut rows, mut cols) = (Vec::new(), Vec::new());
        for t in 0..PATCH {
            let off = self.const_index(block, t * IMMA_TILE)?;
            rows.push(self.addi(block, row_base, off)?);
            cols.push(self.addi(block, col_base, off)?);
        }
        Ok((rows, cols))
    }

    /// Where tile `kt`'s bytes sit for this thread's column: block `kt / 2`,
    /// group `4 (kt % 2) + g`.
    fn qgemm_tile_at(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        lanes: &Lanes<'c>,
        kt: Value<'c, 'c>,
    ) -> Result<TileAt<'c>> {
        let two = self.const_index(block, 2)?;
        let four = self.const_index(block, 4)?;
        let blk = self.divui(block, kt, two)?;
        let half = self.remui(block, kt, two)?;
        let ib = self.muli(block, half, four)?;
        let ib = self.addi(block, ib, lanes.g)?;
        // Block `8 blk + j % 8` of the group's row.
        let blk_bytes = self.const_index(block, fmt.block_bytes())?;
        let group_w = self.const_index(block, RAW_GROUP)?;
        let d_blk = self.muli(block, blk, group_w)?;
        let d_col = self.addi(block, d_blk, lanes.in_group)?;
        let blk_off = self.muli(block, d_col, blk_bytes)?;
        Ok(TileAt { blk_off, ib, d_col })
    }

    /// The thread's global reads for tile `kt`, into registers: its share of
    /// the activation tile and scales, the column's block scale, then
    /// whatever the format's decode needs (see `qgemm_fmt.rs`).
    #[allow(clippy::too_many_arguments)]
    fn qgemm_load(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        lanes: &Lanes<'c>,
        aq: &MemVal<'c>,
        asc: &MemVal<'c>,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        kt: Value<'c, 'c>,
    ) -> Result<Vec<Value<'c, 'c>>> {
        let vec16 = Type::vector(&[CHUNK_BYTES as u64], self.i8_t);
        let vec2_f32 = Type::vector(&[2], self.f32_t);

        let kt_w = self.const_index(block, QGEMM_KT)?;
        let kbase = self.muli(block, kt, kt_w)?;
        let mut regs = Vec::with_capacity(lanes.a_rows.len() + 8);
        for (row, chunk) in lanes.a_rows.iter().zip(&lanes.a_chunks) {
            let col = self.addi(block, kbase, *chunk)?;
            regs.push(self.vec_load_al(block, aq.mem, &[*row, col], vec16, CHUNK_BYTES)?);
        }
        let groups = self.const_index(block, QGEMM_KT / Q8_BLOCK)?;
        let gbase = self.muli(block, kt, groups)?;
        let sa_col = self.addi(block, gbase, lanes.sa_col)?;
        regs.push(self.vec_load_al(block, asc.mem, &[lanes.sa_row, sa_col], vec2_f32, 8)?);

        let at = self.qgemm_tile_at(block, fmt, lanes, kt)?;
        let dv = self.push(block, memref::load(d.mem, &[lanes.qb_row, at.d_col], self.loc))?;
        regs.push(self.numeric_cast(block, dv, self.f32_t)?);
        self.qgemm_fmt_load(block, fmt, lanes, qb, &at, &mut regs)?;
        Ok(regs)
    }

    /// The thread's part of a stage: its activation pieces and scales copied
    /// in, its four weight fragments decoded, its column scales written, and
    /// for a format with a minimum the activation's group sums and the
    /// column minimums beside them.
    #[allow(clippy::too_many_arguments)]
    fn qgemm_store(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        lanes: &Lanes<'c>,
        regs: &[Value<'c, 'c>],
        a_st: &MemVal<'c>,
        w_st: &MemVal<'c>,
        sa_st: &MemVal<'c>,
        sw_st: &MemVal<'c>,
        planes: (Option<&MemVal<'c>>, Option<&MemVal<'c>>),
        tabs: &[MemVal<'c>],
    ) -> Result<()> {
        let (mw_st, as_st) = planes;
        let pieces = lanes.a_rows.len();
        for ((v, row), dst) in regs[..pieces].iter().zip(&lanes.a_rows).zip(&lanes.a_dst) {
            self.vec_store_al(block, *v, a_st.mem, &[*row, *dst], CHUNK_BYTES)?;
        }
        self.vec_store_al(block, regs[pieces], sa_st.mem, &[lanes.sa_row, lanes.sa_col], 8)?;
        if let Some(as_st) = as_st {
            // The activation's sum over each 32-element group: a thread's
            // sixteen-byte piece is half a group and the thread beside it
            // holds the other half, so one xor shuffle joins them and both
            // store the same word.
            let words_t = Type::vector(&[4], self.i32_t);
            let group_w = self.const_index(block, KQ_GROUP)?;
            for ((v, row), chunk) in regs[..pieces].iter().zip(&lanes.a_rows).zip(&lanes.a_chunks) {
                let w = self.vec_bitcast(block, *v, words_t)?;
                let words: Vec<Value<'c, 'c>> = (0..4)
                    .map(|i| self.vec_extract(block, w, &[i], self.i32_t))
                    .collect::<Result<_>>()?;
                let seed = self.zero_scalar(block, self.i32_t)?;
                let half = self.kq_byte_sum(block, &words, seed)?;
                let as_f = self.push(block, arith::bitcast(half, self.f32_t, self.loc))?;
                let other = self.shfl_xor_f32(block, as_f, 1)?;
                let other = self.push(block, arith::bitcast(other, self.i32_t, self.loc))?;
                let sum = self.push(block, arith::addi(half, other, self.loc))?;
                let group = self.divui(block, *chunk, group_w)?;
                block.append_operation(memref::store(sum, as_st.mem, &[*row, group], self.loc));
            }
        }
        let dv = regs[pieces + 1];
        let stage = Stage {
            lanes,
            w_st,
            sw_st,
            mw_st,
            tabs,
            dv,
        };
        self.qgemm_fmt_decode(block, fmt, &stage, &regs[pieces + 2..])
    }

    /// The warp's patch over one stage: four 32-element groups, each two
    /// `ldmatrix.x4` an operand, sixteen tensor tiles, and the scale
    /// epilogue.
    #[allow(clippy::too_many_arguments)]
    fn qgemm_contract(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        lanes: &Lanes<'c>,
        a_st: &MemVal<'c>,
        w_st: &MemVal<'c>,
        sa_st: &MemVal<'c>,
        sw_st: &MemVal<'c>,
        planes: (Option<&MemVal<'c>>, Option<&MemVal<'c>>),
        accs: &[Value<'c, 'c>],
    ) -> Result<Vec<Value<'c, 'c>>> {
        let (mw_st, as_st) = planes;
        let (i8_t, i32_t, f32_t) = (self.i8_t, self.i32_t, self.f32_t);
        let four_i8 = Type::vector(&[4], i8_t);
        let frag_t = Type::vector(&[1, 4], i8_t);
        let quad_t = Type::vector(&[PATCH as u64, 4], i8_t);
        let acc_t = Type::vector(&[1, 2], i32_t);
        let two_f32 = Type::vector(&[2], f32_t);
        let shape = self.mma_shape(IMMA_TILE, IMMA_TILE, IMMA_K)?;
        let zero_i = self.zero_scalar(block, i32_t)?;
        let empty = self.vec_broadcast(block, zero_i, acc_t)?;
        let halves = Q8_BLOCK / IMMA_K;
        let split = fmt.scale_run() == IMMA_K;
        let per_col = if split { halves } else { 1 };
        let (rows, cols) = self.qgemm_patch_coords(block, lanes)?;
        let chunk_w = self.const_index(block, CHUNK_BYTES)?;

        let mut accs = accs.to_vec();
        for s in 0..(QGEMM_KT / Q8_BLOCK) {
            // The weight scales this group needs: one a column, or one a
            // column and half.
            let mut sw = Vec::with_capacity((PATCH * per_col) as usize);
            for col in &cols {
                for h in 0..per_col {
                    let s_idx = self.const_index(block, s * per_col + h)?;
                    sw.push(self.vec_load_al(block, sw_st.mem, &[s_idx, *col], two_f32, 8)?);
                }
            }
            let s_idx = self.const_index(block, s)?;
            let mut sa = Vec::with_capacity(PATCH as usize);
            for row in &rows {
                sa.push(self.push(block, memref::load(sa_st.mem, &[*row, s_idx], self.loc))?);
            }
            // The minimum term's operands, where the format has one: the
            // column minimums like the scales, the row sums like the
            // activation scales.
            let mut mw = Vec::with_capacity(PATCH as usize);
            if let Some(mw_st) = mw_st {
                for col in &cols {
                    mw.push(self.vec_load_al(block, mw_st.mem, &[s_idx, *col], two_f32, 8)?);
                }
            }
            let mut asum = Vec::with_capacity(PATCH as usize);
            if let Some(as_st) = as_st {
                for row in &rows {
                    let raw = self.push(block, memref::load(as_st.mem, &[*row, s_idx], self.loc))?;
                    asum.push(self.small_int_to_f32(block, raw)?);
                }
            }

            let (mut a_frags, mut w_frags) = (Vec::new(), Vec::new());
            for h in 0..halves {
                let chunk = self.const_index(block, s * halves + h)?;
                let phys = self.push(block, arith::xori(chunk, lanes.ldm_xor, self.loc))?;
                let col = self.muli(block, phys, chunk_w)?;
                let a4 = self.ldmatrix(block, a_st.mem, [lanes.ldm_a_row, col], PATCH, false, quad_t)?;
                let w4 = self.ldmatrix(block, w_st.mem, [lanes.ldm_w_row, col], PATCH, false, quad_t)?;
                for t in 0..PATCH {
                    let a = self.vec_extract(block, a4, &[t], four_i8)?;
                    a_frags.push(self.vec_shape_cast(block, a, frag_t)?);
                    let w = self.vec_extract(block, w4, &[t], four_i8)?;
                    w_frags.push(self.vec_shape_cast(block, w, frag_t)?);
                }
            }
            let frag = |h: i64, t: i64| (h * PATCH + t) as usize;

            for r in 0..PATCH {
                for c in 0..PATCH {
                    // One accumulator over both halves where a scale covers
                    // the group, one a half where it does not.
                    let mut sums = Vec::with_capacity(per_col as usize);
                    let mut sum = empty;
                    for h in 0..halves {
                        let seed = if split { empty } else { sum };
                        sum = self.mma_sync(
                            block,
                            a_frags[frag(h, r)],
                            w_frags[frag(h, c)],
                            seed,
                            shape,
                            acc_t,
                        )?;
                        if split {
                            sums.push(sum);
                        }
                    }
                    if !split {
                        sums.push(sum);
                    }
                    for dj in 0..2i64 {
                        let slot = ((r * PATCH + c) * 2 + dj) as usize;
                        let sw_of = |cg: &mut Self, h: usize| {
                            cg.vec_extract(block, sw[c as usize * per_col as usize + h], &[dj], f32_t)
                        };
                        let raw_of = |cg: &mut Self, h: usize| -> Result<Value<'c, 'c>> {
                            let raw = cg.vec_extract(block, sums[h], &[0, dj], i32_t)?;
                            cg.small_int_to_f32(block, raw)
                        };
                        accs[slot] = if split {
                            // acc += sa * (c0 * sw0 + c1 * sw1): two scale
                            // products a group rather than four.
                            let (w0, w1) = (sw_of(self, 0)?, sw_of(self, 1)?);
                            let (c0, c1) = (raw_of(self, 0)?, raw_of(self, 1)?);
                            let inner = self.push(block, arith::mulf(c0, w0, self.loc))?;
                            let inner = self.elem_mac(block, f32_t, c1, w1, inner)?;
                            self.elem_mac(block, f32_t, inner, sa[r as usize], accs[slot])?
                        } else if mw_st.is_some() {
                            // acc += sa * (c * sw - asum * mw): the run's
                            // minimum, weighed by the activation's sum.
                            let w0 = sw_of(self, 0)?;
                            let as_f = raw_of(self, 0)?;
                            let mw0 = self.vec_extract(block, mw[c as usize], &[dj], f32_t)?;
                            let inner = self.push(block, arith::mulf(as_f, w0, self.loc))?;
                            let sub = self.push(block, arith::mulf(asum[r as usize], mw0, self.loc))?;
                            let inner = self.push(block, arith::subf(inner, sub, self.loc))?;
                            self.elem_mac(block, f32_t, inner, sa[r as usize], accs[slot])?
                        } else {
                            let w0 = sw_of(self, 0)?;
                            let scale = self.push(block, arith::mulf(sa[r as usize], w0, self.loc))?;
                            let as_f = raw_of(self, 0)?;
                            self.elem_mac(block, f32_t, as_f, scale, accs[slot])?
                        };
                    }
                }
            }
        }
        Ok(accs)
    }
}

/// What a format's decode writes into: the stage's weight tile and scale
/// plane, the tables already in shared memory, and the column's block scale.
pub(super) struct Stage<'a, 'c> {
    pub(super) lanes: &'a Lanes<'c>,
    pub(super) w_st: &'a MemVal<'c>,
    pub(super) sw_st: &'a MemVal<'c>,
    /// The column minimums, for a format that subtracts one.
    pub(super) mw_st: Option<&'a MemVal<'c>>,
    pub(super) tabs: &'a [MemVal<'c>],
    pub(super) dv: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {
    /// Write the thread's `l`-th octet, two i32 words of four bytes, where
    /// the tile keeps elements `32 g + 8 l ..+ 8` of column `j`: chunk
    /// `2 g + l / 2`, permuted by the row's phase, low or high half.
    pub(super) fn qgemm_put_octet(
        &mut self,
        block: &Block<'c>,
        stage: &Stage<'_, 'c>,
        l: i64,
        w0: Value<'c, 'c>,
        w1: Value<'c, 'c>,
    ) -> Result<()> {
        let two_i32 = Type::vector(&[2], self.i32_t);
        let eight_i8 = Type::vector(&[8], self.i8_t);
        let two = self.const_index(block, 2)?;
        let chunk_w = self.const_index(block, CHUNK_BYTES)?;
        let g2 = self.muli(block, stage.lanes.g, two)?;
        let chunk_off = self.const_index(block, l / 2)?;
        let chunk = self.addi(block, g2, chunk_off)?;
        let phys = self.push(block, arith::xori(chunk, stage.lanes.j_xor, self.loc))?;
        let dst = self.muli(block, phys, chunk_w)?;
        let half_off = self.const_index(block, (l % 2) * 8)?;
        let dst = self.addi(block, dst, half_off)?;
        let pair = self.vec_broadcast(block, w0, two_i32)?;
        let pair = self.vec_insert(block, w1, pair, &[1])?;
        let bytes = self.vec_bitcast(block, pair, eight_i8)?;
        self.vec_store_al(block, bytes, stage.w_st.mem, &[stage.lanes.j, dst], 8)
    }

    /// Write the thread's group minimum, `dmin * m`, for a format that
    /// subtracts one.
    pub(super) fn qgemm_put_min(
        &mut self,
        block: &Block<'c>,
        stage: &Stage<'_, 'c>,
        value: Value<'c, 'c>,
    ) -> Result<()> {
        let Some(mw_st) = stage.mw_st else {
            bail!("a minimum for a format that has none");
        };
        block.append_operation(memref::store(value, mw_st.mem, &[stage.lanes.g, stage.lanes.j], self.loc));
        Ok(())
    }

    /// Write the thread's group scale, or the `h`-th of its `per_group`.
    pub(super) fn qgemm_put_scale(
        &mut self,
        block: &Block<'c>,
        stage: &Stage<'_, 'c>,
        per_group: i64,
        h: i64,
        value: Value<'c, 'c>,
    ) -> Result<()> {
        let per = self.const_index(block, per_group)?;
        let row = self.muli(block, stage.lanes.g, per)?;
        let off = self.const_index(block, h)?;
        let row = self.addi(block, row, off)?;
        block.append_operation(memref::store(value, stage.sw_st.mem, &[row, stage.lanes.j], self.loc));
        Ok(())
    }
}
