// The `<fmt>_qgemm_t` family: a prompt projection against a raw quantized
// weight, with both operands staged through shared memory.
//
// For each 128 elements of `k`, the CTA stages 128 activation rows and 64
// decoded weight columns as int8 tiles, plus the column scales. It contracts
// them with `ldmatrix` and the integer `mma.sync` (see `qgemm_contract.rs`).
// The next tile's reads are issued before this one is contracted. With
// `cp.async`, and room for it in the static 48 KB, the activation tile is
// doubled and copied straight into the half not being contracted, so only
// the weight passes through registers.
//
// Format decode lives in `qgemm_fmt.rs`. The epilogue matches
// `iq1s_qmma_t`'s, so both agree bit for bit. `q8_qgemm_t` is the same
// kernel over a Q8_0 weight held as planes, its quants staged as they are.
// Given the activation's (scale, sum) pairs, a format with a minimum takes
// the minimum term through the f16 tensor op once per 256-element block
// instead of through the epilogue (see `qgemm_contract.rs`); the stage
// writers below lay out its two operands.

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

/// Bytes of a 16-element `k` chunk, which is what one `ldmatrix` row
/// carries.
pub(super) const CHUNK_BYTES: i64 = 16;

/// Tiles of 8 x 8 a warp's patch spans, each way.
pub(super) const PATCH: i64 = 4;

/// Words of a row of the activation scale stage when the (scale, sum) pairs
/// are given: the four groups' scales, four spare words, then the four
/// product words. At 32 bytes a row, the eight rows a warp's quads read for
/// one group would fall on four banks twice over; at 48 they spread over
/// eight, four banks apart, and a row's scales load as one aligned 16
/// bytes.
const PAIR_STRIDE: i64 = PAIR_PRODUCTS + 4;

/// The first product word of such a row: word `PAIR_PRODUCTS + 2 half + p`
/// holds `f16(scale * sum)` for groups `2 p` and `2 p + 1` of the stage
/// that is half `half` of a 256-element block, the minimum term's A operand
/// (see `qgemm_min_term`).
pub(super) const PAIR_PRODUCTS: i64 = 2 * (QGEMM_KT / Q8_BLOCK);

/// Bytes of the stage with the activation tile doubled, `row_sums` when it
/// holds the activation's group sums as a plane of its own, `pairs` when
/// they sit beside the activation scales.
fn async_stage_bytes(fmt: QgFormat, row_sums: bool, pairs: bool) -> i64 {
    let scales = QGEMM_KT / fmt.scale_run() * QGEMM_TN * 4;
    let sums = if row_sums { QGEMM_TM * (QGEMM_KT / KQ_GROUP) * 4 } else { 0 };
    // With pairs, the minimums are f16 for both halves of a block: the same
    // bytes as one stage of f32.
    let mins = if fmt.has_min() { scales + sums } else { 0 };
    let sa = QGEMM_TM * 4 * if pairs { PAIR_STRIDE } else { QGEMM_KT / Q8_BLOCK };
    let a = 2 * QGEMM_TM * QGEMM_KT + sa;
    a + QGEMM_TN * QGEMM_KT + scales + mins + fmt.tables().iter().sum::<i64>()
}

/// The loop-invariant geometry of one thread.
pub(super) struct Lanes<'c> {
    /// The warp's patch origin and the lane's place in a fragment.
    pub(super) i0: Value<'c, 'c>,
    pub(super) j0: Value<'c, 'c>,
    pub(super) quad: Value<'c, 'c>,
    pub(super) in_quad: Value<'c, 'c>,
    /// The `ldmatrix` row this lane addresses in each operand's tile, and
    /// the swizzle its chunk index is permuted by.
    pub(super) ldm_a_row: Value<'c, 'c>,
    pub(super) ldm_w_row: Value<'c, 'c>,
    pub(super) ldm_xor: Value<'c, 'c>,
    /// The same for `m16n8k32`'s whole fragments, on a chip that has it.
    pub(super) k32: Option<K32Lanes<'c>>,
    /// Activation staging: the row and chunk of each of the thread's four
    /// 16-byte pieces, and the permuted column each lands at.
    a_rows: Vec<Value<'c, 'c>>,
    a_chunks: Vec<Value<'c, 'c>>,
    a_dst: Vec<Value<'c, 'c>>,
    /// Activation scale staging: the row, the first of the two scales, and
    /// which pair of the stage's groups they are.
    sa_row: Value<'c, 'c>,
    sa_col: Value<'c, 'c>,
    sa_half: Value<'c, 'c>,
    /// The weight column the thread decodes and its 32-element group within
    /// the stage, plus `j % 8`, the swizzle phase of that row.
    pub(super) j: Value<'c, 'c>,
    pub(super) g: Value<'c, 'c>,
    j_xor: Value<'c, 'c>,
    /// The row of the grouped weight holding the column's bytes, `8 (j / 8)`,
    /// and `j % 8`. See [`RAW_GROUP`].
    pub(super) qb_row: Value<'c, 'c>,
    in_group: Value<'c, 'c>,
}

/// The `ldmatrix.x4` addresses of `m16n8k32`'s fragments: lanes 0..15 name
/// A's 16 rows at a group's first half and lanes 16..31 at its second; B's
/// quad is columns 0..7 at both halves, then columns 8..15.
pub(super) struct K32Lanes<'c> {
    /// A's row and the half of the group the lane's octet reads.
    pub(super) a_row: Value<'c, 'c>,
    pub(super) a_half: Value<'c, 'c>,
    /// The same for B.
    pub(super) w_row: Value<'c, 'c>,
    pub(super) w_half: Value<'c, 'c>,
}

/// Where a tile's bytes sit in the weight: the 256-element block, and the
/// thread's group within it.
pub(super) struct TileAt<'c> {
    /// The byte offset of the block in the group's row, and its scale's
    /// entry in the scale plane. Both are `8 blk + j % 8` blocks in.
    pub(super) blk_off: Value<'c, 'c>,
    pub(super) d_col: Value<'c, 'c>,
    /// The thread's group, 0 to 7, within the block.
    pub(super) ib: Value<'c, 'c>,
}

impl<'c> Codegen<'c> {
    /// out[i, j] = sum_k a[i, k] * w[j, k] with `w` decoded from `fmt`, both
    /// operands staged through shared memory. See the module comment.
    ///
    /// For a format with a minimum, `pairs` is the activation's (scale, sum)
    /// per row and 32-element group, `[128, 2 K/32] f32`, the sum over the
    /// group's quantized values as the integer it is. Given, the stage holds
    /// each pair's scale and, as f16, its product, `asc` is not read, and
    /// nothing is summed in the kernel.
    ///
    /// For a plane-held weight (`q8_qgemm_t`), `qb` is the weight's `[64, K]`
    /// plane and `d` its `[K/32, 64]` f32 scales.
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
        pairs: Option<&MemVal<'c>>,
        out: &MemVal<'c>,
    ) -> Result<()> {
        let name = fmt.intrinsic();
        for (v, what) in [(aq, "a"), (asc, "a scales"), (qb, "qb"), (d, "d")]
            .into_iter()
            .chain(tables.iter().map(|t| (t, "table")))
            .chain(pairs.map(|p| (p, "a pairs")))
        {
            if v.shape.len() != 2 {
                bail!("`{name}` {what} must be a rank-2 tile");
            }
            if v.is_masked() {
                bail!("`{name}` {what} must be a fully in-bounds slice");
            }
            if v.swizzle.is_some() {
                bail!("`{name}` {what} must not be swizzled");
            }
        }
        if aq.elem != self.i8_t || qb.elem != self.i8_t || tables.iter().any(|t| t.elem != self.i8_t)
        {
            bail!("`{name}` contracts an int8 activation against the format's raw bytes");
        }
        if asc.elem != self.f32_t {
            bail!("`{name}`'s activation scale must be f32");
        }
        if let Some(p) = pairs {
            if !fmt.has_min() {
                bail!("`{name}` takes no activation pairs: its format subtracts no minimum");
            }
            if p.elem != self.f32_t {
                bail!("`{name}`'s activation pairs must be f32");
            }
            self.check_shapes(&[asc.shape[0]], &[p.shape[0]], "qgemm a pairs rows")?;
            if asc.shape[1] != DYN && p.shape[1] != DYN && p.shape[1] != 2 * asc.shape[1] {
                bail!("`{name}`'s activation pairs hold a (scale, sum) per group: twice the scales' columns");
            }
        }
        let planar = fmt.planar();
        if planar && d.elem != self.f32_t {
            bail!("`{name}`'s weight scales must be f32, one per 32 elements and column");
        }
        if !planar && d.elem != self.f16_t {
            bail!("`{name}`'s block scale must be f16");
        }
        if out.elem != self.f32_t || out.is_masked() {
            bail!("`{name}` needs a fully in-bounds f32 destination");
        }
        if !self.has_int8_mma() {
            bail!("`{name}` needs the integer tensor cores (sm_75 or later)");
        }
        if self.cta_threads != QGEMM_THREADS {
            bail!("`{name}` is written for a CTA of {QGEMM_THREADS} threads");
        }
        let (md, nd) = (aq.shape[0], qb.shape[0]);
        if md != QGEMM_TM || nd != QGEMM_TN {
            bail!("`{name}` takes a {QGEMM_TM} x {QGEMM_TN} output tile, got {md} x {nd}");
        }
        self.check_shapes(&[md, nd], &out.shape, "qgemm destination")?;
        self.check_shapes(&[md], &[asc.shape[0]], "qgemm a scale rows")?;
        if planar {
            // The planes: the weight `[N, K]` and its scales `[K/32, N]`.
            self.check_shapes(&[nd], &[d.shape[1]], "qgemm weight scale columns")?;
            if qb.shape[1] != DYN && aq.shape[1] != DYN && qb.shape[1] != aq.shape[1] {
                bail!("`{name}`'s weight and activation must share K");
            }
            if d.shape[0] != DYN && aq.shape[1] != DYN && d.shape[0] * Q8_BLOCK != aq.shape[1] {
                bail!("`{name}`'s weight scales must cover K/32 rows");
            }
        } else {
            self.check_shapes(&[nd], &[d.shape[0]], "qgemm d rows")?;
        }
        let table_bytes = fmt.tables();
        if tables.len() != table_bytes.len()
            || tables.iter().zip(table_bytes).any(|(t, &want)| t.shape[1] != want)
        {
            bail!("`{name}` wants tables of {table_bytes:?} bytes");
        }
        if aq.shape[1] != DYN && aq.shape[1] % 256 != 0 {
            bail!("`{name}` needs a whole number of 256-element blocks");
        }

        let (i8_t, f32_t) = (self.i8_t, self.f32_t);
        let scales_per_tile = QGEMM_KT / fmt.scale_run();

        // The stage: both operand tiles, the scales of each, and the tables.
        // Pairs given are staged as they are, so a row's (scale, sum) for a
        // group is one load. Otherwise, with `m16n8k32`, the contraction sums
        // the activation's groups itself, so the stage holds no sums for a
        // format with a minimum; before it, a plane of them is summed at
        // stage time.
        let paired = pairs.is_some();
        let row_sums = fmt.has_min() && !paired && !self.qgemm_k32(fmt);
        let a_async = self.has_cp_async() && async_stage_bytes(fmt, row_sums, paired) <= self.static_smem_per_block();
        let a_rows = if a_async { 2 * QGEMM_TM } else { QGEMM_TM };
        let a_st = self.alloc_tile_shaped(block, i8_t, &[a_rows, QGEMM_KT])?;
        let w_st = self.alloc_tile_shaped(block, i8_t, &[QGEMM_TN, QGEMM_KT])?;
        let sa_cols = if paired { PAIR_STRIDE } else { QGEMM_KT / Q8_BLOCK };
        let sa_st = self.alloc_tile_shaped(block, f32_t, &[QGEMM_TM, sa_cols])?;
        let sw_st = self.alloc_tile_shaped(block, f32_t, &[scales_per_tile, QGEMM_TN])?;
        // For a format with a minimum: `dmin * m` per column and group, and
        // the activation's sum per row and group, filled at stage time (see
        // `kquant.rs`). Given the pairs, the minimums are `-(dmin * m)` as
        // f16 for both halves of a block, `[column, 2 group + half]`: the
        // minimum term's B operand, in the same bytes.
        let mw_st = match (fmt.has_min(), paired) {
            (false, _) => None,
            (true, false) => Some(self.alloc_tile_shaped(block, f32_t, &[scales_per_tile, QGEMM_TN])?),
            (true, true) => Some(self.alloc_tile_shaped(block, self.f16_t, &[QGEMM_TN, 2 * scales_per_tile])?),
        };
        let as_st = match row_sums {
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

        // The m16n8k32 regrouping holds about twice the fragment registers.
        // At two CTAs an SM it spills, at one it does not and runs faster, so
        // the kernel asks for one whatever its `@launch` says.
        if self.int8_mma_k() == 32
            && let Some(launch) = self.launch.as_mut()
        {
            launch.min_blocks = Some(1);
        }
        let lanes = self.qgemm_lanes(block, fmt)?;

        // Stage the first tile before the loop. The loop stages the next one
        // at the end of every turn.
        let zero_k = self.const_index(block, 0)?;
        let first_half = a_async.then_some(zero_k);
        let into = first_half.map(|half| (&a_st, half));
        let operands = QgOperands { aq, asc, pairs, qb, d };
        let (first, group) = self.qgemm_load(block, fmt, &lanes, &operands, zero_k, into)?;
        if let Some(group) = group {
            self.async_wait(block, group)?;
        }
        let planes = (mw_st.as_ref(), as_st.as_ref());
        let stage = (&a_st, &w_st, &sa_st, &sw_st);
        let first_block_half = match paired {
            true => Some(zero_k),
            false => None,
        };
        self.qgemm_store(block, fmt, &lanes, &first, first_half, stage, planes, &tabs, first_block_half)?;
        self.barrier(block)?;

        let acc_count = (PATCH * PATCH * 2) as usize;
        let mut args = vec![(self.index_t, self.loc)];
        args.extend(std::iter::repeat_n((f32_t, self.loc), acc_count));
        let kb = Block::new(&args);
        let kt = detach(kb.argument(0)?.into());
        let accs: Vec<Value<'c, 'c>> = (0..acc_count)
            .map(|slot| Ok(detach(kb.argument(slot + 1)?.into())))
            .collect::<Result<_>>()?;

        // Issue the next tile's reads first, so they are in flight during
        // this tile's contraction. The last turn re-reads its own tile rather
        // than branching.
        let last = self.push(&kb, arith::subi(nk, one, self.loc))?;
        let after = self.addi(&kb, kt, one)?;
        let next = self.push(&kb, arith::minui(after, last, self.loc))?;
        // The activation half this turn contracts, and the one the next
        // tile is copied into. The turn's parity is also which half of a
        // 256-element block its tile is, which the minimum term goes by.
        let (parity, halves) = match a_async {
            true => {
                let two = self.const_index(&kb, 2)?;
                let tm = self.const_index(&kb, QGEMM_TM)?;
                let parity = self.remui(&kb, kt, two)?;
                let cur = self.muli(&kb, parity, tm)?;
                (Some(parity), Some((cur, self.push(&kb, arith::subi(tm, cur, self.loc))?)))
            }
            false if paired => {
                let two = self.const_index(&kb, 2)?;
                (Some(self.remui(&kb, kt, two)?), None)
            }
            false => (None, None),
        };
        let into = halves.map(|(_, nxt)| (&a_st, nxt));
        let (fetched, group) = self.qgemm_load(&kb, fmt, &lanes, &operands, next, into)?;

        let cur = halves.map(|(cur, _)| cur);
        let block_half = parity.filter(|_| paired);
        let accs = self.qgemm_contract(&kb, fmt, &lanes, (&a_st, cur), &w_st, (&sa_st, block_half), &sw_st, planes, &accs)?;

        // Every warp must finish with the stage before it is overwritten, and
        // the new stage must be complete before it is read.
        self.barrier(&kb)?;
        if let Some(group) = group {
            self.async_wait(&kb, group)?;
        }
        let nxt = halves.map(|(_, nxt)| nxt);
        let next_block_half = match paired {
            true => {
                let two = self.const_index(&kb, 2)?;
                Some(self.remui(&kb, next, two)?)
            }
            false => None,
        };
        self.qgemm_store(&kb, fmt, &lanes, &fetched, nxt, stage, planes, &tabs, next_block_half)?;
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

        // A lane's two adjacent columns go out as one 8-byte store when the
        // rows keep them aligned, so a warp's store fills whole sectors.
        let paired = out.align_div == 0 || out.align_div % 2 == 0;
        let pair_t = Type::vector(&[2], f32_t);
        let (rows, cols) = self.qgemm_patch_coords(block, &lanes)?;
        for (r, row) in rows.iter().enumerate() {
            for (c, col) in cols.iter().enumerate() {
                let slot = (r * PATCH as usize + c) * 2;
                let lo = detach(loop_op.result(slot)?.into());
                let hi = detach(loop_op.result(slot + 1)?.into());
                if paired {
                    let v = self.vec_broadcast(block, lo, pair_t)?;
                    let v = self.vec_insert(block, hi, v, &[1])?;
                    self.vec_store_al(block, v, out.mem, &[*row, *col], 8)?;
                    continue;
                }
                for (dj, value) in [lo, hi].into_iter().enumerate() {
                    let at = self.addi(block, *col, self.const_index(block, dj as i64)?)?;
                    block.append_operation(memref::store(value, out.mem, &[*row, at], self.loc));
                }
            }
        }
        self.barrier(block)?;
        Ok(())
    }

    /// Copies a `[1, bytes]` table into shared memory, 16 bytes per thread
    /// per turn.
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

    /// Whether the stage contracts through `m16n8k32` whole fragments. A
    /// format with a scale per 16 elements would take `m16n8k16` per half,
    /// slower than the native `m8n8k16`, so it waits for Blackwell, which
    /// has no native `m8n8k16`.
    fn qgemm_k32(&self, fmt: QgFormat) -> bool {
        self.int8_mma_k() == 32 || (self.has_int8_mma_k32() && fmt.scale_run() != IMMA_K)
    }

    /// The thread's place in everything: its fragment lane, its patch, and
    /// which pieces of each stage it copies or decodes.
    fn qgemm_lanes(&mut self, block: &Block<'c>, fmt: QgFormat) -> Result<Lanes<'c>> {
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

        // Warps 0..7 as four down and two across, 32 rows or columns each.
        let patch_w = c(self, PATCH * IMMA_TILE)?;
        let wr = self.divui(block, warp, two)?;
        let wc = self.remui(block, warp, two)?;
        let i0 = self.muli(block, wr, patch_w)?;
        let j0 = self.muli(block, wc, patch_w)?;

        // ldmatrix.x4: lanes 8t..8t+7 address the eight rows of tile t. So a
        // lane's row is its index within the patch, and its chunk is permuted
        // by that row's place in the bank period.
        let ldm_a_row = self.addi(block, i0, lane)?;
        let ldm_w_row = self.addi(block, j0, lane)?;
        let ldm_xor = self.remui(block, lane, eight)?;
        let k32 = match self.qgemm_k32(fmt) {
            true => {
                let sixteen = c(self, 16)?;
                let a_in = self.remui(block, lane, sixteen)?;
                let a_half = self.divui(block, lane, sixteen)?;
                let w_pair = self.muli(block, a_half, eight)?;
                let w_in = self.addi(block, ldm_xor, w_pair)?;
                let octet = self.divui(block, lane, eight)?;
                Some(K32Lanes {
                    a_row: self.addi(block, i0, a_in)?,
                    a_half,
                    w_row: self.addi(block, j0, w_in)?,
                    w_half: self.remui(block, octet, two)?,
                })
            }
            false => None,
        };

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

        // Weight staging: four threads per column, one 32-element group each.
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
            k32,
            a_rows,
            a_chunks,
            a_dst,
            sa_row,
            sa_col,
            sa_half,
            j,
            g,
            j_xor,
            qb_row,
            in_group,
        })
    }

    /// The rows of A and the even output columns this lane's accumulators
    /// cover: `i0 + 8r + quad` and `j0 + 8c + 2 in_quad`.
    pub(super) fn qgemm_patch_coords(
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
    /// the activation tile, scales and sums, the column's block scale, then
    /// whatever the format's decode needs (see `qgemm_fmt.rs`), or for a
    /// plane-held weight its group scale and quants. With `into`,
    /// the activation pieces go straight to that stage at that row offset
    /// instead, as one `cp.async` group, which is returned.
    fn qgemm_load(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        lanes: &Lanes<'c>,
        operands: &QgOperands<'_, 'c>,
        kt: Value<'c, 'c>,
        into: Option<(&MemVal<'c>, Value<'c, 'c>)>,
    ) -> Result<(Vec<Value<'c, 'c>>, Option<Value<'c, 'c>>)> {
        let QgOperands { aq, asc, pairs, qb, d } = *operands;
        let vec16 = Type::vector(&[CHUNK_BYTES as u64], self.i8_t);
        let vec2_f32 = Type::vector(&[2], self.f32_t);
        let vec4_f32 = Type::vector(&[4], self.f32_t);

        let kt_w = self.const_index(block, QGEMM_KT)?;
        let kbase = self.muli(block, kt, kt_w)?;
        let mut regs = Vec::with_capacity(lanes.a_rows.len() + 8);
        let mut group = None;
        match into {
            None => {
                for (row, chunk) in lanes.a_rows.iter().zip(&lanes.a_chunks) {
                    let col = self.addi(block, kbase, *chunk)?;
                    regs.push(self.vec_load_al(block, aq.mem, &[*row, col], vec16, CHUNK_BYTES)?);
                }
            }
            Some((a_st, half)) => {
                for ((row, chunk), dst) in lanes.a_rows.iter().zip(&lanes.a_chunks).zip(&lanes.a_dst) {
                    let col = self.addi(block, kbase, *chunk)?;
                    let at = self.addi(block, *row, half)?;
                    self.async_copy(block, aq, &[*row, col], a_st, &[at, *dst], CHUNK_BYTES)?;
                }
                group = Some(self.async_create_group(block)?);
            }
        }
        // The thread's two scales, or its two (scale, sum) pairs.
        let groups = self.const_index(block, QGEMM_KT / Q8_BLOCK)?;
        let gbase = self.muli(block, kt, groups)?;
        let sa_col = self.addi(block, gbase, lanes.sa_col)?;
        match pairs {
            Some(pairs) => {
                let two = self.const_index(block, 2)?;
                let col = self.muli(block, sa_col, two)?;
                regs.push(self.vec_load_al(block, pairs.mem, &[lanes.sa_row, col], vec4_f32, 16)?);
            }
            None => regs.push(self.vec_load_al(block, asc.mem, &[lanes.sa_row, sa_col], vec2_f32, 8)?),
        }

        if fmt.planar() {
            self.q8_gemm_load(block, lanes, qb, d, kt, &mut regs)?;
            return Ok((regs, group));
        }
        let at = self.qgemm_tile_at(block, fmt, lanes, kt)?;
        let dv = self.push(block, memref::load(d.mem, &[lanes.qb_row, at.d_col], self.loc))?;
        regs.push(self.numeric_cast(block, dv, self.f32_t)?);
        self.qgemm_fmt_load(block, fmt, lanes, qb, &at, &mut regs)?;
        Ok((regs, group))
    }

    /// The thread's part of a stage: activation pieces and scales copied in,
    /// four weight fragments decoded, column scales written, and for a
    /// format with a minimum, the activation's group sums and column
    /// minimums beside them. With `copied`, the activation pieces are already
    /// in the stage at that row offset, and the sums read them back. With
    /// `block_half`, the (scale, sum) pairs came in whole: their scales go
    /// in, and the products of the thread's two as f16 in the word of that
    /// half of a 256-element block, and the minimums go in as f16.
    #[allow(clippy::too_many_arguments)]
    fn qgemm_store(
        &mut self,
        block: &Block<'c>,
        fmt: QgFormat,
        lanes: &Lanes<'c>,
        regs: &[Value<'c, 'c>],
        copied: Option<Value<'c, 'c>>,
        stage: StageTiles<'_, 'c>,
        planes: (Option<&MemVal<'c>>, Option<&MemVal<'c>>),
        tabs: &[MemVal<'c>],
        block_half: Option<Value<'c, 'c>>,
    ) -> Result<()> {
        let (a_st, w_st, sa_st, sw_st) = stage;
        let (mw_st, as_st) = planes;
        let pieces = if copied.is_some() { 0 } else { lanes.a_rows.len() };
        for ((v, row), dst) in regs[..pieces].iter().zip(&lanes.a_rows).zip(&lanes.a_dst) {
            self.vec_store_al(block, *v, a_st.mem, &[*row, *dst], CHUNK_BYTES)?;
        }
        match block_half {
            Some(half) => {
                // The scales of the thread's two (scale, sum) pairs, where
                // the scales alone would go, then the pairs' products as one
                // f16 word (see `PAIR_PRODUCTS`).
                let two_f32 = Type::vector(&[2], self.f32_t);
                let s0 = self.vec_extract(block, regs[pieces], &[0], self.f32_t)?;
                let s1 = self.vec_extract(block, regs[pieces], &[2], self.f32_t)?;
                let scales = self.vec_broadcast(block, s0, two_f32)?;
                let scales = self.vec_insert(block, s1, scales, &[1])?;
                self.vec_store_al(block, scales, sa_st.mem, &[lanes.sa_row, lanes.sa_col], 8)?;
                self.qgemm_put_products(block, lanes, regs[pieces], sa_st, half)?;
            }
            None => self.vec_store_al(block, regs[pieces], sa_st.mem, &[lanes.sa_row, lanes.sa_col], 8)?,
        }
        if let Some(as_st) = as_st {
            // The activation's sum over each 32-element group. A thread's
            // sixteen-byte piece is half a group, and its neighbour holds the
            // other half. One xor shuffle joins them, and both store the same
            // word.
            let words_t = Type::vector(&[4], self.i32_t);
            let group_w = self.const_index(block, KQ_GROUP)?;
            let a_pieces = match copied {
                None => regs[..pieces].to_vec(),
                Some(half) => {
                    let vec16 = Type::vector(&[CHUNK_BYTES as u64], self.i8_t);
                    let mut out = Vec::with_capacity(lanes.a_rows.len());
                    for (row, dst) in lanes.a_rows.iter().zip(&lanes.a_dst) {
                        let at = self.addi(block, *row, half)?;
                        out.push(self.vec_load_al(block, a_st.mem, &[at, *dst], vec16, CHUNK_BYTES)?);
                    }
                    out
                }
            };
            for ((v, row), chunk) in a_pieces.iter().zip(&lanes.a_rows).zip(&lanes.a_chunks) {
                let w = self.vec_bitcast(block, *v, words_t)?;
                let words: Vec<Value<'c, 'c>> = (0..4)
                    .map(|i| self.vec_extract(block, w, &[i], self.i32_t))
                    .collect::<Result<_>>()?;
                let seed = self.zero_scalar(block, self.i32_t)?;
                let half = self.kq_byte_sum(block, &words, seed)?;
                let as_f = self.push(block, arith::bitcast(half, self.f32_t, self.loc))?;
                let other = self.shfl_xor(block, as_f, 1)?;
                let other = self.push(block, arith::bitcast(other, self.i32_t, self.loc))?;
                let sum = self.push(block, arith::addi(half, other, self.loc))?;
                let group = self.divui(block, *chunk, group_w)?;
                block.append_operation(memref::store(sum, as_st.mem, &[*row, group], self.loc));
            }
        }
        let dv = regs[pieces + 1];
        let decode = Stage {
            lanes,
            w_st,
            sw_st,
            mw_st,
            tabs,
            dv,
            block_half,
        };
        self.qgemm_fmt_decode(block, fmt, &decode, &regs[pieces + 2..])
    }
}

/// A stage's activation and weight tiles and their scale planes.
type StageTiles<'a, 'c> = (&'a MemVal<'c>, &'a MemVal<'c>, &'a MemVal<'c>, &'a MemVal<'c>);

/// The intrinsic's global operands: the activation, its scales and, for a
/// format with a minimum, its (scale, sum) pairs if given; the weight and
/// its block scales.
#[derive(Clone, Copy)]
struct QgOperands<'a, 'c> {
    aq: &'a MemVal<'c>,
    asc: &'a MemVal<'c>,
    pairs: Option<&'a MemVal<'c>>,
    qb: &'a MemVal<'c>,
    d: &'a MemVal<'c>,
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
    /// Which half of a 256-element block the stage is, when the minimums
    /// are the f16 plane of both halves.
    pub(super) block_half: Option<Value<'c, 'c>>,
}

impl<'c> Codegen<'c> {
    /// Writes the thread's `l`-th octet, two i32 words, where the tile keeps
    /// elements `32 g + 8 l ..+ 8` of column `j`. That is chunk `2 g + l / 2`,
    /// permuted by the row's phase, in its low or high half.
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

    /// Writes the thread's group minimum, `dmin * m`, for a format that
    /// subtracts one: into the f32 plane, or as `-(dmin * m)` in f16 at
    /// `[column, 2 group + half]` when the minimum term is the tensor op's.
    pub(super) fn qgemm_put_min(
        &mut self,
        block: &Block<'c>,
        stage: &Stage<'_, 'c>,
        value: Value<'c, 'c>,
    ) -> Result<()> {
        let Some(mw_st) = stage.mw_st else {
            bail!("a minimum for a format that has none");
        };
        let (g, j) = (stage.lanes.g, stage.lanes.j);
        match stage.block_half {
            Some(half) => {
                let neg = self.push(block, arith::negf(value, self.loc))?;
                let h = self.truncf(block, neg, self.f16_t)?;
                let two = self.const_index(block, 2)?;
                let col = self.muli(block, g, two)?;
                let col = self.addi(block, col, half)?;
                block.append_operation(memref::store(h, mw_st.mem, &[j, col], self.loc));
            }
            None => {
                block.append_operation(memref::store(value, mw_st.mem, &[g, j], self.loc));
            }
        }
        Ok(())
    }

    /// Writes `f16(scale * sum)` of the thread's two pairs, `pairs` as they
    /// came in, as its row's product word for block half `half` (see
    /// `PAIR_PRODUCTS`).
    fn qgemm_put_products(
        &mut self,
        block: &Block<'c>,
        lanes: &Lanes<'c>,
        pairs: Value<'c, 'c>,
        sa_st: &MemVal<'c>,
        half: Value<'c, 'c>,
    ) -> Result<()> {
        let (f16_t, f32_t) = (self.f16_t, self.f32_t);
        let product = |cg: &mut Self, i: i64| -> Result<Value<'c, 'c>> {
            let scale = cg.vec_extract(block, pairs, &[2 * i], f32_t)?;
            let sum = cg.vec_extract(block, pairs, &[2 * i + 1], f32_t)?;
            let p = cg.push(block, arith::mulf(scale, sum, cg.loc))?;
            cg.truncf(block, p, f16_t)
        };
        let (h0, h1) = (product(self, 0)?, product(self, 1)?);
        let word = self.vec_broadcast(block, h0, Type::vector(&[2], f16_t))?;
        let word = self.vec_insert(block, h1, word, &[1])?;
        let word = self.vec_bitcast(block, word, Type::vector(&[1], f32_t))?;
        let two = self.const_index(block, 2)?;
        let first = self.const_index(block, PAIR_PRODUCTS)?;
        let col = self.muli(block, half, two)?;
        let col = self.addi(block, col, lanes.sa_half)?;
        let col = self.addi(block, col, first)?;
        self.vec_store_al(block, word, sa_st.mem, &[lanes.sa_row, col], 4)
    }

    /// Writes the thread's group scale, or the `h`-th of its `per_group`.
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
