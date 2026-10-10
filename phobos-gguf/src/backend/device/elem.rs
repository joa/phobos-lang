// Strided copies and the pointwise kernels.

use super::*;

impl DeviceBackend {
    /// The quantized SwiGLU of two dense `[rows, width]` buffers without the
    /// f32 result, at the tile `swiglu_q_rows` takes.
    pub(super) fn swiglu_q_unkept(&self, gate: Buf, up: Buf, rows: usize, width: usize) -> Result<Option<QAct>> {
        let len = rows * width;
        let tile = if len.is_multiple_of(ELEM_TILE_WIDE) { ELEM_TILE_WIDE } else { ELEM_TILE };
        if !len.is_multiple_of(tile) {
            return Ok(None);
        }
        let slot = self.act_slot_transient(rows, width)?;
        Ok(Some(self.swiglu_q_into(slot, (gate, 0), (up, 0), None, len, tile)?))
    }

    /// The quantizing SwiGLU over a flat span of `len`, into `slot`, `tile`
    /// elements a CTA. `out` gets the f32 result too, if given.
    pub(super) fn swiglu_q_into(
        &self,
        slot: (QAct, u64, u64),
        (gate, gate_at): (Buf, usize),
        (up, up_at): (Buf, usize),
        out: Option<Buf>,
        len: usize,
        tile: usize,
    ) -> Result<QAct> {
        if let Some(out) = out {
            self.check_distinct("swiglu_q", out, &[gate, up]);
        }
        ensure!(
            len.is_multiple_of(tile),
            "a quantizing SwiGLU needs a length ({len}) that is a multiple of {tile}"
        );
        let blocks = tile / RMS_LANE;
        let (act, qa_ptr, das_ptr) = slot;
        let rb = (len / RMS_LANE) as i64;
        let lane = RMS_LANE as i64;
        self.with_kernel(
            &self.gated_swiglu,
            (blocks, out.is_some()),
            "swiglu_q",
            || swiglu_q_src(blocks, out.is_some()),
            |module| {
                let mut operands = vec![(self.ptr(gate, gate_at)?, [rb, lane]), (self.ptr(up, up_at)?, [rb, lane])];
                if let Some(out) = out {
                    operands.push((self.ptr(out, 0)?, [rb, lane]));
                }
                operands.extend([(qa_ptr, [rb, lane]), (das_ptr, [rb, 1])]);
                self.launch(module, "swiglu_q", &operands, ((len / tile) as u32, 1, 1))
            },
        )?;
        Ok(act)
    }

    /// A strided block copy between two planes already resolved to a pointer
    /// and a pitch, converting if the two sides differ in width.
    pub(super) fn strided(
        &self,
        kind: Strided,
        src: (u64, usize),
        dst: (u64, usize),
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let aligned = src.1.is_multiple_of(width) && dst.1.is_multiple_of(width);
        self.with_kernel(
            &self.splits,
            (kind, width, aligned),
            kind.kernel(),
            || copy_2d_src(kind, width, aligned),
            |module| {
                self.launch(
                    module,
                    kind.kernel(),
                    &[
                        (src.0, [rows as i64, src.1 as i64]),
                        (dst.0, [rows as i64, dst.1 as i64]),
                    ],
                    (rows as u32, 1, 1),
                )
            },
        )
    }

    /// [`DeviceBackend::strided`], but two independent plane pairs in one
    /// launch; see [`Backend::store_2d_pair`].
    pub(super) fn strided_pair(
        &self,
        a_src: (u64, usize),
        a_dst: (u64, usize),
        b_src: (u64, usize),
        b_dst: (u64, usize),
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let aligned = [a_src, a_dst, b_src, b_dst]
            .iter()
            .all(|&(_, pitch)| pitch.is_multiple_of(width));
        self.with_kernel(
            &self.store_pairs,
            (width, aligned),
            "store_2d_pair",
            || store_2d_pair_src(width, aligned),
            |module| {
                self.launch(
                    module,
                    "store_2d_pair",
                    &[
                        (a_src.0, [rows as i64, a_src.1 as i64]),
                        (b_src.0, [rows as i64, b_src.1 as i64]),
                        (a_dst.0, [rows as i64, a_dst.1 as i64]),
                        (b_dst.0, [rows as i64, b_dst.1 as i64]),
                    ],
                    (rows as u32, 1, 1),
                )
            },
        )
    }

    /// A pointwise launch over `len` elements of one or more flat buffers.
    /// Long runs use the wide kernel, see [`ELEM_TILE_WIDE`].
    pub(super) fn pointwise(
        &self,
        name: &'static str,
        buffers: &[(Buf, usize)],
        len: usize,
    ) -> Result<()> {
        let ptrs = buffers.iter().map(|&(buf, offset)| self.ptr(buf, offset)).collect::<Result<Vec<_>>>()?;
        self.pointwise_raw(name, &ptrs, len)
    }

    /// [`DeviceBackend::pointwise`] over raw device addresses, such as a
    /// mapped host buffer's.
    pub(super) fn pointwise_raw(&self, name: &'static str, ptrs: &[u64], len: usize) -> Result<()> {
        let operands: Vec<_> = ptrs.iter().map(|&ptr| (ptr, [1i64, len as i64])).collect();
        let (module, tile) = if len >= WIDE_FLOOR {
            (&self.pointwise_wide, ELEM_TILE_WIDE)
        } else {
            (&self.pointwise, ELEM_TILE)
        };
        self.launch(module, name, &operands, (len.div_ceil(tile) as u32, 1, 1))
    }
}
