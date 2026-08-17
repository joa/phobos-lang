// Strided copies and the pointwise kernels.

use super::*;

impl DeviceBackend {
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
    #[allow(clippy::too_many_arguments)]
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

    /// A pointwise launch over `len` elements of one or more flat buffers. Past
    /// enough of them to fill the card the wide kernel takes over; see
    /// [`ELEM_TILE_WIDE`].
    pub(super) fn pointwise(
        &self,
        name: &'static str,
        buffers: &[(Buf, usize)],
        len: usize,
    ) -> Result<()> {
        let mut operands = Vec::with_capacity(buffers.len());
        for &(buf, offset) in buffers {
            operands.push((self.ptr(buf, offset)?, [1i64, len as i64]));
        }
        let (module, tile) = if len >= WIDE_FLOOR {
            (&self.pointwise_wide, ELEM_TILE_WIDE)
        } else {
            (&self.pointwise, ELEM_TILE)
        };
        self.launch(module, name, &operands, (len.div_ceil(tile) as u32, 1, 1))
    }
}
