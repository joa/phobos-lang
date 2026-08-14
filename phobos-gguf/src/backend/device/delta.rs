// The chunked delta rule.

use super::*;

impl DeviceBackend {
    /// The chunked gated delta rule, in two passes; see [`delta_wy_src`].
    ///
    /// The packed operands are one row per (position, head), where a chunk's `C`
    /// consecutive positions of one head sit `C` rows apart. Read as one row per
    /// position with the heads side by side, the same memory, a chunk becomes a
    /// column window of consecutive rows and so an ordinary tile. The gates and
    /// the first pass's three `[C, C]` matrices are laid out to match.
    pub(super) fn delta_chunked(
        &self,
        packed: Buf,
        rows: usize,
        heads: usize,
        head_dim: usize,
        state: Buf,
        out: Buf,
    ) -> Result<()> {
        let (span, gates) = (rows * heads * head_dim, rows * heads);
        let (n, width) = (rows as i64, (heads * head_dim) as i64);
        let wy_width = heads * 4 * DELTA_CHUNK;
        let eye = self.identity(DELTA_CHUNK)?;
        let wy = self.alloc(rows * wy_width)?;
        let result = self.with_kernel(
            &self.chunks,
            (heads, head_dim),
            "delta_chunk",
            || {
                let mut source = delta_wy_src(heads, head_dim, DELTA_CHUNK);
                source.push_str(&delta_scan_src(
                    heads,
                    head_dim,
                    DELTA_CHUNK_TN,
                    DELTA_CHUNK,
                ));
                source
            },
            |module| {
                let wy_ptr = self.ptr(wy, 0)?;
                let wy_dims = [n, wy_width as i64];
                self.launch(
                    module,
                    "delta_wy",
                    &[
                        (self.ptr(packed, 0)?, [n, width]),
                        (self.ptr(packed, span)?, [n, width]),
                        (self.ptr(packed, 3 * span)?, [n, heads as i64]),
                        (self.ptr(packed, 3 * span + gates)?, [n, heads as i64]),
                        (eye, [DELTA_CHUNK as i64, DELTA_CHUNK as i64]),
                        (wy_ptr, wy_dims),
                    ],
                    (heads as u32, (rows / DELTA_CHUNK) as u32, 1),
                )?;
                self.launch(
                    module,
                    "delta_scan",
                    &[
                        (self.ptr(packed, 0)?, [n, width]),
                        (self.ptr(packed, span)?, [n, width]),
                        (self.ptr(packed, 2 * span)?, [n, width]),
                        (wy_ptr, wy_dims),
                        (
                            self.ptr(state, 0)?,
                            [(heads * head_dim) as i64, head_dim as i64],
                        ),
                        (self.ptr(out, 0)?, [n, width]),
                    ],
                    (heads as u32, (head_dim / DELTA_CHUNK_TN) as u32, 1),
                )
            },
        );
        self.release(wy);
        result
    }
}
