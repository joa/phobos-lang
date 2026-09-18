// The Hadamard transform ahead of a folded weight; the kernel is
// `kernels/hadamard.rs`.

use super::*;

impl DeviceBackend {
    pub(super) fn hadamard_rows(
        &self,
        x: Buf,
        rows: usize,
        width: usize,
        signs: Buf,
        perm: Option<HeadPerm>,
        out: Buf,
    ) -> Result<()> {
        self.check_distinct("hadamard", out, &[x, signs]);
        ensure!(
            width.is_multiple_of(HADAMARD_BLOCK),
            "a {width}-wide row is not a whole number of Hadamard blocks"
        );
        if let Some(p) = perm {
            ensure!(
                p.head_dim.is_multiple_of(HADAMARD_SIDE)
                    && HADAMARD_BLOCK.is_multiple_of(p.head_dim)
                    && p.head_dim * p.groups * p.repeat == width,
                "{p:?} does not regroup a {width}-wide row in whole tiles"
            );
        }
        let h = self.constant("hadamard.h32", &hadamard_matrix())?;
        let side = HADAMARD_SIDE as i64;
        let tiles = (rows * width / HADAMARD_SIDE) as i64;
        let blocks = (rows * width / HADAMARD_BLOCK) as u32;
        self.with_kernel(
            &self.hadamards,
            (width, perm),
            "hadamard",
            || hadamard_src(width, perm),
            |module| {
                self.launch(
                    module,
                    "hadamard",
                    &[
                        (self.ptr(x, 0)?, [tiles, side]),
                        (self.ptr(signs, 0)?, [(width / HADAMARD_SIDE) as i64, side]),
                        (self.ptr(h, 0)?, [side, side]),
                        (self.ptr(out, 0)?, [tiles, side]),
                    ],
                    (blocks, 1, 1),
                )
            },
        )
    }
}
