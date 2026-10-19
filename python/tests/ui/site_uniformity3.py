# From website/src/content/docs/docs/ctx/uniformity.mdx.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.launch(256)
def k(A: ph.tensor[ph.f32][1, 32], O: ph.tensor[ph.f32][1, 32]):
    t: ph.var[ph.tile[ph.f32][1, 32]] = A[0:+1, 0:+32]
    t[0, ph.ctx.lane] = 1.0
    O[0:+1, 0:+32] = t
