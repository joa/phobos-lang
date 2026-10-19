# From website/src/content/docs/docs/ctx/collectives.mdx.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.launch(128)
def k(A: ph.tensor[ph.f32][1, 128], O: ph.tensor[ph.f32][1, 1]):
    x: ph.let = A[0, ph.ctx.thread]
    r: ph.var = 0.0
    if ph.ctx.lane == 0:
        r = ph.sum(x, over=ph.ctx.lane)
    if ph.ctx.thread == 0:
        O[0, 0] = r
