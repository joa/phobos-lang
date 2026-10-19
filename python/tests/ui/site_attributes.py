# From website/src/content/docs/docs/attributes/index.mdx.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.unroll
@ph.autotune(BLOCK=[1024])
def copy(x: ph.tensor[ph.f32][N], y: ph.tensor[ph.f32][N]):
    base: ph.let = ph.ctx.block.x * BLOCK
    y[base:+BLOCK] = x[base:+BLOCK]
