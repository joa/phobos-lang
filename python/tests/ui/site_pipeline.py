# From website/src/content/docs/docs/attributes/pipeline.mdx.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.pipeline
def flat(A: ph.tensor[ph.f32][N], C: ph.tensor[ph.f32][N]):
    i: ph.let = ph.ctx.block.x
    C[i:+1] = A[i:+1] * 2.0
