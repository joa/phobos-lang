# A name one typo away from a known one is offered in its place.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.launch(128)
def k(A: ph.tensor[ph.f32][N], B: ph.tensor[ph.f32][N]):
    scale: ph.let = 2.0
    x: ph.let = A[ph.ctx.thread] * scael
    B[ph.ctx.thread] = ph.sqtr(x)
