# `min`, `max` and `pow` become operators as they parse, with two
# arguments; any other count is an error, not an unknown function.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(A: ph.tensor[ph.f32][N], B: ph.tensor[ph.f32][N]):
    x: ph.let = A[ph.ctx.thread]
    B[ph.ctx.thread] = ph.pow(x)
