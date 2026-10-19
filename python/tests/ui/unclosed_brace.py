from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(X: ph.tensor[ph.f32][N], Y: ph.tensor[ph.f32][N]):
    i: ph.let = ph.ctx.thread
    if i < N:
        Y[i = X[i]
