from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(X: ph.tensor[ph.f32][N], Y: ph.tensor[ph.f32][N]):
    Y[ph.ctx.thread] = ph.exp(X[ph.ctx.thread] * 2.0
