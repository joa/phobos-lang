from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(Y: ph.tensor[ph.f32][N]):
    Y[ph.ctx.thread] = 1.0 $ 2.0
    big: ph.let = 99999999999999999999
    e: ph.let = 1e
