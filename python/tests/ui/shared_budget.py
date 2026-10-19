# A tile larger than the 48 KB a block may hold statically.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def big(X: ph.tensor[ph.f32][N, 256], Y: ph.tensor[ph.f32][N, 256]):
    r: ph.let = ph.ctx.block.x * 64
    x: ph.var[ph.tile[ph.f32][64, 256]] = X[r:+64, :]
    Y[r:+64, :] = x * 2.0
