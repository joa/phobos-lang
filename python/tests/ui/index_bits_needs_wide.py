# `@tensorcore` takes mma.sync, which no 32-bit index reaches.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.index_bits(32)
@ph.tensorcore
def scale(X: ph.tensor[ph.f32][N], Y: ph.tensor[ph.f32][N]):
    Y[ph.ctx.thread] = X[ph.ctx.thread] * 2.0
