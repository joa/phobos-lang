# The kernels of one module share one index width.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.autotune(IB=[32, 64])
@ph.index_bits("IB")
def first(X: ph.tensor[ph.f32][N]):
    X[ph.ctx.thread] = 1.0


@ph.kernel
@ph.index_bits(64)
def second(X: ph.tensor[ph.f32][N]):
    X[ph.ctx.thread] = 2.0
