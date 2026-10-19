# Not a parse error: the build reports it, at the attribute.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.unroll(4)
def copy(X: ph.tensor[ph.f32][N], Y: ph.tensor[ph.f32][N]):
    Y[ph.ctx.thread] = X[ph.ctx.thread]
