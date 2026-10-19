from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(Y: ph.tensor[ph.i32][N]):
    Y[ph.ctx.thread] = ph.ctx.lanez
    ctx: ph.let = 1
