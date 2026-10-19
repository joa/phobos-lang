from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def staged(X: ph.tensor[ph.f32][N]):
    i: ph.let = ph.ctx.block.x
    X[i] = X[i] + 1.0
    ph.grid_barrier()
