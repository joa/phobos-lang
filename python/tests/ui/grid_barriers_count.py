from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.launch(256)
@ph.persistent("BLOCKS")
@ph.grid_barriers(0)
@ph.autotune(BLOCKS=[8])
def k(X: ph.tensor[ph.f32][1, 256], Y: ph.tensor[ph.f32][1, 256]):
    for i in range(0, 4):
        u: ph.let = ph.ctx.block.x + i * ph.ctx.blocks.x
        if u < 32:
            Y[0:+1, u * 8:+8] = X[0:+1, u * 8:+8]
    for i in range(0, 4):
        u: ph.let = ph.ctx.block.x + i * ph.ctx.blocks.x
        if u < 32:
            X[0:+1, u * 8:+8] = Y[0:+1, (u + 1) % 32 * 8:+8]
