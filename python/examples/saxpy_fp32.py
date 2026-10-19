from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.autotune(BLOCK=[16, 8192])
def saxpy(x: ph.tensor[ph.f32][N], y: ph.tensor[ph.f32][N], out: ph.tensor[ph.f32][N], alpha: ph.f32):
    base: ph.let = ph.ctx.block.x * BLOCK
    out[base:+BLOCK] = alpha * x[base:+BLOCK] + y[base:+BLOCK]
