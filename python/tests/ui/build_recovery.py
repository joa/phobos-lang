# Three mistakes the build finds, each once: `y` uses `x`, whose
# declaration failed, and says nothing more.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(A: ph.tensor[ph.f32][N], B: ph.tensor[ph.f32][N]):
    x: ph.let = nope + 1
    y: ph.let = x * 2
    z: ph.let = ph.wobble(A)
    for i in range(0, N):
        B[i] = A
    B[ph.ctx.thread] = A[ph.ctx.thread] + y
