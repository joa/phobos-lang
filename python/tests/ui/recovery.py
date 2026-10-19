from __future__ import annotations

import phobos.lang as ph


# Three mistakes that do not depend on each other; Python stops at the first.
@ph.kernel
@ph.launch(64)
def scale(X: ph.tensor[ph.f32][1, 64], Y: ph.tensor[ph.f32][1, 64]):
    s: ph.let = 2.0 * * 3.0
    Y[0, ph.ctx.thread] = X[0, ph.ctx.thread] * s s


@ph.kernel
def copy(X: ph.tensor[ph.f32][N], Y: ph.tensor[ph.f32][N]):
    i: ph.let = ph.ctx.block.x * ph.ctx.threads + ph.ctx.thread
    if i < N:
        Y[i] = X[i] +


@ph.kernel
def fill(Y ph.tensor[ph.f32][N]):
    Y[ph.ctx.thread] = 0.0
