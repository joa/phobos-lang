from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(Y: ph.tensor[ph.f32][N]):
    for i in steps(0, N):
        Y[i] = 0.0
    for i in ph.twice(2, range(0, N)):
        Y[i] = 1.0
