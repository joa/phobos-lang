from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(X: ph.tensor[ph.f32][N, 64], Y: ph.tensor[ph.f32][N, 64]):
    x: ph.var = X[0:+16, :]
    ph.rms_norm(x, 0.00001)
    Y[0:+16, :] = x
