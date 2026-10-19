from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(X: ph.tensor[ph.f32][1, 256]):
    H: ph.var[ph.tensor[ph.f32][8, 32]]
    H[0:+1, 0:+32] = X[0:+1, 0:+32]
