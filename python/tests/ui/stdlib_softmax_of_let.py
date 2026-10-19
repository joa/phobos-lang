# A library function that writes its argument needs a `var`; the error
# points at the call, not into the library.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(X: ph.tensor[ph.f32][N, 64], Y: ph.tensor[ph.f32][N, 64]):
    x: ph.let = X[0:+16, :]
    ph.softmax(x)
    Y[0:+16, :] = x
