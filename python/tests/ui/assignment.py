from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(A: ph.tensor[ph.f32][N], B: ph.tensor[ph.f32][N]):
    1 = 2
    A[0][1] = 1.0
    A[0] = (B[0] + 1.0
