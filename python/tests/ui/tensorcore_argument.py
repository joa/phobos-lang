from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.tensorcore("wmmma")
@ph.launch(128)
def k(A: ph.tensor[ph.f32][64, 64], B: ph.tensor[ph.f32][64, 64], C: ph.tensor[ph.f32][64, 64]):
    acc: ph.var[ph.tile[ph.f32][64, 64]] = 0.0
    acc += ph.dot(A[0:+64, 0:+64], B[0:+64, 0:+64])
    C[0:+64, 0:+64] = acc
