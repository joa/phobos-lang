# A generated tile is computed afresh at each read, so the function that
# computes it may read memory but not write it.

from __future__ import annotations

import phobos.lang as ph


@ph.fn(uses=("C", "W"))
def wq(j, q):
    v: ph.let = ph.word(W[j, 4 * q:+4], 0)
    C[0, 0] = 1.0
    v


@ph.kernel
@ph.launch(128)
def k(A: ph.tensor[ph.i8][128, 32], AS: ph.tensor[ph.f32][128, 1], W: ph.tensor[ph.i8][64, 32], WS: ph.tensor[ph.f32][1, 64], C: ph.tensor[ph.f32][128, 64]):
    acc: ph.var[ph.tile[ph.f32][128, 64]] = 0.0
    for kt in range(0, 32, 32):
        a: ph.let = A[0:+128, 0:+32]
        w: ph.let[ph.tile[ph.i8][64, 32]] = ph.generate(wq)
        ws: ph.let = WS[0:+1, 0:+64]
        da: ph.let = AS[0:+128, 0:+1]
        acc += ph.f32(ph.dot_t(a, w)) * ws * da
    C[0:+128, 0:+64] = acc
