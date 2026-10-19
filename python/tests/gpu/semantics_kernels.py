"""The semantics table's rows as kernels whose operands arrive at run time,
for `semantics.py` to run on the card."""

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def division(I: ph.tensor[ph.i32][4], a: ph.i32, b: ph.i32):
    I[0] = a // b
    I[1] = a % b


@ph.kernel
def remainder(X: ph.tensor[ph.f32][4], a: ph.f32, b: ph.f32):
    X[0] = a % b


@ph.kernel
def wrap(J: ph.tensor[ph.i64][4], n: ph.i32):
    J[0] = 2147483647 + n


@ph.kernel
def widened(D: ph.tensor[ph.f64][4]):
    D[0] = 0.1


@ph.kernel
@ph.launch(256)
def chained(I: ph.tensor[ph.i32][N], lo: ph.i32, hi: ph.i32):
    i: ph.let = ph.ctx.thread
    I[i] = ph.where(lo < i - 1 < hi, 1, 0)


@ph.fn(uses=("s",))
def scaled(x):
    return x * s


@ph.kernel
def free_name(I: ph.tensor[ph.i32][4], s: ph.i32):
    I[0] = scaled(3)
