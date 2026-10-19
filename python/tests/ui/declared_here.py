# An error about a use of a name points back at its declaration: a `let`
# assigned to, and a store of the wrong type.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.launch(128)
def k(A: ph.tensor[ph.f32][N], B: ph.tensor[ph.i32][N]):
    x: ph.let = A[ph.ctx.thread]
    x = 2.0
    m: ph.var = 0
    m = x
    B[ph.ctx.thread] = x
