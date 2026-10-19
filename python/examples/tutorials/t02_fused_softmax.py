"""Triton's tutorial 02, the fused row softmax, in Phobos.

A block takes `ROWS` rows whole, `BLOCK` columns wide, and leaves the
columns past `n` out of the maximum and the sum. The tutorial's input is
1823 x 781, and a Phobos launch takes extents that are multiples of 4, the
row-pitch ABI: the rows are stored 784 wide and 1824 long, `n` is 781, and
the 1823 x 781 the tutorial compares are compared, with its tolerance,
`torch.allclose`'s defaults.
"""

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.autotune(BLOCK=[1024], ROWS=[4])
def softmax(X: ph.tensor[ph.f32][M, NP], Y: ph.tensor[ph.f32][M, NP], n: ph.i32):
    row: ph.let = ph.ctx.block.x * ROWS
    x: ph.var[ph.tile[ph.f32][ROWS, BLOCK]] = X[row:+ROWS, 0:+BLOCK]
    x = ph.where(ph.iota(1) < n, x, -1.0e30)
    m: ph.var[ph.tile[ph.f32][ROWS, 1]] = ph.rowmax(x)
    x = ph.exp(x - m)
    s: ph.var[ph.tile[ph.f32][ROWS, 1]] = ph.rowsum(x)
    x = x / s
    Y[row:+ROWS, 0:+BLOCK] = x


def check(torch):
    torch.manual_seed(0)
    rows, cols = 1823, 781
    x = torch.randn(rows, cols, device="cuda")
    xp = torch.zeros(rows + 1, cols + 3, device="cuda")
    xp[:rows, :cols] = x
    yp = torch.empty_like(xp)
    softmax[lambda meta: (meta["M"] // meta["ROWS"],)](xp, yp, cols)
    torch.cuda.synchronize()
    got, want = yp[:rows, :cols], torch.softmax(x, axis=1)
    ok = torch.allclose(got, want)
    return ok, f"max |torch - phobos| = {torch.max(torch.abs(got - want)).item():.3g}, allclose (rtol 1e-5, atol 1e-8): {ok}"
