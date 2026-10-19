"""Triton's tutorial 01, vector addition, in Phobos: `out = x + y`.

A block adds `BLOCK` elements; the last block's slice past `N` is masked
by the compiler, which Triton's `mask = offsets < n_elements` does by
hand. The check is the tutorial's: the largest difference from torch,
which is 0.
"""

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.autotune(BLOCK=[1024])
def add(x: ph.tensor[ph.f32][N], y: ph.tensor[ph.f32][N], out: ph.tensor[ph.f32][N]):
    base: ph.let = ph.ctx.block.x * BLOCK
    out[base:+BLOCK] = x[base:+BLOCK] + y[base:+BLOCK]


def check(torch):
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    out = torch.empty_like(x)
    add[lambda meta: (-(-meta["N"] // meta["BLOCK"]),)](x, y, out)
    torch.cuda.synchronize()
    worst = torch.max(torch.abs(x + y - out)).item()
    return worst == 0.0, f"max |torch - phobos| = {worst}"
