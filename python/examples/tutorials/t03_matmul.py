"""Triton's tutorial 03, matrix multiplication, in Phobos: `C = A B` with
fp16 operands on the tensor cores and an fp32 accumulator, narrowed to fp16
once, on the store, as the tutorial's `accumulator.to(tl.float16)` does. The
check is the tutorial's: 512 x 512 inputs in [-0.5, 0.5) against
`torch.matmul`, `allclose` with atol 1e-2 and rtol 0.
"""

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.autotune(TILE_M=[64], TILE_N=[64], TILE_K=[32])
@ph.launch(256)
@ph.tensorcore
@ph.aligned(M="TILE_M", N="TILE_N", K="TILE_K")
def matmul(A: ph.tensor[ph.f16][M, K], B: ph.tensor[ph.f16][K, N], C: ph.tensor[ph.f16][M, N]):
    pm: ph.let = ph.ctx.block.x
    pn: ph.let = ph.ctx.block.y
    acc: ph.var[ph.tile[ph.f32][TILE_M, TILE_N]] = 0.0
    for kt in range(0, K, TILE_K):
        a: ph.var = A[pm * TILE_M:+TILE_M, kt:+TILE_K]
        b: ph.var = B[kt:+TILE_K, pn * TILE_N:+TILE_N]
        acc += ph.dot(a, b)
    C[pm * TILE_M:+TILE_M, pn * TILE_N:+TILE_N] = acc


def check(torch):
    torch.manual_seed(0)
    a = torch.rand((512, 512), device="cuda", dtype=torch.float16) - 0.5
    b = torch.rand((512, 512), device="cuda", dtype=torch.float16) - 0.5
    c = torch.empty((512, 512), device="cuda", dtype=torch.float16)
    matmul[lambda meta: (meta["M"] // meta["TILE_M"], meta["N"] // meta["TILE_N"])](a, b, c)
    torch.cuda.synchronize()
    want = torch.matmul(a, b)
    ok = torch.allclose(c, want, atol=1e-2, rtol=0)
    return ok, f"max |torch - phobos| = {torch.max(torch.abs(c.float() - want.float())).item():.3g}, allclose (atol 1e-2): {ok}"
