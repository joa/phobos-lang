# From website/src/content/docs/docs/attributes/swizzle.mdx.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.launch(128)
@ph.swizzle
@ph.aligned(M=64, N=64, K=128)
def proj(A: ph.tensor[ph.i8][M, K], AS: ph.tensor[ph.f32][M, KB], W: ph.tensor[ph.i8][N, K], WS: ph.tensor[ph.f32][KB, N], C: ph.tensor[ph.f32][M, N], E: ph.tensor[ph.i8][N, K]):
    pm: ph.let = ph.ctx.block.x
    pn: ph.let = ph.ctx.block.y
    w: ph.var[ph.tile[ph.i8][64, 128]]
    acc: ph.var[ph.tile[ph.f32][64, 64]] = 0.0
    for kt in range(0, K, 128):
        w = W[pn * 64:+64, kt:+128]
        ph.barrier()
        a0: ph.let = A[pm * 64:+64, kt:+32]
        w0: ph.let = w[0:+64, 0:+32]
        da: ph.let = AS[pm * 64:+64, kt // 32:+1]
        ws: ph.let = WS[kt // 32:+1, pn * 64:+64]
        acc += ph.f32(ph.dot_t(a0, w0)) * ws * da
        E[pn * 64:+64, kt:+128] = w
    C[pm * 64:+64, pn * 64:+64] = acc
