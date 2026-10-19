from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.autotune(TILE_M=[32, 256], TILE_N=[32, 256], TILE_K=[4, 32], U=[1, 8], T=[256, 128], IB=[32, 64])
@ph.launch("T")
@ph.index_bits("IB")
@ph.pipeline
@ph.aligned(M="TILE_M", N="TILE_N", K="TILE_K")
def gemm(A: ph.tensor[ph.f32][M, K], B: ph.tensor[ph.f32][K, N], C: ph.tensor[ph.f32][M, N], alpha: ph.f32, beta: ph.f32):
    pm: ph.let = ph.ctx.block.x
    pn: ph.let = ph.ctx.block.y
    acc: ph.var[ph.tile[ph.f32][TILE_M, TILE_N]] = 0.0
    for kt in ph.unroll(U, range(0, K, TILE_K)):
        a: ph.var = A[pm * TILE_M:+TILE_M, kt:+TILE_K]
        b: ph.var = B[kt:+TILE_K, pn * TILE_N:+TILE_N]
        acc += ph.dot(a, b)
    if beta == 0.0:
        C[pm * TILE_M:+TILE_M, pn * TILE_N:+TILE_N] = alpha * acc
    else:
        c_old: ph.let = C[pm * TILE_M:+TILE_M, pn * TILE_N:+TILE_N]
        C[pm * TILE_M:+TILE_M, pn * TILE_N:+TILE_N] = alpha * acc + beta * c_old
