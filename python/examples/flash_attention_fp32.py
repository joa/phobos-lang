# FlashAttention-2 in fp32: https://tridao.me/publications/flash2/flash2.pdf

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.cluster(BR=[1024, 4096])
@ph.tensorcore
@ph.launch(128)
@ph.autotune(D=[64], BR=[4, 128], BC=[4, 128])
@ph.aligned(Nq="BR", Nk="BC")
def flash_attention(Q: ph.tensor[ph.f32][Nq, D], K: ph.tensor[ph.f32][Nk, D], V: ph.tensor[ph.f32][Nk, D], O: ph.tensor[ph.f32][Nq, D], scale: ph.f32):
    pid: ph.let = ph.ctx.block.x
    row: ph.let = pid * BR
    q: ph.let = Q[row:+BR, :]
    acc: ph.var[ph.tile[ph.f32][BR, D]] = 0.0
    m: ph.var[ph.tile[ph.f32][BR, 1]] = -1.0e30
    l: ph.var[ph.tile[ph.f32][BR, 1]] = 0.0
    for kt in range(0, Nk, BC):
        k: ph.let = K[kt:+BC, :]
        v: ph.let = V[kt:+BC, :]
        s: ph.var[ph.tile[ph.f32][BR, BC]] = ph.dot_t(q, k)
        s = s * scale
        mnew: ph.var[ph.tile[ph.f32][BR, 1]] = ph.rowmax(s)
        mnew = ph.tmax(m, mnew)
        s = ph.exp(s - mnew)
        corr: ph.var[ph.tile[ph.f32][BR, 1]] = ph.exp(m - mnew)
        l = l * corr
        l += ph.rowsum(s)
        acc = acc * corr
        acc += ph.dot(s, v)
        m = mnew
    acc = acc / l
    O[row:+BR, :] = acc
