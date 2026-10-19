"""Triton's tutorial 06, fused attention forward and backward, in Phobos:
FlashAttention-2 in fp16 with fp32 softmax state and accumulators.

The batch and the heads are flattened into the rows, `[Z H N_CTX, D]`, and
a block's second grid index picks the head. The forward pass keeps each
row's log-sum-exp for the backward pass, which recomputes the
probabilities from it: one kernel for `delta = rowsum(dO * O)`, one for
`dK` and `dV` a block of keys at a time over the queries, and one for
`dQ` a block of queries at a time over the keys. Causal attention masks a
key past the query and stops the walk at the diagonal.

The check is the tutorial's `test_op` at Z 1, H 2, N_CTX 1024, HEAD_DIM 64
and `sm_scale` 0.5, causal and not, forward and backward: the output and
the three gradients against torch, `assert_close` with atol 1e-2 and rtol 0.
"""

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.tensorcore
@ph.launch(128)
@ph.autotune(D=[64], BR=[32], BC=[32])
@ph.aligned(R="BR")
def attn_fwd(Q: ph.tensor[ph.f16][R, D], K: ph.tensor[ph.f16][R, D], V: ph.tensor[ph.f16][R, D],
             O: ph.tensor[ph.f16][R, D], L: ph.tensor[ph.f32][R, 1], n_ctx: ph.i32, scale: ph.f32, causal: ph.i32):
    base: ph.let = ph.ctx.block.y * n_ctx
    q0: ph.let = ph.ctx.block.x * BR
    q: ph.let = Q[base + q0:+BR, :]
    acc: ph.var[ph.tile[ph.f32][BR, D]] = 0.0
    m: ph.var[ph.tile[ph.f32][BR, 1]] = -1.0e30
    l: ph.var[ph.tile[ph.f32][BR, 1]] = 0.0
    end: ph.let = ph.where(causal == 1, q0 + BR, n_ctx)
    for kt in range(0, end, BC):
        k: ph.let = K[base + kt:+BC, :]
        v: ph.let = V[base + kt:+BC, :]
        s: ph.var[ph.tile[ph.f32][BR, BC]] = ph.dot_t(q, k)
        s = s * scale
        if causal == 1:
            s = ph.where(kt + ph.iota(1) <= q0 + ph.iota(0), s, -1.0e30)
        ph.online_softmax(acc, m, l, s, v)
    acc = acc / l
    O[base + q0:+BR, :] = acc
    L[base + q0:+BR, 0:+1] = m + ph.log(l)


@ph.kernel
@ph.autotune(D=[64], BR=[64])
def attn_bwd_delta(O: ph.tensor[ph.f16][R, D], DO: ph.tensor[ph.f16][R, D], Delta: ph.tensor[ph.f32][R, 1]):
    r0: ph.let = ph.ctx.block.x * BR
    p: ph.var[ph.tile[ph.f32][BR, D]] = ph.f32(O[r0:+BR, :]) * ph.f32(DO[r0:+BR, :])
    Delta[r0:+BR, 0:+1] = ph.rowsum(p)


@ph.kernel
@ph.tensorcore
@ph.launch(128)
@ph.autotune(D=[64], BR=[32], BC=[32])
def attn_bwd_dkdv(Q: ph.tensor[ph.f16][R, D], K: ph.tensor[ph.f16][R, D], V: ph.tensor[ph.f16][R, D],
                  DO: ph.tensor[ph.f16][R, D], L: ph.tensor[ph.f32][R, 1], Delta: ph.tensor[ph.f32][R, 1],
                  DK: ph.tensor[ph.f16][R, D], DV: ph.tensor[ph.f16][R, D], n_ctx: ph.i32, scale: ph.f32, causal: ph.i32):
    base: ph.let = ph.ctx.block.y * n_ctx
    k0: ph.let = ph.ctx.block.x * BC
    k: ph.let = K[base + k0:+BC, :]
    v: ph.let = V[base + k0:+BC, :]
    dk: ph.var[ph.tile[ph.f32][BC, D]] = 0.0
    dv: ph.var[ph.tile[ph.f32][BC, D]] = 0.0
    start: ph.let = ph.where(causal == 1, k0 // BR * BR, 0)
    for qt in range(start, n_ctx, BR):
        q: ph.let = Q[base + qt:+BR, :]
        do: ph.let = DO[base + qt:+BR, :]
        s: ph.var[ph.tile[ph.f32][BR, BC]] = ph.dot_t(q, k)
        p: ph.var[ph.tile[ph.f32][BR, BC]] = ph.exp(s * scale - L[base + qt:+BR, 0:+1])
        if causal == 1:
            p = ph.where(k0 + ph.iota(1) <= qt + ph.iota(0), p, 0.0)
        pt: ph.var[ph.tile[ph.f32][BC, BR]] = ph.transpose(p)
        dv += ph.dot(pt, do)
        dp: ph.var[ph.tile[ph.f32][BR, BC]] = ph.dot_t(do, v)
        ds: ph.var[ph.tile[ph.f32][BR, BC]] = p * (dp - Delta[base + qt:+BR, 0:+1])
        dst: ph.var[ph.tile[ph.f32][BC, BR]] = ph.transpose(ds)
        dk += ph.dot(dst, q)
    DK[base + k0:+BC, :] = dk * scale
    DV[base + k0:+BC, :] = dv


@ph.kernel
@ph.tensorcore
@ph.launch(128)
@ph.autotune(D=[64], BR=[32], BC=[32])
def attn_bwd_dq(Q: ph.tensor[ph.f16][R, D], K: ph.tensor[ph.f16][R, D], V: ph.tensor[ph.f16][R, D],
                DO: ph.tensor[ph.f16][R, D], L: ph.tensor[ph.f32][R, 1], Delta: ph.tensor[ph.f32][R, 1],
                DQ: ph.tensor[ph.f16][R, D], n_ctx: ph.i32, scale: ph.f32, causal: ph.i32):
    base: ph.let = ph.ctx.block.y * n_ctx
    q0: ph.let = ph.ctx.block.x * BR
    q: ph.let = Q[base + q0:+BR, :]
    do: ph.let = DO[base + q0:+BR, :]
    dq: ph.var[ph.tile[ph.f32][BR, D]] = 0.0
    end: ph.let = ph.where(causal == 1, q0 + BR, n_ctx)
    for kt in range(0, end, BC):
        k: ph.let = K[base + kt:+BC, :]
        v: ph.let = V[base + kt:+BC, :]
        s: ph.var[ph.tile[ph.f32][BR, BC]] = ph.dot_t(q, k)
        p: ph.var[ph.tile[ph.f32][BR, BC]] = ph.exp(s * scale - L[base + q0:+BR, 0:+1])
        if causal == 1:
            p = ph.where(kt + ph.iota(1) <= q0 + ph.iota(0), p, 0.0)
        dp: ph.var[ph.tile[ph.f32][BR, BC]] = ph.dot_t(do, v)
        ds: ph.var[ph.tile[ph.f32][BR, BC]] = p * (dp - Delta[base + q0:+BR, 0:+1])
        dq += ph.dot(ds, k)
    DQ[base + q0:+BR, :] = dq * scale


def run(torch, q, k, v, causal, sm_scale, dout):
    z, h, n, d = q.shape
    rows = z * h * n
    flat = lambda t: t.reshape(rows, d).contiguous()  # noqa: E731
    qf, kf, vf, dof = flat(q), flat(k), flat(v), flat(dout)
    o = torch.empty_like(qf)
    lse = torch.empty((rows, 1), dtype=torch.float32, device="cuda")
    attn_fwd[lambda meta: (n // meta["BR"], z * h)](qf, kf, vf, o, lse, n, sm_scale, int(causal))
    delta = torch.empty_like(lse)
    attn_bwd_delta[lambda meta: (rows // meta["BR"],)](o, dof, delta)
    dq, dk, dv = torch.empty_like(qf), torch.empty_like(kf), torch.empty_like(vf)
    attn_bwd_dkdv[lambda meta: (n // meta["BC"], z * h)](qf, kf, vf, dof, lse, delta, dk, dv, n, sm_scale, int(causal))
    attn_bwd_dq[lambda meta: (n // meta["BR"], z * h)](qf, kf, vf, dof, lse, delta, dq, n, sm_scale, int(causal))
    torch.cuda.synchronize()
    shape = (z, h, n, d)
    return o.reshape(shape), dq.reshape(shape), dk.reshape(shape), dv.reshape(shape)


def check(torch):
    results = []
    for causal in (False, True):
        torch.manual_seed(20)
        z, h, n, d, sm_scale = 1, 2, 1024, 64, 0.5
        q = torch.empty((z, h, n, d), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
        k = torch.empty((z, h, n, d), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
        v = torch.empty((z, h, n, d), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
        mask = torch.tril(torch.ones((n, n), device="cuda"))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if causal:
            p[:, :, mask == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).half()
        ref_out = torch.matmul(p, v).half()
        dout = torch.randn_like(q)
        ref_out.backward(dout)
        refs = (ref_out, q.grad.clone(), k.grad.clone(), v.grad.clone())
        got = run(torch, q.detach(), k.detach(), v.detach(), causal, sm_scale, dout)
        for name, g, r in zip(("out", "dq", "dk", "dv"), got, refs):
            try:
                torch.testing.assert_close(g, r, atol=1e-2, rtol=0)
                ok = True
            except AssertionError:
                ok = False
            results.append((ok, f"{'causal' if causal else 'full'} {name} max diff {torch.max(torch.abs(g.float() - r.float())).item():.3g}"))
    return all(ok for ok, _ in results), ", ".join(d for _, d in results) + " (assert_close atol 1e-2, rtol 0)"
