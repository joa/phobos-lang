"""Triton's tutorial 05, layer norm forward and backward, in Phobos.

The forward pass is the tutorial's: a block takes a row, walks it in
`CB`-column chunks for the mean, again for the variance, and a third time
to normalize, keeping the mean and the reciprocal deviation for the
backward pass. The backward pass computes `dx` a row a block, as the
tutorial does; `dw` and `db` sum over the rows, which the tutorial does
with a lock and a second reduction, and which here is a second kernel
over column chunks walking the rows.

The tutorial's input is 1151 x 8192 fp16; a Phobos launch takes extents
that are multiples of 4, so the rows are stored 1152 long, the last one
zero, which adds nothing to `dw` and `db`. The check is the tutorial's:
`y`, `dx`, `dw` and `db` against torch, `allclose` with atol 1e-2.
"""

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.autotune(CB=[1024])
def ln_fwd(X: ph.tensor[ph.f16][M, N], W: ph.tensor[ph.f16][1, N], B: ph.tensor[ph.f16][1, N],
           Y: ph.tensor[ph.f16][M, N], Mean: ph.tensor[ph.f32][M, 1], Rstd: ph.tensor[ph.f32][M, 1], eps: ph.f32):
    row: ph.let = ph.ctx.block.x
    total: ph.var[ph.tile[ph.f32][1, 1]] = 0.0
    for c in range(0, N, CB):
        total += ph.rowsum(ph.f32(X[row:+1, c:+CB]))
    mu: ph.var[ph.tile[ph.f32][1, 1]] = total / ph.f32(N)
    squares: ph.var[ph.tile[ph.f32][1, 1]] = 0.0
    for c in range(0, N, CB):
        d: ph.var[ph.tile[ph.f32][1, CB]] = ph.f32(X[row:+1, c:+CB]) - mu
        squares += ph.rowsum(d * d)
    r: ph.var[ph.tile[ph.f32][1, 1]] = ph.rsqrt(squares / ph.f32(N) + eps)
    Mean[row:+1, 0:+1] = mu
    Rstd[row:+1, 0:+1] = r
    for c in range(0, N, CB):
        y: ph.var[ph.tile[ph.f32][1, CB]] = (ph.f32(X[row:+1, c:+CB]) - mu) * r
        Y[row:+1, c:+CB] = y * ph.f32(W[0:+1, c:+CB]) + ph.f32(B[0:+1, c:+CB])


@ph.kernel
@ph.autotune(CB=[1024])
def ln_bwd_dx(X: ph.tensor[ph.f16][M, N], W: ph.tensor[ph.f16][1, N], Mean: ph.tensor[ph.f32][M, 1],
              Rstd: ph.tensor[ph.f32][M, 1], DY: ph.tensor[ph.f16][M, N], DX: ph.tensor[ph.f16][M, N]):
    row: ph.let = ph.ctx.block.x
    mu: ph.var[ph.tile[ph.f32][1, 1]] = Mean[row:+1, 0:+1]
    r: ph.var[ph.tile[ph.f32][1, 1]] = Rstd[row:+1, 0:+1]
    a: ph.var[ph.tile[ph.f32][1, 1]] = 0.0
    b: ph.var[ph.tile[ph.f32][1, 1]] = 0.0
    for c in range(0, N, CB):
        xhat: ph.var[ph.tile[ph.f32][1, CB]] = (ph.f32(X[row:+1, c:+CB]) - mu) * r
        wdy: ph.var[ph.tile[ph.f32][1, CB]] = ph.f32(W[0:+1, c:+CB]) * ph.f32(DY[row:+1, c:+CB])
        a += ph.rowsum(xhat * wdy)
        b += ph.rowsum(wdy)
    c1: ph.var[ph.tile[ph.f32][1, 1]] = a / ph.f32(N)
    c2: ph.var[ph.tile[ph.f32][1, 1]] = b / ph.f32(N)
    for c in range(0, N, CB):
        xhat: ph.var[ph.tile[ph.f32][1, CB]] = (ph.f32(X[row:+1, c:+CB]) - mu) * r
        wdy: ph.var[ph.tile[ph.f32][1, CB]] = ph.f32(W[0:+1, c:+CB]) * ph.f32(DY[row:+1, c:+CB])
        DX[row:+1, c:+CB] = (wdy - (xhat * c1 + c2)) * r


@ph.kernel
@ph.autotune(CB=[64], RT=[32])
def ln_bwd_dwdb(X: ph.tensor[ph.f16][M, N], DY: ph.tensor[ph.f16][M, N], Mean: ph.tensor[ph.f32][M, 1],
                Rstd: ph.tensor[ph.f32][M, 1], DW: ph.tensor[ph.f16][1, N], DB: ph.tensor[ph.f16][1, N]):
    col: ph.let = ph.ctx.block.x * CB
    dw: ph.var[ph.tile[ph.f32][1, CB]] = 0.0
    db: ph.var[ph.tile[ph.f32][1, CB]] = 0.0
    for r0 in range(0, M, RT):
        dy: ph.var[ph.tile[ph.f32][RT, CB]] = ph.f32(DY[r0:+RT, col:+CB])
        p: ph.var[ph.tile[ph.f32][RT, CB]] = (ph.f32(X[r0:+RT, col:+CB]) - Mean[r0:+RT, 0:+1]) * Rstd[r0:+RT, 0:+1] * dy
        dw += ph.colsum(p)
        db += ph.colsum(dy)
    DW[0:+1, col:+CB] = dw
    DB[0:+1, col:+CB] = db


def check(torch):
    torch.manual_seed(0)
    rows, cols, eps = 1151, 8192, 1e-5
    dtype = torch.float16
    weight = torch.rand((cols,), dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.rand((cols,), dtype=dtype, device="cuda", requires_grad=True)
    x = -2.3 + 0.5 * torch.randn((rows, cols), dtype=dtype, device="cuda")
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    y_ref = torch.nn.functional.layer_norm(x, (cols,), weight, bias, eps).to(dtype)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [t.grad.clone() for t in (x, weight, bias)]

    padded = rows + 1
    xp = torch.zeros((padded, cols), dtype=dtype, device="cuda")
    xp[:rows] = x.detach()
    dyp = torch.zeros_like(xp)
    dyp[:rows] = dy
    w, b = weight.detach().reshape(1, cols).contiguous(), bias.detach().reshape(1, cols).contiguous()
    y, dx = torch.empty_like(xp), torch.empty_like(xp)
    mean = torch.empty((padded, 1), dtype=torch.float32, device="cuda")
    rstd = torch.empty_like(mean)
    dw, db = torch.empty_like(w), torch.empty_like(b)
    ln_fwd[(padded,)](xp, w, b, y, mean, rstd, eps)
    ln_bwd_dx[(padded,)](xp, w, mean, rstd, dyp, dx)
    ln_bwd_dwdb[lambda meta: (cols // meta["CB"],)](xp, dyp, mean, rstd, dw, db)
    torch.cuda.synchronize()

    results = []
    for name, got, want in (("y", y[:rows], y_ref), ("dx", dx[:rows], dx_ref), ("dw", dw[0], dw_ref), ("db", db[0], db_ref)):
        ok = torch.allclose(got, want, atol=1e-2, rtol=0)
        results.append((ok, f"{name} max diff {torch.max(torch.abs(got.float() - want.float())).item():.3g}"))
    return all(ok for ok, _ in results), ", ".join(d for _, d in results) + " (allclose atol 1e-2)"
