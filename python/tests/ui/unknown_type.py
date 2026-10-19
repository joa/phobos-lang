from __future__ import annotations

import phobos.lang as ph


@ph.kernel
def k(X: ph.tensor[ph.f23][N], Y: ph.tile[ph.f32][N]):
    pass


@ph.kernel
def j(X: ph.tensr[ph.f32][N]):
    pass
