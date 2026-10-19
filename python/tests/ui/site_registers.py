# From website/src/content/docs/docs/attributes/registers.mdx.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.launch(256)
@ph.registers("acc")
def k(O: ph.tensor[ph.f32][3, 5]):
    acc: ph.var[ph.tile[ph.f32][3, 5]] = 0.0
    O[0:+3, 0:+5] = acc
