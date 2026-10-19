# From website/src/content/docs/docs/attributes/launch.mdx.

from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.launch(48)
def k(O: ph.tensor[ph.i32][1, 1]):
    O[0, 0] = ph.i32(ph.ctx.threads)
