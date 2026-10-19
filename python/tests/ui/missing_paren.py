from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.launch(64)
def k:
    pass


@ph.launch(64)
x: ph.let = 1
