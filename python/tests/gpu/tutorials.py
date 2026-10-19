"""Triton's tutorials 01, 02, 03, 05 and 06 as ported to Phobos
(`python/examples/tutorials`), each run on the card and checked against
torch with the tutorial's own tolerance.

    python python/tests/gpu/tutorials.py

Needs a GPU and torch with CUDA.
"""

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "tutorials"))

import torch  # noqa: E402

TUTORIALS = ["t01_vector_add", "t02_fused_softmax", "t03_matmul", "t05_layer_norm", "t06_fused_attention"]


def main():
    failed = 0
    for name in sys.argv[1:] or TUTORIALS:
        try:
            ok, detail = importlib.import_module(name).check(torch)
        except Exception as e:  # a failure to compile or launch is a finding, reported with the rest
            ok, detail = False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:<22} {detail}")
    print(f"{len(sys.argv[1:] or TUTORIALS)} tutorials, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
