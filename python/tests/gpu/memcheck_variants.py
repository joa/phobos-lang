"""Every non-persistent variant of `gemm_fp32` and `flash_attention_fp32`
launched once, small, on buffers with canary bytes after them: run under
`compute-sanitizer --tool memcheck` it reports any access outside an
allocation, and the canaries any write past a tensor inside the pool.

    compute-sanitizer --tool memcheck python python/tests/gpu/memcheck_variants.py

A persistent kernel waits at its grid barrier for blocks the sanitizer
serializes, so it hangs there and is left out. Needs a GPU.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

from phobos import compiler, driver  # noqa: E402
from phobos.array import DeviceArray  # noqa: E402
from phobos.autotune import CANARY, CANARY_BYTES  # noqa: E402

SHAPES = {
    # 256 is a whole number of every tile either space holds.
    "gemm": ({"A": (256, 256), "B": (256, 256), "C": (256, 256)}, (1.0, 0.5),
             lambda meta: (256 // meta["TILE_M"], 256 // meta["TILE_N"])),
    "flash_attention": ({"Q": (256, 64), "K": (256, 64), "V": (256, 64), "O": (256, 64)}, (0.125,),
                        lambda meta: (256 // meta["BR"],)),
}


def main():
    from flash_attention_fp32 import flash_attention
    from gemm_fp32 import gemm

    capability = driver.compute_capability()
    rng = np.random.default_rng(0)
    launched, failed = 0, []
    for kernel in (gemm, flash_attention):
        tensors, scalars, grid = SHAPES[kernel.__name__]
        with kernel.files() as (ph, origin):
            variants, skipped = compiler.compile_variants(ph, capability=capability, extra=["--origin", str(origin)])
        for settings, compiled in variants:
            k = compiled.kernel(kernel.__name__)
            if k.contract.persistent is not None:
                continue
            args = [DeviceArray.from_bytes(rng.uniform(-1, 1, shape).astype(np.float32).tobytes(), shape, "f32",
                                           pad_bytes=CANARY_BYTES) for shape in tensors.values()]
            for a in args:
                a.fill_pad(CANARY)
            try:
                k.launch(grid, *args, *scalars)
                driver.synchronize()
            except driver.CudaError as e:
                failed.append(f"{kernel.__name__} {settings}: {e}")
                break
            launched += 1
            for name, a in zip(tensors, args):
                if a.pad() != bytes([CANARY]) * CANARY_BYTES:
                    failed.append(f"{kernel.__name__} {settings}: a write past `{name}`")
    for f in failed:
        print(f"FAIL {f}")
    print(f"{launched} variants launched, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
