"""Writes a stand-in CUDA driver and cuBLAS, so the `cuda` features build,
link and run their device-free tests on a machine without a card.

    python scripts/ci/cuda_stub.py DIR

leaves `DIR/lib64/libcuda.so` and `DIR/lib64/libcublas.so`, whose every
function returns an error, as a machine without a card would. Point
`CUDA_LIBRARY_PATH` at DIR, where cust's build script looks for the driver,
and `LD_LIBRARY_PATH` at `DIR/lib64`. The functions are the ones cust's
bindings declare and the ones phobos-kbench binds itself.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# CUDA_ERROR_NO_DEVICE and CUBLAS_STATUS_NOT_INITIALIZED.
DRIVER_ERROR = 100
CUBLAS_ERROR = 1


def functions(source, prefix):
    return sorted(set(re.findall(rf"\bfn ({prefix}\w*)\s*\(", source.read_text(encoding="utf-8"))))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    lib = Path(sys.argv[1]).resolve() / "lib64"
    lib.mkdir(parents=True, exist_ok=True)

    command = ["cargo", "metadata", "--format-version", "1", "--locked", "--filter-platform", "x86_64-unknown-linux-gnu"]
    meta = json.loads(subprocess.check_output(command, cwd=ROOT))
    cust_raw = next(p for p in meta["packages"] if p["name"] == "cust_raw")
    driver = functions(Path(cust_raw["manifest_path"]).parent / "src" / "cuda.rs", "cu")
    cublas = functions(ROOT / "phobos-kbench" / "src" / "cublas.rs", "cublas")

    for name, names, error in (("cuda", driver, DRIVER_ERROR), ("cublas", cublas, CUBLAS_ERROR)):
        source = lib / f"{name}.c"
        source.write_text("".join(f"int {f}(void) {{ return {error}; }}\n" for f in names))
        subprocess.check_call(["cc", "-shared", "-fPIC", "-o", str(lib / f"lib{name}.so"), str(source)])
    print(f"{len(driver)} driver and {len(cublas)} cuBLAS functions under {lib}")


if __name__ == "__main__":
    main()
