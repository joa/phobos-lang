"""The semantics table on the card: each row's kernel launched with its
operands at run time, the value it computes against the value the table
documents, and against what Python computes where the two differ.

    python python/tests/gpu/semantics.py

Needs a GPU and torch with CUDA.
"""

import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import semantics_kernels as k  # noqa: E402
from phobos import driver  # noqa: E402


def launch(kernel, grid, *args, index_bits=None):
    compiled = kernel.compile(capability=driver.compute_capability(), index_bits=index_bits)
    compiled.kernel(kernel.__name__).launch(grid, *args)
    torch.cuda.synchronize()


def main():
    rows = []
    i = torch.zeros(4, dtype=torch.int32, device="cuda")
    for a, b in ((-7, 2), (7, -2), (-7, 3), (7, 3)):
        launch(k.division, (1,), i, a, b)
        got = i[:2].tolist()
        want = [int(a / b), a - b * int(a / b)]
        rows.append((f"{a} // {b}, {a} % {b}", got, want, [a // b, a % b]))
    x = torch.zeros(4, dtype=torch.float32, device="cuda")
    try:
        launch(k.remainder, (1,), x, -7.5, 2.0)
        rows.append(("-7.5 % 2.0", [x[0].item()], [-1.5], [-7.5 % 2.0]))
    except driver.CudaError as e:
        rows.append(("-7.5 % 2.0", [f"{e}"], [-1.5], [-7.5 % 2.0]))
    j = torch.zeros(4, dtype=torch.int64, device="cuda")
    launch(k.wrap, (1,), j, 1)
    rows.append(("2147483647 + n, n = 1, 32-bit index", [j[0].item()], [-2147483648], [2147483648]))
    launch(k.wrap, (1,), j, 1, index_bits=64)
    rows.append(("2147483647 + n, n = 1, 64-bit index", [j[0].item()], [2147483648], [2147483648]))
    d = torch.zeros(4, dtype=torch.float64, device="cuda")
    launch(k.widened, (1,), d)
    rows.append(("0.1 into an f64", [d[0].item()], [struct.unpack("<f", struct.pack("<f", 0.1))[0]], [0.1]))
    c = torch.zeros(256, dtype=torch.int32, device="cuda")
    launch(k.chained, (1,), c, 0, 200)
    want = [int(0 < t - 1 < 200) for t in range(256)]
    rows.append(("0 < i - 1 < 200 over 256 threads", [sum(c.tolist())], [sum(want)], [sum(want)]))
    rows.append(("  ... element by element", [c.tolist() == want], [True], [True]))
    launch(k.free_name, (1,), i, 7)
    rows.append(("a fn's `s` is its caller's, s = 7", [i[0].item()], [21], [21]))

    failed = 0
    for what, got, want, python in rows:
        ok = got == want
        failed += not ok
        differs = "" if want == python else f" (Python: {python})"
        print(f"{'ok  ' if ok else 'FAIL'} {what:<40} {got}{differs}")
    print(f"{len(rows)} rows, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
