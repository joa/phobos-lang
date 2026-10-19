"""Replays launches a Rust launcher recorded, through the Python launcher, on
the same PTX and input bytes, and compares both launchers byte for byte.

    cargo run --release -p phobos-kernels --features cuda --example lang_check -- --record DIR
    cargo run --release -p phobos-kernels --features cuda --example cross_record -- --record DIR
    python python/tests/gpu/replay.py DIR [--skip-persistent]

For each recording, the Python launcher binds the same arguments from the
contract alone. The slots it packs must equal the Rust launcher's, pointers
aside, at each slot's width; the tensors it leaves must equal the bytes the
Rust launch left. A recording without `.out` files (a launch that trapped)
is skipped, and with `--skip-persistent` so is a `@persistent` kernel, which
hangs under compute-sanitizer. Needs a GPU.
"""

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import phobos  # noqa: E402
from phobos import driver  # noqa: E402
from phobos.launch import SCALARS  # noqa: E402


def decode_scalar(param, word):
    elem = param["elem"]
    raw = struct.pack("<Q", word)
    if elem == "bf16":
        (bits,) = struct.unpack("<H", raw[:2])
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    fmt = SCALARS[elem]
    return struct.unpack(fmt, raw[: struct.calcsize(fmt)])[0]


def replay(at, skip_persistent=False):
    launch = json.loads((at / "launch.json").read_text())
    if not all((at / f"{t['name']}.out").exists() for t in launch["tensors"]):
        return "skip", "no outputs recorded"
    compiled = phobos.Compiled(at.joinpath("kernel.ptx").read_text(), json.loads(at.joinpath("kernel.json").read_text()))
    kernel = compiled.kernel(launch["kernel"])
    c = kernel.contract
    if skip_persistent and c.persistent is not None:
        return "skip", "persistent"
    slots = launch["slots"]
    problems = []
    if [c.cta_threads, 1, 1] != launch["block"]:
        problems.append(f"block: the Rust launch used {launch['block']}, the contract {c.cta_threads}")

    args, arrays = [], {}
    for i, p in enumerate(c.declared):
        if p["shape"] is None:
            args.append(decode_scalar(p, slots[p["slot"]]))
            continue
        mask = 0xFFFFFFFF if c.index_bits == 32 else (1 << 64) - 1
        rank = len(p["shape"])
        shape = [slots[p["slot"] + 3 + d] & mask for d in range(rank)]
        data = (at / f"{p['name']}.in").read_bytes()
        arrays[p["name"]] = phobos.DeviceArray.from_bytes(data, shape, p["elem"])
        args.append(arrays[p["name"]])

    bound = kernel.launch(tuple(launch["grid"]) if c.persistent is None else (), *args)
    driver.synchronize()

    pointers = {a.ptr for a in arrays.values()}
    for i, (got, want, slot) in enumerate(zip(bound.slots, slots, c.slots)):
        want = struct.pack("<Q", want)[: slot["bytes"]]
        if slot["part"] in ("allocated", "aligned"):
            if struct.unpack("<Q", got)[0] not in pointers and c.params[slot["param"]]["role"] == "declared":
                problems.append(f"slot {i}: a pointer to no argument")
        elif got != want:
            problems.append(f"slot {i} {slot['part']}: Python {got.hex()}, Rust {want.hex()}")
    for name, a in arrays.items():
        want = (at / f"{name}.out").read_bytes()
        got = a.to_bytes()
        if got != want:
            diff = sum(x != y for x, y in zip(got, want))
            problems.append(f"`{name}`: {diff} of {len(want)} bytes differ")
    return ("FAIL" if problems else "ok"), "; ".join(problems)


def main():
    root = Path(sys.argv[1])
    skip_persistent = "--skip-persistent" in sys.argv[2:]
    failed = 0
    runs = sorted(p for p in root.iterdir() if (p / "launch.json").exists())
    for at in runs:
        try:
            verdict, detail = replay(at, skip_persistent)
        except Exception as e:  # a launch error is a finding, reported with the rest
            verdict, detail = "FAIL", f"{type(e).__name__}: {e}"
        failed += verdict == "FAIL"
        print(f"{verdict:<4} {at.name:<40} {detail}")
    print(f"{len(runs)} recordings, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
