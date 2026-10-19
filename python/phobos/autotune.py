"""The autotuner: every point of a kernel's own `@autotune` space, compiled
in one race-checked process, checked against a reference and timed on the
card, the fastest kept for each value of the kernel's key extents.

There is no list of configurations to write: the space is the one the
kernel declares, and `@ph.kernel(key=("M", "N"))` names the extents a winner
holds for. Nothing is specialized on an argument's value, and alignment is
what `@aligned` promises, not something the tuner infers.

A variant is accepted only if it computes what the reference does. It runs
on copies of the arguments with canary bytes after each, which a launch
writing past its tensor would change; an integer output must match exactly,
a float one within the tolerance its type's rounding allows, which a caller
may widen for a path that rounds more, as tensor cores round operands.

Timing is CUDA events around each launch, after a warm-up that lasts long
enough for the clocks to ramp; the variants take turns, a launch each a
round, so they share the card's clock and thermal state. Each is the
fastest of its rounds, reported with the spread (slowest minus fastest).
Before every timed launch the L2 is flushed by writing a buffer twice its
size, outside the events, so a variant is timed reading its inputs from
memory, as a kernel in a model mostly does; `flush_l2=False` times it warm.

A result is kept in the kernel cache, beside the chip's compiled kernels,
under `autotune/`: one file per kernel source, keyed by the source's cache
key, the chip, the card's name and the key extents' values.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import compiler, driver
from .array import DeviceArray
from .launch import LaunchError

CANARY_BYTES = 256
CANARY = 0xA5

# The relative and absolute error each float type's rounding allows a
# result whose summation order differs between variants.
TOLERANCE = {"f64": (1e-12, 1e-12), "f32": (1e-5, 1e-6), "f16": (1e-2, 1e-3), "bf16": (2e-2, 1e-2)}


def cache_root():
    """The kernel cache's directory, as `phobos_kernels::util::kernel_cache_dir`
    finds it: `PHOBOS_KERNEL_CACHE_DIR`, or `~/.phobos/kernel-cache`."""
    env = os.environ.get("PHOBOS_KERNEL_CACHE_DIR")
    if env is not None:
        return Path(env) if env else None
    return Path.home() / ".phobos" / "kernel-cache"


@dataclass
class Timing:
    settings: dict
    samples_ms: list = field(default_factory=list)
    verified: bool = False
    detail: str = ""

    @property
    def min_ms(self):
        return min(self.samples_ms, default=float("inf"))

    @property
    def max_ms(self):
        return max(self.samples_ms, default=0.0)

    @property
    def median_ms(self):
        s = sorted(self.samples_ms)
        return s[len(s) // 2] if s else float("inf")

    @property
    def spread_ms(self):
        return self.max_ms - self.min_ms


@dataclass
class Result:
    kernel: str
    chip: str
    card: str
    key: dict
    cache_key: str
    compiler_fingerprint: str
    l2_flush: str
    winner: dict
    variants: list = field(default_factory=list)
    skipped: str = ""


def results_path(chip, kernel, cache_key):
    root = cache_root()
    if root is None:
        return None
    return root / chip / "autotune" / f"{kernel}-{cache_key.rsplit('-', 1)[-1][:32]}.json"


def entry_key(card, key):
    return card + "|" + ",".join(f"{k}={v}" for k, v in sorted(key.items()))


def load(chip, kernel, cache_key, card, key):
    path = results_path(chip, kernel, cache_key)
    if path is None or not path.exists():
        return None
    entries = json.loads(path.read_text(encoding="utf-8"))
    found = entries.get(entry_key(card, key))
    return Result(**found) if found else None


def store(result):
    path = results_path(result.chip, result.kernel, result.cache_key)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    entries[entry_key(result.card, result.key)] = asdict(result)
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def host(arr):
    """A device array's bytes as a numpy array of its type."""
    import numpy as np

    cai = arr.__cuda_array_interface__
    shape = tuple(cai["shape"])
    nbytes = int(np.prod(shape)) * np.dtype(cai["typestr"]).itemsize
    data = driver.download(int(cai["data"][0]), nbytes)
    return np.frombuffer(data, dtype=np.dtype(cai["typestr"])).reshape(shape).copy()


def padded_copy(arr, elem):
    """`arr` copied into a buffer of its own with canary bytes after it."""
    copy = DeviceArray(arr.__cuda_array_interface__["shape"], elem, pad_bytes=CANARY_BYTES)
    copy_from(copy, arr)
    copy.fill_pad(CANARY)
    return copy


def copy_from(dst, src):
    cai = src.__cuda_array_interface__
    driver.call("cuMemcpyDtoD_v2", driver.CUdeviceptr(dst.ptr), driver.CUdeviceptr(int(cai["data"][0])),
                driver.c_size_t(dst.nbytes))


def compare(name, elem, got, want, rtol, atol):
    """Why `got` is not `want`, or None."""
    import numpy as np

    want = np.asarray(want)
    if got.shape != want.shape:
        return f"`{name}` has shape {got.shape}, the reference {want.shape}"
    if elem not in TOLERANCE:
        bad = np.count_nonzero(got != want)
        return f"`{name}`: {bad} of {got.size} elements differ" if bad else None
    r, a = TOLERANCE[elem]
    r, a = rtol if rtol is not None else r, atol if atol is not None else a
    g, w = got.astype(np.float64), want.astype(np.float64)
    if not np.all(np.isfinite(g) == np.isfinite(w)):
        return f"`{name}`: non-finite where the reference is finite"
    err = np.abs(g - w)
    bad = np.count_nonzero(err > a + r * np.abs(w))
    if bad:
        return f"`{name}`: {bad} of {got.size} elements past rtol {r} atol {a}, worst {err.max():.3g}"
    return None


def tune(kernel, grid, args, reference, rtol=None, atol=None, rounds=20, warmup_ms=300.0, flush_l2=True, index_bits=None,
         log=print):
    """Tunes `kernel` for the key values `args` bind. `reference(*host_args)`
    takes the arguments as numpy arrays and scalars and returns the expected
    outputs, a dict from parameter name to array. `index_bits` compiles at
    that index width, which the runtime otherwise chooses."""
    import numpy as np

    capability = driver.compute_capability()
    card = driver.device_name()
    base = kernel.compile(capability=capability, index_bits=index_bits)
    contract = base.kernel(kernel.__name__).contract
    bound_dims = _bind_dims(contract, args)
    key = {k: bound_dims[k] for k in kernel.key}
    with kernel.files() as (ph, origin):
        variants, skipped = compiler.compile_variants(ph, capability=capability, index_bits=index_bits,
                                                      extra=["--origin", str(origin)])
    if skipped.strip():
        log(skipped.strip())
    if not variants:
        raise compiler.CompileError(f"no variant of `{kernel.__name__}` compiled:\n{skipped}")

    declared = contract.declared
    host_args = [host(a) if p["shape"] is not None else a for p, a in zip(declared, args)]
    expected = reference(*host_args)
    outputs = {name: np.asarray(v) for name, v in expected.items()}
    unknown = set(outputs) - {p["name"] for p in declared}
    if unknown:
        raise LaunchError(f"the reference returns {sorted(unknown)}, which `{kernel.__name__}` does not take")

    # One set of copies for every variant, each check starting from the
    # arguments as given and fresh canaries.
    copies = [padded_copy(a, p["elem"]) if p["shape"] is not None else a for p, a in zip(declared, args)]
    timings, runs = [], []
    for settings, compiled in variants:
        t = Timing(settings)
        timings.append(t)
        k = compiled.kernel(kernel.__name__)
        for p, c, a in zip(declared, copies, args):
            if p["shape"] is not None:
                copy_from(c, a)
                c.fill_pad(CANARY)
        try:
            k.launch(grid, *copies)
            driver.synchronize()
        except (driver.CudaError, LaunchError) as e:
            t.detail = f"launch failed: {e}"
            continue
        problems = []
        for p, c in zip(declared, copies):
            if p["shape"] is None:
                continue
            if c.pad() != bytes([CANARY]) * CANARY_BYTES:
                problems.append(f"`{p['name']}`: a write past its end")
            if p["name"] in outputs:
                why = compare(p["name"], p["elem"], host(c), outputs[p["name"]], rtol, atol)
                if why:
                    problems.append(why)
        if problems:
            t.detail = "; ".join(problems)
            continue
        t.verified = True
        runs.append((t, k))

    if not runs:
        raise LaunchError(f"no variant of `{kernel.__name__}` computes what the reference does:\n"
                          + "\n".join(f"  {t.settings}: {t.detail}" for t in timings))

    flush = None
    if flush_l2:
        l2 = driver.attribute(driver.DEVICE_L2_CACHE_SIZE)
        flush = DeviceArray([2 * l2], "u8")
    start, end = driver.Event(), driver.Event()

    def once(k):
        if flush is not None:
            driver.memset(flush.ptr, 0, flush.nbytes)
        start.record()
        k.launch(grid, *copies)
        end.record()
        end.synchronize()
        return end.elapsed_ms(start)

    began, done = time.perf_counter(), 0
    while done < 3 or (time.perf_counter() - began) * 1e3 < warmup_ms:
        for _, k in runs:
            once(k)
        done += 1
    for _ in range(rounds):
        for t, k in runs:
            t.samples_ms.append(once(k))

    best = min((t for t, _ in runs), key=lambda t: t.min_ms)
    result = Result(
        kernel=kernel.__name__, chip=base.chip, card=card, key=key, cache_key=base.cache_key,
        compiler_fingerprint=base.fingerprint,
        l2_flush=f"memset of {2 * driver.attribute(driver.DEVICE_L2_CACHE_SIZE)} bytes before each timed launch" if flush_l2 else "none",
        winner=best.settings,
        variants=[{**asdict(t), "min_ms": t.min_ms, "median_ms": t.median_ms, "spread_ms": t.spread_ms} for t in timings],
        skipped=skipped.strip(),
    )
    store(result)
    return result


def _bind_dims(contract, args):
    from .launch import bind

    # Binding needs pointers for the scratch only to pack them; the dims
    # come from the declared arguments alone.
    extra = [0] * (len(contract.params) - len(contract.declared))
    return bind(contract, list(args), extra).dims


__all__ = ["Result", "Timing", "cache_root", "load", "store", "tune"]
