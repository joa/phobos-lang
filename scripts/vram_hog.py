#!/usr/bin/env python3
"""Holds device memory in a process of its own, the way another program would.

Allocates MIB of device memory in 256 MiB chunks, prints `holding N MiB`
once it has them, then rewrites every chunk each TOUCH_MS milliseconds until
its stdin closes. The rewriting is what makes it pressure: the driver pages an
idle allocation out to the host before an active one, so a hog that only
allocates hands its memory straight back to whoever is busy.

`--free` prints what the driver reports free from a context of its own and
exits, which says whether one process sees memory another one holds.

Usage:
    python scripts/vram_hog.py 2048
    python scripts/vram_hog.py 2048 --touch-ms 250
    python scripts/vram_hog.py --free

Talks to the CUDA driver through ctypes, so it needs only the driver.
"""

import argparse
import ctypes
import os
import sys
import threading
import time

CHUNK_BYTES = 256 << 20


class Driver:
    def __init__(self):
        name = "nvcuda.dll" if os.name == "nt" else "libcuda.so.1"
        self.lib = ctypes.WinDLL(name) if os.name == "nt" else ctypes.CDLL(name)
        self.call("cuInit", 0)
        dev = ctypes.c_int()
        self.call("cuDeviceGet", ctypes.byref(dev), 0)
        self.ctx = ctypes.c_void_p()
        self.call("cuCtxCreate_v2", ctypes.byref(self.ctx), 0, dev)

    def call(self, fn, *args):
        code = getattr(self.lib, fn)(*args)
        if code:
            raise RuntimeError(f"{fn} failed with CUDA error {code}")

    def free_total(self):
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.call("cuMemGetInfo_v2", ctypes.byref(free), ctypes.byref(total))
        return free.value, total.value

    def alloc(self, size_bytes):
        ptr = ctypes.c_uint64()
        self.call("cuMemAlloc_v2", ctypes.byref(ptr), ctypes.c_size_t(size_bytes))
        return ptr

    def fill(self, ptr, size_bytes, value):
        self.call("cuMemsetD32_v2", ptr, ctypes.c_uint(value), ctypes.c_size_t(size_bytes // 4))

    def sync(self):
        self.call("cuCtxSynchronize")


class NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def nvml_free_total():
    """Free and total device memory across every process, as nvidia-smi
    counts it, or None without NVML."""
    try:
        lib = ctypes.CDLL("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
    except OSError:
        return None
    handle, mem = ctypes.c_void_p(), NvmlMemory()
    if lib.nvmlInit_v2() or lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)):
        return None
    if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)):
        return None
    return mem.free, mem.total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mib", type=int, nargs="?", help="device memory to hold")
    ap.add_argument("--touch-ms", type=int, default=250, help="milliseconds between rewrites")
    ap.add_argument("--free", action="store_true", help="print free device memory and exit")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    drv = Driver()
    if args.free:
        free, total = drv.free_total()
        print(f"this process: free {free >> 20} MiB of {total >> 20} MiB")
        if card := nvml_free_total():
            print(f"whole card:   free {card[0] >> 20} MiB of {card[1] >> 20} MiB")
        return
    if not args.mib:
        ap.error("give the MiB to hold, or --free")

    chunks = []
    try:
        for _ in range(-(-args.mib * (1 << 20) // CHUNK_BYTES)):
            chunks.append(drv.alloc(CHUNK_BYTES))
    except RuntimeError as e:
        print(f"allocation stopped at {len(chunks) * CHUNK_BYTES >> 20} MiB: {e}")
    for ptr in chunks:
        drv.fill(ptr, CHUNK_BYTES, 0)
    drv.sync()
    print(f"holding {len(chunks) * CHUNK_BYTES >> 20} MiB, rewritten every {args.touch_ms} ms")

    closed = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.read(), closed.set()), daemon=True).start()
    value = 0
    while not closed.wait(args.touch_ms / 1e3):
        value += 1
        for ptr in chunks:
            drv.fill(ptr, CHUNK_BYTES, value)
        drv.sync()


if __name__ == "__main__":
    main()
