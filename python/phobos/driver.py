"""The CUDA driver API through ctypes, as much of it as a launch needs.

Nothing here loads the driver until a function is first called, so a host
without a GPU can import the package, compile kernels and pack arguments.
The context is the device's primary context, the one the CUDA runtime and
so torch and CuPy use, so their allocations and ours are interchangeable.
"""

import ctypes
import sys
from ctypes import byref, c_char_p, c_float, c_int, c_size_t, c_uint, c_void_p

CUdeviceptr = ctypes.c_uint64

# CUfunction_attribute and CUdevice_attribute values from cuda.h.
FUNC_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
DEVICE_MULTIPROCESSOR_COUNT = 16
DEVICE_COMPUTE_CAPABILITY_MAJOR = 75
DEVICE_COMPUTE_CAPABILITY_MINOR = 76
DEVICE_L2_CACHE_SIZE = 38

# The shared memory a block holds before the launch must raise its ceiling.
STATIC_SHARED_LIMIT_BYTES = 48 * 1024


class CudaError(RuntimeError):
    pass


_lib = None
_context = None


def _load():
    global _lib
    if _lib is None:
        name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
        try:
            _lib = ctypes.CDLL(name)
        except OSError as e:
            raise CudaError(f"no CUDA driver ({name}): {e}") from e
        _check(_lib.cuInit(0), "cuInit")
    return _lib


def _check(status, what):
    if status != 0:
        name = c_char_p()
        if _lib is not None:
            _lib.cuGetErrorName(status, byref(name))
        label = name.value.decode() if name.value else str(status)
        raise CudaError(f"{what}: {label}")


def call(name, *args):
    """Calls driver function `name` and raises on a status other than success."""
    _check(getattr(_load(), name)(*args), name)


def context(device=0):
    """Makes the device's primary context current on this thread."""
    global _context
    lib = _load()
    if _context is None:
        dev = c_int()
        call("cuDeviceGet", byref(dev), device)
        ctx = c_void_p()
        call("cuDevicePrimaryCtxRetain", byref(ctx), dev)
        _context = (dev.value, ctx)
    _check(lib.cuCtxSetCurrent(_context[1]), "cuCtxSetCurrent")
    return _context[0]


def attribute(which, device=0):
    value = c_int()
    call("cuDeviceGetAttribute", byref(value), which, context(device))
    return value.value


def compute_capability(device=0):
    return attribute(DEVICE_COMPUTE_CAPABILITY_MAJOR, device) * 10 + attribute(DEVICE_COMPUTE_CAPABILITY_MINOR, device)


def device_name(device=0):
    buf = ctypes.create_string_buffer(256)
    call("cuDeviceGetName", buf, 256, context(device))
    return buf.value.decode()


class Module:
    """A loaded PTX module. Functions are looked up once and kept."""

    def __init__(self, ptx):
        context()
        self.handle = c_void_p()
        image = ptx.encode() + b"\0"
        call("cuModuleLoadData", byref(self.handle), image)
        self._functions = {}

    def function(self, name):
        if name not in self._functions:
            func = c_void_p()
            call("cuModuleGetFunction", byref(func), self.handle, name.encode())
            self._functions[name] = func
        return self._functions[name]

    def __del__(self):
        if _lib is not None and getattr(self, "handle", None):
            _lib.cuModuleUnload(self.handle)


def raise_dynamic_shared(func, bytes_):
    if bytes_ > STATIC_SHARED_LIMIT_BYTES:
        call("cuFuncSetAttribute", func, FUNC_MAX_DYNAMIC_SHARED_SIZE_BYTES, c_int(bytes_))


def resident_blocks(func, threads, dynamic_shared_bytes):
    """Blocks of `func` resident at once on the whole device, as a
    `@persistent` grid must be: blocks per SM times the SM count."""
    per_sm = c_int()
    call("cuOccupancyMaxActiveBlocksPerMultiprocessor", byref(per_sm), func, c_int(threads), c_size_t(dynamic_shared_bytes))
    if per_sm.value < 1:
        raise CudaError(f"a persistent kernel fits no block per SM at {threads} threads and {dynamic_shared_bytes} dynamic shared bytes")
    return per_sm.value * attribute(DEVICE_MULTIPROCESSOR_COUNT)


def launch(func, grid, block, shared_bytes, stream, params):
    """Launches `func`; `params` is a ctypes array of pointers to the slots."""
    gx, gy, gz = grid
    bx, by, bz = block
    call("cuLaunchKernel", func, c_uint(gx), c_uint(gy), c_uint(gz), c_uint(bx), c_uint(by), c_uint(bz),
         c_uint(shared_bytes), c_void_p(stream), params, None)


def alloc(nbytes):
    context()
    ptr = CUdeviceptr()
    call("cuMemAlloc_v2", byref(ptr), c_size_t(max(nbytes, 1)))
    return ptr.value


def free(ptr):
    if _lib is not None and ptr:
        _lib.cuMemFree_v2(CUdeviceptr(ptr))


def upload(ptr, data):
    context()
    call("cuMemcpyHtoD_v2", CUdeviceptr(ptr), data, c_size_t(len(data)))


def download(ptr, nbytes):
    context()
    out = ctypes.create_string_buffer(nbytes)
    call("cuMemcpyDtoH_v2", out, CUdeviceptr(ptr), c_size_t(nbytes))
    return out.raw


def memset(ptr, byte, nbytes):
    context()
    call("cuMemsetD8_v2", CUdeviceptr(ptr), ctypes.c_ubyte(byte), c_size_t(nbytes))


def synchronize():
    context()
    call("cuCtxSynchronize")


class Event:
    def __init__(self):
        context()
        self.handle = c_void_p()
        call("cuEventCreate", byref(self.handle), 0)

    def record(self, stream=0):
        call("cuEventRecord", self.handle, c_void_p(stream))

    def synchronize(self):
        call("cuEventSynchronize", self.handle)

    def elapsed_ms(self, start):
        ms = c_float()
        call("cuEventElapsedTime", byref(ms), start.handle, self.handle)
        return ms.value

    def __del__(self):
        if _lib is not None and getattr(self, "handle", None):
            _lib.cuEventDestroy_v2(self.handle)


def pointer_array(buffers):
    """A `void**` over ctypes buffers, which must outlive the launch."""
    return (c_void_p * len(buffers))(*[ctypes.cast(b, c_void_p) for b in buffers])


__all__ = [
    "CudaError", "Event", "Module", "alloc", "attribute", "compute_capability", "context", "device_name",
    "download", "free", "launch", "memset", "pointer_array", "raise_dynamic_shared", "resident_blocks",
    "synchronize", "upload",
]
