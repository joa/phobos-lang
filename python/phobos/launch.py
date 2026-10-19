"""Launching a compiled kernel from its launch contract.

`phobos-compile -o K.ptx` writes `K.json` beside the PTX: each kernel's
`.entry` parameters, index width, CTA size, shared memory, published scratch,
barrier word and persistent requirement. A launch binds the arguments to that
contract and nothing else, so whatever the compiler decides, the launch
follows.

A tensor argument is any object with `__cuda_array_interface__` (torch,
CuPy, Numba, or `phobos.DeviceArray`), row-major and contiguous, with its
element type, rank and fixed extents as declared. A scalar is a Python number.
"""

import ctypes
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from . import driver

# Element type: the `__cuda_array_interface__` typestr kinds it accepts, and
# its width. bf16 has no typestr of its own; its bits travel as 2-byte words.
ELEMENTS = {
    "f16": (("f2",), 2),
    "bf16": (("V2", "u2", "i2"), 2),
    "f32": (("f4",), 4),
    "f64": (("f8",), 8),
    "i8": (("i1",), 1),
    "u8": (("u1",), 1),
    "u16": (("u2",), 2),
    "i32": (("i4",), 4),
    "u32": (("u4",), 4),
    "i64": (("i8",), 8),
    "bool": (("b1",), 1),
}

SCALARS = {
    "f16": "<e", "f32": "<f", "f64": "<d", "i8": "<b", "u8": "<B", "u16": "<H",
    "i32": "<i", "u32": "<I", "i64": "<q", "bool": "<?",
}


class LaunchError(ValueError):
    pass


@dataclass
class Contract:
    """One kernel's launch contract, as `phobos-compile` writes it."""

    raw: dict

    def __getattr__(self, name):
        try:
            return self.raw[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def declared(self):
        return [p for p in self.raw["params"] if p["role"] == "declared"]


@dataclass
class Bound:
    """Arguments bound to a contract: each slot's bytes, and the symbols the
    tensors' extents bound."""

    slots: list
    dims: dict


def row_major(shape):
    strides, acc = [], 1
    for extent in reversed(shape):
        strides.append(acc)
        acc *= extent
    return list(reversed(strides))


def bf16_bits(x):
    """`x` rounded to bf16, nearest even, as its 16 bits."""
    (bits,) = struct.unpack("<I", struct.pack("<f", x))
    if (bits & 0x7F800000) == 0x7F800000 and bits & 0x7FFFFF:
        return (bits >> 16) | 0x40
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def interface(arg, name):
    cai = getattr(arg, "__cuda_array_interface__", None)
    if cai is None:
        raise LaunchError(f"`{name}` is a tensor parameter; pass an object with __cuda_array_interface__, not {type(arg).__name__}")
    return cai


def bind(contract, args, extra_pointers=()):
    """Binds `args`, the declared parameters in order, to `contract`'s slots.
    `extra_pointers` are the scratch and barrier word buffers, in order."""
    declared = contract.declared
    if len(args) != len(declared):
        names = ", ".join(p["name"] for p in declared)
        raise LaunchError(f"`{contract.kernel}` takes {len(declared)} arguments ({names}), got {len(args)}")
    params = contract.params
    values = list(args) + list(extra_pointers)
    shapes, pointers = {}, {}
    for i, (param, value) in enumerate(zip(params, values)):
        if param["shape"] is None:
            continue
        if param["role"] != "declared":
            shapes[i] = [int(e) for e in param["shape"]]
            pointers[i] = value
            continue
        name = param["name"]
        cai = interface(value, name)
        kinds, width = ELEMENTS[param["elem"]]
        if cai["typestr"][1:] not in kinds:
            raise LaunchError(f"`{name}` is {param['elem']}, got typestr {cai['typestr']}")
        shape = [int(e) for e in cai["shape"]]
        if len(shape) != len(param["shape"]):
            raise LaunchError(f"`{name}` has rank {len(param['shape'])}, got shape {tuple(shape)}")
        strides = cai.get("strides")
        if strides is not None and [s // width for s in strides] != row_major(shape) and 0 not in shape:
            raise LaunchError(f"`{name}` must be row-major and contiguous, got strides {tuple(strides)}")
        ptr = int(cai["data"][0])
        if ptr % param["align_bytes"]:
            raise LaunchError(f"`{name}` must start on a {param['align_bytes']}-byte boundary, got address {ptr:#x}")
        for axis, (want, got) in enumerate(zip(param["shape"], shape)):
            if isinstance(want, int) and want != got:
                raise LaunchError(f"`{name}` axis {axis} is {want}, got {got}")
        shapes[i], pointers[i] = shape, ptr

    dims = {}
    for d in contract.dims:
        if d["axis"] is None:
            value = int(values[d["param"]])
        else:
            value = shapes[d["param"]][d["axis"]]
        if value % d["multiple_of"]:
            raise LaunchError(f"`{d['name']}` is {value}, which the kernel needs to be a multiple of {d['multiple_of']}")
        dims[d["name"]] = value
    for i, param in enumerate(params):
        for axis, want in enumerate(param["shape"] or []):
            if isinstance(want, str) and want in dims and shapes[i][axis] != dims[want]:
                raise LaunchError(f"`{param['name']}` axis {axis} is `{want}`, bound to {dims[want]}, got {shapes[i][axis]}")

    index = "<i" if contract.index_bits == 32 else "<q"
    slots = []
    for slot in contract.slots:
        i, part = slot["param"], slot["part"]
        if part in ("allocated", "aligned"):
            slots.append(struct.pack("<Q", pointers[i]))
        elif part == "offset":
            slots.append(struct.pack(index, 0))
        elif isinstance(part, dict):
            ((kind, axis),) = part.items()
            shape = shapes[i]
            v = shape[axis] if kind == "size" else row_major(shape)[axis]
            try:
                slots.append(struct.pack(index, v))
            except struct.error:
                raise LaunchError(f"`{params[i]['name']}` {kind} {v} does not fit the kernel's {contract.index_bits}-bit index") from None
        elif part == "value":
            slots.append(scalar(params[i], values[i]))
        else:
            raise LaunchError(f"unknown slot part {part!r}")
    for got, want in zip(slots, contract.slots):
        assert len(got) == want["bytes"], (got, want)
    return Bound(slots, dims)


def scalar(param, value):
    elem, name = param["elem"], param["name"]
    if elem == "bool":
        if not isinstance(value, bool):
            raise LaunchError(f"`{name}` is a bool, got {type(value).__name__}")
        return struct.pack("<?", value)
    if elem in ("f16", "f32", "f64", "bf16"):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise LaunchError(f"`{name}` is {elem}, got {type(value).__name__}")
        return struct.pack("<H", bf16_bits(float(value))) if elem == "bf16" else struct.pack(SCALARS[elem], float(value))
    if not isinstance(value, int) or isinstance(value, bool):
        raise LaunchError(f"`{name}` is {elem}, got {type(value).__name__}")
    try:
        return struct.pack(SCALARS[elem], value)
    except struct.error:
        raise LaunchError(f"`{name}` is {elem}, and {value} does not fit") from None


class Kernel:
    """One kernel of a compiled module, launched as `kernel[grid](*args)`.

    `grid` is up to three block counts, or a function of the bound symbols
    and `@autotune` constants returning them. A `@persistent` kernel takes no
    grid, `kernel[()]`: it is the compiled-in block count, or as many blocks
    as stay resident at once.
    """

    def __init__(self, compiled, contract):
        self.compiled, self.contract = compiled, contract
        self._scratch = None

    @property
    def name(self):
        return self.contract.kernel

    def __getitem__(self, grid):
        return lambda *args, stream=0: self.launch(grid, *args, stream=stream)

    def _extra(self):
        """The published scratch and the barrier word, allocated once. The
        word is zeroed once; scratch is uninitialized, as the kernel expects."""
        if self._scratch is None:
            self._scratch = []
            for p in self.contract.params:
                if p["role"] == "declared":
                    continue
                nbytes = ELEMENTS[p["elem"]][1]
                for e in p["shape"]:
                    nbytes *= int(e)
                ptr = driver.alloc(nbytes)
                if p["role"] == "barrier_word":
                    driver.memset(ptr, 0, nbytes)
                self._scratch.append(ptr)
        return self._scratch

    def grid(self, grid, dims):
        c = self.contract
        if c.persistent is not None:
            if grid not in ((), None):
                raise LaunchError(f"`{c.kernel}` is @persistent; it sizes its own grid, launch it as kernel[()]")
            blocks = c.persistent["blocks"]
            if blocks is None:
                blocks = driver.resident_blocks(self.function(), c.cta_threads, c.dynamic_shared_bytes)
            return (blocks, 1, 1)
        if callable(grid):
            grid = grid({**c.autotune, **dims})
        if isinstance(grid, int):
            grid = (grid,)
        grid = tuple(int(g) for g in grid)
        if not 1 <= len(grid) <= 3 or min(grid) < 1:
            raise LaunchError(f"a grid is one to three positive block counts, got {grid}")
        return grid + (1,) * (3 - len(grid))

    def function(self):
        func = self.compiled.module().function(self.contract.kernel)
        driver.raise_dynamic_shared(func, self.contract.dynamic_shared_bytes)
        return func

    def launch(self, grid, *args, stream=0):
        c = self.contract
        bound = bind(c, args, self._extra())
        grid = self.grid(grid, bound.dims)
        buffers = [ctypes.create_string_buffer(s, len(s)) for s in bound.slots]
        params = driver.pointer_array(buffers)
        driver.launch(self.function(), grid, (c.cta_threads, 1, 1), c.dynamic_shared_bytes, stream, params)
        return bound


class Compiled:
    """A compiled module: its PTX and every kernel's contract."""

    def __init__(self, ptx, descriptor):
        self.ptx, self.descriptor = ptx, descriptor
        self.contracts = {k["kernel"]: Contract(k) for k in descriptor["kernels"]}
        self._module = None
        self._kernels = {}

    @classmethod
    def load(cls, ptx_path):
        """The PTX at `ptx_path` and the contract `phobos-compile` wrote beside it."""
        ptx_path = Path(ptx_path)
        return cls(ptx_path.read_text(), json.loads(ptx_path.with_suffix(".json").read_text()))

    @property
    def cache_key(self):
        return self.descriptor["cache_key"]

    @property
    def chip(self):
        return self.descriptor["chip"]

    @property
    def fingerprint(self):
        return self.descriptor["compiler_fingerprint"]

    def kernel(self, name=None):
        if name is None:
            if len(self.contracts) != 1:
                raise LaunchError(f"the module has kernels {sorted(self.contracts)}; name one")
            (name,) = self.contracts
        if name not in self._kernels:
            self._kernels[name] = Kernel(self, self.contracts[name])
        return self._kernels[name]

    def module(self):
        """The module loaded into the driver, on first use."""
        if self._module is None:
            self._module = driver.Module(self.ptx)
        return self._module
