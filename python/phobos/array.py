"""A device buffer with `__cuda_array_interface__`, for when no array library
is at hand: a launch takes it as it takes a torch or CuPy array."""

from math import prod

from . import driver
from .launch import ELEMENTS

TYPESTR = {
    "f16": "<f2", "bf16": "<u2", "f32": "<f4", "f64": "<f8", "i8": "|i1", "u8": "|u1",
    "u16": "<u2", "i32": "<i4", "u32": "<u4", "i64": "<i8", "bool": "|b1",
}


class DeviceArray:
    """`shape` elements of `elem` in device memory, row-major, with `pad_bytes`
    more allocated after them: room for canary bytes a launch must leave
    alone."""

    def __init__(self, shape, elem, pad_bytes=0):
        self.shape, self.elem = tuple(int(e) for e in shape), elem
        self.nbytes = prod(self.shape) * ELEMENTS[elem][1]
        self.pad_bytes = pad_bytes
        self.ptr = driver.alloc(self.nbytes + pad_bytes)

    @classmethod
    def from_bytes(cls, data, shape, elem, pad_bytes=0):
        a = cls(shape, elem, pad_bytes)
        if len(data) != a.nbytes:
            raise ValueError(f"{len(data)} bytes for a {shape} {elem} array of {a.nbytes}")
        driver.upload(a.ptr, bytes(data))
        return a

    @property
    def __cuda_array_interface__(self):
        return {"shape": self.shape, "typestr": TYPESTR[self.elem], "data": (self.ptr, False), "version": 3, "strides": None}

    def to_bytes(self):
        return driver.download(self.ptr, self.nbytes)

    def fill_pad(self, byte):
        if self.pad_bytes:
            driver.memset(self.ptr + self.nbytes, byte, self.pad_bytes)

    def pad(self):
        return driver.download(self.ptr + self.nbytes, self.pad_bytes)

    def __del__(self):
        driver.free(getattr(self, "ptr", 0))
