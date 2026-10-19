"""The Phobos language spelled in Python: `import phobos.lang as ph`.

A function decorated `@ph.kernel` is a Phobos kernel and `@ph.fn` a Phobos
`fn`. Their bodies are never run: the front end reads their source and
writes the Phobos it spells, so every name in a body is a spelling of a
Phobos construct, resolved from this module by what the module imports.
README.md lists the spellings; `lang.pyi` lists what this module holds.

Nothing else here does anything at run time. The attribute decorators
(`@ph.autotune(...)`, `@ph.launch(...)`) hand the function on unchanged,
and the front end reads them from the source as it reads the body.
"""

import functools as _functools


class Marker:
    """A name the front end reads from the source: a type, `let`, `var`, a
    built-in. Subscripting one gives one, so `ph.tensor[ph.f32][M, K]` can be
    evaluated where Python insists, as a default annotation is."""

    def __init__(self, name):
        self._name = name

    def __getitem__(self, _):
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Marker(f"{self._name}.{name}")

    def __call__(self, *args, **kwargs):
        # A bare decorator, `@ph.tensorcore`, is called with the function.
        if len(args) == 1 and not kwargs and callable(args[0]) and not isinstance(args[0], Marker):
            return args[0]
        return Attribute(self._name)

    def __repr__(self):
        return f"ph.{self._name}"


class Attribute(Marker):
    """`@ph.name(args)`: the attribute's arguments live in the source."""

    def __call__(self, fn):
        return fn


let = Marker("let")
var = Marker("var")
tensor = Marker("tensor")
tile = Marker("tile")
ctx = Marker("ctx")
f16, bf16, f32, f64 = Marker("f16"), Marker("bf16"), Marker("f32"), Marker("f64")
i8, i32, i64, u8, u16, u32 = Marker("i8"), Marker("i32"), Marker("i64"), Marker("u8"), Marker("u16"), Marker("u32")
# `ph.bool` comes from `__getattr__`, so this module's `bool` stays Python's.


def unroll(n, loop):
    """`for i in ph.unroll(n, range(lo, hi))` is `for<unroll n> i in range(lo, hi)`."""
    return loop


def kernel(fn=None, *, key=None, reference=None):
    """`@ph.kernel`: a Phobos kernel. `key` names the extents the autotuner
    tunes for, one winner per value they take; with `reference`, a launch at
    key values with no winner yet tunes first. See `phobos.autotune`."""
    if fn is None:
        return _functools.partial(kernel, key=key, reference=reference)
    from .jit import Kernel

    return Kernel(fn, key=key, reference=reference)


def fn(f=None, *, uses=()):
    """`@ph.fn`: a Phobos `fn`, inlined at every call. A name its body reads
    that is neither a parameter nor its own resolves where it is called, as
    in Phobos, and has to be listed in `uses`."""
    if f is None:
        return _functools.partial(fn, uses=uses)
    from .jit import Function

    return Function(f, uses=tuple(uses))


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return Marker(name)
