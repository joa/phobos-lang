"""What `@ph.kernel` and `@ph.fn` make of a Python function: the Phobos it
spells, compiled on first use for the card, and launched from the contract."""

import json
import tempfile
from pathlib import Path

from . import compiler, frontend


def _module_text(fn):
    path = fn.__code__.co_filename
    try:
        return path, Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise compiler.CompileError(f"`{fn.__name__}`: no source to read at {path}: {e}") from None


def translate(path, text, only=None):
    """[`frontend.translate`], its errors rendered as the compiler renders
    its own and raised as a `CompileError`."""
    try:
        return frontend.translate(path, text, only)
    except frontend.FrontendError as e:
        raise render(e) from None


def render(e, error_format="human"):
    """The `CompileError` a front end's diagnostics make."""
    with tempfile.TemporaryDirectory(prefix="phobos-") as d:
        diag = Path(d) / "errors.json"
        diag.write_text(json.dumps({"file": e.path, "text": e.text, "diagnostics": e.diagnostics}), encoding="utf-8")
        try:
            compiler.run(["--render", str(diag)], error_format=error_format)
        except compiler.CompileError as rendered:
            return rendered
    return compiler.CompileError(str(e))


class Function:
    """A Phobos `fn`; it is written into every kernel of its module."""

    def __init__(self, fn, uses=()):
        self.fn, self.uses = fn, uses
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __call__(self, *args):
        raise TypeError(f"`{self.__name__}` is a Phobos `fn`; call it from a kernel")


class Kernel:
    """A Phobos kernel, launched as `kernel[grid](*args)`, each `@autotune`
    constant at its first choice unless a launch pins it."""

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self._program = None
        self._compiled = {}

    @property
    def program(self):
        """The Phobos program this kernel spells, with its span map."""
        if self._program is None:
            path, text = _module_text(self.fn)
            self._program = translate(path, text, only=self.fn.__name__)
        return self._program

    @property
    def source(self):
        """The Phobos text this kernel spells."""
        return self.program.text

    def compile(self, chip=None, capability=None, index_bits=None, pins=None):
        """The kernel compiled for the target at the `@autotune` pins given,
        each other constant at its first choice. A compile is kept."""
        key = (chip, capability, index_bits, tuple(sorted((pins or {}).items())))
        if key not in self._compiled:
            with self.files() as (ph, origin):
                self._compiled[key] = compiler.compile_file(ph, chip, capability, index_bits, pins, extra=["--origin", str(origin)])
        return self._compiled[key]

    def files(self):
        """The program and its origin written out, for the compiler."""
        program = self.program

        class Files:
            def __enter__(self_):
                self_.dir = tempfile.TemporaryDirectory(prefix="phobos-")
                ph = Path(self_.dir.name) / f"{self.__name__}.ph"
                origin = ph.with_suffix(".origin.json")
                # Byte offsets index the text as written, so no newline
                # translation.
                ph.write_text(program.text, encoding="utf-8", newline="\n")
                origin.write_text(program.origin_json(), encoding="utf-8", newline="\n")
                return ph, origin

            def __exit__(self_, *exc):
                self_.dir.cleanup()

        return Files()

    def __getitem__(self, grid):
        def launch(*args, stream=0, pins=None):
            from . import driver

            compiled = self.compile(capability=driver.compute_capability(), pins=pins)
            return compiled.kernel(self.__name__).launch(grid, *args, stream=stream)

        return launch

    def __call__(self, *args):
        raise TypeError(f"`{self.__name__}` is a Phobos kernel; launch it as `{self.__name__}[grid](*args)`")
