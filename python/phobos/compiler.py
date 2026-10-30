"""Running `phobos-compile`, the compiler a release ships.

The binary is `PHOBOS_COMPILE` when set, else `phobos-compile` on the PATH,
else the one beside `python/` in a release, else the one this checkout built
under `target/release`. It reads and writes
the runtime's kernel cache, so a kernel the Rust side compiled is a cache hit
here and the other way round.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .launch import Compiled


class CompileError(RuntimeError):
    """A compile that failed, with what the compiler printed."""

    def __init__(self, stderr, diagnostics=None):
        super().__init__(stderr)
        self.stderr = stderr
        self.diagnostics = diagnostics or []


def compiler():
    found = os.environ.get("PHOBOS_COMPILE") or shutil.which("phobos-compile")
    if found:
        return found
    root = Path(__file__).resolve().parents[2]
    name = "phobos-compile.exe" if os.name == "nt" else "phobos-compile"
    for exe in (root / name, root / "target" / "release" / name, root / "target" / "debug" / name):
        if exe.is_file():
            return str(exe)
    raise CompileError("no phobos-compile: set PHOBOS_COMPILE, put it on the PATH, or `cargo build --release -p phobos-compile`")


def target_args(chip=None, capability=None, index_bits=None):
    """The target flags: `chip`, else `PHOBOS_CHIP`, else the newest supported
    chip the card of `capability` runs; `None` for every one leaves the
    compiler's default."""
    chip = chip or os.environ.get("PHOBOS_CHIP")
    args = []
    if chip:
        args += ["--chip", chip]
    elif capability is not None:
        args += ["--capability", str(capability)]
    if index_bits is not None:
        args += ["--index-bits", str(index_bits)]
    return args


def run(args, cwd=None, error_format="human"):
    """Runs the compiler with `args` and returns its stdout. A failure raises
    with what it printed, uncolored; with `error_format="json"` that is one
    JSON object per error, which `CompileError.diagnostics` also holds."""
    proc = subprocess.run([compiler(), *args, "--color", "never", "--error-format", error_format],
                          capture_output=True, text=True, encoding="utf-8", cwd=cwd)
    if proc.returncode != 0:
        diagnostics = []
        if error_format == "json":
            for line in proc.stderr.splitlines():
                try:
                    diagnostics.append(json.loads(line))
                except ValueError:
                    pass
        raise CompileError(proc.stderr, diagnostics)
    return proc.stdout


def compile_file(path, chip=None, capability=None, index_bits=None, pins=None, extra=()):
    """`path` compiled to one module at the `@autotune` pins given, else each
    constant's first choice."""
    with tempfile.TemporaryDirectory(prefix="phobos-") as out:
        ptx = Path(out) / "kernel.ptx"
        args = [str(path), *target_args(chip, capability, index_bits), "-o", str(ptx), *extra]
        for name, value in (pins or {}).items():
            args += ["--set", f"{name}={value}"]
        run(args)
        return Compiled.load(ptx)


def compile_variants(path, chip=None, capability=None, index_bits=None, pins=None, check_races=True, extra=()):
    """Every `@autotune` variant of `path` left after `pins`, compiled in one
    process, race-checked unless `check_races` is off. Returns the settings
    and module of each variant that compiled, and the compiler's report of
    the ones it skipped."""
    with tempfile.TemporaryDirectory(prefix="phobos-") as out:
        stem = Path(out) / "kernel"
        args = [str(path), *target_args(chip, capability, index_bits), "--all-variants", "-o", str(stem), *extra]
        if check_races:
            args.append("--check-races")
        for name, value in (pins or {}).items():
            args += ["--set", f"{name}={value}"]
        proc = subprocess.run([compiler(), *args], capture_output=True, text=True)
        variants = []
        for line in proc.stdout.splitlines():
            ptx, _, settings = line.partition(" ")
            values = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(-?\d+)", settings))
            variants.append((values, Compiled.load(ptx)))
        if proc.returncode != 0 and not variants:
            raise CompileError(proc.stderr)
        return variants, proc.stderr
