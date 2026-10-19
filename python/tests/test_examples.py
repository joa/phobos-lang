"""Every kernel under `python/examples`, the ports of `examples/` and of
Triton's tutorials, compiles and is race-free on the targets the GPU runs
use, at each `@autotune` constant's first choice. What they compute is
checked on the card, by `tests/gpu/tutorials.py`."""

import tempfile
import unittest
from pathlib import Path

from phobos import compiler, frontend

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class Examples(unittest.TestCase):
    def test_every_example_kernel_compiles_race_free(self):
        modules = sorted(EXAMPLES.rglob("*.py"))
        self.assertGreaterEqual(len(modules), 9)
        failures = []
        for py in modules:
            text = py.read_text(encoding="utf-8")
            program = frontend.translate(str(py), text)
            if not program.kernels:
                continue
            with tempfile.TemporaryDirectory(prefix="phobos-ex-") as d:
                ph, origin = Path(d) / f"{py.stem}.ph", Path(d) / f"{py.stem}.origin.json"
                ph.write_text(program.text, encoding="utf-8", newline="\n")
                origin.write_text(program.origin_json(), encoding="utf-8", newline="\n")
                for chip in ("sm_75", "sm_80"):
                    try:
                        compiler.run([str(ph), "--chip", chip, "--origin", str(origin), "--check-races", "-o", str(Path(d) / "out.ptx")])
                    except compiler.CompileError as e:
                        failures.append(f"{py.relative_to(EXAMPLES)} on {chip}:\n{e.stderr}")
        self.assertFalse(failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
