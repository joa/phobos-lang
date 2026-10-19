"""Each `tests/ui/*.py` fails to compile, and what the compiler prints for
it, uncolored, against the `.py`, is the `.stderr` beside it.
`PHOBOS_BLESS=1` rewrites the `.stderr` files from what it prints now;
review the diff.

Each `.py` is `phobos-lang/tests/ui/*.ph` spelled in Python, printed by
`ph2py`, except the sources that do not parse, which have no Python
spelling to print and are written by hand with the same mistakes. A
printed one fails as its `.ph` does: the same errors, in the same order.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from phobos import compiler, frontend
from phobos.jit import render

HERE = Path(__file__).resolve().parent
UI = HERE / "ui"
PHOBOS_UI = HERE.parents[1] / "phobos-lang" / "tests" / "ui"

# Written by hand: their `.ph` does not parse, and Python's parser stops at
# the first syntax error where Phobos's recovers.
HAND_WRITTEN = {
    "assignment", "ctx_field", "for_range", "missing_paren", "recovery", "unclosed_brace", "unclosed_paren",
    "unexpected_char", "unknown_type",
}


def printed(py):
    """What compiling `py` prints, as the tests name it."""
    name = f"tests/ui/{py.name}"
    text = py.read_text(encoding="utf-8")
    try:
        program = frontend.translate(name, text)
    except frontend.FrontendError as e:
        return str(render(e).stderr)
    with tempfile.TemporaryDirectory(prefix="phobos-ui-") as d:
        ph, origin = Path(d) / f"{py.stem}.ph", Path(d) / f"{py.stem}.origin.json"
        ph.write_text(program.text, encoding="utf-8", newline="\n")
        origin.write_text(program.origin_json(), encoding="utf-8", newline="\n")
        proc = subprocess.run([compiler.compiler(), str(ph), "--origin", str(origin), "--color", "never", "-o", os.devnull],
                              capture_output=True, text=True, encoding="utf-8")
    if proc.returncode == 0:
        raise AssertionError(f"{name} compiles, but a UI test must fail")
    return proc.stderr


def headlines(text):
    return [line for line in text.splitlines() if line.startswith(("error", "warning"))]


class Ui(unittest.TestCase):
    def test_every_ui_source_prints_its_stderr(self):
        bless = os.environ.get("PHOBOS_BLESS", "").lower() in ("1", "on", "yes", "true")
        sources = sorted(UI.glob("*.py"))
        self.assertTrue(sources)
        failures = []
        for py in sources:
            out = printed(py).replace("\r\n", "\n")
            expected = py.with_suffix(".stderr")
            if bless:
                expected.write_text(out, encoding="utf-8", newline="\n")
                continue
            want = expected.read_text(encoding="utf-8").replace("\r\n", "\n") if expected.exists() else ""
            if out != want:
                failures.append(f"{py.name}: printed\n{out}\nbut {expected.name} holds\n{want}")
        self.assertFalse(failures, "\n".join(failures) + "\nPHOBOS_BLESS=1 rewrites the .stderr files")

    def test_every_phobos_ui_source_has_a_python_one(self):
        missing = sorted(p.stem for p in PHOBOS_UI.glob("*.ph") if not (UI / f"{p.stem}.py").exists())
        self.assertEqual(missing, [])

    def test_a_printed_source_fails_as_its_phobos_does(self):
        for py in sorted(UI.glob("*.py")):
            if py.stem in HAND_WRITTEN:
                continue
            with self.subTest(py.stem):
                phobos = (PHOBOS_UI / f"{py.stem}.stderr").read_text(encoding="utf-8")
                python = py.with_suffix(".stderr").read_text(encoding="utf-8")
                self.assertEqual(headlines(python), headlines(phobos))


if __name__ == "__main__":
    unittest.main()
