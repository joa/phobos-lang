"""Every ```python fence on the website's Python pages is a module whose
kernels compile, through the front end and `phobos-compile` at sm_75, as
`site -- check` compiles the `phobos` fences. A `python fragment` fence is
part of a module and is not compiled; a `python expect-error` fence must
fail, in the front end or the compiler."""

import re
import tempfile
import unittest
from pathlib import Path

from phobos import compiler, frontend

PAGES = Path(__file__).resolve().parents[2] / "website" / "src" / "content" / "docs" / "docs" / "python"
FENCE = re.compile(r"^```python([^\n]*)\n(.*?)^```", re.M | re.S)


def fences():
    for page in sorted(PAGES.rglob("*.md*")):
        text = page.read_text(encoding="utf-8")
        for m in FENCE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            yield f"{page.relative_to(PAGES)}:{line}", m.group(1).split(), m.group(2)


def compile_module(name, text):
    """Compiles the kernels `text` spells; raises on the first error."""
    program = frontend.translate(name, text)
    if not program.kernels:
        raise ValueError("no `@ph.kernel` in it; mark a fence without one `python fragment`")
    with tempfile.TemporaryDirectory(prefix="phobos-site-") as d:
        ph, origin = Path(d) / "fence.ph", Path(d) / "fence.origin.json"
        ph.write_text(program.text, encoding="utf-8", newline="\n")
        origin.write_text(program.origin_json(), encoding="utf-8", newline="\n")
        compiler.run([str(ph), "--chip", "sm_75", "--origin", str(origin), "-o", str(Path(d) / "fence.ptx")])


class Site(unittest.TestCase):
    def test_python_fences_compile(self):
        found = list(fences())
        self.assertGreaterEqual(len(found), 10, f"no Python pages under {PAGES}")
        failures = []
        for where, info, text in found:
            if "fragment" in info:
                continue
            try:
                compile_module(where, text)
            except (frontend.FrontendError, compiler.CompileError, ValueError) as e:
                if "expect-error" not in info:
                    failures.append(f"{where}: {getattr(e, 'stderr', e)}")
            else:
                if "expect-error" in info:
                    failures.append(f"{where}: an `expect-error` fence compiled")
        self.assertFalse(failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
