"""The autotuner's host side: comparing a variant's result, keeping the
winner, and compiling the space in one race-checked process. The timing
and the checks on the card are in `tests/gpu`."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phobos import autotune, compiler, frontend

HEADER = "from __future__ import annotations\n\nimport phobos.lang as ph\n\n\n"


class Compare(unittest.TestCase):
    def test_an_integer_output_matches_exactly(self):
        got = np.array([1, 2, 3], dtype=np.int32)
        self.assertIsNone(autotune.compare("C", "i32", got, [1, 2, 3], None, None))
        self.assertIn("1 of 3 elements differ", autotune.compare("C", "i32", got, [1, 2, 4], None, None))

    def test_a_float_output_matches_within_its_types_rounding(self):
        want = np.array([1.0, 2.0], dtype=np.float32)
        self.assertIsNone(autotune.compare("C", "f32", want * (1 + 1e-6), want, None, None))
        self.assertIn("past rtol", autotune.compare("C", "f32", want * (1 + 1e-3), want, None, None))
        self.assertIsNone(autotune.compare("C", "f32", want * (1 + 1e-3), want, 1e-2, None))

    def test_a_nan_where_the_reference_is_finite_fails(self):
        got = np.array([np.nan], dtype=np.float32)
        self.assertIn("non-finite", autotune.compare("C", "f32", got, [1.0], None, None))


class Store(unittest.TestCase):
    def test_a_winner_is_kept_per_card_and_key(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("PHOBOS_KERNEL_CACHE_DIR")
            os.environ["PHOBOS_KERNEL_CACHE_DIR"] = d
            try:
                result = autotune.Result(kernel="gemm", chip="sm_75", card="Card A", key={"M": 256}, cache_key="gemm-" + "ab" * 32,
                                         compiler_fingerprint="f", l2_flush="none", winner={"TILE_M": 64})
                autotune.store(result)
                autotune.store(autotune.Result(**{**result.__dict__, "card": "Card B", "winner": {"TILE_M": 128}}))
                self.assertEqual(autotune.load("sm_75", "gemm", result.cache_key, "Card A", {"M": 256}).winner, {"TILE_M": 64})
                self.assertEqual(autotune.load("sm_75", "gemm", result.cache_key, "Card B", {"M": 256}).winner, {"TILE_M": 128})
                self.assertIsNone(autotune.load("sm_75", "gemm", result.cache_key, "Card A", {"M": 512}))
                self.assertTrue(next(Path(d, "sm_75", "autotune").glob("gemm-*.json")))
            finally:
                if old is None:
                    del os.environ["PHOBOS_KERNEL_CACHE_DIR"]
                else:
                    os.environ["PHOBOS_KERNEL_CACHE_DIR"] = old


class Variants(unittest.TestCase):
    def test_the_space_compiles_in_one_race_checked_process(self):
        text = HEADER + """@ph.kernel(key=("N",))
@ph.autotune(BLOCK=[64, 256])
def scale(x: ph.tensor[ph.f32][N], y: ph.tensor[ph.f32][N]):
    base: ph.let = ph.ctx.block.x * BLOCK
    y[base:+BLOCK] = x[base:+BLOCK] * 2.0
"""
        with tempfile.TemporaryDirectory() as d:
            py = Path(d) / "scale.py"
            py.write_text(text, encoding="utf-8", newline="\n")
            program = frontend.translate(str(py), text)
            ph, origin = Path(d) / "scale.ph", Path(d) / "scale.origin.json"
            ph.write_text(program.text, encoding="utf-8", newline="\n")
            origin.write_text(program.origin_json(), encoding="utf-8", newline="\n")
            variants, skipped = compiler.compile_variants(ph, chip="sm_75", extra=["--origin", str(origin)])
        self.assertEqual(sorted(v["BLOCK"] for v, _ in variants), [64, 128, 256])
        self.assertEqual(skipped.strip(), "")
        for settings, compiled in variants:
            self.assertEqual(compiled.kernel("scale").contract.autotune, settings)


class Races(unittest.TestCase):
    def test_a_race_is_reported_at_the_python_lines(self):
        text = HEADER + """@ph.kernel
@ph.launch(64)
def mirror(X: ph.tensor[ph.f32][1, 64], Y: ph.tensor[ph.f32][1, 64]):
    t: ph.var[ph.tile[ph.f32][1, 64]] = 0.0
    t[0, ph.ctx.thread] = X[0, ph.ctx.thread]
    Y[0, ph.ctx.thread] = t[0, 63 - ph.ctx.thread]
"""
        with tempfile.TemporaryDirectory() as d:
            py = Path(d) / "mirror.py"
            program = frontend.translate(str(py), text)
            ph, origin = Path(d) / "mirror.ph", Path(d) / "mirror.origin.json"
            ph.write_text(program.text, encoding="utf-8", newline="\n")
            origin.write_text(program.origin_json(), encoding="utf-8", newline="\n")
            with self.assertRaises(compiler.CompileError) as e:
                compiler.run([str(ph), "--origin", str(origin), "--check-races", "-o", str(Path(d) / "m.ptx")])
        out = e.exception.stderr
        self.assertIn("a thread reads shared memory another warp writes", out)
        self.assertIn("mirror.py:11:27", out)
        self.assertIn("10 |     t[0, ph.ctx.thread] = X[0, ph.ctx.thread]", out)
        # The edit is Phobos, so Python gets the help alone.
        self.assertNotIn("barrier()\n", out)


if __name__ == "__main__":
    unittest.main()
