"""What the Python spellings compute where Phobos and Python differ, each row
of the table in README.md shown on the host: the kernel is written in
Python, compiled to PTX, and the value read off the constant the compiler
folded, or the instruction it chose. The device runs of the same rows are in
`lang_check`."""

import re
import struct
import tempfile
import unittest
from pathlib import Path

from phobos import CompileError, compiler, frontend
from phobos.jit import translate

HEADER = "from __future__ import annotations\n\nimport phobos.lang as ph\n\n\n"


def ptx_of(body, kernel="k", chip="sm_75", index_bits=None):
    """The PTX `body`, a module after its imports, compiles `kernel` to."""
    with tempfile.TemporaryDirectory(prefix="phobos-sem-") as d:
        py = Path(d) / "sem.py"
        py.write_text(HEADER + body, encoding="utf-8", newline="\n")
        program = translate(str(py), py.read_text(encoding="utf-8"), only=kernel)
        ph, origin = Path(d) / "sem.ph", Path(d) / "sem.origin.json"
        ph.write_text(program.text, encoding="utf-8", newline="\n")
        origin.write_text(program.origin_json(), encoding="utf-8", newline="\n")
        return compiler.compile_file(ph, chip=chip, index_bits=index_bits, extra=["--origin", str(origin)]).ptx


def stores(ptx):
    """Each immediate stored to global memory, in order."""
    return [int(v) for v in re.findall(r"st\.global\.b(?:32|64)\s+\[[^\]]+\],\s*(-?\d+);", ptx)]


def bits_f32(x):
    return struct.unpack("<i", struct.pack("<f", x))[0]


class Division(unittest.TestCase):
    def test_floor_division_of_integers_truncates_toward_zero(self):
        ptx = ptx_of("""@ph.kernel
def k(I: ph.tensor[ph.i32][4, 4]):
    I[0, 0] = -7 // 2
    I[0, 1] = 7 // 2
    I[0, 2] = 7 // -2
""")
        # Python: -4, 3, -4.
        self.assertEqual(stores(ptx), [-3, 3, -3])

    def test_remainder_takes_the_dividends_sign(self):
        ptx = ptx_of("""@ph.kernel
def k(I: ph.tensor[ph.i32][4, 4], X: ph.tensor[ph.f32][4, 4]):
    I[0, 0] = -7 % 3
    I[0, 1] = 7 % -3
    X[0, 0] = -7.5 % 2.0
""")
        # Python: 2, -2, 0.5.
        self.assertEqual(stores(ptx), [-1, 1, bits_f32(-1.5)])

    def test_true_division_of_integers_is_refused_at_the_python_line(self):
        with self.assertRaises(CompileError) as e:
            ptx_of("""@ph.kernel
def k(I: ph.tensor[ph.i32][4, 4], n: ph.i32):
    I[0, 0] = n / 2
""")
        self.assertIn("this `/` divides two integers, as integers", e.exception.stderr)
        self.assertIn("sem.py:8:15", e.exception.stderr)

    def test_floor_division_of_floats_is_refused_at_the_python_line(self):
        with self.assertRaises(CompileError) as e:
            ptx_of("""@ph.kernel
def k(X: ph.tensor[ph.f32][4, 4]):
    X[0, 0] = X[0, 1] // 2.0
""")
        self.assertIn("this `//` divides floats", e.exception.stderr)
        self.assertIn("sem.py:8:15", e.exception.stderr)

    def test_floor_division_is_refused_on_every_path_that_types_it(self):
        # An op-assign, a tile, and a tensor-core fragment accumulator, which
        # the build updates in its registers.
        cases = {
            "x //= 2.0": ("""@ph.kernel
def k(X: ph.tensor[ph.f32][4]):
    x: ph.var[ph.f32] = X[0]
    x //= 2.0
    X[0] = x
""", "sm_75"),
            "t //= 2.0": ("""@ph.kernel
def k(X: ph.tensor[ph.f32][64, 64]):
    t: ph.var[ph.tile[ph.f32][64, 64]] = X[0:+64, 0:+64]
    t //= 2.0
    X[0:+64, 0:+64] = t
""", "sm_75"),
            "acc // l": ("""@ph.kernel
@ph.tensorcore
@ph.launch(128)
@ph.autotune(D=[64], BR=[32], BC=[32])
def k(Q: ph.tensor[ph.f16][N, D], K: ph.tensor[ph.f16][N, D], V: ph.tensor[ph.f16][N, D], O: ph.tensor[ph.f16][N, D]):
    q: ph.let = Q[0:+BR, :]
    acc: ph.var[ph.tile[ph.f32][BR, D]] = 0.0
    l: ph.var[ph.tile[ph.f32][BR, 1]] = 0.0
    for kt in range(0, N, BC):
        k: ph.let = K[kt:+BC, :]
        v: ph.let = V[kt:+BC, :]
        s: ph.var[ph.tile[ph.f32][BR, BC]] = ph.dot_t(q, k)
        l += ph.rowsum(s)
        acc += ph.dot(s, v)
    acc = acc // l
    O[0:+BR, :] = acc
""", "sm_80"),
        }
        for what, (body, chip) in cases.items():
            with self.subTest(what), self.assertRaises(CompileError) as e:
                ptx_of(body, chip=chip)
            self.assertIn("this `//` divides floats", e.exception.stderr)

    def test_true_division_of_floats_divides_as_floats(self):
        # A scalar index does not meet a float without a conversion.
        ptx = ptx_of("""@ph.kernel
def k(X: ph.tensor[ph.f32][4, 4]):
    X[0, 0] = ph.f32(7) / 2.0
""")
        self.assertEqual(stores(ptx), [bits_f32(3.5)])


class Literals(unittest.TestCase):
    def test_a_float_literal_is_an_f32_widened_to_f64(self):
        ptx = ptx_of("""@ph.kernel
def k(D: ph.tensor[ph.f64][4, 4]):
    D[0, 0] = 0.1
""")
        (value,) = stores(ptx)
        widened = struct.unpack("<d", struct.pack("<q", value))[0]
        self.assertEqual(widened, 0.10000000149011612)
        self.assertNotEqual(widened, 0.1)

    def test_literals_past_an_f32_or_a_32_bit_index_are_refused(self):
        with self.assertRaises(CompileError) as e:
            ptx_of("""@ph.kernel
def k(J: ph.tensor[ph.i64][4, 4], X: ph.tensor[ph.f32][4, 4], n: ph.i32):
    J[0, 0] = 5000000000 + n
    X[0, 0] = 1.0e39
    X[0, 1] = 1.0e-50
""")
        for message in ("integer literal `5000000000` does not fit a 32-bit index",
                        "float literal `1.0e39` is infinite as an `f32`", "float literal `1.0e-50` is zero as an `f32`"):
            self.assertIn(message, e.exception.stderr)

    def test_a_wide_literal_fits_a_64_bit_index(self):
        ptx = ptx_of("""@ph.kernel
def k(J: ph.tensor[ph.i64][4, 4], n: ph.i32):
    J[0, 0] = 5000000000 + n
""", index_bits=64)
        self.assertIn("5000000000", ptx)

    def test_run_time_integers_wrap_at_the_index_width(self):
        body = """@ph.kernel
def k(J: ph.tensor[ph.i64][4, 4], n: ph.i32):
    J[0, 0] = 2147483647 + n
"""
        self.assertRegex(ptx_of(body), r"add\.s32\s+%r\d+, %r\d+, 2147483647;")
        self.assertRegex(ptx_of(body, index_bits=64), r"add\.s64\s+%rd\d+, %rd\d+, 2147483647;")

    def test_constant_integer_arithmetic_folds_exactly(self):
        # Folded before the index narrows, so a constant sum does not wrap
        # where the same sum at run time would.
        ptx = ptx_of("""@ph.kernel
def k(J: ph.tensor[ph.i64][4, 4]):
    J[0, 0] = 2147483647 + 1
""")
        self.assertEqual(stores(ptx), [0, 2147483648])


class Comparisons(unittest.TestCase):
    def test_a_chained_comparison_is_both_comparisons(self):
        ptx = ptx_of("""@ph.kernel
def k(I: ph.tensor[ph.i32][4, 4]):
    I[0, 0] = ph.where(0 < 2 < 3, 1, 0)
    I[0, 1] = ph.where(0 < 5 < 3, 1, 0)
    I[0, 2] = ph.where(3 > 2 == 2, 1, 0)
""")
        self.assertEqual(stores(ptx), [1, 0, 1])

    def test_and_evaluates_both_sides(self):
        # The guard does not guard the load: `A[i]` is read for every `i`.
        ptx = ptx_of("""@ph.kernel
def k(A: ph.tensor[ph.f32][N], B: ph.tensor[ph.f32][N], n: ph.i32):
    i: ph.let = ph.ctx.thread
    if i < n and A[i] > 0.0:
        B[i] = 1.0
""")
        body = ptx[ptx.index("{"):]
        load = body.index("ld.global")
        self.assertNotIn("bra", body[:load])
        self.assertIsNone(re.search(r"@%p\d+\s+ld\.global", body))


class FreeNames(unittest.TestCase):
    MODULE = """@ph.fn(uses=("s",))
def scaled(x):
    return x * s


@ph.kernel
def twice(I: ph.tensor[ph.i32][4, 4]):
    s: ph.let = 2
    I[0, 0] = scaled(3)


@ph.kernel
def thrice(I: ph.tensor[ph.i32][4, 4]):
    s: ph.let = 3
    I[0, 0] = scaled(3)
"""

    def test_a_free_name_resolves_where_the_fn_is_called(self):
        self.assertEqual(stores(ptx_of(self.MODULE, "twice")), [6])
        self.assertEqual(stores(ptx_of(self.MODULE, "thrice")), [9])

    def test_an_unlisted_free_name_is_refused(self):
        with self.assertRaises(CompileError) as e:
            ptx_of(self.MODULE.replace('@ph.fn(uses=("s",))', "@ph.fn"), "twice")
        self.assertIn("`s` is read from where the `fn` is called", e.exception.stderr)

    def test_a_listed_name_the_module_also_binds_is_refused(self):
        with self.assertRaises(CompileError) as e:
            ptx_of("s = 5\n\n\n" + self.MODULE, "twice")
        self.assertIn("`s` is both a name of this module and one `scaled` reads from its caller", e.exception.stderr)


class Declarations(unittest.TestCase):
    def test_a_declaration_names_let_or_var(self):
        with self.assertRaises(CompileError) as e:
            ptx_of("""@ph.kernel
def k(X: ph.tensor[ph.f32][64, 64]):
    t: ph.tile[ph.f32][64, 64] = X[0:+64, 0:+64]
""")
        self.assertIn("a declaration says `ph.let` or `ph.var`", e.exception.stderr)

    def test_let_is_a_view_and_var_a_copy(self):
        program = frontend.translate("m.py", HEADER + """@ph.kernel
def k(X: ph.tensor[ph.f32][64, 64], Y: ph.tensor[ph.f32][64, 64]):
    v: ph.let = X[0:+64, 0:+64]
    t: ph.var = X[0:+64, 0:+64]
    u: ph.var[ph.tile[ph.f32][8, 8]]
    Y[0:+64, 0:+64] = v + t
""")
        self.assertIn("let v = X[0 :+ 64, 0 :+ 64]", program.text)
        self.assertIn("var t = X[0 :+ 64, 0 :+ 64]", program.text)
        self.assertIn("var u: tile<f32>[8, 8]\n", program.text)

    def test_a_span_and_a_range_stay_distinct(self):
        program = frontend.translate("m.py", HEADER + """@ph.kernel
def k(X: ph.tensor[ph.f32][64], Y: ph.tensor[ph.f32][64], s: ph.i32):
    Y[s:+16] = X[s:s + 16]
""")
        self.assertIn("Y[s :+ 16] = X[s : s + 16]", program.text)


if __name__ == "__main__":
    unittest.main()
