"""Binding arguments to a launch contract, with no GPU: what each slot
carries, and every argument the contract refuses."""

import struct
import tempfile
import unittest
from pathlib import Path

import phobos
from phobos.launch import LaunchError, bind, row_major

ROOT = Path(__file__).resolve().parents[2]


class Fake:
    """An array at a made-up device address: enough for binding."""

    def __init__(self, shape, typestr="<f4", ptr=0x7F0000000000, strides=None):
        self.__cuda_array_interface__ = {"shape": tuple(shape), "typestr": typestr, "data": (ptr, False), "version": 3, "strides": strides}


def descriptor(ptr, shape, index):
    words = [struct.pack("<Q", ptr)] * 2 + [struct.pack(index, 0)]
    words += [struct.pack(index, e) for e in shape] + [struct.pack(index, s) for s in row_major(shape)]
    return words


def compile_source(text, **target):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "k.ph"
        path.write_text(text)
        return phobos.compile_file(path, **target)


class Saxpy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = phobos.compile_file(ROOT / "examples/saxpy_fp32.ph", chip="sm_75").kernel().contract

    def test_three_descriptors_at_32_bits_then_the_scalar(self):
        x, y, out = Fake([1024], ptr=0x1000), Fake([1024], ptr=0x2000), Fake([1024], ptr=0x3000)
        bound = bind(self.c, [x, y, out, 2.5])
        want = descriptor(0x1000, [1024], "<i") + descriptor(0x2000, [1024], "<i") + descriptor(0x3000, [1024], "<i")
        self.assertEqual(bound.slots, want + [struct.pack("<f", 2.5)])
        self.assertEqual(bound.dims, {"N": 1024})

    def test_refusals(self):
        ok = [Fake([1024]), Fake([1024]), Fake([1024]), 1.0]
        cases = {
            "typestr": [Fake([1024], "<f2"), *ok[1:]],
            "rank": [Fake([32, 32]), *ok[1:]],
            "multiple of 4": [Fake([1022]), Fake([1022]), Fake([1022]), 1.0],
            "bound to 1024": [Fake([1024]), Fake([512]), Fake([1024]), 1.0],
            "16-byte": [Fake([1024], ptr=0x1004), *ok[1:]],
            "is f32": [*ok[:3], "1.0"],
            "takes 4": ok[:3],
            "__cuda_array_interface__": [[0.0] * 1024, *ok[1:]],
        }
        for what, args in cases.items():
            with self.subTest(what), self.assertRaisesRegex(LaunchError, what):
                bind(self.c, args)

    def test_a_strided_view_is_refused_and_a_contiguous_one_is_not(self):
        with self.assertRaisesRegex(LaunchError, "row-major"):
            bind(self.c, [Fake([1024], strides=(8,)), Fake([1024]), Fake([1024]), 1.0])
        bind(self.c, [Fake([1024], strides=(4,)), Fake([1024]), Fake([1024]), 1.0])

    def test_a_grid_function_sees_the_bound_symbols_and_constants(self):
        k = phobos.compile_file(ROOT / "examples/saxpy_fp32.ph", chip="sm_75", pins={"BLOCK": 256}).kernel()
        seen = {}
        grid = k.grid(lambda meta: seen.update(meta) or (meta["N"] // meta["BLOCK"],), {"N": 4096})
        self.assertEqual((grid, seen), ((16, 1, 1), {"N": 4096, "BLOCK": 256}))


class Gemm(unittest.TestCase):
    def test_the_tensor_core_gemm_at_sm_80_takes_64_bit_indices(self):
        c = phobos.compile_file(ROOT / "examples/gemm_fp16tc_fp32acc.ph", chip="sm_80").kernel().contract
        self.assertEqual(c.index_bits, 64)
        a, b, out = Fake([128, 64], ptr=0x10000), Fake([64, 96], ptr=0x20000), Fake([128, 96], ptr=0x30000)
        bound = bind(c, [a, b, out, 1.0, 0.0])
        want = descriptor(0x10000, [128, 64], "<q") + descriptor(0x20000, [64, 96], "<q") + descriptor(0x30000, [128, 96], "<q")
        self.assertEqual(bound.slots, want + [struct.pack("<f", 1.0), struct.pack("<f", 0.0)])
        # `@aligned(M = TILE_M)` at TILE_M's first choice, 32.
        with self.assertRaisesRegex(LaunchError, "multiple of 32"):
            bind(c, [Fake([100, 64]), b, Fake([100, 96]), 1.0, 0.0])

    def test_an_extent_past_a_32_bit_index_is_refused(self):
        c = compile_source("kernel k(A: tensor<i8>[N]) { A[0] = 1 }", chip="sm_75").kernel().contract
        with self.assertRaisesRegex(LaunchError, "32-bit index"):
            bind(c, [Fake([1 << 31], "|i1")])


class Persistent(unittest.TestCase):
    SRC = """@persistent(4)
        kernel k(X: tensor<f32>[64, 64], Y: tensor<f32>[64, 64]) {
            var S: tensor<f32>[64, 64]
            if ctx.block.x == 0 {
                S[0 :+ 64, 0 :+ 64] = X[0 :+ 64, 0 :+ 64]
            }
            grid_barrier()
            Y[0 :+ 64, 0 :+ 64] = S[0 :+ 64, 0 :+ 64] * 2.0
        }"""

    def test_scratch_and_the_barrier_word_follow_the_declared_arguments(self):
        k = compile_source(self.SRC, chip="sm_75").kernel()
        x, y = Fake([64, 64], ptr=0x1000), Fake([64, 64], ptr=0x2000)
        bound = bind(k.contract, [x, y], [0x9000, 0xA000])
        want = descriptor(0x1000, [64, 64], "<i") + descriptor(0x2000, [64, 64], "<i")
        want += descriptor(0x9000, [64, 64], "<i") + descriptor(0xA000, [2, 1], "<i")
        self.assertEqual(bound.slots, want)
        self.assertEqual(k.grid((), {}), (4, 1, 1))
        with self.assertRaisesRegex(LaunchError, "persistent"):
            k.grid((8,), {})


class Cache(unittest.TestCase):
    def test_a_compile_names_its_fingerprint_and_cache_entry(self):
        m = phobos.compile_file(ROOT / "examples/saxpy_fp32.ph", chip="sm_75")
        self.assertRegex(m.cache_key, r"^saxpy-[0-9a-f]{64}$")
        self.assertRegex(m.fingerprint, r"^[0-9a-f]+$")
        self.assertEqual(m.chip, "sm_75")

    def test_a_capability_picks_the_newest_chip_it_runs(self):
        m = phobos.compile_file(ROOT / "examples/saxpy_fp32.ph", capability=87)
        self.assertEqual(m.chip, "sm_86")


if __name__ == "__main__":
    unittest.main()
