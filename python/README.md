# Phobos from Python

A Python function decorated `@ph.kernel` is a Phobos kernel, spelled in
Python with Phobos's semantics. The front end reads the function's source,
never runs it, and writes the Phobos it spells; every Python construct here
is a spelling of one construct in [SPEC.md](../SPEC.md), and the Python side
adds nothing to the language. `phobos-compile` compiles it, and a launch
binds the arguments to the launch contract the compiler writes beside the
PTX.

```python
from __future__ import annotations

import phobos.lang as ph


@ph.kernel
@ph.autotune(BLOCK=[16, 8192])
def saxpy(x: ph.tensor[ph.f32][N], y: ph.tensor[ph.f32][N], out: ph.tensor[ph.f32][N], alpha: ph.f32):
    base: ph.let = ph.ctx.block.x * BLOCK
    out[base:+BLOCK] = alpha * x[base:+BLOCK] + y[base:+BLOCK]


saxpy[lambda meta: (meta["N"] // meta["BLOCK"],)](x, y, out, 2.0)   # x, y, out: torch, CuPy, ...
print(saxpy.source)                                                  # the Phobos it spells
```

A module of kernels starts `from __future__ import annotations`, so Python
never evaluates a parameter's type and its dimensions need not be names
Python knows. Names resolve through the module's imports of `phobos.lang`:
`import phobos.lang as ph`, `from phobos import lang as ph`, or
`from phobos.lang import let, var, ...`.

The package needs no third-party module. It runs `phobos-compile`: `PHOBOS_COMPILE`, else
the one on the `PATH`, else this checkout's `target/release` build.

## Spellings

| Phobos | Python |
|---|---|
| `kernel k(A: tensor<f32>[M, K], alpha: f32) { .. }` | `@ph.kernel` on `def k(A: ph.tensor[ph.f32][M, K], alpha: ph.f32):` |
| `fn f(a, b) = e` | `@ph.fn` on `def f(a, b): return e` |
| `fn f(a, b) { .. }` | `@ph.fn` on `def f(a, b):` and the statements; a last bare expression is the value a `generate` takes |
| a name a `fn` body reads from its caller | listed: `@ph.fn(uses=("W", "C"))` |
| `@name`, `@name(1, k = v, x)` | `@ph.name`, `@ph.name(1, k=v, "x")`: a name is a string |
| `@autotune(X in [32, 256])` | `@ph.autotune(X=[32, 256])` |
| `let x = e`, `let x: T = e` | `x: ph.let = e`, `x: ph.let[T] = e` |
| `var x = e`, `var t: T = e`, `var t: T` | `x: ph.var = e`, `t: ph.var[T] = e`, `t: ph.var[T]` |
| `x = e`, `x[..] = e`, `x += e`, `x op= e` | the same |
| `tensor<f16>[M, 64]`, `tile<f32>[R, C]`, `f32` | `ph.tensor[ph.f16][M, 64]`, `ph.tile[ph.f32][R, C]`, `ph.f32` |
| `for i in range(lo, hi, s) { }` | `for i in range(lo, hi, s):`, and `range(hi)` from 0 |
| `for<unroll n> i in range(lo, hi) { }` | `for i in ph.unroll(n, range(lo, hi)):` |
| `while c { }`, `if c { } else if d { } else { }` | `while c:`, `if c: .. elif d: .. else: ..` |
| `break`, `continue`, `return` | the same |
| `A[i, j]`, `A[s : e]`, `A[s :+ n]`, `A[s:]`, `A[:e]`, `A[:]` | `A[i, j]`, `A[s:e]`, `A[s:+n]`, `A[s:]`, `A[:e]`, `A[:]` |
| `true`, `false`, `!a`, `a && b`, `a \|\| b` | `True`, `False`, `not a`, `a and b`, `a or b` |
| `-` `~` `+` `-` `*` `%` `&` `\|` `^` `<<` `>>` `==` `!=` `<` `<=` `>` `>=` | the same |
| `/` of floats; `/` of integers | `/`; `//` |
| `a < b && b < c` | `a < b < c` |
| `ctx.block.x`, `ctx.lane`, .. | `ph.ctx.block.x`, `ph.ctx.lane`, .. |
| a built-in or library function, `exp(x)`, `min(a, b)`, `f32(x)`, `softmax(t)` | `ph.exp(x)`, `ph.min(a, b)`, `ph.f32(x)`, `ph.softmax(t)` |
| `sum<over ctx.lane, width 8>(x)`, `reduce<over ctx.thread>(x, f)` | `ph.sum(x, over=ph.ctx.lane, width=8)`, `ph.reduce(x, f, over=ph.ctx.thread)` |
| `shfl<xor 16>(v)`, `shfl<from s>(v)` | `ph.shfl(v, xor=16)`, `ph.shfl(v, from_=s)` |
| `generate(f)` | `ph.generate(f)`, `f` a `@ph.fn` of the module |

`lang.pyi` lists every name `phobos.lang` holds, generated from the compiler
(`cargo run -p phobos-lang --features trace --example site -- stubs`); a test
fails while it lags.

A declaration names `ph.let` or `ph.var` and nothing else does: a bare
`x = e` on a name never declared is an assignment, and the compiler refuses
it at the Python line. `let` binds a value or a view of a tensor slice,
`var` a mutable value, a tile, or a copy of a slice into one.

## Where Phobos and Python differ

Each row is refused at the Python line, or computes as written and has a
test showing the value (`tests/test_semantics.py` on the host, from the
constants the compiler folds; `tests/gpu/semantics.py` on the card). Who
refuses: the front end, from the source, or the compiler, from the
operand types, through the spelling the span map carries.

| Python | Phobos | Here |
|---|---|---|
| `/` of two integers gives a float | divides as integers | refused by the compiler: write `//`, or convert an operand |
| `//` of integers floors: `-7 // 2` is `-4` | truncates | computes `-3` |
| `//` of a float floors | no floor division | refused by the compiler: write `/`, or `ph.floor` of it |
| `%` takes the divisor's sign: `-7 % 3` is `2` | takes the dividend's, as C's `fmod` | computes `-1`, and `7 % -3` is `1`; a float `%` of constants folds to `-7.5 % 2.0` = `-1.5`, but one computed at run time does not load yet: it lowers to `__nv_fmodf`, which nothing links |
| `a < b < c` | an error | `a < b && b < c`; `b` is written twice, safe since an expression has no side effects |
| a float literal is an `f64` | an `f32`, widened where it meets an `f64` | `0.1` into an `f64` tensor is `0.10000000149011612` |
| a float literal outside an `f32` | | refused by the compiler, as for a `.ph` |
| an integer literal past 32 bits | an index, as wide as the kernel's | refused by the compiler at a 32-bit index past what a 32-bit word spells (up to 4294967295, read as its bits), as for a `.ph` |
| integers never overflow | wrap at the index width at run time | `2147483647 + n`, `n = 1`, is `-2147483648` at 32 bits, `2147483648` at 64; a constant sum folds exactly |
| `and`, `or` stop early | evaluate both sides | `i < n and A[i] > 0.0` reads `A[i]` for every `i` |
| a free name in a `def` is the module's | resolves where the `fn` is called | must be listed in `uses`; one the module also binds is refused |

Python's other constructs (`**`, `@`, `is`, `in`, conditional expressions,
lambdas, comprehensions, tuples, strings, `with`, `try`, nested functions,
`*args`, defaults) have no Phobos counterpart and are refused where they are
written. A kernel calls the `@ph.fn`s of its own module: every one of them is
written into the program its kernels compile in.

## Errors

Every error, the front end's and the compiler's, is reported at the Python
line, laid out as rustc lays out its own, and so is everything else that
names a location: `--error-format json`, the race check (`--check-races`),
the launch report and the visualizer's traces. The front end writes a span
map beside the Phobos, and `phobos-compile --origin MAP` reads it.
Messages are the compiler's, so some still spell a construct in Phobos.
Python's own parser stops at the first syntax error.

`cargo run -p phobos-lang --example ph2py -- FILE.ph` prints a Phobos source
in this spelling; `tests/roundtrip.py` checks that every source in the tree,
the snapshot corpus and the kernel dumps compiles through Python to the
module it compiles to directly, under every target the emit sweep diffs.

## Launching

`phobos-compile -o K.ptx` writes `K.json` beside the PTX: for each kernel,
its `.entry` parameter slots (width, and the role each plays in a memref
descriptor or as a scalar), the index width, the CTA size, the static and
dynamic shared memory, the scratch it publishes, the barrier word, whether it
is persistent, the chip, the compiler's fingerprint and the cache entry. A
launch reads nothing else. `phobos-compile` reads and writes the runtime's
kernel cache, so a kernel the Rust side compiled is found here and the other
way round.

A tensor argument is any object with `__cuda_array_interface__`, row-major
and contiguous, of the declared element type, rank and fixed extents, its
base aligned as the kernel assumes. A dynamic extent is bound from the first
tensor that has it and must agree everywhere else, and must be a multiple of
4, the row-pitch ABI, or of what `@aligned` promises.

`kernel[grid](*args)` takes up to three block counts or a function of the
bound extents and the `@autotune` constants; a `@persistent` kernel takes
`kernel[()]` and sizes its own grid. A launch runs on the legacy default
stream unless `stream=` names another; an array's
`__cuda_array_interface__` stream is not waited on.

## Tests

- `python -m unittest discover -s python/tests`: no GPU. The launch
  contract against the arguments it binds, the semantics table and the UI
  sources (`tests/ui`, `PHOBOS_BLESS=1` reblesses).
- `python python/tests/roundtrip.py [DIR...]`: the round trip.
- On the card, `tests/gpu`: `replay.py` (the Python launcher against the
  Rust one, byte for byte, on recordings `lang_check --record` and
  `cross_record --record` make) and `semantics.py`.
