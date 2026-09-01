# Beat llama.cpp on Qwen3.8-27B-UD-IQ1_M: pp128 and tg128

Opened 2026-08-31 at f772026. RTX 2080 SUPER, 8 GiB, 496 GB/s.

## Definition of done

bench.py interleaved, same session, ratio > 1.0 on **both** pp128 and tg128,
holding across rounds rather than in a favourable one. The opponent's bars are
stable to four digits and are the numbers to clear:

| test | llama.cpp | phobos at open | need |
| --- | ---: | ---: | ---: |
| pp128 | 477.11 +/- 0.39 | 21.5 healthy, 4.3 degraded | 22x |
| tg128 | 21.95 +/- 0.01 | 11.07 healthy, 9.59 mean | 2.0x |

Standing constraint: **never run the 27B above tg128.**

## What the opening traces say

Model: 26.9B parameters, 64 layers, d 5120, ffn 17408, weights 6.22 GiB,
6.95 GiB free on the card. KV is 64 KiB a token, so it is not a factor.

**Prefill, one 128-row pass (nsys, `-p 128 -n 0 -r 1 --no-warmup`):**

| kernel | ms | share |
| --- | ---: | ---: |
| matmul_tc | 529.2 | 33.4% |
| iq1s_qdecode | 395.6 | 25.0% |
| iq2xxs_qdecode | 268.2 | 16.9% |
| q3k_qdot_matvec (head, m=1) | 69.6 | 4.4% |
| iq2s / iq1m / iq2xs / iq3xxs _qdecode | 203.7 | 12.9% |
| q8_mma | 28.2 | 1.8% |
| everything else | 87 | 5.5% |
| **total GPU** | **1582** | |

but pp128 wall is 5950 ms. **73% of a prompt pass is not kernel time.**

**Decode (nsys, `-p 0 -n 16 -r 1`), per token:**

| kernel | ms | share |
| --- | ---: | ---: |
| q3k_qdot_matvec (LM head) | 58.7 | 42% |
| iq1s_qdot_matvec | 34.6 | 25% |
| iq2xxs_qdot_matvec | 20.4 | 15% |
| everything else | ~25 | 18% |

## Three root causes, one shared mechanism

1. **The LM head is read over PCIe, every token.** 521 MiB in 58.7 ms is
   8.9 GB/s. The same kernel at the same shape does 3.26 ms / 170.9 GB/s
   standalone (see the `evicted-vs-slow-kernel` note): 18x. It is eviction, not
   a slow kernel, and 0dcc7d7's trim did **not** fix it -- this trace is the
   A/B `docs/HANDOFF.md` left open, and the answer is "still paged".

2. **The prompt scratch is what evicts it.** `project_raw_dense` allocates
   `k * strip` and `m * strip` per weight and releases them to a pool keyed on
   *exact length*, so every distinct shape becomes its own entry:
   **741-886 MiB measured**. 6.22 GiB of weights + 0.87 GiB of scratch against
   6.95 GiB free is over the line, and the driver answers by paging out the
   largest cold allocation, which is the 521 MiB head.

3. **`Pool::trim` is what makes pp128 bimodal.** Measured directly:

       warmup   [vram] trimmed 741 MiB
       pp128 rep 1/2: 30.150 s     <- re-allocates what the warmup trimmed
       [vram] trimmed 886 MiB
       pp128 rep 2/2:  5.923 s

   30.150 s is 4.25 t/s; the memory note's "pp128 steps down at round 4 and
   never recovers, 21.5 -> 4.3" is this, and nothing else. The trim hands
   ~800 MiB back and the next prompt pass has to `cuMemAlloc` it again on a
   card at 96%.

So one mechanism -- the prompt pass's dequant scratch -- causes the decode
gap, the prefill bimodality, and (via the dequant/matmul round trip) most of
the prefill gap.

## Budget arithmetic, so the target is not guessed

pp128 is 6.4 TOP of contraction. llama.cpp's 477 t/s is 23.9 TOP/s, 52% of
this card's f16-TC-f32-acc peak. A fused int8 path at the efficiency phobos
already measures standalone (47% of peak on `gemm_fp16tc_fp32acc`) is
~150 ms of MMA + 13.5 ms of weight read against a 268 ms budget: **1.8x of
margin.** A two-pass design cannot get there at any tuning level -- expanding
25e9 elements at the best rate the qdot family has ever shown is ~240 ms
before a single MMA runs.

tg128 needs 45.6 ms a token. Residency alone takes 90 -> ~54 ms (18.5 t/s,
which is what `docs/HANDOFF.md` recorded as 18.24 in a session where the head
happened to be resident). The last 1.2x has to come from the qdot family.

## Beams

### A. residency: one scratch buffer, no trim   [LANDED, measured]
Parent: HEAD. Hypothesis: the pool's exact-length keying, not the budget, is
what makes the scratch 800 MiB; one buffer sized to the widest `k * strip`,
held for the process, is ~131 MiB and removes the need to trim at all.
Changes: `project_raw_dense`'s alloc/release pair, `residency.rs`.
Predicts: head resident (q3k off 58.7 ms), tg 11 -> ~18, pp128 steady at
~21.5 with the 30 s rep gone. Does **not** on its own reach either goal.

**Result.** `dense_scratch(which, len)` in `residency.rs`, prefix-sliced,
grown with a stream drain, never released; `trim_after_dense` deleted.

| pp128, `-p 128 -n 16 -r 3` | rep 1 | rep 2 | rep 3 | mean |
| --- | ---: | ---: | ---: | ---: |
| before (`-r 2`) | 30.150 s | 5.923 s | -- | 21.5 healthy / 14.6 measured |
| after | **1.644 s** | 3.447 s | 3.469 s | **50.63 t/s** |

The 30 s rep is gone and pp128 is 2.4-3.5x up. A 2.1x step from rep 1 to
rep 2 remains and is not yet explained -- headroom after the change is only
~617 MiB against a 521 MiB head, so it is probably still on the residency
line rather than clear of it.

Correctness: `backend_check` is **numerically identical** to HEAD, all 198
ops to four digits. Note HEAD already fails one, pre-existing and unrelated:
`matmul_raw IQ4_XS [64 x 512 x 128] rel err 2.674e-2`, the f16-strip shape,
identical value with and without this change. Worth a separate look; IQ3_S at
the same shape is 4.044e-3, so that shape is the loosest path in the check.

### B. fused IQ qmma: decode inside the tensor-core matmul   [LANDED for IQ1_S]
Parent: `tile_qmma_t` (Q8_0) + `tile_iq1s_qdot_t`. Hypothesis: an IQ grid
entry is already i8, so the weight fragment for `mma.m8n8k16.s8` can be
decoded in registers and never touch global. Kills `iq1s_qdecode` +
`iq2xxs_qdecode` + `matmul_tc` (1193 ms of 1582) *and* the scratch that beam A
is working around, so it subsumes A.
The delta: IQ1 weights are `dl * (grid + delta)`, so the contraction needs
`delta * sum_k q[i,k]` beside the dot -- `qdot_i8_delta.rs` already does this
at m=1; the batched form needs an activation row-sum plane from
`quantize_act`.
Scope by share: IQ1_S (25.0%) then IQ2_XXS (16.9%); Q2_K and IQ4_XS are 1%
between them and keep `_dequant` forever.

**The delta turned out to be free, which was the main risk.** An IQ1_S grid
byte is -1, 0 or 1 and the delta is an eighth, so `dl * (g +- 1/8)` is
`(dl / 8) * (8g +- 1)` and `8g +- 1` is an exact i8 in -9..9. No activation
row-sum plane, no Q8_1, none of what the `dp4a` path needs -- and the sign
becomes an index bit rather than arithmetic by carrying both foldings in the
table (`quant::iq1s_signed_grid`, 32 KB). Checked, not assumed: the grid's
distinct signed byte values across all 2048 entries are exactly {-1, 0, 1}.
IQ2_XXS is the same story with no delta at all, magnitudes {8, 25, 43} and a
sign, so it fits i8 too and is the next one.

The other fit is structural: IQ1_S's scale group is 32 elements and so is a
Q8_0 activation block, so both scales land on the same k step and the
accumulators stay in registers, which is exactly what `qmma_t`'s doc says is
the difference between 12.6 TOPS and 2.3.

**Built.** `tile_iq1s_qmma_t` in `phobos-lang/src/codegen/tile/iq1s_qmma.rs`,
`iq1s_qmma_t` in `call.rs`, the kernel source and tiles in
`kernels/iq1s.rs`, the table in `quant/iq1_s.rs`, dispatch in
`backend/device/qmma_raw.rs` behind `PHOBOS_RAW_QMMA` so the two paths can be
compared in one binary. Emits and verifies: 64 `mma.sync` a warp step, patch
resolved to rm=4, rn=8. Three codegen tests pass. `backend_check` gets a
`(128, 512, 128)` shape, the only one in `RAW_SHAPES` the fused path can
take, since without it the prompt path would go untested.

**The delta fold is checked exactly, host-only.**
`iq1s_folds_its_delta_into_an_exact_int8` rebuilds a block the way the kernel
does, `(dl / 8) * signed_grid[(idx * 2 + sign) * 8 + y]`, and asserts bitwise
equality against the reference dequantizer for all 256 elements. It passes,
so the table and the algebra are right whatever the kernel then does with
them.

**One defect caught before it cost a GPU run: the output went through a
shared tile.** `ptxas -v` on the emitted PTX said **220 registers and 32768
bytes of shared memory**, and 32768 is exactly a `[128, 64]` f32 tile.
`qmma_t` avoids this with a `stmt.rs` arm that rewrites
`C[slice] = qmma_t(..)` into `qmma_t_into`, and the new intrinsic had no such
arm, so it allocated the tile `qmma_t`'s own doc warns about. Adding the arm:

| | registers | shared | barriers |
| --- | ---: | ---: | ---: |
| through a tile | 220 | 32768 B | 1 |
| direct to global | **168** | **0** | 1 |

3 CTAs/SM rather than 2, no spills either way. The load histogram is what the
design predicts: 16 grid `b32`, 16 scale `b16`, 8 activation `b32`, 48 `b8` of
block bytes per k step.

### C. why `_qdecode` is 11x slower than `_qdot` on the same bytes   [ANSWERED]
Same IQ1_S weights: 34.6 ms through `iq1s_qdot_matvec`, 395.6 ms through
`iq1s_qdecode`. Both decode every element exactly once. If the gap is the
global write, beam B removes it for free; if it is the lane geometry, beam B
inherits the defect and has to fix it first. **Host-only PTX histogram
settles which, and it gates B's expected value.** Do this before B's tiling
is chosen.

**Verdict: write-side. B inherits nothing.** Full note in
`autoresearch/beams/iq1s-qdecode-gap.md`. The two k-loop bodies are **88 SASS
instructions each**, same 5 global loads per 8 elements, no barrier and no
shared memory in either, same 4 CTAs/SM. The only difference is 8 stores.
IQ1_S is 11.901e9 elements in 2.3245 GB; `_qdecode` additionally writes
23.803 GB. Both move DRAM at the same rate (67.2 against 66.0 GB/s), and
11.24x the bytes at 1.017x the cost per byte is 11.43x, the measured ratio.

Two riders, both recorded there: the emitted PTX is byte-identical to the
kernel-cache entries the model actually loaded, so this is what ran; and
`_qdecode`'s f16 store is half-sector, worth ~1.5x if a two-pass path ever
survives, but it dies with the store under B.

It also corrects a number this file had wrong. `qdot` runs at **344e9
elements/s**, not the ~54e9 the old docs imply, so B's decode is ~74 ms of
25.6e9 elements rather than 240, and B's margin is better than stated.

### Killed / not started
- Requantizing the head to make it fit: the goal names this file; a narrower
  head is a different model and the comparison would be invalid.
- An IQ fuse pass (`backend/fuse/`): capped at ~2 ms a token by launch-cost
  arithmetic, per `docs/PERF-QWEN27B.md`. Not the lever.

## Log

- 2026-08-31 opened; traces above taken at f772026 with the card at 818 MiB
  desktop, the quietest conditions in any recorded session.

- **Beam A landed on prefill, and did not move decode.** Traced at the same
  shape as the opening decode trace, `q3k_qdot_matvec` went 58.7 -> 46.9 ms.
  521 MiB in 46.9 ms is 11.1 GB/s: **still PCIe, still paged.**

- **The eviction victim moves with whatever is held.** In the same trace
  `delta_rule` went 20.3 us a launch to 737, min 14.8 us and max 1643 us,
  bimodal, on a kernel the change cannot touch. Holding 131 MiB through the
  decode steps simply chose a different loser.

- **Releasing the scratch when the pass shape changes is not the fix either.**
  `note_decode_projection` + release at the next pass boundary: pp128 rep 1
  38.4 s (the warmup's own decode step arms the release, so the first timed
  prefill pays the re-allocation), rep 2 1.789 s, tg128 **8.21 t/s**. Worse
  than the 11.07 it is trying to beat.

- **This kills the whole "the scratch evicts the head" theory, including the
  premise beam A was built on.** The decisive fact was in the opening trace all
  along: HEAD frees **741 MiB** before decode and its head is *still* paged at
  58.7 ms. 741 MiB is more than the head needs; if the scratch were what
  displaced it, HEAD would already be fast. Something else holds the headroom.

  Prime suspect, and HANDOFF's never-measured lever #1: **the loaded modules.**
  `resident_probe`'s header says the model path compiles two hundred kernels
  before it reaches one, and `mem_get_info` reports 0 MiB free with the weights
  at 6.22 GiB of a 6.95 GiB budget. Being measured now via `vram_mark` at three
  boundaries: context up, kernels compiled, weights uploaded.

  If modules are the holder, the fix is cheap and already half-built: the
  shape-change hook can `cuModuleUnload` the prefill-only modules
  (`*_qdecode`, `*_dequant`, `matmul_tc`) when decode starts, and the PTX cache
  makes reloading them milliseconds.

- **pp128 has a second bottleneck, not yet sized.** Beam A's steady reps are
  3.45 s wall against ~1.58 s of traced GPU, so ~1.9 s a pass is host-side --
  against a 268 ms budget for the whole row. Beam B cuts the node count hard
  (3 launches and a copy per strip become 1), but the host time has to be
  attributed before assuming B absorbs it. Open.

- **The pool's free list is where the headroom went, and it is now capped.**
  Attribution at a pass boundary, `PHOBOS_VRAM=1`:

      cuda context up:             7115 of 8192 MiB free
      kernels compiled and loaded: 7113 free   (-2 MiB)
      pass 1:                         0 free   (-7105 MiB)
        live 486 bufs 206 MiB, raw weights 6006 MiB

  **The loaded modules cost 2 MiB, not the 300+ HANDOFF's lever #1 assumed**,
  so that theory is dead, measured rather than argued. Of the 7105 MiB a pass
  commits, 6006 is raw weights and 206 is live buffers; the ~529 MiB left over
  is the pool's free list, which keys on exact length and so grows an entry per
  distinct shape the model ever asks for rather than per shape it needs at
  once. `Pool` now caps the free list at 96 MiB and hands the tail back to the
  driver.

  The arithmetic that says this is the whole game: weights 6370 + context 196 +
  desktop 881 is **7447 MiB before a single runtime buffer**, on a card whose
  measured eviction cliff sits between 7249 and 7761. llama.cpp fits the same
  file with a compute buffer of a couple of hundred MiB; phobos was asking for
  735.

- **`release_scratch_for_decode` removed.** It cost the first timed prompt pass
  38.4 s, because `bench`'s warmup ends in decode steps and so arms the release
  before the first pp rep. The pool cap frees ~430 MiB without that cost, which
  is the larger lever anyway.

- Open and not chased: the 1.64 -> 3.45 s step between beam A's rep 1 and
  rep 2.

## Measured, 2026-08-31 evening

**The fused IQ1_S projection is correct.** `backend_check`, the shape added for
it: `matmul_raw IQ1_S [128 x 512 x 128] rel err 3.265e-3`, against IQ3_S's
4.044e-3 on the neighbouring `[64 x 512 x 128]`. All ten IQ1_S rows pass.

Still failing, and **pre-existing**, unrelated to any of this: `matmul_raw
IQ4_XS` at `[64 x 512 x 128]` (2.674e-2 at HEAD, bit-identical with beam A
applied) and now also at the `[128 x 512 x 128]` shape this work added
(4.893e-2). IQ4_XS has no `_qdecode` and does not take the fused path; it is
the f16-strip arm that is wrong. Worth its own look.

**pp128 A/B, one binary, one session, `PHOBOS_RAW_QMMA` the only difference:**

| | fused ON | fused OFF |
| --- | ---: | ---: |
| pp128 | **96.76 +/- 0.42** | 77.81 +/- 0.53 |
| reps | 1.327, 1.319 s | 1.653, 1.637 s |

**1.24x from the fused kernel alone, and 96.76 against the 21.5 t/s this
started at.** The rest is beam A: the 30 s prompt pass is gone and the reps
agree to three digits, where they used to be 30.150 s and 5.923 s.

llama.cpp is 477.11, so pp128 is **0.20x**, up from 0.045x. Not met.

**tg128 regressed, and it was my doing.** 11.07 -> 6.00 (fused off) and 4.93
(fused on). The VRAM lines say why: `live 488 bufs 336 MiB` during decode,
where HEAD trimmed to ~78 MiB before the first decode step. Holding the
scratch across the shape change was the mistake; two attempts to fix it in the
wrong place both failed, and both failures are worth recording:

- **Releasing at a decode boundary faults.** `an illegal memory access was
  encountered`. A pass graph is cached and replayed and its kernel nodes carry
  raw device pointers, so freeing a pooled buffer after that graph exists
  leaves a replay reading memory the driver has taken back. The first pass
  after a dense one is the only safe point, which is where HEAD put it.
- **Capping `Pool`'s free list on every `put` faults for a different reason.**
  A caller releases a buffer *while the pass is still being recorded*, and the
  pool is what keeps it alive until those launches run; a cap that frees on
  `put` breaks that contract. The cap is also redundant once the trim is back,
  since the trim empties the free list at that boundary anyway. Reverted.

The configuration under test now is HEAD's trigger and HEAD's `Pool`, with
beam A's one-buffer sizing so the re-allocation the trim forces is a single
128 MiB `cuMemAlloc` rather than 741-886 MiB across seven entries. That is
what made the trim cost 30 s a prompt pass in the first place.

## What is left to reach the goal

pp128 needs 4.9x more. The fused path covers IQ1_S, which is 25.0% of a prompt
pass's kernel time and 49% of its parameters; `iq2xxs_qdecode` (16.9%),
`iq2s`/`iq1m`/`iq2xs`/`iq3xxs`/`iq3s` (12.9% between them) and the `matmul_tc`
they feed (33.4%) are all still on the expansion path. IQ2_XXS is next and its
i8 fit is already checked. Whether that is enough depends on the TOPS the fused
kernel actually reaches, which is not yet measured on its own -- `ppsweep` is
the place for that, and it is the next measurement, not another format.

tg128 needs the regression undone first, then 2x. The head is still read over
PCIe (46.9 ms a token at last trace), and the arithmetic that says why has not
moved: weights 6370 + context 196 + desktop 881 is 7447 MiB before a single
runtime buffer, against a cliff between 7249 and 7761.

## Settled, `-r 1`, the shape `bench.py` scores

`-r 2` and `-r 3` are unreadable on this card: the same row measured 6.2, 8.7,
15.5, 42.7, 56.9 and 85.9 seconds across reps. At `-r 1` in a fresh process the
numbers repeat to three digits, and that is also how `bench.py` runs a round.
One session, `bench -p 128 -n 128 -r 1`, each config twice:

| config | pp128 | tg128 |
| --- | ---: | ---: |
| what this replaced | 3.35, 18.84 | 7.52, 7.52 |
| **one buffer, no free-list trim, activation slots freed** | **82.32** | **8.19** |
| the same plus the fused IQ1_S projection | 101.78, 101.56 | 6.23, 6.23 |

**The middle row is better than what it replaced on both rows**, which is what
makes it the default. pp128 4.4x against the better of the two old readings and
26x against the worse; tg128 1.09x.

The fused projection is another **1.24x on pp128** and costs **1.31x on
tg128**, so it stays behind `PHOBOS_RAW_QMMA=1`. The decode cost is not the
kernel, which never runs at one row: it is the activation slot the fused path
takes per projection, `m * k` bytes, 163 of them for IQ1_S alone. They are
freed at the pass boundary now -- that alone moved tg128 from 4.28 to 6.23 --
but allocating and freeing 163 buffers a prompt pass still leaves decode worse
off than never taking them. **Quantizing the activation once per distinct
input rather than once per weight is the fix, and it is the next thing to do
on this beam**: in this architecture q/k/v share one input and gate/up share
another, so most of those slots are the same tensor quantized again.

## Against the goal

| | phobos now | llama.cpp | ratio |
| --- | ---: | ---: | ---: |
| pp128 | 82.3 (101.8 fused) | 477.11 | 0.17x (0.21x) |
| tg128 | 8.19 | 21.95 | 0.37x |

**Not met.** Both rows moved the right way and neither is close. What the
session establishes is why, and it is the same answer for both: the model is
6370 MiB of weights plus a 196 MiB context on a card whose desktop holds
881 MiB and whose eviction cliff sits between 7249 and 7761 MiB, so **7447 MiB
is committed before a single runtime buffer exists**. Every change measured
here trades one phase's allocations against the other's, which is what a card
past its cliff does. llama.cpp fits the same file because its compute buffer is
a couple of hundred megabytes against phobos's 500 to 700.

Ranked, what is left:

1. **The activation slots** (above). Cheap, bounded, and it is the only thing
   standing between the fused projection and a strictly-better default.
2. **The remaining six formats.** IQ1_S is 25.0% of a prompt pass and 49% of
   its parameters; `iq2xxs_qdecode` is 16.9%, the rest 12.9%, and the
   `matmul_tc` they feed is 33.4%. IQ2_XXS is next and its int8 fit is already
   checked: magnitudes {8, 25, 43} and a sign, no delta at all.
3. **Measure the fused kernel's TOPS on its own**, in `ppsweep`. 6.24 TOP in
   268 ms is llama.cpp's 23.3 TOPS; `q8_qmma` reaches 30.64 on Q8_0 weights
   that are 3.8x the bytes. Whether the fused path clears 23.3 decides whether
   finishing item 2 is enough, and it is one host-side kernel measurement.
4. **The allocator.** 500 to 700 MiB of exact-length pool entries against
   llama.cpp's couple of hundred is the whole residency story, and neither a
   cap nor a trim fixes it -- both were tried here and both faulted or
   regressed. An arena with stable reuse is the shape of the answer.

## The activation slots, half fixed

`Linear::project_shared` already carries a caller's pre-quantized `act` for the
int8 path, and `rms_norm_q` already leaves the rows quantized for QKV; the raw
arm threw it away, on a comment ("a raw kernel decodes to f32 ... has nothing
to offer it") that the fused projection made stale. Threading it through
(`Backend::matmul_raw_act`, defaulting to ignoring it, overridden on the
device) means the fused path takes the caller's copy where there is one instead
of a slot per weight.

Same session, each configuration twice, `-r 1`:

| | pp128 | tg128 |
| --- | ---: | ---: |
| default | 77.92, 77.66 | 6.04, 6.02 |
| fused | 100.18, 100.12 | 5.31, 5.31 |

The fused path's decode cost is **1.31x -> 1.14x**, prefill still 1.29x. Not
eliminated, because only QKV has a caller's `act` to share; the MLP and the
rest still reach `matmul_raw` through `project_into`, which has none. Giving
those a shared activation is the rest of this item.

**Note the session drift.** The default configuration read tg128 8.19 in the
previous session and 6.02 here, unchanged code. Only same-session ratios mean
anything on this card; see [[bench_r1_fresh_process]].

## Correction: the free list is 8 MiB, not 529

Earlier in this note I attributed ~529 MiB to `Pool`'s exact-length free list.
That was a residual from subtraction, not a measurement, and it is **wrong**.
Instrumented directly, every device-side pool at the pass after a prompt pass:

    live 488 bufs 336 MiB | raw 6006 | quants 28 | act 2 | fused 0 MiB
    free list: 2 distinct lengths, 8 MiB

The app holds about **6380 MiB**, of which ~6208 is weights (6006 raw, 28 Q8_0,
~174 f32) and only ~162 MiB is runtime buffers. `mem_get_info` reporting `0 of
8192 free` is not the app having taken 7105: WDDM holds roughly 900 MiB of its
own before the reported limit, which `docs/PERF-QWEN27B.md` had already
recorded and I did not apply. 6208 + 880 desktop + ~900 WDDM is **7988 of
8192**.

**So there is no allocator win here, and the size-class pool this note was
about to recommend would have bought nothing.** Freeing every runtime buffer
phobos has is ~160 MiB against a 521 MiB head. The remaining levers on
residency are the weights themselves or the desktop's 880 MiB -- not phobos's
allocator.

## What the arithmetic now says about the goal

**pp128.** Traced with the fused path on, `-p 128 -n 0 -r 1 --no-warmup`:
`iq1s_qmma` 199.1 ms for 3.05 TOP is **15.3 TOPS**, `q3k_qdot_matvec` 96.6 ms
for one paged launch, everything else ~90 ms. llama.cpp does the whole 6.24 TOP
pass in 268 ms, so 23.3 TOP/s *including* its head and attention. For phobos to
clear 268 ms with the head resident (~3 ms) and the rest at ~90, the
projections have to fit in ~175 ms, which is **35.7 TOPS across every format**.

The emitted balance says that is out of reach for this kernel shape: 1598
instructions a k step against 128 `mma.sync`, and the tile sweep (host-only,
four CTA widths and nine shapes) bottoms out at 12.5 instructions per tensor
instruction with 251 of 255 registers used and no spill. `q8_qmma` reaches
30.64 TOPS on the same tensor cores with the same 128-accumulator epilogue and
none of the decode, so ~25 TOPS is the plausible ceiling here. **Short by about
1.4x even after fusing every remaining format.**

**tg128.** Residency-bound, and per the correction above that is not a software
problem on this card at this desktop footprint.

Both rows are therefore blocked on something this beam cannot reach: prefill on
a kernel structure that would have to beat `q8_qmma` while doing strictly more
work, decode on 8 GiB of card holding a 6.2 GiB model behind an 880 MiB
desktop and a 900 MiB driver reserve.

## Correction, again: there *was* an allocator win, and it is the count

The section above concluded "there is no allocator win here" because the app
holds ~6380 MiB and the probe showed 6100 MiB of ballast running the head at
full speed. Both numbers were right and the conclusion was wrong, because
`resident_probe` allocated its ballast as **one** buffer and phobos allocates
**~1300**.

Giving the probe a second argument for the allocation count makes it visible.
Same total in every row, the output head's matvec at this model's shape:

| ballast | allocations | head matvec |
| ---: | ---: | ---: |
| 5500 MiB | 1 | 2.93 ms, 190 GB/s |
| 5500 MiB | 400 | 3.12 ms, 179 GB/s |
| 6000 MiB | 1 | 2.96 ms, 188 GB/s |
| 6000 MiB | 256 | 2.94 ms, 189 GB/s |
| 6000 MiB | 320 | 24.88 ms, 22 GB/s |
| 6000 MiB | 400 | 47.03 ms, 12 GB/s |
| 6000 MiB | 900 | **96.40 ms, 5.8 GB/s** |

96.40 ms is what the prompt trace measured for `q3k_qdot_matvec`. **The cliff
needs the footprint and the count together**: at 5500 MiB the count is worth 6%,
at 6000 MiB it is worth 33x. Testing one variable at a time is exactly what
made this invisible three times over -- first as the loaded modules, then as
the pool's free list, then as "no lever left".

`arena.rs` bump-allocates the bulk weights out of 128 MiB slabs, first fit
across all of them. 44 slabs, 6034 MiB held for 6006 used.

**Only the bulk weights.** The Q8_0 scale planes and the f32 constants are 644
allocations but only 200 MiB, and putting them in cost tg128 9.74 -> 8.65 ->
7.82: a hot 0.1 MiB plane in a 128 MiB slab drags the slab resident, where its
own allocation cost the driver almost nothing.

Measured with `PHOBOS_ARENA` so it is an A/B rather than a memory, one session,
each row twice:

| arena | fused | pp128 | tg128 |
| --- | --- | ---: | ---: |
| off | off | 78.6, 78.9 | 6.15, 6.16 |
| off | on | 106.1, 106.5 | 5.32, 5.32 |
| on | off | 80.8 | **7.83** |
| on | on | **111.8** | 6.44 |

1.27x on tg128, 1.03x on pp128, and it stacks with the fused projection.

**And the absolute numbers move with the desktop.** The same arena-on
configuration read tg128 9.74 with 880 MiB of desktop VRAM and 7.83 with 1102,
unchanged code. That is 1.24x sitting in whatever else is on the card, and it
is the one lever left that costs no engineering: at 880 MiB the model is 9.74,
and the trace arithmetic says a fully resident head is worth another ~1.5x on
top.

## The decode path was on the wrong kernel the whole time

With the head resident, a clean decode-only trace (`-p 0 -n 32 -r 1
--no-warmup`, no warmup so no prompt launches mixed in) is finally readable:

| kernel | ms a token | share |
| --- | ---: | ---: |
| iq1s_qdot_matvec | 36.9 | 41.6% |
| iq2xxs_qdot_matvec | 21.6 | 24.4% |
| iq2s / iq1m / iq2xs / iq3xxs | 19.8 | 17.6% |
| q3k_qdot_matvec (head) | **3.02** | 3.4% |
| rms_norm_q | 2.2 | 2.5% |
| delta_rule | **1.0** | 1.1% |

Two of these correct earlier readings from this same beam. The head is
**3.02 ms**, against the 2.94 ms `resident_probe` gets standalone: the arena
made it resident, and it is no longer a lever. And `delta_rule` is **20.7 us a
launch**, not the 905 the previous trace showed -- that trace included the
warmup's 128-row prompt launches averaged in with decode's one-row ones, so
"delta_rule is the new number one" was an artifact of tracing with a warmup.

What is left is 83.6% quantized matvecs, and `resident_probe` had the answer
sitting in its own output all along:

    iq1s     5120 17408  TN=8    0.268 ms   64.9 GB/s
    iq1s-i8  5120 17408  TN=64   0.134 ms  129.4 GB/s

**The dp4a int8-activation matvec is 2x the float one, and the model was not
using it.** `PHOBOS_IQ1S_DP4A` gates it, off "pending a whole-model quality
check", and nothing since had gone back to do the check.

    dp4a=0  pp128 82.89, 83.00   tg128 11.20, 11.20
    dp4a=1  pp128 82.95, 82.75   tg128 **18.05, 18.07**

**tg128 1.61x from one flag**, prefill untouched. That is 0.82x of llama.cpp's
21.95, from 0.34x at the start of this beam.

Quality, since that is what the flag was waiting on: the dp4a bodies agree with
the float ones they replace to **1.096e-6 (IQ1_S) and exactly 0 (IQ2_XXS,
IQ3_XXS)** on identical inputs, so the kernels are not the question -- the
question is quantizing the activation, which the f32 host reference does not
do. `argmax_check` on the 27B produces **identical tokens for all 32 greedy
decode steps** either way, and activation quantization is already what every
Q8_0 projection in this backend does by default.

## What pp128 would need, calibrated rather than guessed

Re-traced with the arena in and the fused path on, `-p 128 -n 0 -r 1
--no-warmup`. **1117.8 ms of kernels against a 1.116 s wall: prefill is now
entirely GPU-bound**, where it used to be 73% host. The 30 s prompt pass and
the 4.4 s of allocation churn are both gone.

| kernel | ms | share |
| --- | ---: | ---: |
| matmul_tc | 269.6 | 24.1% |
| iq2xxs_qdecode | 268.1 | 24.0% |
| iq1s_qmma (fused) | 201.6 | 18.0% |
| iq1m / iq2s / iq2xs / iq3xxs _qdecode | 206.9 | 18.5% |
| q3k_qdot_matvec (head, one launch) | 58.1 | 5.2% |
| q8_mma | 28.4 | 2.5% |

Note the head is **58.1 ms here and 3.02 ms in a decode step** -- the same
kernel at the same shape. The arena makes it resident for decode; during a
prompt pass the dequant scratch is live and it goes out again. Worth 5% of the
row, not the lever.

**The lever, and why it does not reach.** Emit both projections at the same
CTA and count, host-only:

| | mma | global loads | instructions | inst/mma | TOPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q8_qmma` | 64 | 44 | 634 | 9.9 | 30.64 |
| `iq1s_qmma` | 64 | 92 | 1090 | 17.0 | 15.3 |

Instruction count predicts throughput to within 15% across the pair, which is
this codebase's usual experience right up until an emulated instruction
appears. So the fused IQ kernel is 1.7x the instructions of a Q8_0 one that
does no decoding, and gets 0.5x the TOPS.

Now the budget. llama.cpp runs the whole 6.24 TOP pass in 268 ms, so 23.3
TOP/s **including its attention, its norms and its head**. For phobos to clear
268 ms with everything else at the ~80 ms it currently costs, the projections
have to fit in ~175 ms: **35.7 TOPS, or about 8.6 instructions a tensor op.**

That is fewer instructions than `q8_qmma` spends without decoding anything at
all. Fusing the remaining six formats at the rate `iq1s_qmma` achieves lands
at 6.24 / 15.3 + 80 = ~490 ms, about 260 t/s; even at `q8_qmma`'s own 30.6 it
is ~290 ms, about 440 t/s. **Neither clears 477.**

What that rules out is this kernel *shape*, not the goal: llama.cpp reaches
23.3 TOP/s whole-pass on the same file, so its MMQ is roughly 2x our fused
kernel. The structural difference is where the decode lands. This one decodes
into registers, so every warp that needs a column decodes it again and the
decode is 17 instructions a fragment; MMQ decodes into shared memory once a
CTA and reads fragments back with `ldmatrix`, which is one instruction. **A
shared-staging rewrite is the next thing on this beam, and it is the only
thing left that can move pp128 by the factor required.**

## Dead ends measured here, do not repeat

- **A wider CTA for the dp4a matvecs.** 1.19x standalone at `@launch(1024)`,
  0.91x in the model. Same shape as the larger-CTA result already on file for
  the float matvecs.
- **Pooling anything hot into the arena.** Q8_0 planes, f32 constants, the
  recurrent state: all three lose. Cold and bulk is the rule.
- **The remaining opt-in flags.** `PHOBOS_PERSIST_QDOT`, `PHOBOS_ATTN_PERSIST`
  (already on), `PHOBOS_FUSED` (Q8_0 only, never engages on an IQ file): all
  within noise of each other, 12.89 to 12.93. `PHOBOS_IQ1S_DP4A` was the only
  flag with a win behind it.

## The shared-staging rewrite: built, and worth 1.05x

Predicted at 1.13x on the kernel from the instruction count, measured 1.047x on
the whole prompt row, which is more than the count alone accounts for -- a
column now leaves DRAM once a CTA instead of once a warp, so the loads it saves
are worth more than their issue slots.

| | mma | global | shared | instructions | inst/mma |
| --- | ---: | ---: | ---: | ---: | ---: |
| registers | 128 | 104 | 0 | 1598 | 12.5 |
| staged | 128 | 72 | 20 | 1452 | 11.3 |
| `q8_qmma`, decoding nothing | 128 | 88 | 0 | 1268 | 9.9 |

**pp128 117.9, 117.6 against 112.6, 112.5**, each twice, `PHOBOS_QMMA_STAGE`
the only difference. Decode untouched at 8.22: this kernel never runs at one
row. `backend_check` gives the same 3.265e-3 to four digits, so the two forms
agree.

The structural note worth keeping: staging is only expressible where one warp
owns one patch. The register form carries its accumulators across `k` inside a
patch loop, and staging needs every warp at the same `k` at once, so the patch
loop has to go. `qmma_patch` caps a patch at `QMMA_TILES`, so a wider output
tile would take more patches than warps and the intrinsic refuses. That is the
same constraint that pins `q8_mma` at 2.3 TOPS against `qmma_t`'s 30.6, met
from the other side.

It closes about a third of the gap to `q8_qmma` and does not change the
conclusion above: the remaining two thirds is the epilogue, which both kernels
pay identically, and the per-group scale, which the format dictates.

## Fusing the second format, and the constraint that stops the third

IQ2_XXS fits the tensor cores for the same reason IQ1_S does -- magnitude times
sign, both already i8, an exact weight in -43..43 with no correction term -- and
reuses `iq2xxs_qdot_t`'s own decode helpers: `iq2xxs_lane` maps a lane index to
the geometry and a group's four lanes are `4 * ib + l`, so the staging loop
calls the same helper the matvec does.

    fused formats        pp128
    none                 80.6, 81.1
    IQ1_S                117.7, 117.6
    IQ1_S and IQ2_XXS    155.9, 156.7

1.33x on top of IQ1_S, 1.93x over the expansion path.

**IQ2_S and IQ2_XS decode identically and cannot use it.** Their scale is a
nibble, low for lanes 0 and 1 and high for 2 and 3, so they carry **two scales
per 32-element block** where this kernel shape assumes one -- and that
assumption is what keeps the accumulators in registers across `k`, which is the
whole difference between `qmma_t` at 30.6 TOPS and `q8_mma` at 2.3. Generated
from the macro anyway they read **2.5e-1** against the dense path where IQ2_XXS
reads 1.8e-3. Backed out.

The fix is available and costed: the two halves of a k step are already
separate `mma` operations at exactly the boundary the nibble changes on, so
scaling each half rather than their sum is correct, at the price of doubling an
epilogue that is a third of the kernel. Those two formats are 7% of a prompt
pass, so it wants measuring rather than assuming.

**The device oracle is what made this visible at all.** Against the host both
read about 2.4e-2 -- and so does *correct* IQ2_XXS, because the fused path
quantizes its activation and the host reference does not. A right kernel and a
wrong one are indistinguishable there. Against the dense path they replace,
device to device in one session, they are 1.8e-3 and 2.5e-1. That is the second
time this session the host has been the wrong oracle, after the `m = 1` dp4a
rows, and the second time an oracle change was the thing that unblocked a
measurement rather than a kernel change.

## Four formats fused, and one measurement still owed

The nibble scale is expressible after all. A k step is already two `mma`
operations divided on exactly the boundary the nibble changes on, so a split
format keeps an accumulator and a scale per half instead of chaining them --
chaining is only sound when the two halves share a scale, which is what the
macro's `split` flag now says. All four fused formats verify against the dense
path they replace:

    IQ1_S    1.984e-3
    IQ2_XXS  1.802e-3
    IQ2_S    1.998e-3   (2.529e-1 before the fix)
    IQ2_XS   1.679e-3   (2.496e-1 before)

250 ok, and the two IQ4_XS rows that fail at the parent commit.

**The speed of the four-format version is not measured, and should not be
guessed from the two-format one.** The card went from 974 MiB of desktop VRAM
to 1476 while this was being built, and at 1476 the fused path is *slower*:
52.4, 54.0, 54.5 against 75.7, 76.8, 76.9 with it off, three interleaved pairs,
reproducible. Two things changed at once -- the desktop and the format list --
so that number attributes to neither. The last clean reading, at 974 MiB with
two formats fused, was 156.7 against 80.6.

What it does suggest, and what the decode rows have said all session, is that
the fused path trades footprint for arithmetic: it takes an activation slot a
projection, and on a card this far past its cliff that is the more expensive
half of the trade. The measurement owed is the four-format A/B on a quiet card,
and a per-format toggle would make it attributable.

Where the prompt pass stood before this, traced with two formats fused, 932.1
ms total:

| kernel | ms | share |
| --- | ---: | ---: |
| iq1s_qmma + iq2xxs_qmma | 310.0 | 33.3% |
| matmul_tc | 132.6 | 14.2% |
| iq2s_qdecode | 99.4 | 10.7% |
| q3k_qdot_matvec (head, one launch) | 94.3 | 10.1% |
| iq1m + iq3xxs + iq2xs _qdecode | 178.3 | 19.1% |

IQ1_M has the same nibble pair and can follow the split path; IQ3_XXS has
IQ2_XXS's geometry outright. The head at 10.1% is still the second largest
single item and still costs what the dequant scratch keeps it from -- it goes
resident only when no format needs a scratch at all, which means Q2_K and
IQ4_XS too, and those have no `_qdecode` to build one from.

## Attributed: two formats pay, two do not

`PHOBOS_RAW_QMMA` takes a list now, which answers the question the previous
entry left open. One card state, `-p 128 -n 0 -r 1`, each row twice:

| fused | pp128 |
| --- | ---: |
| none | 71.8, 79.1 |
| IQ1_S | 114.3, 110.2 |
| **IQ1_S, IQ2_XXS** | **149.6, 149.4** |
| and IQ2_S, IQ2_XS | 135.3, 107.0 |

**Adding the two split-scale formats loses**, and the spread says why: three
digits of agreement for the pair, 135 against 107 for the four. That is
residency, not arithmetic. Fusing a format trades its expansion for an
activation slot a projection, and this card is far enough past its cliff that
the slots are the expensive half.

So the earlier entry was wrong to imply the split-scale fix would pay once it
was correct. It is correct -- 1.998e-3 and 1.679e-3 against the dense path, and
`backend_check` reads 250 ok with all four named against 248 with the default
two -- and it still loses. Both facts are kept, and the formats stay behind
their names.

**This also reframes what is left.** IQ1_M and IQ3_XXS are the next two
expansions by size, and there is now no reason to expect fusing them to pay
either: IQ1_M has the same nibble pair as the two that lost, and every fused
format adds slots. The thing that would change that is the activation slot
itself -- one quantized copy per distinct input rather than per projection --
which is already on this beam from the decode side and is worth more than any
further format.

Confirmed at the new default: **pp128 149.08, 149.21 against 79.07, 79.18**,
1.89x, at 1376 MiB of desktop VRAM. The same pair read 156.7 at 974 MiB.

## Reversed: the slots were the cost, not the kernels

The activation slot was the thing to fix, and fixing it inverts the entry
above. A pass takes a fresh slot per `quantize_act` because a caller may hold
the handle across many projections -- `rms_norm_q` does, for QKV and for gate
and up -- but the fused path has one caller that does not: the down projection,
whose input is the SwiGLU output and whose only reader is itself. At 128 rows of
a 17408-wide FFN that is 2.2 MiB a layer, 143 MiB across the model.

A recorded pass is a chain of kernel nodes in issue order, so the quantize that
overwrites a ring slot cannot run before the projection that read it. The ring
only has to exceed how many are live at once, which is one; it is four.

Same card, `-p 128 -n 128 -r 1`, each row twice:

| fused | a slot a projection | a ring of four |
| --- | ---: | ---: |
| none | 71.8, 79.1 | 79.2, 79.0 |
| IQ1_S, IQ2_XXS | 149.6, 149.4 | 151.8, 157.9 |
| **all four** | 135.3, 107.0 | **188.0, 183.8** |

The two split-scale formats went from losing to winning by a wide margin, and
the four-format spread went from 135-to-107 to three digits of agreement. So the
negative result was real and was about residency: 143 MiB of slots across the
model really was more than those kernels were worth. Default goes back to all
four. **pp128 188.0, 2.38x the expansion path and 0.39x of llama.cpp's 477.11.**

What this predicts for the two expansions still unfused, IQ1_M and IQ3_XXS, is
the opposite of what the previous entry predicted: their slots are now nearly
free, so they are worth the arithmetic they save. They are 19.1% of the traced
prompt pass between them.

## The ceiling, and where the fused kernel spends its issue slots

Arithmetic first, because it decides what is worth trying. The tensor histogram
of the file, host-only:

| type | tensors | elements | MiB | share |
| --- | ---: | ---: | ---: | ---: |
| IQ1_S | 163 | 11.90e9 | 2216.8 | 34.6% |
| IQ2_XXS | 110 | 6.54e9 | 1607.5 | 25.1% |
| Q3_K (`output.weight`) | 1 | 1.27e9 | 521.0 | 8.1% |
| IQ2_S | 25 | 1.65e9 | 504.5 | 7.9% |
| Q2_K (1.27e9 of it `token_embd`) | 8 | 1.39e9 | 435.6 | 6.8% |
| IQ1_M | 35 | 2.05e9 | 427.7 | 6.7% |
| IQ2_XS | 25 | 1.16e9 | 320.9 | 5.0% |
| IQ3_XXS | 21 | 0.74e9 | 271.8 | 4.2% |
| IQ3_S | 9 | 0.13e9 | 53.7 | 0.8% |
| Q8_0, F32, IQ4_XS, Q4_K | 454 | 0.05e9 | 47.6 | 0.7% |

The four fused formats are **72.6% of the bytes and 21.25e9 of the elements.**
A 128-row pass is 2 x 24.35e9 x 128 = **6.23 TFLOP** of projection, so:

- llama.cpp's 477.11 tok/s is 268 ms, **23.2 TOP/s end to end**.
- phobos at 188 is 681 ms, **9.1 TOP/s**.
- `iq1s_qmma` runs at **15.2 TOPS**, from two independent readings: 310.0 ms of
  trace over the 4.72 TFLOP the two fused formats are, and the 11.3 inst/mma
  calibration against `q8_qmma`'s 9.9 at 30.6.

So fusing what is left cannot get there on its own. At 15.2 TOPS the fused
72.6% alone is 358 ms and pp128 caps at 316 even if everything else were free.
**The fused kernel has to roughly double**, and q8-parity is only borderline:
5.44 TFLOP at 30.6 is 178 ms, plus ~69 ms of unfused `matmul_tc` plus attention
and norms and the head, which lands right at the line.

(Inference, not measurement: 23.2 TOP/s is above this card's 22.3 TFLOPS f16
with f32 accumulate, so llama.cpp is reaching the int8 tensor cores rather than
dequantizing to f16 and calling cuBLAS.)

### The SASS says the loop body, not occupancy

Host-only, `phobos-lang --example ptx` into `ptxas -arch=sm_75 -O3` and
`nvdisasm`. Both kernels at TM=128, TN=64, CTA=64, both doing **128
IMMA.8816.S8 per k iteration**, so the bodies divide straight across:

| | `q8_qmma` | `iq1s_qmma` staged |
| --- | ---: | ---: |
| k-loop instructions | 768 | 1074 |
| per mma | 6.0 | 8.4 |
| registers | 238 | 239 |
| memory ops in the loop | 56 | 92 |
| `LDG.E.U8` | 0 | **28** |
| `LDG.E.U16` | 0 | 16 |
| `LDG.E`, `LDG.E.64` | 56 | 24, 4 |
| `LDS.U`, `STS.64` | 0 | 16, 4 |
| `FMUL`/`FFMA`/`FADD` | 128/128/128 | 160/128/128 |

Three readings:

- **Occupancy is not a lever.** 239 registers pins the kernel at 65536/239, 8
  warps a multiprocessor, whatever the CTA is; `q8_qmma` reaches 30.6 TOPS at
  the same 238. A tile sweep is not worth a GPU run.
- **The 2x is the body.** 1.4x of it is the instruction count and the rest is
  stalls, since 92 memory instructions against 56 at 8 warps has nothing to
  hide behind.
- **28 byte loads a k step, at `[Rn+0x20]` and `[Rn+0x21]`.** That is the block
  `d`, an `f16` read as two `LDG.E.U8` and reassembled, and `d` is per 256
  elements while the loop steps 32: it is read eight times per block it belongs
  to. `ptxas` drops the macro's unread grid and sign loads, as the comment
  claims, but it does not common these up.

### The epilogue is the second half

Both kernels spend 384 float operations per 128 mma, three per accumulator
element per 32-element block: `I2F`, the scale `FMUL`, the `FFMA`. That is the
same 3:1 against the tensor work in both, and it is why `q8_qmma`'s 30.6 is a
ceiling rather than a target.

It is breakable. The per-32 weight scale is an **odd integer** in all four fused
formats (IQ1_S is `2*ls+1` up to 15; IQ2_XXS is `(2s+1)/8` with the eighth
folded into `d`), so it can be applied by an `IMAD` into an `i32` accumulator
and the float epilogue moved out to the 256-element block. Headroom: IQ1_S
reaches 4.4e6 in an `i32` over a 256 block and IQ2_XXS 4.3e7, both fine, and
IQ2_XXS only 2.4x clear if it were carried over the whole of `k` -- which is
the reason to stop at 256 rather than go per row.

## The tile is not the lever; the instruction count is, exactly

`PHOBOS_QMMA_TILE=TMxTNxCTA` overrides the fused tile, so the host-only
register and instruction sweep can be checked against the card. Four shapes,
`-p 128 -n 0 -r 2`, one session, fused path on:

| tile | warps a multiprocessor | inst/mma (host) | pp128 |
| --- | ---: | ---: | ---: |
| 128x64x64 (default) | 8 | 8.38 | **187.24** |
| 128x128x128 | 8 | 8.35 | 186.77 |
| 128x32x64 | 16 | 9.50 | 176.12 |
| 64x64x128 | 28 | 13.03 | 146.96 |

Fit `1/pp = T_other + T_k * (inst_per_mma / 8.38)` to the first and third rows
and the fourth predicts 148.4 against 146.96 measured, **within 1%**. So:

- **Time is linear in the fused kernel's instruction count, slope one.** Every
  instruction taken out of the k loop is time.
- **Occupancy does nothing.** Eight warps a multiprocessor and twenty-eight
  measure the same once the instruction count is divided out. `QMMA_TILES` caps
  a warp's patch at 64 tiles, which is 128 accumulator registers a lane and is
  what pins the register count, not the CTA shape.
- The split is **T_k 323 ms and T_other 361 ms** in a 684 ms pass, so the fused
  kernel is 47% of it and a free kernel would only reach 355 tok/s.

A correction to the entry above: the four fused formats reach **16.8 TOPS**, not
15.2, at this card state.

### Two things that did not work

`qh` is read as two `LDG.E.U8` because the tile language sees `tensor<i8>`. Read
instead as a two-element vector bitcast to `i16`, `ptxas` emits `LDG.E.U16`
plus a `PRMT` to split the halves back apart and the k loop grows from 1073
instructions to 1123: four fewer loads for fifty more instructions, which by the
law above is a loss. A single `u16` operand aliasing the same buffer would get
it, and is worth trying only after the epilogue.

Shrinking an activation slot when the batch shrinks (a prompt pass sizes every
slot at 128 rows and the decode steps after it carry that) measured **no change
at all** in `tg128`. Not committed.

## The opponent's numbers are not 477

`scripts/bench.py`, interleaved, 3 rounds, `-p 128 -n 128 -r 1`, fused path on:

| | phobos | llama.cpp | ratio |
| --- | ---: | ---: | ---: |
| pp128 | 196.84 +/- 1.30 | 79.12 +/- 69.78 | 2.49x |
| tg128 | 8.17 +/- 0.02 | 21.37 +/- 0.19 | 0.38x |

The llama.cpp `pp128` stderr is the whole story: **218.68 in round 1 and 9.52,
9.17 in rounds 2 and 3.** It has the card to itself in every round -- the
between-round check found nothing all three times -- so the collapse is what
running a second 6.4 GiB model on an 8 GiB card leaves behind, and it is not
something phobos earned. Round one is llama.cpp's honest `pp128` and **196.84
against 218.68 is 0.90x**, not 2.49x and not the 0.41x the 477.11 at the top of
this note implies. That 477.11 is a remembered figure from another session and
does not reproduce here; it should not be quoted again.

**So the goal is now one number.** `pp128` is within 10% and moving, and
`tg128` at 0.38x is where the work is. Per token that is 122 ms against
llama.cpp's 46.8, or **52 GB/s of weight against 137**, on a card that can do
496 and is paging because 6.4 GiB of weights plus 1.2 GiB of desktop do not fit
in 8.

## Decode: the output head is half of it, and only after a prompt pass

Two `nsys` traces, `--cuda-graph-trace=node --no-warmup`, both with the fused
path on, both steady state (the second half of the decode, so the load and the
first passes are outside the window):

| | `-p 0 -n 16` | `-p 128 -n 32` |
| --- | ---: | ---: |
| wall a token | 52.5 ms | 97.6 ms |
| GPU busy a token | 49.3 ms | 94.3 ms |
| **utilization** | **93.9%** | **96.6%** |
| `q3k_qdot_matvec` (the head) | **3.03 ms** | **47.19 ms** |
| `iq1s_qdot_i8_matvec` | 17.76 | 16.56 |
| `iq2xxs_qdot_i8_matvec` | 12.50 | 11.61 |

Every other kernel is the same to within a few percent. **The head alone is
15x, and it is 50% of a decode step.** 521 MiB at 47.19 ms is 11.6 GB/s, which
is the bus, not the card: the head is read over PCIe every token. With it
resident a decode step is 53.4 ms, which is **18.7 tok/s**.

So decode is not idle and not launch-bound. The 2.6 ms a token of gap in the
first trace is one `argmax_finish` to `copy`, the host reading the sampled
token, and there is nothing else. **The whole tg deficit is one evicted
allocation.**

### What it is not

- **Allocation count.** The run holds **2126 live `cuMemAlloc_v2`** against a
  cliff `resident_probe` puts between 256 and 320. Moving 644 of them into
  slabs (`PHOBOS_ARENA_CONST=1`) measures nothing: three interleaved rounds
  read 10.02/10.01, 8.21/11.26, 11.26/11.25. At 2126 the curve is flat; only
  getting under a few hundred could show.
- **The pool free list.** `PHOBOS_TRIM=1` reads tg128 11.14, 11.19 against
  11.27, 8.99 and costs pp128 208 -> 3.69 in one round. The free list is 8 MiB;
  it was never the problem.
- **Activation slots sized by the prompt.** Shrinking a slot when the batch
  shrinks: no change. Decode never asks for one, so it never fires.

### What it is

The card has **0 MiB free** from the moment the model is up. The weights are
6034 MiB of arena, the constants and Q8_0 planes about 200 more, and the live
pool is **336 MiB with a 127 MiB dequant scratch in it** -- against a desktop
that drifts around 1200 MiB of the 8192. The working set is over the card by
roughly the size of the head, so the head is what goes.

Two ways out, and they are the same work: fuse the formats that still need the
scratch, and put the pool on a diet.

## Where it stands: pp128 won, tg128 is 130 MiB short

`scripts/bench.py`, 4 interleaved rounds, `-p 128 -n 128 -r 1`, every default in
force:

| | phobos | llama.cpp | ratio |
| --- | ---: | ---: | ---: |
| pp128 | 193.28 +/- 0.30 | 9.04 +/- 0.03 | **21.4x** |
| tg128 | 18.02 +/- 0.01 | 21.66 +/- 0.19 | 0.83x |

llama.cpp's `pp128` is 9.04 in all four rounds, in the round it goes first as
well, and **9.04 +/- 0.01 run on its own with nothing else on the card** -- so
it is llama.cpp's number here and not something phobos left behind. The 218.68
it read once in an earlier session does not reproduce; neither does the 477.11
at the top of this note. Its `tg128` is rock solid at 21.66 either way.

### The tg deficit, measured

`resident_probe` holds ballast in 44 chunks, the arena's shape, and runs the
head's matvec against it. It prints what the ballast actually cost now, so the
knee reads off directly:

| free VRAM | `q3k_qdot_matvec` |
| ---: | --- |
| 276 MiB | 2.969 ms, 187.4 GB/s |
| 186 MiB | 3.020 ms, 184.2 GB/s |
| 40 MiB | 24.834 ms, 22.4 GB/s |
| 0 MiB | 58.032 ms, 9.6 GB/s |

The model runs at **0 MiB free**: peak 7844 of 8192 by `nvidia-smi`, of which
the desktop is 950 to 1070 and drifting. Solving for how much of the head is
paged from its traced 37 GB/s, against 178 MiB/ms resident and 11.1 paged,
gives **129.5 MiB**. That is the deficit, and it is the whole remaining gap:
with the head resident a decode step is 46.1 ms rather than 55.5, which is
**21.7 tok/s against llama.cpp's 21.66**. Parity, from residency alone.

The desktop proves the sensitivity on its own. Same binary, same defaults:

| desktop VRAM | tg128 |
| ---: | ---: |
| 910 MiB | 18.03, 18.03 |
| 941 MiB | 16.92, 16.69 |
| 1066 MiB | 12.90, 12.90 |

### What the kernels are worth, standalone

`resident_probe` on an empty card, against what the same kernels read inside the
model:

| kernel | standalone | in the model |
| --- | ---: | ---: |
| `q3k_qdot_matvec` (head) | 182-187 GB/s | **37** |
| `iq1s_qdot_i8_matvec` | 132-136 | 131 |
| `iq2xxs_qdot_i8_matvec` | 147 | 135 |
| `iq3xxs_qdot_i8_matvec` | 168 | 150 |

**Only the head is evicted.** Every other decode kernel runs at its own ceiling,
so there is no second residency problem hiding behind the first, and the qdot
family's 131-150 GB/s is a tuning gap rather than a defect. The tile sweep says
that gap does not open with the tile: see `qdot_i8_tn`.

### The 130 MiB, itemized

`PHOBOS_VRAM=1` now names every allocation over 8 MiB and every constant over 8
MiB. What is reducible, largest first:

- **72 MiB**: the delta net's 48 recurrent states, 3 MiB each in `f32`. Halving
  them is a numerics change and wants its own measurement.
- **29 MiB**: `blk.35.attn_v.weight` and `blk.51.attn_v.weight` are Q4_K, which
  has no device path at all, so they are dequantized to `f32` at upload and held
  dense: **40 MiB for 11 MiB of weight**. Either a Q4_K kernel or a
  requantization to Q8_0 at load gets it back.
- **23 MiB**: the dequant scratch, once IQ1_M, IQ3_XXS and IQ3_S are fused and
  only Q2_K and IQ4_XS are left to need one, at which point the budget can drop
  to 8 MiB for nothing.
- **28 MiB**: the arena's tails, 6034 MiB held for 6006 handed out.

That is 152 MiB against a 130 MiB deficit, so the target is reachable, but no
single item gets there and the desktop's own drift is the same size.

### Measured and not kept

- The tile and CTA of the decode matvecs (see `qdot_i8_tn`): standalone gains do
  not survive the model.
- Taking the dequant scratch before any weight is uploaded, so the driver never
  has to place it on a full card: tg128 16.95, 16.99 against 16.92, 16.69.
- `PHOBOS_SLAB_MIB` at 32, 64 and 256 against the 128 default: 10.03, 11.30 and
  10.03 against **12.90**, twice each. The default is already the best of the
  four, and the sweep is unusually reproducible, which is what a clean residency
  measurement looks like.
- Allocation-count granularity: 900 allocations of 72 KiB cost 66 MiB for 64
  asked, so the driver's page is small and the count is not a hidden footprint.

## Correction: llama.cpp's pp128 is bistable, and 470 is its healthy number

The entry above said `pp128` was won at 21.4x. **It is not.** The same
`scripts/bench.py`, four interleaved rounds, with IQ3_XXS and IQ3_S fused as
well:

| | phobos | llama.cpp | ratio |
| --- | ---: | ---: | ---: |
| pp128 | 214.87 +/- 1.06 | **469.91 +/- 0.50** | **0.46x** |
| tg128 | 16.46 +/- 0.49 | 21.66 +/- 0.01 | 0.76x |

llama.cpp's `pp128` swings between **9.04 +/- 0.01 and 469.91 +/- 0.50** and is
four-digit stable inside each state. The desktop is what moves it: 9.04 was read
at 1066 MiB of desktop VRAM, standalone with nothing else on the card, four
rounds running; 469.91 at 963. So it falls off the same residency cliff phobos
does, only in its prompt pass rather than its decode, and 469.91 is its honest
number. **The 477.11 at the top of this note was right and should be quoted
again.** The two entries that called it a remembered figure were wrong, and the
21.4x is an artifact of catching the opponent on the wrong side of its own
cliff.

What survives from that entry: phobos's `pp128` is stable where llama.cpp's is
not (+/- 1.06 across four rounds against a 52x swing), and every phobos number
in this note was measured on the same card in the same session as the llama.cpp
number beside it.

**So neither half of the goal is met.** `pp128` needs 2.2x and `tg128` 1.32x.
Where each stands:

- `pp128` 214.87 is 0.596 s a pass, **10.5 TOP/s** against llama.cpp's 22.9.
  The fused projection is 47% of it at 8.4 instructions an `mma`, and time is
  linear in that count, so the epilogue rewrite (int32 accumulation inside a
  256-block, worth about 2.5 instructions an `mma`) is the lever that is
  already scoped. The rest is the expansion path that is left -- Q2_K, IQ4_XS
  and Q4_K -- plus the output head, which a prompt pass pays too.
- `tg128` 16.46 is one evicted allocation and 130 MiB, itemized above.
