---
parent: fcfdf8b (HEAD, autoresearch branch)
status: killed as a grid-width lever; its own Result log's follow-on
  recommendation (a warp-scope parallelism primitive) landed as a genuine
  win, see the "2026-08-17" section at the end of the Result log
---

# Beam: cache-length-bucketed decode attention split count

## Hypothesis

`ATTN_SPLITS = 8` (`phobos-gguf/src/backend/device/kernels/attn.rs:462`) is a
global constant. minicpm5-1b's decode-attention grid is `(n_head / qgroup) *
splits = 8 * 8 = 64` blocks regardless of cache length, so it underfills the
card's 48 SMs worse as the cache grows and each split covers a bigger share of
it. This is a fixed grid against a growing amount of work per block, which is
consistent with phobos's measured tg slope degrading with cache length
(-8.7% tg32->tg2048) while llama.cpp's stays flat (-0.9%) on the same model.
The kernel's own doc comment says a cache-length-scaled split count is the
intended design, not implemented: see the excerpt in
`autoresearch/beams/BEAMS.md`.

Qwen3.5-0.8B is not expected to move much either way: it only takes this
kernel on 6 of 25 layers (the rest are SSM/delta-rule state), so it is not
where the win or the regression risk is. **Do not retune for Qwen; if a change
regresses Qwen, that is disqualifying, but Qwen is not the target.**

## What to change

1. Replace the flat `ATTN_SPLITS` constant with a function of cache length
   (`spec.total()` at the call site, `phobos-gguf/src/backend/device/attn.rs:
   168-225`, specifically line 182 `let splits = ATTN_SPLITS * qgroup;`).
   Bucket coarsely -- powers of two on `spec.total()` are the natural choice
   since the kernel doc already frames the cost as "a count that grows with
   the cache reshapes the pass every few dozen tokens, and each change costs a
   graph rebuild." Something like: splits doubles (8 -> 16 -> 32) as cache
   length crosses fixed thresholds. Do not make it track cache length exactly;
   that would rebuild the CUDA graph on every decode step, which is far worse
   than the problem being fixed (see `decode-step-is-launch-bound` memory: a
   graph rebuild is not free).
2. `with_kernel`'s cache key at `attn.rs:190-192` is currently `(spec.n_head,
   spec.group(), spec.head_dim)`. It has to grow to include whichever split
   count the bucket picked, or a second bucket silently replays the first
   bucket's compiled kernel (wrong split count baked into the PTX via
   `@autotune`, not just a runtime argument -- `attention_split_src` and
   `attention_merge_src`'s `S`/`splits` template parameter is compiled in).
   Get this wrong and it is a silent correctness bug, not a slowdown: the
   merge kernel's `S` has to match what the split kernel actually wrote or it
   reads uninitialized `P`/`ML` rows.
3. `attn_scratch` (`phobos-gguf/src/backend/device/mem.rs:79`) sizes the
   partials/stats buffers off `splits * n_head`. Check `grow_scratch`
   (`mem.rs:158`) handles the buffer growing when a bucket boundary is
   crossed for the first time in a session without breaking the "arenas grow
   only until they have seen the widest projection, so after the warmup no
   pass flushes" invariant (`graph.rs:36-40` comment) for steady-state
   replay once the largest bucket in a given run has been reached once.

## Correctness gate (non-negotiable, per CLAUDE.md)

`cargo run --release -p phobos-gguf --features cuda --example fuse_check` and
`backend_check` both have to pass before any timing from this beam counts. A
kernel that writes past its allocated output damages the *next* allocation and
not its own result, so op-level correctness is not sufficient on its own --
run a whole-model check too (`model_check`).

## How to measure

Fast iteration: `cargo run --release -p phobos-gguf --features cuda --example
attndecode`, minicpm5-1b row, across its full cache-length sweep (it uses
primes, deliberately -- see the file's own comment on why powers of two hide a
real regression). This isolates the attention kernel from the rest of a decode
step and is much cheaper than a full `bench.py` round.

Confirmation, same session, same card, interleaved against llama.cpp:

    python scripts/bench.py -m models/minicpm5-1b-Q8_0.gguf -p 128 -n 32 128 512 1024 2048 -r 3 -R 3

Compare against the fresh baseline in `autoresearch_baseline.csv` at repo root
(same invocation, captured before this beam's change). Also run the Qwen row
once as a regression check:

    python scripts/bench.py -m models/Qwen3.5-0.8B-Q8_0.gguf -p 128 -n 32 128 512 1024 2048 -r 3 -R 3

## Result log

**Verdict: killed.** The split count is not the bottleneck. Implemented,
passed every correctness gate, and still measured neutral-to-negative on the
real benchmark. Details below for whichever beam picks up decode attention
next.

### What was built

`attn_splits(spec, qgroup)` in `phobos-gguf/src/backend/device/kernels/attn.rs`:
`ATTN_SPLITS` (8) below a cache length of 256 (`ATTN_SPLIT_BUCKET`),
`ATTN_SPLITS_WIDE` (12, i.e. `S = 24` once multiplied by `qgroup = 2`) past
it, capped by a `ATTN_MERGE_TILE_BUDGET_BYTES` shared-memory budget. Folded
`splits` into `attention_decode`'s `with_kernel` cache key
(`(spec.n_head, spec.group(), spec.head_dim, splits)`) and widened
`split_attn`'s `HashMap` key type to match. `attn_scratch`'s existing
too-small check already tolerates the buffer growing at a bucket crossing
without flushing every replay after -- no change needed there.

All three correctness gates passed on this implementation: `backend_check`
(98/98 ops, worst rel err 2.9e-4, including attention at cache lengths 300,
512, 600 on both `16/2 x 128` and `8/4 x 256` shapes), `fuse_check` on both
models (`launched`/`fused` paths agreed, no decided flips), `model_check` on
both models ("backends agree"). `cargo clippy` clean, `source_size` test
untouched (files stayed at 569/275/385 lines, cap 900, no grandfathering
needed).

### The split-count sweep (`attndecode`, minicpm shape, warm card)

Global `ATTN_SPLITS` constant (no bucketing, so this is `S` at `qgroup = 2`),
`attn/step` at cache 521 / 1031 / 2053, all warm-card same-session numbers:

| `ATTN_SPLITS` (S) | 521 | 1031 | 2053 |
| --- | --- | --- | --- |
| 8 (16) baseline | 705.7u | 875.3u | 1113.2u |
| 10 (20) | 621.1u | 748.4u | 1045.2u |
| 12 (24) | 620.6u | 745.3u | 1043.4u |
| 14 (28) | 620.0u | 748.9u | 1047.6u |
| 16 (32) | -- | -- | 1441.7u / 1453.3u (two runs) |

S = 20 to 28 is a flat plateau, about 15% faster than baseline at 1031/2053
in isolation. S = 32 is a **cliff**, not a slope past the plateau -- worse
than the *baseline* S=16, not just short of the plateau's gain. This rules
out the brief's suggested "splits doubles" design (8 -> 16 -> 32): the second
doubling lands past the cliff. Picked `ATTN_SPLITS_WIDE = 12` (S=24), the
middle of the flat stretch, with margin on both sides.

Shared-memory note for whoever revisits this: the merge kernel's `[S, head_dim]`
tile failed to compile at exactly `S=48, head_dim=256` (49152 bytes, the
`STATIC_SHARED_LIMIT` constant with zero margin) even though the doc comment
on `ATTN_SPLITS` only named the `S=64` (65536-byte) case as the known wall.
The merge kernel's per-thread reduction scratch takes real room on top of the
tile; a cap needs headroom below `STATIC_SHARED_LIMIT`, not equality with it.

### The real benchmark: it does not reach `tg`

Three same-session interleaved `bench.py` runs, minicpm5-1b-Q8_0,
`-p 128 -n 32 128 512 1024 2048 -r 3 -R 3`, ratio = phobos / llama.cpp CUDA:

| test | baseline (2026-08-16 fresh) | bucketed (256/24) | wide-from-start (0/24, no crossing) |
| --- | --- | --- | --- |
| tg32  | 0.921 | 0.941 | 0.890 |
| tg128 | 0.920 | 0.920 | 0.876 |
| tg512 | 0.912 | 0.879 | 0.805 |
| tg1024| 0.892 | 0.892 | 0.727 (round 2 contaminated, see below) |
| tg2048| 0.862 | 0.863 | 0.822 |

tg32/tg128 never cross the 256-position bucket in this profile (`start_pos`
tops out at 128+32=160 and 128+128=256, and the bucket is `total() >
256` so 256 itself stays in the small bucket), so those rows run the exact
same kernel as baseline in both the bucketed and (mostly) the wide-from-start
build; the bucketed row's small uptick (0.921 -> 0.941) is session noise, not
signal -- confirmed by the wide-from-start row *also* using the unchanged
small-bucket kernel at these lengths yet reading lower (0.890, 0.876), in the
opposite direction. Take tg32/tg128 as noise-bounded and flat.

tg1024 and tg2048 are where the hypothesis should have shown up cleanest: the
wide bucket is active for nearly the whole run, the one-time graph rebuild at
the bucket crossing amortizes to nothing over that many tokens, and
`attndecode` said this shape should be ~15% faster. The bucketed build's
ratios (0.892, 0.863) are identical to baseline (0.892, 0.862) to three
decimal places. The kernel-level gain measured in isolation does not reach
the decode step at all.

tg512 is the one row that moved, and it moved the wrong way (0.912 ->
0.879): this run's cache crosses the 256 bucket about a quarter of the way
through a 512-token generation, and one CUDA graph rebuild at that crossing
(see `graph.rs`'s `reusable` check -- a changed `func` forces
`build_graph`) is a large enough one-time cost to show up over only 512
decode steps, while it is invisible by tg1024/tg2048's step count. Not
chased further since the beam's best case elsewhere already reads zero.

The wide-from-start build (`ATTN_SPLIT_BUCKET = 0`, so `S=24` from the first
decode token, no crossing anywhere) is the control for the crossing-cost
theory: same kernel as the bucketed build at tg1024/tg2048's cache lengths,
zero rebuilds, and it is worse than baseline on every row, not merely flat.
Round 2 of that run had a contaminated sample (tg1024 120.08 t/s, tg512
143.47 t/s, round-level -8.99% against its own median -- looks like a
thermal/power excursion, not the code path) and was excluded from that
conclusion; rounds 1 and 3 alone (tg1024 232-234 t/s, tg512 240-241 t/s) are
still below baseline (244.20, 249.48). So it is not the mid-run crossing
that costs tg512, and it is not the crossing's absence that would have made
tg1024/tg2048 win -- a wider decode-attention grid plainly does not pay
inside a real decode step.

### Why `attndecode` disagreed with `tg`

`attndecode` issues 24 back-to-back attention calls with nothing else in the
pass -- exactly the shape a wider grid helps, more concurrent blocks with
nothing else contending for the SMs. A real decode step interleaves
attention with the projections, norms and lm_head, and per the
`decode-step-is-launch-bound` memory the step is launch-bound with roughly
220 launches left post-fusion. Widening attention's grid from 128 to 192
blocks buys latency-hiding the step has no room to spend, and the merge
kernel's larger reduction is pure added cost. `attndecode` remains valid for
measuring the attention kernel itself; it is not a proxy for `tg`, and a
result from it needs the real benchmark before it counts as evidence, same
as CLAUDE.md's whole-model-check rule for correctness.

### Kill reasoning and what it rules out

Per `autoresearch/AGENT.md`: singleton (bucketed) and the plausible
combination-in-reverse (wide-from-start, isolating the crossing-cost
variable) both measured neutral-to-negative on the metric that matters.
Profile evidence (the flat plateau from S=20 to S=28, three times the
baseline's S=16, with zero gain reaching `tg`) says the grid width is not
what limits decode attention's contribution to a real step. Combining this
beam with anything else would need a reason to believe grid width matters
under some other condition not tested here; none is in evidence.

This is useful negative evidence for beam 3
([[launch-bound-headroom]] in `BEAMS.md`): if tripling the attention kernel's
own grid does not move `tg`, the step's cost is very likely concentrated in
launch/dispatch overhead or in the other kernels a step runs, not in
attention's occupancy. That beam's residual-launch-count question is now the
more promising next shot at minicpm's slope.

### 2026-08-16, later round: the isolated-vs-real contradiction, resolved

The user asked this round to resolve, before designing anything, why this
beam's own `attndecode` sweep found a real ~15% isolated gain from widening
`ATTN_SPLITS` (8->12, i.e. `S` 16->24) at cache 1031/2053, while the real
`bench.py` run above showed nothing at the same cache lengths. Rebuilt the
exact global (unbucketed) `ATTN_SPLITS=12` change this file already swept in
isolation, and instead of trusting `attndecode` or `bench.py` alone, measured
the same kernel inside a real, full-model, CUDA-graph-replayed decode step
with `nsys profile --trace cuda --cuda-graph-trace=node` (same method
`wide-vocab-lm-head.md` used), comparing `attention_split`'s own GPU-measured
duration baseline-vs-wide at the *same absolute cache depth* `attndecode`
tested.

**Method.** `cargo run --release -p phobos-gguf --features cuda --example
bench -- -m models/minicpm5-1b-Q8_0.gguf -p 0 -n 1024 -r 1 --no-warmup` run
under `nsys` for both `ATTN_SPLITS=8` (S=16, shipped baseline) and
`ATTN_SPLITS=12` (S=24, this beam's own "wide-from-start" build) trees, back
to back, same session, same card, uncontended
(`autoresearch/beams/attn_puzzle_{baseline,wide}_tg1024*`). `-n 1024` alone
(no `-p`, `--no-warmup`) decodes 1024 steps from a fresh 1-token prime, so
cache grows 1 -> 1024 across the run; the last 20 of 1025 captured steps
(cache ~1004-1024) are the steady-state window this beam's own isolated sweep
predicted a gain at (its own cache-1031 row). Per-step kernel-busy time
summed from `attention_split`/`attention_merge`'s own `Duration (ns)`
column, 24 layers per step; wall time from consecutive `rms_norm` (the
output norm, once/step) launch timestamps.

**Result 1: the kernel-level gain is real and reproduces inside the actual
CUDA-graph replay, at almost exactly the magnitude `attndecode` predicted.**

| | baseline (S=16) | wide (S=24) | delta |
| --- | --- | --- | --- |
| `attention_split`/step (24 calls) | 805,433 ns | 661,060 ns | **-17.9%** |
| `attention_merge`/step (24 calls) | 88,916 ns | 101,889 ns | +14.6% |
| combined (split+merge)/step | 894,349 ns | 762,949 ns | **-14.7%** |
| step wall time (avg, 19 steady steps) | 4,613,377 ns | 4,471,119 ns | -3.08% |
| combined share of step wall | 19.4% | 17.1% | |

`attndecode`'s own isolated prediction at cache 1031 was -14.99% combined
(875.3us -> 745.3us, per this file's earlier table) to -15.06% (re-measured
this round, 875.3 baseline vs 743.5 rebuilt-wide, same session as this
table). The in-pass measurement (-14.7% combined) lands inside a percentage
point of that. **This rules out both of the brief's leading candidate
explanations**: L2 cache pressure differing between an isolated repeated-call
benchmark and a real pass sharing L2 with ~290 other kernels, and CUDA graph
replay overlapping/serializing differently than a bare launch loop. Neither
would produce a real-pass kernel-duration delta this close to the isolated
prediction. `attndecode` is a trustworthy proxy for a kernel's own GPU
execution time and for the sign and rough magnitude of a code change's effect
on it, confirmed by measuring inside an actual graph-replayed pass rather than
assumed.

(Caveat on the wall-time row: it comes from two separate `nsys`-profiled runs,
not a single back-to-back session, so cross-run drift of a percent or two is
possible on that specific number the way `[[flash-attention-decode]]`'s own
"win size re-verified" section found; take "kernel-level delta confirmed
in-pass" as the solid claim and "-3.08% wall" as directionally right but not
pinned tighter than that by these two runs alone.)

**Result 2: the same two trees, timed cleanly (no `nsys`, no prefill,
back-to-back same session) at the actual benchmark's own tg1024/tg2048,
reproduce this beam's original null finding.**

| test | baseline (S=16) | wide (S=24) | delta |
| --- | --- | --- | --- |
| tg1024 | 239.43 +/- 0.95 | 240.57 +/- 0.64 | +0.48% (noise) |
| tg2048 | 231.74 +/- 1.94 | 229.12 +/- 0.48 | -1.13% (noise) |

(`autoresearch/beams/attn_puzzle_{baseline,wide}_bench.log`.) Both deltas sit
inside stderr. This is the same "identical to three decimal places" result
this file's kill decision already recorded, now reproduced in a fresh,
isolated A/B rather than inferred from the original bucketed/wide-from-start
runs.

**Result 3: reconciling 1 and 2 -- `tg` is a whole-trajectory average, and the
gain is real but concentrated where the trajectory barely visits.** `tg1024`
averages throughput over decode steps at *every* cache depth from 1 to 1024,
weighted by each step's own duration in the sum, not just the deep end
`attndecode` and the nsys window above measured. Attention is the one
per-step cost that scales with cache length (`BEAMS.md`'s own architectural
note): its share of a decode step is close to zero at cache ~1 and ~19% at
cache ~1024 (measured above), so the wide split's real ~15% kernel-level
saving is worth a genuine, measured ~3% of step wall time at the *deep* end of
the trajectory, but worth close to nothing at the shallow end, where most of
attention's own cost is fixed per-call overhead rather than data volume (this
file's own doc-comment evidence: "a cache with fewer tiles than pieces idles
the last blocks, and those are the caches whose attention is too cheap to
matter") and where the wider `attention_merge`'s own +14.6% cost has nothing
large to offset it against. Averaged over the whole 1..1024 (or 1..2048)
trajectory, this nets out to a fraction of a percent -- inside `bench.py`'s
own normal run-to-run stderr band, which is exactly what "ratio identical to
three decimal places" and this round's fresh "+0.48%/-1.13%, both noise"
look like empirically. **The tool is not broken; the metric is an average
that dilutes a deep-cache-only win.** A future decode-attention change with
the same shape (bigger at longer cache, small-to-negative at short cache)
needs `tg4096` or a direct in-pass nsys measurement at the cache length it
targets to be judged fairly -- averaging it into `tg1024` will understate it,
and `tg32/128/512` would nearly erase it.

**Real hardware counters, landed mid-round via the user's own elevated-shell
`ncu` run** (`ncu --set full -k regex:attention_split -c 5` /
`regex:attention_merge`, stock split+merge, minicpm shape, cache 4099, 5
launches averaged; reports at `autoresearch/beams/ncu_split_4099.ncu-rep` and
`ncu_merge_4099.ncu-rep`, readable from a normal shell with `ncu --import
<file> --page details`): `attention_split` (~86us/call, the dominant cost)
measures 38% of peak Memory Throughput, 14% of peak Compute (SM) Throughput,
63.8% Achieved Occupancy, grid (8,16,1) = 128 blocks, 256 threads/block, 64
registers/thread, Block Limit Registers = 4, Block Limit Warps = 4.
`attention_merge` (~7us/call) measures 5-13% Memory, ~1.5% Compute, 25%
Occupancy, grid = 16 blocks only.

Neither throughput number saturated alongside a mid-range occupancy is the
signature of a latency-bound kernel: enough warps resident to do real work,
mostly stalled rather than issuing at a high rate. The likely serializer is
`attention_split_src`'s per-tile online-softmax recurrence -- `rowmax` needs
the previous `m`, `exp` needs the new max, the rescale needs the `exp` --
which chains one key-tile's compute to the next with no independent work to
overlap it against. **This also explains this beam's own S=24 plateau and
S=32 cliff precisely**: sm_75 caps at 32 resident warps/SM; this kernel's
256-thread (8-warp) blocks hit that ceiling at exactly 4 blocks/SM, which is
*both* the register limit ncu reports *and* the architecture's absolute
maximum for this launch configuration, not merely a resource-driven ceiling
below 100%. S=24 (192 blocks / 48 SMs = 4.0 exactly) already reaches 100% of
the occupancy this launch shape can ever have on this card; S=32 (256 blocks
= 5.33/SM) cannot fit and forces a second grid-strided wave, which is why it
is a cliff and not a diminishing slope. **Widening the grid further is
therefore not merely unpromising, it is provably exhausted** -- there is no
more occupancy-driven latency-hiding left to buy with this kernel's current
per-block resource footprint.

### Two follow-on experiments this round, both negative, both reverted

Per the ncu diagnosis, the remaining lever (since occupancy is maxed) is
shortening the per-block *serial critical path itself* rather than adding
more resident blocks. Two cheap, in-language ways to do that were tried on
`attention_split` and measured in isolation (`attndecode`, minicpm shape)
before touching anything else:

1. **`@pipeline` on `attention_split`.** Already a supported phobos-lang
   attribute (`phobos-lang/src/codegen/pipeline.rs`), already used by
   `examples/flash_attention_fp32.ph` on a for-loop of exactly the same
   shape (leading `var`-staged static tensor slices, `dot_t`/`rowmax`/`exp`
   accumulate after). `attention_split`'s own `for kt in range(lo, full,
   BC) { var k = ...; var v = ...; ... }` loop already stages `k`/`v` with
   `var` (the doc comment above `attention_split_src` explains why, for a
   different reason -- vectorized loads), which happens to be exactly what
   `pipeline_candidate` wants, so this was a one-line attribute add. Result:
   flat, within noise, at every cache length (cache 1031: 867.3us vs the
   committed baseline's 873.2us, -0.7%; cache 37: 305.9 vs 305.4, +0.2%).
   Reverted, not committed. Consistent with the occupancy math above: sm_75
   has no `cp.async`, so `@pipeline` here can only double-buffer through
   registers/shared memory rather than overlap with real async-copy
   hardware, and the kernel is already at this card's occupancy ceiling, so
   there is no slack left for the extra live state to spend.
2. **Manual 2-way independent online-softmax chain inside one block.** Each
   block splits its own per-block key range into two disjoint halves,
   each carried by an independent `(m, l, acc)` triple, interleaved in one
   loop stepping `2*BC` per iteration (chain A's tile and chain B's tile
   both touched in the same iteration, so their loads/compute sit adjacent
   in the instruction stream), combined once after the loop via the same
   two-term online-softmax merge `attention_merge` already does across
   splits, before falling through to the existing single-key remainder loop
   and the existing `P`/`ML` writes -- fully expressible in today's
   language, no new primitive, the external split/merge interface and
   `attn.rs` call site untouched. Result: a clean **regression** at every
   cache length (cache 1031: 903.6us vs 873.2us baseline, +3.5%; cache 67:
   262.7us vs 235.2us, +11.7%; cache 4099: 2020.9us, no directly comparable
   baseline row but the same direction). Reverted, not committed. Diagnosed
   (from the occupancy math above, not separately re-profiled under this
   round's remaining budget): doubling the block's live `(m, l, acc)` state
   and doubling the simultaneously-staged K/V tiles (four live at once
   instead of two) raises register/shared-memory pressure on a kernel that
   is *already* sitting at this card's absolute 4-blocks/SM occupancy
   ceiling (32 warps/SM, the sm_75 maximum) -- any increase in per-block
   footprint can only push occupancy *below* that ceiling, and the added
   instruction-level parallelism was not enough to offset losing resident
   warps that were already hiding the same class of dependency-chain
   latency.

Net: two straightforward ways to shorten the per-block critical path without
a new phobos-lang primitive both failed to find headroom this round. Given
occupancy is provably maxed at 256 threads/block, the surviving unexplored
lever is a genuinely different unit of parallelism that buys independence
with *thread count* rather than *per-thread register/shared-memory
footprint* -- multiple warps per block, each owning a disjoint key
sub-range and its own accumulator, combined by a fast intra-block reduce at
the end, so the added parallelism costs occupancy nothing (this
architecture is warp-count-limited via block residency, not raw
thread-count) the way the register-heavy two-chain attempt above did. That
is a genuine new phobos-lang construct (warp-scope tile ownership /
block-thread-partitioning), not expressible in today's language without one,
and this round's remaining budget did not reach implementing it -- see
`autoresearch/beams/BEAMS.md`'s ranking update for how it stacks against the
concurrent shared-memory-pooling fix this round also found in progress on
`phobos-gguf/src/backend/device/attn.rs` /
`phobos-lang/src/codegen/{mod,tile/alloc}.rs` (not this beam's work; noted
here only so the next agent knows those files were mid-edit by another
process during this round and should be checked fresh rather than assumed
from this snapshot).

### Beam status: still killed as a standalone lever, but the puzzle behind its kill is now understood

The original kill verdict stands -- widening `ATTN_SPLITS` alone does not
move `tg`, and this round adds the reason why (trajectory-average dilution
of a real but cache-deep-concentrated gain, plus an occupancy ceiling this
card has already reached at the plateau this file found). Not worth
reopening as a grid-width lever; worth remembering as background for
whichever beam next changes `attention_split`'s own per-block work, since any
such change should be judged at `tg4096` or via in-pass `nsys`, not
`tg1024`/`tg32`, to see its true size.

### 2026-08-17: the follow-on recommendation landed -- warp-partitioned online softmax

This file's own previous section ends by naming the surviving lever: "a
genuinely new phobos-lang unit-of-parallelism primitive (warp-scope ownership
of a key sub-range, several independent warps per block instead of one wide
serial chain per block, so added parallelism costs thread count rather than
the per-thread register/shared-memory footprint that sank the two-chain
attempt)". Dispatched as its own round (`autoresearch/beams/BEAMS.md`'s "Beam
3"), and it landed as a genuine, large win. Full mechanism and thread-mapping
findings are in `flash-attention-decode.md`'s own new Result-log section
(this file records the numbers this beam's own recommendation predicted;
that file is where the code lives and where a future reader should start).

**What was found before writing anything.** Read the actual thread-to-work
mapping in `attention_split_src`'s loop body (source, not guessed): at
minicpm's shape (`QG=2`, `D=128`, `BC=16`), `dot_t`/`exp`/`dot` each
`distribute()` their small `[QG, BC]`-shaped work over only 32 of the block's
256 threads (one thread per output element, each doing a serial 128-deep FMA
reduction over the head dimension), `rowmax`/`rowsum` reach 64 via their
warp-shuffle path, and every one of those ops ends in a CTA-wide
`gpu.barrier`. Confirmed in emitted MLIR
(`cargo run -p phobos-lang --example emit` on a reduced kernel), not just
inferred from the DSL source, per the brief's own instruction. Net: six or
seven of the block's eight warps sat idle at a barrier for the entire loop,
exactly the "otherwise-idle warp" capacity this beam's own prior section
predicted existed but did not have a primitive to reach.

**What was built.** Not a general "warp-scoped mode" of `dot_t`/`rowmax`/
`dot` (too much blast radius -- every other kernel in the tree uses those).
Instead, one new, purpose-built `phobos-lang` builtin, `warp_partial`
(`phobos-lang/src/codegen/tile/warp_attn.rs`), that computes one warp's own
online-softmax partial entirely in registers: the head dimension is spread
across the warp's 32 lanes (`D / 32` elements each), one key at a time (no
more `BC`-wide tiling), the QK dot product is a short per-lane
multiply-accumulate followed by a five-step `gpu.shuffle` xor-butterfly
all-reduce (reusing the shuffle primitive `rowreduce_warp` already had, not a
new one), and the accumulator update needs no shuffle at all since each lane
only ever owns its own slice of `acc[i, :]`. No shared-memory staging of
`K`/`V`, and critically no CTA barrier inside the per-key loop -- the only
synchronization is the self-converging shuffle, which only needs the 32
lanes of the issuing warp, so different warps' independent loops (different
trip counts, since each warp's own `[lo, hi)` sub-range is a further
`ceil`-divided slice of the block's own range) can genuinely interleave on
the SM scheduler instead of converging on a shared barrier every iteration.
One CTA-wide barrier at the very end (after every warp's loop has finished,
called exactly once by all 256 threads regardless of their own warp's trip
count) publishes the `WCT` warps' partials to a small per-block shared
scratch (`wm`/`wl`/`wacc`, `[QG, WCT]`/`[QG, WCT]`/`[QG * WCT, D]`), which a
new small Rust-generated combine block (`attention_combine` in
`phobos-gguf/src/backend/device/kernels/attn.rs`) then folds with the exact
same `rowmax`/`exp`/`rowsum`/`dot` online-softmax merge `attention_merge`
already runs across splits -- just at `WCT` (8) width instead of `S`, so
`attention_merge` and its own shared-memory ceiling (this file's own earlier
section found it walling out around `S ~ 28`) are untouched: `S` (the
block-level split count) did not change, only what happens inside one
block's own share of the work.

`attention_split_src` and `attention_persist_src`'s phase one both changed
(the doc comment already promised phase one is the split kernel's body
verbatim; it still is). `attention_merge` and phase two are byte-for-byte
unchanged.

**Correctness, `PHOBOS_ATTN_PERSIST=1` forced on, both models:**
`backend_check` (worst relative error 2.902e-4, identical to the documented
baseline, every decode-shaped attention case at every previously-tested
cache length still passing), `model_check` (both "backends agree", spreads
in the same band as before), `batch_check` (both "batched and sequential
agree", spreads in the documented bands), `fuse_check` (both prompt passes
exact, decode spreads 8.1e-3 to 1.33e-2, zero top-token flips on either
model -- cleaner than the previously documented one tied flip on minicpm).
`cargo test -p phobos-lang` (148 pass, all pre-existing) and
`cargo test -p phobos-base -p phobos-gguf -p phobos-onnx -p phobos-inference
-p phobos-kernels` all pass; `phobos-base`'s `source_size` ratchet passes
(`expr.rs` trimmed by one comment line to stay at its grandfathered 916
after the one-line `warp_partial` dispatch addition; `kernels/attn.rs` grew
to 628 lines, still well under the 900 cap). `cargo clippy -p phobos-gguf -p
phobos-lang --features cuda -- -D warnings` clean. Every `.ph` under
`examples/` re-verified under both the default target and
`PHOBOS_CHIP=sm_80 PHOBOS_INDEX_BITS=64`, zero failures -- expected, since
the `phobos-lang` diff is provably additive (`git diff --stat` shows only
insertions in every existing file the change touches, and the new
`warp_partial` codegen path is unreachable from any kernel that does not
call it by name).

**`attndecode`, stock (non-persistent) split-plus-merge, both shapes, before
vs after (`us`):**

| cache | minicpm before | minicpm after | delta | Qwen before | Qwen after | delta |
| --- | --- | --- | --- | --- | --- | --- |
| 521  | 705.7 | 291.7 | **-58.7%** | -- | 90.3  | -- |
| 1031 | 875.3 | 380.0 | **-56.6%** | 282.9 | 122.5 | **-56.7%** |
| 2053 | 1113.2 | 550.9 | **-50.5%** | 372.9 | 187.3 | **-49.8%** |
| 4099 | -- | 894.3 | -- | -- | 317.3 | -- |

(minicpm/Qwen "before" figures at 1031/2053 from this file's own earlier
table and `flash-attention-decode.md`'s `attndecode` launched-path table;
521/4099 have no directly comparable prior row, included for completeness.)
A uniform ~50-59% reduction in the kernel's own GPU time, at every cache
length tested, on both shapes -- unlike the grid-width lever this file
killed, which only ever helped at deep cache and needed `tg4096`/in-pass
`nsys` to see past `tg`'s dilution. This one does not have that problem, see
the `bench.py` numbers below.

**`bench.py`, same session, interleaved against llama.cpp, 3 rounds x 3 reps,
`PHOBOS_ATTN_PERSIST=1`, uncontended every round:**

minicpm5-1b-Q8_0, against the freshest recorded `PHOBOS_ATTN_PERSIST=1`
baseline -- `BEAMS.md`'s "Current-state snapshot, 2026-08-17,
post-pooling-fix" section, landed concurrently with this round and marked
there as the reference point to compare future changes against, superseding
the slightly earlier `flash-attention-decode.md` pooling-landing numbers
(the two agree within session noise: 243.27 vs 243.44 at tg1024, etc., so
neither reading changes the conclusion, this just cites the one its own
commit asked future rounds to use):

| test | before (t/s) | after (t/s) | delta | ratio before | ratio after |
| --- | --- | --- | --- | --- | --- |
| tg1024 | 243.27 | 260.75 | **+7.2%** | 0.88x | **0.95x** |
| tg2048 | 231.12 | 255.76 | **+10.7%** | 0.85x | **0.94x** |
| tg4096 | 209.68 | 244.43 | **+16.6%** | 0.79x | **0.92x** |

Every row up, and the gain *grows* with cache length exactly as the
mechanism predicts (attention's share of a decode step grows with cache
length; this change cuts that share's own cost) -- the opposite of the
grid-width lever's shape, and no dilution to correct for: `tg1024` alone
already shows most of the win. minicpm closes from a widening 0.88x -> 0.85x
-> 0.79x slope (the worst ratio this session had measured, from the
snapshot this round found waiting) to a flat-to-improving **0.95x -> 0.94x
-> 0.92x** against llama.cpp's FA-on decode -- the closest this whole
session has gotten, and the slope this beam note opened with no longer
widens with cache length.

Qwen3.5-0.8B-Q8_0, same protocol, against the same fresh snapshot:

| test | before (t/s) | after (t/s) | delta |
| --- | --- | --- | --- |
| tg1024 | 271.40 | 274.25 | +1.0% |
| tg2048 | 267.37 | 272.12 | +1.8% |
| tg4096 | 258.67 | 267.39 | +3.4% |

Small further gain, no regression -- Qwen's ratio against llama.cpp moves
from 1.02-1.05x to **1.06-1.07x**, still comfortably ahead.

**Why this is not merely a repeat of the grid-width lever's null result.**
That lever added more *blocks* (more resident warps of the *same* kind of
work, hitting the SM's 32-warp/SM ceiling at 4 blocks/SM with nothing left
to spend). This lever adds no blocks and no warps at all -- it gives warps
that were *already resident and already idle* independent work, which is
why it does not run into the same occupancy ceiling and why, unlike the
grid-width lever, it shows up at `tg1024` without needing `tg4096` or
`nsys` to see past trajectory-average dilution.

**Committed on `autoresearch`.** New files:
`phobos-lang/src/codegen/tile/warp_attn.rs`. Changed:
`phobos-lang/src/codegen/expr.rs` (dispatch), `phobos-lang/src/codegen/tile/mod.rs`
(module registration), `phobos-gguf/src/backend/device/kernels/attn.rs`
(`attention_split_src`/`attention_persist_src` rewritten, new
`attention_combine` helper, new `ATTN_WARP_SPLITS` constant),
`phobos-gguf/src/backend/device/attn.rs` (call sites updated to the new
function signatures).

**What is not chased this round.** `warp_partial` does one key at a time,
unvectorized scalar loads for its `dpl`-wide per-lane K/V slice (`dpl = 4`
for minicpm's `D=128`, `8` for Qwen's `D=256`) -- batching a few keys per
warp iteration and/or vectorizing those loads (per the
`vector-width-is-bytes-not-elements` memory) is a plausible further
increment, untried here, and this round's numbers are strong enough that it
was not necessary to reach a clear win. `ATTN_WARP_SPLITS = 8` (one warp per
group, matching the block's whole warp count) was not swept against smaller
values (multiple warps sharing a group, needing named-barrier or a
different combine shape) -- the advisor's v1 recommendation, taken as-is
and not revisited since it worked cleanly on the first attempt.
`emit_warp_partial` bails at compile time if `WM`/`WL`/`WACC` are ever sized
for a warp count that does not match the launching CTA's actual warp count
(`cta_threads / 32`), so a future attempt at fewer, wider warp groups is
caught as a build error rather than a silent out-of-bounds shared-memory
write from the extra warps' unaccounted-for final stores.

**Post-commit gate check.** The advisor's one open question after this
landed -- does `attn_persist_plan`'s occupancy gate still *accept* both
shapes under the new footprint, or did they silently fall back to the
(also-faster) launched path, making the "both persist" framing above
unverified -- was checked directly rather than left inferred:
`PHOBOS_PASS_REPORT=9 PHOBOS_ATTN_PERSIST=1`, `bench -m <model> -p 0 -n 12
-r 1 --no-warmup`, both models. minicpm: `attention_persist`, shared 9952
bytes, 4 blocks/SM, 4.00 waves (clean single pass). Qwen: `attention_persist`,
shared 19680 bytes, 3 blocks/SM, 3.00 waves (clean single pass, total 214
launches -- an exact match to the previously documented count from before
this round's kernel rewrite). Both shapes confirmed live on the persistent
path with no second grid-strided pass, so the bench numbers above are what
they claim to be.

### Round 2: vectorized K/V loads + a warp-count sweep

Picked up the round's own "what is not chased" list. Two agents ran this
round in parallel (this beam's vectorization + sweep, and
`[[launch-bound-headroom]]`'s independent re-examination of the fusion
levers below it); both went quiet after several resumes and were finished
by direct takeover rather than a subagent report -- see `BEAMS.md`'s
process-lesson entries for what that involved.

**Vectorization.** `warp_partial`'s per-lane K/V loads now issue as a
single wide vector load (`vector<8xf16>` for Qwen's `dpl=8`, `vector<4xf16>`
for minicpm's `dpl=4`) instead of `dpl` separate scalar loads, widened to
f32 in one `vec_extf` and unpacked with `vec_extract`. Safety argument
(alignment holds for every lane and every row, not just the common case)
is in the code comment above the change in `warp_attn.rs`; falls back to
scalar for a `dpl` the `[8,4,2,1]` ladder doesn't evenly divide, not
reached by either shipped shape. `attention_split_src`'s `@launch(...)`
width changed from a hardcoded `256` to `wct * WARP_THREADS`, so a future
warp-count change can't silently desync the launch width from the
kernel's own warp count.

**Warp-count sweep.** Swept `ATTN_WARP_SPLITS` at 4, 6, and 8 (the shipped
default) via `attndecode`, minicpm shape, cache 4099: 964.0us / 776.0us /
708.9us respectively (4.0x / 3.2x / 3.0x the bandwidth floor, `device copy
bandwidth` now reading a properly-warmed 422 GB/s rather than the original
run's cold 142 GB/s). **8 wins outright** -- fewer, wider-working warps
each pay a longer critical path than the combine overhead they save.
Confirms the shipped default rather than finding a better one; no code
change from this half of the round.

**Correctness**, on the combined tree (this round's vectorization plus
`[[launch-bound-headroom]]`'s concurrent fusion work, verified together
since that is the tree that actually shipped): `backend_check` (worst rel
err 2.902e-4, unchanged), `fuse_check` both models (minicpm 9.703e-3
average/1.325e-2 worst, Qwen 8.085e-3 average/1.051e-2 worst, 0 top-token
flips either model), `batch_check` both models ("batched and sequential
agree", spread errors in the same 0.7e-2-1.6e-2 band this session has seen
throughout), `model_check` both models ("backends agree"). `cargo check
--workspace` and `cargo clippy -p phobos-gguf -p phobos-lang --features
cuda -- -D warnings` both clean.

**Benchmark**, same session, interleaved against llama.cpp,
`PHOBOS_ATTN_PERSIST=1`, combined tree
(`autoresearch/beams/takeover_combined_bench.{csv,json,log}`), 3 of 3
rounds uncontended:

| model | test | before (round 1) | after (round 2, combined) | ratio |
| --- | --- | --- | --- | --- |
| minicpm5-1b | tg1024 | 260.75 | 264.56 | 0.96x |
| minicpm5-1b | tg2048 | 255.76 | 261.74 | 0.96x |
| minicpm5-1b | tg4096 | 244.43 | 254.07 | 0.95x |
| Qwen3.5-0.8B | tg1024 | 274.25 | 279.39 | 1.08x |
| Qwen3.5-0.8B | tg2048 | 272.12 | 279.25 | 1.09x |
| Qwen3.5-0.8B | tg4096 | 267.39 | 275.73 | 1.08x |

A further small, real improvement over round 1's already-large win, on both
models, no regression -- but the vectorization and the concurrent
`[[launch-bound-headroom]]` fusion work landed together and were verified
and benchmarked as one combined tree, so this table cannot separate their
individual contributions. Given both were independently correctness-gated
before combining and the combined result strictly improves on round 1's
number, that ambiguity was accepted rather than spending a further round
isolating it. **minicpm still has not crossed 1.0x** (0.95-0.96x) -- closest
this session has gotten, not yet the stated goal.

**Committed on `autoresearch`** (both this round's and
`[[launch-bound-headroom]]`'s changes, one commit, since they were verified
together as one tree and splitting the commit after the fact would not
reflect what was actually tested).

### ncu on the post-vectorization `attention_persist` kernel

Re-profiled `attention_persist` (the committed, vectorized kernel, not the
pre-`warp_partial` one the session's earlier ncu data is from) at cache
length 4099, minicpm's shape:

    ncu -k "regex:attention_persist" -c 5 --set full -f \
      -o autoresearch/beams/ncu_persist_post_vectorize_4099 -- \
      target/release/examples/attndecode.exe
    (PHOBOS_ATTNDECODE_SHAPE=minicpm PHOBOS_ATTNDECODE_LENGTH=4099 PHOBOS_ATTN_PERSIST=1)

Occupancy is no longer the story: **Achieved Occupancy ~98%** (up from the
pre-vectorization 63.8% this beam's own earlier round measured), 31.5-31.8
of 32 theoretical warps active per SM. But throughput is still low --
Memory Throughput 36-38%, Compute (SM) Throughput 33-36% -- so the kernel
is latency-bound with occupancy already at its ceiling, not thread-starved.

Warp-stall breakdown (raw `smsp__average_warps_issue_stalled_*` counters,
per-instruction average of ~21.3 cycles):

| stall reason | cycles | share |
| --- | --- | --- |
| barrier (waiting on sibling warps at the intra-CTA combine sync) | 9.09 | 42.7% |
| long scoreboard (global memory latency) | 5.16 | 24.2% |
| wait (fixed-latency, e.g. arithmetic pipe) | 2.23 | 10.5% |
| short scoreboard (shared memory latency) | 1.62 | 7.6% |
| selected (issuing) | 1.00 | 4.7% |
| not selected | 0.61 | 2.9% |
| math pipe / MIO throttle | 0.75 | 3.5% |

Barrier stall alone is the single largest cost, ahead of raw memory
latency. **Correction, discriminating measurement below: this is not
`warp_partial`'s intra-CTA combine sync.** Initial writeup guessed the
8-warp combine barrier; the actual source is `attention_persist`'s
`grid_barrier()`, the cross-block sync between the persistent kernel's
split and merge phases.

Ran the identical profile against the *launched* `attention_split` kernel
(same shape/length, `PHOBOS_ATTN_PERSIST` unset) -- it contains
`warp_partial`'s combine barrier but no `grid_barrier()`:

    ncu -k "regex:attention_split" -c 5 --set full -f \
      -o autoresearch/beams/ncu_split_post_vectorize_4099 -- \
      target/release/examples/attndecode.exe
    (PHOBOS_ATTNDECODE_SHAPE=minicpm PHOBOS_ATTNDECODE_LENGTH=4099)

Barrier stall on `attention_split`: **1.43 cycles, ~12% of ~11.5 total**
(vs 9.09 cycles, 42.7%, on `attention_persist`). The dominant cost there
is long-scoreboard memory latency (4.82 cycles, ~42%) plus fixed-latency
wait (2.15, ~19%) -- an ordinary latency-bound profile, not
barrier-dominated. Occupancy on `attention_split` is 64.9% (unchanged from
this beam's pre-vectorization baseline; vectorization's occupancy gain was
specific to `attention_persist`'s launch shape, not `warp_partial` itself).
Raw report: `autoresearch/beams/ncu_split_post_vectorize_4099.ncu-rep`.

**What this means for the next round.** The barrier cost is a
`grid_barrier()` / cross-block problem, not a `warp_partial` one: some of
the persistent kernel's 144 resident blocks finish their grid-strided
split-phase work later than others (plausibly the same memory-latency
variance the long-scoreboard share hints at, now expressed as cross-block
skew rather than cross-warp skew) and every block idles at the barrier for
the slowest one before the merge phase can start. Two fixes worth trying,
in the persistent kernel's structure rather than `warp_partial`: (a)
better-balanced work assignment across the grid-strided split so blocks
finish closer together, or (b) replace the full grid barrier with an
atomic arrival-counter-gated merge that lets early-finishing blocks start
merging as data becomes available rather than all waiting on the
slowest. Key-batching/ILP tricks remain out of scope here too -- neither
occupancy (98% on the persistent kernel) nor compute throughput (35%) is
the ceiling. Raw report for the persist kernel itself:
`autoresearch/beams/ncu_persist_post_vectorize_4099.ncu-rep`.

### Round 3: the imbalance was real and structural, not just latency variance -- fixed with (a)

Picked up the previous round's diagnosis and checked which of the two
named directions applies before picking one: is the barrier-stall
imbalance a genuine *size* mismatch in phase one's work assignment, or
just memory-latency variance among equally-sized units expressed as
cross-block skew?

**It is a size mismatch, and a large one.** `attn_persist_plan`
(`phobos-gguf/src/backend/device/attn.rs`) settles the persistent kernel's
resident grid from the driver's own occupancy answer, independent of
`ATTN_SPLITS`. `attention_decode` (same file) then handed the persistent
path the *launched* kernel's own split count (`ATTN_SPLITS * qgroup`, a
constant tuned for the launched kernel's own grid, `blocks = groups *
splits` exactly) with no connection to the persistent kernel's actual
settled grid. Checked with a fresh `PHOBOS_PASS_REPORT=9` pass report on
this card, current tree (`ac86bfd`'s descendants): minicpm's shape settles
144 -> now 192 resident blocks (`4608 blocks / 24 calls`, 4/SM,
`clean single pass`) since the vectorization/fusion rounds shrank the
kernel's shared-memory footprint further, but phase one's assigned unit
count stayed fixed at `groups * splits = 8 * 16 = 128` -- so **a third of
the resident grid (64 of 192 blocks) starts phase one with `u >= U1`,
skips the whole split body, and goes straight to `grid_barrier()`**, where
it then sits idle for the entire time the other 128 blocks spend doing
real work. Qwen's shape is worse in relative terms: settled grid is 144,
assigned units `4 * 16 = 64`, so **80 of 144 blocks (56%) are pure-idle
barrier-waiters from the start of the kernel.** This is exactly the
"remainder chunk" shape of imbalance the task brief asked to check for,
not latency noise -- the 24.2% long-scoreboard share the previous round
measured is real too, but it is not what the 42.7% barrier figure is
mostly made of.

**Fix: `attn_persist_plan` now settles its own split count from the grid
it settles, not the launched kernel's `ATTN_SPLITS`.** `splits = blocks /
groups` (floored) makes `groups * splits` land on `blocks` exactly
whenever it divides evenly, which both shipped shapes do on this card
(minicpm: `192 = 8 * 24`; Qwen: `144 = 4 * 36`) -- every resident block
gets a real unit of phase-one work, none idle at the barrier from the
start. `attention_decode` no longer threads a `splits` value into the
persistent path at all; `attn_persist_plan` returns `(blocks, splits)`
together and `AttnPersistKey` dropped `splits` from its five-tuple (it
was never load-bearing for identity -- the persist split count is now a
pure function of the other four shape fields plus the device's own
occupancy answer). Full mechanism and the two-stage design (see below) are
in the doc comments on `attention_decode` and `attn_persist_plan`.

**A real bug surfaced along the way, worth recording so it is not
rediscovered.** The first version of this fix put `splits = blocks /
groups` *inside* the existing occupancy-settling loop, recomputed every
candidate `blocks` right alongside it. That is unstable: a wide first-try
`blocks` guess (the loop's starting point, `per_sm * sms`, before any
shrinking) picks a correspondingly wide `splits`, which grows phase two's
`mv`/`c` tiles (sized `[1, S]`) enough to measurably shrink what
`cuOccupancyMaxActiveBlocksPerMultiprocessor` allows; the *next* iteration
reads that shrunk `blocks`, picks a *narrower* `splits` for it, and the
occupancy query on that narrower kernel comfortably allows the original
wide grid again -- but the loop only ever compares `allowed` against its
own already-shrunk candidate, never revisits the wider one it abandoned,
so it settles on the first accidentally-small `blocks` it happens to land
on. Measured on Qwen's shape: this version of the loop settled at 48
blocks (1/SM) where a fixed-`splits` probe finds 144 (3/SM) is genuinely
resident -- a 3x narrower grid than necessary, and non-deterministic
between process runs (`bench.exe` calls with `PHOBOS_PASS_REPORT=9`
against the same shape reproduced 48, then 144, then 48 again across
otherwise-identical invocations, tracking which candidate the loop's
compiled-module churn happened to land the shared-memory query on). It
also intermittently corrupted a *later, unrelated* kernel launch
(`rms_norm: CUDA_ERROR_INVALID_VALUE`, reproduced on roughly half of
repeated runs) via a second, independent bug the varying-`S` loop exposed:
`shared_of`'s cache (`self.func_shared`, `phobos-gguf/src/backend/device/
launch.rs`) is keyed on a raw `CUfunction` pointer, and a discarded loop
candidate's module unloading can hand that address to a later, wholly
unrelated compile; with a constant `splits` (the pre-existing code, and
this round's stage-one probe) every candidate's shared-byte value was
identical, so a stale cache entry was always coincidentally correct and
this was invisible. Once `splits` varied by candidate, a stale entry could
be wrong, and once it was wrong the byte count handed to some later
kernel's launch (whatever unrelated function next reused that address)
could exceed what that kernel's own `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED
_SIZE_BYTES` allows, which the driver rejects outright. Fixed two ways:
(1) split settling `blocks` and settling `splits` into two stages so nothing
feeds back on itself (stage one settles `blocks` with a fixed probe
`splits = ATTN_SPLITS * qgroup`, exactly reproducing the pre-existing,
already-validated loop; stage two computes the real `splits` from the
settled `blocks` once and recompiles a single time, verifying occupancy
still holds before committing to it and falling back to the probe module
otherwise), and (2) evicting a discarded candidate's `func_shared` entry
before its module is dropped, defensively, in case a future change
reintroduces per-candidate variation. Reproduced the bug and confirmed the
fix with 6+ repeated `PHOBOS_PASS_REPORT=9` runs each on both models: the
buggy version failed roughly every other run, the two-stage version has
not failed once across more than a dozen repeats on both models.

**Correctness**, current tree, `PHOBOS_ATTN_PERSIST=1` forced, both
models: `backend_check` (worst rel err 2.902e-4, unchanged from every
prior round), `batch_check` (minicpm gpu spread err up to 1.350e-2, Qwen
up to 1.551e-2, both "batched and sequential agree", same band as every
prior round), `model_check` (`backends agree`, both models), `fuse_check`
(minicpm worst 1.332e-2/9.944e-3 average, Qwen worst 1.051e-2/8.085e-3
average, both 0 top-token flips -- identical to the documented baseline to
the digit). `cargo test -p phobos-gguf --features cuda --lib` (55 passed).
`cargo clippy --release -p phobos-gguf --features cuda -- -D warnings`
clean. All of this was run twice: once in the shared working tree, and
once more in an isolated `git worktree` containing only this beam's exact
diff (a concurrent, unrelated argmax-feature edit landed mid-round in the
same shared tree and left it non-building at points during this round --
see the process note below), to rule out cross-contamination from that
concurrent work. Both runs agree to the digit.

**Numbers.** `attndecode`, minicpm shape, cache 4099, mean of 3 runs each,
before this round's fix (`ee8874c`, saved as a separate binary before
editing) vs after:

| | before | after | delta |
| --- | --- | --- | --- |
| attn/step | 690.8us | 626.0us | **-9.4%** |
| vs floor | 2.9x | 2.6x | |

Qwen shape, cache 4099, same protocol:

| | before | after | delta |
| --- | --- | --- | --- |
| attn/step | 217.8us | 217.4us | -0.2% (noise) |

`ncu` on minicpm's `attention_persist`, cache 4099, same protocol as the
previous round (`smsp__average_warps_issue_stalled_*_per_issue_active.ratio`):

| stall reason | before (cycles) | after (cycles) | before share | after share |
| --- | --- | --- | --- | --- |
| barrier | 9.09 | 3.77 | 42.7% | 25.0% |
| long scoreboard | 5.16 | 4.70 | 24.2% | 31.2% |
| wait | 2.23 | 2.23 | 10.5% | 14.8% |
| short scoreboard | 1.62 | 2.36 | 7.6% | 15.7% |
| selected | 1.00 | 1.00 | 4.7% | 6.6% |
| not selected | 0.61 | 0.97 | 2.9% | 6.4% |
| misc | 0.75 | 0.02 | 3.5% | 0.1% |

Barrier stall's absolute cost dropped 58% (9.09 -> 3.77 cycles) and the
total average stall-cycles-per-instruction dropped 29% (~21.3 -> ~15.05),
tracking the 9.4% wall-clock reduction (stall cycles are not 1:1 with wall
time, but the direction and rough proportion both confirm the mechanism
rather than merely correlating with it). Occupancy stayed at its ceiling
(99.68%, matching the pre-fix 98%), and compute throughput actually rose
(33-36% before -> 39.62% after) since less of each SM's cycles are now
spent on blocks doing nothing. Grid confirmed as `(192, 1, 1)` in the raw
report, matching the settled-blocks math above. Raw reports:
`autoresearch/beams/ncu_persist_rebalanced_4099.ncu-rep`.

**Qwen's honest secondary finding: fixing the idle-block count did not
fix Qwen's barrier stall, and the wall-clock stayed flat rather than
improving.** `ncu` on Qwen's `attention_persist` post-fix (grid `(144, 1,
1)`, `autoresearch/beams/ncu_persist_qwen_rebalanced_4099.ncu-rep`) shows
barrier stall *still* dominant -- 15.88 of ~26.46 total cycles, ~60%, if
anything a larger share than minicpm ever measured -- and achieved
occupancy down at 74.31% (well under minicpm's 99.68%), even though the
same fix eliminated 100% of Qwen's previously-idle blocks (0 of 144 now
unassigned, same as minicpm). The likely mechanism, not chased further
this round: Qwen's `splits = 36` is a much bigger multiplier over its
`ATTN_SPLITS`-derived baseline (`16 -> 36`, 2.25x) than minicpm's (`16 ->
24`, 1.5x), because Qwen's `groups = 4` is smaller and the same settled
`blocks = 144` divides into fewer, bigger multiples of it; each phase-one
unit's own key range (`per = ceil(NK / S)`) is correspondingly narrower
(roughly 114 keys per unit at cache 4099, against minicpm's ~171), and at
that granularity per-block fixed overhead (launch/scheduling skew,
register/shared-memory setup, `warp_partial`'s own startup cost) may be
large enough relative to the shrunk real work that finish-time variance
gets *worse* even though the assigned-work variance (the 0-vs-nonzero
split this round targeted) is now zero. This is a distinct, second
imbalance source from the one this round fixed -- over-fragmentation
rather than under-assignment -- and it nets out flat rather than negative
on Qwen's wall-clock, so it was not chased further: the `-9.4%` win on
minicpm (the task's actual target) is real, reproducible, and does not
regress Qwen, which is the bar this round needed to clear. A cap on how
far `splits` can grow past the launched kernel's own tuned value (e.g.
`splits <= 2 * (ATTN_SPLITS * qgroup)`, leaving some idle blocks on a
small-`groups` shape like Qwen's rather than over-fragmenting) is a
plausible follow-up if Qwen's flatness does not hold up under `bench.py`,
but was not implemented speculatively against a result that already reads
as a clean pass.

**Process note.** A concurrent, unrelated agent landed an in-progress
argmax feature (new files and struct fields across `phobos-lang`,
`phobos-gguf/src/backend/mod.rs`, `device/mod.rs`, `device/kernels/mod.rs`)
in this same shared working tree partway through this round, at one point
leaving the tree unable to build for reasons entirely unrelated to this
beam. Per the session's existing worktree-isolation lesson, this beam's
own correctness gates were re-run inside an isolated `git worktree`
carrying only this beam's exact diff (`attn.rs` copied whole, `mod.rs`'s
three hunks reapplied by hand rather than via `git diff | git apply`,
since a plain `git diff` on a file two concurrent agents are both editing
silently captures both sets of hunks together -- learned the hard way when
a first attempt at this isolation pulled the other agent's incomplete
argmax additions along for the ride and failed to build for *their*
reasons, not this beam's). Do not trust `git diff <file>` to isolate one
beam's changes when another agent is concurrently editing the same file;
diff against a known-clean base and hand-verify the hunks instead, or
avoid shared files entirely per the existing lesson.

**Nothing committed this round.** Per this round's brief, `scripts/bench.py`
confirmation and the final commit are left to the orchestrator. Changed:
`phobos-gguf/src/backend/device/attn.rs` (`attention_decode`,
`attn_persist_plan`), `phobos-gguf/src/backend/device/mod.rs`
(`AttnPersistKey` narrowed to four fields, new `AttnPersistEntry` type
alias, `attn_persist_modules`'s value type). `phobos-gguf/src/backend/
device/kernels/attn.rs` (the kernel source generator) is untouched --
this round is entirely in the caller's grid/split arithmetic, matching
the task brief's "contained" option.
