---
parent: fcfdf8b (HEAD, autoresearch branch)
status: killed
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
