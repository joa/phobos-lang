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
