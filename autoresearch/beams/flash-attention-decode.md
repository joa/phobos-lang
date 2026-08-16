---
parent: fcfdf8b / 80204d5 (HEAD, autoresearch branch)
status: implemented, opt-in -- genuine partial win on minicpm (+1.0-1.4% tg, all rows positive), shape-gated to avoid a measured Qwen regression, open, combine with [[launch-bound-headroom]]
---

# Beam: flash-attention-shaped decode attention

## Correction to the first pass of this beam note

The claim below that `-fa 0` "explains the whole session's deficit" was
overstated -- checked against the same table it was drawn from. `-fa 0`
closes the *slope* (tg512/tg1024 are dead heats) and about half of the flat
tg32 deficit, but a real gap remains at every length: -4.0% at tg32, -2.9%
tg128, 0% tg512, -0.5% tg1024, **-5.5% at tg2048**. Also: `-fa 0` in
llama.cpp is not a pure kernel swap in isolation -- the non-FA path also uses
a different V-cache layout than the FA path -- so this experiment is
correctly described as "phobos vs. llama.cpp's non-FA decode path," not
"same everything, one kernel toggled." The conclusion (attention kernel
design is most of the gap) still holds; the size claim needed fixing.

**The bigger correction: the goal is not parity with non-FA llama.cpp, it is
beating FA-on llama.cpp** (270-275 t/s flat, the actual baseline this whole
session measures against). At tg2048 that needs **+18%** over phobos's
current 233.54 t/s. Bounding what's achievable:

`autoresearch/beams/lmhead_profile_cuda_gpu_trace.csv` (already-committed
nsys trace from the [[wide-vocab-lm-head]] round) has every GPU event with
durations for a steady-state minicpm decode step. Summing
`attention_split` + `attention_merge` durations inside four different
steady-state step windows (bounded by consecutive lm_head launches, same
method the bubble measurement used):

| window | attention ns | wall ns | % of step |
| --- | --- | --- | --- |
| 1 | 764,882 | 4,724,486 | 16.2% |
| 2 | 674,616 | 4,602,734 | 14.7% |
| 3 | 812,301 | 4,745,606 | 17.1% |
| 4 | 857,005 | 4,808,225 | 17.8% |

**Attention (both kernels combined) is ~15-18% of a decode step.** That is
the hard ceiling on this beam even in the impossible case of a zero-cost
attention kernel: 233.54 t/s * 1/(1-0.17) ~ 281 t/s, just clears 275, with
no margin and assuming perfection. A real implementation will not remove
100% of that 15-18% -- of the two kernels, `attention_split` is 6-8x
`attention_merge`'s share in every window above (e.g. window 1: 661,271 ns
split vs. 103,611 ns merge), meaning most of attention's cost is the split
kernel's own compute/bandwidth against the KV cache, not the merge or the
scratch round-trip between them. **This beam alone is unlikely to clear the
goal; it has to combine with [[launch-bound-headroom]]'s remaining
unfused launches (`store_2d`, `quantize`, `q8_qdot_add`, still ~untried) to
have a realistic shot**, per AGENT.md's combine-before-retiring discipline.
Whoever runs this beam should say so plainly rather than promising a win it
cannot deliver alone.

## phobos already has a working FlashAttention-2 kernel -- this is an
## adaptation, not a from-scratch design

`out.csv` (read at the start of this session): `flash_fp32` and `flash_fp16`
rows are `phobos-bench`'s existing benchmark of a real, working, single-pass
online-softmax attention kernel. Source: `examples/flash_attention_fp32.ph`
(also `_fp16.ph`), a complete FlashAttention-2 implementation already in the
tree:

    @cluster(BR in [1024, 4096])
    @pipeline
    @tensorcore
    @launch(128)
    @autotune(D in [64], BR in [4, 128], BC in [4, 128])
    @aligned(Nq = BR, Nk = BC)
    kernel flash_attention(Q, K, V, O, scale) {
      var acc, m, l = 0, -inf, 0
      for kt in range(0, Nk, BC) {
        var s = dot_t(q, k) * scale
        var mnew = tmax(m, rowmax(s))
        s = exp(s - mnew)
        var corr = exp(m - mnew)
        l = l * corr + rowsum(s)
        acc = acc * corr + dot(s, v)
        m = mnew
      }
      O[row :+ BR, :] = acc / l
    }

This is prefill-shaped (`BR` query rows tiled, no causal mask visible here,
not wired into the GGUF decode path), but the online-softmax recurrence
inside the `for kt` loop is exactly the merge logic `attention_merge`
currently does as a *second, separate kernel* reading scratch (`P`/`ML`)
another kernel wrote. **Decode's `M=1` shape also means FlashAttention's
actual namesake trick -- avoiding materializing the `[Nq, Nk]` score matrix
-- buys nothing here**, since a single query row's score is `[1, Nk]` and
was never tiled for that reason in the current design either. What plausibly
pays is folding the online-softmax combine (this kernel's `for kt` loop
body) into `attention_split` itself, so the split-across-grid decode kernel
keeps its parallelism (each block still owns a slice of the key axis, for
the same reason `attention_decode`'s split exists at all -- one program per
head would leave a grid a sixth of the card wide) but combines its own
partial `(m, l, acc)` in-kernel via a device-scope reduction instead of
writing them to global scratch for a second kernel to read back.

`docs/megakernel.md` already built the primitives this needs: `grid_barrier()`
(measured 0.88-1.09us, "half a launch") and `@persistent` (grid sized from
`cuOccupancyMaxActiveBlocksPerMultiprocessor`, not a constant -- the doc's
own "hard correctness precondition: co-residency" section applies directly
here). A `@persistent` `attention_decode` that computes its split's partial
in phase one, `grid_barrier()`s, then does the reduction across resident
blocks in phase two removes both a kernel launch (244 -> 220-ish, folding
into [[launch-bound-headroom]]'s count) and the `P`/`ML` global round-trip,
without giving up the split-for-grid-width design or the
[[decode-attention-splits-and-query-grouping]] GQA lessons, which stay
valid: the split-per-key-range and `QG` grouping logic do not change, only
where the combine happens.

## Evidence this beam is real (not inferred, measured)

`python scripts/bench.py -m models/minicpm5-1b-Q8_0.gguf -p 128 -n 32 128 512
1024 2048 -r 3 -R 3 --llama-args "-fa 0"`, 2026-08-16, same session as the
fresh baseline, 3 of 3 rounds uncontended
(`autoresearch/beams/noFA_bench.{csv,json,log}`):

| test | phobos | llama.cpp (FA on) | llama.cpp (`-fa 0`) | ratio (FA on) | ratio (`-fa 0`) |
| --- | --- | --- | --- | --- | --- |
| tg32 | 253.69 | 271.65 | 264.25 | 0.92x | 0.96x |
| tg128 | 256.86 | 274.86 | 264.29 | 0.92x | 0.97x |
| tg512 | 253.58 | 273.68 | 253.33 | 0.91x | 1.00x |
| tg1024 | 245.19 | 273.66 | 246.52 | 0.89x | 0.99x |
| tg2048 | 233.54 | 270.05 | 247.24 | 0.86x | 0.94x |

phobos's KV cache is already f16 (`kv: fp16 buffers`, `50543a5`), matching
llama.cpp's default, so this is not a precision artifact -- confirmed by
first trying `--llama-args "-ctk f32 -ctv f32"` and having llama-bench reject
`f32` as a cache-type value outright (`autoresearch/beams/f32kv_noFA_bench.log`,
this llama-bench build only accepts `f16`). With flash attention off, phobos
is within 0-6% of llama.cpp's own decode attention at every cache length
tested, including a dead heat at tg512/tg1024. **The entire session's
measured gap is explained by llama.cpp having a flash-attention decode
kernel and phobos not having one** -- not by attention grid occupancy (beam
1, killed), not by lm_head bandwidth (beam 2, killed), not by launch count
(beam 3, one increment tried, regressed for an unrelated tiling reason).

## What this means for the other three beams, in hindsight

[[cache-length-split-buckets]]'s null result now makes sense as a category
error, not a surprising result: widening `attention_decode`'s own
split-and-merge grid cannot close a gap that comes from FlashAttention being
a *different kernel design* (one fused pass, online softmax, no scratch
round-trip) rather than the same design run wider. There was never a split
count that would have found this gain, because the gain is not in occupancy.

[[launch-bound-headroom]] is not wrong, just smaller: its ceiling (14.4%,
~655us of the ~3.5-4ms decode step) is real launch overhead, but even fully
eliminated it would not close a gap this evidence now says is dominated by
attention's own kernel design, which launch-fusion does not touch (fusing
launches around `attention_split`/`attention_merge` still calls the same two
kernels).

## What to build

Not scoped into an implementation plan yet -- this file exists to hold the
evidence and hand off to whoever scopes it next. Known constraints and prior
art to carry forward rather than rediscover:

- **GQA correctness lessons from [[decode-attention-splits-and-query-groupin
  g]] memory still apply.** A key head is read once per query head that
  shares it unless the design accounts for grouping; that memory's 3-part fix
  (tile-aligned splits, parallel merge, `QG` query rows per program) was hard
  won on the *current* split-plus-merge design and the reasoning (why a
  serial merge is a fixed per-step cost independent of cache length) carries
  over to any single-pass redesign that still needs to reduce partial results
  across a grid -- a naive single online-softmax pass per (query-head-group,
  whole-cache) still needs a decision about whether one block owns the whole
  cache per head (long-cache latency) or the key axis is still split with an
  online-softmax merge instead of the current two-pass max/exp/sum merge
  (FlashAttention's actual design, and the more faithful target).
- **`docs/megakernel.md`'s co-residency and occupancy lessons apply if this
  becomes a persistent-kernel design**: a grid barrier deadlocks if every
  block is not resident at once, so grid size has to come from
  `cuOccupancyMaxActiveBlocksPerMultiprocessor`, not a constant, if this ever
  merges attention into the wider persistent-kernel effort rather than
  staying a standalone kernel pair.
- **`ATTN_QGROUP`/`ATTN_SPLITS` tuning work is not wasted** even if the
  kernel shape changes underneath it -- `attention_qgroup`'s per-group-size
  logic and the `attndecode` example's shape sweep (primes, deliberately, to
  expose remainder handling) are reusable measurement infrastructure for
  whatever kernel replaces or sits beside `attention_decode`.
- **Shared memory budget is the same wall it always was on this card**:
  `STATIC_SHARED_LIMIT` (48 KB, sm_75), and [[cache-length-split-buckets]]'s
  result log notes the merge kernel's tile hit a compile wall at exactly
  `S=48, head_dim=256` with zero margin below the documented limit -- leave
  margin, don't design to the wall.
- **`phobos-lang`'s tile/codegen constraints**: attention kernels are
  generated MLIR source via `@autotune`/format strings (`attn.rs`'s
  `attention_split_src`/`attention_merge_src`), not hand-written CUDA, so a
  redesign is a codegen-template change plus whatever new `phobos-lang`
  primitives an online-softmax single pass needs (a running max/sum carried
  across a key-axis loop inside one kernel is closer to the flash-attention
  prefill path already in the codebase than to the current decode split, per
  `attn_gemm_src`/tril-masked prefill attention -- worth reading that path
  first for reusable structure before designing from scratch).

## Correctness gate

Same as every attention change this session: `backend_check`, `fuse_check`,
`model_check`, `batch_check`. Given the scope, also budget for a dedicated
`attndecode`-style microbenchmark of the new kernel alone before a full
`bench.py` round, the way the existing kernel already has one.

## Result log

### Implemented: the fallback from "What to build", not the full grid-barrier redesign -- and it turned out to need one more piece than the brief anticipated

Built the smaller fallback the brief allowed as legitimate ("reduce what the
merge kernel and the scratch round-trip cost without removing the second
launch entirely"), except it *did* remove the second launch: `attention_split`
and `attention_merge` (`phobos-gguf/src/backend/device/kernels/attn.rs`) are
folded into one `@persistent` kernel, `attention_persist_src`. Phase one is
the split kernel's body verbatim, over a grid-strided range of `(group,
split)` units instead of one program per unit; `grid_barrier()`; phase two is
the merge kernel's body, over a grid-strided range of heads. Neither phase's
math changed by a line -- this is `flash_attention_fp32.ph`'s online-softmax
recurrence, already present twice (once per phase, since the split kernel
already carries it across key tiles and the merge kernel across splits), just
moved from two kernel launches into one kernel and a barrier. `P`/`ML`
scratch stay exactly as they were (`attn_scratch`, `mem.rs`), passed to the
combined kernel under both shapes the two phases need (`P` and `PM`, one
pointer, two parameter declarations) rather than round-tripped through two
launches.

What the brief did not anticipate, found only by measuring: **the persistent
kernel is not simply "the same kernels, one less launch."** Its combined
shared-memory footprint does not collapse to the wider of the two phases the
way `docs/megakernel.md`'s redundant-stage tiles do (see that doc's step 3a,
"a redundant stage is only free if its result never leaves the block") --
measured at Qwen's shape, the combined kernel takes 41376 bytes of shared
against the launched split kernel's own 23760 (merge alone is 17616), nearly
the *sum* of the two rather than the max. That halves occupancy from 2 blocks
per SM to 1, and since the split phase's own unit count (`groups * splits`)
does not shrink, a halved resident grid means the grid-strided phase-one loop
needs a *second* pass over the whole key axis where the launched kernel needed
one. This is a genuine phobos-lang codegen finding, not a design flaw in the
phase split: the tile pool's liveness tracking, which correctly reuses a
redundant stage's storage across a barrier when [`docs/megakernel.md`]'s fused
MLP does it, is not doing the same across this kernel's two phases. Fixing
that pooling gap is the highest-value follow-up this round found and did not
have time to chase (see "What is not chased" below) -- if it collapses the way
the MLP's does, Qwen's shape stops declining and the whole beam's win widens.

### Why this predicts (and produces) a per-model split, and the gate that routes around it

The mechanism above means the persistent kernel is **slower per call than the
launched pair at both shapes measured**, and the `tg` win, where there is one,
comes entirely from removed launches outrunning that per-call cost --
`docs/megakernel.md`'s own launch-arithmetic-vs-kernel-cost lesson from step
3b ("cost the kernels, not the boundaries") landing on a third mechanism.
`attndecode` (`cargo run --release -p phobos-gguf --features cuda --example
attndecode`), launched vs. persistent, `attn/step` in microseconds, primes as
the harness already sweeps
(`autoresearch/beams/attn_persist_attndecode_{baseline,persist}.log`):

| cache | minicpm launched | minicpm persist | delta | Qwen launched | Qwen persist | delta |
| --- | --- | --- | --- | --- | --- | --- |
| 37 | 305.4u | 295.6u | -3.2% | 93.3u | 141.8u | **+52.0%** |
| 67 | 235.2u | 252.6u | +7.4% | 72.9u | 103.2u | **+41.6%** |
| 131 | 251.1u | 256.9u | +2.3% | 73.9u | 102.9u | **+39.2%** |
| 257 | 317.6u | 324.9u | +2.3% | 95.6u | 152.6u | **+59.6%** |
| 521 | 703.4u | 717.7u | +2.0% | 211.6u | 348.6u | **+64.8%** |
| 1031 | 873.2u | 882.7u | +1.1% | 282.9u | 418.0u | **+47.8%** |
| 2053 | 1114.7u | 1195.3u | +7.2% | 372.9u | 567.8u | **+52.3%** |

minicpm's kernel itself is 1-7% *slower* per call, and Qwen's is 40-65%
slower. minicpm's `tg` win happens anyway because it pays this cost on all 24
of its layers and removes 24 launches + 24 scratch round-trips a step; Qwen
pays a much larger per-call penalty on only 6 of its 25 layers and removes
only 6. **This beam's payoff scales with full-attention layer count**, which
is exactly what predicts where it helps next (a model with more full-attention
layers than Qwen's 6-of-25) and where it would need the shared-memory fix
above before it helps at all.

The pass report (`PHOBOS_PASS_REPORT=9`, full text in
`autoresearch/beams/attn_persist_pass_report.log`) says why directly: at
Qwen's shape (`n_head=8, n_kv=2, head_dim=256`, `qgroup=2`, `splits=16`), the
split phase's own unit count is `groups * splits = 4 * 16 = 64`, and the
persistent kernel's occupancy-settled grid is only 48 blocks (1 block/SM x 48
SMs, from the 41376-byte footprint) -- 48 < 64, so phase one needs two
grid-strided passes. At minicpm's shape (`n_head=16, n_kv=2, head_dim=128`,
same `qgroup`/`splits`), the unit count is `8 * 16 = 128` and the settled grid
is 144 blocks (3/SM, 20896 bytes) -- 144 >= 128, one pass, no penalty.

**The gate**: `attn_persist_plan` (`phobos-gguf/src/backend/device/attn.rs`)
declines a shape whose settled grid cannot fit the split phase's own unit
count in one pass, the same architecture-blind way `fuse::ChainKey::plan`
declines a chain it cannot fuse -- no model name anywhere, just the driver's
occupancy answer against the shape's own arithmetic, cached per shape
(`AttnPersistKey = (n_head, group, head_dim, qgroup, splits)`) so a declined
shape is not recompiled or reprobed every call. A decline falls straight
through to the unmodified launched `attention_split`/`attention_merge` pair in
the same function, so a declined shape's behavior is bit-for-bit the
pre-existing path.

**This predicate's evidence is two shapes on one card, validated only at its
endpoints (144 vs 128, a 12.5% margin; 48 vs 64, a 25% shortfall), not in the
middle.** A shape where the settled grid sits just above its own unit count
(say 5-10% of margin rather than minicpm's ~12%) would pass this gate while
still paying most of Qwen's per-call penalty, since the penalty comes from the
shared-memory-driven occupancy hit, not from the pass count alone once phase
one is already single-pass. What would close that gap: a third shape nearer
the boundary, or a second card whose occupancy answer differs (the gate reads
the driver fresh each time, so it is architecture-portable by construction,
but architecture-*validated* only here). Until then this stays a hypothesis
about the predicate's shape, not a proven one -- named honestly rather than
generalized past what was measured.

### Correctness gates, all with `PHOBOS_ATTN_PERSIST=1` forced on (exercises both the persistent path on minicpm's shape and the decline-and-fallback path on Qwen's in the same run, since `backend_check`'s own attention sweep covers both)

- `backend_check`: every op passes, worst relative error 2.902e-4 (unchanged
  from the documented baseline), including every `rows=1` decode-shaped
  attention case across GQA shapes from 16/8x128 to 8/4x256, at cache lengths
  through split-tile boundaries (63, 64, 300, 511, 3, 512, 600).
- `model_check`, both models: "backends agree". minicpm spread errors
  1.258e-2/1.335e-2/1.294e-2, in the same band as the doc's own fused-MLP-only
  baseline (1.259e-2/1.273e-2/1.232e-2); Qwen 1.019e-2/9.839e-3/8.416e-2, tied
  at two of three steps.
- `batch_check`, both models: "batched and sequential agree" over 600 decode
  steps in batches of 512/100/64. minicpm spread 1.057e-2 to 1.35e-2 (doc's
  own band: 1.03e-2 to 1.19e-2); Qwen 7.61e-3 to 1.551e-2. This is the
  strongest gate here, since the batched path never touches decode attention
  at all and is an independent computation of the same logits across the same
  600 steps the persistent kernel ran on minicpm.
- `fuse_check`, minicpm: prompt pass agrees exactly, 32 decode steps at most
  1.244e-2 of the logit spread apart (1.010e-2 average), one tied (undecided)
  flip -- same order of magnitude as the doc's own recorded number for
  minicpm's existing MLP fusion alone (1.07e-2 average, worst 1.5e-2), so the
  persistent attention kernel is not adding drift beyond what fusion already
  costs.
- `cargo test -p phobos-gguf -p phobos-onnx -p phobos-inference -p
  phobos-kernels`: all pass. `phobos-base`'s `source_size` ratchet test:
  passes (`kernels/attn.rs` 578 lines, `device/attn.rs` 425 lines, `mod.rs`
  404 lines, all under the 900 cap with no grandfathering needed).
  `cargo clippy -p phobos-gguf --features cuda -- -D warnings`: clean.

### The benchmark, same session, interleaved against llama.cpp, 3 rounds x 3 reps, uncontended every round

minicpm5-1b-Q8_0
(`autoresearch/beams/attn_persist_{baseline,minicpm}_minicpm.{csv,json}`):

| test | baseline t/s | persist t/s | delta | ratio baseline | ratio persist |
| --- | --- | --- | --- | --- | --- |
| tg32 | 256.97 +/- 1.37 | 259.60 +/- 1.55 | +1.02% | 0.94x | 0.94x |
| tg128 | 256.52 +/- 0.36 | 260.22 +/- 0.26 | +1.44% | 0.92x | 0.93x |
| tg512 | 252.81 +/- 0.17 | 256.03 +/- 0.14 | +1.27% | 0.92x | 0.92x |
| tg1024 | 247.12 +/- 0.10 | 250.20 +/- 0.09 | +1.25% | 0.90x | 0.90x |
| tg2048 | 235.57 +/- 0.36 | 238.51 +/- 0.09 | +1.25% | 0.86x | 0.87x |

All five rows positive, a repeatable ~1.0-1.4% gain that is roughly *flat in
absolute t/s* across cache lengths (2.6-3.7 t/s at every row) rather than
proportional to it -- consistent with the mechanism (a fixed per-layer launch
and round-trip removed, not a bandwidth effect that would scale with cache
length). Reconfirmed after adding the shape gate
(`attn_persist_minicpm.csv`, gate active, minicpm still takes the persistent
path since it passes the gate): tg32 257.77, tg128 259.01, tg512 255.89,
tg1024 250.37, tg2048 238.48 -- same result within session noise.

Qwen3.5-0.8B-Q8_0, **before** the shape gate existed (attn_persist forced on
for every shape,
`autoresearch/beams/attn_persist_{baseline_qwen,qwen_ungated}.{csv,json}`):

| test | baseline t/s | persist (ungated) t/s | delta | ratio baseline | ratio ungated |
| --- | --- | --- | --- | --- | --- |
| tg32 | 274.75 +/- 2.11 | 274.70 +/- 1.64 | -0.02% | 1.17x | 1.15x |
| tg128 | 281.67 +/- 0.38 | 276.20 +/- 0.37 | **-1.94%** | 1.10x | 1.08x |
| tg512 | 281.22 +/- 0.62 | 275.68 +/- 0.53 | **-1.97%** | 1.08x | 1.06x |
| tg1024 | 278.81 +/- 0.52 | 272.93 +/- 0.61 | **-2.11%** | 1.07x | 1.05x |
| tg2048 | 274.43 +/- 0.69 | 266.11 +/- 0.50 | **-3.03%** | 1.06x | 1.03x |

A clean regression growing with cache length -- exactly the shape the
mechanism above predicts (more decode steps, more times the double-pass
penalty pays out). This is what motivated the gate.

Qwen3.5-0.8B-Q8_0, **after** the shape gate
(`autoresearch/beams/attn_persist_qwen_gated.{csv,json}`), same env var set
but `attn_persist_plan` now declines this shape and falls back:

| test | baseline t/s | persist (gated) t/s | delta |
| --- | --- | --- | --- |
| tg32 | 274.75 | 277.68 | +1.07% |
| tg128 | 281.67 | 282.77 | +0.39% |
| tg512 | 281.22 | 282.28 | +0.38% |
| tg1024 | 278.81 | 280.09 | +0.46% |
| tg2048 | 274.43 | 275.62 | +0.43% |

All within normal session noise of the baseline (the small positive drift
matches the direction of noise seen elsewhere in this session, e.g. the
round-level diagnostics in the raw JSON), confirming the gate routes Qwen back
to the unmodified launched path: the pass report confirms 220 launches, the
doc's own documented default count, with `attention_split`/`attention_merge`
both present and `attention_persist` absent.

### Against the actual goal

**This does not clear llama.cpp's FA-on baseline**, exactly as flagged before
starting: minicpm's ratio moves from 0.86-0.94x to 0.87-0.94x, roughly +0.01 at
every cache length. The beam file's own ceiling math said a *zero-cost*
attention kernel would only reach ~281 t/s against llama.cpp's ~275-278 (this
session's numbers), no margin, and this is a partial removal of one of
attention's two kernels' overhead, not a zero-cost kernel. tg2048 moved
235.57 -> 238.51 t/s; llama.cpp CUDA sits at ~274-278 t/s in the same session.
The gap remaining is dominated by attention's own compute/bandwidth cost
(`attention_split`'s share was already documented as 6-8x `attention_merge`'s
in every profiled window), which this round did not touch, plus the
[[launch-bound-headroom]] launches this beam does not reach (`store_2d`,
`quantize`, `q8_qdot_add` per layer, still unfused per that beam's own open
items).

### What is not chased this round

- **The shared-memory pooling gap** (combined kernel ~41KB vs the launched
  split kernel's ~24KB, not collapsing to the max of the two phases the way
  `docs/megakernel.md`'s redundant-stage fix does for the fused MLP) is a
  phobos-lang codegen investigation of its own, out of this round's budget.
  If fixed, Qwen's shape likely stops declining and the whole beam's win
  widens closer to the ~15-18% ceiling; this is the single highest-value
  follow-up either for this beam or for `docs/megakernel.md` generally, since
  the same pooling gap would affect any future two-phase persistent kernel.
- **Reducing `attention_split`'s own per-call cost** (documented as most of
  attention's share, 6-8x the merge kernel's) is untouched; this round only
  removed the launch/round-trip between two otherwise-unchanged kernel bodies.

### Beam status: kept open, not killed, one genuine partial win landed

Per AGENT.md's kill criteria, none apply: no correctness failure, no
across-the-board regression (the regression found was diagnosed and gated
around, not repeated), and the targeted cost (a launch + scratch round-trip
per full-attention layer) is real and partially removed, not shown immaterial.
Shipped as **opt-in** (`PHOBOS_ATTN_PERSIST=1`, unset means off) rather than
flipping the default: the gate's predicate is validated only at two shapes'
endpoints on one card (see above), and a shipped default should not
extrapolate past that on a change touching grid barriers. Whoever revisits
this: the next increment is either the pooling fix (widens which shapes win)
or a third shape/card to firm up the gate's predicate before proposing
default-on.

Combine-with-[[launch-bound-headroom]] verdict: **yes, still needed, and this
round adds evidence for it rather than replacing it.** This beam's own ceiling
math already said 15-18% attention share bounds a zero-cost kernel at ~281
t/s with no margin; a real (non-zero-cost, launch-count-only) implementation
reaching +1.2% on minicpm is consistent with that bound and nowhere near
closing it alone. `launch-bound-headroom`'s remaining unfused launches
(`store_2d`, `quantize`, `q8_qdot_add`) are untouched by this round and stay
the next lever.

### Win size, re-verified with a clean back-to-back run

A cross-session comparison (this beam's landed control run vs. the much
earlier `autoresearch_baseline.csv`) briefly suggested the win was actually
+2.1-3.7%, larger than the +1.0-1.4% recorded above. Before editing the
number, re-ran it properly: one `bench.py` invocation with
`PHOBOS_ATTN_PERSIST` unset immediately followed by one with it set to `1`,
same session, minimal gap, same card
(`autoresearch/beams/persist_{off,on}_clean.{csv,json,log}`):

| test | off | on | delta |
| --- | --- | --- | --- |
| tg32 | 255.01 | 257.80 | +1.09% |
| tg128 | 254.89 | 259.64 | +1.86% |
| tg512 | 251.96 | 254.63 | +1.06% |
| tg1024 | 245.45 | 249.64 | +1.71% |
| tg2048 | 234.91 | 237.66 | +1.17% |

+1.06% to +1.86%, matching the originally recorded +1.0-1.4% closely. **The
cross-session number was noise, not a bigger real win** -- confirmed by the
`off` run's own llama.cpp column showing a round-1 contention/clock blip
(-5.29% off that row's median, tg1024 stderr +/-36.73) despite passing
`bench.py`'s own contention gate, evidence that even an "uncontended" run
can carry session-level drift large enough to produce a false +2x reading if
the two sides being compared aren't measured back-to-back. **Lesson for
future rounds in this project: CLAUDE.md's "a performance number is only
comparable to one measured in the same session" is not satisfied by two
different sessions both being individually clean -- it needs one session,
both sides, minimal gap.** The beam's recorded number stands: +1.0-1.4% on
minicpm, unchanged.

### A bigger, separate finding: decode attention is far from its own bandwidth floor

Independent of the win-size question above, ran `attndecode`'s built-in
bandwidth-floor comparison (`autoresearch/beams/attndecode_roofline.log`) --
a tool this session had run for split-count sweeps but never read the
`floor`/`vs floor` columns of, which divide the *distinct* KV bytes a shape
must move by a measured device-copy bandwidth to get a lower bound, then
compare against the measured `attn/step` time. Stock (non-persistent)
split-plus-merge, minicpm shape, at the longest cache tested:

| cache | attn/step | distinct bytes | floor | vs floor |
| --- | --- | --- | --- | --- |
| 2053 | 3218.0us | 48.12 MB | 354.6us | **9.1x** |

The `group 1, same cache` diagnostic shape (2 heads instead of 16, `qgroup`
irrelevant since there is no group to share a read across -- isolates
whether GQA's redundant re-read explains the gap) still shows **8.1x** at
the same cache length. Since removing all redundant re-reads barely moves
the ratio, **the split-plus-merge kernel's inefficiency is not primarily the
GQA re-read this session's earlier `ATTN_QGROUP` tuning already addressed --
it is something else in the kernel/algorithm**, unexplored by any beam this
session tried. This is a fundamentally different situation from
[[wide-vocab-lm-head]]'s `q8_qdot` kernel, independently roofline-checked in
that beam at 92.8% of peak (nothing left to find). Decode attention has
real, unclaimed headroom -- how much of the session's 15-18%-of-a-step
attention cost is recoverable is unknown without deeper profiling (ncu, not
just this coarse floor check), but "far from its own floor" is a much
stronger starting position than "already near peak, nothing to find" for
justifying further work here.

Caveat: the `floor` figure depends on `attndecode`'s own device-copy
bandwidth probe (142 GB/s measured at the top of its run), which is well
below what a warmed, well-tuned kernel demonstrably achieves on this card
(`q8_qdot` measured 460.5 GB/s in [[wide-vocab-lm-head]]'s roofline). If the
probe under-measures true achievable bandwidth (plausible: `attndecode` has
no explicit card-warming step the way `bench.py` does), the true floor is
lower and the `vs floor` ratios above are *conservative* -- the real
multiple could be larger, not smaller. Either way the qualitative
conclusion (large, GQA-redundancy-independent headroom) holds; the exact
multiple needs a proper profiling pass to pin down before it drives an
implementation decision.
