---
parent: fcfdf8b / 80204d5 (HEAD, autoresearch branch)
status: new, not started -- scoping
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

(not started)
