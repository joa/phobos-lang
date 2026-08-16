---
parent: fcfdf8b / 80204d5 (HEAD, autoresearch branch)
status: new, not started -- scoping
---

# Beam: flash-attention-shaped decode attention

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
