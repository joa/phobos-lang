---
parent: fcfdf8b (HEAD, autoresearch branch)
status: killed (profiled, no viable lever within this beam's scope)
---

# Beam: decode-step lm_head projection efficiency

## Hypothesis

phobos trails llama.cpp on minicpm5-1b even at tg32 (259.5 vs 278.8, 0.93x),
before cache-length effects have room to matter much (see
[[cache-length-split-buckets]] for the length-dependent part of the gap).
Something length-*independent* accounts for a piece of the deficit. The
lm_head projection (`hidden[1,1536] x weight[1536,130560] -> logits[1,
130560]`) runs once per decode step regardless of cache length and is the
single largest matvec in the model (minicpm has no tied/smaller output head
distinct from this). Worth checking whether phobos's decode matvec path is
well-suited to this specific shape (wide N, quantized Q8_0 weight, M=1) or
whether it is leaving bandwidth on the table relative to llama.cpp's kernel
for the same shape.

Note this hypothesis is weaker than [[cache-length-split-buckets]]: minicpm's
vocab (130,560) is *smaller* than Qwen's (248,320), and Qwen already wins, so
vocab width alone does not explain the loss. The angle here is not "the
lm_head is big," it is "is phobos's kernel choice/tuning for *this exact*
K=1536 (minicpm's embedding_length), N=130560, Q8_0, M=1 shape as good as
llama.cpp's for the same shape" -- a shape-specific tuning question, not a
generic size one.

## Where to look

- `phobos-gguf/src/backend/device/backend.rs:172-204` (`fn matmul`): decode
  (`m == 1`) dispatches to `self.matvec.pick(n.is_multiple_of(MV_TN))`, an f32
  path. Confirm whether the lm_head actually goes through this or through a
  quantized path -- the model's weights are Q8_0, and
  `phobos-gguf/src/backend/device/kernels/quant.rs` plus the `qdot`
  fused-quantized-dot kernel (see `qdot-fused-quantized-dot` and
  `qmma-prefill-projection` memory) are what the rest of the decode
  projections use. Find the actual call site for the output/lm_head
  projection (`phobos-gguf/src/llama.rs:368`, `self.head.project_into(...)`)
  and trace `project_into` to whichever kernel it lowers to.
- Compare against every other decode matvec in the model (QKV projections,
  MLP up/down) which are `K` in the ~1536-4608 range but `N` far narrower
  (head_dim*n_head or feed_forward_length, all under 5000). The lm_head is
  the only one with `N` in six figures. Check whether `MV_TN` (matvec tile
  width) or the qdot kernel's tiling was chosen with the narrow shapes in
  mind and never re-checked against a vocab-width output -- grid size,
  blocks/SM, and whether `n.div_ceil(MV_TN)` blocks is enough to fill the
  card or whether launch overhead (single kernel, so unlikely, but check) or
  memory access pattern (row-major weight layout, coalescing) is off for a
  matrix this wide.
- Check `phobos-gguf/src/backend/device/backend.rs:123-152` (`fn read`):
  logits readback happens synchronously every decode step
  (`self.stream.synchronize()`) through a pinned staging buffer sized to
  `out.len()` (vocab). This is a genuine per-token sync point whose cost
  scales with vocab width. It is present for both models, so on its own it
  cannot be *the* differentiator (Qwen's is bigger and Qwen wins), but
  measure its share of one minicpm decode step regardless -- it may be worth
  a fixed number of tokens/sec at any vocab size and small relative to
  Qwen's larger step time but not to minicpm's smaller one.

## Correctness gate

Any kernel-selection or tiling change to the projection path needs
`backend_check` (every device op against the host reference) and
`model_check` (whole-model logits, device against host) before its timing
counts, per CLAUDE.md: an op that matches in isolation can still corrupt the
*next* allocation.

## How to measure

Isolate the lm_head cost with a targeted profile first, before changing
anything -- `nsys profile --trace cuda --cuda-graph-trace=node` on a decode
step (see `decode-step-is-launch-bound` memory for why the graph-trace flag
is required or nsys attributes nothing inside the CUDA graph), filtered to
the kernel(s) `project_into` lowers to. Get a number for what fraction of one
minicpm decode step this is before deciding whether it is worth chasing
further. If it is a single-digit percent of the step, this beam is weak
evidence against itself and should be down-ranked in favor of
[[cache-length-split-buckets]], not abandoned outright per AGENT.md's beam
discipline (kill only on inherent correctness failure or a profiled cost
that is no longer material -- a profile showing this is small **is** that
kill condition, so act on it if it comes back small).

## Result log

### Round 1: trace + profile, no code changes

**Kernel confirmed.** `self.head.project_into` (`phobos-gguf/src/llama.rs:368`) ->
`Linear::project_shared` (`phobos-gguf/src/layers.rs`) -> for a Q8_0 weight it
goes through `DeviceBackend::project_q8`
(`phobos-gguf/src/backend/device/matmul.rs:7`), never the f32 `matvec` path in
`backend.rs:172-204` (that path is dense-weight only). At M=1, N=130560
(divisible by `Q8_QDOT_TN=8`), `persist_qdot` off by default, this dispatches
to the single `q8_qdot` kernel (`phobos-gguf/src/backend/device/kernels/quant.rs:164-175`),
grid `(n.div_ceil(8), 1, 1) = (16320, 1, 1)`, 256 threads/block, one warp per
output column.

**Profile setup.** `nsys profile --trace cuda --cuda-graph-trace=node` on
`cargo run --release -p phobos-gguf --features cuda --example bench -- -m
models/minicpm5-1b-Q8_0.gguf -p 0 -n 32 -r 1` (warmup on). Card confirmed
uncontended first (`nvidia-smi`: 6% util, only desktop-compositor processes,
2551/8192 MiB used by something idle -- no other CUDA workload). Raw outputs
kept: `autoresearch/beams/lmhead_profile.nsys-rep`,
`lmhead_profile_cuda_gpu_kern_sum.csv`, `lmhead_profile_cuda_gpu_trace.csv`.
`q8_qdot` launches at grid=16320 are unambiguously the lm_head: every other
decode matvec in minicpm has N in the low thousands, so grid=16320 (=N/8)
never occurs anywhere else. Every `model.forward` call (prefill or decode)
projects only the last row through the head, so lm_head fires once per
forward call including the one 128-token warmup prefill; the 38 captured
`q8_qdot`-at-16320 instances = 1 prefill-warmup + 4 single-token warmup + 1
prime + 32 timed decode steps, confirmed by matching count against
`rms_norm` (the output norm, also once per forward call) and against
`fused`/`attention_split` counts (24 layers x 37 decode-shaped calls = 888,
excluding only the one wide prefill call).

**Time share**, steady-state (last 32 lm_head calls, i.e. the timed tg32
loop; per-step wall time taken as the delta between consecutive lm_head
launch timestamps, which brackets one full decode step including that
step's own lm_head and readback):

| | ns | % of decode step |
| --- | --- | --- |
| `q8_qdot` kernel (lm_head) | 491,050 avg | 10.9% (of nsys-profiled step, 4,495,420ns avg) / 12.3% (of bench.py's un-profiled tg32 step, 3,997,900ns from the fresh baseline's 250.13 t/s) |
| D2H readback (`backend.rs:123-152` `read`, pinned staging + `stream.synchronize()`) | 77,423 avg | +1.7% / +1.9% |
| **combined** | 568,473 avg | **12.6% / 14.2%** |

nsys inflates host/launch time more than kernel time, so the profiled-step
percentage (12.6%) is the floor and the un-profiled-step percentage (14.2%,
against bench.py's own measured tg32 rate) is closer to what actually happens
outside the profiler. Either way: **double-digit, not the single-digit
"weak-evidence-against-itself" case the brief flagged as an automatic
down-rank.** This part of the hypothesis is confirmed materially real.

**But the kernel is not under-tuned -- it is near the memory-bandwidth
roofline, and the "wider tile" fix has no headroom to give.**

Bytes a correct Q8_0 matvec at this shape must move, all irreducible (no
duplication: `row_scales`, the row-major copy `constant_quant` builds
alongside the block-major `scales`, is a *layout* of the same per-block-scale
data qdot's access pattern needs, not an extra copy of information already
present elsewhere in a form qdot could use for free):
- weight: N x K = 130,560 x 1,536 = 200,540,160 bytes (191.3 MiB)
- scales: N x (K/32) x 4 bytes = 130,560 x 48 x 4 = 25,067,520 bytes (23.9 MiB)
- output write: N x 4 = 522,240 bytes (negligible)
- total ~= 226.1 MB

Card: RTX 2080 Super, 256-bit bus, GDDR6 at `nvidia-smi
--query-gpu=clocks.max.memory` = 7751 MHz -> 15,502 MT/s -> theoretical peak
15,502e6 x 32 bytes/s = 496.06 GB/s.

Achieved: 226.1 MB / 491,050 ns = 460.5 GB/s = **92.8% of theoretical peak**.
(The repo's own `attndecode` "device copy bandwidth" benchmark -- a simple
D2D `backend.copy` pointwise kernel warmed 6s -- measured only 142 GB/s on
this card in this session, so lm_head's qdot is already running at 3.2x that
codebase-standard reference. That gap says more about the plain copy kernel
than about a lower achievable ceiling: theoretical peak, cross-checked against
the driver-reported memory clock, is the meaningful bound here.)

Cross-check against the narrower unfused `q8_qdot`/`q8_qdot_add` calls the
same trace captured (o_proj and, inferred from grid size, a fused QKV
projection -- K unverified for these, flagged as such): grid=320
(N=2560) ~370 GB/s, grid=192 (N=1536) ~317 GB/s, both below lm_head's 461
GB/s. Direction is unambiguous even with K unverified: **the six-figure-N
shape is the best-utilized matvec in the model, not the worst.** "Under-tiled
for wide N" is backwards -- the narrow shapes are the ones with more
launch/ramp overhead relative to their tiny working set; the grid this shape
gets (16320 blocks, one warp each) is exactly what lets it amortize that
overhead into near-roofline throughput. There is no grid-starvation or
under-parallelization to fix.

**Headroom ceiling, not just share.** The number that actually matters is
how much of the tg32 gap a perfect lm_head could close. Fresh baseline
(`autoresearch_baseline.csv`, same-session, 2026-08-16): phobos tg32 = 250.13
t/s (3997.9 us/token), llama.cpp CUDA tg32 = 271.65 t/s (3681.1 us/token).
Gap to close: 316.8 us/step (7.9% of phobos's step time).

A kernel hitting 100% of theoretical peak on the same 226.1 MB (not
achievable in practice, but it's the ceiling) would take 455.8 us, saving
491.05 - 455.8 = **35.3 us** off the kernel -- 0.9% of a decode step, about
11% of the 316.8 us that needs to be found. Even zeroing the kernel
entirely (physically impossible, the bytes have to move) caps out at 491 us,
~10.9-12.3% of the step -- more than the whole gap in isolation, but not a
credible target since the floor is 455.8 us, not zero.

One real, narrow lever exists and is being recorded rather than chased:
GGUF stores Q8_0 scales natively as f16; `constant_quant`
(`phobos-gguf/src/backend/device/backend.rs:206-231`) upconverts to f32 for
both `scales` and `row_scales`. Narrowing just `row_scales` (the one qdot
reads) back to f16 would cut its 25.07 MB to 12.53 MB, saving ~5.5% of
lm_head's traffic (~27 us, ~0.6% of a decode step). Not implemented: the
payoff is roughly a twentieth of the gap, and touching the Q8_0 upload path
and the `qdot_t` builtin's expected operand width is blast-radius-wide
relative to that payoff (every Q8_0 matvec in the model reads `row_scales`
or `scales`, not just lm_head's).

**Conclusion: kill, not a viable target for a tiling/kernel change within
this beam's scope.** The lm_head cost is real and double-digit (confirmed,
against the brief's own down-rank threshold), but it is not evidence of a
tunable inefficiency -- it is a near-roofline memory-bandwidth cost that
scales with vocab width and cannot be substantially reduced without either
(a) a marginal, high-blast-radius scale-width change worth under 1% of a
decode step, or (b) not computing the full logits vector at all (a sampling-
architecture change, out of scope for a matmul-path beam). No code changed;
`backend.rs`/`quant.rs`/`matmul.rs`/`llama.rs` are unmodified. Per
AGENT.md, this is the "profile evidence shows the targeted cost is no
longer material [to the proposed fix]" kill condition, refined: material as
a cost, immaterial as a *lever*.

**Free datum for [[launch-bound-headroom]]**, from the same trace, no extra
runs: summing every GPU event's duration inside one steady-state decode-step
window (bounded by consecutive lm_head launch timestamps) against that
window's wall time, averaged over 7 windows: wall 4,561,603 ns avg, kernel-
busy 3,906,354 ns avg, bubble (gap between launches, not accounted for by any
kernel's own execution) 655,249 ns avg = **14.4% of a decode step**, fairly
stable (13.7-15.5% across windows), 294 GPU-side events (kernels + memcpys)
per step across 24 layers. This lines up with that beam's own estimate
(~220 nodes x ~3.3us =~ 0.73ms) almost exactly and is bigger than lm_head's
own share -- logged in `launch-bound-headroom.md` directly rather than
repeated here.

Raw evidence kept: `autoresearch/beams/lmhead_profile.nsys-rep`,
`lmhead_profile_cuda_gpu_kern_sum.csv`, `lmhead_profile_cuda_gpu_trace.csv`
(the regenerable `.sqlite` nsys emits alongside was deleted). Reproduce with:

    nsys profile --trace cuda --cuda-graph-trace=node -o autoresearch/beams/lmhead_profile --force-overwrite=true -- \
      ./target/release/examples/bench.exe -m models/minicpm5-1b-Q8_0.gguf -p 0 -n 32 -r 1
    nsys stats --report cuda_gpu_kern_sum --format csv --output autoresearch/beams/lmhead_profile autoresearch/beams/lmhead_profile.nsys-rep
    nsys stats --report cuda_gpu_trace --format csv --output autoresearch/beams/lmhead_profile autoresearch/beams/lmhead_profile.nsys-rep
