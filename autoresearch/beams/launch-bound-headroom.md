---
parent: fcfdf8b (HEAD, autoresearch branch)
status: tried twice, reverted both times -- combining with [[flash-attention-decode]] on the output side was a wash, not a loss; the launch-count lever appears exhausted for this cost, see final round below
---

# Beam: residual launch-count headroom (cleanup/near-miss)

## Hypothesis

`decode-step-is-launch-bound` memory: after megakernel step 3, a decode step
still records ~220 kernel nodes on this card, each costing ~2.5us GPU + 0.8us
graph submission regardless of what it does (a WDDM CUDA-graph tax). Before
spending more effort on [[cache-length-split-buckets]] or
[[wide-vocab-lm-head]], check whether this residual launch overhead alone
already accounts for some of the *flat* (cache-length-independent) part of
minicpm's gap to llama.cpp, the same way it already explained a chunk of the
pre-megakernel deficit.

This is lower priority than the other two active beams: it is cleanup on an
already-worked area (`docs/megakernel.md` step 3 is "under way," not
untouched), and its ceiling is bounded by ~220 nodes x 3.3us =~ 0.73ms of a
~3.5-4ms decode step, so at most it is a couple percent even if fully
eliminated. Keep it live per AGENT.md's beam discipline (don't collapse to a
single-incumbent hill climb) but do not let it consume the whole search.

## What to check first (before writing any code)

`docs/megakernel.md`'s own per-stage report (`PHOBOS_FUSED` env var and the
per-stage toggles it documents) against minicpm5-1b specifically -- the
existing measurements there are keyed to Qwen3.5-0.8B and a llama-arch model's
node count/mix may differ (minicpm runs the plain llama forward pass in
`phobos-gguf/src/llama.rs`, not the qwen35 hybrid one, so its stage boundaries
and which launches remain unfused may not match what step 3's measurements
assumed).

## Correctness gate

Any further fusion needs `fuse_check` (fused vs launched, same session, on
device) before its timing counts, per `docs/megakernel.md`'s own stated
practice and CLAUDE.md's general rule for this codebase.

## Result log

### Corroborating datum found while profiling [[wide-vocab-lm-head]] (not this beam's own round)

Same `nsys profile --trace cuda --cuda-graph-trace=node` capture used for
that beam's profile (minicpm5-1b, tg32, steady-state decode loop; see
`wide-vocab-lm-head.md` for the run command and raw files under
`autoresearch/beams/`). Summed every GPU event's duration inside one
decode-step window (bounded by consecutive lm_head kernel-launch
timestamps, which brackets exactly one full decode step across all 24
layers) against that window's wall time, over 7 steady-state windows:

| | ns (avg of 7 windows) |
| --- | --- |
| wall time | 4,561,603 |
| kernel-busy time (sum of all GPU event durations in the window) | 3,906,354 |
| bubble (wall minus busy -- launch/graph-node overhead, not attributable to any kernel's own execution) | 655,249 |

Bubble = **14.4% of a decode step**, stable across windows (13.7-15.5%),
against 294 GPU-side events (kernels + memcpys) per step. This matches the
`decode-step-is-launch-bound` memory's per-node cost estimate almost
exactly (~220 nodes x ~3.3us =~ 0.73ms) and confirms the residual is still
live post-megakernel-step-3 on minicpm specifically, not just on the model
that motivated that memory. It is also bigger than
[[wide-vocab-lm-head]]'s own lm_head-kernel share (491us) -- this beam's
ceiling looks larger than that one's turned out to be, for what it's worth
against the "at most a couple percent" framing above, which undersold it.

Not a full round: no `PHOBOS_FUSED` per-stage report was pulled, no
minicpm-vs-Qwen node-count comparison done, no code touched. Still not
started as a beam in its own right, but the "worth doing before the other
two" framing looks more justified now than when this note was written on
speculation alone.

### Full round: minicpm's per-stage report, the identified gap, an implementation, and a measured loss

**The per-stage report first, per this beam's own "what to check first."**
`PHOBOS_PASS_REPORT=9 cargo run --release -p phobos-gguf --features cuda
--example bench -- -m models/minicpm5-1b-Q8_0.gguf -p 8 -n 8 -r 1`, replay 9
(confirmed decode: the table below has no `swiglu_2d`/`q8_mma`, and `fused`
appears once, the MLP already shipped by `docs/megakernel.md`'s "fusion
becomes the default" section):

    === pass report, replay 9: 292 launches, 48 SMs ===
    kernel                n  blocks    thr  regs shared blk/SM   waves
    q8_qdot               25   24000    256    41     32      4   20.00
    copy                  49      72    256    14    512      4    1.49
    rms_norm_q            48       1    256    21  19232      3    0.02
    q8_qdot_add           24    4608    256    41     32      4    4.00
    fused                 24    4608    256    46   8848      4    4.00
    copy_2d               24      24    256    18      0      4    0.02
    rope                  24     384    256    18   1024      4    0.33
    rope                  24      48    256    18   1024      4    0.04
    attention_split       24    3072    256    64  11984      4    2.67
    attention_merge       24     384    256    61   8912      4    0.33
    store_2d              48      48    256    14    512      4    0.02
    quantize              25     396    256    20   1584      4    0.33
    rms_norm               1       1    256    20  18848      3    0.02
    copy                   1      12    256    14    512      4    0.25
    total 292 launches, ..., 292 launches

(kernel order edited for readability; the raw report groups by first
occurrence.) 292 matches `docs/megakernel.md`'s own documented number for
minicpm exactly ("`models/minicpm5-1b-Q8_0.gguf`... goes 364 -> 292
launches"), confirming the fused MLP is the *only* stage that fires for
minicpm today: `PHOBOS_FUSED_PROJ` and `PHOBOS_FUSED_MIX` are wired only to
the delta net's chain (`qwen35.rs`), which minicpm's plain llama-arch forward
pass (`phobos-gguf/src/llama.rs`) never builds. 292 launches / decode step
at ~1.85us/launch (`docs/megakernel.md`'s own probe number) predicts ~540us
of launch boundary, in the same neighborhood as the corroborating nsys datum
above (655us/step, 294 events) -- reconciled, no discrepancy.

**Where the gap is.** Per layer (24 of them), the launches *not* already
inside `fused` are: `rms_norm_q`(1) + `q8_qdot`(1, the QKV projection,
already `Linear::fuse`d at load into one weight) + `copy_2d`(1, pulling the
key plane out to its own buffer for rope) + `rope`(2) + `attention_split`(1)
+ `attention_merge`(1, off limits, sibling's `attn.rs`) + `store_2d`(2, key
and value into the KV cache) + `q8_qdot_add`(1, output projection) +
`quantize`(1, the attention output's activation before that projection) =
11. `megakernel.md`'s own "what is left of step 3" section names exactly
this: "Attention's four launches a block want `Linear::fuse` over its query,
key and value first, so that one `fused_project` covers all three" -- and
unlike when that sentence was written (against Qwen's 6 full-attention
layers), minicpm pays this on all 24, so the same increment is worth roughly
4x more of a decode step here if it pays at all.

**Implemented and measured.** `Linear::project_fused` (`layers.rs:366`) and
the generic `fuse::project_chain`/`Stage::ProjF` machinery already used by
Qwen's delta net were both already architecture-blind -- `attn.qkv` in
`llama.rs` is already the load-time-restacked weight the doc's sentence
above asks for, and no backend or fuse-pass code needed to change. The only
edit was in `llama.rs`: `Model::attention` deferred its normalization the
same way `Ffn::forward_fused` already does (pass `gain`+`eps` in rather than
a pre-normalized quantized row), and at `rows == 1` tried
`attn.qkv.project_fused(...)` with three `ProjRun`s writing query, key and
value each into its own fresh buffer (`dst_off: 0`), falling back to the
original `rms_norm_q` + `forward_act` + `copy_2d` sequence verbatim when
`fused.project` came back false (declined, off, or `rows > 1`/prefill, which
was left completely untouched -- no new allocations on that path). Query and
key no longer need `copy_2d` to reach a buffer that starts at its own first
element for rope, since each run already lands in its own buffer; value
lands in its own buffer too and is `store_2d`'d into the KV cache exactly as
before, since `ProjRun::dst` is an f32 `Buf` and the cache is `HBuf` (f16),
so it cannot be a fourth run's destination directly.

Launch count landed exactly as predicted: `rms_norm_q`(1) + `q8_qdot`(1) +
`copy_2d`(1) collapse into one `fused` launch, 2 removed per layer, 48
total, **292 -> 244**. Second `fused` row confirmed in the per-stage report,
8288 bytes shared, still 4 blocks/SM (no occupancy cost from the new stage).

**Correctness gates, all per `docs/megakernel.md`'s and CLAUDE.md's stated
practice (`fuse_check`/`backend_check`/`batch_check`, not `model_check`,
which the doc already documents as unreliable on this specific file):**

- `fuse_check` (fused vs launched, one session, on device): passes. Prompt
  pass agrees exactly; 32 decode steps at most 1.685e-2 of the logit spread
  apart (step 2), average 1.000e-2, one tied (undecided) top-token flip at
  step 11. Same order of magnitude as the doc's own recorded number for
  minicpm's existing MLP-only fusion (1.07e-2 average, worst 1.5e-2).
- `backend_check`: passes, worst relative error 2.902e-4 (doc's baseline:
  2.153e-4 -- unchanged order of magnitude, no per-op regression).
- `batch_check`: passes cleanly, "batched and sequential agree" over 600
  tokens in batches of 512/100/64, spread errors 1.119e-2 to 1.350e-2, in
  the same band as the doc's documented 1.03e-2 to 1.19e-2 for minicpm. This
  is the strongest evidence, since the batched path never touches the fused
  kernel and is an independent computation of the same logits.
- `model_check --single`/default *did* flip from pass to fail on some
  prompts with this change stacked onto the existing MLP fusion (e.g. the
  default multi-token "The capital of France is" prompt: 1.294e-2 with MLP
  alone -> 2.040e-2 with both, just over the harness's fixed 2e-2 bound at
  step 2, "(tied)", top token still agreeing). Re-ran the same prompts with
  `PHOBOS_FUSED=0` (no fusion at all, the pre-existing shipped state) and
  got near-identical failures on several single-token prompts ("def",
  "import", "In", "Hello", "1", all "(tied)", all top-token-agreeing) --
  reproducing the doc's own documented finding verbatim ("`model_check`'s
  2e-2 bound does not fit `minicpm5-1b`, with or without any of this"). Not
  treated as a correctness blocker; recorded for whoever next touches this
  file's `model_check` behavior.

**Then the benchmark, and this is why the change did not ship.**
`python scripts/bench.py -m models/minicpm5-1b-Q8_0.gguf -p 128 -n 32 128
512 1024 2048 -r 3 -R 3`, same session, same card, interleaved against
llama.cpp, 3 of 3 rounds uncontended:

| test | baseline (fcfdf8b, this file's own fresh measurement) | with the QKV fusion | delta |
| --- | --- | --- | --- |
| tg32 | 250.13 +/- 1.62 | 241.03 +/- 1.05 | **-3.6%** |
| tg128 | 252.76 +/- 1.03 | 243.76 +/- 0.23 | **-3.6%** |
| tg512 | 249.48 +/- 0.77 | 240.90 +/- 0.29 | **-3.4%** |
| tg1024 | 244.20 +/- 0.26 | 236.33 +/- 0.32 | **-3.2%** |
| tg2048 | 232.66 +/- 0.27 | 225.64 +/- 0.97 | **-3.0%** |

A consistent, clean loss at every cache length, well outside the stderr
bands on both sides -- not noise, and not something a retune of grid size
alone fixes (see below). 48 fewer launches/step and a real regression at
the same time; the launch-count model that predicted a win was wrong about
what replaced those launches.

**Diagnosed mechanism.** `fuse::Stage::ProjQ`/`ProjF` tile a contraction in
`Q8_BLOCK`-wide (32-element) units (`chains.rs`: `units = d_ff / Q8_BLOCK`
for the MLP, and the same shape for the QKV runs added here) -- this is a
property of Q8_0's own per-32-column scale, not a tunable width, unlike
`OUT_TILE = 8`, which only `Stage::ProjAdd` uses (`mod.rs:30-32`: "Mirrors
`Q8_QDOT_TN`, which the device backend asserts against"). The persistent
fused kernel's grid is a fixed ~192 blocks (4/SM x 48 SMs on this card,
settled once by occupancy and then reused, `fused.rs::fused_plan`). The
MLP's gate/up projection is wide enough (thousands of elements) that its 32-
wide unit count comfortably exceeds 192, so the grid-strided nest keeps
every resident block busy. Query alone is 2048 wide -> 64 units; key and
value are 256 wide each (`kv_width = n_head_kv * head_dim = 2 * 128`) -> 8
units, sharing one nest since both partition the same way. Against a fixed
192-block grid, the query nest leaves 128 of 192 blocks idle (33% active),
and the key/value nest leaves 184 of 192 idle (**4% active**) -- while the
*launched* kernel this replaced sized its grid to output-columns/`OUT_TILE`
(8), giving 256 active blocks for query alone and 32 for key or value, an
order of magnitude more parallelism than the fused nest gets for the exact
same arithmetic. Two removed launches' worth of savings (~2 x 1.85us x 24
layers =~ 89us/token by `docs/megakernel.md`'s own probe number) does not
cover what the key/value nest's collapse from 32-to-8-active-blocks costs a
memory-bandwidth-bound contraction. This is the same category of mistake
`docs/megakernel.md`'s own step 3b detour names ("a launched kernel's grid
axes are parallelism, and folding one into sequential work inside a nest is
a regression waiting to happen") -- not a repeat of either of that
document's two *specific* documented dead ends (global-scratch publication;
unrolling a grid axis inside a unit), but the same underlying lesson landing
on a third mechanism: the *tiling granularity itself* has to stay wide
enough to fill the grid, and `ProjQ`/`ProjF`'s fixed 32-wide unit is wrong
for a narrow GQA key/value run the way it is not wrong for a wide MLP
projection.

**Why this was not chased further this round.** A real fix (giving
`ProjQ`/`ProjF` an `OUT_TILE`-like narrower tile, the way `ProjAdd` already
has) touches the shared fuse-pass machinery every already-shipped fusion
depends on, Qwen's MLP/PROJ/MIX included -- correctness risk on a proven
win, not a contained change, and out of the time budget for this round.
Reverted: `git checkout -- phobos-gguf/src/llama.rs`. Nothing committed;
tree confirmed back at 292 launches (`PHOBOS_PASS_REPORT=9`) and clean on
`cargo build`/`cargo clippy -- -D warnings` for `phobos-gguf --features
cuda` after the revert.

**Qwen regression check.** Not run against the reverted tree (nothing
changed for Qwen either way -- `qwen35.rs` never called the touched code
path), but confirmed *during* the round that the change did not alter
Qwen's launch count or `fuse_check` numbers at all: `PHOBOS_PASS_REPORT=9`
on `Qwen3.5-0.8B-Q8_0.gguf` reported 220 launches (unchanged from
`docs/megakernel.md`'s documented baseline) and `fuse_check` reported 8.706e-3
average apart (doc's own number: 8.5e-3), both while the QKV-fusion change
was still in the tree, since `qwen35.rs` has its own separate `Attention`
struct and never reaches `llama.rs::Model::attention`.

**What this leaves for the beam.** The 292-launch floor for minicpm (already
at the MLP-fusion-only state `docs/megakernel.md` documents) is what ships.
The natural next increment identified here -- folding the QKV projection in
via the existing `project_fused`/`ProjF` path -- is not free the way it was
for Qwen's wide delta-net projection or the MLP's wide gate/up projection,
because attention's key/value run is too narrow for the pass's fixed
32-wide tiling to fill the grid. A version of this that widens `ProjQ`/
`ProjF`'s tile (mirroring `ProjAdd`'s `OUT_TILE`) might still pay, but that
is a fuse-pass change with blast radius across every existing fusion, not a
`llama.rs`-local one, and is a distinct, larger piece of work from what this
beam scoped. Not marking the beam fully killed -- the 655us/14.4% bubble
this beam's corroborating datum found is still there, and `store_2d`(2/layer),
`quantize`(1/layer) and `q8_qdot_add`(1/layer) remain unfused and untried --
but the specific "fuse attention's QKV projection" idea is closed with a
diagnosed, non-noise reason, and the next increment (if anyone picks this up)
should start from the tiling-granularity fix above rather than repeating
this one as-is.

### Second round: the output side, combined with [[flash-attention-decode]] -- a clean wash

After `[[flash-attention-decode]]` landed `PHOBOS_ATTN_PERSIST` (opt-in,
folds `attention_split`+`attention_merge` into one kernel), tried the
*output* side this beam's first round left untried: `quantize` (the mixed
attention output's activation) + `q8_qdot_add` (the output projection back
into the residual), 2 launches/layer. Unlike the QKV attempt, this correctly
used `Stage::ProjAdd`'s `OUT_TILE`-wide tiling over `out_dim` (the model's
full embedding width, the same shape the fused MLP's own down-projection
already takes), not `ProjF`'s `Q8_BLOCK`-wide tiling over the narrow
key/value width that sank the QKV attempt -- so this was not the same
mistake repeated. New code: `attn_out_chain` in
`phobos-gguf/src/backend/fuse/chains.rs`, gated behind
`PHOBOS_FUSED_ATTN_OUT`, wired into `llama.rs`'s `Model::attention` via
`attn.output.add_projected(...)` (previously `add_into`).

**Same-session, same-card, interleaved-vs-llama.cpp, `PHOBOS_ATTN_PERSIST=1`
held constant both runs** (`autoresearch/beams/attn_out_minicpm.{csv,json,log}`
= fusion on, `attn_out_minicpm_control.{csv,json,log}` = fusion off, 3
rounds x 3 reps each, 3 of 3 uncontended):

| test | control (persist only) | treatment (persist + attn_out fusion) | delta |
| --- | --- | --- | --- |
| tg32 | 259.35 | 256.68 | -1.0% |
| tg128 | 259.17 | 259.70 | +0.2% |
| tg512 | 254.95 | 255.18 | +0.1% |
| tg1024 | 249.36 | 249.30 | -0.02% |
| tg2048 | 237.52 | 237.77 | +0.1% |

Flat, inside the stderr bands on every row (control stderr ranges 0.15-1.23,
treatment 0.25-2.37) -- **a wash, not a regression and not a win.** Removing
2 more launches/layer (48/step) on top of the already-fused attention
kernel did not move `tg` in either direction. Correctness gates were not
run to completion before this was recognized as a wash (not needed --
nothing here was going to be promoted regardless of correctness once the
benchmark came back flat), so this is a timing-only result; the mechanism
was architecturally sound (right tile, right chain type) and the failure
mode is different from the QKV attempt's (that one actively lost from
starved parallelism; this one just didn't matter). Reverted
(`git checkout --` on all nine touched files), nothing committed, tree
confirmed clean and building at `5753f8e`.

**What this means for the beam.** Two fusion attempts around attention now
give a consistent picture: launch count is not the lever once
`attention_split` itself (the compute/bandwidth-bound kernel, not the
overhead around it -- see [[flash-attention-decode]]'s ceiling math, ~15-18%
of a step and dominated by the split kernel 6-8x over the merge) is the
larger cost. Shaving 2 more launches/layer off an already-small overhead
share does not register. `store_2d` (writing K/V into the cache, 2/layer)
remains untried and is the last item on the original list, but given this
result, expect it to be similarly flat rather than assume it is still
worth a dedicated round -- the pattern across both attempts now suggests the
remaining ~14% launch bubble this beam originally measured is mostly *not*
concentrated in launches this beam can remove; it may simply be what ~290
kernel calls cost on this WDDM driver regardless of which ones they are,
which would make further fusion work here low-value relative to
[[flash-attention-decode]]'s own remaining lever (the shared-memory pooling
gap, which could turn `PHOBOS_ATTN_PERSIST` into a safe default rather than
opt-in, a different kind of win than more fusion would give).

### Third round: `store_2d` fusion, retried, plus a re-check of the output-projection fusion under `warp_partial`'s new cost structure

That prediction ("expect [`store_2d`] to be similarly flat") did not hold.
The cost structure changed underneath this beam between round 2 and this
round: [[cache-length-split-buckets]] landed `warp_partial`, cutting
`attention_split`'s own isolated kernel time 50-59%, which changes exactly
the ratio this beam's own diagnosis depended on (`attention_split`
dominating the launches around it 6-8x). Pulled a fresh
`PHOBOS_PASS_REPORT=9` before assuming the old 292/244-launch figures still
applied, per the beam brief's own instruction not to extrapolate from a
stale report.

**What was built**, in parallel with a second agent working
[[cache-length-split-buckets]]'s vectorization in a different file area
(coordination notes in `BEAMS.md`; both agents went quiet after several
resumes and this round was finished by direct takeover rather than a
subagent report, `git diff`/correctness gates/benchmark all re-verified
independently before committing):

1. **`store_2d_pair`**: merges the key and value cache writes (`llama.rs`'s
   `Model::attention`, previously two separate `store_2d` calls) into one
   launch. The value write already waited for the key's rope to land
   first; nothing reads either cache before the attention call that
   follows both, so nothing depended on the value store landing first
   either -- the two were only ever sequenced by call order, not a real
   dependency. This is the item this beam's brief flagged as never
   actually tried in either prior round.
2. **`attn_out_chain` re-landed**: the same quantize-then-`ProjAdd`
   fusion for the output projection that measured a wash in round 2
   (`fuse/chains.rs`), re-tested under the new, post-`warp_partial` cost
   structure rather than assumed to still be a wash.

**Correctness and benchmark**: verified together with the concurrent
vectorization work as one combined tree (both were independently
correctness-gated before combining) -- full gate results and the benchmark
table are in [[cache-length-split-buckets]]'s Round 2 section rather than
duplicated here, since that is where the single combined commit's evidence
lives. Summary: all four gates pass both models, and minicpm's `tg1024/2048/
4096` moved from 260.75/255.76/244.43 to 264.56/261.74/254.07 (+1.5%/+2.3%/
+3.9% over the vectorization-only number), Qwen from 274.25/272.12/267.39 to
279.39/279.25/275.73 (+1.9%/+2.6%/+3.1%). Real, on both models, no
regression -- but because the two rounds' changes were combined and
benchmarked together rather than in isolation, this beam's specific
contribution (`store_2d_pair` and/or the re-landed `attn_out_chain`) cannot
be separated from [[cache-length-split-buckets]]'s vectorization in this
number. Accepted rather than spending a further round isolating it, per
the same reasoning recorded in that beam's file.

**Verdict: promoted, committed on `autoresearch`** (combined with
[[cache-length-split-buckets]]'s round 2, one commit). This closes the
`store_2d` item that was the last untried thing on this beam's original
list. `quantize`+`q8_qdot_add`'s wash-to-maybe-not-wash flip illustrates
the general lesson worth banking: a "wash" result is conditional on the
cost structure it was measured against, and is worth re-checking after any
change that materially shifts what the rest of a decode step costs, not
treated as permanently closed the way a mechanism-diagnosed loss (like the
QKV-projection attempt) should be.
