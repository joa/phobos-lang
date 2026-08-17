# q8_qmma split-K for starved prefill grids

Beam: give `q8_qmma`'s deep tile more resident blocks on a starved grid by
splitting `k` across `program_id(2)`, instead of shrinking `TM`/`TN` (which an
earlier probe in this same investigation measured as a net loss on shapes
that were already reasonably fed). Diagnosis (ncu, `-c 20 --set full`) is in
the task brief this beam was given; this file covers what was built and
measured on top of it.

## Result summary

**Second update, after final review: this beam is a negative result at the
level that matters.** The per-kernel numbers below are real and held up
under an isolated `ncu` A/B. But an end-to-end `scripts/bench.py --no-llama`
A/B on minicpm's `pp128` (isolated from llama.cpp/cross-process noise, four
runs, consistent direction) found split-K **costs 14-19% of end-to-end
throughput**: `PHOBOS_QMMA_SPLIT` off averages 5524 t/s, on averages
4480-4774 t/s. Two full `bench.py` runs against llama.cpp confirm the same
direction: the established pre-split ratio of 0.76x dropped to 0.61x and
0.54x with split-K on. See "End-to-end result and the graph-node mechanism"
below for the numbers and the nsys trace that explains them: every
split-routed projection replaces one launch with two (write, then reduce),
and those extra serialized graph nodes cost more in aggregate wall clock on
this platform than the kernels save, even though no individual kernel got
slower. **`qmma_split` now defaults off** (`PHOBOS_QMMA_SPLIT=1` opts in).
Everything below the per-shape kernel numbers -- the mechanism, the two
compiler-proof bugs, the correctness verification -- is kept as documentation
of real, working code that just doesn't pay for itself end to end on this
platform at these shapes; it may be worth revisiting at a different
node-count/model mix (Qwen's per-kernel wins were larger than minicpm's,
noted below).

Per-kernel, on minicpm5-1b-Q8_0's real `pp128` projection shapes (gate
requires `splits == Q8_QMMA_SPLIT_MAX`; see "Gate fix" further down -- qkv
is excluded from routing entirely, not regressed, at the per-kernel level):

| shape | baseline | split-K (write+reduce) | delta |
| --- | --- | --- | --- |
| o_proj (n=1536, k=2048, S=8) | 90.7 us | 50.9 + 19.4 = 70.3 us | **-22.5%** |
| down_proj (n=1536, k=4608, S=8) | 191.8 us | 89.5 + 18.5 = 108.0 us | **-43.7%** |
| qkv (n=2560, k=1536) | 73.7 us | 73.7 us (unsplit, correctly excluded) | 0% |
| gate/up (n=9216, unsplit, sanity check) | 140.2 us | 141.7 us | +1.1% (noise, correctly declined) |

Qwen3.5-0.8B-Q8_0 shows the same per-kernel pattern, more favorably: every
shape it routes through split-K wins, by more than minicpm's win:

| shape | baseline | split-K | delta |
| --- | --- | --- | --- |
| n=1024, k=2048, S=8 | 87.9 us | 41.9 + 13.2 = 55.1 us | **-37.3%** |
| n=1024, k=3584, S=8 | 147.9 us | 64.8 + 14.2 = 79.1 us | **-46.5%** |

Correctness holds on both models at every routed shape (see below), and
still holds with `qmma_split` defaulted off and re-enabled as an opt-in.

## Mechanism

`q8_qmma`'s deep tile launches `(rows / 128) * (n / TN)` blocks and nothing
else. On minicpm's `pp128` (M=128 exactly), that grid is a constant 12, 20 or
72 blocks depending on projection width, against 48 SMs. Freshly measured
this session (`ncu_qmma_splitk_baseline.ncu-rep`, matches the task's own
diagnosis numbers closely): achieved occupancy sits at 12.5% for grid=12 and
20, 19.1-19.2% for grid=72, against the 25% register-bound ceiling this
kernel's 128-accumulator patch caps a CTA to (2 resident blocks/SM at
`Q8_QMMA_CTA=128` threads/block, sm_75, 238 registers/thread).

Splitting `k` across more blocks (same `TM`, same `TN`, same arithmetic
intensity, same tensor-core efficiency) is the fix that adds blocks without
paying the tile-shrink probe's efficiency cost. The obstacle was entirely a
compiler-proof one, not an algorithmic one, and it took two failed designs
before landing on one that works:

1. **First attempt**: fold the split index into the destination offset,
   `P[ps * M + pm * TM :+ TM, ...]`. Verified via a throwaway `.ph` file
   through `cargo run -p phobos-lang --example emit` that this compiles
   *without error* -- but silently declines `qmma_t`'s direct-to-global
   write and falls back through a shared-memory round trip instead (visible
   in the emitted MLIR as `assume_align` + a masked `scf.if` copy-out). The
   reason: a dynamic tensor-shape symbol like `M`, used in arithmetic, is
   only ever assumed a divisor of 4 (the row-pitch ABI), regardless of what
   `@aligned` promises about the tensor it indexes -- `@aligned`'s promise
   attaches to a specific operand's declared extent, not to the symbol's
   general arithmetic use. `gcd(expr_div(ps*M), expr_div(pm*TM))` bottoms
   out at 1, so the bounds proof (`dyn_in_bounds` in
   `phobos-lang/src/codegen/util.rs`) never clears, and `qmma_t`'s
   direct-write optimization (`phobos-lang/src/codegen/stmt.rs`, gated on
   `!target.is_masked()`) declines. That fallback caps the kernel at one CTA
   per SM (a `[128,64]` f32 shared tile is 32 KB) -- worse than the 25%
   ceiling this beam is trying to raise, so this design would have bought
   nothing.
2. **Working design**: give each split its own output *operand* and its own
   `if ps == i` branch, so every write keeps the exact `pm * TM, pn * TN`
   offset the unsplit kernel already proves unmasked (verified the same way,
   confirmed clean MLIR with no `scf.if`/`assume_align` beyond the `S`
   branch conditionals themselves, which are uniform across the CTA -- a
   predicated dispatch, not warp divergence). `k`'s slice bounds are baked in
   as *literals* per branch (`from = i * slice`), which sidesteps the same
   proof issue from the other side: a literal offset's own divisor is
   itself, so `@aligned(K = slice, KB = slice / 32)` clears the bounds proof
   on `A`, `AS`, `W` and `WS` with no dependency on `program_id` at all.
3. **Reduce kernel, second bug**: the first version summed `S` output tiles
   with a plain `[TM, TN]`-shaped add (`P0[...] + P1[...] + ...`). That's not
   `qmma_t`, so it always stages through shared memory regardless of
   masking -- and at the real `TM=128, TN=128` case that's a 64 KB tile, over
   the 48 KB static ceiling. This passed MLIR verification *and* LLVM/NVPTX
   codegen (`cargo run -p phobos-lang --example ptx` produced valid PTX text
   using the kernel's smaller default autotune choice), and only failed at
   `Module::from_ptx` -- a driver-level PTX JIT rejection neither `emit` nor
   `ptx` would ever catch, since both stop before loading the module. Caught
   by running a real `pp128` shape through `bench.rs` with temporary debug
   instrumentation. Fixed by reducing one row at a time (`[1, TN]` tiles,
   grid `(rows, n/wide, 1)`): a size-1 span is unmasked unconditionally
   (`dyn_in_bounds`'s `size <= 1` case divides by 1 either way), matching
   the existing `Q8_SPLIT_SRC`/`q8_reduce`'s own row-at-a-time pattern.

## What was built

- `phobos-gguf/src/backend/device/kernels/quant.rs`:
  - `Q8_QMMA_SPLIT_THRESHOLD` (48), `Q8_QMMA_SPLIT_TARGET` (96),
    `Q8_QMMA_SPLIT_MAX` (8) -- the gating constants from the task brief.
  - `q8_qmma_splits(rows, n, k, wide)` -- returns 1 (unsplit) when the
    deep tile's own grid is already at or above the threshold, else a split
    count that lands near the target block count and divides `k`'s block
    count evenly, halving down when it doesn't (mirrors `q8_splits`'s own
    halving loop for the decode path). **Only returns a split when the
    halving loop still lands exactly on `Q8_QMMA_SPLIT_MAX` (8)** -- see
    "Gate fix" below; anything short of the max falls back to 1 (unsplit).
  - `q8_qmma_split_src(block, k, s)` -- the branch-per-split kernel
    described above.
  - `q8_qmma_reduce_src(block, s)` -- the row-at-a-time reduction.
- `phobos-gguf/src/backend/device/matmul.rs`: `project_q8`'s deep-tile
  iteration now computes `splits` and, when `> 1`, calls the new
  `launch_qmma_split` instead of the plain `q8_qmma` launch. The shallow
  remainder pass and the deep pass's own dispatch order are untouched, as
  asked. `launch_qmma_split` compiles and caches both kernels per
  `(wide, k, splits)`, reuses the existing `split_partials` scratch
  allocator (shared with the decode path's own split-K, same
  grow-then-flush recording-rule handling, no new allocator code), and
  issues the write launch (grid `(rows/TM, n/wide, splits)`) followed by the
  reduce launch (grid `(rows, n/wide, 1)`).
- `phobos-gguf/src/backend/device/mod.rs`: a `q8_qmma_split` module cache
  (`HashMap<(wide, k, splits), (Module, Module)>`) and a `qmma_split: bool`
  field, gated by `PHOBOS_QMMA_SPLIT` -- initially built default-on/opt-out
  (matching this codebase's `PHOBOS_ATTN_PERSIST`/`PHOBOS_FUSED_MLP`
  convention) since that's what let the same-session register/occupancy/
  timing A/B below run both paths without reverting code between
  measurements; **flipped to default-off/opt-in** after the end-to-end
  result below, `PHOBOS_QMMA_SPLIT=1` now required to route through it at
  all.

No change touches `qmma_t`'s builtin codegen, `q8_qmma_src`,
`Q8_QMMA_TM`/`Q8_QMMA_WIDTHS`, or the existing deep/shallow dispatch order,
per the task's standing rules.

## Emit diff-sweep

`git diff --stat` shows only `phobos-gguf` files touched; zero changes under
`phobos-lang` (the compiler `emit` runs). Since `emit`'s output is a pure
function of `phobos-lang` + the `.ph` text and neither changed, the
before/after diff is zero by construction. Ran it anyway as a sanity check:
every `.ph` file under `examples/` and `phobos-lang/examples/` still compiles
clean under both the default target and `PHOBOS_CHIP=sm_80
PHOBOS_INDEX_BITS=64`.

## Correctness

`cargo clippy -p phobos-gguf --release --features cuda -- -D warnings`:
clean. All files stay well under the 900-line cap
(`quant.rs` 412, `matmul.rs` 281, `mod.rs` 472).

The four standing gates (`backend_check`, `batch_check`, `model_check`,
`fuse_check`), both models, `--release --features cuda`: all pass. But it's
worth being explicit that **none of those, at their default arguments,
exercise `splits > 1`** -- `model_check`/`fuse_check`'s default prompts are
~5 tokens (never reach the M=128 deep tile at all), `batch_check`'s hardcoded
batch sizes (512/100/64) push the deep tile's own row count past where
`unsplit` clears the 48-block threshold, and `backend_check`'s one M=128
synthetic shape lands on `splits=1` after the divisibility-halving loop. That
gap is exactly how the shared-memory reduce bug above slipped past all four
gates and was only caught by a manual `bench.rs -p 128` run.

So the real validation is `model_check -p "<128+ token prompt>"`, which
compares device against the host reference at the actual routed shapes:

- **minicpm**, prompt tokenizing to 128 rows: routes `n=1536,k=2048
  splits=8`, `n=1536,k=4608 splits=8`, `n=2560,k=1536 splits=4`,
  `n=9216,k=1536 splits=1` (declined) -- `backends agree`, spread error
  1.1e-2 to 7.4e-3 across the first 3 steps, same magnitude as the
  pre-existing non-split baseline's own host/device noise, no top-token
  flips.
- **minicpm**, a 256-row prompt: routes splits 4/4/2/1 -- `backends agree`.
- **Qwen**, 128-row prompt: routes `n=1024,k=2048 splits=8`,
  `n=1024,k=3584 splits=8`, `n=512,k=1024 splits=8`, and *correctly declines*
  `n=4096,k=1024` (unsplit=32 is under the 48 threshold, but
  `1024/32=32` k-blocks isn't evenly divisible by any split count down to 1,
  so the divisibility guard falls back to unsplit rather than construct an
  invalid K-slice) -- `backends agree`.
- Re-ran all 8 standard gates (4 checks x 2 models) once more after removing
  debug instrumentation and adding the `PHOBOS_QMMA_SPLIT` toggle: all still
  pass, in both toggle states.
- Re-ran all 8 standard gates plus both models' 128-token `model_check` a
  third time after the post-review gate fix (below): all still pass. The
  numbers in this section (shapes, split counts, spread errors) predate that
  fix and describe the state where qkv still routed through split-K at
  `S=4`; the fix does not change what any of these shapes compute, only
  whether qkv is routed at all, so none of this evidence needed to change.

**Device-pool recording-rule check**: `launch_qmma_split` calls the existing
`split_partials`, unchanged, which already implements "flush pending launches
before growing" (see `[[device_pool_recording_rule]]`). `begin_pass`/
`end_pass` wrap prefill the same as decode, so `self.recording` is `true`
during the very prefill calls that hit this new path -- this isn't a new
scenario, it's the existing mechanism at a larger size. Every `model_check`
run above exercises the real `begin_pass`/pass-recording/`end_pass` flow (not
a bypass), hit `splits > 1`, and matched the host reference exactly, which is
the same empirical bar the existing decode-path split-K was held to.

## Register / occupancy / stall evidence

Freshly measured this session, `ncu.bat --metrics ...` plus one `--set full`
pass per kernel kind. GPU confirmed idle before profiling
(`nvidia-smi`: ~3% util, clocks at their idle floor before the warm-up
launches).

| kernel | shape | regs/thread | occupancy | notes |
| --- | --- | --- | --- | --- |
| `q8_qmma` (baseline) | any deep-tile shape | 238 | 12.5% (grid 12/20), 19.1% (grid 72) | matches diagnosis's remembered figures |
| `q8_qmma_split` | o_proj (grid 1,12,8) | 248 | 22.4% (light) / 21.3-23.1% (`--set full`) | |
| `q8_qmma_split` | down_proj (grid 1,12,8) | 248 | 23.4% (light) / 23.1% (`--set full`) | |
| `q8_qmma_split` | qkv (grid 1,20,4), pre-gate-fix | 248 | 19.7% (light) / 19.7-20.4% (`--set full`) | short of the 25% ceiling at `S=4`; this shape no longer routes through split-K, see "Gate fix" below |
| `q8_qmma_reduce` | all three | 26-27 | 89-96% | tiny kernel, expected |

`--set full` files: `ncu_qmma_splitk_full_oproj.ncu-rep`,
`ncu_qmma_splitk_full_downproj.ncu-rep`, `ncu_qmma_splitk_full_qkv.ncu-rep`,
`ncu_qmma_splitk_full_oproj_reduce.ncu-rep`.

**Stall breakdown -- an honest nuance, not the clean "long_scoreboard drops"
story.** The baseline's `long_scoreboard` ratio (measured fresh this
session) is 0.78-0.86 for grid=12/20 and 1.22-1.32 for grid=72, matching the
diagnosis. The *split* kernel's `long_scoreboard` ratio is **higher**, not
lower: 1.65-1.73 (o_proj/down_proj), 1.53-1.85 (qkv). This isn't the fix
failing -- `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` is an
average count of concurrently-stalled warps per issued instruction, and
adding resident warps mechanically raises it even when latency-hiding
improves: more warps now have outstanding memory requests at any given
instant, each individually still waiting on the same absolute DRAM latency,
which is a *higher* number on this metric even as the SM's scheduler has more
non-stalled warps to issue from and the kernel finishes faster. The metric
that actually says whether latency got hidden is duration (dropped 22-44% on
two of three shapes) combined with occupancy (rose 1.6-1.9x), not the raw
stall ratio. Reporting the number as measured rather than the story I
expected it to tell.

## Why qkv regressed, and the gate fix

The reduce pass's cost is dominated by moving the partial buffer:
`S * rows * n` floats read plus `rows * n` floats written. For qkv
(`rows=128, n=2560, S=4`, its split count *before* the fix below): 4*128*2560*4
+ 128*2560*4 bytes = ~6.55 MB, at this card's ~427 GB/s sustained bandwidth
that's ~15 us -- matches the measured 16.0 us almost exactly, i.e. the reduce
pass is bandwidth-bound on the partials traffic, not launch-latency-bound.
That cost is roughly fixed by `(rows, n, S)` regardless of how much the split
*saves*. qkv's own baseline duration (73.7 us) is the smallest of the three
routed shapes, and its `S=4` (`96/20` floored) only lifted occupancy to
19.7%, short of the 25% ceiling the `S=8` shapes get closer to -- so the
compute-time saved (73.7 to 61.0 us, -17%) didn't clear the ~16 us the reduce
pass added back. o_proj and down_proj won because their baseline durations
are large enough, and their `S=8` split large enough, that the same
roughly-fixed reduce cost is a smaller fraction of what's saved.

**Gate fix** (post-review): every shape that won -- o_proj, down_proj, and
both routed Qwen shapes -- reached the full `S = Q8_QMMA_SPLIT_MAX` (8). The
one that regressed, qkv, was the only one the halving-to-fit-K-blocks loop
couldn't lift past `S=4`. That's the same mechanism above, not a separate
coincidence: a smaller `S` buys proportionally less compute-time saving per
unit of the reduce pass's roughly-fixed cost. So `q8_qmma_splits` now only
returns a split when the halving loop lands exactly on `Q8_QMMA_SPLIT_MAX`;
anything short of the max falls back to 1 (unsplit) rather than route a
partial split. This is a gate fix grounded in the pattern already present in
the measured data above, not a new tuning pass against a single number, and
it touches only `q8_qmma_splits`'s return -- `Q8_QMMA_SPLIT_THRESHOLD` and
`Q8_QMMA_SPLIT_TARGET` are unchanged.

Re-verified after the fix: qkv now shows as a plain `q8_qmma` launch, grid
`(1, 20, 1)`, no `q8_qmma_split`/`q8_qmma_reduce` pair at all
(`ncu_qmma_splitk_gatefix.ncu-rep`/`.csv`), duration 71.8-76.1 us across three
observations -- matching its original 73.7 us baseline, i.e. truly unaffected
rather than merely improved. o_proj, down_proj and gate/up are unchanged in
the same capture. All four standard gates and both models' 128-token
`model_check` (`backends agree` on both) were re-run against the fixed gate
and still pass.

## End-to-end result and the graph-node mechanism

**This is the section that overrides "Result summary" above and decided the
default.** Everything before this point was measured with `ncu` isolating
individual kernels; none of it saw the graph they run inside.

The coordinator ran the end-to-end check I didn't (I don't run `bench.py`
myself, per the task's standing rules): `scripts/bench.py --no-llama` on
minicpm's `pp128`, isolated specifically to rule out cross-process/llama.cpp
contention, four separate rounds, consistent direction every time:

| `PHOBOS_QMMA_SPLIT` | pp128 t/s |
| --- | --- |
| `0` (off) | 5524 |
| `1` (on, default before this fix) | 4480-4774 |

A **14-19% end-to-end throughput loss**, not a gain. Two full `bench.py` runs
against llama.cpp confirm the same direction at the whole-comparison level:
the established pre-split ratio of 0.76x dropped to 0.61x and 0.54x with
split-K on.

The coordinator traced the mechanism with `nsys --trace=cuda` (both traces
captured and exported to sqlite: `nsys_qmma_split_on.nsys-rep`/`.sqlite`,
`nsys_qmma_split_off.nsys-rep`/`.sqlite`), and ruled out the two most likely
explanations before landing on the real one:

- **Not the kernels themselves.** `cuda_gpu_kern_sum`'s in-context durations
  roughly match the isolated `ncu` numbers above (e.g. `q8_qmma_reduce`
  averaged 18.7 us in-context vs 18.5-19.4 us isolated). Total raw GEMM
  kernel time is genuinely lower with split-K on in this trace, consistent
  with the per-shape savings measured above. The kernels are not lying, and
  are not the problem.
- **Not graph-rebuild frequency.** `cuGraphInstantiate_v2` fires exactly
  twice in both traces (`cuda_api_sum`), a pre-existing prefill/decode
  cache-eviction pattern in `graph.rs`'s single-slot `self.pass` cache,
  unrelated to split-K. ON's two instantiate calls do cost more on average
  (1.46 ms vs 1.06 ms -- presumably a bigger graph takes longer to
  instantiate) but that's at most ~0.8 ms total, nowhere near the gap.
- **What's left**: `launch_qmma_split` replaces one launch with two (write,
  then reduce) for every split-routed projection -- net +1 graph node each.
  With 2 routed shapes x however many layers, that's dozens of extra
  serialized nodes added to a graph built, per `graph.rs`'s own doc comment,
  as "a chain rather than a dependency analysis: these launches shared one
  stream, so serial order is the ordering they already relied on" -- every
  node has a hard dependency edge on the previous one, no parallelism
  between nodes. Per-kernel durations aren't inflated, but aggregate wall
  clock still grows, which points at inter-node dispatch/sync overhead on
  this platform (WDDM) scaling with node count rather than with kernel
  content. The coordinator was not able to isolate the exact per-node cost
  with the reports pulled so far (instance-count accounting in the
  graph-replay case didn't cleanly decompose across passes); further
  precision here has diminishing returns for the decision that actually
  matters, which is the default.

**This is a structural cost, not a bug with a cheap fix.** `qmma_split` now
defaults off; the code, the two compiler-proof bugs it took to get here, and
the correctness verification all stay, since the diagnosis and the mechanism
are real and the tradeoff could tip differently at a different node-count or
model shape mix -- Qwen's per-kernel wins (-37% to -47%) were larger than
minicpm's, which is the kind of thing worth another look if this comes up
again with a model whose split-routed projection count or shape differs
enough to change the graph-node-count-vs-kernel-savings balance.

## Files

- `ncu_qmma_splitk_baseline.ncu-rep` / `.csv` -- minicpm, `PHOBOS_QMMA_SPLIT=0`, all 4 shapes, light metrics, 12 launches (3 layers).
- `ncu_qmma_splitk_split.ncu-rep` / `.csv` -- minicpm, split-K on, all 4 shapes, light metrics, 21 launches (3 layers).
- `ncu_qmma_splitk_full_oproj.ncu-rep`, `_full_downproj.ncu-rep`, `_full_qkv.ncu-rep`, `_full_oproj_reduce.ncu-rep` -- `--set full` mechanism confirmation, one launch each (plus their `_check.csv` exports).
- `ncu_qmma_splitk_qwen_baseline.ncu-rep` / `.csv`, `ncu_qmma_splitk_qwen_split.ncu-rep` / `.csv` -- same light-metrics comparison on Qwen3.5-0.8B-Q8_0.
- `ncu_qmma_splitk_gatefix.ncu-rep` / `.csv` -- minicpm, post-gate-fix, confirms qkv now takes the plain unsplit `q8_qmma` path and o_proj/down_proj/gate-up are unaffected.
- `nsys_qmma_split_on.nsys-rep` / `.sqlite`, `nsys_qmma_split_off.nsys-rep` / `.sqlite` -- the coordinator's whole-pass traces (`nsys --trace=cuda`) that found the graph-node mechanism behind the end-to-end regression; see "End-to-end result and the graph-node mechanism" above.
- `qmma_splitk_off.csv`/`.json`, `qmma_splitk_on.csv`/`.json` -- the coordinator's phobos-only `bench.py --no-llama` A/B (minicpm, pp128) that first isolated the regression from cross-process noise: off 5524 t/s, on 4480-4774 t/s.
- `qmma_splitk_confirm.csv`/`.json`, `qmma_splitk_confirm2.csv`/`.json` -- the coordinator's two full `bench.py` runs against llama.cpp with split-K still on by default, before the fix: 0.61x, 0.54x against the established 0.76x baseline.
- `qmma_splitk_final_confirm.csv`/`.json` -- the coordinator's confirmation run after the default flip to off: pp128 5732.69 t/s, ratio 0.73x against llama.cpp, back in the neighborhood of the pre-split 0.76x baseline (within this session's observed clock-noise band on this card).

## Current state

- `qmma_split` defaults **off** (`PHOBOS_QMMA_SPLIT` is now opt-in:
  `1`/`on`/`yes`/`true`; anything else, including unset, stays off). All
  four standard gates and both models' 128-token `model_check` re-run and
  pass with the new default (the well-tested existing unsplit path, so
  unsurprising), and re-confirmed `PHOBOS_QMMA_SPLIT=1` still routes and
  still matches the host reference as an explicit opt-in.
- I don't run `scripts/bench.py` myself, per the task's standing rules --
  the end-to-end numbers in this file are the coordinator's, not mine.
- The gate fix in "Why qkv regressed, and the gate fix" above is grounded in
  the S=8-vs-S=4 pattern already present across all four measured shapes, not
  a new tuning pass fit to qkv's number alone. It's still in the code and
  still correct; it just no longer changes the default outcome, since
  nothing routes through split-K by default now.
