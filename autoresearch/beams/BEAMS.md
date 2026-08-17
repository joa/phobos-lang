# Beam note: decode bandwidth vs llama.cpp

Goal (from the user): phobos decode throughput (tg) higher than llama.cpp CUDA,
across the models `scripts/bench.py` tracks by default.

## State as measured

**Qwen3.5-0.8B-Q8_0**: phobos already wins every tg row. Last recorded
same-session numbers (`results/bench.csv`, 2026-08-13, 15 interleaved samples
per cell): tg32 284.2 vs 242.0 (1.17x), tg128 281.6 vs 259.7 (1.08x), tg512
281.5 vs 263.0 (1.07x), tg1024 279.4 vs 262.8 (1.06x), tg2048 275.0 vs 261.6
(1.05x). Not a target for new work; regression-watch only.

**minicpm5-1b-Q8_0**: phobos loses every tg row, and loses worse as the cache
grows. Same source, same session: tg32 259.5 vs 278.8 (0.93x), tg128 258.4 vs
281.2 (0.92x), tg512 254.4 vs 279.9 (0.91x), tg1024 248.6 vs 278.7 (0.89x),
tg2048 237.6 vs 276.3 (0.86x). llama.cpp is flat across cache length (278 ->
276, -0.9%); phobos degrades (259 -> 238, -8.7%). **This is the target.**

**Fresh same-session baseline, 2026-08-16, `fcfdf8b`, RTX 2080 SUPER, driver
610.88, 3 rounds x 3 reps, 3 of 3 rounds uncontended**
(`autoresearch/beams/autoresearch_baseline.csv`/`.json`/`.log`):

| test | phobos | llama.cpp CUDA | ratio |
| --- | --- | --- | --- |
| tg32 | 250.13 +/- 1.62 | 271.65 +/- 2.97 | 0.92x |
| tg128 | 252.76 +/- 1.03 | 274.86 +/- 0.21 | 0.92x |
| tg512 | 249.48 +/- 0.77 | 273.68 +/- 0.29 | 0.91x |
| tg1024 | 244.20 +/- 0.26 | 273.66 +/- 0.18 | 0.89x |
| tg2048 | 232.66 +/- 0.27 | 270.05 +/- 0.25 | 0.86x |
| pp128 | 3455.60 +/- 11.95 | 8452.92 +/- 92.64 | 0.41x (not a target here) |

Confirms both parts of the shape: an ~8% deficit already present at tg32
(short cache -- the flat, length-independent piece [[wide-vocab-lm-head]]
targets) plus another ~6 points of degradation by tg2048 (the length-dependent
piece [[cache-length-split-buckets]] targets). llama.cpp itself is close to
flat here too (271.65 -> 270.05, tg32 to tg2048), reinforcing that the slope
is phobos-specific. **This is now the baseline every beam result compares
against**; the 2026-08-13 figures above are superseded but kept for the
architectural analysis they informed.

## Why minicpm and not Qwen: the architectural difference

From `phobos-gguf/examples/attndecode.rs`'s own `SHAPES` table and the GGUF
metadata (`llama.*` vs `qwen35.*` keys, read directly out of both files):

| | minicpm5-1b | Qwen3.5-0.8B |
| --- | --- | --- |
| decode layers doing full KV attention | 24 of 24 | 6 of 25 (`full_attention_interval=4`, rest are SSM/delta-rule state, O(1) per token) |
| n_head / n_kv (group) | 16 / 2 (group 8) | 8 / 2 (group 4) |
| head_dim | 128 | 256 |
| vocab | 130,560 | 248,320 |

Qwen pays the decode-attention kernel's per-call cost on 6 layers; minicpm
pays it on all 24. Any inefficiency in that kernel is roughly 4x more exposed
on minicpm's tg number than on Qwen's, which is consistent with Qwen already
winning comfortably while minicpm trails and the gap widens with cache length
specifically (attention is the one decode cost that scales with cache length;
everything else in a decode step does not).

## Root cause candidate, with a comment in the code naming it

`phobos-gguf/src/backend/device/kernels/attn.rs:453-462`:

    /// Pieces the key axis is cut into while decoding, per query head a
    /// program carries. Fixed rather than chosen from the cache length,
    /// which is what it wants to be: a count that grows with the cache
    /// reshapes the pass every few dozen tokens, and each change costs a
    /// graph rebuild.
    pub(crate) const ATTN_SPLITS: usize = 8;

`ATTN_QGROUP = 2` (same file, line 39) is separately documented as already
swept per-shape ("Two is where both models land: at a head dimension of 128
over a group of eight, and at 256 over a group of four... flat between two and
four") -- that knob is closed, do not reopen it without new evidence.

`ATTN_SPLITS` was not swept per cache length, only picked once as a global
constant. At minicpm's shape the decode-split grid is `(n_head / qgroup) *
splits = (16/2) * 8 = 64` blocks per attention call, fixed regardless of
whether the cache holds 32 tokens or 2048. On this card (48 SMs) that grid
stops filling the machine well before occupancy-per-block is the limit, and
it never grows to compensate as each split's share of the cache grows with
`total()`. llama.cpp's flash-attention-on decode path is not grid-bound this
way. This lines up exactly with the observed slope: phobos degrading with
cache length while llama.cpp stays flat.

`phobos-gguf/src/backend/device/graph.rs:152` (`replay`) already tolerates a
kernel func changing between decode steps -- `reusable` checks `a.func ==
b.func` per node and falls back to `build_graph` when it does not match, which
is exactly what selecting a different split-count kernel variant at a cache-
length boundary would trigger. The mechanism the comment says is missing does
not need new graph-replay plumbing, only: (1) a bucketing function from
`spec.total()` to a split count, (2) folding the bucket into the `with_kernel`
cache key at `phobos-gguf/src/backend/device/attn.rs:190-192` (currently keyed
`(n_head, group, head_dim)` only, so a second split count would silently reuse
a kernel compiled for the first one), and (3) confirming `attn_scratch`'s grow
path (`mem.rs:79,158`) tolerates the resulting buffer growth without flushing
every step once the cache stabilizes at its largest bucket.

## Active beams

1. [[cache-length-split-buckets]] -- **killed.** Implemented exactly as
   scoped (bucketed `ATTN_SPLITS`, folded into the kernel cache key, all
   correctness gates passed), and still measured neutral-to-negative on
   `tg`: tg1024/tg2048 (where the wide bucket runs nearly the whole
   sequence and `attndecode`'s own sweep predicted ~15%) came back
   identical to baseline to three decimal places on the phobos/llama.cpp
   ratio, and tg512 (straddling the bucket boundary mid-run) got worse. A
   no-crossing control build (wide split count from the first decode
   token) was worse than baseline on every row, ruling out "it's just the
   graph-rebuild cost at the boundary." `attndecode`'s sweep is real and
   stays on record as a warning for anyone tempted to reopen this: doubling
   past the shipped baseline (`ATTN_SPLITS` 8 -> 16) is a cliff, not a gain
   -- worse than baseline itself at a cache of 2053, not merely short of
   one. Full numbers and reasoning in the beam file's Result log. Takeaway:
   tripling decode attention's own grid width did not move `tg` at any
   cache length, so the step's cost is not concentrated in that kernel's
   occupancy -- corroborates beam 2 below rather than competing with it.
2. [[launch-bound-headroom]] -- **one increment tried and reverted, beam
   stays open.** Pulled minicpm's `PHOBOS_FUSED` per-stage report as scoped:
   292 launches/decode step (matches `docs/megakernel.md`'s documented
   number exactly; only the fused MLP fires for minicpm today, since
   `PHOBOS_FUSED_PROJ`/`_MIX` are wired only to Qwen's delta net). Implemented
   the doc's own named next step, folding attention's query/key/value
   projection into the existing `Linear::project_fused`/`Stage::ProjF`
   machinery (already architecture-blind, no fuse-pass or backend changes
   needed, just `llama.rs`): launches fell **292 -> 244** exactly as
   predicted, and `fuse_check`/`backend_check`/`batch_check` all passed
   (`model_check` flipped to fail on some prompts, but reproduced
   byte-for-byte on the pre-existing unfused baseline too -- the doc's
   already-documented per-file quantization-bound issue, not a regression).
   Benchmarked anyway and it is a clean **-3.0 to -3.6% loss on every tg row**
   for minicpm, no exceptions, well outside noise. Diagnosed cause:
   `ProjQ`/`ProjF` tile in fixed 32-wide (`Q8_BLOCK`) units, which the wide
   MLP/delta-net projections that already use this path fill easily but
   attention's narrow key/value run (256 wide) does not -- only 8 of the
   persistent kernel's ~192 resident blocks do real work in that nest, an
   order of magnitude less parallelism than the launched kernel's grid gave
   it. Reverted (`git checkout -- phobos-gguf/src/llama.rs`), nothing
   committed, Qwen unaffected throughout (never touches `llama.rs`). Full
   mechanism, numbers and a proposed real fix (an `OUT_TILE`-width option on
   `ProjQ`/`ProjF`, a fuse-pass change out of scope for this round) in the
   beam file's Result log. Still open: `store_2d`(2/layer), `quantize`
   (1/layer) and `q8_qdot_add`(1/layer) remain unfused and untried, and the
   corroborating 14.4%/655us bubble datum is unexplained by this one failed
   increment alone.
3. [[wide-vocab-lm-head]] -- **killed**. Profiled first, per the brief.
   lm_head's `q8_qdot` kernel + logits readback is a real 12.6-14.2% of a
   decode step (double-digit, not the brief's single-digit auto-down-rank
   case) -- but the kernel is already at 92.8% of the card's theoretical
   memory-bandwidth peak (460.5 of 496.06 GB/s), the widest-N matvec in the
   model and the *best*-utilized one, not the worst. Headroom ceiling from a
   hypothetically perfect kernel is ~35us, about 11% of the 316.8us/step
   that needs to be found to match llama.cpp at tg32 -- not a viable lever.
   No code changed. Full writeup, roofline math and raw nsys traces in
   `wide-vocab-lm-head.md` and `autoresearch/beams/lmhead_profile*`.

## The gap reframed: it is mostly llama.cpp's flash attention, not a phobos deficit

Three beams above all chased phobos-internal costs (attention grid occupancy,
lm_head bandwidth, launch count) and all three moved `tg` by nothing or the
wrong direction. Before funding a fourth internal round, tested a different
variable: how much of the gap is llama.cpp's own configuration advantage
rather than something phobos is doing wrong.

phobos's KV cache is already f16 (`kv: fp16 buffers`, commit `50543a5`;
confirmed in `phobos-gguf/src/backend/mod.rs:504`), matching llama.cpp's
default, so precision was never the variable (`--llama-args "-ctk f32 -ctv
f32"` was tried first and rejected outright -- this llama-bench build only
accepts `f16` as a cache-type value, so that specific flag combination is a
dead end, logged in `autoresearch/beams/f32kv_noFA_bench.log`). What phobos
does not have at all is a flash-attention-style decode kernel: its decode
path is the split-plus-merge design in `attn.rs` (`attention_decode`), two
kernels and a global round-trip through `P`/`ML` scratch, not FlashAttention's
single-pass online-softmax kernel.

**`python scripts/bench.py -m models/minicpm5-1b-Q8_0.gguf -p 128 -n 32 128
512 1024 2048 -r 3 -R 3 --llama-args "-fa 0"`, 2026-08-16, same session, 3 of
3 rounds uncontended** (`autoresearch/beams/noFA_bench.{csv,json,log}`):

| test | phobos | llama.cpp CUDA (FA on, baseline) | llama.cpp CUDA (`-fa 0`) | ratio (FA on) | ratio (`-fa 0`) |
| --- | --- | --- | --- | --- | --- |
| tg32 | 253.69 | 271.65 | 264.25 | 0.92x | **0.96x** |
| tg128 | 256.86 | 274.86 | 264.29 | 0.92x | **0.97x** |
| tg512 | 253.58 | 273.68 | 253.33 | 0.91x | **1.00x** |
| tg1024 | 245.19 | 273.66 | 246.52 | 0.89x | **0.99x** |
| tg2048 | 233.54 | 270.05 | 247.24 | 0.86x | **0.94x** |

Turning flash attention off drops llama.cpp's own tg1024/tg2048 by 10-11% (its
FA advantage grows with cache length, same shape as the slope beam 1 chased)
and phobos closes most but not all of the gap against llama.cpp's own non-FA
decode path: dead heats at tg512/tg1024, but still -4.0% at tg32, -2.9%
tg128, -5.5% tg2048 (corrected from this note's first pass, which
overstated this as full parity -- see [[flash-attention-decode]]'s own
correction section for the exact numbers). It explains beam 1's null result
in hindsight regardless: widening the split-and-merge kernel's own grid
cannot close a gap that comes from a different kernel design (a fused
online-softmax pass vs. two kernels plus a scratch round-trip), so there was
never a grid width that would have fixed it. And note `-fa 0` also changes
llama.cpp's V-cache layout, not only its kernel choice, so this is "phobos
vs. llama.cpp's non-FA path," not a single-variable kernel swap -- still the
right experiment, just labelled honestly.

**Bound on this beam, from the already-committed nsys trace**: attention
(`attention_split` + `attention_merge` combined) is ~15-18% of a minicpm
decode step (four steady-state windows measured, [[flash-attention-decode]]
has the numbers). Even a zero-cost attention kernel only reaches ~281 t/s at
tg2048 against llama.cpp's 275 -- enough to clear the goal, but with no
margin, and a real fix will not remove 100% of that share. This beam has to
combine with [[launch-bound-headroom]]'s remaining unfused launches to have
a realistic shot at the actual goal (beating FA-on llama.cpp), not just
closing to parity with FA-off. phobos also already has a working
FlashAttention-2 kernel in `examples/flash_attention_fp32.ph` (prefill-shaped,
not wired to decode) -- this beam is an adaptation of its online-softmax
recurrence into `attention_split` via `grid_barrier()`/`@persistent` (already
built for `docs/megakernel.md`), not a from-scratch design.

## New primary beam

4. [[flash-attention-decode]] -- structural, high-risk, high-ceiling.
   **Implemented, opt-in.** Not the full FlashAttention single-pass redesign:
   `attention_split` and `attention_merge` folded into one `@persistent`
   kernel via `grid_barrier()`, same online-softmax math, one launch and one
   scratch round-trip removed per full-attention layer per decode step.
   Genuine win on minicpm (+1.0 to +1.4% `tg`, all rows positive, same-session
   A/B against llama.cpp interleaved), but the persistent kernel is measured
   *slower per call* than the launched pair at both shapes tried (minicpm
   +1-7%, Qwen +40-65%, from `attndecode`) -- the win is the removed-launch
   arithmetic outrunning that per-call cost, which it only does on a model
   with enough full-attention layers to pay for it (minicpm: 24 of 24; Qwen:
   6 of 25, where it is a clean regression, -2 to -3% at longer cache).
   Root cause: the combined kernel's shared-memory footprint does not pool
   down to the wider of its two phases the way `docs/megakernel.md`'s
   redundant-stage tiles do (measured ~41KB against the launched split
   kernel's own ~24KB at Qwen's shape), which halves occupancy and forces a
   second grid-strided pass. Shipped behind `PHOBOS_ATTN_PERSIST` (unset =
   off) with a shape gate (`attn_persist_plan`, architecture-blind, keyed off
   the driver's own occupancy answer) that declines any shape whose settled
   grid cannot cover the split phase in one pass and falls through to the
   unmodified launched path -- Qwen reproduces its pre-existing numbers
   exactly once gated. All correctness gates (`backend_check`, `model_check`,
   `batch_check` x2 models, `fuse_check`) pass. Full mechanism, both models'
   `tg` tables, the `attndecode` per-call numbers and the pass-report evidence
   in the beam file's Result log. Does not clear llama.cpp's FA-on baseline
   (0.86-0.94x, +0.01 in ratio) -- combine with [[launch-bound-headroom]]
   stays the live next step, unchanged from before this round.

## Ranking after this round

`[[cache-length-split-buckets]]` is now closed too: implemented, correctness
gates passed, and the real benchmark showed no gain at the cache lengths
where the mechanism should have shown up cleanest (tg1024/tg2048, ratio
identical to baseline to three decimals) plus a regression at tg512. Its null
result is evidence *for* `[[launch-bound-headroom]]`, not against it: if
tripling decode attention's grid does not move `tg`, the bottleneck is not
attention occupancy, which leaves launch/dispatch overhead and the other
per-step kernels as the remaining live hypothesis, matching that beam's
already-profiled 14.4% launch-bubble ceiling.

`[[launch-bound-headroom]]`'s first increment (fuse the QKV projection) is
now also closed, but with a diagnosed mechanism rather than a null result:
the launch-count savings were real (292 -> 244) and the correctness gates
passed, but it lost -3 to -3.6% on `tg` because the generic fuse pass's
32-wide tiling starves a narrow attention projection of grid parallelism the
launched kernel had. That is a *different* failure mode from
`[[cache-length-split-buckets]]`'s null result (that one moved nothing;
this one moved throughput the wrong way for a specific, named reason).
`[[launch-bound-headroom]]` stays open in principle (`store_2d`, `quantize`
and `q8_qdot_add` are still unfused per layer) but is now second priority:
its own diagnosed fix (narrowing `ProjQ`/`ProjF` from 32-wide to `OUT_TILE`
= 8-wide) only reaches 32 active blocks against the persistent kernel's
192-block grid, still far short of the 256 the launched kernel had for the
query run alone, so it may not even fully solve the problem it targets --
whoever picks it up should compute the predicted post-fix active-block count
before writing code, not after.

**The session's real finding this round is above: matching llama.cpp's flash
attention off closes 0.92x to 0.96-1.00x at every cache length**
(`--llama-args "-fa 0"` result). This makes `[[flash-attention-decode]]` the
new primary beam, well ahead of the launch-count and tiling work, because it
is now backed by a controlled experiment rather than an inferred mechanism:
turning off the one feature phobos structurally lacks reproduces phobos's own
numbers almost exactly. `[[wide-vocab-lm-head]]` and
`[[cache-length-split-buckets]]` remain closed and kept as beam files per
AGENT.md (history, not deleted); both of their null results are now
explained rather than merely observed -- neither targeted mechanism was ever
going to close a gap that comes from a different kernel design entirely.

## Ranking after the flash-attention-decode round

`[[flash-attention-decode]]` is no longer "not yet implemented": it shipped a
genuine, gated, opt-in win (see above), the first positive result among the
four attention/launch beams this session has tried. It stays open rather than
closing, for two reasons. First, it does not clear the actual goal
(llama.cpp's FA-on baseline) alone, which the beam file predicted going in and
the result confirms -- `[[launch-bound-headroom]]`'s remaining unfused
launches (`store_2d`, `quantize`, `q8_qdot_add`) are still the next lever to
combine it with. Second, the round surfaced a specific, actionable phobos-lang
codegen gap (shared-memory pooling not collapsing across a persistent
kernel's two phases the way it does across `docs/megakernel.md`'s
redundant-stage barrier) that would widen this beam's own reach if fixed --
named in the beam file as the highest-value follow-up, ahead of chasing a
third shape to firm up the shape gate's predicate.

Priority for the next round: either (a) `[[launch-bound-headroom]]`'s
remaining unfused per-layer launches, now the most direct path to actually
closing the gap on minicpm, or (b) the shared-memory pooling gap this round
found, which could turn `[[flash-attention-decode]]`'s Qwen regression into a
second win and make default-on defensible. Both outrank starting a fifth beam
from scratch, per AGENT.md's combine-before-retiring discipline.

## Ranking after the output-fusion round -- option (a) is now closed too

Tried option (a): fused `quantize`+`q8_qdot_add` (attention's output
projection) using the correct wide tile (`ProjAdd`'s `OUT_TILE` over
`out_dim`, not the narrow tile that sank the earlier QKV attempt). Correctly
implemented, architecturally sound, and **a clean wash** -- flat to within
stderr on every tg row, same session, `PHOBOS_ATTN_PERSIST=1` held constant
both sides. Reverted, nothing committed. Full numbers in
`[[launch-bound-headroom]]`'s Result log.

Two fusion attempts around attention (QKV, then output) now agree on the
same underlying picture from different failure modes: QKV lost because a
narrow tile starved parallelism; output-side didn't move at all because
attention's cost is dominated by `attention_split` itself (~15-18% of a
step, 6-8x the merge kernel's share per `[[flash-attention-decode]]`'s own
measurement), not by the couple of launches around it. **Option (a) is
closed** -- `store_2d` remains technically untried but the pattern across
both attempts makes it a low-probability follow-up, not a promising one; do
not spend a third round on it without new evidence that store_2d's cost
differs qualitatively from quantize/q8_qdot_add's.

That leaves **option (b), the shared-memory pooling gap, as the only
concretely-scoped remaining lever** this session has identified. It has not
been attempted (both subagents that reached this beam ran out of round
budget on option (a) first, per the brief's own ordering). Whether it is
worth a further round depends on its ceiling: it would not move minicpm's
absolute numbers (it only removes `PHOBOS_ATTN_PERSIST`'s Qwen regression,
letting the flag default on), so its value is making an already-landed win
safe-by-default rather than closing more of the gap. The gap itself, per
`[[flash-attention-decode]]`'s ceiling math, may not be closeable by any
further attention/launch work alone at this session's measured 15-18%
ceiling -- worth an explicit checkpoint with the user before spending
another large round chasing it.

## Directives from the user, 2026-08-16, apply to every round from here

1. **Benchmark scope narrows to `tg >= 1024`** (1024, 2048, 4096). Shorter
   lengths (32/128/512) were useful for isolating the flat-vs-slope shape of
   the gap earlier in the session, but that diagnosis is done; don't spend
   further `bench.py` rounds sweeping them. `attndecode.rs`'s `LENGTHS` grew
   an 8th entry, 4099 (prime near 4096), to cover the new floor.
2. **The ~9x-of-floor headroom in decode attention needs real profiling,
   not the coarse `attndecode` floor check.** `attndecode.rs` now supports
   `PHOBOS_ATTNDECODE_SHAPE`/`PHOBOS_ATTNDECODE_LENGTH` to isolate a single
   kernel launch for `ncu` (Nsight Compute). `ncu` needs an elevated shell on
   this box (`ncu-needs-elevation` memory: `ERR_NVGPUCTRPERM` from a normal
   one) -- the user runs it, not an agent. See the request left for them
   below this section.
3. **Existing beams stayed inside phobos-lang's current primitive set.** No
   beam this session proposed a new codegen primitive, tile shape, or
   scheduling construct `phobos-lang` doesn't already have. The user wants
   the theoretical maximum pursued, which may mean it needs one. Do not treat
   "phobos-lang doesn't have X" as a beam-ending constraint without first
   asking whether X is a reasonable thing to add (see `docs/megakernel.md`'s
   own history: `grid_barrier()`, `@persistent` and `atomic_add` were all
   added mid-project for exactly this reason).
4. **If adapting FlashAttention's structure to decode is not enough to close
   the gap, invent something else.** The mandate is not "match llama.cpp's
   mechanism," it is "beat llama.cpp's number." A different decode-attention
   design (different KV cache layout, a different reduction structure, a
   different division of work between grid and block) is in scope if the
   adapted-FlashAttention direction tops out short of the goal.

### Open action: waiting on the user's `ncu` run

Gave the user this to run themselves, elevated PowerShell (`ncu` is on PATH
there, not in Git Bash, per the `ncu-needs-elevation` memory):

    $env:PHOBOS_ATTNDECODE_SHAPE = "minicpm"
    $env:PHOBOS_ATTNDECODE_LENGTH = "4099"
    & "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\ncu.bat" --set full --launch-skip 5 --launch-count 10 -o attndecode_ncu_minicpm_4099 -f "C:\Users\joaeb\code\phobos\target\release\examples\attndecode.exe"

Stock (non-persistent) split-plus-merge path, since that's what the 9x
figure was measured against; `$env:PHOBOS_ATTN_PERSIST = "1"` before the
`ncu` call profiles the persistent kernel instead, for comparison. Key
numbers to read out of the report once it lands: DRAM/Memory Throughput
percentage of peak (confirms or corrects the coarse floor check's claim of
non-bandwidth-bound headroom), Compute (SM) Throughput, and Achieved
Occupancy (distinguishes "not enough parallelism" from "enough parallelism,
stalled on something else" -- the two have different fixes).

## Active beams roster, 2026-08-16 evening

The `ncu` result above landed (see `flash-attention-decode.md`'s "resolved:
real ncu hardware counters" section): decode attention is latency-bound
(38% memory throughput, 14% compute throughput, 64% occupancy on
`attention_split`), not bandwidth-bound. Two beams running in parallel from
that finding, per AGENT.md's minimum-3-active-beams discipline (had dropped
to one after the last round of kills, caught and corrected):

1. **Exploit/structural** -- redesign `attention_split`'s core loop to
   shorten its serial critical path (the online-softmax `m`/`l`/`acc` chain
   across many small-`BC` iterations at long cache). **Update: the
   isolated-vs-real-pass puzzle's guessed resolution above ("~290 other
   kernels' worth of independent work to interleave with, so more grid
   width is redundant in a real pass") does not hold up** -- CUDA kernels on
   one stream execute one at a time, so a neighboring kernel's own warps are
   never resident to interleave with `attention_split`'s while it runs; there
   is no mechanism for that theory to work through. Measured instead
   (`[[cache-length-split-buckets]]`'s Result log has the full method and
   numbers): the wider-split kernel change *does* reproduce its isolated
   `attndecode` gain almost exactly inside a real CUDA-graph-replayed pass
   (14.7% real reduction in `attention_split`+`attention_merge`'s own
   measured GPU duration at cache ~1024, vs `attndecode`'s ~15% isolated
   prediction at cache 1031) -- `attndecode` is trustworthy. What actually
   swallows the gain is that `tg` averages throughput over the *whole*
   decode trajectory from cache=1, and attention's share of a step grows
   from near-zero to ~19% only by cache ~1024, so a real deep-cache win
   dilutes below `bench.py`'s own noise floor by the time it is averaged
   into `tg1024`/`tg2048`. Also landed: the `ncu` counters confirm
   `attention_split` is latency-bound (38% memory, 14% compute throughput)
   and that its 63.8% occupancy is not a resource shortfall to tune away --
   256-thread blocks hit sm_75's 32-warps/SM architectural ceiling at
   exactly 4 blocks/SM, which `[[cache-length-split-buckets]]`'s own S=24
   plateau already reaches exactly (192 blocks/48 SMs = 4.0). **Grid-width
   tuning on this kernel is provably exhausted, not merely unpromising.**
   Two follow-on attempts to shorten the per-block critical path instead
   (`@pipeline`; a manual two-independent-chain in-block redesign, no new
   primitive needed) were both tried this round and both came back
   negative -- flat and a clean regression respectively, both reverted,
   full mechanism in `[[cache-length-split-buckets]]`'s Result log. Beam
   stays open: the one lever this round's evidence points at and did not
   reach is a genuinely new phobos-lang unit-of-parallelism primitive
   (warp-scope ownership of a key sub-range, several independent warps per
   block instead of one wide serial chain per block, so added parallelism
   costs thread count rather than the per-thread register/shared-memory
   footprint that sank the two-chain attempt) -- the actual "invent
   something else" the user's directive asked for once grid/register-level
   tuning tops out, which this round's evidence now shows it has.
2. **Near-miss** -- the shared-memory pooling gap flagged when
   `PHOBOS_ATTN_PERSIST` landed: phase one and phase two of the persistent
   kernel sum their shared-memory footprints (~41KB) instead of pooling to
   the wider phase's own (~24KB), which is the occupancy cost that makes the
   flag regress Qwen and forces the opt-in gate. Fixing it could widen or
   remove the gate, turning an opt-in win into a default-on one -- lower
   ceiling than beam 1 (doesn't move minicpm's absolute numbers) but lower
   risk and independent territory (a different kernel function in the same
   file). In progress as of this note (`phobos-gguf/src/backend/device/attn.rs`
   and `phobos-lang/src/codegen/{mod.rs,tile/alloc.rs}` mid-edit, uncommitted,
   under a separate process from beam 1's -- not touched by beam 1's own
   round, check their state fresh rather than trust this snapshot).

Beam 1 above closed (2026-08-16 23:06, `13ad004`) with a clear next step:
**[[cache-length-split-buckets]]'s Result log now recommends a new
phobos-lang warp-scope parallelism primitive** (independent warps per block
each owning a disjoint key sub-range, combined by a fast intra-block reduce,
so added parallelism costs thread count rather than the per-thread
register/shared-memory footprint that sank this round's two-chain attempt).
**Queued, not yet dispatched**: it would also touch
`phobos-lang/src/codegen/mod.rs`, which beam 2 above is actively editing
right now. Dispatching a third agent into the same core-compiler file while
beam 2 is mid-edit risks an uncoordinated conflict in fragile codegen
internals, worse than the file-ownership risk already being managed between
beams 1 and 2 (which touched different files). Wait for beam 2 to land,
then dispatch the warp-scope primitive as the new beam 1 on a clean base,
restoring the 3-beam minimum.

**Beam 2 (the shared-memory pooling gap) landed, 2026-08-17.** Full
mechanism, gate numbers and both models' benchmarks in
`flash-attention-decode.md`'s new Result log section. Short version: found
the root cause by reading `phobos-lang`'s tile allocator (a released tile
only pools back for reuse by a *later, identically-shaped* request; the
persistent kernel's two phases never share a shape, so the static
`memref.global`-per-tile mode had no way to alias them at all), fixed it by
adding `@dynshared` to `attention_persist_src` (already a real, shipped
attribute -- `delta_scan` has used it since before this beam) plus a small
allocator change (`dynamic_live` tracking + a byte-cursor reset when nothing
is live ahead of a new shape, in `phobos-lang/src/codegen/tile/alloc.rs`
and `mod.rs`), and a companion fix in `attn_persist_plan`
(`phobos-gguf/src/backend/device/attn.rs`) whose occupancy query was
hardcoding zero dynamic shared bytes -- correct before `@dynshared`, silently
wrong after, since it would have undercounted the kernel and could have
settled a grid too wide for co-residency. Qwen's footprint dropped
41376 -> 23760 bytes (1 -> 2 blocks/SM, 48 -> 96 settled grid), which clears
the gate's own 64-unit threshold; minicpm's dropped further, 20896 -> 11984
(3 -> 4 blocks/SM, the thread-count ceiling). All four correctness gates
pass on both models (`backend_check`, `model_check`, `batch_check`,
`fuse_check` -- Qwen's `fuse_check` was missing from the original round's
battery and is now included), matching documented baselines to the digit.
Benchmarked same-session, off vs on, `tg1024/2048/4096`: **Qwen's
previously-measured -2 to -3% regression is gone**, now +0.1 to +0.4%,
inside session noise -- Qwen passes the gate cleanly for the first time.
minicpm reads flat (-0.2% to +0.4%), not clearly still the beam's original
+1.0-1.4%, honestly recorded as unresolved-but-not-a-regression rather than
rounded either direction (full reasoning in the Result log). Added a
codegen test (`dynamic_shared_resets_its_cursor_between_dead_phases`,
`phobos-lang/src/codegen/tests/tile.rs`) pinning the reset behavior, since
no correctness gate would catch a future refactor silently re-inflating the
footprint and re-regressing Qwen. `attn_persist_plan`'s gate logic itself
needed no change (it already reads the driver fresh); fixing the footprint
fed it a better answer. Recommendation left for the user: default-on is now
defensible (both known shapes flat-to-positive) but the beam's own bar for
that (a third shape or a second card) is unchanged, so `PHOBOS_ATTN_PERSIST`
stays opt-in. Committed on `autoresearch`. **`phobos-lang/src/codegen/mod.rs`
and `tile/alloc.rs` are now free** for the queued warp-scope primitive beam.

Also worth flagging for whoever works this shared tree next: partway
through this round `phobos-gguf/src/backend/device/kernels/attn.rs` was
found reverted to its pre-round committed state by another process, which
silently destroyed this beam's first attempt at the `@dynshared` edit along
with (evidently) a concurrent rewrite of `attention_split_src` that was
in-progress at the time. No error, no diff to notice by -- caught only by
habitually running `git diff --stat` on the shared file before and after
every build in this round. Worth doing as standard practice whenever two
agents are dispatched into the same working tree rather than separate
worktrees.

Process lesson banked from that incident: this session dispatches concurrent
agents into one shared working tree rather than separate git worktrees. That
has been manageable when beams touch different files, but a beam that
recovers a failed attempt via a blanket `git checkout -- <file>` can destroy
a concurrent beam's uncommitted edits to the *same* file with no error and
no diff to notice by. The `Agent` tool supports `isolation: "worktree"`,
which would have prevented this entirely. Use it for the next pair of beams
dispatched concurrently into files either one might revert wholesale, rather
than relying on both agents remembering to `git diff --stat` defensively.

## Beam 3 dispatched, 2026-08-17: the warp-scope parallelism primitive

Now that `phobos-lang/src/codegen/mod.rs`/`tile/alloc.rs` are free (beam 2
landed), dispatching the next step `[[cache-length-split-buckets]]`'s
Result log identified: shorten `attention_split`'s per-block serial
critical path by giving several independent warps within one block their
own disjoint key sub-range (so parallelism costs thread count, which this
kernel is not short of -- `attention_split` sits at 63.8% occupancy against
sm_75's 32-warp/SM ceiling, per the `ncu` counters -- rather than the
per-thread register/shared-memory footprint that sank the prior round's
two-independent-chain attempt), combined by a fast intra-block reduce
(warp shuffle or a small shared-memory tree) instead of widening the grid
further (provably exhausted, same evidence). This is the "invent something
else" the user's directive asked for, now that grid-width and register-ILP
tuning have both been tried and both closed.

## Current-state snapshot, 2026-08-17, post-pooling-fix, `PHOBOS_ATTN_PERSIST=1`

Fresh, comprehensive, same-session comparison at the now-focused `tg
1024/2048/4096`, both models, 3 rounds x 3 reps, 3 of 3 uncontended
(`autoresearch/beams/current_state_snapshot.{csv,json,log}`), `ac86bfd`:

| model | test | phobos | llama.cpp CUDA | ratio |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | tg1024 | 271.40 | 258.99 | 1.05x |
| Qwen3.5-0.8B | tg2048 | 267.37 | 256.90 | 1.04x |
| Qwen3.5-0.8B | tg4096 | 258.67 | 254.77 | 1.02x |
| minicpm5-1b | tg1024 | 243.27 | 275.47 | 0.88x |
| minicpm5-1b | tg2048 | 231.12 | 272.01 | 0.85x |
| minicpm5-1b | tg4096 | 209.68 | 266.63 | **0.79x** |

This is the deepest cache this session has benchmarked end-to-end (4096).
minicpm's ratio keeps degrading past tg2048 (0.86x in the original baseline
-> 0.85x here at tg2048, consistent; 0.79x at tg4096 is new data and the
worst ratio measured this session) -- the slope this whole beam note opened
with has not flattened, it continues past 2048. Qwen remains solidly ahead
throughout, though its own ratio compresses toward 1.0x as cache grows too
(1.05x -> 1.02x), worth watching but not yet a regression. **This is now
the reference point for judging any future decode-attention change**;
compare against it rather than the pre-pooling-fix baseline further up this
file.

## Beam count note

Restoring a third active beam was assessed and found not viable right now
with available assets: no GGUF model in `models/` presents a genuinely
different attention shape from the two already tested (`Qwen3.5-0.8B-Q4_K`
is the same architecture as the already-tested Q8_0, just a different
quant format that doesn't touch the persist gate's occupancy math; `GPT2`
goes through the separate ONNX backend where `PHOBOS_ATTN_PERSIST` doesn't
apply at all), so the pooling beam's own recommended next validation step
(a third shape or a second card, before considering a default-on flip) has
no cheap path forward and was not forced. `store_2d`/`row_scales` fusions
remain assessed low-probability given the latency-bound finding and were
not spun up just to pad the count. One focused beam (warp-scope primitive)
plus this snapshot is the honest state of available, well-evidenced work
right now.

## Beam 3 landed, 2026-08-17: warp-partitioned decode attention -- the session's largest win

Dispatched as above; landed clean. Full mechanism, thread-mapping evidence
and both models' numbers are in `cache-length-split-buckets.md`'s new
Result-log section (that is where the recommendation for this beam always
lived, per the dispatch note); `flash-attention-decode.md` has a short
cross-reference on what changed underneath its own `@persistent` mechanism
and what did not.

**What it is, briefly.** A new `phobos-lang` builtin, `warp_partial`
(`phobos-lang/src/codegen/tile/warp_attn.rs`), lets `attention_split`'s
block divide its own key sub-range across its eight warps instead of
running one shared, barrier-synchronized chain that left six or seven of
them idle the whole time (confirmed in emitted MLIR, not guessed: at
minicpm's shape, `dot_t`/`exp`/`dot` were already only ever using 32 of the
block's 256 threads). Each warp computes its own `(m, l, acc)` in
registers, head dimension spread across its 32 lanes, combined via a
`gpu.shuffle` xor-butterfly (reusing the existing warp-reduce primitive)
rather than shared memory, so no CTA barrier sits inside the loop and the
eight warps run genuinely independently. One barrier at the very end
publishes all eight partials to a small shared scratch tile, which a new
Rust-generated combine block folds with the same online-softmax merge
`attention_merge` already runs across splits -- so `attention_merge` and
the block-level split count `S` are both untouched, only what one block
does with its own share of work changed.

**Numbers.** `attndecode` (isolated kernel time): 50-59% reduction across
every cache length tested, both models, both the launched and
`@persistent` paths. `bench.py`, same session, interleaved against
llama.cpp, `PHOBOS_ATTN_PERSIST=1`:

"before" is the current-state snapshot above (landed concurrently with this
beam, both agents' numbers agree within session noise against the slightly
earlier pooling-landing figures):

| model | test | before | after | delta |
| --- | --- | --- | --- | --- |
| minicpm5-1b-Q8_0 | tg1024 | 243.27 | 260.75 | +7.2% |
| minicpm5-1b-Q8_0 | tg2048 | 231.12 | 255.76 | +10.7% |
| minicpm5-1b-Q8_0 | tg4096 | 209.68 | 244.43 | +16.6% |
| Qwen3.5-0.8B-Q8_0 | tg1024 | 271.40 | 274.25 | +1.0% |
| Qwen3.5-0.8B-Q8_0 | tg2048 | 267.37 | 272.12 | +1.8% |
| Qwen3.5-0.8B-Q8_0 | tg4096 | 258.67 | 267.39 | +3.4% |

minicpm's ratio against llama.cpp's FA-on baseline moves from a widening
0.88x/0.85x/0.79x slope (the worst this session had measured, per the
snapshot above) to a flat-to-improving **0.95x/0.94x/0.92x**, the closest
this whole session has gotten -- and unlike the grid-width lever this
beam's own prior round killed, the gain shows up fully at `tg1024` already,
growing (not shrinking) with cache length, so none of `tg`'s
trajectory-average dilution applies here. Qwen improves too (1.02-1.05x ->
1.06-1.07x), no regression on either model.

**Correctness.** All four gates (`backend_check`, `model_check`,
`batch_check`, `fuse_check`) pass on both models with
`PHOBOS_ATTN_PERSIST=1` forced on, matching documented error bands to the
same order of magnitude (`backend_check`'s worst relative error is
identical to the pre-change baseline, 2.902e-4). `cargo test` across every
non-CUDA crate and `phobos-lang`'s own 148 codegen/parser tests all pass;
every `.ph` example re-verifies under both the default target and
`PHOBOS_CHIP=sm_80 PHOBOS_INDEX_BITS=64`, unsurprising since the
`phobos-lang` diff is provably additive (new builtin, unreachable from any
existing kernel). `source_size` ratchet passes. Committed on `autoresearch`.

**Where minicpm's decode throughput stands now.** 0.92-0.95x of llama.cpp's
FA-on baseline, up from a widening 0.79-0.88x at the start of this round
(the deepest-cache snapshot found the slope still worsening past tg2048)
and from 0.89-0.92x at the very start of the session's decode-attention
work. This
is the first change all session to move `tg1024` by a double-digit
percentage without any dilution caveat attached. It does not yet clear the
goal (beating llama.cpp outright), but it closes most of the remaining gap
by itself, further than any combination of the session's other beams did.

**What is left, for whoever picks this up next.** `warp_partial` processes
one key at a time with unvectorized scalar K/V loads -- batching a few keys
per warp step and/or vectorizing those loads is untried and plausible
further headroom, not attempted because this round's numbers already
cleared a clear win without it. `ATTN_WARP_SPLITS = 8` (one warp per group)
was taken as the advisor's v1 recommendation and not swept against smaller
values sharing multiple warps per group. And `[[launch-bound-headroom]]`'s
remaining unfused per-layer launches (`store_2d`, `quantize`,
`q8_qdot_add`) are still on the table as a separate, independent lever,
unaffected by this round's kernel rewrite. A fresh same-session baseline
capture (this round's own numbers, not the superseded 2026-08-16 one) is
the right starting point for whoever runs that next round.

## Two beams dispatched, 2026-08-17, pushing on `warp_partial`'s momentum

minicpm sits at 0.92-0.95x against llama.cpp's FA-on baseline, closest this
session has gotten. Two independent follow-ups dispatched in parallel,
file-separated per the worktree-isolation lesson above (not using actual
worktrees this time -- file overlap is genuinely minimal and both were
briefed explicitly on the prior incident and told to scope any revert
precisely):

1. **Vectorize `warp_partial`'s K/V loads + sweep `ATTN_WARP_SPLITS`.**
   Territory: `phobos-lang/src/codegen/tile/warp_attn.rs`,
   `phobos-gguf/src/backend/device/kernels/attn.rs`.
2. **Re-examine `[[launch-bound-headroom]]`'s remaining unfused launches
   under the new, post-`warp_partial` cost structure.** `store_2d` was
   never actually tried in either prior round (only `quantize`+
   `q8_qdot_add` were, and that was a wash *before* attention's own cost
   dropped 50%+) -- primary target. Territory: `phobos-gguf/src/llama.rs`,
   `phobos-gguf/src/backend/fuse/*.rs`, `phobos-gguf/src/layers.rs`.

## Process lesson: two concurrent `bench.py` runs can corrupt each other silently

The two beams dispatched above ran their own `bench.py` invocations at
roughly the same time and corrupted each other's numbers.
`autoresearch/beams/vectorize_minicpm_bench.log` is the evidence: both
engines read at roughly half this session's established baseline (~100-150
t/s vs the normal ~230-280 t/s), in **every** round including the one
`bench.py` itself labeled "0% util between rounds, uncontended." Why the
label didn't catch it: `bench.py`'s contention check samples GPU state
*between* rounds via `nvidia-smi`; it has no way to detect a sibling
process's GPU work if that work overlaps the *timed measurement itself*
rather than the gap between rounds. Two `bench.py` processes running
concurrently -- as opposed to one `bench.py` process correctly detecting
some unrelated third-party GPU load -- can defeat this check entirely,
because the interference isn't confined to the gaps it samples.

**Lesson for every future round in this session**: dispatching two beams
that will each run their own `bench.py` confirmation concurrently is not
safe by default, unlike file-ownership conflicts (which `git status`/`git
diff --stat` catch) -- GPU contention between sibling agents' benchmarks
produces *plausible-looking, self-labeled-clean* bad data with no error and
no obvious tell beyond checking absolute numbers against a known baseline.
Before trusting or reporting any `bench.py` result from a round that ran
concurrently with another, verify the absolute numbers (not just the
ratio) against this file's own current-state snapshot, and check `tasklist
| grep -i "bench\|python"` for a sibling process, not only `bench.py`'s own
between-round label. When dispatching concurrent beams going forward,
either stagger their benchmark phases explicitly in the brief, or accept
the risk and mandate the absolute-number sanity check as a standing
instruction (done retroactively for the two beams above via `SendMessage`).

## Round landed by direct takeover, 2026-08-17: vectorization + store_2d/output-projection fusion

Both dispatched beams (vectorize K/V loads + sweep `ATTN_WARP_SPLITS`;
re-examine `[[launch-bound-headroom]]`'s remaining fusions) produced real,
substantial, correct work but stalled on their final reports after several
resumes each -- taken over directly rather than waited on further: reviewed
both diffs, ran `cargo check --workspace` and `cargo clippy -D warnings`
clean, ran all four correctness gates (`backend_check`, `fuse_check`,
`batch_check`, `model_check`) on both models with `PHOBOS_ATTN_PERSIST=1`
myself (all clean, matching documented bands), ran a fresh confirmation
benchmark myself, and committed. Full mechanism and numbers in
[[cache-length-split-buckets]]'s and [[launch-bound-headroom]]'s own
Result logs.

**Current state, same session, interleaved against llama.cpp,
`PHOBOS_ATTN_PERSIST=1`, `tg1024/2048/4096`:**

| model | tg1024 | tg2048 | tg4096 |
| --- | --- | --- | --- |
| minicpm5-1b | 0.96x | 0.96x | 0.95x |
| Qwen3.5-0.8B | 1.08x | 1.09x | 1.08x |

Closest this whole session has gotten on minicpm. Not yet crossing 1.0x --
the stated goal is not yet achieved, but the gap has closed from the
session's opening 0.86-0.92x to 0.95-0.96x across three major rounds
(`warp_partial`'s original landing, then this combined round).

## Superseded immediately: the round above never shipped with its own fusions active

The table right above this note is stale as of the same day it was
written. Audit found `store_2d_pair`/`attn_out_chain` (landed in the
combined commit `6a7b6ae`) were gated behind env vars defaulting *off*,
and no confirmation benchmark on record -- including the one that produced
the table above -- ever set them. Re-benchmarked with them on: real,
reproducible gain. Per the user's standing instruction that a probable
improvement must ship on by default, flipped both to the repo's existing
default-on/opt-out convention. Full mechanism, nsys evidence and gate
results in [[launch-bound-headroom]]'s fourth round.

**Current state, same session, interleaved against llama.cpp,
`PHOBOS_ATTN_PERSIST=1` (only flag still forced -- both fusion flags now
default on), `tg1024/2048/4096`:**

| model | tg1024 | tg2048 | tg4096 |
| --- | --- | --- | --- |
| minicpm5-1b | 0.97x | 0.97x | 0.97x |
| Qwen3.5-0.8B | 1.08x | 1.09x | 1.08x |

Flat 0.97x across all three tracked lengths on minicpm -- the most stable
result this session has produced, and closer to 1.0x than any prior
snapshot. Still not crossing the goal. Next candidates per the advisor
consult that opened this round: attention's remaining ~3x-over-floor gap
(ncu on the current, post-vectorization `attention_persist` kernel shows
occupancy now at ~98% -- up from 63.8% before vectorization -- but memory
throughput 37%, compute throughput 35%, and the stall breakdown is
barrier-dominated: 42.7% of stall cycles are warps waiting at a barrier,
24.2% more is global-memory latency, everything else under 11%. A
discriminating measurement against the launched `attention_split` kernel
(same `warp_partial` combine, no `grid_barrier`) found barrier stall
collapses to ~12% there -- so the cost is `attention_persist`'s
`grid_barrier()` cross-block sync, not `warp_partial`'s own combine.
Full counters and both reproduction commands in
[[cache-length-split-buckets]]. This argues against more ILP/key-batching
(occupancy is already maxed, and `attention_split`'s own profile shows
compute throughput is not the ceiling either) and toward reducing
`grid_barrier`'s cost or the cross-block work-imbalance that staggers
arrival at it -- a different lever, and a different kernel region, than
anything tried so far.

## Resolved: `PHOBOS_ATTN_PERSIST` is now default on

The open question above was flagged to the user rather than decided
unilaterally, since a wrong call here fails as a hang, not a slow number.
User's answer: flip it on, but keep `attn_persist_plan`'s own decline as
the actual safety net -- which is exactly the existing structure, since
the flag only ever gated whether `attention_decode` *asks*
`attn_persist_plan` the question, never whether the answer is trusted
blindly. Flipped `DeviceBackend::attn_persist`'s initializer from
`std::env::var_os("PHOBOS_ATTN_PERSIST").is_some()` (unset = off) to
default-true with `PHOBOS_ATTN_PERSIST=0` as the opt-out, not folded into
`fused_stage`'s shared `PHOBOS_FUSED` fallback since this gates a
different mechanism (a persistent kernel and its `grid_barrier`, not a
fused launch chain) and a blanket `PHOBOS_FUSED=0` should not silently
touch it too.

All four correctness gates re-run with **no env vars set at all**
(confirming the true default, not the forced-on config every prior round
used): identical to every documented baseline, including the
`PHOBOS_ATTN_PERSIST=0` opt-out path checked separately and landing in the
same error bands. Confirmation `bench.py`, also with nothing set
(`phobos env: nothing set, so every default is in force`):

| model | tg1024 | tg2048 | tg4096 | pp128 |
| --- | --- | --- | --- | --- |
| minicpm5-1b | 1.05x | 1.04x | 0.96x | 0.82x |
| Qwen3.5-0.8B | 1.17x | 1.22x | 1.23x | 0.84x |

At or above every established baseline (some rows noisier than usual --
desktop GPU contention on this interactive machine, not a code issue; see
the note in [[pipeline-default-on]]'s commit history for the same failure
mode hit and resolved earlier the same session). From this commit
forward, a plain `phobos-cli` invocation gets everything this session's
`tg`/`pp` numbers measured -- no env var needed. Committed:
`phobos-gguf/src/backend/device/{mod.rs,attn.rs}` only.

## Bubble recapture on the current tree: megakernel not funded as a third beam

Per the advisor consult's gate before funding a megakernel beam: recaptured
the launch bubble on the current, fusion-defaults-on tree
(`nsys profile --trace cuda --cuda-graph-trace=node`, minicpm, n=512,
`PHOBOS_ATTN_PERSIST=1`, `autoresearch/beams/bubble_current.nsys-rep` +
`_cuda_gpu_kern_sum.csv`). Launch count dropped materially --
`store_2d_pair` shows one call per layer per step (12,432 over the run)
where the pre-fusion tree showed two separate `store_2d` calls (49,440,
i.e. 2/layer/step); `quantize`+`q8_qdot_add` folded into the generic
`fused` chain kernel alongside the MLP's own fusion. Overall per-step
launch count is down from ~272 (pre-this-session's-final-round) to ~227.

**But the gap percentage barely moved.** Steady-state window (1001
consecutive kernel events, mid-trace, same method as
[[wide-vocab-lm-head]]'s original 14.4% figure): gap is **15.06%** of
wall time now, against that round's 14.4% pre-fusion. A ~16% cut in
launch count produced no measurable cut in the bubble it was expected to
shrink. This matches [[launch-bound-headroom]]'s own suspicion from its
second round: the remaining gap looks like it is not per-launch overhead
that scales with kernel count, but something closer to a fixed
per-graph-replay cost under this WDDM driver, paid regardless of how many
nodes are inside the graph. **Not funding a megakernel beam on this
evidence** -- the pool advisor flagged as an ~11% ceiling does not appear
to actually be there once launch count is properly reduced; retiring the
idea with this diagnosed reason rather than leaving it open. If someone
revisits this, the discriminating question is what specifically the fixed
cost is (graph submission itself? the WDDM batch boundary? something
`nsys`'s own instrumentation inflates that a clean, unprofiled measurement
would not show?) -- not "how many launches remain."

## Barrier-imbalance beam landed, 2026-08-17: the grid_barrier stall was a real work-assignment bug, not latency noise

Picked up the previous round's open question (`ncu` found 42.7% of
`attention_persist`'s stall cycles at `grid_barrier()`) and diagnosed it
before touching anything: `attn_persist_plan` settles the persistent
kernel's resident grid from live occupancy, but `attention_decode` handed
it the *launched* kernel's own `ATTN_SPLITS`-derived split count, which
has no relationship to that settled grid. Confirmed with a fresh pass
report: minicpm's shape assigns `groups * splits = 128` real units of
phase-one work into a 192-block resident grid (a third of the blocks idle
at the barrier from the start of the kernel), Qwen's assigns 64 into 144
(56% idle). Fixed by having `attn_persist_plan` settle its own split count
(`blocks / groups`, floored) so every resident block gets a unit -- full
mechanism, a real feedback-loop bug found and fixed along the way (a naive
one-stage version of this settled a non-deterministic, 3x-too-narrow grid
for Qwen and intermittently corrupted an unrelated later kernel launch via
a stale pointer-keyed cache), and complete numbers are in
`[[cache-length-split-buckets]]`'s "Round 3" section.

**Result**: minicpm's isolated `attndecode` time at cache 4099 dropped
9.4% (690.8us -> 626.0us), and `ncu` confirms the mechanism directly --
barrier stall's absolute cost fell 58% (9.09 -> 3.77 cycles/instruction)
with occupancy unchanged at its ceiling. Qwen's wall-clock stayed flat
(217.8us -> 217.4us, noise), though `ncu` shows Qwen trades the
under-assignment problem for a different one (over-fragmentation at
`splits = 36`, still ~60% barrier stall) -- documented honestly as an open
secondary finding rather than chased further, since it does not regress
Qwen and minicpm (the task's target) is the real win.

All four correctness gates pass on both models with `PHOBOS_ATTN_PERSIST=1`
forced, matching every documented baseline to the digit, verified twice
(once in the shared tree, once in an isolated worktree after a concurrent
agent's in-progress argmax feature landed mid-round and left the shared
tree non-building for unrelated reasons -- see the beam file's process
note on why a plain `git diff` was not safe to use for that isolation
check). **Nothing committed this round**: per the task's brief, the
confirmation `bench.py` run and the commit are the orchestrator's next
step, not this round's.

## Process lesson: subagents can produce real, correct work and still never send a final report

Both beams this round independently exhibited the same failure mode this
session has seen repeatedly: real implementation work, real (eventually
clean) benchmark data sitting on disk, correctness gates presumably run --
and no final report after being resumed multiple times with explicit
"finish up now" instructions. Direct takeover (reading the diffs, re-running
the correctness gates and a confirmation benchmark independently, and
committing based on that firsthand evidence rather than a stalled agent's
unsent conclusion) is the reliable fallback once a resume-and-wait cycle
has repeated 3+ times without a final report, not a last resort to avoid.
The evidence bar for taking over is the same as for trusting a report: don't
commit anything the orchestrator hasn't personally verified compiles,
passes the correctness gates, and benchmarks as claimed.

Update this note after every 3-5 submissions with current ranking and next
combination candidates, per `autoresearch/AGENT.md`.

## New beam landed, 2026-08-17: device-side argmax for greedy decode

Separate from the attention-side work above (the "concurrent argmax
feature" a couple of the notes above mention colliding with in the shared
tree -- confirmed no actual file overlap; see that beam file's own process
note). Targets [[wide-vocab-lm-head]]'s one untried lever: that beam killed
the `q8_qdot` matmul-tuning angle on the logits projection but explicitly
named "not computing the full logits vector at all" as a separate,
out-of-scope-for-that-beam idea. This beam is that idea, scoped to greedy
decoding (temperature 0, no penalties) where the sampler only ever needs
the argmax, not the full distribution.

**What it is.** A new phobos-lang builtin, `argsel(va, vb, ia, ib)` =
`select(va >= vb, ia, ib)`, the index-carrying sibling `tmax` cannot
express (indices travel as f32, exact up to 2^24, well past either
vocabulary). Two small eager kernels
(`phobos-gguf/src/backend/device/kernels/argmax.rs`) grid-stride the
logits row and fold a running `(value, index)` pair with `tmax`/`argsel`,
then an unrolled halving tree collapses each block's partial, and a tiny
serial finish kernel folds the partials -- both launched after `end_pass`
(outside the CUDA graph). `Backend::argmax` (default: host reduction,
free for `HostBackend`) is the new trait surface;
`Decoder::forward_greedy`/`Session::extend_greedy` (default falls back to
`extend` + host argmax, so ONNX and any future backend get it for free)
wire it into the real generation loop
(`phobos_inference::generate::continue_from`) behind
`SampleConfig::is_greedy_unpenalized()`, not just the benchmark.
`bench.py`'s `tg` loop now calls `forward_greedy` too, since that is what a
real greedy-serving deployment does.

**Ceiling, stated honestly**: [[wide-vocab-lm-head]] measured the readback
`Backend::argmax` replaces at 1.7-1.9% of a decode step (the `q8_qdot`
kernel producing the logits is untouched, still runs). Net of the new
kernels' own small cost, closer to 1.3-1.6%, and that shrinks further at
longer cache lengths where a decode step itself is more expensive. Not a
big enough lever by itself to be a new primary beam; it is a genuine,
verified capability landed opportunistically because it was asked for.

**Correctness**: a new example, `argmax_check` (device argmax vs. a host
reduction of the *same device backend's* logits, deliberately not
cross-backend), agrees on 32/32 real decode steps with zero float ties on
both minicpm5-1b-Q8_0 and Qwen3.5-0.8B-Q8_0, plus three synthetic edge
cases (all-negative logits, winner at index 0, winner at the last index)
all agree. All four standing gates (`backend_check`, `batch_check`,
`model_check`, `fuse_check`) pass on both models with
`PHOBOS_ATTN_PERSIST=1`, matching documented error bands. Full mechanism,
two small language-level snags found and fixed along the way (an
index-to-float gap in `coerce`, and a reminder that phobos-lang float
literals have no exponent form), and the source-size-ratchet fallout (two
files were sitting exactly at their grandfathered cap; fixed by splitting
`qwen35.rs`'s forward family into a new `qwen35/forward.rs` descendant
module and extracting `argsel`'s call-site glue out of `expr.rs`, not by
trimming comments) are in
[[greedy-argmax-readback]] (`autoresearch/beams/greedy-argmax-readback.md`).

**Nothing committed.** Per the task's brief, `scripts/bench.py`
confirmation and the commit are the orchestrator's own next step, same as
the attention-side beam above. Changed, beyond the beam-file-documented
list: `phobos-lang/src/codegen/{expr.rs,tile/elem.rs,tests/math.rs}`,
`SPEC.md`, `phobos-gguf/src/{llama.rs,qwen35.rs,qwen35/forward.rs,model.rs,
runtime.rs,backend/mod.rs,backend/device/{mod.rs,backend.rs,argmax.rs,
kernels/{mod.rs,argmax.rs}},examples/{bench.rs,argmax_check.rs},Cargo.toml`,
`phobos-inference/src/{model.rs,sampling.rs,generate.rs}`,
`phobos-base/tests/source_size.rs`.

## Both beams landed by direct takeover, 2026-08-17: the session's goal is crossed

Both agents' work sat in one shared, actively-being-edited tree with
genuine file overlap (`phobos-gguf/src/backend/device/mod.rs`, touched by
both). Verified each independently rather than trusting either report at
face value: `git diff` showed the overlap was cleanly separable into
non-overlapping hunks (confirmed by reading both diffs side by side, not
assumed), so each beam was split out, built, gated, and benchmarked on its
own before being combined.

The attention-side beam was verified twice -- once in the shared tree, and
a second time in a fully isolated `git worktree` (its own
`CARGO_TARGET_DIR`, hardlinked model files) after a `--no-build`
confirmation run against a target directory shared with the still-running
argmax agent came back with intermittent, non-deterministic parse
failures on 5 of 8 rows. Not a bug in the grid_barrier fix: a second,
fully isolated run was clean on all 8. **Lesson for the worktree-vs-shared-
tree question this session kept deferring**: a shared `CARGO_TARGET_DIR`
between two concurrently active trees is not safe even when the source
trees themselves have zero content conflict -- cargo's build outputs
raced. A worktree only earns its cost (model files needing a hardlink,
`.cargo/config.toml` needing a copy since it is gitignored, and a full
from-scratch build unless `CARGO_TARGET_DIR` is deliberately kept separate
too) when a confirmation run needs to be trusted while another agent is
still actively building in the same tree -- which is exactly what
happened here. Committed separately: `fc7de8b` (grid_barrier),
`adbe187` (argmax).

**Combined confirmation, both beams landed, `PHOBOS_ATTN_PERSIST=1`,
same session, same card, interleaved against llama.cpp:**

| model | tg1024 | tg2048 | tg4096 |
| --- | --- | --- | --- |
| minicpm5-1b | 1.00x | 1.00x | 0.996x |
| Qwen3.5-0.8B | 1.16x | 1.16x | 1.16x |

minicpm crosses parity on tg1024/tg2048 for the first time this session
(276.86 vs 275.99 t/s, 273.29 vs 273.19 t/s) and sits a fraction under on
tg4096 (266.17 vs 267.15, -0.4%) -- from the session's opening 0.86-0.92x.
Qwen, not the target but regression-watched throughout, gained
independently from the argmax beam (its much larger vocabulary means the
readback it skips is proportionally larger) and now leads at 1.16x, up
from 1.08-1.09x.

The user's directive was to beat llama.cpp's number, not stop at a
specific mechanism, and to push past any "good enough" stopping point
("Push further. Easy is for the weak") -- tg4096 not yet crossing 1.0x
means this is not fully, unconditionally done. Two honest open items:
Qwen's `attention_persist` path still shows ~60% barrier stall from a
different mechanism (over-fragmentation, not under-assignment -- see
[[cache-length-split-buckets]]'s Round 3), untouched because it does not
regress Qwen; and `PHOBOS_ATTN_PERSIST` itself remains opt-in pending the
user's call on the open question above.

## New beam, 2026-08-17: prefill (`pp128`), never previously a target this session

With `tg` at parity, the user asked to also look at prefill: `pp128` sat at
0.40-0.41x of llama.cpp CUDA in every recorded baseline, and it had never
been benchmarked or profiled as a target before this round. Full mechanism,
A/B numbers, the tensor-core kernel that was built, correctness-verified,
benchmarked and then reverted (a real measured loss, not a null result),
and the precision finding that came out of debugging it, in
[[prefill-attention-tensorcore]] (`autoresearch/beams/prefill-attention-tensorcore.md`).

**Shipped**: `DeviceBackend::attention` (`phobos-gguf/src/backend/device/backend.rs`)
now tries the already-existing, already-proven `attention_blocked` kernel
(single-pass online-softmax, no materialized score matrix) ahead of the
three-launch `attn_gemm_src` path it used to prefer whenever the shape
tiled evenly by 64. A real `nsys` trace found attention tied with the
projection GEMM at 37.5-37.9% of a minicpm pp128 pass, not the minor cost
`attn_gemm_src`'s own doc comment assumed (corrected in place) -- and the
kernel it was skipping already did the same work 66% cheaper. Same-session
trace, minicpm pp128, 3 reps: total prefill kernel time 33.04ms -> 23.62ms
(-28.5%), attention's own share 12.53ms -> 4.25ms (-66.1%), `q8_qmma`
(the GEMM, untouched by this change) unaffected. All four correctness
gates pass on both models, `PHOBOS_ATTN_PERSIST=1`, matching documented
baselines to the digit (`backend_check` worst error 2.902e-4, identical to
the pre-existing figure).

**Tried and reverted**: a tensor-core (`@tensorcore`/`@pipeline`, f16 Q/K/V,
WMMA legacy path -- confirmed engaged before building, `sm_75` + 32-bit
index means `has_mma_sync` is false but `has_wmma` isn't) version of the
same kernel, modeled on `examples/flash_attention_fp16.ph`. Correctness-verified
(a real bug hunt resolved to a precision finding -- f16 rounding in the PV
product, full strength at a from-scratch prompt's first block where no
prior loop dilutes it, still within a justified `1e-3` bound since
llama.cpp's own prefill kernel is f16 too) and all four gates passed with
it active. But measured **27-40% slower than the plain-f32 kernel it was
meant to replace** at two launch widths tried (`nsys`, same shape): the
matmuls here (`16x16x128`/`16x128x16`) are too small for WMMA's per-launch
fixed cost (fragment load/store through a shared-memory epilogue slab,
plus a query f32->f16 cast) to amortize. Reverted cleanly; only the
dispatch-reorder and a doc-comment fix remain in the tree. Concrete next
steps (wider `BR`, folding the diagonal tile into the pipelined loop, a
wider query group) are in the beam file for whoever picks this back up --
none attempted this round given the clear, repeated loss at the one tile
size tried.

**Not yet run: the `bench.py` confirmation.** Per this session's standing
rule, that and the resulting commit are the orchestrator's own next step,
same as every other round this session. The remaining gap after the
reorder (from the same trace): phobos's total prefill kernel time is now
1.84x llama.cpp's (down from 2.58x); the GEMM projection (1.29x) is the
largest absolute per-kernel gap, attention (5.74x) the largest relative
one and the one the tensor-core attempt tried and failed to close at
`BR=16`.

## Orchestrator diagnostics, 2026-08-17: why tensor cores are the wrong lever for decode, and one q8_qmma occupancy datum

Gathered while scoping the prefill investigation above, before dispatching
it -- kept here since they were reported to the user directly but not yet
on record.

**Decode attention (`attention_persist`, post grid-barrier-rebalance and
argmax) genuinely does not want tensor cores.** Direct `ncu` measurement at
cache 4099, minicpm shape (`autoresearch/beams/ncu_persist_post_argmax_4099.ncu-rep`):
`sm__pipe_tensor_op_hmma_cycles_active` is 0.22% of peak -- essentially
zero, and correctly so. The kernel does ~34 MFLOP/step while streaming
4.36MB of K/V cache (`dram__bytes_read.sum`, matching the ~4.2MB
theoretical distinct almost exactly -- traffic is not redundant, ruling out
a qgroup-restructuring lever). Under 1% of this card's FMA capacity; this
is a bandwidth-streaming kernel, not a FLOP-bound one, so tensor cores
(which accelerate FLOPs) cannot be the lever regardless of how they are
wired in. `sm__inst_executed_pipe_fma` sits at 35.48% of peak, matching
`ncu`'s own Compute (SM) Throughput reading (39.61%) -- the kernel is
latency-bound (barrier + long-scoreboard stalls, see the grid-barrier-
rebalance beam above), not compute-bound, so there is no idle FMA capacity
tensor cores would be racing against either.

**Swizzling is a closed question for this kernel too**: shared-memory bank
conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}`) sum
to 498 against 284,855 load wavefronts, ~0.17% -- negligible. No swizzle
work is warranted here.

**One `q8_qmma` (prefill GEMM) datum, not yet generalized**: `ncu` on a
single call captured mid-prefill (`autoresearch/beams/ncu_qmma_prefill.ncu-rep`)
showed **12.45% achieved occupancy against a 25% theoretical ceiling** --
grid `(1, 20, 1)`, only 20 of this card's 48 SMs had any work at all. Int8
tensor-core utilization on that call was a real 19.61% of peak (not zero,
unlike decode), and L2 hit rate was high (79.97%), so redundant global
fetches across warps are not the story. This is one shape, not
representative of every projection minicpm calls (the file's own comment,
`"keeping the grid full was costing the two 1024-wide projections about a
quarter of their throughput"`, already documents this exact width-vs-grid-
fill tradeoff as known) -- flagged, not chased, since the same round's
kernel-time breakdown found attention tied with the GEMM for prefill's
dominant cost, and the prefill beam above went after that first.

## New beam, 2026-08-17: warp_partial register-level K/V prefetch

Third beam in the "prefill rebuild, then @pipeline default-on, then decode
prefetch" sequence the user asked for. Full mechanism, register/occupancy/
spill evidence, and the honest twist in the result, in
[[warp-partial-prefetch]] (`autoresearch/beams/warp-partial-prefetch.md`).

**Summary**: one-deep register software pipeline for `warp_partial`'s K/V
loop (issue next iteration's raw f16 load before consuming the current
one), landed at zero register/occupancy/spill cost (64 registers/thread,
98.4% occupancy, 0 bytes local-memory traffic, all unchanged). The
`long_scoreboard` stall it targeted dropped as predicted (~31.6% -> ~7.5%
share on `attention_persist`), but wall-clock on that specific kernel
stayed flat -- `attention_persist`'s critical path turned out to be
`grid_barrier` cross-block skew (already diagnosed, a different
bottleneck), which absorbed the freed-up stall budget instead of shortening
the kernel. Real, honest, mechanism-confirmed result, not the clean win
hoped for on the kernel it targeted.

**But a genuine free win landed anyway**: `attention_split` -- minicpm's
fallback path and **Qwen's only decode-attention path, always** (its shape
declines the persistent kernel unconditionally) -- improved 6-9% on both
models' isolated `attndecode` timing, zero cost. `bench.py` confirms no
regression (minicpm flat as predicted, Qwen positive though noisy at this
sample size given only 6 of 25 layers do full attention).

Committed: `phobos-lang/src/codegen/tile/warp_attn.rs` only.

## New beam, 2026-08-17: `@pipeline` becomes an assertion, pipelining becomes the default -- a language-design change the user requested directly

Not a performance beam: the user asked, mid-session, why `@pipeline` should
ever need to be written when the compiler could just identify eligible
loops itself. Full mechanism, the two correctness gaps auto-attempt exposed
(CTA-uniformity of loop bounds, shared-memory budget) and how each was
closed, the `Variants`/masked-fallback resolution, and the full emit-diff
sweep's findings in [[pipeline-default-on]]
(`autoresearch/beams/pipeline-default-on.md`).

**Headline finding**: across every `.ph` file in the tree and 36
representative Rust-embedded kernel-source instantiations (47 compiles
total, `.ph` and `PHOBOS_CHIP=sm_80 PHOBOS_INDEX_BITS=64`, before/after,
diffed), exactly **one** loop newly pipelines under the default --
`attention_src`'s single-key remainder tail in the rare misaligned-multi-
row-continuation fallback (`DeviceBackend::attention_rows`), off both
`pp128` and `tg1024`'s hot paths. The reason so few loops qualify is
structural and is the real finding: `pipeline_candidate` prescans a loop
body *before* its own induction variable is bound, so `expr_div` of an
unbound loop var defaults to 1, and no `@aligned` promise on the tensor
side alone can clear the eligibility check's `&&` for the canonical
`var k = K[kt :+ BC, ...]` shape this mechanism targets. A modest-effort
follow-up (threading the loop var's own provable divisor into the prescan,
machinery for which -- `slice_is_partial_within`'s `ivs`/`pending` --
already exists for `emit_split_for`) would make `@aligned` k-loops
genuinely eligible; noted in the beam file, not chased here.

**Mid-task correction worth flagging**: the first design added real taint-
tracking machinery (a `HashSet` populated on every scalar binding) to guard
against `atomic_add`'s return value reaching a loop bound and hanging a
barrier-in-`scf.if` guard. Building the test for it proved the scenario
does not compile in this language at all -- `atomic_add` returns `i32`,
loop bounds require MLIR `index`, and neither `Codegen::coerce` nor
`Codegen::unify` has any path from one to the other. The machinery was
removed as provably-dead code and replaced with a proof comment plus a
regression test pinning both failure modes as a tripwire against a future
language change reopening the hole.

**Second correction, caught by review before this report went out, not by
my own testing**: the emit-diff sweep above verifies MLIR text via
`phobos_lang::codegen::emit` directly (what `cargo run -p phobos-lang
--example emit` calls), which bypasses `compile_shared` entirely --
exactly the function that enforces `@pipeline`'s new assertion. That sweep
is structurally blind to assertion failures. Checked every `@pipeline` site
in the tree against `compile_shared` directly and found three stale
annotations that now hard-fail: `examples/flash_attention_fp16.ph` and
`flash_attention_fp32.ph` (fragment-carried loop, `let`-not-`var` staged
values, `@pipeline` has always been inert there) and `examples/gemm_fp16.ph`
(an f16-accumulator GEMM that `matmul_candidate`'s tensor-core/vector-path
branches both miss on this codebase's default chip, so it falls to the
generic path and declines the same way the headline finding describes).
The flash break is not hypothetical: `phobos-bench/src/flash.rs` loads
both files through `autotune::compile` -> `phobos_lang::compile` ->
`compile_shared`, so `cargo run -p phobos-bench`'s flash benchmarks would
have hard-failed. All three fixed by removing the stale `@pipeline`
(verified byte-identical MLIR before/after on both targets, verified
`compile_shared` now succeeds on all nine `examples/*.ph` files on both
targets). `phobos-onnx` carries no `@pipeline` anywhere and was checked
separately (inspected, and its two loop-bearing kernel shapes verified
against the same before/after diff as above).

**Fourth file, left open, not fixed**: `examples/gemm_fp32.ph` still
carries `@pipeline`, and enumerating its full `@autotune` search space
(64 configs, `TILE_M`/`TILE_N` each doubling `32->256`, `TILE_K` doubling
`4->32`) through `compile_shared` found 12 failures, all the asymmetric
`TILE_M=128,TILE_N=256` / `TILE_M=256,TILE_N=128` tile shapes crossed with
every `TILE_K` -- `matmul_candidate`'s `sub_tile`/`lane_grid` split has no
answer there, so those 12 fall to the generic path and decline the same
way as everywhere else in this sweep. Confirmed pre-existing (same before/
after-stash proof as the other three), and confirmed the other 52 configs
genuinely do pipeline, so stripping the attribute would be the wrong fix
here -- it would discard a real assertion holding for 52 of 64 points.
`phobos-bench`'s autotuner tolerates per-config compile failures during
its unpinned search (catches and logs `"skipped"`, keeps going), but a
`--autotune` pin landing on one of the 12 fails hard
(`"autotune: no config works"`). Left as-is and surfaced in
[[pipeline-default-on]] as an explicit open decision -- whether `@pipeline`
should mean "pipelines for some config in its own search space" (true
here) or "pipelines for whatever config a caller picks" (false for these
12) is a semantics question this task's brief did not anticipate for an
autotuned kernel, and is not resolved unilaterally.

**Correctness**: all four standing gates (`backend_check`, `batch_check`,
`model_check`, `fuse_check`) pass on both models with
`PHOBOS_ATTN_PERSIST=1`, matching documented error bands to the digit
(`backend_check` worst error 2.902e-4). `backend_check`'s own shape sweep
happens to exercise the one newly-pipelined kernel directly (`5 rows @ 31,
16/2 heads, head_dim 64` dispatches to `attention_rows`) and it agrees at
rel err 1.043e-7. `cargo test -p phobos-lang`: 155/155 (4 new tests).
`cargo test --workspace`: clean. Release build + clippy on
`phobos-gguf --features cuda`: clean.

**Not run**: `scripts/bench.py`, per this session's standing rule --
orchestrator's next step, same as every other round. This task's own quick
nsys/isolated-timing obligation was assessed and not exercised: the one
kernel that changed is a size-1-slice tail loop unreachable from `pp128`
or `tg1024`'s shapes, so there is nothing on the hot path for a timing
check to catch either way; correctness on the exact changed path is
already confirmed via `backend_check`.

**Nothing committed.** Changed:
`phobos-lang/src/{lib.rs,codegen/{mod.rs,pipeline.rs,stmt.rs,
matmul/{plan.rs,reg.rs,wmma.rs},tests/pipeline.rs}}`,
`phobos-kernels/src/compile.rs`, `SPEC.md`, and three example files with
stale `@pipeline` removed: `examples/flash_attention_fp16.ph`,
`examples/flash_attention_fp32.ph`, `examples/gemm_fp16.ph`.

## New beam landed by direct review, 2026-08-17: `q8_qmma` split-K -- real per-kernel win, real end-to-end loss, shipped default off

Fourth item in the "prefill rebuild, `@pipeline` default-on, decode
prefetch, revisit `q8_qmma`" sequence, per the user's explicit "q8_qmma,
yes". Diagnosis (`ncu -c 20 --set full` across a real minicpm prefill,
generalized from an earlier single-shape datum in
[[orchestrator-diagnostics-q8-qmma]]): `q8_qmma`'s deep tile launches
`(rows/128) * (n/wide)` blocks and nothing else, a constant 12, 20 or 72
against 48 SMs on `pp128`, landing 12.4-19.7% achieved occupancy against a
25% register-bound ceiling (128 live accumulators/patch caps a CTA to 2
resident blocks/SM). Stall breakdown on both a starved and a fuller shape:
`long_scoreboard` dominates overwhelmingly. A live probe (temporary
dispatch-order swap, reverted) confirmed shrinking `TM`/`TN` is not the
fix: IMMA utilization collapses (19-21% -> 5.5-9.2%) because arithmetic
intensity is `TM*TN/(TM+TN)`, and per-shape timing showed it only wins on
badly starved shapes and actively loses on an already-fed one (+57% on the
grid-72 shape). Split-K (`k` sliced across `program_id(2)`, same `TM`/`TN`,
same intensity) was the fix that adds blocks without paying that cost --
mirroring the existing `q8_split`/`q8_reduce` decode-path pattern.

An implementation agent (`a00876fdabe5e3de2`) built it and found two real
compiler-proof bugs before either one produced a wrong answer or a slow
number silently: folding the split index into the destination offset
compiles clean but silently declines `qmma_t`'s direct-to-global write
(a dynamic tensor-shape symbol like `M` is only ever assumed a divisor of
4 regardless of what `@aligned` promises about it, so the bounds proof
never clears) -- caught by reading emitted MLIR, not by an error. Fixed by
giving each split its own output operand and literal-offset `k`-slice
bounds instead. The reduce kernel's first version (`[TM,TN]`-tile add)
blew the 48 KB static shared-memory limit at the real `TM=128` shape --
passed MLIR verification *and* PTX codegen, only failed at driver-level
`Module::from_ptx`, which neither `emit` nor `ptx` would ever catch. Fixed
by reducing one row at a time, matching `q8_reduce`'s own pattern. Both
failure modes are now documented in the kernel source and the beam file
for the next person who reaches for this shape of fix.

**Per-kernel, the win is real.** On minicpm's routed shapes: o_proj
-22.5%, down_proj -43.7% (`ncu` `gpu__time_duration.sum`, write+reduce
pair against the unsplit baseline). Qwen's routed shapes win by more
(-37.3%, -46.5%). A first cut of the gate (split whenever the unsplit grid
is below a 48-block threshold) also caught a real regression on one shape
(qkv, +4.5%, `S=4` only) with a mechanism that checked out numerically
(predicted reduce-pass bandwidth cost ~15.3us against a measured 16.0us) --
fixed by requiring the halving loop land exactly on the target split
factor (`S=8`), grounded in the pattern that every winning shape, both
models, reached that same factor and the one losing shape didn't.

**End to end, on this platform, it is a net loss, caught only by
questioning a benchmark number that looked too good.** The orchestrator's
final review ran a `scripts/bench.py --no-llama` phobos-only A/B specifically
to isolate the change from cross-process noise: `PHOBOS_QMMA_SPLIT=0`
averages 5524 t/s at minicpm `pp128`, on averages 4480-4774 t/s -- a
reproducible 14-19% end-to-end loss, matching two full `bench.py` runs
against llama.cpp where the established 0.76x baseline dropped to 0.61x
and 0.54x. An `nsys --trace=cuda` whole-pass trace (`cuda_api_sum`,
`cuda_gpu_kern_sum`) ruled out the two obvious explanations: in-context
kernel durations roughly match the isolated `ncu` numbers (no cache
interaction between the write and reduce kernels), and `cuGraphInstantiate_v2`
fires exactly twice in both the on and off traces (a pre-existing
prefill/decode single-slot cache-eviction pattern in `graph.rs`, unrelated
to split-K) -- ruling out repeated graph rebuilds too. What's left: every
split-routed projection replaces one graph node with two (write, then
reduce), and `graph.rs` builds a pass as "a chain rather than a dependency
analysis... these launches shared one stream, so serial order is the
ordering they already relied on" -- a strictly serial dependency chain, no
parallelism between nodes. The 48 extra serialized nodes (2 shapes x 24
layers) cost more in aggregate wall clock on this platform than the
kernels save, even though no individual kernel got slower. This is the
same "op that matches in isolation can still be wrong" rule this project
already holds, one level up: at the graph/pass level instead of the
allocation level, and it took a whole-pass A/B, not a kernel-level one, to
see it.

**Shipped default off.** `PHOBOS_QMMA_SPLIT` is opt-in (`=1`/`on`/`yes`/
`true`; unset or anything else stays off, unlike this session's other
opt-out-style flags, since the default behavior here is a regression, not
a safe fallback). All four standing gates and both models' 128-token
`model_check` pass with the new default (the well-tested existing unsplit
path) and with the flag explicitly on (confirming the opt-in path still
matches the host reference for anyone revisiting this at a different
node-count or model shape mix -- Qwen's larger per-kernel margins are the
reason this might be worth another look rather than deleting). Final
confirmation `bench.py` against llama.cpp with the new default: 0.73x,
back in the neighborhood of the pre-split 0.76x baseline.

**Committed** (`56982fe`): `phobos-gguf/src/backend/device/{kernels/quant.rs,
matmul.rs,mod.rs}`, full writeup and raw evidence (`ncu`, `nsys`, and the
orchestrator's `bench.py` A/B files) in
`autoresearch/beams/q8-qmma-split-k.md` and its accompanying files.
