# q8_qmma narrow-CTA path for starved prefill grids

Fifth beam on `q8_qmma` this session. The first four are `q8-qmma-split-k.md`
(real per-kernel win, real end-to-end loss from a 6.3MB-per-shape partials
round trip), `q8-qmma-kloop-prefetch.md` (register-neutral, regressed 8.8-42%
anyway), `prefill-attention-tensorcore-wide-br.md` (a different kernel, same
lesson: check the budget before writing code), and
`q8-qmma-load-width-interleave.md` (killed at a producer census before any
code). This beam had to confirm it was not a retread of the "tile-shrink
probe" `q8-qmma-split-k.md` references and `Q8_QMMA_SPLIT_THRESHOLD`'s own
doc comment names, before writing anything.

## Retread check -- cleared

The tile-shrink probe (and `Q8_QMMA_WIDTHS`' existing 64-wide fallback, which
is the same configuration) shrinks `TN` at the CTA's *unchanged* 128-thread
size. Hand-traced `qmma_patch` (`phobos-lang/src/codegen/tile/qmma.rs`) for
that case: `qmma_patch(rt=16, ct=8, warps=4)` resolves to `(rm=4, rn=8)`, a
worse-intensity per-warp patch than the shipped 128-wide config's
`qmma_patch(rt=16, ct=16, warps=4) = (rm=8, rn=8)`. That is why it lost.

This beam instead halves the CTA *together with* the column tile
(`Q8_QMMA_NARROW_CTA=64`, `Q8_QMMA_NARROW_TN=64`):
`qmma_patch(rt=16, ct=8, warps=2)` also resolves to `(rm=8, rn=8)` -- the
same patch, same 64-tile register-bound cap, same tensor-core intensity.
Confirmed against emitted MLIR, not just hand-arithmetic: a throwaway `.ph`
probe at each config, diffed through `cargo run -p phobos-lang --example
emit`, gave structurally identical 1792-line output (both direct-to-global
writes, no `scf.if`/shared-memory decline), with the loop trip counts
(`scf.for %arg5 = %warp to %c2 step %c2` vs `... to %c4 step %c4`) matching
the hand calc exactly. Real, load-bearing distinction, not a retread.

## Two pinned facts

From `ncu --import` on `autoresearch/beams/ncu_qmma_multishape.ncu-rep`
(import needs no elevation, only capture does):

1. `Q8_QMMA_TM=128`, `Q8_QMMA_WIDTHS=[128,64]`, `Q8_QMMA_CTA=128`. All four
   routed pp128 shapes resolve `wide=128`, grids `(1,12,1)` o_proj,
   `(1,12,1)` down_proj, `(1,20,1)` qkv, `(1,72,1)` gate/up -- matches the
   task brief's numbers exactly. `238 regs/thread`, `Block Limit
   Registers = 2`, `Theoretical Occupancy 25%`, this card confirmed 48 SMs
   (`ncu`'s own advisory text: "grid... is less than the 48 multiprocessors").
2. 4 warps cooperate on one CTA's `[128,128]` output tile at the routed
   shapes (not `[128,64]`; the shallow `Q8_QMMA_TN=64` constant is only used
   by the remainder tile, never the deep one at these shapes). Each warp owns
   a `64x64` quadrant, one patch per warp, no loop -- `qmma_t_into`'s
   `scf.for(warp, total, warps, ...)` runs exactly one iteration per warp
   since `total == warps`.

## Paper DRAM-delta accounting -- cleared

Grid.x is 1 for every routed shape (`rows == Q8_QMMA_TM`), so every CTA in a
launch already shares the same `pm=0` row-block and already re-reads the
identical A tile today (12 to 72 times depending on shape) -- this
redundancy is not new, only its multiplicity changes. A is 192KB (qkv/gate-
up, k=1536), 256KB (o_proj, k=2048) or 576KB (down_proj, k=4608): trivially
L2-resident on this card's L2 (TU104, 4MB), and the existing kernel already
measures 79.97% L2 hit rate on this exact read pattern
(`autoresearch/beams/BEAMS.md`'s orchestrator-diagnostics section). Doubling
the CTA count only doubles the number of *readers* of an unchanged-size, already-hot
block -- structurally unlike split-K's new 6.3MB write-then-read round trip
with zero prior producer. Cleared on paper; see "Cache-control cross-check"
below for the empirical confirmation.

## Implementation

`phobos-gguf/src/backend/device/kernels/quant.rs`: `Q8_QMMA_NARROW_CTA=64`,
`Q8_QMMA_NARROW_TN=64`, `q8_qmma_narrow_eligible(rows, n, wide)` -- gates on
`wide == Q8_QMMA_WIDTHS[0]` (128, the widest option) and the unsplit grid
being under `Q8_QMMA_SPLIT_THRESHOLD` (48, this card's SM count -- the same
question `q8_qmma_splits` asks for split-K, so it reuses the same constant).
`phobos-gguf/src/backend/device/matmul.rs`: `launch_qmma_narrow`, lazily
compiling a `RefCell<Option<Module>>` (single fixed shape, no per-call
variation, unlike split-K's/stream-K's per-`k` keyed caches), checked ahead
of split-K in `project_q8`'s dispatch. `phobos-gguf/src/backend/device/mod.rs`:
`qmma_narrow: bool` field, `PHOBOS_QMMA_NARROW=1/on/yes/true` opt-in, default
off. No reduction pass, no scratch buffer -- the narrow kernel writes its
output tile directly, same as the unsplit path, just at half the width and
half the CTA.

`q8_qmma_narrow_eligible` declines gate/up (grid 72 >= 48) by construction,
so the already-well-fed shape is untouched -- both the mechanism's own
design and (per the numbers below) the right call.

## Correctness -- all four gates pass, both models, both toggle states

`cargo check`/`cargo clippy -p phobos-gguf --release --features cuda -- -D
warnings`: clean. Files stay under the 900-line cap (quant.rs 516, matmul.rs
439, mod.rs 532). `backend_check`, `batch_check`, `fuse_check`: identical to
the documented baseline (worst relative error 2.902e-4) in all four
combinations (2 models x 2 toggle states). `model_check` run with a 128+
token prompt (the default ~5-token prompt never reaches the routed shape --
`matmul_quant [128 x 1024 x 2048]` in `backend_check`'s own sweep is the one
that actually exercises this path, grid 16 < 48, and its error is the one
that hits the 2.902e-4 worst-case figure in every run): `backends agree`,
identical spread errors to the off-state, both models, both toggle states.
`cargo test --workspace`: 306 tests, clean. Toggle-off confirmed a true
no-op (lazy compile; the module is never even constructed unless the env
var and the eligibility check both fire).

## GPU kill-check ladder

GPU verified idle by the orchestrator before this ladder ran. `nvidia-smi`
itself showed fluctuating background utilization (6-36%, clocks bouncing
300MHz-1650MHz) from ordinary desktop load (Chrome, Slack, WhatsApp, window
compositor) even after the orchestrator's own check -- rather than trust
either reading blind, ran a fresh baseline capture first as a self-
consistency check: it reproduced the pre-existing `ncu_qmma_multishape.ncu-rep`
durations (73/89-91/140/190us) to within 1-2%, which is the actual evidence
the desktop noise was not perturbing these short, focused kernel captures.
Proceeded on that basis.

### Register / Block-Limit confirmation

`--set full`, one launch per routed shape, `PHOBOS_QMMA_NARROW=1`
(`ncu_narrow_full_qkv.ncu-rep`, `_oproj.ncu-rep`, `_downproj.ncu-rep`):

| shape | regs/thread | static shared/block | Block Limit Registers | Block Limit Shared Mem | Theoretical Occupancy |
| --- | --- | --- | --- | --- | --- |
| qkv (grid 1,40,1) | 237 | 0 B | **4** | 16 | 25% |
| o_proj (grid 1,24,1) | 237 | 0 B | **4** | 16 | 25% |
| down_proj (grid 1,24,1) | 237 | 0 B | **4** | 16 | 25% |

Exactly as predicted: regs/thread unchanged (238 -> 237, noise from the
CTA-size-dependent index arithmetic, not a real register-count change),
`Block Limit Registers` moved 2 -> 4 blocks/SM (same math as
`q8-qmma-kloop-prefetch.md`'s own `65536 / 128 / 240 = 2` check, now
`65536 / 64 / 240 = 4`), no shared-memory usage so `Block Limit Shared Mem`
stays non-binding at 16 (never the constraint either config), and
`Theoretical Occupancy` stays exactly 25% (8 resident warps/SM either way:
2 blocks x 4 warps before, 4 blocks x 2 warps after -- confirms the
mechanism's own claim that this raises SM count reached, not the occupancy
ceiling itself).

### Cache-control cross-check (DRAM read traffic)

`ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum`, o_proj shape,
`--cache-control all` (forces a flush, worst case) and `--cache-control
none` (real back-to-back-launch L2 state), baseline vs narrow:

| config | cache-control | dram reads | dram writes |
| --- | --- | --- | --- |
| baseline (grid 1,12,1) | all | 3.85 MB | 1.81 MB |
| baseline (grid 1,12,1) | none | 3.85 MB | 1.51 MB |
| narrow (grid 1,24,1) | all | 3.85 MB | 169.66 KB |
| narrow (grid 1,24,1) | none | 3.85 MB | 178.43 KB |

`dram__bytes_read.sum` is **byte-for-byte identical (3.85 MB) across all
four captures** -- doubling the CTA count added zero measurable extra DRAM
read traffic, in either cache-control mode. This is a stronger result than
the paper accounting predicted ("should be small, L2-absorbed"); it came
back exactly flat.

o_proj is the paper argument's best case (smallest A, 256KB) and weakest
generalization -- down_proj has the biggest A (576KB, 2.25x bigger) and the
biggest `k` (4608, so the W streaming volume that could evict A's L2 lines
is also 2.25x bigger, which is exactly the eviction mechanism the stream-K
beam's own cross-check found for a different read pattern). Repeated the
`--cache-control none` half of the check at down_proj as the harder case:

| config | cache-control | dram reads | dram writes |
| --- | --- | --- | --- |
| baseline (grid 1,12,1) | none | 8.65 MB | 188.32 KB |
| narrow (grid 1,24,1) | none | 8.66 MB | 208.58 KB |

0.01 MB apart (~0.1%), within measurement noise -- the flat result
generalizes to the bigger-A, bigger-`k` shape, not just o_proj's easier
case. The write-side numbers differ but are not evidence of
anything here -- no new buffer is written in this design at all (no scratch,
no reduce pass), and `--cache-control`'s write-flush-timing artifacts are
the same effect `q8-qmma-streamk.md`'s own cross-check flagged and set
aside for its baseline capture.

### Per-shape duration, and the falsifiable prediction

Predicted before capturing (per the task brief's "waves per grid" framing,
corrected): **the brief's own `grid/(SM*ceiling)` formula is invariant
under this change** -- 0.125/0.125/0.208/0.75 before and after, since grid
and ceiling both double together. The metric that actually moves is
**SMs touched**: o_proj/down_proj 12->24 of 48 (still half-covered), qkv
20->40 of 48 (nearly saturates), gate/up 48->48 unchanged (already
saturated both sides, and declined by the gate regardless). Predicted:
large, real movement on the three routed shapes, near-flat on gate/up.

Light-metrics capture, `--kernel-name regex:"^q8_qmma$"`, 3 repeats per
shape (`ncu_narrow_off_light.ncu-rep`, `ncu_narrow_on_light.ncu-rep`):

| shape | baseline (avg of 3) | narrow (avg of 3) | delta |
| --- | --- | --- | --- |
| qkv (grid 20->40) | 72.13 us | 59.31 us | **-17.8%** |
| o_proj (grid 12->24) | 90.49 us | 71.34 us | **-21.2%** |
| down_proj (grid 12->24) | 189.78 us | 157.12 us | **-17.2%** |
| gate/up (grid 72, declined) | 141.01 us | 139.38 us | -1.2% (noise) |

Isolated sum of one layer's four shapes: 493.41us -> 427.15us, **-13.4%**.

**Directional prediction confirmed cleanly, fine-grained ordering did not
hold exactly.** All three routed shapes moved substantially (17-21%), the
declined shape stayed flat within noise -- the qualitative
routed-vs-declined split the mechanism predicts is real. But the ordering
inside the routed group does not track SMs-touched fraction alone: o_proj
(25%->50% coverage) shows the *largest* win, not qkv (41.7%->83.3%
coverage, the shape closest to full saturation), and down_proj's win
(25%->50%, same coverage jump as o_proj) is close to qkv's rather than
matching o_proj. Since all three shapes' grids exactly double regardless
of `k` or baseline duration, the SMs-touched-fraction story alone does not
fully explain the cross-shape ordering; something shape-dependent (`k`
depth, or fixed per-launch overhead being a larger fraction of qkv's
shorter baseline duration) also modulates the win. Reporting this
honestly rather than fitting the ordering after the fact -- the ladder's
own instruction was to say so plainly if the gradient does not track the
prediction's shape, and here it tracks the coarse (routed/declined) split
but not the fine one.

**Alternative hypothesis (halving warps/SM to 2 could starve per-SM
latency-hiding) is not what happened.** All three routed shapes improved
substantially despite dropping from 4 to 2 resident warps/SM -- doubling
SMs touched dominated any loss from thinner per-SM occupancy, on this
kernel's stall profile (`long_scoreboard`-dominated, per the diagnosis
this beam started from).

### The 75.75 -> 65.49 t/s line visible in the raw captures, explained

Every `ncu`-wrapped `bench.exe` run above prints `bench.rs`'s own bottom-line
`pp128` t/s figure, host-timed, unrelated to the per-kernel
`gpu__time_duration.sum` numbers the table above is built from. Those lines
show toggle-off at 75.75 t/s and toggle-on at 65.49 t/s -- a 13.5% *slower*
whole-run number under the very toggle that wins 17-21% per kernel. Worth
running down rather than leaving in the logs unexplained.

Cause: `launch_qmma_narrow` compiles its module lazily, on first use
(`RefCell<Option<Module>>`, same pattern as `launch_qmma_split`). Every
capture above used `--no-warmup -r 1`, so the *first* prefill pass in the
process is also the *only, timed* one -- the one-time PTX JIT compile
(hundreds of ms) lands inside the timed region. The rep time in the raw
logs confirms it: 1.690s (off) vs 1.955s (on), +265ms, in JIT-compile
territory; model load time was identical either way (2491-2533ms), ruling
out a load-time confound.

Confirmed by re-running with warmup restored (`bench.exe -p 128 -n 0 -r 3`,
no `--no-warmup`, so the compile lands in the untimed warmup pass instead):

| toggle | pp128 t/s |
| --- | --- |
| off | 6440.38 +/- 361.82 |
| on | 6882.97 +/- 413.04 |

Narrow wins the whole isolated prefill pass by **+6.9%** once the one-time
compile is out of the timed region -- consistent with, not contradicting,
the per-kernel table above. (These are `bench.exe`-only numbers, isolated
from llama.cpp and from decode -- not a replacement for the orchestrator's
own `scripts/bench.py` confirmation, just closing out this specific
discrepancy.)

Checked whether compiling a module while `self.recording.get()` is true is
safe, rather than assuming it from precedent: `graph.rs` shows this
backend's "recording" is its own Rust-level `Vec<Recorded>` buffering, not
a live CUDA `cuStreamBeginCapture` -- the actual `cuGraphCreate`/
`cuGraphAddKernelNode`/`cuGraphInstantiate_v2` calls only happen later, once
recording stops. `compile()`'s `cuModuleLoadDataEx` has no interaction with
CUDA's graph-capture API at all in this design, so a lazy compile mid-pass
is an ordinary driver call with no capture-state hazard -- confirmed by
reading, not inferred from the correctness gates passing (though those also
already exercised this exact path: `model_check`/`fuse_check`'s first
forward call is the first-ever `q8_qmma_narrow` compile, during a recorded
pass, and both passed clean).

One real, worth-flagging cost: the first prefill pass on a fresh process
pays a one-time ~265ms stall the first time a starved shape routes through
this path. Invisible in steady-state throughput (every later pass reuses
the cached module) and consistent with `q8_qmma_split`'s and
`q8_qmma_streamk`'s identical lazy-compile shape, so this is not a new
category of cost this beam introduces -- flagging it because this specific
comparison surfaced it clearly, not because it is unique to this design.

## Recommendation

Implement, opt-in, `PHOBOS_QMMA_NARROW=1`. Every kill-check on the ladder
cleared: not a retread of the tile-shrink probe (confirmed against emitted
MLIR), paper DRAM accounting cleared and the empirical cache-control
cross-check came back flat (not just small -- literally byte-identical
DRAM reads across all four conditions), the register/Block-Limit story
landed exactly as predicted with no new binding constraint, and per-shape
duration moved 17-21% on the three starved shapes with the declined shape
staying flat, confirming both halves of the mechanism (adds CTAs) and the
gate (correctly declines the already-fed shape).

**Reliability caveat, stronger than usual for this kernel**: `q8_qmma`'s
isolated per-kernel wins have been wrong twice this session already
(split-K, K-loop prefetch) -- real in isolation, not in the whole pass.
This design has no DRAM-traffic mechanism resembling either prior failure
(no scratch buffer, no reduction pass, and the cache-control cross-check
found literally zero extra DRAM read traffic rather than merely "small"),
which is a structural reason to expect it survives whole-pass contact
better than split-K did. It is not proof. The orchestrator's own
`scripts/bench.py` whole-pass confirmation is the actual arbiter here, more
than usual for this specific kernel.

## Orchestrator whole-pass confirmation (2026-08-18)

Merged (`cef45aa`/`eb2c813`/`c302f74` cherry-picked onto main), all four
gates re-verified on both models in both toggle states after merging on
top of the barrier-fusion beam (also landed the same session, first time
the two were tested together) -- identical to the documented baselines to
the last digit, `model_check` re-run with a genuinely 140-token prompt
after catching my own first attempt silently falling back to the 5-token
default (wrong flag, `--prompt` instead of `-p`; caught before trusting
the result, not after).

`scripts/bench.py`, two clean-window pairs (`util between` 0-2% every
round both sides, this session's own diagnostic column, not just my
`nvidia-smi` polling), toggle off vs `PHOBOS_QMMA_NARROW=1`:

| model | test | off | on | delta |
| --- | --- | --- | --- | --- |
| minicpm5-1b-Q8_0 | pp128 | 0.80x | 0.91x | **+0.11** |
| minicpm5-1b-Q8_0 | pp512 | 0.58x | 0.59x | +0.01 (flat) |
| Qwen3.5-0.8B-Q8_0 | pp128 | 0.83x | 0.87x | **+0.04** |
| Qwen3.5-0.8B-Q8_0 | pp512 | 0.78x | 0.73x | **-0.05** |

**pp128, the beam's actual target, is a real win on both models**,
confirming the whole-pass survival this beam's own reliability caveat
called out as unproven -- unlike split-K, this design's whole-pass number
moves the same direction as its per-kernel numbers.

**pp512 surfaces a real regression on Qwen the beam's own kill-check
ladder never tested** (it profiled only the four `pp128`-shaped launches).
The mechanism traces to the eligibility gate itself:
`q8_qmma_narrow_eligible`'s `unsplit < Q8_QMMA_SPLIT_THRESHOLD` check
computes `unsplit` from `rows`, and `pp512`'s `rows=512` (4x `pp128`'s 128)
multiplies every shape's block count by 4x before the gate is checked.
Minicpm's o_proj/down_proj-sized shapes (`n=1536`) land at
`unsplit = 4 * (1536/128) = 48`, exactly at `Q8_QMMA_SPLIT_THRESHOLD`
(the gate is a strict `<`, so this declines -- flat result, correctly).
Qwen's smaller `n=1024` shapes land at `unsplit = 4 * (1024/128) = 32`,
still under the threshold, so narrow still engages -- but at a grid this
much less starved, halving warps/CTA (the mechanism's actual cost, per
the beam's own named alternative hypothesis) apparently costs more than
doubling SM coverage still buys back. The fixed SM-count threshold that
works well for genuinely starved `pp128`-scale grids does not degrade
gracefully as a shape approaches it from below.

**Decision: keep opt-in only, do not promote to default-on.** The
toggle-off path is untouched by any of this (confirmed identical to
baseline above), so nothing shipped regresses by landing this beam. But
the current fixed threshold is not yet safe to flip on by default across
both `pp128` and `pp512` scales on every model. A follow-up beam should
either tighten the threshold (a smaller cutoff than the SM count, or a
ratio-based check relative to how starved the grid actually is rather
than a flat `< SM count`) or make the gate row-count-aware directly,
before this is reconsidered for default-on.

## Files

- `ncu_narrow_off_light.ncu-rep` / `ncu_narrow_on_light.ncu-rep` -- light
  metrics (grid, block, regs/thread, occupancy, duration), all four shapes,
  3 repeats, `PHOBOS_QMMA_NARROW=0`/`1`.
- `ncu_narrow_full_qkv.ncu-rep`, `_oproj.ncu-rep`, `_downproj.ncu-rep` --
  `--set full`, one launch each, `PHOBOS_QMMA_NARROW=1`, the register/
  Block-Limit confirmation.
- `ncu_narrow_dram_oproj_baseline_all.ncu-rep`, `_baseline_none.ncu-rep`,
  `_narrow_all.ncu-rep`, `_narrow_none.ncu-rep` -- the cache-control
  cross-check, o_proj shape, baseline vs narrow x `--cache-control all`/`none`.
