# q8_qmma K-loop software pipeline (A-operand prefetch)

Beam: reduce `q8_qmma`'s own per-block memory latency exposure by
software-pipelining the K-loop inside `qmma_t`'s tile-codegen builtin
(`phobos-lang/src/codegen/tile/qmma.rs`), mirroring the one-deep
register pipeline this session already built for `warp_partial`
(`phobos-lang/src/codegen/tile/warp_attn.rs`, see
`autoresearch/beams/warp-partial-prefetch.md`). Unlike the split-K beam
(`autoresearch/beams/q8-qmma-split-k.md`), this touches `phobos-lang`
directly and sits on the shipped default path for every real prefill
projection shape in every model -- no opt-in flag, no grid-shape change.

## Result summary

**Negative result: the register kill-check cleared, but real timing
evidence kills it.** Registers/thread came in at 254 (predicted ~254,
confirmed on the digit), under the 256 cliff, and achieved occupancy held
exactly flat against baseline. But every one of the four real prefill
shapes measured *slower* after the change, by 8.8% to 42%, in tight,
non-overlapping distributions across 24-48 launches each. The mechanism
section below explains why: the explicit loop-carried prefetch consumed
register headroom `ptxas` was apparently already using for its own
scheduling, and added a serialized address-computation dependency to the
loop's critical path, without producing the latency-hiding the change was
built for -- the `long_scoreboard` stall ratio measured *higher* after the
change, not lower, at identical occupancy.

**Recommendation: do not commit.** The diff is preserved below/in the tree
for review, but this beam should not ship. Revert
`phobos-lang/src/codegen/tile/qmma.rs`.

## Register kill-check (the hard gate, checked first)

Measured via `ncu.bat --metrics launch__registers_per_thread,
launch__registers_per_thread_allocated,launch__occupancy_limit_registers,
sm__warps_active.avg.pct_of_peak_sustained_active,
l1tex__t_bytes_pipe_lsu_mem_local_op_{ld,st}.sum`, minicpm5-1b-Q8_0,
`pp128`, `PHOBOS_QMMA_SPLIT=0`, GPU verified idle first (`nvidia-smi`: 2%
util, clocks at idle floor, no lingering `tasklist`/`Get-Process` hits).
Same-session baseline re-measured fresh (not reused from the split-K
beam), aggregated over every `q8_qmma` launch in one `pp128` pass:

| grid (shape) | regs/thread used | regs/thread allocated | occupancy limit (blocks/SM) | achieved occupancy | local mem spill |
| --- | --- | --- | --- | --- | --- |
| baseline, all shapes | 238 | 240 | 2 | 12.50% (grid 12/20), 19.11% (grid 72) | -- |
| after, all shapes | **254** | **256** | 2 (unchanged) | 12.50% (grid 12/20), 19.07% (grid 72) | **0 bytes both directions** |

- `launch__registers_per_thread`: 238 -> 254, **+16**, matching the task
  brief's hand-traced prediction (`rm=8, halves=2` -> 16 A fragments)
  almost exactly.
- `launch__registers_per_thread_allocated` (the number that actually
  drives the occupancy math): 240 -> 256. Both round up to the same
  2-blocks/SM regime (`65536 / 128 / 240 = 2`, `65536 / 128 / 256 = 2`
  exactly), so `launch__occupancy_limit_registers` and achieved occupancy
  are unchanged, and there is **zero register spill to local memory**
  either direction.
- The gate's literal condition (regs/thread <= 256, occupancy not worse
  than baseline) **passed**. But note the allocated count moved from
  240/256 (16 registers of slack within the 2-block regime) to exactly
  256/256 (zero slack) -- this detail turns out to matter for the
  mechanism, see below.

Evidence: `ncu_qmma_kloop_baseline.ncu-rep`/`.csv`/`.log` (fresh baseline,
all shapes, light metrics), `ncu_qmma_kloop_after.ncu-rep`/`.csv`/`.log`
(same, after the change), `ncu_qmma_kloop_after_spill.ncu-rep`/`.csv`
(spill check).

Per this beam's own standing instructions, clearing this gate was licence
to proceed to timing evidence, not licence to skip it.

## Per-shape timing: every real shape regressed

Same `ncu` capture as the register table above, `gpu__time_duration.sum`
per launch, aggregated across all layers in one `pp128` pass (minicpm has
enough layers to give 24 launches for the two shapes that appear once per
layer and 48 for the one grid size two shapes share). o_proj and
down_proj share grid `(1, 12, 1)` (both `n=1536`) and were split by launch
parity within that group (`o_proj, down_proj, o_proj, down_proj, ...` --
confirmed by the resulting medians landing almost exactly on the split-K
beam's independently-measured baseline for the same two shapes, 90.7us/
191.8us there vs 89.1us/190.3us here, which also serves as a
same-session cross-check that this fresh baseline isn't contaminated).

| shape | grid | baseline median (n) | after median (n) | delta |
| --- | --- | --- | --- | --- |
| o_proj (n=1536, k=2048) | (1,12,1) | 89.06us (24) | 115.38us (24) | **+29.5%** |
| down_proj (n=1536, k=4608) | (1,12,1) | 190.30us (24) | 270.32us (24) | **+42.0%** |
| qkv (n=2560, k=1536) | (1,20,1) | 71.34us (24) | 100.08us (24) | **+40.3%** |
| gate/up (n=9216, k=1536) | (1,72,1) | 138.67us (24) | 150.88us (24) | **+8.8%** |

Distributions are tight and do not overlap (e.g. qkv: baseline range
70.24-72.83us, after range 95.20-102.21us across 24 samples each) -- this
is a real, repeatable regression, not noise. Every shape got worse; the
regression is *worst* on down_proj (grid=12, 12.5% occupancy, `k=4608`,
the deepest of the four K-loops, +42%) and *smallest* on the one shape
with the highest baseline occupancy (gate/up, 19.1% achieved, only
+8.8%) -- consistent with the mechanism below: the regression tracks
inversely with occupancy and directly with K-loop trip count, since a
deeper loop pays the per-iteration overhead this change adds more times.

## Mechanism: the intended latency-hiding did not happen

`--set full` on the first `q8_qmma` launch of the same `pp128` pass
(qkv shape, grid `(1,20,1)`), before and after, same-session, GPU
reconfirmed idle before each capture:

| metric | before | after |
| --- | --- | --- |
| `gpu__time_duration.sum` | 71.90us | 94.94us |
| `launch__registers_per_thread` | 238 | 254 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 12.44% | 12.54% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 0.750 | **1.231** |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.020 | 0.015 |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0 | 0.053 |
| `smsp__average_warp_latency_per_inst_issued.ratio` | 3.39 cycles | 4.53 cycles |
| `smsp__inst_executed.sum` | 2,950,480 | 3,027,920 (+2.6%) |

The `long_scoreboard` stall ratio (global-memory-latency stall) went
**up**, not down, at essentially identical achieved occupancy (12.44% vs
12.54%). This is the same metric the split-K beam flagged as
occupancy-sensitive (it mechanically rises when more warps become
resident, even while latency hides better) -- but that caveat does not
apply here, because occupancy is unchanged before and after. A
higher `long_scoreboard` ratio at *constant* occupancy is a genuine
worsening of exposed memory latency, not a measurement artifact.

Instruction count rose only 2.6% (the extra address arithmetic and
prologue/epilogue loads this change adds), which does not explain a
32-34% duration increase or a 64% jump in the stall ratio on its own.
The more likely story, supported by the register-allocation numbers
above: the *unmodified* K-loop has no loop-carried dependency on its
operand loads at all (`a_frags`/`w_frags` are freshly computed from the
loop induction variable every iteration, with no register threading
between iterations), which leaves `ptxas` free to hoist and interleave
loads across iterations on its own, using whatever of the 240/256
allocated-but-unused registers it needs to do so. This beam's explicit
one-deep pipeline forces a *specific* one-iteration-ahead schedule
through `iter_args`, consumes those same registers itself (240/256 slack
-> 256/256, zero slack), and adds a genuine chain dependency on the
address path (`next_k = min(k + step, kd - step)`, then two more `addi`s
per fragment) that the original code's freely-hoistable loads did not
have. In other words: this kernel's compiler backend may already have
been doing an equivalent or better job of hiding the load latency on its
own, using register headroom this change then took away, and replaced
with a strictly worse, hand-scheduled version. This is consistent with
the cross-shape gradient above (the shapes with the least occupancy
headroom and the most loop iterations to pay the per-iteration tax on
regressed the most).

Evidence: `ncu_qmma_kloop_full_qkv_before.ncu-rep` /
`ncu_qmma_kloop_full_qkv_before_check.csv` /
`ncu_qmma_kloop_full_qkv_before_check2.csv` /
`ncu_qmma_kloop_full_qkv_before_check3.csv`,
`ncu_qmma_kloop_full_qkv_after.ncu-rep` /
`ncu_qmma_kloop_full_qkv_after_check.csv`.

## What was built

Mirrors `warp_partial`'s one-deep register software pipeline shape,
inside `qmma_t_into`'s `kb`/`tb` loop structure
(`phobos-lang/src/codegen/tile/qmma.rs`):

- A prologue (in `tb`, before the `scf.for` is built) loads the first
  iteration's raw (uncast) i8 vec4 A fragments at `k=0`.
- The K-loop's `iter_args` now carry those raw A fragments alongside the
  existing `lanes` f32 accumulators (`args`/`operands`/`result_types` all
  extended by `pf = rm * halves` slots of `vec4_i8`).
- Each iteration issues the *next* iteration's raw A load first (before
  touching the carried-in current value), addressed via
  `next_k = min(k + step, kd - step)` -- the same branch-free clamp
  `warp_partial` uses for its own loop's final row, always in-bounds since
  `kd` is guaranteed a whole number of `Q8_BLOCK`-sized blocks.
- The carried-in raw A fragment is cast to the `mma_sync` fragment shape
  once per iteration, on the value that just became "current" (not stored
  widened in the loop-carried state).
- Only A is prefetched, not W -- the two operands are register-symmetric
  (same fragment count each), and a full A+W double buffer was
  hand-estimated at +32 registers (238 -> ~270), certain to cross the
  256-register cliff; the A-only variant's actual +16 (238 -> 254) came
  in exactly where predicted.

Only `phobos-lang/src/codegen/tile/qmma.rs` changed. 374 lines (cap is
900; the file was well under it already).

## Emit diff-sweep

Ran `cargo run -p phobos-lang --example emit` over every `.ph` file under
`examples/` and `phobos-lang/examples/` (10 files), under both the
default target and `PHOBOS_CHIP=sm_80 PHOBOS_INDEX_BITS=64`, before (via
`git stash` on just `qmma.rs`) and after this change. Diffed every
resulting `.mlir` file individually (`diff -q`, not assumed from
`git diff --stat`, since this change is inside `phobos-lang` itself and
the task flagged that as not automatically zero the way the split-K
beam's was): **zero differences**, all 20 file/target combinations. No
`.ph` file in the tree calls `qmma_t` directly, so this is an expected
empirical no-op, now actually verified rather than assumed.

`cargo test -p phobos-lang`: **155/155**, unchanged count from before this
change.

## Correctness

`cargo build -p phobos-lang` and `cargo build --release -p phobos-gguf
--features cuda --example bench` (plus the four gate examples): clean.
`cargo clippy --release -p phobos-lang -- -D warnings`: clean
(`phobos-lang` has no `cuda` feature to gate on).
`cargo clippy --release -p phobos-gguf --features cuda -- -D warnings`:
clean.

All four standing gates, both models, `--release --features cuda`:

- **backend_check**: both models, all ops `ok`, worst relative error
  `2.902e-4` (identical on both models -- this op-level check doesn't
  touch the changed deep-tile path differently per model).
- **batch_check**: both models, `batched and sequential agree`. Worst gpu
  spread error: minicpm `1.350e-2` (batches of 100), Qwen `1.540e-2`
  (batches of 512).
- **model_check**: both models, default short prompt (`--single`-adjacent,
  ~5 tokens, does not reach the deep tile) and a fresh ~180-token prompt
  (confirmed via the `encode` example: the 6x-repeated test sentence
  tokenizes to well over 128 tokens on minicpm's tokenizer, comfortably
  past the `M=128` deep-tile threshold that the split-K beam's own
  postmortem flagged as necessary to actually exercise this path). All
  four runs `backends agree`:
  - minicpm, short prompt: spread err `1.258e-2` / `1.335e-2` / `1.763e-2`
    (tied) over 3 steps.
  - minicpm, ~180-token prompt: spread err `1.270e-2` / `9.792e-3` /
    `8.689e-3` over 3 steps.
  - Qwen, short prompt: spread err `1.019e-2` (tied) / `7.097e-3` (tied) /
    `1.074e-2` over 3 steps.
  - Qwen, ~180-token prompt: spread err `9.688e-3` / `8.621e-3` /
    `7.932e-3` over 3 steps.
- **fuse_check**: both models, `the prompt pass agrees exactly` in both
  cases, 0 top-token flips over 32 decode steps. minicpm: launched/fused
  at most `1.332e-2` of the logit spread apart (step 31), `9.944e-3`
  average. Qwen: at most `1.051e-2` apart (step 11), `8.085e-3` average.

All spread-error magnitudes are in the same band this session's other
beams already documented as the pre-existing host/device noise floor for
this kernel family (split-K beam: `1.1e-2` to `7.4e-3`; warp_partial beam:
similar), not a new source of disagreement. **Correctness holds
throughout** -- this beam's failure is purely a performance regression,
not a correctness one.

## Current state

`phobos-lang/src/codegen/tile/qmma.rs` carries the diff described above.
**Recommend reverting it** -- every one of the four real shapes this
kernel serves measured slower by 8.8-42%, at constant occupancy, with a
stall-ratio increase that contradicts the intended mechanism. The
register/occupancy gate this beam was built to respect passed on its own
terms (254 <= 256, occupancy unchanged, no spill), which is exactly why
the timing evidence -- not the register count -- is what should decide
this one. I don't run `scripts/bench.py` myself, per the task's standing
rules; the numbers above are `ncu`-isolated `gpu__time_duration.sum` on
real shapes, same convention as this session's other beams.
