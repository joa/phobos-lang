# Beam: register-level K/V prefetch inside warp_partial

## Diagnosis this acts on

After this session's grid-barrier-rebalance fix (`[[cache-length-split-buckets]]`'s
Round 3), `attention_persist` sat at 98.4% occupancy with barrier stall down
to ~22-25% of stall cycles (was 42.7%) and `long_scoreboard` (global memory
latency) the largest remaining category at ~27-31%. At maxed occupancy, more
warps cannot hide that latency further -- only more independent in-flight
loads *per warp* can. Direct read of `warp_partial`'s codegen
(`phobos-lang/src/codegen/tile/warp_attn.rs`, the `carry_loop` closure)
confirmed a strict load-then-use pattern: each iteration loads this warp's
K/V slice for key position `kt`, then immediately consumes it, no overlap
with the next iteration's load. sm_75 (this card, Turing) has no hardware
`cp.async`, so any fix is software pipelining via early load issue, not a
hardware async-copy primitive.

## What was built

One-deep register-level software pipeline in `carry_loop`'s closure: a
prologue loads the first row's raw (unwidened) K/V vectors; each loop
iteration issues the *next* row's raw load before widening/consuming the
row carried in via `iter_args` from the previous iteration. Both K and V
are prefetched (the cheaper "raw f16 vectors, widen once consumed" variant
worked; the K-only fallback was never needed). Addressing is branch-free:
`min(kt+1, ghi-1)` clamps both the empty-warp-range case and the loop's
final iteration into always-valid rows without a conditional. Only
`phobos-lang/src/codegen/tile/warp_attn.rs` changed (94 insertions, 20
deletions) -- no other file in the tree.

`@pipeline` (the DSL-level statement-lowering pass) was deliberately not
used: it only fires on `.ph`-level for-loops staging *shared-memory* tile
slices in a specific `var t = <static slice>` shape, and `warp_partial` is
hand-written Rust codegen loading straight into *registers* -- a different
mechanism the DSL pass structurally cannot reach.

## Register/occupancy/spill -- checked before any benchmark, as instructed

The whole risk of this change is register pressure: the kernel sits at
exactly 64 registers/thread with `Block Limit Registers = 4` **binding**
the occupancy ceiling, already at 98.4% achieved occupancy, so any growth
risks a 25% occupancy cliff.

- `launch__registers_per_thread`: **64, unchanged**
- `launch__occupancy_limit_registers`: **4 blocks, unchanged**
- Achieved occupancy: **~98.3-98.8%**, matching baseline (independently
  reverified by the orchestrator: 98.84% via `sm__warps_active.avg.pct_of_peak_sustained_active`,
  `autoresearch/beams/ncu_persist_prefetch_4099.ncu-rep`)
- `l1tex__t_bytes_pipe_lsu_mem_local_op_{ld,st}.sum`: **0 bytes both
  directions** -- no spilling to local memory

`ptxas` absorbed the extra raw-vector state within the existing 64-register
footprint. The cheaper variant (raw f16, not widened-f32) was the right
call: a naive widened-f32 double-buffer would have cost +8 registers on a
kernel with zero headroom.

## Mechanism confirmed, with an honest twist

`ncu` stall breakdown on `attention_persist` (minicpm, cache 4099,
`--set full`): `long_scoreboard` dropped from ~31.6% share (4.70 cycles) to
~7.5% (0.91-0.96 cycles, orchestrator's independent capture matches) -- the
prefetch does exactly what it was built to do. But `barrier` stall's share
rose to absorb it (24.1% -> 35.6%, orchestrator's capture: 4.62 cycles),
and per-kernel `gpu__time_duration.sum` barely moved (29.72us -> 29.59us,
~0.4%, noise).

**Reading**: `attention_persist`'s actual critical path is `grid_barrier`
cross-block skew (the already-diagnosed bottleneck from the grid-barrier-
rebalance round), not `warp_partial`'s own memory latency. Freeing up
long-scoreboard stall inside `warp_partial` doesn't shorten a kernel whose
wall-clock is set by the slowest block reaching the barrier, not by its
own issue-stall mix. This is not a failure of the diagnosis -- the
mechanism this beam targeted is real and was closed exactly as predicted
-- it is a case of two real bottlenecks stacked on the same kernel, and
closing the second-largest one first (this beam) doesn't move wall-clock
until the largest one (the barrier) is closed further too.

## Result: flat on the targeted kernel, a real free win elsewhere

Isolated `attndecode` at cache 4099, 5 reps each, orchestrator-verified in
a fully isolated `git worktree` (own `CARGO_TARGET_DIR`, branched from
clean `de5ebf7`, immune to a concurrent agent's in-flight edits to
`phobos-lang/src/codegen/{pipeline.rs,stmt.rs,mod.rs}` elsewhere in the
tree at the time):

| | before | after | delta |
| --- | --- | --- | --- |
| minicpm `attention_persist` | 622.9us | 624.1us (agent) / 627.6us (orchestrator recheck) | flat (noise) |
| minicpm `attention_split` | 710.5us | 662.5us (agent) / 674.0us (orchestrator recheck) | **-6 to -7%** |
| Qwen `attention_persist` | 218.1us | 194.9us (agent) / 199.2us (orchestrator recheck) | **-9 to -11%** |
| Qwen `attention_split` | 216.4us | 197.0us (agent) / 204.9us (orchestrator recheck) | **-8 to -9%** |

`attention_persist` is minicpm's default decode path (the shape fits the
persistent kernel and `PHOBOS_ATTN_PERSIST=1` is this session's established
config) -- flat there, as the mechanism above explains. But
`attention_split` is what minicpm falls back to whenever persist is off or
declines, and it is **Qwen's only decode-attention path, always** --
`attn_persist_plan` declines Qwen's shape unconditionally (`[[flash-attention-decode]]`).
So this beam is a real, free, zero-register-cost win for Qwen's actual
production decode-attention kernel, and for minicpm whenever the
persistent path isn't engaged.

## Correctness

`cargo build --release -p phobos-gguf --features cuda --examples`:
clean. `cargo clippy --release -p phobos-gguf --features cuda --examples
-- -D warnings`: clean. `cargo test -p phobos-lang`: 151/151, including
`warp_partial_vectorizes_kv_loads_at_dpl_4`/`_8` unchanged. All four
standing gates (`backend_check`, `batch_check`, `model_check`,
`fuse_check`) clean on both `minicpm5-1b-Q8_0` and `Qwen3.5-0.8B-Q8_0`
with `PHOBOS_ATTN_PERSIST=1`, matching documented baselines to the digit
-- verified independently by both the dispatched agent and the
orchestrator, in the same isolated worktree.

No `.ph` example exercises `warp_partial` directly (it is reached only
through the GGUF device backend's decode-attention kernels, not named by
any `.ph` file in the tree), so the CLAUDE.md emit-diff-sweep requirement
has nothing to diff for this change; the GGUF-side correctness gates are
the applicable verification here.

## Confirmation benchmark

`bench.py`, both models, `PHOBOS_ATTN_PERSIST=1`, isolated worktree,
`autoresearch/beams/warpprefetch_confirm.csv`/`.json`:

| model | tg1024 | tg2048 | tg4096 |
| --- | --- | --- | --- |
| minicpm5-1b | 1.00x | 1.00x | 0.99x |
| Qwen3.5-0.8B | 1.12-1.14x | | |

minicpm flat, exactly as the mechanism predicts (its tracked `tg` path
uses `attention_persist`). Qwen's round 1 had a contaminated sample
(phobos's own tg1024 read 257.79 t/s against 294-299 t/s in rounds 2-3,
a clear transient-contention outlier, not a regression) inflating the
reported stderr and pulling the headline ratio down from this session's
recent 1.15-1.16x baseline; still solidly positive and consistent with
no regression. Qwen's attention is only 6 of 25 layers doing full KV
attention (the rest are SSM/delta-rule state, see `[[cache-length-split-buckets]]`'s
architecture table), so `attention_split`'s real ~9% kernel-level win
dilutes into a small, easily noise-dominated `tg` effect at this sample
size -- consistent with, not contradicting, the clean isolated
measurements above.

## Process note: a second, distinct instance of the shared-target-dir /
concurrent-agent-in-same-tree contamination this session already banked

This beam's own agent independently rediscovered and applied the
worktree-isolation lesson from the grid-barrier-rebalance round (a
`git stash` on just its own file still left the shared tree's other files
at a concurrent agent's mid-flight, uncommitted state, risking a confounded
comparison even for a diff that never touched those files) without being
told to -- it built `../phobos-wt-warpprefetch` with its own
`CARGO_TARGET_DIR` unprompted. Separately, the orchestrator hit live GPU
contention (not a build-artifact race this time, but two processes -- the
agent's own gate run and the orchestrator's `attndecode` timing check --
racing for the same physical GPU inside that same isolated worktree) that
produced one clearly-outlier reading (1807.6us against an expected
~626-691us range) before a clean rerun confirmed the real number. Worth
generalizing: worktree isolation solves the *build-artifact* race, but the
GPU itself is one physical resource no worktree can isolate -- the same
"check `tasklist` before trusting a timing number" discipline `bench.py`
already needed applies to `attndecode`/`ncu` runs too, not just `bench.py`.
