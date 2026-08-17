# q8_qmma proper stream-K (boundary fixup, not full-plane split-K)

Beam: build real stream-K for `q8_qmma`'s starved deep-tile grids (the same
shapes `autoresearch/beams/q8-qmma-split-k.md` targeted), this time assigning
contiguous ranges of total K-work across the grid so most output tiles stay
single-owned (whole `k`, one direct write, no partial, no reduce), and only a
boundary tile that a work-assignment split gets a small `[TM, TN]`-scale
fixup, not the prior beam's `S` full-output-sized scratch planes. Full design
context and the prior beam's postmortem (why it shipped default-off: 6.3 MB
of partials-traffic-per-shape contending for DRAM/L2 bandwidth with the whole
pass, not just the routed kernels) are in `q8-qmma-split-k.md`; not repeated
here except where this beam's own measurements bear on it directly.

## Status: kill-check run, clean negative result -- beam stops here

**Headline: the two-way probe's extra DRAM traffic is roughly 172 KB per
output tile, about 1.7x the postmortem's ~100 KB/launch bar from a single
tile's boundary fixup alone**, before any accounting for how many tiles a
real starved shape would actually need split (see "Whole-shape total" and
"Is the full multi-way design likely to clear the bar" below: no, and the
reasoning says it would cost *more* per tile, not less). This beam stops
here per the task's own standing instruction that a clean negative result at
this checkpoint is the valued outcome, not a reason to keep pushing.

Everything under "What was built" through "De-risking the slot" below is the
design and validation that got to this measurement; the numbers themselves
are in "DRAM kill-check result" further down.

## What was built

A minimal two-kernel instance of the boundary-fixup mechanism, deliberately
not the full multi-tile work-assignment scheme yet: an unconditional 50/50
split of `q8_qmma`'s deep tile's own `k`, one shape at a time, gated behind
`PHOBOS_QMMA_STREAMK=1` (default off, independent env var from
`PHOBOS_QMMA_SPLIT`, mutually exclusive with it in `project_q8`'s dispatch --
streamk is checked first). The point of building this first, per the task's
own instruction, is to get a real `ncu` number for the boundary-fixup
mechanism's DRAM cost before investing in the full assignment scheme that
decides *which* tiles need it.

- `phobos-gguf/src/backend/device/kernels/quant.rs`:
  - `q8_qmma_sk_lo_src(block, tm, tn, half)` -- the low half of the split:
    `C[...] = qmma_t(A[..., 0 :+ half], ...)`, structurally identical to the
    existing unsplit `q8_qmma_src` except the tile spans half of `k` at a
    literal offset. Confirmed via `cargo run -p phobos-lang --example emit`
    on a throwaway `.ph` (kept nowhere permanent, this description is the
    record) that this takes `qmma_t`'s direct-to-global write path exactly
    like the unsplit kernel: zero `scf.if`, zero `memref.global` (no shared
    tile at all) in the emitted body, one barrier (from `qmma_t_into`
    itself, same as baseline).
  - `q8_qmma_sk_hi_src(block, tm, tn, half)` -- the high half:
    `C[...] += qmma_t(A[..., half :+ half], ...)`. `+=` with `qmma_t` on the
    right does **not** take the direct-write path -- that special case in
    `phobos-lang/src/codegen/stmt.rs`'s `store_tile` is gated on
    `op == AssignOp::Set` only. So this falls through to the generic
    `Rv::Tile` path: `qmma_t(...)` evaluates via `tile_qmma_t`, which
    allocates a **fresh shared tile** (`alloc_tile_shaped`, always shared
    memory, see `phobos-lang/src/codegen/tile/alloc.rs`) the same size as
    the destination slice, writes into that via the same `qmma_t_into`
    fast path, and then the assignment machinery does a plain
    `tile_binary(Add, target, src, target)`: load the destination's current
    value from global memory, add the shared tile, store back. Confirmed via
    the same `emit` sweep: the high kernel's body is exactly this --
    `vector.load` on the destination subview, `vector.load` on the shared
    tile, `arith.addf`, `vector.store` back to the destination subview. This
    load-add-store on the real output buffer, not a separate scratch
    plane, is the mechanism this beam is measuring the cost of, and it is
    real DRAM traffic (a genuine read of already-computed output plus a
    second write), not merely a shared-memory round trip.
  - Two kernels rather than one launch with an `if` branch, because the high
    half's add has to observe the low half's write, and that ordering only
    holds across a kernel boundary on the same CUDA stream -- not across
    concurrently scheduled blocks within one launch, which is what an
    in-launch branch would be.
- `phobos-gguf/src/backend/device/matmul.rs`: `launch_qmma_streamk`, wired
  into `project_q8`'s deep-tile dispatch ahead of the existing
  `qmma_split`/`splits > 1` branch (checked first, so the two mechanisms
  never both fire for the same call). Compiles and caches both kernels per
  `(wide, half)`.
- `phobos-gguf/src/backend/device/mod.rs`: `q8_qmma_streamk` module cache
  (`HashMap<(usize, usize), (Module, Module)>`, keyed by output width and the
  low half's `k`) and a `qmma_streamk: bool` field, `PHOBOS_QMMA_STREAMK=1`
  to opt in (`0`/unset/anything else stays off, matching `qmma_split`'s own
  convention, not `PHOBOS_ATTN_PERSIST`'s default-on one -- this is an
  unvalidated probe, not a tuned default candidate).

## A second compiler-proof hazard, caught the same way the first beam caught
## its own (driver-level, not `emit`/`ptx`-level)

First build used the caller's own column width (`wide`, which is 128 for any
`n` divisible by 128, e.g. `qkv`/`gate_up`-shaped calls) for *both* kernels.
`backend_check --features cuda` under `PHOBOS_QMMA_STREAMK=1` got through
every non-quant check and 30 `matmul_quant` cases, then failed hard:

```
Error: loading q8_qmma_sk_hi PTX
Caused by:
    "a PTX JIT compilation failed"
```

Same failure mode as `q8-qmma-split-k.md`'s own "Reduce kernel, second bug":
`emit`/`ptx` both stop before `Module::from_ptx`, so neither ever sees this.
Root cause here is analogous but not identical -- not a plain tile add
(that beam's bug), but `tile_qmma_t`'s own shared-tile allocation for the
`+=` fallback: at `TM = 128, TN = wide = 128` that shared accumulator is
128*128*4 = 64 KB, over the 48 KB static ceiling this kernel is compiled
under. Fixed by decoupling the high half's column width from the low half's:
the low half keeps the caller's `wide` (its direct-write path has no shared
tile at all, so no ceiling to hit), and the high half is hardcoded to
`Q8_QMMA_TN` (64, a 32 KB tile) with its own grid over `n`, regardless of
what the low half used. `Q8_QMMA_TN` divides every width `qmma_width` can
return by construction (`Q8_QMMA_WIDTHS = [128, 64]`), so this needs no new
divisibility guard.

## Correctness spot check (not yet the four standing gates)

`PHOBOS_QMMA_STREAMK=1 cargo run -p phobos-gguf --release --features cuda --example backend_check`:
all cases pass after the fix above, including the one that exercises the
deep tile with the streamk path live: `matmul_quant [128 x 1024 x 2048]`
(`M = 128 = Q8_QMMA_TM`, `N = 2048`, divisible by 128, so `wide = 128`,
`splits` via the old mechanism would not even apply here -- this is purely
the new unconditional-50/50 probe path). Worst relative error across the
whole suite: **2.871e-4**, in the same neighborhood as the documented
`backend_check` baseline (2.902e-4) despite this being a minimal probe, not
a tuned or fully-generalized design. This is not one of the four standing
gates and is not a substitute for them -- it is a smoke check that the
mechanism (two launches, direct-write low half, load-add-store high half)
is arithmetically sound before spending a GPU timing slot on it. The four
standing gates on both models, plus `model_check`/`fuse_check` at the real
routed shapes, are pending the same slot as the kill-check itself (a value
comparison could run under contention per the task's protocol, but running
it before the kill-check clears would be measuring correctness of a design
that may not survive the bandwidth bar at all).

## De-risking the slot: real routed shapes JIT-compile and agree (value
## check, run under contention, no slot needed)

The `backend_check` smoke check above only proves one `(wide, half)`
combination survives `Module::from_ptx` (`wide=128, half=512`). Every real
routed shape JIT-compiles its own fresh module pair at first use and none of
minicpm's or Qwen's had been through the driver yet -- and this session has
now hit two driver-level PTX rejections that neither `emit` nor `ptx` catch
(one from the prior beam, one from this one, above), so this was worth
ruling out before spending a timing slot on a build that might still fault
mid-capture. Ran both models' `model_check` with a 128+-token prompt (same
kind of check `q8-qmma-split-k.md` used for its own routed-shape validation),
`PHOBOS_QMMA_STREAMK=1`, under contention (a value comparison, allowed
without a slot per the task's protocol):

- **minicpm**, 128+-token prompt: `step 0: spread err 1.390e-2`, `step 1:
  1.173e-2`, `step 2: 1.040e-2` (steps 1-2 tied top token) -- `backends
  agree`. Same order of magnitude as `q8-qmma-split-k.md`'s own routed-shape
  spread errors (1.1e-2 to 7.4e-3), unsurprising since both are the same
  kind of host/device summation-order noise on real Q8_0 projections, not a
  new error source.
- **Qwen3.5-0.8B**, same prompt: `step 0: 8.841e-3` (tied), `step 1: 1.958e-2`
  (tied), `step 2: 8.395e-3` -- `backends agree`.

Both loaded, JIT-compiled every shape they routed through streamk, and
matched the host reference. The slot is safe to spend on timing.

## DRAM kill-check result

GPU slot granted by the coordinator; `nvidia-smi` showed no CUDA compute
process (only two sibling agents' `cargo` builds), ~34% util / ~765 MHz from
background desktop apps -- a known non-compute pattern on this box, and
irrelevant regardless since `dram__bytes_*.sum` is a byte count intrinsic to
the kernel's own execution, not a wall-clock number.

Measured on `backend_check`'s existing `matmul_quant [128 x 1024 x 2048]`
case (`M=128=Q8_QMMA_TM`, `K=1024`, `N=2048`, `wide=128`, 16 output column
tiles), the only shape in that binary that reaches the deep tile with a full
128-row grid, via `ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum`,
isolated with `-k "regex:^q8_qmma$"` (baseline) and
`-k "regex:^q8_qmma_sk_..$"` (probe: `q8_qmma_sk_lo` + `q8_qmma_sk_hi`, both
launches, same shape), captured at both `--cache-control all` (ncu's
default, flushes caches between kernels) and `--cache-control none` (leaves
L2 as a real back-to-back launch pair on one stream would see it):

| | read | write | total |
| --- | --- | --- | --- |
| baseline, `--cache-control all` | 2.56 MB | 1.59 MB | 4.15 MB |
| probe (lo+hi), `--cache-control all` | 1.27+2.45=3.72 MB | 1.45+1.74=3.19 MB | 6.91 MB |
| **extra, `--cache-control all`** | **+1.16 MB** | **+1.60 MB** | **+2.76 MB** |
| baseline, `--cache-control none` | 2.53 MB | 11.58 KB* | 2.54 MB* |
| probe (lo+hi), `--cache-control none` | 1.28+2.35=3.63 MB | 1.60+1.73=3.33 MB | 6.96 MB |
| extra read, `--cache-control none` | **+1.10 MB** | -- | -- |

\* Baseline's `--cache-control none` write is not a real number: with cache
flushing disabled, ncu's own measurement window can end before a kernel's
dirty L2 lines are evicted to DRAM, so a single write-only kernel's write
count is an artifact of when the profiler happened to stop watching, not the
true write traffic. This is why the write-side comparison uses
`--cache-control all` only (a mode that flushes deliberately, so it counts
everything); the `--cache-control none` capture exists specifically to
cross-check the *read* side, where the question is "does the high half's
read of the low half's output hit L2 instead of DRAM," and a deferred-flush
artifact does not corrupt that answer for reads that must complete within
the kernel's own execution to be usable by later instructions in the same
kernel.

**The L2-residency question is settled, and settled against the design**:
extra read is 1.16 MB flushed vs. 1.10 MB unflushed -- essentially identical.
L2 does not absorb the high half's read of the low half's write in practice,
even though the two kernels run back to back on the same stream with nothing
else scheduled between them. The mechanism is visible in the numbers
themselves: each kernel's own K-loop streams roughly 1.27 MB of weight data
through L2 while computing its half, and that streaming volume alone is
enough to evict a small (~1 MB) C-region write well before the high half's
final accumulate step gets around to reading it back. A production pass,
which streams far more weight data per step than this isolated two-kernel
probe, would evict it at least as fast.

**Per-tile normalization**: this probe's single launch pair splits all 16 of
the shape's output column tiles at once (`lo`'s grid is `(1, 16, 1)` at
`wide=128`; `hi`'s is `(1, 32, 1)` at `Q8_QMMA_TN=64`, same total output
region). Extra traffic per tile: **2.76 MB / 16 tiles is approximately 172
KB per tile** -- already about **1.7x the postmortem's ~100 KB/launch bar
from a single tile's own two-way boundary fixup**, before any accounting for
how many tiles a real shape needs split.

**Whole-shape total, for scale against the prior beam's design**: this
probe's 2.76 MB extra, for one M=128 deep-tile call at K=1024, N=2048, is
already about 44% of the prior split-K beam's own reported 6.3 MB/shape
figure (`autoresearch/beams/q8-qmma-split-k.md`) -- and that number used
`S=8` (an 8-way split with full-plane scratch), while this is only a 2-way
split with no scratch buffer at all, using the cheapest possible fixup
mechanism (direct load-add-store against the real output, not a separate
plane). The mechanism this beam built is real and is cheaper *per split* than
the prior beam's, exactly as intended -- but the per-tile cost is still far
above the bar the postmortem set, because the tile granularity itself
(`[Q8_QMMA_TM, Q8_QMMA_TN]` = `[128, 64]`, 32 KB in f32) is simply too coarse
for any boundary-touching design at this shape to move under 100 KB: a
single tile's own footprint is already about a third of the whole bar.

## Is the full multi-way design likely to clear the bar? No.

Two independent reasons point the same way, not one weak signal from a
simplified probe:

1. **The measured 2-way number is already ~1.7x over budget per tile**, and
   this is the *cheapest* non-trivial split (fewest boundary crossings, no
   scratch buffer, no reduce pass -- just one accumulate). Any design that
   splits a tile more ways pays this load-add-store cost more times, not
   fewer: an `S`-way split of one tile needs `S-1` accumulate passes each
   reading and writing that tile's full region, so per-tile extra traffic
   scales *up* with more splits, roughly linearly in `S` for the same
   mechanism. There is no version of "split more finely" that reduces this
   number; finer splitting is strictly worse per tile, not better.
2. **The starved shapes this beam exists to help cannot use few splits per
   tile.** See "A caveat on the design brief's own 'most tiles single-owned'
   framing" below: at `n=1536, k=2048` (12 output tiles, 48 SMs), a
   work-assignment chunk can be at most 16 k-blocks if all 48 SMs are kept
   busy, and a tile is 64 k-blocks deep, so every tile needs at least a
   4-way split to be covered at all -- not the 2-way this probe measured,
   and not "mostly whole tiles, a few boundaries." A 4-way split's
   load-add-store cost (3 accumulate passes per tile, not 1) would be
   measured, not assumed, but per point 1 it can only be *more* than this
   probe's already-over-budget 2-way number, never less.

Both point to the same conclusion without needing a second measurement to
confirm it: the full multi-tile work-assignment design is not a candidate
that a smarter implementation could still land under the bar. The tile
granularity (`Q8_QMMA_TM x Q8_QMMA_TN` fixed at 128x64/128x128, chosen for
tensor-core efficiency, not revisited here per the task's standing
instruction not to re-litigate the tile-shrink probe this session already
ran and rejected) is the binding constraint, and it binds regardless of how
the work-assignment scheme is tuned on top of it. This is not "needs more
slot time to prove it's fine" -- it is the number the checkpoint existed to
produce, and it says stop.

## A caveat on the design brief's own "most tiles single-owned" framing

Worth stating plainly before the measurement, not softened after it: for
`n=1536, k=2048` (12 output tiles, `Q8_QMMA_TM x wide` each, `k_blocks = 64`
per tile), there is no work assignment across up to 48 SMs where most tiles
keep a single owner. A block's assigned chunk of total K-work can be at most
`768 / 48 = 16` k-blocks if all 48 SMs are to be used, and `16 < 64` means
every tile's own depth exceeds one block's chunk -- every tile needs at
least `64 / 16 = 4` blocks to cover it, not zero or one. The design brief's
own illustrative example ("most output tiles ... are still owned
start-to-finish by a single block ... e.g. `n=1536` giving a 12-tile grid")
does not hold for the shape it names: a chunk can only exceed a tile's own
depth (avoiding any split at all) when the target block count is *smaller*
than 4x the tile count, which is a mild, not a starved, grid -- exactly the
opposite of the case this beam (and the prior one) exists to help. This
doesn't invalidate the boundary-fixup mechanism itself (a smaller, cheaper
partial than `S` full planes either way), but it does mean the realistic
outcome for these specific starved shapes is closer to "every tile split
some number of ways, each way cheaper than the old design's" rather than
"only a few boundary tiles pay anything" -- so the per-tile number from the
kill-check, not a best-case handful-of-boundaries number, is what a
generalized design here would actually cost, and that number times the
tile count is the honest projection for the real shape.

## Pre-measurement working estimate (kept for the record; confirmed by
## the measurement above, not superseded by it)

Written before the `ncu` capture, from PTX-level reasoning about the
generated code alone: the high half's load-add-store touches a
`[Q8_QMMA_TM, Q8_QMMA_TN]` = `[128, 64]` f32 region of the real output
buffer, a 32 KB read plus a 32 KB write beyond what the unsplit baseline
does, per column tile the high half's grid covers -- already at or above the
postmortem's ~100 KB bar from one tile's own fixup, before counting how many
tiles a real starved shape needs split. The one open question flagged at the
time was whether L2 residency (the high half's read following the low
half's write within microseconds on the same stream) would let the real
number come in below this PTX-level estimate. It does not: the measured
extra read (1.10-1.16 MB across both cache-control modes for the whole
16-tile shape, matching this estimate's own per-tile 32 KB x 16 = 512 KB
order of magnitude reasonably given real write amplification and DRAM
sector granularity) confirms the read genuinely reaches DRAM rather than
being absorbed, for the reason given in "DRAM kill-check result" above (the
kernel's own weight streaming evicts the small C write from L2 before the
accumulate step runs). The working estimate and the measurement agree in
direction and rough magnitude; the measurement is the number reported as the
result.

## Files

- `ncu_sk_baseline_call.ncu-rep` / `.csv` -- baseline `q8_qmma` (deep tile,
  `M=128,K=1024,N=2048`), `--cache-control all`, isolated via `-c 1` to the
  first (and only, for this shape) matching launch.
- `ncu_sk_baseline_none.ncu-rep` / `.csv` -- same, `--cache-control none`
  (write numbers not reliable, see the table's footnote; read numbers are).
- `ncu_sk_probe_all.ncu-rep` / `.csv` -- `q8_qmma_sk_lo` + `q8_qmma_sk_hi`,
  same shape, `PHOBOS_QMMA_STREAMK=1`, `--cache-control all`.
- `ncu_sk_probe_none.ncu-rep` / `.csv` -- same, `--cache-control none`.

## Emit diff-sweep

`git diff --stat` for this beam touches only `phobos-gguf` -- no
`phobos-lang` changes, so `emit`'s output over every `.ph` in `examples/`
and `phobos-lang/examples/` is unchanged by construction (a pure function of
`phobos-lang` plus the `.ph` text, and neither this beam's code nor any
`.ph` file changed). Same argument the prior split-K beam made for the same
reason.

## Current state

`qmma_streamk` defaults off (`PHOBOS_QMMA_STREAMK` opt-in only, matching
`qmma_split`'s convention). The code stays in the tree as a real, measured
negative result and as the working template for the boundary-fixup
mechanism (direct-write low half, load-add-store high half at a literal K
offset) should a future attempt want to build on it at a different tile
granularity or a different card's SM count -- but per "Is the full multi-way
design likely to clear the bar?" above, this beam does not recommend
pursuing the full multi-tile work-assignment design at `q8_qmma`'s current
`Q8_QMMA_TM x Q8_QMMA_TN` tile size without first revisiting that tile size
itself, which is out of this beam's scope (the tile-shrink probe this
session already ran and rejected on efficiency grounds, a separate
tradeoff from the one measured here). The four standing correctness gates on
both models were not run to completion for this design, since the kill-check
failing is the reported outcome and further correctness/timing investment on
a design that already fails its own acceptance bar would not change that
outcome.
