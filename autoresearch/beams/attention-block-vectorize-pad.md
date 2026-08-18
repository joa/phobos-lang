# attention_block: K/V vector widths and bank-conflict padding

Task: implement the two fixes `autoresearch/beams/prefill-sass-audit.md` found
and scoped for `attention_block` (`phobos-gguf/src/backend/device/kernels/attn.rs`,
`attention_block_src`, 4.47ms/pass baseline): item 4 (vectorize K/V staging)
and item 2 (pad K/V's shared tile against bank conflicts). This report covers
both fixes' implementation, correctness gates on both models, fix B's
occupancy kill-check, and three-point `ncu` timing (baseline / Fix A / Fix
A+B) on the real `pp128` shape, all in the same GPU slot and session so the
numbers are directly comparable.

**Headline result**: launch duration drops from **175.9us (baseline) to
138.4us (Fix A alone, -21.3%) to 92.5us (Fix A+B, -47.4% from baseline)**,
averaged over 5 launches each, `attention_block` at minicpm's real `pp128`
autotune shape. The average shared-load bank conflict drops from **5.0-way
to 2.4-way** once Fix B lands (unmoved by Fix A alone, as expected -- Fix A
vectorizes the staging *copy*, not the fragment *read* inside `dot`/`dot_t`
that Fix B targets). See "Timing" below for the full breakdown, including a
second, unplanned mechanism Fix A turned out to fix along the way (a
pool-release defect that was quietly costing this kernel a third of its
shared-memory budget).

**Headline residual on Fix B**: padding only reaches the main causal loop's
K/V tiles (the ~88% of trips Fix A's own residual already named). The
diagonal tile (`dk`/`dv`) *and* `q` itself stay unpadded, both because their
slices are masked (the compiler cannot prove their offsets in bounds) and
the masked materialization path never consults the new padding decision.
This is consistent with the bank-conflict number improving substantially but
not clearing to 1-way -- see "What actually got padded" below for the full
mechanism.

## Fix A: vectorize K/V staging

**Status: done, MLIR-level verified on a hand-built replica of the real
shape, and exercised end to end by the correctness gates below (which
compile the actual shipped `attn.rs` string, not a probe).**

Two changes to `attention_block_src`, exactly as the audit specced:
1. Added `@aligned(KW = D)` to the kernel.
2. Changed `let k`/`let v`/`let dk`/`let dv` to `var k`/`var v`/`var dk`/
   `var dv`.

Both together, not one alone, per the project's banked
`[[aligned_and_staging_go_together]]` lesson the audit re-confirmed for this
kernel.

Checked at the MLIR level by hand-building a standalone probe at minicpm's
real `pp128` autotune shape (`NH=16 G=8 D=128 BR=8`) matching the post-edit
`attn.rs` source structurally (the probe's scale constant is a fixed
9-decimal literal rather than whatever `{scale:.9}` prints for `128f32.sqrt().recip()`,
so this is a hand-built replica, not a byte-diff against the generated
string) and compiling it with `cargo run -p phobos-lang --example emit`.
Clean compile (exit 0). The four correctness gates below then compiled and
ran the actual `attention_block_src` format-string output, which is the real
end-to-end check that the shipped edit compiles and produces correct
results. The main causal loop's K and V copies both lower to:

```
%81 = vector.load %subview_125[%79, %80] {alignment = 16 : i64}
    : memref<8x128xf16, strided<[?, 1], offset: ?>, 1>, vector<8xf16>
vector.store %81, %assume_align_126[%79, %80] {alignment = 16 : i64}
    : memref<8x128xf16, 3>, vector<8xf16>
```

`vector<8xf16>`, the 16-byte-wide path -- matching the audit's own probe
exactly. The diagonal tile (`dk`/`dv`) stays scalar and masked, the accepted
residual the audit named: `base = NK - R + qt*BR` is a one-shot
`program_id`-relative offset, not a loop-trimmed induction variable, so no
`@aligned` promise reaches it. Confirmed directly: `memref.load %subview_127[...]`
(scalar, no vector width) for the diagonal tile's copy, same as before the
edit. Barrier count unchanged at 48 (`gpu.barrier` grep), matching the audit's
own cross-kernel invariant check.

## Fix B: pad K/V's shared tile against bank conflicts

**Status: done. Compiler-verified, occupancy kill-check passes both on paper
and confirmed on the card, and the real `ncu` capture shows the bank
conflict dropping from 5.0-way to 2.4-way (see "Timing" below).**

### Why this needed `phobos-lang` work, not a `.ph`-level lever

Checked first, per the task's own instruction: `dot`/`dot_t` only take the
`mma.sync`/padded-or-swizzled staging path under `@tensorcore` (confirmed in
`SPEC.md`'s own `@tensorcore` doc: "Tile-level `dot`/`dot_t` (flash attention)
take the same `mma.sync` path under the same gate"). `attention_block` carries
no `@tensorcore` tag and is not going to acquire one (this kernel's `[BR,BR]`-shaped
score tile and the fp32 accumulation path are exactly what the two prior
tensor-core redesigns of this same kernel already tried and shelved -- see
`prefill-attention-tensorcore.md` and `-wide-br.md`). So `alloc_tile_padded`
(`phobos-lang/src/codegen/tile/alloc.rs:247`) is reachable only by teaching
the generic (non-tensor-core) staging path to call it, which means a
`phobos-lang` change.

### The scoping problem, and how it was resolved

The K/V staging (`var k = K[...]`, etc.) goes through `stmt.rs`'s
`Stmt::Var` branch, the one place any `var name = <tensor slice>` with no
explicit tile type lands (`emit_expr` returns `Rv::Tile(src)` with
`src.owned == false`, then `alloc_tile_shaped` + `tile_copy` stage it). That
branch is shared by every kernel in the tree using this pattern, not just
`attention_block` -- `attention_src` and `attention_split_src` both stage
`[*, D=128]` f16 K/V slices through the exact same branch, and are live decode
paths this task has no bench/ncu budget to re-verify. Padding that branch
unconditionally would silently change every one of those kernels' shared
layout with no way to confirm the timing effect this session.

Fixed by making the padding decision two-gated, both conditions required:

1. **Opt-in per kernel**: a new attribute, `@padstage`, added to
   `Kernel`/`Codegen` exactly like `@dynshared`/`@pipeline`
   (`Kernel::wants_padded_stage` in `phobos-lang/src/ast.rs`, `pad_stage: bool`
   field on `Codegen`, `phobos-lang/src/codegen/mod.rs`). Only
   `attention_block_src` in `attn.rs` gets the attribute; `attention_src` and
   `attention_split_src` are untouched, so this fix reaches only the one
   kernel it was scoped to.
2. **Even when opted in, only a tile whose row pitch is itself an exact
   multiple of the shared-memory bank period** (`SHARED_BANK_BYTES = 128`,
   `phobos-lang/src/codegen/mod.rs`) gets padded --
   `Codegen::should_pad_stage` (`phobos-lang/src/codegen/tile/alloc.rs`)
   checks `cols * elem_bytes % 128 == 0` before choosing `alloc_tile_padded`
   over `alloc_tile_shaped` in `stmt.rs`'s staging branch. This is the actual
   defect the audit's arithmetic named (every row landing on the same bank at
   a fixed column), not a blanket "always pad" -- a tile whose pitch already
   misses the period is left alone.

Padding reuses the existing `WMMA_SMEM_PAD = 8` elements the WMMA path
already uses (`alloc_tile_padded` itself is unchanged); for the f16 K/V tile
that is `+16` bytes/row (`256 -> 272`), still a multiple of 16 so the Fix A
vector stores stay 16-byte aligned (checked directly, see below).

### What actually got padded, verified in the emitted MLIR

Compiled the post-edit `attention_block_src` (Fix A + Fix B both applied,
`@aligned(KW=D)` + `@padstage`, `var` staging) at the same minicpm `pp128`
shape:

- **K and V's main-loop tiles**: `memref<8x136xf16, 3>` (128 -> 136, the `+8`
  elements `WMMA_SMEM_PAD` adds). Row pitch 272 bytes, `272 % 128 = 16`,
  breaking the exact-multiple collision the audit's arithmetic identified.
  Fix A's `vector<8xf16>` loads/stores survive unchanged into this padded
  buffer (confirmed: same two `vector.load`/`vector.store` pairs as the
  Fix-A-only probe, now targeting `memref<8x136xf16, 3>` instead of
  `memref<8x128xf16, 3>`) -- the padded row stride (272 bytes) is still a
  multiple of 16, so every 8-element vector access stays on a legal boundary.
- **The diagonal tile (`dk`/`dv`)**: **stays unpadded**, `memref<8x128xf16, 3>`.
  This mirrors Fix A's own accepted residual for the identical reason: `dk`/`dv`'s
  slice is masked (`base` is `program_id`-relative, not proven in-bounds), so
  it materializes through `check.rs`'s `materialize_masked`, which calls
  `alloc_tile_shaped` directly and never consults `should_pad_stage`. Extending
  the masked path to also pad was judged out of scope for this task (real
  additional design work, touching a path every masked slice in the tree
  goes through, not just this kernel's ~12% diagonal-trip residual) -- flagging
  it as follow-up, not attempting it here, same call the audit made for Fix
  A's own diagonal-tile residual.
- **`q`**: also unpadded, for the same masking reason -- `Q`'s row slice
  (`qt*BR:+BR` against `R`) and column slice (`qcol:+D` against `QW`) have no
  `@aligned` promise, so `q`'s staging is also masked and routes through
  `materialize_masked`, never reaching `should_pad_stage`. This was not the
  audit's target (the audit's mechanism and ncu evidence are specifically
  about K/V's f16 tile) and unmasking `q` would need its own correctness
  argument about `R`/`QW`'s actual divisibility that this task did not chase.
- **The `[BR,BR]` f32 score tile `s`, flagged for a check in the task brief**:
  confirmed **not part of this defect and correctly left unpadded**. `s` is
  declared with an explicit tile type (`var s: tile<f32>[BR,BR] = dot_t(q,k)`),
  which never goes through the `var name = <tensor slice>` staging branch at
  all (it's a computed value via `emit_tile_decl`, not a tensor-slice
  adoption) -- so `should_pad_stage` is never even consulted for it. Had it
  been, the arithmetic still excludes it: `BR*4 = 32` bytes at `BR=8`, and
  `32 % 128 != 0`, so `should_pad_stage` would return false anyway. Two
  independent reasons it's untouched, matching the audit's own prediction.

Barrier count: unchanged at 48 (`gpu.barrier` grep on the combined-fix probe),
so this doesn't interact with the audit's item-3 finding either.

### Occupancy kill-check (per the task's standing instruction, run before
calling this a win)

Static shared-memory delta from padding, computed off the emitted
`memref.global` list: K and V's main-loop tiles each grow from 128 to 136
elements at 2 bytes (f16), `+16 bytes/row * 8 rows = +128 bytes` each, `+256
bytes` total for the two tiles this fix actually touches. `dk`/`dv` and `q`
stay their original unpadded size, per the masking finding above -- this
session's own initial estimate (made before the masking interaction was
found, and not a claim from the audit, which never discussed padding `q`)
assumed `q` would also grow; the real delta is smaller than that estimate.

Audit's own baseline for this kernel: `Block Limit Shared Mem: 2` blocks/SM
at 25.25 KB/block static, the binding occupancy constraint (ahead of register
or warp limits). A `+256` byte increase against a 25.25 KB (25,856-byte)
budget is about a 1% growth -- nowhere near enough to push the per-block
footprint past whatever threshold would drop the SM's resident-block count
from 2 to 1 (that would need roughly a doubling, not a 1% move, on a card
whose full per-SM shared budget is tens of KB past the 2-block point).
**Kill-check passes on paper.**

**Confirmed on the card** (see "Timing" below for the full three-point
capture): `Block Limit Shared Mem` is **3** for both Fix A alone and Fix
A+B, unchanged by Fix B's padding -- the kill-check holds. It moved from the
audit's documented `2` to `3` already at Fix A, for a reason neither this
task nor the audit anticipated: see "An unplanned second effect of Fix A"
under Timing.

### Emit-diff-sweep (Fix A + Fix B combined, both targets, all `.ph` examples)

Ran `cargo run -p phobos-lang --example emit` over all 9 files in
`examples/*.ph` and both files in `phobos-lang/examples/*.ph`, at the default
target and at `PHOBOS_CHIP=sm_80 PHOBOS_INDEX_BITS=64` (22 combinations
total), diffed byte-for-byte against a baseline captured with `git stash`
before either fix landed. **Zero differences** (`diff -rq` exit 0, all 22
files identical). This confirms both fixes are a true no-op outside
`attention_block`: Fix A only edits `attn.rs`'s format string (nothing else
calls `attention_block_src`), and Fix B's `phobos-lang` codegen change is
gated behind `@padstage`, which none of these example kernels declare.

`cargo test -p phobos-lang`: **158 passed, 0 failed** (155 pre-existing, plus
three new tests added in `codegen/tests/tile.rs` pinning `@padstage`'s
behavior: `padstage_pads_a_bank_period_pitch_tile` (a `[8,128]` f16 staging
tile pads to `memref<8x136xf16, 3>` under `@padstage`),
`without_padstage_the_same_tile_stays_unpadded` (the same statement without
the attribute stays `memref<8x128xf16, 3>`), and
`padstage_leaves_a_sub_period_pitch_tile_alone` (a `[8,32]` f16 tile, 64
bytes/row, stays unpadded even with `@padstage` on, since its pitch is not a
bank-period multiple -- the same reasoning that excludes `attention_block`'s
own `s` tile)).

`cargo clippy --release --features cuda -p phobos-lang -p phobos-gguf -- -D
warnings`: clean, no warnings.

## Correctness gates, both fixes together, both models, `PHOBOS_ATTN_PERSIST=1`

All four gates run `--release --features cuda` against an idle-but-shared
GPU (per the standing protocol: value-comparison gates run freely under
concurrent contention; only the timing capture below needs an exclusive
slot).

**`backend_check`** (no model argument; synthetic op suite): worst relative
error **2.902e-4** -- identical to the documented baseline, no regression.

**`batch_check`**:
- minicpm5-1b-Q8_0: gpu spread err up to 1.350e-2, "batched and sequential
  agree", exit 0.
- Qwen3.5-0.8B-Q8_0: gpu spread err up to 1.540e-2, "batched and sequential
  agree", exit 0.

**`model_check`**:
- minicpm5-1b-Q8_0: 3 steps, spread err up to 1.763e-2, "backends agree",
  exit 0.
- Qwen3.5-0.8B-Q8_0: 3 steps, spread err up to 1.074e-2, "backends agree",
  exit 0.

**`fuse_check`**:
- minicpm5-1b-Q8_0: prompt pass agrees exactly; 32 decode steps, at most
  1.332e-2 of the logit spread apart, 9.944e-3 average, 0 top-token flips.
- Qwen3.5-0.8B-Q8_0: prompt pass agrees exactly; 32 decode steps, at most
  1.051e-2 of the logit spread apart, 8.085e-3 average, 0 top-token flips.

All four gates pass on both models with both fixes applied together.

## Timing

Three `ncu --set full -k "regex:attention_block" -c 5` captures, same
session, same idle card (`nvidia-smi` checked immediately before each: 5-33%
transient util but clock at the 420-540MHz idle floor throughout, no
`bench`/`check`/`ncu` process besides this one running), all against
`target\release\examples\bench.exe -m models\minicpm5-1b-Q8_0.gguf -p 128 -n
0 -r 1 --no-warmup`, `PHOBOS_ATTN_PERSIST=1`, minicpm's real `pp128`
autotune shape (`NH=16 G=8 D=128 BR=8`, grid `(16,16,1)`, block `(256,1,1)`).
Each state built by editing `attn.rs` back to that state, `cargo build
--release --features cuda --example bench`, capture, then restoring toward
Fix A+B (final diff checked byte-identical to the working state via `git
diff` after restoring). 5 launches captured per state; per-launch spread was
within about 8% peak-to-peak in every state (baseline 170.3-184.2us, Fix A
136.5-143.5us, Fix A+B 90.0-96.1us), but the three states' ranges do not
overlap at all, so single-launch noise cannot account for any of the deltas
below.

One process note on this capture round: the first `ncu` invocation
accidentally profiled the *main* tree's `bench.exe`
(`C:\Users\joaeb\code\phobos\...`) rather than this worktree's, because the
PowerShell tool's working directory defaulted there rather than to the
worktree (this session's Bash calls had all explicitly `cd`'d into the
worktree per command and never hit this; the PowerShell tool apparently
starts elsewhere and needs an explicit `Set-Location` once). Caught
immediately from the process path `ncu` printed, the resulting `.ncu-rep`
was deleted before being read, no other files in the main tree were
touched, and `Set-Location` was added before every capture below -- the
three captures used for this report all show
`C:\Users\joaeb\code\phobos-wt-attnfix\target\...` as the profiled process
path.

| state | launch duration (5-launch avg) | bank conflict | conflicted wavefronts | `Block Limit Shared Mem` | achieved occupancy |
| --- | --- | --- | --- | --- | --- |
| baseline (neither fix) | 175.9us | 5.0-way, 2,229,017 conflicts | 67.83% of 3,286,284 | 2 blocks/SM | 47.14% |
| Fix A alone | 138.4us (-21.3%) | 5.0-way, 2,229,390 conflicts (unchanged) | 67.83% of 3,286,619 (unchanged) | 3 blocks/SM | 67.35% |
| Fix A + Fix B | 92.5us (-33.2% vs A, -47.4% vs baseline) | **2.4-way**, 508,910 conflicts (-77.2%) | **32.49%** of 1,566,154 | 3 blocks/SM (unchanged from A) | 65.05% |

(Raw per-launch numbers, all five: baseline `[174.24, 179.36, 184.22, 170.30,
171.55]`us; Fix A `[136.48, 137.63, 143.46, 138.02, 136.58]`us; Fix A+B
`[91.49, 91.97, 92.99, 90.02, 96.13]`us. Evidence files:
`ncu_attnblock_baseline_pp128.ncu-rep`, `ncu_attnblock_fixA_pp128.ncu-rep`,
`ncu_attnblock_fixAB_pp128.ncu-rep`, and a `_details.txt` export of each.)

**Fix A alone is a real, substantial win on its own** (-21.3%), which is
notable since Fix A's own MLIR-level story was "vectorize 88% of the K/V
staging copies" -- a smaller-sounding change than the result. The
bank-conflict metric confirms this speedup is *not* coming from Fix B's
mechanism (conflicts are statistically unchanged, 2,229,017 -> 2,229,390):
Fix A vectorizes the staging *copy* (global memory into shared), not the
fragment *read* inside `dot`/`dot_t` that actually collides on banks --
those are two different instruction streams reading/writing the same
buffer, and only Fix B's padding touches the read side's collision pattern.

**Fix B clears most, not all, of the conflict**: 5.0-way down to 2.4-way,
conflicted-wavefront share 67.83% down to 32.49%, raw conflict count down
77.2%. This matches the predicted residual exactly -- `q` and the diagonal
tile (`dk`/`dv`) still collide (both unpadded, both read by `dot_t`/`dot` at
the same frequency as the now-padded main-loop `k`/`v`), so some conflict
remains rather than clearing to 1-way. Total shared-load wavefronts also
roughly halve (3.29M -> 1.57M) alongside the conflict drop, consistent with
fewer, wider transactions once the collision pattern breaks.

### An unplanned second effect of Fix A

The `Block Limit Shared Mem` jump from the audit's documented 2 to 3 blocks/SM
**happens already at Fix A, before any padding**, which the occupancy
arithmetic above did not predict (padding was expected to cost bytes, not
save them). Traced by comparing the emitted `memref.global` list across all
three probes at the same shape:

- **Baseline: 16 distinct shared tiles**, six of them `[8,128]f16` (a `k`/`v`
  pair for what turns out to be *two* separate static copies of the causal
  loop's body -- this kernel's `for kt in range(0, base, BR)` compiles to a
  trimmed main pass plus a masked remainder tail, each its own static site --
  plus separate `dk`/`dv` globals for the diagonal tile). Total static
  footprint sums to ~24.7KB, matching the audit's measured 25.25KB/block
  closely.
- **Fix A: 12 distinct shared tiles**, only two of them `[8,128]f16` (`k`
  and `v`, shared by both loop-body copies, with `dk`/`dv` reusing the same
  pool slots once `k`/`v` are dead rather than minting their own). Total
  static footprint sums to ~16.7KB.

The mechanism: `let`-bound tiles (`Binding::View`) never go through the
buffer-pool's `release()` (its own doc: "No-op for views, params and named
tiles"), so every masked slice that materializes into shared memory via
`let k = K[...]` (the baseline's masking path, since no `@aligned` promise
exists yet) holds its buffer for the rest of the kernel -- forcing the
second loop-body copy's `k`/`v` and the diagonal tile's `dk`/`dv` to each
mint fresh globals instead of reusing the first copy's now-idle ones.
Switching to `var` (needed anyway, for the staging reason Fix A's own MLIR
check already established) makes these proper pool-tracked `Binding::Tile`
values that release when their last read passes, so the second loop-body
copy and the diagonal tile reuse the first copy's freed slots instead of
growing the footprint. This was already necessary for the vectorization fix
and wasn't chased as a distinct change -- flagging it here because it is
most of why Fix A's occupancy and duration numbers came in higher than the
staging-vectorization story alone would predict, and because the same
`let`-never-releases pattern likely costs other masked-tile kernels in this
tree shared memory they don't need to spend; worth a follow-up audit of the
tree for other `let name = <masked slice>` sites this session did not check.

### Reading these against the audit's own caution

CLAUDE.md and the audit both flag that a benchmark number is only comparable
to one measured in the same session on the same card. All three numbers
above meet that bar (same session, same idle card, same shape, same binary
family, differing only in the one file rebuilt between captures). The
`t/s` figures `bench.exe` printed alongside each capture (12.67, 14.23,
14.75) are **not** cited as throughput -- profiling perturbs timing, per the
audit's own note, and this task did not run `scripts/bench.py`. The
per-kernel launch durations above are the number to trust; an end-to-end
`pp128` throughput comparison is `scripts/bench.py`'s job, which this task
was instructed not to run itself.

## Files touched

- `phobos-gguf/src/backend/device/kernels/attn.rs`: `attention_block_src` --
  `@aligned(KW = D)`, `@padstage`, `let`->`var` for `k`/`v`/`dk`/`dv`.
- `phobos-lang/src/ast.rs`: `Kernel::wants_padded_stage`.
- `phobos-lang/src/codegen/mod.rs`: `SHARED_BANK_BYTES` constant, `pad_stage`
  field on `Codegen`.
- `phobos-lang/src/codegen/tile/alloc.rs`: `Codegen::should_pad_stage`.
- `phobos-lang/src/codegen/stmt.rs`: `Stmt::Var`'s tensor-slice staging branch
  consults `should_pad_stage` to choose `alloc_tile_padded` vs
  `alloc_tile_shaped`.
- `phobos-lang/src/codegen/tests/tile.rs`: three new tests pinning
  `@padstage`'s behavior (padded, unpadded-without-the-attribute, and
  unpadded-below-the-bank-period cases).
- `SPEC.md`: `@padstage` attribute entry.

## Evidence files (`autoresearch/beams/`)

- `ncu_attnblock_baseline_pp128.ncu-rep` / `_details.txt` -- neither fix,
  5 launches, `--set full`.
- `ncu_attnblock_fixA_pp128.ncu-rep` / `_details.txt` -- Fix A alone.
- `ncu_attnblock_fixAB_pp128.ncu-rep` / `_details.txt` / `.csv` -- Fix A+B
  together, the shipped state.
