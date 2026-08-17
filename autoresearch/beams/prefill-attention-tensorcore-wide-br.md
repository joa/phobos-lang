---
parent: e2a692f (autoresearch branch, PHOBOS_ATTN_PERSIST default-on)
status: killed at the shared-memory gate before any kernel reached the tree.
  BR=32, D=128 needs 106.6 KiB of shared memory in the straightforward form
  that mirrors `attention_block_src` (the postmortem's own top-ranked
  follow-up), more than double the 48 KiB static cap and 67% over Turing's
  64 KiB hard per-block ceiling -- `@dynshared` cannot rescue an over-budget
  request past the card's own limit, confirmed by reproducing the same
  106.6 KiB peak under `@dynshared` itself. The best variant tried
  (`var`-form automatic pipelining) still overshoots the 64 KiB hardware
  ceiling by 17%. BR=64 was not attempted: it scales worse, not better, and
  the task's own stopping rule is to not force a design past this gate.
  Nothing was wired into the dispatcher and no repository file changed;
  everything below lives in this beam file and the session's scratchpad.
---

# Beam: prefill attention tensor cores, wider BR

Follow-up to `autoresearch/beams/prefill-attention-tensorcore.md`, which
built and reverted a `BR=16` tensor-core attention kernel (lost 27-40% to
the shipped f32 `attention_block` kernel) and ranked "a wider `BR`" as its
single most promising untried lever, on the theory that `BR=16` is the WMMA
fragment floor rather than a chosen tile size and never amortizes its fixed
per-launch WMMA overhead. This round's job was to build `BR=32` and, only if
that cleared its own gates with room to spare, escalate to `BR=64`.

It did not clear the first gate. The shared-memory kill-check the task
brief asked for up front -- "measure it, don't assume there's room" -- is
where this stopped, with numbers, before any correctness or timing work.

## Design, before the measurement

Kernel: `attention_tc(Q: tensor<f32>[R, QW], K: tensor<f16>[NK, KW], V:
tensor<f16>[NK, KW], O: tensor<f32>[R, QW])`, mirroring
`attention_block_src`'s structure exactly (`phobos-gguf/src/backend/device/kernels/attn.rs`):
one program owns `BR` consecutive query rows of one head, a causal loop over
whole `[BR, D]` key/value tiles with online-softmax accumulation, followed
by one `tril`-masked diagonal tile for the block straddling the causal
boundary. `@tensorcore` added; `@launch(128)`, not `@launch(256)` -- see
below for why. `kcol = h / G * D` unchanged from the existing kernel.

One deliberate change from the reverted `BR=16` kernel: **`Q` stays an f32
kernel parameter**, the same signature `attention_block_src` already uses,
rather than a dedicated `q_cast` kernel pre-casting it to f16 (which the old
postmortem flagged as a real but unresolved cost, ~0.07ms, and listed
folding it into an existing kernel as follow-up #3). `wmma_dot`'s operand
staging (`phobos-lang/src/codegen/matmul/wmma.rs`, `dot_stage` in
`matmul/stage.rs`) already accepts an f32 operand and rounds it to f16
during staging (`stage_to_f16` -> `tile_copy_f16`, an `arith.truncf` per
element), and `hoist.rs`'s loop-invariant-operand hoisting stages `q` once
into a shared f16 buffer before the causal loop rather than once per
iteration -- the same place `q_cast` would have landed, done by construction
instead of a second launch. This resolves old follow-up #3 for free, for
whoever revisits this kernel shape; not benchmarked here since the kernel
never got far enough to run.

`@launch(128)`, not `256`: `wmma_dot`'s `wmma_plan` (`matmul/wmma.rs`)
requires `wm * wn == warps` with both `gm % wm == 0` and `gn % wn == 0` over
the *output* tile's 16x16-fragment grid. At `BR=32`, the score matmul's
output is `[32, 32]` (`gm = gn = 2`); at `@launch(256)` (8 warps), no
factorization of 8 into `(wm, wn)` both dividing 2 exists, so `wmma_plan`
returns `None` and `wmma_dot` silently falls back to the vector path for
that one dot site -- no error, just half the kernel quietly not using tensor
cores. At `@launch(128)` (4 warps), `(wm, wn) = (2, 2)` plans the `[32,32]`
score matmul and `(1, 4)` plans the `[32,128]` PV matmul. Confirmed in the
emitted MLIR (see below) rather than assumed.

## The shared-memory kill-check

Methodology: rather than hand-trace the codegen's tile pool and staging/
hoisting rules (attempted first, got it wrong -- see "what the hand-count
missed" below), wrote the kernel as a standalone `.ph` file and ran `cargo
run -p phobos-lang --example emit` at the default target (`sm_75`, 32-bit
index -- `Context::default()`, matching `phobos-gguf`'s real compile config,
same check the original beam ran), then summed every `memref.global` in the
shared address space from the emitted MLIR. This is what the compiler
actually allocates, not an estimate of it.

Four variants measured, `NH=16, G=8, D=128` (minicpm's shape):

| variant | distinct shared tiles | total bytes | vs 48 KiB static cap | vs 64 KiB Turing hard cap |
| --- | --- | --- | --- | --- |
| `BR=16`, `let`-form (sanity check against the old postmortem) | 17 | 47,936 | fits, 1,216 bytes to spare | fits |
| `BR=32`, `let`-form (mirrors `attention_block_src` as specified) | 18 | 109,184 | **+60,032 (122% over)** | **+43,648 (67% over)** |
| `BR=32`, `@dynshared` on the identical source | 18 (same peak, view-offset confirmed) | 109,184 | n/a (dynamic) | **+43,648 (67% over)** |
| `BR=32`, `var`-form (automatic pipelining of `k`/`v`) | 14 | 76,416 | +28,264 | **+10,880 (17% over)** |

The `BR=16` row is not new work -- it is a direct check that this kernel's
*structure* (not just its tile size) reproduces the original postmortem's
"comfortably under the 48 KB static cap" finding before trusting the same
structure's `BR=32` number. It does: 46.8 KiB against 48 KiB, matching the
old beam's claim, though "comfortably" is generous -- 2.5% headroom is not
much, and it says the `BR=16` kernel was already closer to the wall than
the postmortem's prose suggested.

`BR=32` fails in every form tried. The `@dynshared` row is the same source
file with `@dynshared` added and nothing else changed, to test the task
brief's named fallback directly: the dynamic allocator's byte-offset cursor
(`phobos-lang/src/codegen/tile/alloc.rs`) reaches the same 109,184-byte peak
(confirmed by the highest `memref.view` offset plus its tile's own size
landing exactly on 109,184), because this kernel has no barrier-separated
phases for the allocator's phase-boundary reset to exploit (that reset is
what lets `attention_persist_src` reuse phase-one's footprint for phase
two; a single online-softmax pass never returns `dynamic_live` to zero
mid-kernel). `@dynshared` moves the ceiling from 48 KiB to Turing's 64 KiB
hardware opt-in maximum (`docs/GGUF.md`'s own "Turing's 64K of shared
memory", confirmed against `phobos-gguf/src/backend/device/launch.rs`'s
`compile_dynamic`, which raises `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`
to whatever the kernel needs and lets the driver reject it past the
hardware limit) -- it does not create room past that hardware wall, and
109,184 bytes is past it either way.

The `var`-form (automatic pipelining of the `k`/`v` staging, per the recent
"lang: make pipelining automatic" change) was tried on the theory that it
might, counter to intuition, *reduce* the footprint by giving the pipeline's
own staged buffers a shape wmma_dot's operand staging could reuse more
directly than the plain `let`-form's per-call restaging. It does reduce the
total (18 tiles/109,184 bytes down to 14/76,416), but still misses the 64
KiB hardware ceiling by 10,880 bytes (17%), and this is not the design the
task asked for regardless (`@pipeline` is now an assertion, not an opt-in,
and doubling the K/V staging buffers on top of an already-too-large kernel
is the wrong direction for a kernel already over budget on its unpipelined
form). Recorded for the "why not pipelined" question, not proposed as a fix.

Tensor-core engagement confirmed the same way the original postmortem did,
on the `let`-form `BR=32` emission: 48 `gpu.subgroup_mma_compute`, 84
`gpu.subgroup_mma_load_matrix`, 0 `vector.contract` (the vector-path
fallback) -- every dot site plans and runs on tensor cores at `@launch(128)`,
confirming the shared-memory numbers above are for a kernel that actually
exercises what it set out to measure, not one that silently degraded to the
f32 path first.

## Why it does not fit: the mechanism, not just the number

The `let`-form's 18 distinct tiles, by shape (all in `phobos-gguf/src/backend/device/kernels/attn.rs`-equivalent structure, not yet in that file):

- 2x `f32[32,128]` (32,768 bytes) -- the accumulator `acc`, plus a second
  tile of the identical shape the `O[...] = acc / l` expression
  materializes its result into before the strided store to global memory.
- 8x `f16[32,128]` (65,536 bytes) -- the hoisted `q`, and *seven more*
  copies of the same `[BR, D]` f16 shape: `wmma_dot`'s operand staging
  (`dot_stage` in `matmul/stage.rs`) has no fast path for "this operand is
  already a shared f16 tile of the right shape" -- it always allocates a
  fresh buffer and copies into it (`stage_to_f16`), even when the source is
  the `k`/`v` view the causal loop just read from global memory one
  statement earlier. Combined with this codegen's existing convention
  (already documented in `ATTN_BLOCK_ELEMS`'s own comment on the shipped
  f32 kernel: "about ten" distinct tiles, no reuse across different
  variable names) that a named `var`-declared tile in `.ph` source is never
  automatically released and pooled against a *different* name, the loop's
  `k`/`v` restage tiles and the diagonal tile's separately-named `dk`/`dv`
  restage tiles cannot share a global even though they are never live at
  the same time.
- 2x `f32[32,32]` (8,192 bytes) and 1x `f16[32,32]` (2,048 bytes) -- the
  loop's score tile `s` and the diagonal's separately-named `ds`, same
  story: same shape, different names, no reuse. (The `f16[32,32]` restage
  for `dot(s,v)`/`dot(ds,dv)`'s `a` operand *did* end up pooled to one
  instance in this run -- the reuse is not fully absent, just inconsistent
  across which allocation site happens to trigger it, which is itself a
  sign this is accidental pooling rather than a load-bearing design.)
- 5x `f32[32,1]` (640 bytes) -- `m`, `l`, `mn`/`dm`, `corr`/`dcorr` -- small
  relative to the rest, not a lever.

None of this is `BR=32`-specific: the same duplication exists at `BR=16`,
just at half the linear tiles' size and a quarter of the `[BR,BR]`-shaped
ones', which is exactly why `BR=16` still fit (46.8 KiB) while `BR=32`
doesn't (106.6 KiB) -- **the growth from `BR=16` to `BR=32` is not the ~2x a
uniform doubling would suggest.** The `[BR, D]`-shaped tiles (accumulator,
Q/K/V staging) scale linearly with `BR` and roughly double; the `[BR,
BR]`-shaped tiles (the score matrix and its own f16 restage) scale with
`BR^2` and roughly quadruple. Widening the tile is not a uniform scale-up
of the `BR=16` kernel's footprint, it is a mix of linear and quadratic
terms, and the quadratic ones are what push the total past 2x.

### What the hand-count missed, for the next person who tries this by arithmetic first

An initial pass at this kill-check (before writing the probe kernel) hand-traced the
pipeline/hoist code and predicted roughly 39 KiB fitting comfortably in 48
KiB, reasoning from the `let`-form having no pipeline doubling and assuming
the hoisted `q` buffer, released after the loop, would be reused by the
diagonal tile's own restaging. The measured 106.6 KiB shows that reasoning
underweighted two things actually present in this codegen: `wmma_dot`
restages *every* operand on *every* call regardless of whether it is
already a shared f16 tile (the `q`/`k`/`v` distinction only matters for
*hoisting eligibility*, not for whether a later `dot_stage` call re-copies
the result), and named `var` tiles are not pool-reused across distinct
source-level names even when their lifetimes never overlap. Both are real,
existing properties of this codegen (not new bugs introduced by this
kernel), consistent with the shipped f32 `attention_block_src` already
carrying "about ten" distinct tiles for the same reason at its own smaller
scale. The lesson banked here: for a kernel using more than one or two
`dot`/`dot_t` call sites, measure the emitted `memref.global` list directly
rather than hand-counting from the staging/hoisting source -- the actual
tile count is materially higher than the mechanism descriptions in
`wmma.rs`/`stage.rs`/`hoist.rs` suggest in isolation.

## What was not attempted, and why

**`BR=64`**: not built. The task's own stopping condition -- `BR=32`
clearing its shared-memory gate before escalating -- was not met, and the
mechanism above (quadratic growth in the `[BR,BR]`-shaped tiles) predicts
`BR=64`'s score tile and its restage alone would be `4x` `BR=32`'s already
over-budget `8,192 + 2,048` bytes, before even accounting for the `[BR,D]`
tiles doubling again. Not worth spending a kill-check run to confirm what
the arithmetic already rules out.

**Folding the diagonal tile into the loop's own pipelined path** (the
postmortem's follow-up #2, one call site's tiles instead of two): not
attempted as code, but the mechanism found here says this is the most
promising remaining lever if this line is revisited -- it would remove the
diagonal's separately-named `dk`/`dv`/`ds`/`dm`/`dcorr` tile identities
(roughly the `2x f32[32,32]` and part of the `8x f16[32,128]` counts above)
by making the diagonal tile just one more (masked) trip through the same
loop body, reusing the loop's own `s`/`mn`/`corr`/`k`/`v` names. Rough
accounting from the breakdown above suggests this could remove on the order
of 10-15 KiB, which closes real ground but, combined even with the
`var`-form's already-lower 76,416-byte baseline, does not obviously clear
the 64 KiB hardware ceiling on its own -- speculative, unmeasured, and a
real code change (restructuring the loop to run one extra masked trip),
not attempted this round given the time already spent on the measurement
above and the explicit brief to report a clean negative rather than force
a design that does not fit.

**Correctness gates, the precision-floor tolerance carve-out, and timing
A/B**: none run. The kernel never reaches a state that compiles within this
card's shared-memory budget, so there is nothing to dispatch, nothing to
gate-check, and no `nsys`/`ncu` comparison to make against the 4.47 ms
`attention_block` baseline. Standing correctness-first discipline: no
speculative timing number for a kernel that cannot run.

## State of the tree

No repository file changed. The probe kernels and their emitted MLIR live
only in this session's scratchpad (outside the repository), not committed
anywhere. `DeviceBackend::attention`
(`phobos-gguf/src/backend/device/backend.rs`) is unchanged from
`e2a692f`, still dispatching to `attention_blocked` first exactly as the
original tensor-core postmortem left it, with that postmortem's revert
comment still in place and accurate.

## Recommendation

Do not pursue a wider-`BR` tensor-core prefill attention kernel further on
this card without first landing the diagonal-fold restructuring above (or
some other reduction of comparable size) *and* re-running this exact
shared-memory kill-check on the result, since the 17%-over `var`-form
number here is the closest any variant got and still does not clear.
Absent that, this lever is exhausted for now: attention's own kernel time
(4.47 ms) remains the largest *relative* gap to llama.cpp of any kernel
family, but the two tile sizes now tried (`BR=16`, reverted for losing on
speed; `BR=32`, killed here for not fitting) both say the WMMA legacy path
on `sm_75` is a poor fit for this kernel's shape at any tile width this
card's shared memory can hold. The GEMM projection (`q8_qmma`, 13.29 ms,
the session's other standing lead) remains the larger lever in absolute
terms and was not touched by this round.
