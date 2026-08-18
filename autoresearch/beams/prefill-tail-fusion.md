# Prefill tail-kernel fusion

status: **one fusion shipped and measured** (`rope_gather`, ~0.10-0.14ms/pass
genuine improvement, verified on both models); **one fusion built, gated
correct, measured as a wash-to-small-loss, and reverted** (`swiglu_2d_q` --
see the negative-result section below, patch kept for reference); one lead
(`q8_qmma_add`) not attempted per the orchestrator's explicit call. The
brief's own tail-bucket table was found contaminated by decode kernels and is
corrected below. Ready for review.

## The brief's table was averaging 4 prefill passes with 4 decode passes mixed in

Before touching any kernel, tried to reconcile the brief's per-kernel table
(`rope` "96 instances/pass", `fused` "48 instances/pass", etc., attributed to
minicpm's pp128 shape) against what the code can actually produce. Two things
didn't fit:

- `fused` (the `PHOBOS_FUSED_*` output) is gated `rows != 1` decline at every
  call site (`Ffn::forward_fused`, `Linear::project_fused`,
  `Linear::add_projected`'s `fused_attn_out` -- all in `phobos-gguf/src/layers.rs`).
  minicpm is architecture `llama` (`phobos-gguf/src/llama.rs`), whose whole
  forward pass runs every block at one `rows` value per call; a genuine pp128
  pass never has `rows == 1` anywhere, so `fused` cannot appear in a pure
  prefill trace at all, structurally, not as a possibility that happens not to
  fire.
- Hand-deriving instance counts per layer from `llama.rs::attention` and
  `Ffn::forward` at `rows > 1` gives exactly half the brief's `rms_norm_q`,
  `rope`, `copy_2d` and `store_2d_pair` counts (48/48/48/24 vs the brief's
  72/96/72/48), while `quantize`, `add_into` and `swiglu_2d` matched the
  brief almost exactly.

Traced it to `phobos-gguf/examples/bench.rs`'s warmup routine (lines
168-200): it runs one prefill-shaped `forward()` at the widest prompt size,
**then four decode-shaped `forward_greedy()` calls** to warm the greedy path's
kernels before any timed repetition sees a first-compile cost. The
`nsys_qmma_nodetrace_off.nsys-rep` capture the brief's table came from holds
both. Confirmed by direct measurement, not just code reading: exported
`cuda_gpu_trace` from the `.nsys-rep` (`nsys stats --force-export=true
--report cuda_gpu_trace --format csv`, read-only against a committed data
file, no capture involved) and used `rms_norm`'s 8 instances (fired exactly
once per `forward`/`forward_greedy` call, at the very end of
`forward_to_logits`) as forward-call boundaries. That splits the capture into
8 windows; checking each for `attention_block` (prefill-only) vs
`attention_persist` (decode-only) instances gives:

| window | kernels | attention_block | attention_persist | kind |
| --- | --- | --- | --- | --- |
| 1 | 748 | 24 | 0 | **prefill** (warmup) |
| 2-5 | ~225 each | 0 | 24 each | decode (warmup x4) |
| 6-8 | ~415 each | 24 | 0 | **prefill** (timed x3) |

Exactly the "1 warmup + 3 timed" prefill passes the brief described, plus the
four decode warmup calls the brief's totals never excluded. Summing only
windows 1, 6, 7 and 8 and dividing by 4 gives the corrected per-genuine-pass
table:

| kernel | brief's ms/pass | brief's inst/pass | corrected ms/pass | corrected inst/pass |
| --- | --- | --- | --- | --- |
| `fused` | 1.90 | 48 | **0.000** | **0** |
| `rms_norm_q` | 0.81 | 72 | 0.60 | 48 |
| `rope` | 0.68 | 96 | 0.56 | 48 |
| `quantize` | 0.62 | ~50 | 0.61 | ~49 |
| `swiglu_2d` | 0.50 | 24 | 0.50 | 24 |
| `add_into` | 0.25 | 48 | 0.25 | 48 |
| `copy_2d` | 0.17 | 72 | 0.14 | 48 |
| `store_2d_pair` | 0.10 | 48 | 0.06 | 24 |
| **total** | **5.03** | | **~2.7** | |

(These per-kernel figures are stable across five independent GPU captures
taken this session, warm and cold, before and after this beam's changes --
they agree to within a few percent of each other consistently, which is the
basis for trusting the corrected total despite the run-to-run noise discussed
further down.)

Against llama.cpp's ~2.18ms non-GEMM-non-attention tail (the number the brief
cited), the real addressable gap this beam could go after was **~0.5ms, not
~2.85ms**. Reported this to the orchestrator when found; they confirmed it
was their own mapping error and re-ranked the beam's expected value
accordingly, but asked for the timing capture to proceed anyway.

Also worth noting for whoever reads this next: `rms_norm_q` and `quantize`
both being fed by decode contamination almost exactly cancels in the
"~50/pass" `quantize` row (pure prefill contributes 48, decode contributes
close to 0, the brief's 50 came out nearly right by coincidence) -- a reminder
that a table matching a plausible-sounding total is not the same as being
uncontaminated.

## `copy_2d`'s calls, and why 72 (now 48, corrected) exist

Traced every call site in `phobos-gguf/src/llama.rs::attention` (minicpm's
only mixer; the qwen35 delta-net paths in `phobos-gguf/src/qwen35.rs` are a
different architecture minicpm never takes):

1. **Q's split-out** (`rows > 1` only): the fused QKV projection interleaves
   Q, K and V column-wise, so past one row Q is not contiguous and has to be
   pulled into its own dense buffer before rotation. At `rows == 1` this is
   skipped entirely -- Q happens to be the front of the buffer, so rope runs
   on `qkv` directly.
2. **K's split-out** (unconditional, every row count): same interleaving
   problem, but `Backend::rope` rotates in place from offset 0 of whatever
   buffer it's given, and K sits at a nonzero offset in `qkv`. Unlike Q there
   was no row-count escape hatch, which is exactly why the tail-bucket
   contamination's decode windows still show one `copy_2d`/layer (`k`'s) even
   though prefill's Q-copy vanishes at `rows == 1`.

So `copy_2d`'s role here is structurally required *only* because rotation and
extraction were two separate launches; it is not dead work, a layout
oversight or a leftover from an earlier design. It is exactly the shape
`rope_gather` (below) was built to remove: read the strided window directly
inside rope instead of extracting it first. After this beam's shipped change,
prefill issues zero `copy_2d` calls from `attention()` (both Q and K route
through `rope_gather`), and decode drops from one to zero as well (K no
longer needs its own copy). `store_2d_pair`'s existing fused Q/K store into
the KV cache was untouched -- it already reads K post-rotation from its own
dense buffer, which `rope_gather`'s `dest` still provides.

## Shipped: `rope_gather`

**Target**: the `copy_2d` calls feeding `rope`, per the analysis above, plus
`rope`'s own separate read/write pass over the same data.

**What shipped**: `Backend::rope_gather` (new trait method, default =
`copy_2d` + `rope`, so the host backend and any backend without a fused
kernel get correct, if unfused, behavior for free). The device override is a
new `rope_gather` kernel (`phobos-gguf/src/backend/device/kernels/attn.rs`)
that reads a strided window of the fused QKV buffer directly and writes a
dense, rotated destination in one launch. The strided read uses a
block-reshape trick: the source is declared at `head_dim` granularity with a
`stride_heads` (source pitch / `head_dim`) baked into the generated source,
so a program only ever touches the `heads` slots its caller's pointer offset
already points at, the rest of that physical row (the other of Q/K/V)
correctly falling outside its addressed range.

**Partial rotary -- caught a gap in model-level coverage before it shipped**:
minicpm's `rope_dim == head_dim` (128 = 128, no passthrough channels), and
`rope_gather` is only ever called from `llama.rs::attention`, minicpm's
mixer. Qwen3.5 has the partial-rotary shape this kernel's passthrough branch
exists for (`rope.dimension_count = 64` against `attention.key_length = 256`
-- 192 of each head's 256 channels pass through unrotated), but Qwen runs
`qwen35.rs::attention`, an entirely different mixer this beam never touched:
its rope calls run on `q_normed`/`k_normed`, dense buffers already produced
by a prior QK-norm step, so they take plain `Backend::rope`, never
`rope_gather`. Qwen's model-level gates passing is real evidence for this
fusion's K-path launch-count reduction (Qwen's own attention still benefits
from the K-path change) but **zero evidence for the passthrough branch
specifically** -- a first draft of this report claimed otherwise, which was
wrong; caught on a second pass before shipping.

Fixed by adding two direct cases to `backend_check.rs` rather than relying on
either model: a synthetic strided source shaped like a fused QKV buffer
(`stride_heads` slots of `head_dim` each, gathering `heads` of them from a
nonzero offset `slot`), one shape with `rope_dim == head_dim` (tail = 0) and
one with `rope_dim < head_dim` (tail > 0, `[9 x 4 x 64/32 @ slot 4 of 8]`),
comparing the device kernel against the trait's own default (`copy_2d` +
`rope`, which `HostBackend` inherits unmodified). Both came back `rel err
0.000e0` against the host oracle -- exact, not just within tolerance, which
is the expected result for a pure-arithmetic rotation with no reduction to
reorder. This exercises the tail>0 passthrough copy, the nonzero-offset
pointer math, and the `stride_heads` reshape together, independent of which
model happens to have which rope shape.

`llama.rs::attention` now calls `rope_gather` for K unconditionally (dropping
the row-count asymmetry noted above -- K no longer needs its own `copy_2d` at
any row count, a small decode-side bonus alongside the prefill-side one) and
for Q only when `rows > 1` (the `rows == 1` alias-into-`qkv` path stays, since
it's cheaper still: no extra allocation at all). minicpm's `fuse_check`
exercising 32 decode steps through the K path (tail = 0, since minicpm has no
passthrough range) matching bit-for-bit is genuine coverage of that half;
it's the tail > 0 half that needed the direct test above.

**Correctness gates, both models, `--release --features cuda`,
`PHOBOS_ATTN_PERSIST=1`** (final shipped state, fusion 2 only):

| gate | minicpm5-1b-Q8_0 | Qwen3.5-0.8B-Q8_0 |
| --- | --- | --- |
| `backend_check` (worst rel err, whole suite, incl. 2 new `rope_gather` cases at `rel err 0.000e0`) | 2.902e-4 | (shared run) |
| `batch_check` spread err | 0 (host), 1.128e-2 to 1.350e-2 (gpu) | 0 (host), 7.575e-3 to 1.540e-2 (gpu) |
| `model_check` spread err | 1.258e-2 to 1.763e-2, tokens agree | 7.097e-3 to 1.074e-2, tokens agree |
| `fuse_check` | prompt exact, decode max 1.332e-2 / avg 9.944e-3, 0 flips | prompt exact, decode max 1.051e-2 / avg 8.085e-3, 0 flips |

All numbers match this session's documented baseline band exactly
(`backend_check` 2.902e-4 is the session's own recorded figure, unchanged --
it returns to exactly this number once fusion 1's extra test cases are
removed, see below). Also ran `cargo test -p phobos-gguf -p phobos-onnx -p
phobos-inference -p phobos-kernels` (also the only check of the non-`cuda`
build in this beam's work): 159 passed, 0 failed, 1 ignored.

**Measured timing** (the point of the GPU slot): built two release binaries
from the same worktree via `git stash`/`stash pop` around
`phobos-gguf/src`+`examples/backend_check.rs` -- one with this fusion, one
without -- and captured each with `nsys profile --trace=cuda
--cuda-graph-trace=node` against `bench.exe -p 128 -n 0 -r 5`, warming the
card first with a longer untraced run (idle GPU clock was 375MHz against a
2145MHz boost ceiling; an unwarmed capture measures the ramp, not the
kernel -- see `[[gpu_contention_invalidates_benchmarks]]`/attndecode's own
warm-up discipline). Applied this beam's own window-isolation method (not the
brief's contaminated stats-pull) to both captures to isolate genuine prefill
windows.

Even warm, individual per-kernel numbers still carry real run-to-run noise --
`rms_norm_q`, a kernel this beam never touched, was seen swinging between
0.55ms and 0.95ms/pass across otherwise-identical captures of the same
binary, and one rep in five threw a 3x outlier on an unrelated kernel in one
run. Treating the **whole tail bucket's median across five timed reps** as
the stable quantity (individual small-kernel deltas are not trustworthy at
this granularity without far more repetitions than a beam's timing slot
affords):

| | before | after (fusion 2) |
| --- | --- | --- |
| median tail ms/pass (5 timed reps) | 2.744 | 2.606 |
| per-kernel sum, 4 stable reps each (excludes one 3x outlier rep on the after side) | 2.706 | 2.605 |

**~0.10-0.14ms/pass genuine improvement**, consistent across both the
whole-bucket-median view and the per-kernel-sum view. Mechanism matches
expectation exactly: `copy_2d` (0.14ms/pass) is fully gone, and `rope`
(0.56ms/pass) becomes `rope_gather` at a nearly identical 0.57ms/pass (it now
does strictly more work per launch -- reading the strided source and writing
the dense destination in one kernel -- so it should cost about what `rope`
alone did, and it does). This is a clean case of *deleting a whole kernel's
read+write pass without adding anything to what remains*, which is why it
measured as a straightforward win where the next section's fusion did not.

## Reverted: `swiglu_2d_q`, a measured wash-to-small-loss

**What was built**: `Backend::swiglu_planes_q`, a quantizing epilogue on
`swiglu_planes` mirroring `Backend::swiglu_q`'s existing `rows == 1`
precedent, targeting the `quantize` calls feeding the FFN-down projection's
residual add. Correct on every gate: `backend_check` (two new cases
comparing the dense `O` output directly, `rel err 1.192e-7`), and
`batch_check`/`model_check`/`fuse_check` on both models, numbers unchanged
from the pre-fusion baseline. Found and fixed one real bug along the way (a
shared-memory budget miscalculation caused a PTX JIT load failure on Qwen's
`d_ff = 3584`, minicpm's `4608` happening to dodge it) before it was
correctness-clean.

**Then measured, and it doesn't pay off.** Per-launch, across three
independent warm captures: `swiglu_2d_q` averaged roughly 35-42us/launch
against `swiglu_2d` (21us) plus the FFN-down share of the standalone
`quantize` kernel it replaced (roughly 17us) -- a consistent, if noisy, **net
increase of a few microseconds per launch, times 24 launches/pass**, landing
this fusion somewhere between a wash and a small loss rather than a win.
Tried the obvious lever before accepting this: `swiglu_2d_q`'s tile-size
budget (`OPERANDS = 6`, chosen to fit six live shared-memory tiles under the
48KB static ceiling where the plain kernel needs three) was swept to
`OPERANDS = 4` (a wider tile, 2 blocks/row instead of 3, matching the plain
kernel's own width at minicpm's shape) -- this made it measurably **worse**
(`swiglu_2d_q` rose to ~72us/launch), ruling out "tile too narrow" as the
cause and bracketing the regression from both sides.

**Mechanism, read off the kernel's own emitted MLIR** (`cargo run -p
phobos-lang --example emit` on the generated source, the same tool this
session's `BR=32` attention beam used to find a structurally identical
issue): this tile codegen materializes every `var`-declared value through
its own shared-memory tile with a `gpu.barrier` between stages, and never
pools a tile across a different source-level name even when lifetimes don't
overlap (documented independently in
`autoresearch/beams/prefill-attention-tensorcore-wide-br.md`'s "never reached
the tree" postmortem). The quantizing epilogue adds roughly four more such
staged values (the SwiGLU product, the round-scaled copy, and `O`'s and `Q`'s
own staging) on top of the plain kernel's two (`G`, `U`), each one a
barrier-separated pass over shared memory. `swiglu_q`, the `rows == 1`
precedent this beam modeled the fusion on, avoids this cost class entirely:
at decode scale the kernel is tiny and launch-bound (~2.5us fixed cost per
launch dominates), so removing one launch is a clean win regardless of what
the epilogue adds internally. At prefill scale (17-42us kernels, launch
overhead hidden under the much larger `q8_qmma` calls around them), the
per-stage staging cost is what dominates instead, and it isn't free.

**The transferable lesson, stated plainly**: on this codegen, a fusion that
*deletes* a whole kernel's read-and-write pass (like `rope_gather`) wins
close to for free; a fusion that *adds an epilogue stage* to an existing
kernel pays a real, structural, per-stage cost that a decode-scale precedent
can hide but a prefill-scale one cannot. Worth having on record for the next
beam that reaches for a `_q`-suffixed epilogue kernel at prefill scale.

**Reverted.** `phobos-gguf/src/{backend/mod.rs, backend/device/{mod,backend,elem}.rs,
backend/device/kernels/elem.rs, layers.rs}` and
`phobos-gguf/examples/backend_check.rs` are all back to their pre-beam state
(diffed against git to confirm byte-identical). The full diff, as it stood
correctness-verified and gated before the revert, is kept at
`autoresearch/beams/prefill-tail-fusion-swiglu2dq.patch` for reference (it
also still contains fusion 2's changes to the shared files, since both
landed in the same working tree before this write-up separated them --
`git apply --check` against a tree with fusion 2 already applied is not
expected to succeed cleanly; the value here is documentation, not a
reapplicable patch). Not worth chasing further: the tile sweep bracketed the
regression from both sides and the mechanism is structural to how this
codegen stages shared memory, not this kernel's specific tile choice.
Revisit only if the tile codegen learns to pool live tiles across barriers.

## Not attempted: `q8_qmma_add`

The brief's third lead: fold `add_into` into `q8_qmma`'s own epilogue for
`m > 1`, mirroring `q8_qdot_add`'s existing precedent for `m == 1`. Flagged
to the orchestrator before starting (touches `matmul.rs`, the GEMM/stream-K
sibling beam's file) and the orchestrator said to skip it, for three reasons
beyond the file conflict: it's the smallest of the three levers by the
corrected numbers (~0.25ms against an already-small ~0.5ms total addressable
gap, most of which this beam's one shipped fusion likely already covers
proportionally); the existing `matmul_quant_add` comment noting the `m > 1`
case was a deliberate decision ("a prefill goes through the tensor cores,
where the residual add is a rounding error on the pass rather than a launch
that matters") is a real signal worth trusting; and `matmul.rs` now carries
the stream-K beam's already-merged changes, so touching it here would mean
reconciling two independent diffs on the same file for a fusion judged
marginal even before that cost. Not started.

## What's ready for review

**Shipped**: `rope_gather` alone. Diff touches `phobos-gguf/src/backend/mod.rs`
(one new trait method with an unfused default), `phobos-gguf/src/backend/device/{mod,backend,attn}.rs`
and `phobos-gguf/src/backend/device/kernels/attn.rs` (the kernel and its
launch logic, kept out of `backend.rs` as a thin one-line delegating call per
that file's own "the work lives in the sibling modules" convention),
`phobos-gguf/src/llama.rs` (the call site), and
`phobos-gguf/examples/backend_check.rs` (two new cases covering the tail=0
and tail>0 shapes). `cargo clippy --release --features cuda -- -D warnings`
clean on the crate and its examples, `cargo fmt -- --check` clean on every
touched file, `cargo test -p phobos-gguf -p phobos-onnx -p phobos-inference
-p phobos-kernels` clean (159 passed), all four correctness gates green on
both models with numbers matching this session's documented baseline
exactly, and a measured ~0.10-0.14ms/pass improvement to the genuine prefill
tail from two independent measurement methodologies.

**Not shipped, documented as a negative result**: `swiglu_2d_q`, correct but
measuring as a wash-to-small-loss with a structural (not tuning-fixable)
cause. Fully reverted; patch kept for reference.

**Not attempted**: `q8_qmma_add`, per the orchestrator's explicit call.

**Housekeeping note**: mid-session, a PowerShell tool call's working
directory silently reset to the main tree between commands (this
environment's PowerShell tool does not persist `Set-Location` the way the
tool description implies), and one `nsys profile` invocation using a
relative output path landed a stray `prefill_tail_before.nsys-rep` in
`C:\Users\joaeb\code\phobos\autoresearch\beams\` before this was caught.
Deleted immediately; every subsequent PowerShell command in this session
explicitly `Set-Location`s into the worktree and uses absolute paths for
every argument that touches the filesystem. Flagging in case another agent
working in the main tree noticed the file blip.

GPU timing slot released -- nothing further needs the card. Evidence: two
final capture pairs kept at `autoresearch/beams/prefill_tail_final_before.nsys-rep`
and `autoresearch/beams/prefill_tail_final_after.nsys-rep` (plus their
`.sqlite` exports), five timed prefill reps each, `--cuda-graph-trace=node`,
warmed before capture.
