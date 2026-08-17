---
parent: 8271020 (autoresearch branch, "lang: prefetch warp_partial's K/V one
  iteration ahead")
status: shipped in the tree, not yet committed by this beam (per this
  session's standing rule, the orchestrator's own bench.py confirmation and
  commit come after this report). Correctness-verified on both models; the
  emit-diff sweep found exactly one production loop newly eligible, and it
  is off the pp128/tg1024 hot path. A second, independent sweep -- through
  `compile_shared` itself, not `emit`, see the methodology note below --
  found three stale `@pipeline` annotations that the assertion now turns
  into real compile failures; all three removed, confirmed no-ops before
  removal (byte-identical MLIR) and confirmed compiling clean after. A
  fourth file, `gemm_fp32.ph`, has a real but partial gap (52 of 64
  declared autotune configs pipeline, 12 asymmetric-tile ones do not, same
  latent gap pre-dating this task) left open as a decision for the
  orchestrator rather than resolved unilaterally
---

# Beam: `@pipeline` becomes an assertion, pipelining becomes the default

The user's request, verbatim: make `@pipeline` a default -- "why would
someone have to opt in for more performance? The compiler should be able to
identify this and the annotation should only say 'fail if you cannot use
it'." Confirmed reading: auto-attempt pipelining on every eligible loop,
everywhere, no annotation required; a kernel with no eligible loop silently
takes the plain path (not a failure); `@pipeline`, kept as an explicit
annotation, changes meaning from "opt in to attempting this" to "assert this
kernel pipelines something, or fail to compile."

## The semantic change, and what stayed opt-in

`Codegen`'s `pipeline: bool` field (`phobos-lang/src/codegen/mod.rs`) is
renamed `pipeline_assert: bool` -- still literally "was `@pipeline`
written," but no longer gates *attempting* the generic loop mechanism.
`stmt.rs`'s `emit_for_inner` now always calls `pipeline_candidate`
unconditionally; `@pipeline`'s new role is checked once per kernel in
`emit()` (`mod.rs`), via a `pipelined_any: bool` flag set wherever pipelining
actually happened, and `pipeline_declines: Vec<String>` collecting why each
attempted loop didn't, for an actionable error.

**One thing this pass deliberately left attribute-gated**: the fused-GEMM
backend (`matmul/{plan,reg,wmma}.rs`) reads the same `pipeline_assert` field
to decide its own, *separate* double-buffering (`pairs = if
self.pipeline_assert { 2 } else { 1 }`, for the specialized register-tiled
and tensor-core matmul paths matched by `matmul_candidate` at the
whole-statement level, entirely independent of `pipeline_candidate`). That
mechanism has no legality or shared-memory-budget check of its own -- it
has always blindly trusted the attribute. Flipping it to default-on would
silently double the staging footprint of every production GEMM (`matvec`,
`q8_mma`, and `q8_qmma`, which the orchestrator's own diagnostics this
session already found at 12.45% achieved occupancy against a 25% ceiling --
doubling its shared memory would lower that ceiling further), with none of
the budget/perf analysis this task did for the generic path. Scoped out,
flagged here as the natural next beam if the user wants full parity: it
needs the same two-gap treatment (CTA-uniformity -- trivially satisfied,
see below -- and a real shared-memory budget check) before it's safe to
flip.

`@pipeline`'s assertion is consequently scoped to what it can check: the
`pipeline_candidate` path (`pipelined_any` set in `stmt.rs`'s auto-attempt
branch) and the fused-GEMM path (`pipelined_any` set at the
`matmul_candidate` dispatch site whenever `pipeline_assert` is true, since
that unconditionally means `pairs == 2` for whatever backend runs). A
kernel whose only loop is **fragment-carried** (`emit_frag_for`, the
flash-attention shape: `dot`/`dot_t` calls inside an ordinary loop, threaded
as `scf.for` iter_args) is outside the assertion's domain entirely --
`self.pipeline`/`pipeline_assert` was never consulted anywhere reachable
from `emit_frag_for` before this change, and still isn't. This matters
concretely: `examples/flash_attention_fp16.ph` and `flash_attention_fp32.ph`
both carried `@pipeline` on exactly this shape. It was always inert there
(confirmed against the pre-change codegen too: `diff` against the same
source with `@pipeline` deleted is byte-identical, before this task and
after). Under the *old* semantics an inert attribute was silently harmless.
Under the new assertion semantics it is not: `compile_shared` now bails on
any kernel that wrote `@pipeline` and never set `pipelined_any`, and these
two kernels never do (their loop body is `let k = ...; let v = ...`, not
the `var`-staged prefix `pipeline_candidate` matches, so no leading run is
even found -- confirmed by calling `compile_shared` directly:
`kernel `flash_attention`: @pipeline asserts this kernel can be pipelined,
but nothing in it did: loop `kt`: no leading run of staged tensor-slice
`var` statements`). This is a real, user-facing break, not a hypothetical:
`phobos-bench/src/flash.rs`'s `bench_flash_attention_fp32`/`_fp16` load
these exact files through `phobos-bench/src/autotune::compile`, which calls
`phobos_lang::compile`, which is `compile_shared` under the hood --
`cargo run -p phobos-bench` would have hard-failed on both flash
benchmarks. **Fixed by removing `@pipeline` from both files** (the honest
application of the new semantics: the annotation was asserting something
false, and removing it is a zero-diff change, verified byte-identical MLIR
before/after on both the default target and `PHOBOS_CHIP=sm_80
PHOBOS_INDEX_BITS=64`). A third file had the identical latent problem,
found by the same direct-`compile_shared` check applied systematically to
all nine `examples/*.ph` files rather than trusting the emit-based sweep
(see the methodology note below): `examples/gemm_fp16.ph`. Its `@pipeline`
sat on the fused-GEMM shape (`var acc`, `for kt`, two staged `var` slices,
`dot`), which should route through `matmul_candidate` and be exempt from
this task's changes entirely -- but `matmul_candidate`'s tensor-core branch
only matches when `self.has_wmma() && self.wmma_plan(...).is_some()`, and
its vector-path fallback requires `acc_scalar == F32`, which this kernel's
`f16` accumulator fails; on this codebase's default chip (`sm_75`) neither
branch fires for this exact shape, so the kernel falls through to the
generic per-loop path, where `kt :+ TILE_K`'s offset is judged partial for
the same structural reason as the headline finding below. Confirmed this is
pre-existing behavior, not new: `@pipeline` was already inert for this file
on both `sm_75` (default) and `sm_80` before this task, by the same
before/after diff check. Fixed the same way -- attribute removed, byte-
identical MLIR confirmed on both targets, and `compile_shared` now succeeds
on all nine example files on both targets (verified with a small throwaway
harness, not left in the tree). Extending real pipelining to
fragment-carried loops (prefetching the next chunk's K/V while computing
softmax) is a plausible, larger follow-up -- it would need new staging
machinery inside `frag.rs`, not just flipping a flag -- explicitly out of
scope here per the task's own instruction not to chase a broader eligible-
loop class. Flagged, not chased.

## A second methodology gap, worse than the first: the emit-diff sweep cannot see `@pipeline` assertion failures at all

The temporary sweep tool used below (and `phobos-lang/examples/emit.rs`,
the tool `cargo run -p phobos-lang --example emit` wraps) calls
`phobos_lang::codegen::emit` directly, not `phobos_lang::compile_shared`.
That is precisely the function that enforces `@pipeline`'s new assertion
semantics. An `emit`-based sweep can show that a kernel's MLIR is
byte-identical before and after this change -- and it did, correctly, for
`flash_attention_fp16.ph`, `flash_attention_fp32.ph`, and `gemm_fp16.ph` --
while being structurally blind to the fact that the exact same kernel now
fails to *compile* through the real entry point every caller in this tree
actually uses (`phobos-kernels::compile`/`compile_shared`, which both call
`phobos_lang::compile_shared`). The emit-diff sweep answers "did codegen
change"; it does not and cannot answer "does `@pipeline` still hold." Only
discovered this gap because the advisor flagged it before this report was
sent: traced the enforcement path by hand (`pipeline_assert` -> `emit`'s
`pipeline_failures` -> `compile_shared`'s `bail!`), then verified each of
the three affected files empirically against `compile_shared` directly
(one-off throwaway examples, built and deleted, not left in the tree) and
against a five-line grep for `phobos_lang::compile_shared`/`compile\b`
call sites across `phobos-gguf`, `phobos-onnx`, `phobos-kernels`, and
`phobos-bench` to establish the actual blast radius (only `phobos-bench`'s
flash benchmarks were live callers of the two broken flash files; nothing
in `phobos-gguf`/`phobos-onnx` carries `@pipeline` at all, see the onnx
section below). The corrective step for a future sweep like this: verify
through the same function real callers use, not a tool that exists to
print MLIR for eyeballing. Recorded here as a second, distinct
self-correction from the taint-tracking one above -- not caught by my own
testing this time, caught by review before the report went out.

## Gap 1 (CTA-uniformity of loop bounds): proven closed, not checked at runtime

`emit_pipelined_for`'s guards wrap a `gpu.barrier` inside an `scf.if`, which
hangs (not miscomputes) if a CTA's threads take different branches. Auto-
attempting everywhere means the compiler, not a kernel author who
understood their own bounds, has to be the one certain this can't happen.

**First pass got this wrong and is worth recording as a real mid-task
correction.** The initial design added real machinery: an
`atomic_tainted: HashSet<String>` field on `Codegen`, populated at every
`Let`/`Var`/`Assign` site, and a `loop_bounds_may_diverge` check run before
`pipeline_candidate`, on the premise that `atomic_add`'s return value (the
one builtin in this language whose result genuinely differs per lane, since
distinct threads racing the same RMW slot see distinct "previous" values)
could reach a loop bound: `let n = atomic_add(BAR, 0, 1); for i in
range(0, n) { ... }`. Building the test for this construct surfaced that it
does not compile at all: `range`'s `end` lowers through `emit_index`, which
requires the value already be MLIR's `index` type (`expect_index` rejects
anything else), and `atomic_add` returns `i32` (the tensor it operates on
is `i32`). Tracing the two type-unification functions confirms this is not
an accident: `Codegen::coerce` converts `index` to an integer type
(`index_cast`) but has no arm the other way, and `Codegen::unify` (what a
binary op's operands go through) only widens between float types, bailing
on any int/index mismatch. So neither `atomic_add(...)` directly nor
`atomic_add(...) * 1` (forcing a binop against an index-typed literal) can
ever become index-typed -- both fail to compile, the first at
`expect_index`, the second at `unify`, and both failures are pinned as a
regression test now (`codegen::tests::pipeline::atomic_add_cannot_reach_a_loop_bound`).

Given that, the runtime taint-tracking machinery was provably dead code --
~80 lines that could never fire, on every scalar binding in every kernel,
in a codebase with a line-count ratchet. Removed. What replaced it is a
block comment at the end of `pipeline.rs` giving the full, closed inventory
of what can produce an `index` value in this codegen (integer literals,
`program_id`/`block_id`, a tensor's shape via `memref.dim`, `warp_partial`
and `grid_barrier`'s own returns -- both literally
`Rv::Scalar(self.const_index(block, 0))`, a compile-time constant, not
data-dependent -- and arithmetic over those), and naming the two concrete
ways this could stop being true: an int-to-index conversion added to
`coerce`, or `warp_partial`/`grid_barrier` (in `tile/warp_attn.rs` and
`sync.rs`, files this change does not own) changed to return something
other than a literal zero. The regression test is the tripwire for both --
if either happens, it goes red before anyone ships a hang.

## Gap 2 (shared-memory budget): a real legality check, validated on a real kernel

`pipeline_candidate` (`phobos-lang/src/codegen/pipeline.rs`) now returns
`Result<(staged, rest), PipelineDecline>` instead of a bare `Option`, with
variants (`NoStagedPrefix`, `PartialSlice`, `StagedNameWritten`,
`UnknownElementWidth`, `SharedMemoryBudget { needed_bytes, limit_bytes }`)
that `Display` into the reason strings an `@pipeline` assertion failure
reports. The budget check sums each staged slice's byte footprint (16-byte
aligned, matching `alloc_tile_shaped`), doubles it, and declines if
`self.shared_bytes + doubled` exceeds a `PIPELINE_SHARED_LIMIT_BYTES = 48 *
1024` constant -- duplicated from (can't depend on)
`phobos_kernels::launch::STATIC_SHARED_LIMIT`, since `phobos-kernels`
depends on `phobos-lang`, not the reverse; kept in sync by hand, documented
as such. Conservative on two axes, both documented in the code: it ignores
that a released buffer can be reused from the pool (so it can decline a
loop that would in fact fit), and it is blind to static
(non-`@dynshared`) globals, which this codegen has never tracked in
`shared_bytes` at all -- that blind spot degrades to ptxas's own "uses too
much shared data" error at compile time, not a hang, so it is safe, just
not caught early with a good message.

This is not a theoretical check. `delta_scan_src`
(`phobos-gguf/src/backend/device/kernels/delta.rs`, the chunked delta
rule's second pass, `@dynshared`, and per its own doc comment already at 42
of the 48 KB static shared memory ceiling) has a `for c in range(0, N, C) {
var k = K[c :+ C, qc :+ D]; ... }` loop whose leading `var k` is exactly
`pipeline_candidate`'s staged-prefix shape. It is declined -- correctly --
by this new check with the representative parameters this sweep used
(heads=16, head_dim=128); the emit-diff sweep below confirms zero change to
this kernel's output. Before this change, doubling `k`'s buffer had no gate
at all; now it has one, and it is doing real work on a kernel this session
already spent a full round on (`autoresearch/beams/delta-rule-*` history --
"two rewrites measured worse, only chunking is left").

## `Variants`/masked-fallback resolution

`phobos-kernels`'s `Variants::compile` (`phobos-kernels/src/compile.rs`)
builds an aligned and a masked-fallback variant of one kernel source by
substituting `{ALIGNED}` with two different attribute strings. The masked
variant's slices are partial by construction -- that is what makes it the
fallback -- so `pipeline_candidate` correctly declines to pipeline it. If
`@pipeline` sat in the *shared* part of such a source (outside the
`{ALIGNED}` slot, so both variants carry it textually), a naive per-compile
assertion would fail to compile on exactly the variant that was never
supposed to pipeline.

Resolved by adding `phobos_lang::compile_raw` (`phobos-lang/src/lib.rs`), a
version of `compile_shared` that returns every kernel's raw
`(code, shared, pipeline_failures)` instead of converting an unsatisfied
assertion into an `Err` on the first kernel that has one.
`phobos_lang::compile_shared` (the entry point every other, non-`Variants`
caller here still uses -- direct `.ph` compiles, `phobos-kernels`'s own
`compile()`/`compile_shared()`/`compile_in()` used by every non-`Variants`
kernel in `phobos-gguf`) wraps `compile_raw` and still converts to a hard
`Err` by default, naming the kernel and the decline reasons, so ordinary
single-source callers keep the "fail to compile" guarantee unchanged.
`Variants::compile` instead calls `compile_raw` on both the aligned and the
general source, and only bails if **both** report the same kernel
unsatisfied (aggregated once, after both compiles, not once per compile) --
implementing the "satisfied if any variant pipelines" rule precisely.
No kernel in the tree exercises this today (no `Variants::compile` call
site combines `@pipeline` with `{ALIGNED}`), so this is forward-looking
machinery, exercised by reasoning and by the type system (it compiles and
the existing `Variants`-based kernels -- `matmul`, `matvec`, `q8_dp4a`,
`q8_mma`, `q8_split` -- are confirmed byte-identical in the emit-diff sweep
below, all five instantiated through both `{ALIGNED}` substitutions), not
by a dedicated integration test.

## The emit-diff sweep: what actually changes, and why the mechanism barely fires

Per CLAUDE.md's rule that a codegen-generating change is not a refactor
unless verified as one: every `.ph` file in the tree (`examples/*.ph`,
`phobos-lang/examples/*.ph`, 11 files) and every Rust-embedded kernel
source in `phobos-gguf/src/backend/device/kernels/*.rs` (36 representative
instantiations, covering `matmul.rs`, `quant.rs`, `elem.rs`, `norm.rs`,
`argmax.rs`, `delta.rs`, and `attn.rs`, extracted via a temporary
`#[cfg(test)]` module that dumped each source to a `.ph` file and reused
`emit`, deleted before this report per the plan) went through `cargo run -p
phobos-lang --example emit` under both the default target and
`PHOBOS_CHIP=sm_80 PHOBOS_INDEX_BITS=64`, before and after, diffed. The
`{ALIGNED}`-templated sources (`matmul`, `matvec`, `q8_dp4a`, `q8_mma`,
`q8_split`) were each instantiated both ways, matching their real
`Variants::compile` call sites in `phobos-gguf/src/backend/device/mod.rs`.

**Headline finding, not a footnote**: across all 47 compiled instances,
exactly **one** loop newly pipelines. The reason is structural, not
incidental, and worth stating plainly since it is the real answer to "why
doesn't this fire more": `Codegen::expr_div` (`util.rs`), used by
`dyn_in_bounds`/`dim_in_bounds` to decide whether a slice offset is
provably tile-aligned, defaults an **unbound** name's divisor to 1 (`None
=> self.shape_env.get(name).map_or(1, ...)`). `pipeline_candidate` runs as
a prescan of a loop's body *before* the loop's own induction variable is
bound in scope. The canonical staged-prefix shape this whole mechanism
targets -- `var k = K[kt :+ BC, ...]`, the tensor's own walked dimension
offset by the loop variable itself -- is therefore judged partial
regardless of `@aligned`, because `@aligned`'s promise only raises
`extent_div` (the tensor side), and `dyn_in_bounds`'s check is `extent_div
% size == 0 && expr_div(start) % size == 0` -- an `&&`, and the second
operand can never clear 1 for any `size > 1` under the current prescan.
This is true whether or not `@pipeline` is written; it was already true
before this task, for any kernel author who wrote `@pipeline` on this exact
shape. **A modest-effort follow-up exists and is deliberately not chased
here**: `loop_bounds` already computes `iv_div` (the induction variable's
own provable divisor, from `start`/`step`) for the ordinary path;
`slice_is_partial_within`'s `ivs`/`pending` plumbing already exists (built
for `emit_split_for`'s trimmed-loop prescan) and would, if threaded through
`pipeline_candidate`'s own caller with the about-to-be-bound `var`, make
every `@aligned`, loop-var-offset k-loop genuinely eligible. That is the
real lever this task's diagnostics found and left for whoever picks it up
next.

The one loop that does clear this bar: **`attention_src`'s single-key
remainder tail** (`kernels/attn.rs`, `attention_src` -> `for j in
range(full, visible, 1) { var k1 = K[j :+ 1, col :+ D]; var v1 = V[j :+ 1,
col :+ D]; ... }`, live in `DeviceBackend::attention_rows`
(`backend/device/attn.rs`), the fallback the dispatcher
(`backend/device/backend.rs`'s `attention`) reaches only for a multi-row
prefill continuation that is both unaligned to the block tile *and* does
not tile evenly for the GEMM-decomposed path -- neither `bench.py`'s
`pp128` (fresh prompt, aligned, takes `attention_blocked`) nor `tg1024`
(single-row decode, takes `attention_decode`/`warp_partial`) reach it). It
clears the prescan specifically because its offset dimension has size 1:
`dyn_in_bounds`'s general formula reduces to `extent_div % 1 == 0 &&
expr_div(start) % 1 == 0`, both trivially true for any divisor mod 1 --
alignment and the unbound-loop-var limitation above are simply moot at
size 1. This is a second, narrower general fact worth recording alongside
the headline one: **any staged slice whose dynamic-dimension span is
exactly 1 element is unconditionally eligible**, independent of alignment.

Two apparent diffs in the sweep (`attention_split_src`, `attention_persist_src`)
were investigated and are **not attributable to this change**: both
kernels' `.ph` text has zero `for` statements outside `warp_partial`'s own
internal staging (confirmed by grep of the generated source), so
`pipeline_candidate` is never invoked for either. The observed diff was
traced to a concurrent, unrelated edit landing in the shared tree mid-sweep
(`8271020`, "prefetch warp_partial's K/V one iteration ahead," touching
`phobos-lang/src/codegen/tile/warp_attn.rs`, which both kernels route
through) -- confirmed by re-running the same "after" sweep a second time
and finding it now matches the first "after" run exactly, while still
differing from the pre-change baseline only in those two files. Consistent
with this session's own documented lesson on shared-tree races
(`autoresearch/beams/BEAMS.md`, "Both beams landed by direct takeover"
entry): did not attempt to git-stash the whole tree to isolate this,
stashed only the specific files this task touched
(`git stash push -- <paths>`), leaving the concurrent work untouched.

**Historical precedent, and why it needs a caveat, not a citation**: this
session previously measured `@pipeline` on `attention_split`'s (pre-
`warp_partial`) `var k`/`var v` loop as "flat, within noise" (-0.7% to
+0.2%, `autoresearch/beams/cache-length-split-buckets.md`). By the model
this task's own investigation now establishes, that loop's slices were
`kt`-offset into an unaligned `NK` dimension -- exactly the shape the
headline finding above says the prescan cannot clear. It is plausible that
experiment never actually engaged `emit_pipelined_for` and measured a
no-op, the same trap this task's own first sanity check fell into before
debug tracing corrected it (see below). This does not change the
conclusion for `attention_src`'s remainder loop -- that conclusion rests on
gates passing and the loop being off the hot path, not on the historical
number -- but the historical number should not be read as confirming
"pipelining this shape is safe," since it may not have tested that at all.

## An open question for the orchestrator: `gemm_fp32.ph` pipelines at most, not all, of its declared autotune configs

`examples/gemm_fp32.ph` and `gemm_fp16tc_fp32acc.ph` both still carry
`@pipeline`. Checking one config each was not enough -- both are launched
by `phobos-bench`'s autotuner (`phobos-bench/src/gemm.rs`), which compiles
the *same source text* once per point in its `@autotune`-declared search
space (`phobos_lang::search_space` expands `TILE_M in [32, 256]` to
`{32, 64, 128, 256}` by doubling, not a two-point set -- confirmed by
reading `ast::search_choices`), each with different `shape_overrides`. So
`@pipeline`'s assertion is really being checked once per (kernel,
autotune-config) pair, not once per kernel source. Enumerated both files'
full search space (`cartesian_product` over `search_space`, mirroring
`autotune::compile`'s exact context construction including its
`requires_wide_index` widening) through `compile_shared` on both targets,
with a throwaway harness (not left in the tree):

- `gemm_fp16tc_fp32acc.ph`: 48/48 configs pass, both targets. Clean.
- `gemm_fp32.ph`: **52/64 configs pass, 12 fail, identically on both
  targets.** The 12 failures are exactly the asymmetric tile shapes,
  `TILE_M=128,TILE_N=256` and `TILE_M=256,TILE_N=128`, crossed with every
  `TILE_K` in `{4, 8, 16, 32}`. `matmul_candidate`'s tensor-core branch
  does not apply (no `@tensorcore` on this file) and its vector-path
  fallback further requires a `sub_tile`/`lane_grid` split that apparently
  has no valid answer for a 128x256 (or 256x128) tile at this launch
  config, so the kernel falls through to the generic per-loop path, which
  declines the same `kt`-offset `PartialSlice` as everywhere else in this
  sweep.

**This is not a regression this task introduced.** Checked the identical
`TILE_M=128 TILE_N=256 TILE_K=16` instantiation against the pre-change
codegen (git-stashed): `@pipeline` was already fully inert for that exact
config before this task too -- same byte-identical-with-or-without-the-
attribute proof as the three files above. What changed is only that the
assertion now *notices*. The other 52 configs genuinely do pipeline
(matched `matmul_candidate`, doubled buffers, confirmed by the same
mechanism validated everywhere else in this report), so this is not a
case of a uniformly-dead annotation -- stripping `@pipeline` here, the fix
applied to the three files above, would throw away a real, working
assertion for the 52 configs where it holds.

**Operational consequence, traced through `phobos-bench/src/autotune.rs`**:
the autotuner's stage-1 search loop (`Autotuner::run`) already tolerates a
per-config compile failure -- `probe_short`'s `Err` is caught by the
caller's `match` and logged as `"skipped (...)"`, not propagated -- so an
unpinned `cargo run -p phobos-bench --bench gemm_fp32` would not abort: it
would search the remaining 52 configs and pick a winner among those,
silently smaller than the 64-point space it swept before this task. The
sharper case is a **pinned** run: `--autotune "TILE_M=128 TILE_N=256
TILE_K=16"` narrows the search space to that single point via
`autotune::pin`, and if that single point is one of the 12, stage 1's
`candidates` ends up empty and `anyhow::ensure!(!candidates.is_empty(),
"autotune: no config works")` fires -- a clean, readable error, not a
panic, but a hard failure of a run that would have silently compiled
(just without pipelining) before this task.

**Deliberately not resolved unilaterally.** Two readings of `@pipeline`
are both defensible and this task's brief did not anticipate an autotuned
kernel needing to pick between them: (a) "this kernel source can be
compiled-and-pipelined for at least one shape in its own declared search
space" (true here, 52/64), versus (b) "this kernel pipelines for whatever
specific shape a caller instantiates it at" (false for these 12, and this
is the reading the assertion currently, mechanically implements, since it
runs once per `compile_shared` call and autotuning calls it once per
config). Fixing this the way the three files above were fixed -- deleting
`@pipeline` -- would discard a real, working per-config assertion for the
majority of the space. Narrowing `matmul_candidate`'s `sub_tile`/`lane_grid`
rejection for asymmetric tiles is a real fix but is register-matmul-backend
work this task's scope explicitly excludes (see "what stayed opt-in"
above). Left `@pipeline` in place on both files -- `gemm_fp16tc_fp32acc.ph`
is unconditionally safe, and `gemm_fp32.ph`'s narrowed autotune space is
functionally survivable (the tolerant search path) even though a pinned
CLI run on one of the 12 configs would newly fail. Surfacing this
precisely, with the exact failing configs, is this report's job; deciding
whether that pin-time failure needs a fix (in `sub_tile`, in the assertion
granularity, or by pinning `phobos-bench`'s own defaults away from those
12 points) is the orchestrator's call.

**Orchestrator's resolution**: keep `@pipeline` on `gemm_fp32.ph` exactly
as it stands, no further change. The two readings this report weighed are
not actually in tension once the unpinned/pinned split is taken seriously:
under normal (unpinned) autotune search -- the common case, and what
`cargo run -p phobos-bench` actually exercises, independently re-run and
confirmed clean end to end (`phobos gemm_fp32: ... GFLOP/s`, winner found,
"check: ... correct", no errors) -- the tolerant skip-and-continue path
already handles the 12 declining configs exactly like any other compile
failure, silently and safely. A caller that pins one of those 12 configs
specifically getting a clean, actionable failure is not a regression to
route around; it is the user's own explicit request ("fail if you cannot
use it") firing correctly on the one case where it was always true and
previously silent. Narrowing `sub_tile`/`lane_grid` for asymmetric tiles
remains a legitimate follow-up if someone wants those 12 configs to
actually pipeline, but is unrelated register-matmul-backend work, not a
gap in this task.

## phobos-onnx: inspected, not diff-swept, and why that is sufficient

`phobos-onnx` has no `@pipeline` anywhere (`grep -rn "@pipeline"
phobos-onnx/src/` is empty), so there is no assertion-break risk to check
there. For the auto-attempt side (does any onnx-lowered kernel newly
pipeline), its kernel-source builders are few enough to read directly
rather than build a second dump harness: `phobos-onnx/src/backend/device.rs`
(`layernorm_src`) has no `for` loop at all, so `pipeline_candidate` is never
reached. `phobos-onnx/src/lower.rs` has two loop-bearing shapes, each
appearing twice (`lower_matmul`/`lower_fused_linear` share the GEMM shape;
`lower_flash_attention` is the flash shape), both already characterized
above: the GEMM shape (`var acc`, `for kt`, two staged `var` slices,
`dot`) matches `matmul_candidate` and is confirmed, not just inspected,
byte-identical before/after by extracting `lower_matmul`'s exact generated
source into a standalone `.ph` file and diffing its `emit` output against
the pre-change codegen (stashed and restored via `git stash push -- <the
phobos-lang/phobos-kernels files this task touched>`); `lower_fused_linear`
is textually the same shape plus an epilogue, same conclusion applies by
inspection. The flash shape (`let k = K[...]`; `let v = V[...]`, not `var`)
declines `pipeline_candidate`'s prescan the same way the two
`examples/flash_attention_*.ph` files do, and carries no `@pipeline`, so
there is nothing to newly pipeline and nothing to assert. `phobos-onnx/src/backend/chain.rs`
and `transform.rs` were also grepped for `for `/`kernel `/`_src(` and
contain only host-side Rust loops, no additional kernel-source builders.

## A methodology note worth keeping: MLIR text alone does not prove which mechanism produced it

The first empirical check run in this task (a synthetic bare kernel,
`var a = A[pm * T :+ T, kt :+ T]; acc += a`, no `@aligned`) appeared to
pipeline: two same-shaped `memref.global` tile buffers, a guarded `scf.if`
prefetch, closing barriers. It did not. Debug tracing (`eprintln!` at each
`PipelineDecline` return and at the final `Ok`, removed before this report)
showed `pipeline_candidate` declining with `PartialSlice` for that exact
kernel. The two buffers came from `emit_split_for`'s pre-existing,
unrelated main-loop-plus-masked-remainder structure: a declined,
dynamically-bounded, loop-var-offset slice into an unaligned dimension
triggers `needs_ragged_epilogue`, and `emit_split_for` emits `body`'s
statements twice (once trimmed, once masked), independently minting a
same-named, same-shaped tile buffer each time -- structurally
indistinguishable from genuine double-buffering by shape and count alone.
A **pre-existing test**, `generic_pipeline_double_buffers_a_non_matmul_loop`
(unmodified logic and assertions, `@pipeline` written, predates this
session), was shown by the same tracing to exercise exactly this
incidental path, not the pipeline mechanism its name and comment claim --
its comment is corrected in place; the test itself is left as-is since it
still exercises real codegen and retitling belongs with whoever next
touches `emit_split_for`. The new tests added for this task
(`bare_kernel_auto_pipelines_without_the_attribute` and
`shared_memory_budget_declines_silently_without_the_attribute`; the third
new test, `atomic_add_cannot_reach_a_loop_bound`, replaced an earlier
taint-tracking test that no longer applies once that machinery was found
dead, see Gap 1 above) assert on a **third** buffer's presence or absence
specifically, since `acc` and a single `a` buffer alone already reach two
globals in an eligible kernel and would pass a naive "two buffers exist"
check whether or not real double-buffering happened.

## Design decisions, stated plainly

- **Scope of "eligible everywhere"**: every `.ph`-level `for` loop the
  codegen ever lowers through `emit_for_inner`'s ordinary (non-fragment-
  carried) path, using `pipeline_candidate`'s existing legality shape plus
  the two new checks above -- not a broader class of "staged" value. The
  fused-GEMM backend and fragment-carried loops are explicitly out of this
  task's scope, per above.
- **`@pipeline`'s failure granularity**: once per kernel (via `pipelined_any`
  aggregating across every loop and dispatch site the kernel contains), not
  once per loop -- a kernel with one eligible loop and nine ineligible ones
  satisfies the assertion.
- **Decline reasons are structured, not a bare `None`**: `PipelineDecline`
  is an enum with a `Display` impl; an `@pipeline` failure names the actual
  reason(s) for every loop that was attempted.

## Verification

- `cargo test -p phobos-lang`: 155 passed (151 pre-existing + 4 new:
  `bare_kernel_auto_pipelines_without_the_attribute`,
  `pipeline_assertion_fails_with_the_decline_reason`,
  `shared_memory_budget_declines_silently_without_the_attribute`,
  `atomic_add_cannot_reach_a_loop_bound`), 0 failed. All three pre-existing
  `pipeline`-named tests plus `tensorcore_pipelines_f16_staging` and
  `tensorcore_drops_padding_only_when_maxregs_admits_a_cta` pass unmodified
  (they exercise the fused-GEMM backend or genuinely-eligible generic
  loops, both unaffected in behavior by this change since they already
  wrote `@pipeline` and already pipelined).
- `cargo test --workspace`: every crate green (the `EmitOutput` signature
  change on `codegen::emit` touches call sites in `phobos-cluster` and
  `phobos-sched`, both discard-the-value call sites, unaffected). Re-run
  after the three `@pipeline` removals above; still green.
- All nine `examples/*.ph` files run through `phobos_lang::compile_shared`
  directly (a throwaway harness, not left in the tree) on both the default
  target and `sm_80`/64-bit: all nine `OK` after the three fixes above (all
  nine were `OK` before too, except the three now-fixed files, which failed
  with the `@pipeline` decline-reason error on both targets).
- `cargo build --release -p phobos-gguf --features cuda --examples` and
  `cargo clippy --release -p phobos-gguf --features cuda --examples -- -D
  warnings`: clean. `cargo clippy --workspace --all-targets`: clean (one
  pre-existing, unrelated `unused_imports` warning in
  `phobos-gguf/src/backend/fuse/mod.rs`, not touched by this task).
- Four standing gates, `PHOBOS_ATTN_PERSIST=1`, both models:
  `backend_check` worst relative error **2.902e-4**, identical to the
  documented pre-existing figure -- and its own shape sweep includes `(5
  rows @ 31, 16 heads / 2 kv, head_dim 64)`, which dispatches to
  `attention_rows` (5 > 1 rows, `31` not block-aligned, doesn't tile for
  the GEMM path) -- i.e. the exact newly-pipelined kernel is exercised and
  passes at rel err 1.043e-7. `batch_check`, `model_check`, and
  `fuse_check` all agree on both `minicpm5-1b-Q8_0.gguf` and
  `Qwen3.5-0.8B-Q8_0.gguf`, matching the error bands already on record
  (fuse_check: minicpm max 1.332e-2/step-31, 0 top-token flips; Qwen max
  1.051e-2/step-11, 0 top-token flips).

**Timing: not run, and here is the honest framing for the orchestrator.**
Per this session's standing rule, `scripts/bench.py` is the orchestrator's
own next step. This task's own quick-check obligation (nsys/isolated
kernel timing on the one kernel that changed) was assessed and
deliberately not run: `attention_src`'s newly-pipelined loop is a size-1-
slice tail handling at most `BC - 1` leftover keys, reachable only from a
fallback the pp128/tg1024 shapes `bench.py` measures never take. Building a
synthetic harness to exercise a misaligned-continuation-only code path in
isolation would be new-code work disproportionate to the risk it would be
checking, given correctness is already confirmed on the exact fallback
path via `backend_check`. Correctness verified; performance unmeasured;
unreachable from the shapes the confirmation benchmark covers either way.

## Changed files

`phobos-lang/src/{lib.rs,codegen/{mod.rs,pipeline.rs,stmt.rs,matmul/{plan.rs,
reg.rs,wmma.rs},tests/pipeline.rs}}`, `phobos-kernels/src/compile.rs`,
`SPEC.md`, and three example files whose `@pipeline` was stale and is now
removed: `examples/flash_attention_fp16.ph`, `examples/flash_attention_fp32.ph`,
`examples/gemm_fp16.ph` (each confirmed byte-identical MLIR before/after on
both the default target and `sm_80`/64-bit, and confirmed compiling clean
through `compile_shared` after). Nothing in `phobos-gguf` or `phobos-onnx`
changed (the temporary sweep test file was deleted, and its `mod`
declaration in `phobos-gguf/src/backend/device/kernels/mod.rs` reverted,
before this report).
