---
parent: [[wide-vocab-lm-head]] (the profiled readback cost this targets)
status: implemented, correctness-verified, perf not yet independently confirmed
---

# Beam: device-side argmax for greedy decode

## What this targets

[[wide-vocab-lm-head]] profiled the logits readback (`DeviceBackend::read`,
`backend.rs:123-152`: `stream.synchronize()` then a full device-to-pinned-host
copy of the vocab) at **77,423 ns average, 1.7% of a profiled decode step /
1.9% of `bench.py`'s un-profiled tg32 step**, on minicpm5-1b (see that beam's
own Result log, same nsys trace). That beam killed the *matmul* lever (the
`q8_qdot` kernel feeding the readback is already at 92.8% of the card's
memory-bandwidth roofline, nothing to tune) but explicitly named the
readback itself as a separate, untried lever: "(b) not computing the full
logits vector at all (a sampling-architecture change, out of scope for a
matmul-path beam)." This beam is that change, scoped to the one case where
it is exact rather than approximate: greedy decoding, where the sampler only
ever needs the argmax, not the distribution.

**Ceiling, honestly stated going in**: ~1.7-1.9% of a decode step, not the
12.6-14.2% [[wide-vocab-lm-head]] measured for the *combined* lm_head kernel
+ readback -- the `q8_qdot` kernel itself still runs unchanged, this only
removes the synchronize-and-copy tail. After the couple of microseconds the
new reduction kernels themselves cost (see Mechanism), net is closer to
1.3-1.6%, and at `tg4096`'s longer per-step time that fraction shrinks
further and may sit inside `bench.py`'s own noise floor. This is a real,
verified capability, not a large enough number by itself to be this
session's next primary lever -- record it honestly rather than oversell it.

## Mechanism

**New phobos-lang builtin: `argsel(va, vb, ia, ib)`.** `select(va >= vb, ia,
ib)`, elementwise, the same `cmpf`/`arith.select` shape as the existing
`tmax` builtin but carrying a *payload* (an index) alongside the value
comparison instead of the value itself. This is the one primitive genuinely
missing: phobos-lang had `rowmax`/`tmax` for value-only reductions and
nothing that tracks an index through a fold. Advisor-reviewed before writing
any kernel code specifically to size this addition down from an earlier,
heavier plan (a `rowargmax` reduction primitive plus atomic-based
cross-block combining) to the smallest primitive that composes with what
already exists. Implementation: `phobos-lang/src/codegen/tile/elem.rs`
(`tile_argsel_bc`, the codegen; `emit_argsel`, the call-site glue extracted
into this file rather than `expr.rs` to avoid the workspace's source-size
cap -- see Line-count note below), one dispatch arm in
`phobos-lang/src/codegen/expr.rs`. SPEC.md documents it beside `tmax`.

**Indices travel as f32.** Exact up to 2^24 (16.7M), far past both models'
vocabularies (130,560 and 248,320), which sidesteps needing an integer
reduction path, integer `rowmax`, or an i32 warp-shuffle -- `argsel` and
`tmax` both stay plain float ops. The kernel design folds a `(value, index)`
pair through the reduction with a `tmax`/`argsel` pair at every step: same
comparison, once each, value and payload kept in lock-step.

**Two kernels, both eager (outside the CUDA graph), `phobos-gguf/src/backend/device/kernels/argmax.rs`:**
- `argmax_reduce`: `S` blocks (`ARGMAX_SPLITS = 128`, capped at
  `n / w` so no block ever contributes only its own identity), each
  grid-strides the row in chunks of `W` (`ARGMAX_CHUNK = 512`, or the
  widest power of two at most 512 that divides the vocab --
  `argmax_chunk_width`), folding a running `[1, W]` value tile and index
  tile with `tmax`/`argsel`, then an unrolled halving tree
  (`argsel` before `tmax` at each step, since `argsel` needs both
  un-mutated halves' values) collapses `[1, W]` to `[1, 1]` and writes one
  partial per block to a small scratch buffer.
- `argmax_finish`: one block, a plain serial loop over the (at most 128)
  partials, same `tmax`/`argsel` pair. Small enough that a second
  block-level tree bought nothing.
- Both launch **after `end_pass()`** (i.e. after the CUDA graph that
  produced the logits has already replayed), as two ordinary eager
  launches on the same stream -- not folded into the graph as a third
  node. This was an explicit advisor call: it avoids every question about
  node-arg stability, scratch buffer addresses staying constant across
  replays, and forward/forward_greedy needing different graphs, at a cost
  of maybe 5-6us of launch overhead against the ~50-65us being saved.
- `argmax_chunk_width` always returns a divisor of the vocab (worst case
  1), so `argmax_reduce`'s chunk read is never ragged and never needs a
  masked tail -- important because a masked read's zero fill is not
  `tmax`'s identity (`-3e8`, matching flash attention's own sentinel
  convention; the literal `-3.0e38` scientific-notation form does not
  parse, the grammar's float literal has no exponent form, see the "Two
  small language-level snags" note below), so an all-negative logits row
  would silently lose to a fake zero if a mask ever fired. It does not
  fire here by construction, and the correctness gate below tests the
  all-negative case directly anyway rather than trusting the argument.

**`Backend::argmax(&self, buf, len) -> Result<i64>`** (`phobos-gguf/src/backend/mod.rs`):
default body reads the buffer and reduces on the host
(`phobos_inference::sampling::argmax`, so tie-breaking matches exactly:
last of equal maxima wins) -- free for `HostBackend`, and the fallback any
future backend gets before it writes a device reduction of its own.
`DeviceBackend` overrides it with the kernel pair above
(`phobos-gguf/src/backend/device/argmax.rs`).

**Forward-pass API**: `llama::Model`/`qwen35::Model` each split their
existing `forward` into a shared `forward_to_logits` (everything through the
LM head projection and `end_pass`, returning the device buffers rather than
reading them) plus two thin callers: `forward` (unchanged behavior,
`read_vec` + release) and the new `forward_greedy` (`backend.argmax` +
release, returns `Result<i64>` directly). `model::Decoder::forward_greedy`
dispatches the same way `forward` already does.
`phobos_inference::model::Session` gained `extend_greedy(&mut self, ids) ->
Result<i64>` with a default (`argmax(self.extend(ids)?)`, works for any
existing `Session` impl including ONNX, which this beam does not touch) and
`GgufSession` overrides it to call `Decoder::forward_greedy`. This is the
"real capability, not a benchmark-only path" requirement from the brief:
`phobos_inference::generate::continue_from` now computes
`config.sample.is_greedy_unpenalized()` once
(`SampleConfig::is_greedy_unpenalized`, `phobos-inference/src/sampling.rs`
-- greedy *and* both penalties off, since a penalty rewrites the logits
before argmax and the device kernel only ever sees the raw ones) and takes
the `extend_greedy` branch in the decode loop exactly when it holds,
otherwise the unchanged `extend` + `choose` path. This is the actual
generation loop the CLI, chat rendering and the OpenAI-compatible server all
share, not a bench-only shortcut. `phobos-gguf/examples/bench.rs`'s `tg`
timing loop (and its warmup pass, so the kernels' first compile is not
inside a timed repetition) now calls `forward_greedy` directly, since that
is what a real greedy-serving deployment does.

## Two small language-level snags found and fixed along the way

1. **`coerce` had no index-to-float path.** `phobos-lang/src/codegen/expr.rs`'s
   `coerce` (used when a fused elementwise tree's leaf value needs
   converting to the target tile's element type) handled float-to-float and
   index-to-int, but not index-to-float -- `f32(kt)` inside a fused
   assignment (`var cand: tile<f32>[1, W] = io + f32(kt)`, needed to turn a
   grid-stride loop's index into that chunk's vocab-position offset) hit
   `bail!("type mismatch: cannot store index where f32 is expected")`. This
   is a pre-existing gap, not something new to `argsel`: any future kernel
   doing the same `f32(loop_or_program_id_var)` pattern inside a fused
   assignment would have hit it too. Fixed by delegating that one case to
   `numeric_cast` (which already knows how to route index through i64 to a
   float), a three-line addition. Caught by writing a minimal reproduction
   through `cargo run -p phobos-lang --example emit` *before* wiring
   anything into the GGUF kernel, per the advisor's explicit
   suggestion to verify the exact construct in isolation first.
2. **Scientific-notation float literals do not parse.** `-3.0e38` (an
   attempt to write "very negative" directly, mirroring how the identity is
   written in Rust code elsewhere in this codebase) fails with `expected
   end of statement (newline), found identifier 'e38'` -- the grammar's
   float literal is a plain decimal only (see SPEC.md's own note on this
   next to the RMS-norm epsilon literal, which is why that one is written
   `0.00000001` rather than `1e-8`). Not a bug, a real grammar constraint;
   fixed by using flash attention's own `-300000000.0` sentinel convention
   instead. Caught by `backend_check` failing to even compile the kernel on
   the very first GPU run.

## Line-count note

Both `phobos-gguf/src/qwen35.rs` and `phobos-lang/src/codegen/expr.rs` were
sitting *exactly* at their `phobos-base/tests/source_size.rs` grandfathered
cap (1049 and 916 lines respectively) before this beam touched either, so
any addition at all tripped the ratchet ("may only shrink"). Fixed properly
rather than by trimming comments to fit:
- `qwen35.rs`: the `forward`/`forward_greedy`/`forward_with`/
  `forward_to_logits` family moved into a new descendant module,
  `phobos-gguf/src/qwen35/forward.rs` (the same-name-file-plus-directory
  submodule form, no rename of `qwen35.rs` needed). A descendant module
  keeps full access to `Model`'s private fields and to `attention`/
  `delta_net`, so nothing's visibility changed to make the split possible.
  `ForwardBufs` (the four device buffers a `forward_to_logits` call leaves
  live) also moved out of per-architecture duplication into
  `phobos-gguf/src/model.rs`, shared by both `llama` and `qwen35`.
  `qwen35.rs` down to 933 lines, cap entry tightened 1049 -> 933 in the same
  commit (the ratchet's own "holds nothing stale" test enforces this).
- `expr.rs`: the `argsel` call-site glue (~30 lines, shape/element
  validation, allocation, the four operands' release) moved to
  `phobos-lang/src/codegen/tile/elem.rs` as `emit_argsel`, right beside
  `tile_argsel_bc` -- the same rationale as `tmax`'s own arm staying
  inline (it is short) not applying once `argsel`'s validation is written
  out. `expr.rs` down to 915 lines, cap entry tightened 916 -> 915.

## Correctness verification

**New example**, `phobos-gguf/examples/argmax_check.rs` (registered in
`Cargo.toml`, `required-features = ["cuda"]`): two independent
`DeviceBackend`/`State` pairs fed the identical token sequence, one running
`Decoder::forward` (full logits, host-reduced via
`phobos_inference::sampling::argmax`) and the other `Decoder::forward_greedy`
(the device kernel), so the comparison is device-argmax against a host
reduction of *the same device backend's own logits* -- deliberately not
against `HostBackend`'s logits, which would fold in cross-backend rounding
differences unrelated to whether the argmax reduction itself is correct.

```
cargo run --release -p phobos-gguf --features cuda --example argmax_check -- models/minicpm5-1b-Q8_0.gguf
  32 of 32 decode steps agree device-argmax with host argmax over the same device logits (0 float ties)
  synthetic edge cases: all negative / winner at index 0 / winner at the last index -- all ok

cargo run --release -p phobos-gguf --features cuda --example argmax_check -- models/Qwen3.5-0.8B-Q8_0.gguf
  32 of 32 decode steps agree device-argmax with host argmax over the same device logits (0 float ties)
  synthetic edge cases: all negative / winner at index 0 / winner at the last index -- all ok
```

32 real decode steps on each model, zero disagreements and zero float ties
(the tie path is checked and handled -- see the example's own logic -- but
never exercised by real logits, as expected). Three synthetic edge cases
against `Backend::argmax` directly (bypassing the model): an all-negative
130,560-wide row with the winner in the middle (the case a masking bug would
get wrong), and the winner at index 0 and at the last index of a narrower
row (off-by-one boundaries in the grid-stride loop or the halving tree).
All three agree.

**The four standing gates**, both models, `PHOBOS_ATTN_PERSIST=1` (this
session's default-improvement config): `backend_check` (worst relative
error `2.902e-4`, identical to the pre-existing documented baseline --
`argmax` is not one of `backend_check`'s own op-by-op cases, so this
confirms nothing about it directly, only that nothing else regressed),
`batch_check`, `model_check`, `fuse_check` all pass on both
`minicpm5-1b-Q8_0` and `Qwen3.5-0.8B-Q8_0`, matching documented error bands
(fuse_check's minicpm worst-step spread `1.332e-2`/avg `9.944e-3`, Qwen's
`1.051e-2`/`8.085e-3` -- both in the same range as prior recorded runs).
None of these four exercise `forward_greedy` (they all call `forward`), so
they are confirmation that the *shared* `forward_to_logits` refactor did not
disturb the existing path, not evidence about the new one -- that is what
`argmax_check` is for.

**Build and lint**: `cargo build --release -p phobos-gguf --features cuda
--examples` clean. `cargo clippy --release --workspace -- -D warnings` and
`cargo clippy --release -p phobos-gguf --features cuda --examples -- -D
warnings` both clean. `cargo test -p phobos-lang` (151 passed, including a
new codegen test pinning `argsel`'s `arith.cmpf oge`/`arith.select`
lowering), `cargo test -p phobos-gguf -p phobos-onnx -p phobos-inference
-p phobos-kernels -p phobos-base` all pass (source_size ratchet included).
Every `.ph` example under `examples/` and `phobos-lang/examples/` still
verifies under both the default target and `PHOBOS_CHIP=sm_80
PHOBOS_INDEX_BITS=64`, per CLAUDE.md's codegen-change diffing requirement --
not a full before/after diff (this session's edit is provably additive: a
new builtin unreachable from any existing kernel, and the `coerce` fix only
activates on a path that previously always `bail!`ed), but confirmation
nothing broke.

## Process note: this beam is the "concurrent argmax feature" named elsewhere

[[cache-length-split-buckets]]'s own process note (its "Barrier-imbalance
beam" round) describes "a concurrent, unrelated agent" landing "an in-progress
argmax feature (new files and struct fields across `phobos-lang`,
`phobos-gguf/src/backend/mod.rs`, `device/mod.rs`,
`device/kernels/mod.rs`)" in the same shared tree, at points leaving it
unable to build for reasons unrelated to their own attention work -- that
was this beam. Confirmed no actual file-content collision: this beam never
touched `attn.rs`, `attn_persist_plan`, or `AttnPersistKey`/
`AttnPersistEntry` (that beam's own territory), and the shared files
(`device/mod.rs`, `kernels/mod.rs`) only ever needed *additive* struct-field
and `mod` insertions from each side -- confirmed by reading `git diff` on
`device/mod.rs` before adding this beam's own fields, finding the other
beam's `AttnPersistKey` narrowing already present, and inserting around it
rather than through it. Both beams' diffs coexist cleanly in the working
tree as of this beam's own final verification pass (full workspace build,
clippy, and both beams' correctness gates all pass together). The "tree
temporarily failed to build" episodes on their side were this beam's own
in-progress, not-yet-fixed state (the scientific-notation float-literal
snag below, and the source-size ratchet churn) passing through, not a
genuine conflict.

## What is genuinely uncertain, left for the orchestrator's own bench.py run

This beam's own local timing (wall-clock loop, not `bench.py`) was not run
against the "do not run bench.py yourself" constraint, so there is no
same-session throughput number in this file. The honest ceiling math above
(1.7-1.9% gross, ~1.3-1.6% net of the new kernels' own cost) is a profile-
derived estimate, not a measurement of this change specifically. Two things
worth the orchestrator checking when the real `bench.py` run happens:
- Whether the net gain is visible above `bench.py`'s noise floor at
  `tg1024`, given `tg4096`'s longer per-step time dilutes a fixed
  microsecond saving into a smaller percentage, matching the pattern
  [[cache-length-split-buckets]]'s null result already documented for a
  *different* fixed-cost lever.
- Whether `PHOBOS_ATTN_PERSIST=1` needs to stay forced for this
  comparison, unrelated to this beam but standing practice this session.
