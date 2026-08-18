# Tile-pool `Binding::View` release sweep (no code ships)

Beam: follow up on a side finding from `attention-block-vectorize-pad.md`
-- a masked `let`-bound tensor slice becomes a `Binding::View` in tile
codegen, and `release()` is documented as a no-op for views, so its
shared-memory allocation never returns to the pool. Switching that one
site (`attention_block`'s K/V/dk/dv) from `let` to `var` dropped its
static shared-memory footprint ~24.7KB -> ~16.7KB and moved
`Block Limit Shared Mem` 2 -> 3 blocks/SM. This beam surveyed the rest of
the tree for the same pattern.

## Result: no code ships, mechanism corrected, root fix identified but deferred

**The task brief's framing of the mechanism was imprecise; the real gate
is two-part**, both in `phobos-lang/src/codegen`:

- `util.rs:448-454` (`bind()`) clears `owned = false` on **every** named
  binding, `Binding::View` and `Binding::Tile` alike. The ad-hoc
  `self.release(&a)` calls scattered through `dot`/`dot_t`/`qmma_t`/
  `tmax`/`rowmax` etc. never free a *named* value -- they only matter for
  anonymous temporaries.
- The only release path for named values is `release_named`
  (`stmt.rs:779-791`), driven by `emit_stmts`'s last-use scan
  (`hoist.rs:300`, `last_uses`). It pattern-matches `Binding::Tile` only;
  `Binding::View` (what `let` produces) is silently skipped, forever.

So `var` doesn't work because it releases "sooner" in some ad-hoc sense --
it works because it's the only binding kind `release_named` ever touches
at all. This also retires a worry the beam raised early: a named value
used more than once is never at risk of premature release either way,
since `release_named` only fires after the statement scan's last use.

**Transferable rule this beam established**: the leak only costs bytes
when a *later* allocation of the same `(elem, shape)` exists in the same
kernel to reuse the freed slot. Most kernels checked don't have one.

## Candidates checked

| Site | Verdict |
| --- | --- |
| `quant.rs`: `q8_dp4a`/`q8_mma`/`q8_split`, narrow `@aligned(K=32)`-only variant | Defect real (dp4a -44%, mma -28% shared bytes), fix **blocked**: these kernels' source generators share one `{ALIGNED}` template with a fully-aligned wide variant where the same slice is already an unmasked zero-copy view; applying `var` uniformly regresses the wide variant (+3 to +6 barriers). Needs a per-variant `let`/`var` template placeholder, not a blind swap -- not attempted, out of this beam's scope. |
| Occupancy math for the above | No movement predicted regardless: all footprints (1.3-9.3KB) sit far under this launch width's real binding constraint, which is warp count (`@launch(256)` = 8 warps/block, 4 blocks/SM ceiling from warps alone, per `attn.rs`'s own comment and the attnfix beam's independently-measured [50.1, 74.1)KB per-SM shared budget). Block count here is warp-limited, not shared-memory-limited, all the way up through the region these kernels actually occupy. |
| `attention_split_src`, `attention_persist_src` | **Not applicable** -- already all `var`. |
| `argmax_reduce`/`argmax_finish` | Real mask, **zero benefit**: converting to `var` left tile count/bytes unchanged and added a barrier (17->18, no downstream reuse of the freed shape exists). |
| `phobos-onnx/src/lower.rs`: `lower_flash_attention` | Real mask, **zero benefit, regresses**: identical tile footprint before/after at two distinct shapes, +2 barriers (32->34), no downstream reuse. |
| `fuse/emit.rs`'s `whole()` NormQ stage | **Applied, measured on real generated chains, reverted** -- see below. |
| `phobos-kernels`, other `fuse/emit.rs` sites, `phobos-bench` | No candidates; all tensor-slice bindings already `var`. |

### The one real attempt, and why it was reverted

An isolated 2-loop hand probe of the `whole()` NormQ stage's masked
`xb{s}` slice looked like a clean win (10 tiles/8456B -> 9 tiles/6408B,
-24%, zero barrier change), and reachability was confirmed for both bench
models (`embedding_length` 1536 and 1024, both exact multiples of the
512-row `whole()` threshold, via `cargo run -p phobos-gguf --example
inspect`). But pulling the **real** generated Qwen fused-MLP kernel source
via the existing `fused_source` test harness and diffing all three real
chain plans against the change showed **zero byte reduction and a
consistent +1 barrier** across all three -- the isolated probe only
exercised 2 of the kernel's many downstream stages, and the full kernel
apparently already resolves the pool differently once the rest of the
projection/SwiGLU/gate stages are present. Reverted; a partial MLIR diff
suggested the before-version may already fuse a read of `xb0` twice
straight off the raw masked global subview rather than materializing a
copy at all, but that mechanism was not traced to ground truth before
timeboxing this beam.

**This is the beam's central methodological finding**: an isolated
hand-built probe of a staging pattern can materially disagree with the
same pattern's behavior inside the real, full kernel it's part of.
Validate any future change like this against real generated chains
(`fused_source` test + `emit`), not standalone probes.

## Root-fix recommendation (not implemented)

Extend `release_named` (`stmt.rs:779`) to also handle `Binding::View`
where `mv.global.is_some()` -- this fixes the whole class tree-wide with
no per-kernel edits and no wide/narrow template conflict (a genuinely
unmasked `let` already has `owned=false`, so release stays a no-op there
regardless). Blast radius to check first: `pipeline.rs:426` also creates
`Binding::View`s over prefetch double-buffering staging -- releasing
those early could corrupt the pipeline, needs its own audit before this
lands. Any such change must be validated against real generated chains,
per the lesson above, not hand-built probes.

## Recommendation

No code change ships from this beam. The defect is real and now
precisely characterized, but every site checked is either not worth
fixing on its own (no occupancy-threshold crossing, or a template
conflict with a sibling variant) or, on the one site that looked clean in
isolation, contradicted by measurement on the real generated kernel. The
root fix (extend `release_named` to cover `Binding::View`) is a clean,
tree-wide candidate for a future beam, gated on auditing `pipeline.rs`'s
prefetch-buffer usage first.
