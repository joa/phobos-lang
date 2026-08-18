# q8_qmma load-width interleave (killed at census, no code written)

Beam: act on the SASS/PTX audit's item 1 finding
(`autoresearch/beams/prefill-sass-audit.md`) that `q8_qmma`'s K-loop reads
every operand as a 4-byte `LDG.E.SYS`/`ld.global.b32`, with `ptxas` finding
zero coalescing opportunity, because a lane's two `Q8_BLOCK` halves sit 12
bytes apart in memory and rows are a full pitch apart -- a data-layout
problem, not a scheduling one, distinct from the mechanisms behind the two
already-failed `q8_qmma` beams
([[q8_qmma_occupancy_dead_ends]]: split-K died on DRAM-bandwidth contention,
K-loop register prefetch died fighting `ptxas`'s own free scheduling
headroom).

Proposed fix: permute bytes within each 32-byte Q8_0 block identically on
both the A (activation) and W (weight) operands, so the two K-halves a lane
needs become 4 bytes apart instead of 12, letting one `vec8_i8` load replace
two `vec4_i8` loads. The invariance argument: every consumer computes a
block-local dot product, and a dot is invariant under any within-block
permutation applied identically to both operands, provided block membership
(and therefore the per-block scale lookup) is preserved.

## Result: killed at step 0 (census), before any code

**No code was written and no GPU time was used.** The task brief required a
producer/consumer census as a hard gate before implementation; the census
killed it.

**Prong (a), the invariance argument itself: holds.** Every existing
consumer reads A and W at identical physical column offsets and sums only
within one block:

- `qmma_t` (`phobos-lang/src/codegen/tile/qmma.rs:227-240`) -- the target.
- `qdot_t` (`phobos-lang/src/codegen/tile/qdot.rs:126-140`) -- decode's
  projection kernel.
- the generic dp4a `dot_t` path (`phobos-lang/src/codegen/tile/dp4a.rs:30-59`),
  serving `q8_dp4a`/`q8_mma`/`q8_split`.
- the host oracle's `matmul_quant_act` (`phobos-gguf/src/backend/host.rs:265-277`),
  a positional zip within each block, also invariant, and doesn't need to
  change since it produces its own A independently.

**Prong (b), the producer surface: kills it.** The decisive fact isn't how
many producer sites exist, it's that they're *shared* across kernel
families. A single `QAct` buffer, produced once per `project_q8` call
(`phobos-gguf/src/backend/device/matmul.rs`), feeds `qmma_t`, `qdot_t`, and
`dot_t` consumers depending on row count -- so a permutation can't be scoped
to `qmma_t`'s own operands, it has to apply identically to all five A
producers, or none:

- `phobos-gguf/src/backend/device/kernels/quant.rs:31-43` (`QUANTIZE_SRC`,
  prefill's standalone quantizer)
- `phobos-gguf/src/backend/device/kernels/norm.rs:44-55` (`rms_norm_src`,
  Quantized/GatedQuantized forms)
- `phobos-gguf/src/backend/device/kernels/norm.rs:100-107` (`swiglu_q_src`)
- `phobos-gguf/src/backend/fuse/emit.rs:238-240` (`Stage::NormQ`'s `whole()`)
- `phobos-gguf/src/backend/fuse/emit.rs:344-350` (`Stage::QuantQ`)

W has the same shape of problem on a smaller scale: one upload site
(`phobos-gguf/src/quant/q8_0.rs:36-48`) read by every kernel variant, so it
also can't be scoped to just the `qmma_t` reader.

The two decisive producers, `rms_norm_q` and `swiglu_q`, run on **every
decode step**, not just prefill -- confirmed via `phobos-gguf/src/layers.rs`,
shared by both `llama.rs` (prefill and decode) and `qwen35.rs`. Permuting
their output either narrows a single vectorized 32-wide store into 8
separate 4-byte stores, or adds pre-store permutation statements that the
audit's own item 3 already found get barrier-bracketed per tile-codegen
statement -- real risk to a decode path this session spent multiple beams
hard-winning to parity with llama.cpp, for an unproven prefill-only payoff.
Two closing greps (`tensor<i8>` across all `phobos-gguf` kernel sources, and
`qdot|qmma|Q8_BLOCK` in `phobos-onnx/src`) found no sixth producer and
confirmed ONNX never touches this path, so the count above is exhaustive,
not partial.

## Fallback assessed, also not viable

The brief's named fallback -- confine the fix entirely to `qmma_t`'s reader
side, wide 8-byte load plus an in-register `shfl`/`prmt` shuffle, touching
no producer -- doesn't survive scrutiny either. The two 4-byte chunks a lane
needs are 12 bytes apart, not 8, so a single wide load can't directly cover
them; the only way to make this work is an intra-quad exchange (4 lanes each
load a different 8-byte chunk of the 32-byte block, then `shfl` to
reassemble). That trades 2 independent `LDG.32` per row for 1 `LDG.64` plus
~4 `SHFL`s and lane selects -- a genuine new cross-lane dependency in a loop
`ncu` already showed is latency-bound, the same failure shape as the killed
K-loop prefetch beam (register-neutral on paper, regressed 8.8-42% because
it added a dependency chain `ptxas` didn't have before). It also needs a
`prmt`/`shfl.idx` byte-select instruction `phobos-lang`'s ISA vocabulary
doesn't have yet (`phobos-lang/src/codegen/target/nvidia.rs` only has
`shfl_xor_f32`, used for f32 reductions) -- new compiler work, not a quick
probe.

## Recommendation

Negative result, banked with mechanism, no code changes, worktree left
clean. The load-width finding itself is real and re-confirmed, but neither
way of acting on it survives: the producer-side interleave risks a decode
regression across a shared surface for an unproven prefill-only win, and the
reader-only shuffle fallback re-creates the exact dependency-chain trade
that already killed this kernel's sibling beam.

The audit's own occupancy datum -- 12.45% achieved against a 25%
register-bound ceiling, grid `(1,20,1)` on the smallest shape -- still
points at grid starvation, not load width, as `q8_qmma`'s real remaining
lever. This is consistent with why split-K's per-kernel wins were real but
didn't survive contact with the whole pass (bandwidth contention, not a
grid-shape problem) -- a future beam on this kernel should approach from a
bandwidth-reduction angle at the current grid shape, per that beam's own
closing note, not from either a layout or a splitting angle, both of which
have now been tried.
