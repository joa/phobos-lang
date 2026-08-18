# attention_block: barrier elision for adjacent pure-producer statements

Task: close some of item 3's `prefill-sass-audit.md` gap -- 14 `gpu.barrier`
per dynamic causal-loop iteration against an audit estimate of ~9 truly
necessary sync points -- by making tile-statement barrier emission
conditional in `phobos-lang`'s codegen, scoped to whatever a cheap,
provably-safe mechanism actually supports. Re-ran with the vectorize+pad fix
(`attention-block-vectorize-pad.md`, `3f093db`) already landed, since barrier
stalls grew from 39.6% to 44.0% of the average issue-to-issue gap once that
fix shrank everything around them (`ncu_attnblock_fixAB_pp128_details.txt`,
already in the tree).

**Headline result: the mechanism the task was framed around does not apply
here (kill condition triggers for the general case), but a narrower,
provably-safe special case exists and is implemented.** Every one of the 14
loop-body barriers, traced by hand against the real emitted MLIR, sits
between two statements using genuinely different thread-to-data mappings
(matmul vs. warp-shuffle reduction vs. vectorized elementwise), so "skip the
barrier when the next statement's mapping matches" recovers *nothing* -- the
audit's own "9 real dependency points" undercounted, and the true gap is
duplicated materialization (an `rowsum(s)` temp, a `dot(s,v)` zero-init, a
trivial `m = mn` copy), each needing its own fusion design, not a barrier
toggle. What *is* real and safely removable: two adjacent statements that
each read only global tensor memory (no shared tile) can share one trailing
barrier instead of two, since neither's write can race the other and nothing
reads either until after the later one's own barrier fires anyway. In
`attention_block`'s loop body this is exactly the `var k = K[...]; var v =
V[...]` pair Fix A already vectorized: **14 -> 13 barriers per dynamic
iteration, 48 -> 47 static total**, confirmed in emitted MLIR, PTX
(`bar.sync`), and SASS (`ptxas -arch=sm_75` + `nvdisasm`, `BAR.SYNC`).

## Step 0: mechanism check

**Is barrier emission unconditional?** Mostly, but not absolutely. There is
one pre-existing carve-out: `distribute()` (`phobos-lang/src/codegen/tile/check.rs:188`)
takes a `sync: bool` parameter, used today only by the pipelined-prefetch
path (`pipeline.rs`) to skip a barrier the loop's own closing barrier already
covers. Every other tile-producing op -- `reduce.rs`'s row/lane reduction
(`rowmax`/`rowsum`/`tmax`), `contract.rs::tile_matmul_t` (`dot_t`, via
`distribute(..., true, ...)`), the plain-matmul/warp-fragment path `dot(s,v)`
takes, `qdot.rs`, `imma.rs`, `qmma.rs`, `warp_attn.rs` -- ends its own
lowering with an unconditional `self.barrier(block)`, decided purely by
"which op is this," with zero visibility into what the next statement does.
`stmt.rs`'s `Stmt::Var` branch for `var name = <tensor slice>` (the exact
form `attn.rs`'s Fix A staging uses) hard-coded `tile_copy(..., true, false)`
-- `sync` always `true` -- before this task.

**What would making it conditional need?** Traced every one of the 14
barriers in a hand-built replica of the real `pp128` shape
(`NH=16 G=8 D=128 BR=8`, compiled via `cargo run -p phobos-lang --example
emit`) against the loop-body statement it follows, reading the actual
thread-to-index formula each op emits (not just the audit's source-level
accounting):

- `k`/`v` staging: 64-thread linear `tid -> (row, 0)`, `vector<8xf16>`
  (Fix A's vectorized copy).
- `dot_t(q,k)` (writes `s`): 64 threads, one thread per `(i,j)` output
  element, width 1.
- `s = s * scale`: 16 threads, `vector<4xf32>`, width 4 -- a *different*
  thread count and per-thread element count than `dot_t`'s own write.
- `rowmax(s)`: 256 threads, 32-lane warps, one warp per row, tree-reduced via
  `gpu.shuffle xor`, lane 0 publishes -- nothing like either of the above.
- `tmax(m,mn)`, `exp(s-mn)`, `exp(m-mn)`: each its own `distribute` call,
  its own thread count.
- `rowsum(s)` (feeding `l = l*corr + rowsum(s)`): same warp/lane reduction
  shape as `rowmax`, materializes into its own temp (`tile9`) *before* the
  combine statement that reads it -- a real materialization the audit's
  source-level accounting collapsed into the combine line, not actually
  removable without fusing the reduction into the combine's own store.
- `dot(s, v)` (feeding `acc = acc*corr + dot(s,v)`): a warp/lane
  fragment-accumulation loop (`vector<4x4xf32>` iter_args) into its own
  zeroed temp (`tile10`) -- the zero-init and the accumulate are two more
  materializations with a real read-after-write dependency between them
  (the accumulate reads the zeroed cells), not removable by this mechanism.
- `m = mn`: a plain 8-thread copy, structurally unrelated to every
  neighboring op's mapping.

**Zero of the 14 pairs share a thread-to-data mapping.** "Skip the barrier
when the next statement's access pattern matches the write's" -- the
mechanism the task was framed around -- recovers nothing on this kernel, not
because the idea is wrong but because this codegen's op-lowering functions
each invent their own ad hoc mapping inline (there is no shared,
comparable "access pattern" type anything could check equality on), and in
this specific kernel no two adjacent statements happen to reuse one anyway.

**Kill condition assessment.** Building a general "does the next read need a
fresh barrier" check for the mapping-equality case would need either (a) a
new, reified, comparable access-pattern type threaded through every
tile-producing function in the tree (`reduce.rs`, `contract.rs`, `elem.rs`,
`qdot.rs`, `imma.rs`, `qmma.rs`, `warp_attn.rs`) purely so two could be
compared -- real new infrastructure with no existing precedent -- or (b) a
deferred-barrier scheme (mark a barrier "pending," flush lazily at the next
shared-memory read) that would need every shared-memory *read* site in the
same ~10 files to consult the pending flag before loading, a wide blast
radius for what this kernel alone would net at most one or two more removed
barriers. Both are the kind of "real cross-statement dataflow/alias analysis
that doesn't exist anywhere in this codegen yet" the task's kill condition
names. **Not attempted.**

**The narrower special case that does hold, safely, without any dataflow
analysis:** a `var name = <tensor slice>` statement -- by construction of
`slice_static_shape`'s `Expr::Index`-on-`Binding::Tensor` match -- reads
*only* global tensor memory, never a shared tile. Two or more such statements
in a row therefore cannot race each other no matter what their individual
thread mappings are (there is nothing shared to race over), and nothing
downstream reads any of them until after the last one's own trailing
barrier fires anyway (that barrier still exists; only the earlier ones in
the run are redundant). This needs no per-op access-pattern type and no
liveness tracking -- it falls out of the syntactic shape alone. First found
by hand (an advisor review caught and closed a hole in an earlier, broader
version of this rule -- see "A rule this task considered and rejected"
below), then implemented as a bounded lookahead in `emit_stmts`, the same
style `matmul_candidate`/`frag_acc_candidate` already use over `stmts[i..]`.

### A rule this task considered and rejected

An earlier draft of this rule read: "skip a write's barrier if the
immediately following statement doesn't read what it wrote, and the
following statement ends in its own barrier." That is unsound: consider
`l = l*corr + tmp` (reads `tmp`, releasing it -- `tmp`'s last read) followed
by `var z: tile[...] = 0.0` (a fresh fill, not reading `l` or `tmp`). The
rule would elide the first statement's barrier, but the pool can hand `z`
`tmp`'s just-released buffer, so `z`'s write now races the first statement's
still-in-flight *read* of `tmp` -- a silent WAR hazard, in exactly the
pool-reuse pattern `Codegen::release`'s own doc calls out as needing a
barrier to stay race-free. The shipped rule avoids this by restricting to
statements that read *no* shared tile at all (so nothing they write can ever
be a released buffer another live read depends on) rather than statements
that merely don't read *this particular* write.

## Implementation

`phobos-lang/src/codegen/stmt.rs`:

- `bind_staged_tile(name, src, sync)`: the `Rv::Tile(src)` half of `var name
  = <tile expr>`, extracted unchanged from the `Stmt::Var` branch except for
  the new `sync` parameter threaded into the existing `tile_copy(...)` call
  (which already had a `sync: bool` parameter, previously always `true` at
  this call site).
- `stage_run(stmts)`: length of the maximal prefix of `stmts` matching `var
  name = <tensor slice>` (`ty: None`, `slice_static_shape(value).is_some()`,
  `!slice_is_partial(value)` -- the same two predicates `pipeline_candidate`
  already uses to detect a staged loop prefix, reused rather than
  reimplemented). `slice_is_partial` is what excludes a masked slice (e.g.
  `attention_block`'s diagonal `dk`/`dv`, whose offset is `program_id`-
  relative and not proven in bounds): a masked slice materializes eagerly
  inside `emit_expr` itself (returning an already-`owned` tile,
  `check.rs::materialize_masked`) rather than through this lazy-view path,
  so `stage_run` never sees it and this run's safety argument (`reads no
  shared tile`) never has to reason about it.
- `emit_staging_run(stmts)`: emits each matched statement via
  `bind_staged_tile`, passing `sync = false` for every one but the last.
- `emit_stmts`'s dispatch loop: after the existing `matmul_candidate`/
  `frag_acc_candidate` checks (unaffected -- both match different `Stmt`
  shapes), a new `stage_run(&stmts[i..])` check; a run of 2 or more is
  emitted via `emit_staging_run` and consumes that many statements at once,
  same as the existing multi-statement dispatches.

Three new tests in `phobos-lang/src/codegen/tests/tile.rs`:
`consecutive_staged_slices_share_one_barrier` (two staged slices plus a
store: 2 barriers, not 3, and the shared barrier lands after both copies),
`a_lone_staged_slice_keeps_its_own_barrier` (a single staged slice is
unaffected: still 2), `staging_run_stops_before_a_non_slice_statement` (a
third statement that isn't a bare tensor slice -- `var c = a`, reading an
already-staged tile -- starts a new run rather than folding in: 3 barriers,
confirming the run cannot accidentally swallow a statement with a real
read).

## Verification

**Emit-diff sweep**, all 9 files in `examples/*.ph` and both in
`phobos-lang/examples/*.ph`, at the default target and at
`PHOBOS_CHIP=sm_80 PHOBOS_INDEX_BITS=64` (22 combinations), diffed against a
`git stash`-isolated baseline (stderr filtered from stdout throughout, so no
`Compiling`/`Finished` noise reached the diff, per this session's own
recorded lesson from a prior beam's false positive). **20 of 22 identical.**
The two that differ, `gemm_fp16` at both targets, differ by exactly one
removed `gpu.barrier` line each, nothing else -- `gemm_fp16.ph`'s K-loop body
opens with `var a = A[...]; var b = B[...]` before `acc += dot(a, b)`, the
same pattern as `attention_block`'s `k`/`v`, and the rule fires there too, as
expected for a general codegen change (not a kernel-specific hack). Verified
by hand: `a`'s write and `b`'s write are independent (neither reads the
other), and `dot(a,b)`, the first read of either, occurs after `b`'s own
barrier -- the elision is correct there for the identical reason it's
correct in `attention_block`. `moba_efficient.ph`, `kda_fp32.ph`, and the two
`flash_attention_*.ph` files -- the other kernels with adjacent staging-
looking statements -- did not change, either because their staged slices are
masked (`slice_is_partial`) or because automatic pipelining
(`pipeline_candidate`, which runs first and takes priority) already claims
that loop's staged prefix before this rule ever sees it.

`cargo test -p phobos-lang`: **161 passed, 0 failed** (158 pre-existing +
3 new). No existing test's expected MLIR needed updating -- none of the
pre-existing fixtures happen to chain two bare tensor-slice `var` statements
back to back (the closest, `matmul_kernel_lowers_to_subviews_and_distributed_loops`
in `tests/matmul.rs`, uses `let`, not `var`, a different, pre-existing
lazy-view path this change never touches).

`cargo clippy --release --features cuda -p phobos-lang -p phobos-gguf -- -D
warnings`: clean.

### Per-kernel barrier count, `attention_block` at the real `pp128` shape

Static `gpu.barrier` (MLIR): 48 -> 47. Loop-body dynamic count (14 barriers/
iteration originally): 47 static total resolves to **13 barriers per dynamic
causal-loop iteration**, confirmed by re-slicing the same probe's loop body.
PTX `bar.sync`: 47 (matches). SASS (`ptxas -arch=sm_75 -O3` +
`nvdisasm`, `BAR.SYNC`): 47 (matches; `ptxas` did not independently fold or
add any). This is short of the audit's ~9-per-iteration estimate -- as
explained above, that estimate undercounted the real materialization count
(it treated `rowsum(s)`'s own temp and `dot(s,v)`'s zero-init as free, and
didn't anticipate `m = mn` getting its own full distribute+barrier), and
none of those three gaps are closable by this mechanism; they need genuine
statement-fusion work against a specific op (folding a reduction's result
directly into its consumer's store, or letting a fragment-accumulation loop
skip a separate zero-init), each its own design task, not a general barrier
condition.

## Correctness gates, both models, `--release --features cuda`

**`backend_check`** (synthetic op suite, no model): worst relative error
**2.902e-4** -- identical to the vectorize-pad beam's documented baseline.

**`batch_check`**:
- minicpm5-1b-Q8_0: gpu spread err up to 1.350e-2, batched and sequential
  agree, exit 0.
- Qwen3.5-0.8B-Q8_0: gpu spread err up to 1.540e-2, batched and sequential
  agree, exit 0.

**`model_check`**:
- minicpm5-1b-Q8_0: 3 steps, spread err up to 1.763e-2, backends agree,
  exit 0.
- Qwen3.5-0.8B-Q8_0: 3 steps, spread err up to 1.074e-2, backends agree,
  exit 0.

**`fuse_check`**:
- minicpm5-1b-Q8_0: prompt pass agrees exactly; 32 decode steps, at most
  1.332e-2 of the logit spread apart, 9.944e-3 average, 0 top-token flips.
- Qwen3.5-0.8B-Q8_0: prompt pass agrees exactly; 32 decode steps, at most
  1.051e-2 of the logit spread apart, 8.085e-3 average, 0 top-token flips.

Every one of these numbers matches the vectorize-pad beam's documented
baseline **exactly**, to the last reported digit -- expected, since this
change removes a redundant synchronization point rather than altering any
computed value, and the match is itself a data point that nothing was
silently reordered into a race.

## `ncu` timing

GPU-slot granted after code review; card verified idle immediately before
capture (`nvidia-smi` polled four times: 3-25% transient util, 375-840MHz,
the same desktop-compositor-blip pattern the vectorize-pad beam's own
baseline capture called idle, no other `bench`/`check`/`ncu` process
running). `ncu --set full -k "regex:attention_block" -c 5` against this
worktree's rebuilt `target\release\examples\bench.exe -m
models\minicpm5-1b-Q8_0.gguf -p 128 -n 0 -r 1 --no-warmup`,
`PHOBOS_ATTN_PERSIST=1`, `Set-Location`'d into the worktree explicitly
first (the vectorize-pad beam's own mixup-avoidance step). Confirmed
profiling `C:\Users\joaeb\code\phobos-wt-barrierfuse\target\...` from the
`==PROF== Connected to process` line. 5 launches, same `pp128` shape/grid
`(16,16,1)`/block `(256,1,1)` as every prior capture in this session.
Evidence: `ncu_attnblock_barrierfix_pp128.ncu-rep` / `_details.txt` / `.csv`.

| | before (`fixAB`, this session's standing baseline) | after (this beam) |
| --- | --- | --- |
| duration, 5 launches | 91.488, 91.968, 92.992, 90.016, 96.128 us | 93.024, 94.400, 93.568, 87.520, 92.704 us |
| duration avg | 92.518 us | 92.243 us |
| barrier stall % of issue gap | 44.0% (single launch exported in the baseline `_details.txt`) | 43.6%, 43.5%, 43.7%, 43.8%, 43.3% (avg 43.58%) |
| warp cycles / issued instr | 15.38 | 15.42, 15.44, 15.36, 15.42, 15.44 (avg 15.42) |
| bank conflicts (`..._op_ld.sum`) | 508,874 | 509,080 / 508,841 / 508,898 / 509,002 / 508,929 (unchanged, as expected) |
| `Block Limit Shared Mem` | 3 | 3 (unchanged, as expected) |
| Achieved Occupancy | 65.31% | 65.39% / 64.00% / 63.71% / 65.38% / 65.24% (avg 64.74%) |

**Duration: statistically flat, -0.275us / -0.30% on the 5-launch average,
inside both distributions' own peak-to-peak spread** (before: 90.02-96.13,
a 6.1us/6.6% range; after: 87.52-94.40, a 6.9us/7.5% range -- the two
ranges overlap almost entirely). This confirms the expectation set before
the capture: removing 1 of 13-14 barriers is not a measurable
launch-duration win at this kernel's size, and no honest reading of these
five numbers claims one.

**Barrier-stall share: a small, real, directionally consistent drop** --
44.0% -> 43.58% average, about 0.4 percentage points, roughly 1% relative.
Every one of the 5 new-capture launches lands between 43.3% and 43.8%,
consistently *below* the single baseline launch's 44.0%, which is a
tighter, more one-sided pattern than duration's overlapping noise band
shows -- so this is more likely a real (if small) effect than noise, but it
is nowhere near proportional to "1 barrier removed / 13-14 total = ~7%,"
confirming the other expectation set before the capture: the removed
barrier's own stall cost is whatever the slowest warp was already going to
make everyone wait at the *next* barrier for, not a uniform 1/14th share of
the total. Bank conflicts and `Block Limit Shared Mem` are unchanged, as
they should be for a change that touches only synchronization, not layout
or footprint -- a clean sanity check that nothing else moved.

## Recommendation

**Land the mechanism; do not expect it to move `pp128` throughput on its
own.** It is a genuine, narrow, general codegen correctness-preserving
change (not gated to `attention_block`, verified tree-wide at both targets,
all four gates pass on both models, matching the pre-existing baseline to
the last digit) with a small, real, measured effect: barrier-stall share
drops about 0.4 percentage points (44.0% -> 43.58% average, a tight,
one-sided 5-launch spread against a single baseline point, more likely real
than noise) and launch duration is statistically flat (-0.30% average,
entirely inside both captures' own peak-to-peak spread -- not a claimed
win). The value of this beam is the negative result and the map it leaves
behind, not the barrier count: it closes off "make the barrier
call conditional on the next statement's mapping" as a lever (proven to
recover nothing on this kernel, and the general form of that check is the
kind of dataflow/alias analysis this codegen doesn't have), and it pins the
audit's real 44%-of-issue-gap finding to three specific, named,
still-open materializations -- `rowsum(s)`'s own temp before the `l`
combine, `dot(s,v)`'s zero-init before the accumulate, and the bare `m =
mn` copy -- each its own op-aware fusion design, not a context-free rule,
and each a plausible next beam now that this one has done the tracing.

## Files touched

- `phobos-lang/src/codegen/stmt.rs`: `bind_staged_tile`, `stage_run`,
  `emit_staging_run`; `emit_stmts`'s dispatch loop calls the new run
  detector before falling back to per-statement `emit_stmt`.
- `phobos-lang/src/codegen/tests/tile.rs`: three new tests (see above).

No `phobos-gguf` file changed -- `attention_block`'s barrier drop is a
consequence of the general codegen rule applying to its existing `var
k`/`var v` staging (already `var`, already unmasked, from the vectorize-pad
beam), not a kernel-source edit.
