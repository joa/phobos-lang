# Prefill SASS/PTX/MLIR audit: barriers, bank conflicts, load widths

Task: answer the three questions this whole investigation was framed around that
had not yet been checked with real measurement -- (b) does the compiler elide
barriers it doesn't need, and the swizzling/vector-width half of (c) -- on the
two kernels that dominate minicpm5-1b-Q8_0's `pp128` prefill pass: `q8_qmma`
(13.29ms/pass, the projection GEMM) and `attention_block` (4.47ms/pass, the
prefill attention kernel, `phobos-gguf/src/backend/device/kernels/attn.rs`).
This is a measurement-and-design report; **no repository file changed** other
than this one. Every probe `.ph`/`.mlir`/`.ptx`/`.sass` file used to produce
the findings below lived in the session scratchpad, outside the repository,
and none were left in the tree.

Method, in the order actually used: compile the exact shipped kernel source at
real minicpm shapes through `cargo run -p phobos-lang --example emit`/`ptx`
(this is what `phobos-gguf`'s real compile config produces -- `Context::default()`,
`sm_75`), diff the emitted MLIR/PTX against a hand-edited variant to test a
specific hypothesis, and where that wasn't enough, feed the PTX through
`ptxas -arch=sm_75` and `nvdisasm` directly to see what the driver would
actually schedule, then confirm on the card with `ncu --set full` on the real
`bench.exe -p 128 -n 0` run. GPU verified idle before every capture
(`nvidia-smi`: 14% util, clocks at idle floor, only desktop-compositor
processes in `--query-compute-apps`). Per the standing GPU-slot protocol, I
was told I have first priority for timing windows this round and used a single
slot for the two `ncu` captures below; I did not run `scripts/bench.py`.

## Two-axis ranking

The four findings rank differently by "cheapest to act on" and "biggest
measured stall share" -- both matter for what the orchestrator dispatches
next, so here they are separately rather than collapsed into one list.

**By actionability (least implementation risk first): 4, 2, 1, 3.**
Item 4 is a two-line `.ph` change, compiler-verified end to end, with a named
residual. Item 2 has an ncu-quantified mechanism and two pre-existing,
named compiler mechanisms that could address it, but both need real
`phobos-lang` work to wire onto this kernel's codegen path. Item 1 needs a
coordinated data-layout change across two crates (quantizer + codegen) but the
codegen-side change itself is well scoped. Item 3 turns out, on inspection, to
have no `.ph`-level or hand-edit lever at all -- every barrier found maps to a
real per-statement tile materialization, so acting on it means new
`phobos-lang` statement-fusion work, the least scoped of the four.

**By measured stall share (biggest lever first): 3, 2, 4/1.**
`attention_block`'s own `ncu` capture shows barrier stalls at 39.6% of the
18.6 cycles between issued instructions -- the single largest stall category
measured this round, ahead of the bank-conflict finding's own 36.56%
ncu-estimated speedup. Item 4 (vector width) and item 1 (load width) both
address a smaller, though real, slice of the same kernels' issue-latency
budget.

Read together: the highest-value next beam is probably item 2 (bank
conflicts) -- large measured headroom *and* a concrete mechanism, just not a
one-line fix -- with item 4 as a nearly-free companion change to ship
alongside it since both touch the same kernel and item 4 is already fully
speced. Item 3's barrier count is real but resolves to compiler-fusion work,
not a targeted kernel change, so it's a `phobos-lang` project of its own if
pursued. Item 1 remains real and well quantified but is the most
cross-cutting to implement (quantizer + codegen, two crates).

---

## Item 4: `attention_block`'s K/V vector widths (most actionable)

**Finding: the shipped kernel does not even reach the narrow 4-element (8-byte)
f16 load `tile_copy` supports -- its K/V staging is fully scalar, 2 bytes at a
time, behind a per-element bounds check, for the entire causal loop.** This is
strictly worse than "leaves a width doubling on the table"; it never engages
`tile_copy`'s vectorized path at all.

### Mechanism, confirmed by compiling the real kernel

`attention_block_src(n_head=16, group=8, head_dim=128, tile=8)` (minicpm's
real `pp128` autotune choice: `NH=16, G=8, D=128, BR=8`) was written to a
standalone `.ph` file, byte-identical to the string
`attn.rs`'s `attention_block_src` emits, and compiled with
`cargo run -p phobos-lang --example emit`. The kernel has **no `@aligned`
attribute at all** -- unlike its sibling decode kernels in the same file
(`attention_split_src`/`attention_persist_src` both declare
`@aligned(KW = D)`).

The emitted MLIR for `let k = K[kt :+ BR, kcol :+ D]` (the main causal-loop
body) shows:

```
%subview_131 = memref.subview %assume_align_1[%arg4, %2] [8, 128] [1, 1]
             : memref<?x?xf16, 1> to memref<8x128xf16, strided<[?, 1], offset: ?>, 1>
...
%80 = arith.addi %2, %78 : index
%81 = arith.cmpi ult, %80, %dim_130 : index          // bounds check on the COLUMN dim (KW)
%82 = arith.select %81, %78, %c0_204 : index
%83 = memref.load %subview_131[%79, %82]             // scalar f16 load, one element
%84 = arith.select %81, %83, %cst_205 : f16           // masked zero-fill
memref.store %84, %assume_align_132[%79, %78]         // scalar store into the shared tile
```

The bounds check is on `KW` (the head-dim axis), not `NK` (cache length) --
the row offset (`kt`, a loop induction variable of `for kt in range(0, base, BR)`)
is already proven in-bounds by the loop's own trimming, but the column offset
`kcol = h / G * D` has no promise that `KW` is a whole multiple of `D`, because
nothing declares one. `tile_copy`'s vectorized path
(`phobos-lang/src/codegen/tile/elem.rs:89`) requires `!src.is_masked()`
before it will emit anything wider than a scalar `memref.load`/`memref.store`
pair -- so this kernel's K/V staging, for its entire main loop, is masked and
scalar for the reason [`vector_width_is_bytes_not_elements`] describes: no
`@aligned` promise, no vector load, at any width, not even the narrow one.

### The fix, and why it needs two changes together, both compiler-verified

Adding **only** `@aligned(KW = D)` to the probe and recompiling removes the
masking, but the K/V *staging step disappears entirely* -- `dot_t`/`dot` fall
back to reading straight from the global subview inside a 128-iteration
scalar `scf.for` + `math.fma` reduction loop (confirmed in the emitted MLIR: no
`memref.global` shared tile for K/V at all in this variant). This matches the
project's own banked lesson, `[[aligned_and_staging_go_together]]`: unmasking
a slice also unstages it, unless the slice stays a `var` (which the compiler
treats as pooled/staged regardless of masking status) rather than a `let`.
`attention_block_src`'s shipped kernel uses `let k =`/`let v =`/`let dk =`/
`let dv =`.

Changing **both** -- `@aligned(KW = D)` *and* `let k`/`let v`/`let dk`/`let dv`
to `var k`/`var v`/`var dk`/`var dv` -- and recompiling the identical probe
produces, for the main loop's K/V copy:

```
%81 = vector.load %subview_125[%79, %80] {alignment = 16 : i64}
    : memref<8x128xf16, strided<[?, 1], offset: ?>, 1>, vector<8xf16>
vector.store %81, %assume_align_126[%79, %80] {alignment = 16 : i64}
    : memref<8x128xf16, 3>, vector<8xf16>
```

`vector<8xf16>` is the 16-byte-wide path -- the compiler proves `align_div`
clears 8 once `KW`'s divisibility is known, and confirms this is a real
compile-time-provable win, not a guess. Both files compile clean
(`cargo run -p phobos-lang --example emit`, no errors) and produced identical
total-barrier counts (48) to the unmodified kernel, so this change does not
interact with item 3's finding below.

**Residual gap, honestly reported**: the diagonal tile (`dk = K[base :+ BR, ...]`,
`dv = V[base :+ BR, ...]`) stays masked and scalar even with both changes,
because `base = NK - R + qt*BR` is a one-shot offset built from `qt`
(`program_id`), not a loop-trimmed induction variable -- `dyn_in_bounds`'s own
doc is explicit that an "undeclared program id above all is unbounded from
inside the kernel." No annotation in the current `@aligned` grammar promises
anything about a program-id-relative one-off offset, so this residual is not
fixable by the same lever. At minicpm's `pp128` shape (`BR=8`, 16 query
blocks), the diagonal tile runs once per block regardless of `qt` while the
main loop runs `qt` times -- summed over all 16 blocks that is 16 diagonal
trips against 120 main-loop trips, so the fix above reaches **120 of 136
(~88%) of this kernel's K/V tile-copy trips**; the remaining ~12% stays on the
scalar path.

### What this is worth, and what isn't verified

Not measured: end-to-end timing. This finding is compiler-verified (MLIR
diffed both ways, twice) but never run on the card -- no `ncu`/`bench`
comparison exists for the modified kernel, and it should not be assumed to
win by the byte-count alone (the K-loop prefetch beam's lesson: a register- or
instruction-count argument that looks free on paper still needs a timing A/B).
**Recommendation for the implementing beam**: make both changes together in
`attn.rs`'s `attention_block_src`, run the `emit` diff-sweep over every `.ph`
under both `sm_75`/32-bit and `PHOBOS_CHIP=sm_80 PHOBOS_INDEX_BITS=64` per
CLAUDE.md (nothing else calls `attention_block` directly, so this is expected
to be a no-op outside the one kernel, but verify rather than assume), run all
four correctness gates on both models, and only then an isolated `ncu`
before/after on the real `pp128` shape before calling this a win.

---

## Item 2: bank conflicts in `attention_block`'s shared tiles

**Finding: real, and ncu quantifies it directly.** `ncu --set full` on the
real `pp128` shape (grid `(16,16,1)`, block `(256,1,1)`, `NH=16 G=8 D=128
BR=8`, launch duration 167.90us this capture, `ncu_attnblock_prefill_full.ncu-rep`):

> The memory access pattern for shared loads might not be optimal and causes
> on average a **5.0-way bank conflict** across all 654,464 shared load
> requests. This results in **2,229,171 bank conflicts**, which represent
> **67.83%** of the overall 3,286,451 wavefronts for shared loads.
> **Est. Speedup: 36.56%.**

A related, overlapping ncu finding on the same capture: "uncoalesced shared
accesses resulting in a total of 1,949,696 excessive wavefronts (53% of the
total 3,707,904 wavefronts)," Est. Speedup 41.67%. Shared **stores** are
clean: conflict sum stays under 0.013% of peak across all 8 launches sampled
in a lighter pass (`ncu_attnblock_prefill_light.csv`) -- this is a load-side
problem only.

### Mechanism, confirmed by arithmetic on the tile's actual layout

The K (and V) shared staging tile is `[BR=8, D=128]` f16, row pitch
`128 * 2 = 256` bytes. The shared-memory bank period is 32 banks x 4 bytes =
128 bytes, and 256 is an exact multiple of 128. For byte address
`a(row=j, col=d) = j*256 + 2d`, the bank index is `(a / 4) mod 32 =
(j*64 + d/2) mod 32`. Since `64 = 2*32`, the `j*64` term is `0 mod 32` for
*every* `j` -- so for any fixed column `d`, all 8 rows of this tile map to the
exact same bank. `dot_t`'s fragment-load loop (the generic `vector.contract`
lowering `attention_block` takes, since it carries no `@tensorcore` tag)
reads `k[j, d]` across several `j` at a fixed or near-fixed `d` per lane group
(confirmed directly in the emitted MLIR from the item-4 probe: a `vector.insert`
sequence loading `%assume_align_52[%103, %62]`, `[%103, %64]`,
`[%116, %62]`, `[%116, %64]`... -- varying row, narrow column range) -- an
8-way *structural* collision by construction, diluted to the 5.0-way average
`ncu` measured by whatever accesses in the same kernel don't hit this exact
pattern (the store side, the scale/`m`/`l` reductions, etc).

This is not a guess about "some swizzle would probably help" -- the row pitch
being an exact multiple of the bank period is the precise, nameable defect,
and it is a direct consequence of `D=128` (a real, unavoidable model
dimension) times 2 bytes/f16 landing exactly on 128B under this codegen's
plain row-major shared layout.

### Two pre-existing mechanisms in this compiler, neither wired to this kernel

`phobos-lang` already has a working XOR column-swizzle mechanism built for
exactly this problem: `alloc_tile_swizzled`/`swizzle_col`/`swizzled_index`
(`phobos-lang/src/codegen/tile/alloc.rs:173-240`). Its own doc comment: "so
ldmatrix reads avoid bank conflicts." At `elem_log=3` it XORs the column by
row bits at a 16-byte (8 x f16) granule -- exactly the period that would break
a 256-byte-aligned row pitch. There is also a second, independent mechanism,
`alloc_tile_padded` (`alloc.rs:247`, `WMMA_SMEM_PAD`), which pads a tile's row
pitch off the bank-period multiple instead of permuting columns -- probably
the *simpler* of the two to apply here, since it needs no change to how a
read computes its column index, only to the tile's physical stride.

**Neither is wired to the code path `attention_block` actually uses.**
`swizzle_col`/`swizzled_index` are called only from `matmul/mma_sync.rs:492`
(the `nvgpu.mma.sync` path, sm_80+/64-bit index) and `matmul/wmma.rs:490` /
`matmul/stage.rs:199` (legacy WMMA staging). `attention_block_src` carries no
`@tensorcore` tag, so its `dot`/`dot_t` calls lower through the generic
fallback in `phobos-lang/src/codegen/tile/contract.rs`, whose fragment-load
loop does not call `swizzle_col` anywhere -- confirmed by grepping every
`swizzle_col`/`swizzled_index` call site in the codegen tree. **Turning on
`.swizzle` for K/V's tile allocation without also teaching `contract.rs`'s
reads to route through `swizzle_col` would silently corrupt the result**: the
staging store (which does call `swizzled_index` via `tile_copy`,
`elem.rs:148`) would land data at permuted columns while the unmodified
`vector.contract` reader kept reading unpermuted ones. This is exactly why
this is not a one-line "add `@swizzle`" fix, even though the machinery exists.

### Design proposal, ranked

1. **`alloc_tile_padded` on K/V's shared tile** (and check whether `s`, the
   `[BR,BR]` score tile, needs the same treatment -- its own row pitch,
   `BR*4 = 32` bytes for `BR=8` f32, is far under the 128-byte period so it is
   likely not part of this particular collision, but should be checked with
   the same arithmetic once a candidate fix is built). Padding changes only
   the physical stride the tile allocates and the index arithmetic already
   in place for any padded buffer (`row_stride`, already a first-class
   `MemVal` field) -- no change needed to `contract.rs`'s read loop, since
   physical-address computation from `(row, col)` already respects
   `row_stride` uniformly. This is the lower-risk of the two mechanisms to
   land first.
2. **`alloc_tile_swizzled` + a `contract.rs` change to route fragment loads
   through `swizzle_col`.** More invasive (touches the generic dot lowering
   every non-tensor-core kernel in the tree uses, not just this one kernel),
   but reusable beyond attention if the same row-pitch-is-bank-period pattern
   shows up elsewhere (worth a quick audit: any other f16 tile at a head
   dimension that is a multiple of 64 elements hits the same 256-byte-pitch
   coincidence).

Neither was attempted this round -- this task was measurement-first per its
own brief, and a swizzle/pad change needs a correctness re-verification
(`backend_check` etc.) that a design report shouldn't skip past.

### Free context for whoever picks this up

Same `--set full` capture: **occupancy is capped at 50% theoretically (achieved
47.40%) by shared memory**, not registers or warps (`Block Limit Shared Mem: 2`
blocks/SM at 25.25 KB/block static; `Block Limit Registers: 3`,
`Block Limit Warps: 4`). Grid `(16,16,1)` = 256 blocks over 48 SMs at 2
blocks/SM occupancy is 2.67 waves/SM, so a third of the tail wave runs alone
-- ncu's own "Est. Speedup: 33.33%" note on this. Neither of these was this
task's target, but a future beam that touches this kernel's shared-memory
footprint (a swizzle or pad both leave the *byte count* roughly the same, so
this shouldn't move) should keep the occupancy ceiling in view since it is
already the binding constraint, ahead of either the bank-conflict or barrier
findings in isolation.

---

## Item 3: barrier count vs. theoretical minimum

**Finding: 14 `gpu.barrier` per dynamic causal-loop iteration, static count 48
across the whole kernel, against ~9 truly necessary sync points by the
accounting below -- a real but modest ~1.5x gap. But every barrier maps to a
real per-statement tile boundary, so this is not a "the compiler forgot to
elide something provably safe" finding -- it is a "the tile compiler
materializes every named `var`/reassignment as its own synchronized
shared-memory phase" finding, which needs statement fusion, not barrier
deletion, to close.**

### Count, cross-validated two ways

Static MLIR (`gpu.barrier` grep on the item-4 unmodified probe, the exact
shipped kernel): **48 total**, of which **14 sit inside the causal loop's
single static body** (executed once per dynamic iteration) and 34 are in the
one-time prologue/diagonal-tail/epilogue sections. `ncu`'s own
`launch__barrier_count` metric on the real card capture reports the identical
**48** -- an independent cross-check that the static count is what actually
ships, not an artifact of how the probe was built.

`ncu --set full`'s own Warp State Statistics section names barriers as the
single largest stall category measured this round: **"each warp of this
workload spends 7.4 cycles being stalled waiting for sibling warps at a CTA
barrier... this stall type represents about 39.6% of the total average of
18.6 cycles between issuing two instructions."** That is larger than the
bank-conflict finding's own 36.56% estimated speedup, making barriers the
single biggest lever this whole audit measured, by ncu's own numbers --
though, per the actionability ranking above, the hardest of the four to turn
into a scoped fix.

### Mapping barriers to statements, and the true minimum

Correlating each `gpu.barrier` to the `memref.get_global` immediately before
it (each new `memref.global` is a fresh tile the source's next `var`/
reassignment introduces) against the loop body's source:

```
let k = K[...]            -> stage to shared, barrier            (1)
let v = V[...]             -> stage to shared, barrier            (2)
var s = dot_t(q, k)        -> write s, barrier (before scale reads it)  (3)
s = s * scale              -> read+write s in place, barrier before rowmax (4)
var mn = rowmax(s)         -> write mn, barrier                   (5)
mn = tmax(m, mn)            -> read+write mn, barrier              (6)
s = exp(s - mn)             -> read s+mn, write s, barrier         (7)
var corr = exp(m - mn)      -> write corr, barrier (before l/acc read it) (8)
l = l*corr + rowsum(s)      -> read s+corr, write l
acc = acc*corr + dot(s, v)  -> read s+v+corr, write acc, barrier (WAR: next iter's k/v copy must not overwrite s/v while this dot is still reading) (9)
m = mn
```

That is on the order of **9 real dependency points** by this accounting, not
the 4-5 a first pass at the online-softmax data-flow alone would suggest
(which misses two real ones: a barrier after `corr` is published, since
`l`/`acc`'s update reads it under a different thread mapping than the one
that wrote it, and a WAR barrier at the loop bottom, since the next
iteration's K/V copy would otherwise race the current iteration's still-in-flight
`dot(s, v)` read of the old `v` tile). 14 measured against roughly 9 truly
necessary sync points is a smaller, more defensible gap than the naive
4-5-point estimate implied -- call it **up to ~1.5x**, not the ~2.5-3x a
first look at the raw counts suggests, once every real cross-statement
dependency (including the two easy to miss) is accounted for.

**The remaining gap is not a removable barrier -- it is duplicated
materialization.** Several of the 14 bracket a single logical operation that
the tile language currently expresses as two or three separate statements,
each of which gets its own shared-tile phase: `s = s * scale` is scale
folded into a *later* statement rather than into `dot_t`'s own epilogue;
`exp(s - mn)` and `rowsum(s)` are two passes over the same tile that could in
principle share one; `corr`'s publish-then-consume could fold into the
`l`/`acc` update statements directly instead of round-tripping through its
own named tile. None of these are things `attn.rs`'s `.ph` source text
controls today (barriers are 100% codegen-inserted, invisible at the source
level) or that a human should hand-remove from the emitted MLIR -- the task's
own standing caution applies directly: this project's masking/aliasing rules
(the same ones the BR=32 tensor-core postmortem found force duplicate tile
allocation for same-shape-different-name values) are almost certainly *why*
the codegen brackets each statement this conservatively, and a wrong removal
here is a correctness bug, not a missed optimization.

### Design proposal

**Statement fusion in the tile compiler**, not barrier elision: teach the
codegen to recognize a chain of elementwise/reduction statements over the
same tile identity (e.g., `s = f(s)` immediately followed by a reduction
consuming `s`) and materialize the *chain's* result rather than each
intermediate, collapsing the barrier pairs that currently bracket each link.
This is real `phobos-lang` compiler work -- a new pass or an extension to
however tiles currently pool/dedupe by name -- not a kernel-level change, and
squarely out of scope for a "measurement-first" task. Flagging it as the
concrete next step rather than attempting it here.

---

## Item 1: `qmma_t`'s inner K-loop load widths

**Finding: confirmed at the PTX level and independently at the SASS level
(`ptxas -arch=sm_75` + `nvdisasm`, the actual driver-JIT target) -- every
operand load in the K-loop is a 4-byte `LDG.E.SYS`. Zero merged/widened loads.
This matches the task brief's premise exactly and gives it a hard number.**

### Measurement

Compiled the real deep-tile shape (`TM=TN=128`, matching minicpm's o_proj/
down_proj/qkv projections) via `cargo run -p phobos-lang --example ptx`, then
`ptxas -arch=sm_75 -O3` (the real target; `phobos-gguf` compiles at `sm_75`)
and `nvdisasm` on the resulting cubin:

- PTX: 56 `ld.global.b32` instructions per K-loop iteration, **zero** wider
  (`grep -c "ld.global"` = 56, `grep -o "ld\.global\.[a-z0-9.]*" | sort | uniq -c`
  = `56 ld.global.b32`, no `.64`/`.v2`/`.v4` variant anywhere in the file).
- SASS (post-`ptxas`, the actual scheduled instructions the card runs):
  identical count and width, `56 LDG.E.SYS`, no `.64`/`.128` suffix on any of
  them. `ptxas` did not find a coalescing opportunity across these loads on
  its own.

Of the 56: **32 are the `a_frags`/`w_frags` operand loads** this task's
hypothesis targets (`rm=rn=8` at `TM=TN=128`, `halves=Q8_BLOCK/IMMA_K=2`,
so `16` A fragments + `16` W fragments, matching the task brief's own
hand-traced prediction exactly). The remaining 24 are legitimately-scalar
scale loads (8 `a_rows`-worth of `asc` + `rn*2=16` of `wsc`, `qmma.rs:255,261`)
-- these are per-row/per-column scalars by construction, not a target for the
same widening.

### Why `ptxas` can't merge these on its own

Traced the exact addresses `qmma_t_into` generates
(`phobos-lang/src/codegen/tile/qmma.rs:227-240`): for a fixed row, the two
`halves` loads (`h=0`, `h=1`) read bytes `[k_from, k_from+4)` and
`[k_from+16, k_from+20)` -- 12 bytes apart, not adjacent, because each lane's
`k_off = in_quad*4` is already folded into the base and `IMMA_K=16` separates
the two halves. Across rows, `a_rows`/`w_rows` are a full row-pitch (`kd`,
the contraction depth, hundreds to thousands of elements) apart. Neither gap
is closeable by instruction scheduling alone; `ptxas` cannot invent
contiguity that isn't there in the addresses it's given. A wider load is only
possible if the *data itself* is relaid out so what one lane needs is
physically contiguous -- confirming the task brief's proposed mechanism
(permute the operand layout) is the right shape of fix, not an
instruction-scheduling one.

### Design proposal, sized by tier

- **2x tier (lower risk)**: interleave just the two `Q8_BLOCK`-halves per row
  so they land 4 bytes apart instead of 12 -- one `vec8_i8` (8-byte) load
  replaces each row's two `vec4_i8` loads, on both `A` and `W`. This halves
  the 32 operand loads to 16, no change to which rows/columns a lane touches,
  only how the two halves within one lane's row are packed.
- **4x tier (the task brief's full target, higher risk)**: a
  lane-fragment-major relayout, where a lane's *entire* row-share across
  `halves` (and potentially across the `rm`/`rn` patch, if a viable packing
  exists) is stored contiguously at quantize/upload time -- the full `vec16`
  reach the task brief names. This is a real quantize-format change, not a
  kernel change alone.

**Coordinated change sites, both needed for either tier**: the read side is
`qmma_t_into`'s `kb` block (`phobos-lang/src/codegen/tile/qmma.rs:227-240`,
where `a_frags`/`w_frags` are loaded) -- change the `vec_load_al` width and
the fragment-cast shape to match the new packing. The write side is wherever
`W` (and, for the 4x tier, `A`) gets its on-disk/on-upload byte layout chosen
in `phobos-gguf`'s quantizer/upload path -- not touched or located precisely
in this audit, since this task was scoped to `phobos-lang`'s codegen side; a
future beam should locate it before implementing. **The `A` (activation)
operand's layout is produced by whatever quantizes it at runtime, which is
exactly the sibling tail-fusion beam's `quantize`-kernel work this round** --
flagging the connection per this task's brief, not touching those files.

**Register-neutrality, worth naming explicitly**: unlike the K-loop prefetch
beam (`autoresearch/beams/q8-qmma-kloop-prefetch.md`), which lost 8.8-42% by
trading register headroom `ptxas` was already using for a hand-scheduled
pipeline, this fix reduces *instruction count* for the *same bytes moved per
iteration* -- no new live registers, no loop-carried state, nothing for
`ptxas` to lose scheduling freedom over. That's a materially different risk
profile from the beam that already failed on this same kernel, and worth
flagging to whoever scopes the next attempt so it isn't dismissed by
analogy to that negative result.

---

## Evidence files

Left in `autoresearch/beams/`, matching this session's existing convention:

- `ncu_attnblock_prefill_light.ncu-rep` / `.csv` -- 8 launches, light metrics
  (bank conflicts, long-scoreboard/barrier stall ratios), minicpm `pp128`.
- `ncu_attnblock_prefill_full.ncu-rep` / `.txt` -- one launch, `--set full`,
  the source of the bank-conflict/barrier-stall/occupancy numbers quoted
  above.

All `.ph`/`.mlir`/`.ptx`/`.sass`/`.cubin` probe files used to derive the
item-1 and item-4 findings lived in the session scratchpad
(`C:\Users\joaeb\AppData\Local\Temp\claude\...\scratchpad`), never in this
worktree; none were committed or left behind.

Do not cite this session's `ncu`-captured `t/s` figures (2125, 1695 -- visible
in the two runs above) as throughput numbers; profiling perturbs timing.
The per-kernel launch durations (167-180us across 8 samples) are the usable
number, and cross-check the task's own 4.47ms/pass figure reasonably well
(180us x ~25 layers ~= 4.2-4.5ms).
