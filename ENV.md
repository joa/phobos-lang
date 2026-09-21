# Environment variables

Every `PHOBOS_*` variable the tree reads, what it takes and what it does.
Nothing here is a user setting in the ordinary sense: `phobos-cli`'s flags
are the user's choices, and these are the knobs behind them for measuring,
debugging and sweeping. A variable a document does not list is a variable
the tree does not read; add it here when adding it there.

Two spellings apply throughout, and every toggle uses one of them:

- **opt-in**: on for `1`, `on`, `yes` or `true`; off otherwise, including
  unset.
- **opt-out**: off for `0`, `off`, `no` or `false`; on otherwise, including
  unset.

A variable that takes a value says so. Where a value fails to parse the
default applies. Most are read once, at backend construction or first use,
so they have to be set before the process starts. The helpers are
`phobos_base::env::flag` (opt-in), `flag_on` (opt-out) and `flag_set`
(opt-out, with unset told apart for a toggle that falls back to a wider
one, as the `PHOBOS_FUSED_*` stages fall back to `PHOBOS_FUSED`).

## Logging and tracing

| variable | takes | default | effect |
| --- | --- | --- | --- |
| `PHOBOS_LOG` | `off`/`none`/`0`, `info`, `debug`, `trace` | `info` | The level `phobos_base::log` emits at, for every crate's messages. |
| `PHOBOS_TRACE` | opt-in | off | Prints each block's per-token activation RMS during a forward pass (`llama`, `qwen35`), a readback per block, so only for a diagnosis. |
| `PHOBOS_PASS_REPORT` | a replay number, or empty | unset | On the device backend, prints the launch census of that replay of the pass (kernel, launches, blocks, registers, shared memory, occupancy). Empty means the fourth replay, past the prefill and the warmup passes whose scratch is still growing. |
| `PHOBOS_VRAM` | opt-in | off | Prints free device memory at each pass boundary and every large allocation, constant and scratch, with the pool's free list; how a residency problem is found. |
| `PHOBOS_CHECK_BUFS` | opt-in | off | Checks at every launch that a kernel's destination is not one of its sources. A debugging aid; it reads the slot table per launch. |

## Kernel compilation and the cache

| variable | takes | default | effect |
| --- | --- | --- | --- |
| `PHOBOS_KERNEL_CACHE_DIR` | a directory, or empty | `~/.phobos/kernel-cache` | Where compiled PTX is cached across processes. Empty turns caching off. |
| `PHOBOS_KERNEL_CACHE_EPOCH` | any string | the compiler's build fingerprint | Replaces the fingerprint that keys the cache, so a stale cache can be forced cold or two builds made to share one. |
| `PHOBOS_CHIP` | an SM target, `sm_75`, `sm_80` | the card's, or the context default | The target `phobos-lang`'s `emit`, `ir` and `ptx` examples compile for, to reach a code path the default never takes. |
| `PHOBOS_INDEX_BITS` | `32` or `64` | `32`, widened to 64 by a kernel that wants `ldmatrix` | The index width those examples compile with; `mma.sync` and `cp.async` want 64. |
| `PHOBOS_PRINT_PHASES` | opt-in | off | The `ptx` example prints each lowering phase's IR. |
| `PHOBOS_DUMP_DIR` | a directory | unset | Codegen tests and the raw-format kernel dump write every kernel source they compile there as `.ph`, so the emit sweep covers them. |
| `PHOBOS_SNAPSHOT_SRC` | a directory | unset | The IR builder's test walks every source there and checks each builds, or fails where its `.err` says it should. Skipped without it. |

## The device backend: memory

| variable | takes | default | effect |
| --- | --- | --- | --- |
| `PHOBOS_ARENA` | opt-out | on | Weights go into a few large arena slabs rather than an allocation apiece; WDDM keeps a small set of large allocations resident where it fails a large set of small ones. |
| `PHOBOS_ARENA_CONST` | opt-in | off | f32 constants go into the arena too. |
| `PHOBOS_STATE_ARENA` | opt-in | off | A sequence's recurrent state goes into an arena of its own, reset when the sequence ends. |
| `PHOBOS_SLAB_MIB` | MiB | 128 | Arena slab size. A tensor larger than a slab gets one to itself, which is how the output head can be evicted on its own. |
| `PHOBOS_DEQUANT_MIB` | MiB | 32 | Budget of the strip a prompt pass dequantizes a raw-format weight into, for formats without a fused projection. Smaller buys residency and costs launches. |
| `PHOBOS_DENSE_SCRATCH` | opt-out | on | The prompt pass's dense scratch is one shared allocation for the whole model rather than one per weight. |
| `PHOBOS_TRIM` | opt-in | off | After a prompt pass, hand the dense scratch and the pool's free list back to the driver at the next pass boundary. |

## The device backend: kernel selection

| variable | takes | default | effect |
| --- | --- | --- | --- |
| `PHOBOS_FUSED` | opt-out | on | The fusion pass as a whole: chains of decode stages lowered into one persistent kernel. Off, every stage launches on its own. |
| `PHOBOS_FUSED_MLP` | opt-out | on | The decode MLP (norm, gate and up, SwiGLU, down) as one fused kernel. |
| `PHOBOS_FUSED_PROJ` | opt-out | on | A normalization and the projection reading it as one kernel. |
| `PHOBOS_FUSED_MIX` | opt-out | on | The delta net's convolution and gates as the tail of the fused projection; costs a barrier, so gated apart. |
| `PHOBOS_FUSED_ATTN_OUT` | opt-out | on | Attention's output epilogue (quantize, output projection into the residual) as one kernel. |
| `PHOBOS_FUSED_STORE2D` | opt-out | on | Attention's key and value cache writes in one launch instead of two. |
| `PHOBOS_ATTN_PERSIST` | opt-out | on | The decode attention's persistent split-and-merge kernel. Off, the launched split path. Kept apart from `PHOBOS_FUSED` since a `grid_barrier` that does not fit hangs rather than slows; occupancy is checked before it is taken. |
| `PHOBOS_IQ1S_DP4A` | opt-out | on | The `dp4a` decode matvecs against an int8 activation, for every raw format. Off, the f32 matvec. It quantizes the activation where the host reference does not, so device against host cannot judge it; `backend_check` runs both device ways instead. |
| `PHOBOS_QGEMM` | opt-out | on | The staged tensor-core prompt projection for the raw formats. Off, the register form or the dense expansion. |
| `PHOBOS_RAW_QMMA` | opt-out, or a comma list of formats (`IQ1_S,IQ2_XXS`) | on, all formats | Which raw formats take the fused prompt projection rather than expansion. A list names them; an opt-out spelling means all or none. |
| `PHOBOS_QMMA_SPLIT` | opt-in | off | Split-K on the Q8_0 prompt projection's deep tile when the grid is starved. Measured a net loss; kept to measure again. |
| `PHOBOS_QMMA_NARROW` | opt-in | off | The narrow-CTA form of the Q8_0 prompt projection's deep tile. Wins at pp128, regresses pp512 on some models; opt-in for that reason. |
| `PHOBOS_QMMA_TILE` | `TMxTNxCTA`, e.g. `128x64x256` | the format's | The fused prompt projection's tile and CTA, for a sweep without a rebuild. Each distinct value is a distinct compiled kernel. |
| `PHOBOS_QMMA_STAGE` | opt-out | on | The staged (shared-memory) form of IQ1_S's fused projection. `0` goes back to the register form. |
| `PHOBOS_QDOT_I8_TN` | a power of two, 8 to 256 | 64 | Columns a `dp4a` decode matvec's tile covers, for a sweep. Each value is a distinct compiled kernel. |
| `PHOBOS_QDOT_I8_CTA` | threads | 256 | Threads those matvecs launch with, clamped so a warp's columns fill the CTA whole. Wider reads faster alone and loses in the model. |
| `PHOBOS_PERSIST_QDOT` | opt-in | off | The persistent (card-sized grid) form of the Q8_0 decode matvec, which exists to be measured against the launched one. |
| `PHOBOS_PERSIST_BLOCKS` | a block count | the occupancy answer | Forces that grid's block count, to ask what a matvec loses at the count a fused kernel is stuck with. |

## The device backend: streamed experts

The mixture-of-experts path (`qwen35moe`) keeps a model's experts in a
pinned host mirror and a device cache; see `docs/MOE-QWEN36-35B-A3B.md` for
the design and every measurement behind these defaults. All three are
opt-in and all three measured slower than the default on an RTX 2080 SUPER
whose PCIe link runs at x8; they exist for a wider bus or a faster host.

| variable | takes | default | effect |
| --- | --- | --- | --- |
| `PHOBOS_MOE_LOOKAHEAD` | opt-in | off | At each block's sync point, run the next block's router on the residual as it stands and copy its predicted misses early on a second stream. Right about 80% of the time; the wrong fifth is extra bytes on the bus, and on an x8 link that costs more than the overlap buys (tg128 24.8 against 32.0). |
| `PHOBOS_MOE_CPU_MISS` | opt-in | off | Compute a decode step's misses on the host from the mirror instead of copying them, a zero slot standing in on the device. With the reference decoder a miss is ~1,145 us against ~290 us to copy; a SIMD K-quant dot is what would make it pay. |
| `PHOBOS_MOE_GROUPED` | opt-in | off | Run a prompt pass's experts as grouped GEMMs over rows sorted by expert rather than row by row. Checked, not yet timed; on an x8 link the copies are nine tenths of a prompt pass either way. |

## Benchmarks and examples

| variable | takes | default | effect |
| --- | --- | --- | --- |
| `PHOBOS_ATTNDECODE_LENGTH` | a cache length | unset | `attndecode` sweeps only that length, so an external profiler can isolate one launch. |
| `PHOBOS_ATTNDECODE_SHAPE` | a substring of a shape name | unset | `attndecode` sweeps only the shapes whose name contains it. |

`scripts/bench.py` prints the `PHOBOS_*` variables in its environment as
`phobos env:` before a run, since a benchmark taken with one of these set
is not comparable to one taken without; `PHOBOS_VRAM` in particular has
been found inherited by a shell and left on.
