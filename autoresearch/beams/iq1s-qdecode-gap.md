# Beam C: why `iq1s_qdecode` is 11.4x slower than `iq1s_qdot_matvec`

Opened 2026-08-31 for `autoresearch/beams/qwen38-beat-llamacpp.md` beam C.
Host-only: PTX + SASS + byte accounting. No GPU run was made for this note.

## 1. Verdict

**(a) Write-side. The global store of the expanded weight is the gap; the body
accounts for the residual 1.7%.**

Both kernels decode the same 11.901e9 IQ1_S elements from the same 2.3245 GB of
block bytes, with the same 5 global load instructions per 8 elements per lane,
the same decode arithmetic, zero barriers and zero shared memory inside the
k loop, and identical occupancy (4 CTAs/SM, the sm_75 maximum, for both). The
only instructions that differ are the 8 scalar stores.

| | qdot (decode) | qdecode f16 (prefill) |
| --- | ---: | ---: |
| time to sweep all IQ1_S | 34.6 ms | 395.6 ms |
| elements / s | 344.0e9 | 30.08e9 |
| DRAM bytes / element | 0.1953 | 2.1953 |
| achieved DRAM | **67.2 GB/s** | **66.0 GB/s** |

The two kernels move DRAM bytes at the same rate to within 1.7%. One of them
moves 11.24x more bytes. 11.24 x 1.017 = 11.43, which is the measured ratio.
The residual 1.7% is the whole body-side contribution.

**Consequence for beam B: the fused IQ qmma inherits no defect from
`_qdecode`.** The 8 stores are the cost, and fusing deletes them. Two riders
in section 6 that B does have to respect.

## 2. What was emitted, and the proof it is what ran

`iq1s_qdot_matvec_src(8)`, `iq1s_qdecode_src(8)` and its `f16_scratch` rewrite,
reproduced verbatim and emitted with `target/release/examples/ptx.exe` at the
compile context `phobos_kernels::compile` uses (`Context::default()`: sm_75,
`+ptx90`, `nvptx64-nvidia-cuda`, index bitwidth 32; `@autotune(TN in [8])`
seeds TN=8, so the `("TN", IQ1S_TN)` override is a no-op).

All three emitted texts are **byte-identical to entries already in
`~/.phobos/kernel-cache`** (`iq1s_qdot_matvec-*` 6 entries, `iq1s_qdecode-*`
8 entries, split f32/f16), i.e. to the PTX the model actually loaded. `ptxas -v`
independently reproduces the pass report's `iq1s_qdot_matvec` line (32 bytes
smem, 36 registers), which is a second check that this is the same code.

The pp128 trace's `iq1s_qdecode` is the **f16** variant.
`project_raw_dense` picks it when `narrow`, and for this model `narrow` holds
everywhere: m=128 is a multiple of `TC_TILE_M`=64, every k (5120, 6144, 17408)
is a multiple of `TC_TILE_K`=16, `strip` is rounded down to a multiple of
`TC_TILE_N`=64 and every n is itself a multiple of 64, so no strip is ragged.

## 3. Launch geometry

Identical, which is the first thing to note: the difference is not the shape of
the launch.

| | `iq1s_qdot_matvec` | `iq1s_qdecode` |
| --- | --- | --- |
| `@launch` | 256 threads (8 warps) | 256 threads (8 warps) |
| TN (columns per CTA) | 8 | 8 |
| grid | `(n/8, m, 1)`, m=1 | `(cur/8, 1, 1)` |
| thread map | `j = li/32`, `lane = li%32` | `lane = li/8`, `j = li%8` |
| a warp owns | **1 output column x 32 format lanes** | **8 output columns x 4 format lanes** |
| elements / thread / k-block | 8 | 8 |
| elements / CTA / k-block | 2048 | 2048 |
| outer `li` loop trips | 1 | 1 |
| registers (ptxas -v) | 36 | 46 |
| shared / CTA | 32 B (the `[1,TN]` out tile, outside the k loop) | 0 |
| barriers | 1 (`used 1 barriers`), outside the k loop | **0** |
| CTAs / SM (sm_75, 64K regs, 32 warps) | 4 (full) | 4 (full) |

Both kernels decode every element exactly once per pass, and each thread owns
one IQ1_S format lane and its 8 elements, so the per-thread work is the same.

## 4. Instruction histogram, per 8 decoded elements per lane

One k-loop trip = one 256-element block = 8 elements for this lane. PTX from
`ptx.exe`, SASS from `ptxas -arch=sm_75` + `cuobjdump -sass`.

### Global memory, k-loop body

| op | qdot | qdecode f32 | qdecode f16 |
| --- | ---: | ---: | ---: |
| `ld.global.b8` (qs, qh_lo, qh_hi) | 3 | 3 | 3 |
| `ld.global.b16` (block scale d) | 1 | 1 | 1 |
| `ld.global.v2.b32` (grid entry, 8 B) | 1 | 1 | 1 |
| `ld.global.v4.b32` (activation, 16 B) | **2** | 0 | 0 |
| `st.global.b32` | 0 | **8** | 0 |
| `st.global.b16` | 0 | 0 | **8** |
| `ld.shared` / `st.shared` | 0 | 0 | 0 |
| `bar.sync` | 0 | 0 | 0 |
| `shfl.sync` | 0 | 0 | 0 |

SASS is the same picture: `2 LDG.E.128 + 1 LDG.E.64 + 1 LDG.E.U16 + 3 LDG.E.U8`
for qdot, `1 LDG.E.64 + 1 LDG.E.U16 + 3 LDG.E.U8 + 8 STG.E(.U16)` for qdecode.

### Totals

| | qdot | qdecode f32 | qdecode f16 |
| --- | ---: | ---: | ---: |
| PTX instructions, k-loop body | 101 | 98 | 106 |
| **SASS instructions, k-loop body** | **88** | **88** | **96** |
| SASS instructions, whole kernel | 208 | 136 | 144 |
| f32 mul / add in body | 18 / 17 | 10 / 9 | 10 / 9 |
| `cvt.rn.f16.f32` | 0 | 0 | 8 |
| `prmt.b32` (grid byte extract) | 6 | 6 | 6 |

`iq1s_qdecode` with an f32 scratch is **88 SASS instructions, exactly as many as
`iq1s_qdot_matvec`**, and it reads two fewer global operands. There is no
instruction-shape defect to find. The f16 variant adds 8 `cvt.rn.f16.f32` for
+9%, which is nowhere near 11x.

### Outside the k loop

`iq1s_qdot_matvec` closes with a 5-step `SHFL.BFLY` reduction (a 6th appears in
a cold path off `BRA.DIV`) plus a 4-byte `st.shared` per warp,
then one `bar.sync`, one `LDS.U.128` and one `STG.E.128` to drain the 32-byte
output tile, then a second `bar.sync`. This is per output column, amortised
over K/256 k-loop trips (20 to 68 of them), so it is under 1% of the kernel.
**It is the output tile, not staging**: nothing in the decode passes through
shared memory in either kernel.

## 5. The arithmetic

IQ1_S in `models/Qwen3.8-27B-UD-IQ1_M.gguf`, from
`examples/inspect`: **163 tensors, 11,901,337,600 elements**, 46,489,600 blocks
of 256. (`inspect --tensors` summed per tensor gives the same 11.9013e9 over
163 shapes, all of them 5120 or 17408 on a side.) Device layout drops the
leading f16 into a separate plane
(`Quant::device_block`), so a block costs 48 B of `qb` + 2 B of `d` = 50 B:
**2.3245 GB read by both kernels.**

| | qdot | qdecode f16 |
| --- | ---: | ---: |
| weight reads | 2.3245 GB | 2.3245 GB |
| activation reads | 7.73 MB unique (`sum(k) x 4`), L2-resident at 20-68 KB a matmul | none |
| output writes | 4.91 MB (`sum(n) x 4`, one f32 a column) | **23.803 GB** (`SCRATCH`) |
| total DRAM | 2.325 GB | 26.127 GB |
| bytes / element | 0.1953 | 2.1953 |
| time | 34.6 ms | 395.6 ms |
| **achieved** | **67.2 GB/s** | **66.0 GB/s** |

Byte ratio 11.240x, time ratio 11.434x, **achieved-bandwidth ratio 1.017x**:
qdecode costs 1.7% more per byte moved, and that 1.7% is the entire body-side
contribution. The last two lines are the same two numbers rearranged, so state
the claim as the mechanism: *the same sector rate, moving 11.24x the bytes.*
What makes it
evidence rather than a tautology is section 4: the bodies are the same size, so
there is nothing else the extra time can be.

### Read sectors are identical per CTA, which kills the read-geometry worry

The two thread maps look very different per warp: qdot's warp has `j` fixed and
`lane` 0..31, so its 32 `qs` byte loads are 32 consecutive bytes of one row
(2 sectors); qdecode's warp has `j` 0..7 and `lane` in one group of 4, so its
32 byte loads hit 8 rows, 4 bytes each (8 sectors). Per warp that is an 8x
sector amplification.

**Per CTA it cancels exactly.** Both CTAs cover the same 8 rows x 48 B block +
8 x 2 B scale per k-block: 24 sectors, 768 B, 2048 elements, 0.375 B/element
either way. qdot's warps each take one row whole; qdecode's warps each take a
slice of all eight, and the eight warps union to the same lines. The
intra-warp difference is latency and MSHR pressure, not L2 or DRAM traffic, and
the equal achieved bandwidth in the table above says it is not costing anything
measurable.

## 6. The one body-side defect, and it lives on the store

`SCRATCH` is `[K, N]`, a thread owns 8 consecutive **rows**, so its 8 stores are
strided by the full row pitch. In PTX: `%rd21 = N * 2` and each store is
`add.s64 %rdN, %rdN-1, %rd21`. They can never vectorize. That is inherent to
the layout, not a missed optimization.

The sector fill is the part that is fixable:

| | per warp store instruction | sectors touched | fill |
| --- | --- | ---: | ---: |
| f32 scratch | 4 rows x 8 cols x 4 B = 32 B/row | 4 | **100%** |
| f16 scratch | 4 rows x 8 cols x 2 B = 16 B/row | 4 | **50%** |

The comment on `qdecode_t_into` ("a warp's 32 stores land in four fully covered
32-byte sectors") was written for the f32 destination and is now stale: at f16 a
warp covers half of each of four sectors, and the other half belongs to CTA
`pn^1`. It is recovered only if that CTA's write merges in L2 first.

Supporting figure, **cross-session and therefore weaker** (2026-08-25
`docs/PERF-QWEN27B.md` vs the 2026-08-31 trace, different sessions, not
comparable as absolutes): f32 -> f16 cut useful bytes 1.91x but time only 1.29x.
Exactly the direction half-sector stores predict.

The lever is the tile width. A full 32-byte sector is 16 f16 columns, so
`iq1s_qdecode` wants its own **`TN = 16`** (like `IQ1S_I8_TN`, kept apart from
the `IQ1S_TN = 8` the qdot and `_matvec` paths use). `lane = li/16`, `j = li%16`
then gives a warp 2 rows x 16 consecutive f16 = **2 full sectors** instead of 4
half ones. `total = cols * WARP` becomes 512, so a thread takes two `li` trips
at 256 threads, which the loop already handles, and per-CTA read sectors are
unchanged at 0.375 B/element.

**It is not a one-constant change, and doing it as one corrupts memory
silently.** In `project_raw_dense`, the `tn` taken from the `_dequant` match
table feeds three places: the `cur.is_multiple_of(tn)` gate that picks
`_qdecode` over `_dequant`, and the launch grid `(cur.div_ceil(tn), 1, 1)` used
for **both** kernels. Compile `iq1s_qdecode` at TN=16 while the grid still
divides by 8 and the launch has twice the CTAs it needs; each writes 16 columns
at `pn * 16`, so the tail runs to `2 * cur - 16` and scribbles past the strip.
The kernel carries `@aligned(N = TN)` and does no masking, so nothing catches
it, and per CLAUDE.md the damage lands in the *next* allocation, not in this
kernel's result. The change is: a `IQ1S_QDECODE_TN` beside `IQ1S_TN`, carried
alongside the `qdecode` match arm, and the gate and the grid both switched to
whichever tn the chosen module was compiled at. Gate on `backend_check` plus
`model_check`, not on the timing.

Upside is bounded by the 1.91x/1.29x figure above, so call it up to ~1.5x on
`_qdecode`, worth about 130 ms of a 1582 ms pass if it lands whole. **Beam B
does not inherit any of this**, because it deletes the store; this is only for
a format B never reaches.

## 7. Two riders for beam B

**The gap is explained; the level is not.** Both kernels sit at 13% of this
card's 496 GB/s. A fused kernel does not get 496 GB/s by fiat. Budget beam B
against qdot's *measured* per-element rate: **344e9 elements/s for IQ1_S at
m=1**, from 11.901e9 elements in 34.6 ms. That is ~3x the "best rate the qdot
family has ever shown" the parent beam file assumed (23.3e9 to 54.3e9 from the
08-25 session), so B's "~240 ms before a single MMA runs" is pessimistic by
about that factor and B's margin is better than the beam file states. Recheck
that arithmetic before choosing B's tiling.

**Wave fill is a second-order tax on the level, not on the gap.** At 4 CTAs/SM
the card holds 192 CTAs. `RAW_DEQUANT_BUDGET_BYTES` = 128 MiB gives
strip = 6528 at k=5120 (816 CTAs, 4.25 waves), 5440 at k=6144 (3.54 waves) and
1920 at k=17408 (**240 CTAs, 1.25 waves**). The down-projection strip is the
one that quantizes badly. qdot launches n/8 CTAs over the whole n and fills
better (640 and 2176). One sentence, not a beam.

## 8. Residency confound, bounded

The pp128 trace was taken with the card at 96% VRAM, and the parent beam's root
cause 1 is that a cold allocation gets paged over PCIe at 8.9 GB/s. If
`iq1s_qdecode`'s 2.3245 GB of reads were paged, that alone would be ~260 ms of
the 395.6.

Bounds, host-only:

- The 23.803 GB of writes cannot be over PCIe. That would need 60 GB/s against
  a measured 8.9. The scratch is device-resident and the writes are real VRAM
  writes.
- The paging victim in the parent beam's own diagnosis is the **cold** 521 MiB
  LM head. The IQ1_S weights are touched by every layer of every pass, so they
  are the last thing the driver evicts, and the decode trace confirms they are
  resident there (2.3245 GB in 34.6 ms is 67 GB/s, impossible over PCIe).
- Even in the adversarial case where prefill's reads *are* paged and fully
  serialized (260 ms), the remaining 136 ms is 23.8 GB of writes at 175 GB/s,
  and the store is still the majority of the time. **The verdict does not turn
  on this**: section 4 rules out (b) on instruction shape alone, independently
  of where the bytes came from.

## 9. Beam status

**Beam C: RESOLVED, verdict (a).** It gated beam B's expected value and the
answer is favourable: B removes the cost outright and inherits no lane-geometry
defect. Two carry-overs, both in section 7: budget B against 344e9 elements/s,
not the 08-25 rates; and if a two-pass path survives for a format B never
reaches, section 6's full-sector store is worth ~1.5x on it.

Reproduce: three `.ph` sources per section 2, `ptx.exe FILE.ph`,
`ptxas -arch=sm_75 -v`, `cuobjdump -sass`. No GPU needed.
