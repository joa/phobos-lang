# Phobos Language Spec

1. [Grammar](#grammar)
2. [Lexical](#lexical)
3. [Types](#types)
4. [Statements](#statements)
5. [Subscripts](#subscripts)
6. [Operators and conversions](#operators-and-conversions)
7. [Bounds and masking](#bounds-and-masking)
8. [Tiles and shared memory](#tiles-and-shared-memory)
9. [Built-ins](#built-ins)
10. [Quantized built-ins](#quantized-built-ins)
11. [Attributes](#attributes)
12. [Examples](#examples)

## Grammar

Notation: `{ x }` = zero or more, `[ x ]` = optional, `( ... )` = grouping, `"x"` = literal token, `(* ... *)` = comment.

```ebnf
program     = { kernel } ;

kernel      = { attribute } "kernel" ident "(" [ params ] ")" block ;

attribute   = "@" ident [ "(" [ attr_arg { "," attr_arg } ] ")" ] ;
attr_arg    = ident "in" "[" int { "," int } "]"       (* autotune search dim: TILE_M in [64,128] *)
            | ident "=" literal                        (* keyword argument:    arch = sm_80       *)
            | literal ;                                (* positional / flag:   256 | read_only    *)
literal     = int | float | ident | "true" | "false" ;

params      = param { "," param } ;
param       = ident ":" type ;

type        = scalar
            | "tensor" "<" scalar ">" "[" dims "]"
            | "tile"   "<" scalar ">" "[" dims "]" ;
scalar      = "f16" | "bf16" | "f32" | "f64" | "i8" | "i32" | "i64" | "bool" ;
dims        = dim { "," dim } ;
dim         = ident | int ;                            (* symbolic size (M, TILE_K) or literal *)

block       = "{" { stmt } "}" ;
stmt        = let_stmt | var_stmt | assign_stmt
            | for_stmt | while_stmt | if_stmt | expr_stmt ;

let_stmt    = "let" ident [ ":" type ] "=" expr terminator ;
var_stmt    = "var" ident ( ":" type [ "=" expr ] | "=" expr ) terminator ;
assign_stmt = lvalue ( "=" | "+=" ) expr terminator ;
lvalue      = ident [ "[" subscripts "]" ] ;           (* a name or an indexed name *)
for_stmt    = "for" ident "in" "range" "(" expr "," expr [ "," expr ] ")" block ;
while_stmt  = "while" expr block ;
if_stmt     = "if" expr block [ "else" ( block | if_stmt ) ] ;
expr_stmt   = expr terminator ;

terminator  = newline | ";" ;                          (* may be omitted before "}" or EOF *)

expr        = equality ;
equality    = comparison { ( "==" | "!=" ) comparison } ;
comparison  = term { ( "<" | "<=" | ">" | ">=" ) term } ;
term        = factor { ( "+" | "-" ) factor } ;
factor      = unary { ( "*" | "/" | "%" ) unary } ;
unary       = [ "-" | "!" ] postfix ;
postfix     = primary { "[" subscripts "]" | "(" [ args ] ")" } ;
args        = expr { "," expr } ;
primary     = int | float | "true" | "false" | ident | "(" expr ")" ;

subscripts  = subscript { "," subscript } ;
subscript   = ":"                                      (* full range:  A[:] *)
            | expr [ ( ":" | ":+" ) expr ] ;           (* point A[i] ; range A[start : end] ; span A[start :+ length] *)

ident       = ( letter | "_" ) { letter | digit | "_" } ;
int         = digit { digit } ;
float       = digit { digit } "." { digit } ;
```

## Lexical

| Item | Rule |
|---|---|
| Keywords | `kernel let var if else for in while true false` |
| Contextual identifiers | `tensor`, `tile`, `range`, every built-in and every scalar type name. Ordinary identifiers outside their position; `range` is only special inside `for ... in range(...)`. |
| Comments | `// ...` to end of line. |
| Statement end | A newline or `;`. Optional before `}` or end of file. |
| `else` | Must follow the `}` of the then-block on the same line; a newline there ends the `if`. |

## Types

| Type | Meaning |
|---|---|
| `f16` | IEEE binary16. Float literals are f32, rounded to f16 on store. |
| `bf16` | 8-bit exponent, 7-bit mantissa. Works on every target (native conversion from sm_80, emulated below). |
| `f32`, `f64` | IEEE binary32 / binary64. |
| `i8` | Signed byte. Loads sign-extend. There is no `u8`. |
| `i32`, `i64` | Signed integers. |
| `bool` | Comparison result. No conversion to or from it. |
| `tensor<T>[dims]` | Global memory, kernel parameters only. Dims are literals or symbolic names bound at launch or by `@autotune`. |
| `tile<T>[dims]` | A block of shared memory owned by the CTA. See [Tiles and shared memory](#tiles-and-shared-memory). |

## Statements

| Form | Meaning |
|---|---|
| `let x = e` | Immutable binding. Initializer required. A tensor slice bound with `let` is a view, not a copy. |
| `let x: T = e` | Same, with a declared type. |
| `var x = e` | Mutable binding. `var t = A[...]` copies the slice into a tile. |
| `var t: tile<T>[R, C] = e` | Mutable tile, filled with `e` (a scalar broadcasts). |
| `var t: tile<T>[R, C]` | Uninitialized tile. The type is required and must be a tile. |
| `x = e`, `x[...] = e` | Assign a whole value or a slice. Tiles and tensors are sliced the same way. |
| `x += e`, `x[...] += e` | Accumulate. |
| `for i in range(lo, hi[, step])` | `i` runs over `lo, lo + step, ...` while `< hi`. `step` defaults to 1. |
| `while c { }`, `if c { } else { }` | Scalar condition. |

## Subscripts

| Form | Selects | Example |
|---|---|---|
| `A[i]` | One index | `A[r, c]` |
| `A[s : e]` | `s <= i < e` | `A[0 : 64, :]` |
| `A[s :+ n]` | `s <= i < s + n`, same as `A[s : s + n]` | `A[pm * TM :+ TM, :]` |
| `A[:]` | The whole dimension | `Q[row :+ BR, :]` |
| `A[i:]`, `A[:j]` | Not supported. `:` needs an end, `:+` a length. | |

## Operators and conversions

| Rule | Behavior |
|---|---|
| Arithmetic on tiles | Elementwise. `tile op scalar` and `scalar op tile` broadcast the scalar. |
| Broadcasting | NumPy-style over axes of extent 1: `[R, C] op [R, 1]` stretches the column. |
| Unary `-` | On a tile, elementwise `0 - t`. |
| Unary `!` | Scalars only. |
| Mixed types | Both operands convert to their join first. Float with float: the wider one. `f16` with `bf16`: `f32`. Integer with float: the float. |
| `T(x)` conversion | `f16(x)`, `bf16(x)`, `f32(x)`, `f64(x)`, `i8(x)`, `i32(x)`, `i64(x)`. Tile or scalar. |
| Float to float | Rounds to nearest. |
| Integer to float | Signed conversion. |
| Float to integer | Truncates toward zero. |

## Bounds and masking

Tensor extents are assumed to be multiples of 4. They need not be multiples of the tile.

| Case | Behavior |
|---|---|
| Slice past the end | Out-of-bounds elements read as `0`; their stores are skipped. |
| Static extent | Masked in place. |
| Dynamic extent walked by a `for` | The loop is split into whole tiles plus one masked remainder iteration. Only spans with a static length split; loops with `mma.sync` accumulators or that are pipelined stay masked unless `@aligned` covers the dimension. |
| Dynamic extent indexed by `program_id` | Masked against the runtime extent, unless `@aligned` promises whole tiles. |
| Non-zero reductions | The fill is always `0`. A max or a softmax denominator over a ragged tile needs its own mask. |

`@aligned(DIM = tile)` is unchecked. A wrong promise writes past the tensor, into the following rows.

## Tiles and shared memory

- A tile is a window of one shared-memory buffer per kernel. Static shared memory is capped at 48 KB.
- A tile lives from its declaration to its last use, across any loops in between. Tiles whose lifetimes do not overlap share bytes.
- A tile declared before one loop and read after another is CTA-private storage carried across both.
- A CTA barrier follows a tile store only when a later op reads or writes those bytes from other threads. Two elementwise passes at the same vector width need none; a reduction, `dot` or `transpose` after one does.

## Built-ins

### Indexing and synchronization

| Built-in | Result | Semantics | Requires |
|---|---|---|---|
| `program_id(d)` | index | Block index along grid axis `d`. | `d` a literal `0`, `1` or `2`. |
| `grid_barrier(bar)` | index `0` | Every block of the grid waits for every other. | `bar` a named `i32` tensor of at least 2 elements, zeroed once before launch and not shared by concurrent kernels. 1-D grid. Every block reaches the call. Grid co-resident, see `@persistent`. |
| `atomic_add(t, i, v)` | `i32` | `old = t[i]; t[i] += v` atomically across the device. | `t` a named `i32` tensor parameter; `v` an integer. |

### Elementwise

| Built-in | Semantics | Notes |
|---|---|---|
| `exp(t)` | `e^x` | Approximate (`ex2.approx`). |
| `log(t)` | `ln x` | Approximate (`lg2.approx`). |
| `sqrt(t)` | `sqrt(x)` | Approximate (`sqrt.approx.f32`). |
| `tanh(t)` | `tanh(x)` | Approximate (`tanh.approx.f32`). |
| `round(t)` | Nearest integer, ties to even | Exact (`cvt.rni`). |
| `tmax(a, b)` | `max(a, b)` | Broadcasts. Same element type. |
| `argsel(va, vb, ia, ib)` | `va >= vb ? ia : ib` | Broadcasts. `va`, `vb` share a float type; `ia`, `ib` share the result type. Fold with `tmax` to build an argmax. Carry indices as floats (exact up to `2^24`). |

### Reductions and shape

All take a rank-2 tile with a static shape.

| Built-in | Shape | Semantics |
|---|---|---|
| `rowmax(t)` | `[R, C] -> [R, 1]` | `max_j t[i, j]`. Float only. |
| `rowsum(t)` | `[R, C] -> [R, 1]` | `sum_j t[i, j]`. Float only. |
| `cumsum(t)` | `[R, C] -> [R, C]` | `out[i, j] = sum_{r <= i} t[r, j]`, down the rows. Float only. |
| `tril(t)` | `[R, C] -> [R, C]` | Keeps `t[i, j]` for `j <= i`, zeros the rest. Float only. |
| `transpose(t)` | `[R, C] -> [C, R]` | `out[i, j] = t[j, i]`. |
| `flat(t)` | `[R, C] -> [1, R * C]` | Row-major view of the same bytes, read-only. `t` must be a declared tile, not a slice. |
| `gather(table, idx)` | `idx.shape` | `out[i] = table[idx[i]]`. `table` is `[n]` or `[1, n]`; result has `table`'s element type. |

### Dense contraction

| Built-in | Shape | Semantics |
|---|---|---|
| `dot(a, b)` | `[M, K] x [K, N] -> [M, N]` | `a @ b` |
| `dot_t(a, b)` | `[M, K] x [N, K] -> [M, N]` | `a @ b^T` |

| Operands | Accumulator | Hardware path |
|---|---|---|
| `f32` | `f32` | Tiled FMA, or tensor cores under `@tensorcore` (operands rounded to f16). |
| `f16` | `f16`, or `f32` under `@tensorcore` | Tensor cores need all tile dims and the k-slice to be multiples of 16. |
| `i8` via `dot_t` | `i32` | `mma.m8n8k16.s8` when the output is whole 8x8 blocks and `K % 16 == 0` (sm_75+); else `dp4a` when `K % 4 == 0` (sm_61+); else scalar. |

### Attention

| Built-in | Semantics |
|---|---|
| `warp_partial(q, K, V, lo, hi, col, WM, WL, WACC, scale)` | Split-key online softmax. Warp `w` takes the `w`-th of `W` equal pieces of keys `[lo, hi)` and, for each query row `i`, computes `s_j = scale * q[i, :] . K[j, col :+ D]`, writing `WM[i, w] = max_j s_j`, `WL[i, w] = sum_j exp(s_j - WM[i, w])`, and `WACC[i * W + w, :] = sum_j exp(s_j - WM[i, w]) * V[j, col :+ D]`. Returns `0`. |
| `delta_scan_t(q, k, v, dec, bet, st, o)` | The gated delta rule over the rows of `q`, the state in registers. Views: `q`, `k` `[N, D]`; `v`, `o` `[N, C]`; `dec`, `bet` `[N, 1]`; `st` `[D, C]`, read at the start and written at the end. Per row `t`: `S = dec_t * S`, `S += k_t^T (bet_t * (v_t - k_t S))`, `o_t = q_t S`. The CTA's warps split the columns evenly, a power of two of at most 8 each; within a warp of `cols` columns, lane `l` owns column `l % cols` and the `cols * D / 32` rows of row group `l / cols`, so `D` must divide into `32 / cols` groups. Reductions over rows shuffle across the row groups only. Every view must be in bounds. Returns `0`. |

Shapes: `q` a named `[QG, D]` tile, unmasked; `K`, `V` named `f16` tensor parameters; `WM`, `WL` named `[QG, W]` tiles and `WACC` a named `[QG * W, D]` tile; with `D % 32 == 0` and `W` = CTA threads / 32. Merging the `W` partials is the caller's.

## Quantized built-ins

Symbols used below: `K` is the contraction length, a multiple of 256 unless stated. `N` is output columns. `qb` is the raw block bytes, `[N, K/256 * bytes]` `i8`. `d` is the per-block scale plane, `[N, K/256]` `f16`. Activation scales `a_scales` are Q8_0: one `f32` per 32 elements.

### Q8_0

| Built-in | Shape | Semantics |
|---|---|---|
| `qdot_t(a, a_scales, w, w_scales)` | `[M, K] i8 x [N, K] i8 -> [M, N] f32` | `out[i, j] = sum_b a_scales[i, b] * w_scales[j, b] * sum_{k in b} a[i, k] * w[j, k]`. Scales `[M, K/32]` and `[N, K/32]`. `K` may be dynamic. |
| `qmma_t(a, a_scales, w, w_scales)` | same | Same sum on integer tensor cores. `M`, `N` multiples of 8. `w_scales` is `[K/32, N]` (transposed relative to `qdot_t`). Statement or value. |
| `q50_qdot_t(a, a_scales, w, w_scales)`, `q50_qmma_t(..)` | `[M, K] i8 x [N, K/32 * 20] i8 -> [M, N] f32` | `qdot_t` and `qmma_t` against a Q5_0 weight held as its blocks without their `f16` scale: per 32 elements, the `qh` word, then 16 bytes holding elements `e` and `e + 16` in the low and high nibbles of byte `e`. Each element is `q - 16`, widened as it is loaded. The scales are as for `qdot_t` and `qmma_t`. |
| `rms_norm_q_t(x, gain, eps, [out,] q, scales)` | row as `[K/32, 32]` | `y = x * gain / sqrt(mean(x^2) + eps)`, written to `out` if given, and quantized to Q8_0 into `q` (`i8`) and `scales` (`[K/32, 1]` `f32`). Returns `1 / rms` as `f32`. Operands must be in-bounds tensor slices (use `@aligned`). |
| `rms_norm_gated_q_t(x, gate, gain, eps, [out,] q, scales)` | row as `[K/32, 32]` | As `rms_norm_q_t`, with the normalized row multiplied by `silu(gate) = gate / (1 + exp(-gate))` before it is written and quantized. `gate` has the shape of `x`. Returns `1 / rms` as `f32`. |

### Raw-format families

`<fmt>` is a format name from the [format table](#formats).

| Built-in | Shape | Semantics | Form |
|---|---|---|---|
| `<fmt>_qdot_t(a, qb, d, tables..)` | `[1, K] f32 x qb -> [1, N] f32` | Matvec, decode folded in: `out[j] = sum_k a[k] * w[j, k]`. `N` a multiple of the CTA's warp count. | value |
| `<fmt>_qdot_i8_t(aq, a_scales, qb, d, tables..)` | `[1, K] i8 x qb -> [1, N] f32` | Matvec against a Q8_0 activation (`a_scales` `[1, K/32]`). `N` a multiple of 8. | value |
| `<fmt>_qgemm_t(a, a_scales, qb, d, tables..)` | `[128, K] i8 x [64, ..] -> [128, 64] f32` | Prompt projection on integer tensor cores. Fixed 128 x 64 tile, needs `@launch(256)`. `a_scales` `[128, K/32]`. | statement or value |
| `<fmt>_qdecode_t(qb, d, tables..)` | `qb -> [K, N] f32 or f16` | Dequantize into a scratch tensor slice. Element type follows the destination. `N` a multiple of the destination tile width. | statement |
| `<fmt>_qmma_staged_t(a, a_scales, qb, d, tables..)` | `[M, K] i8 x qb -> [M, N] f32` | Batched projection on integer tensor cores (sm_75+). `M`, `N` multiples of 8, exactly one patch of 8x8 tiles per warp, the CTA's threads dividing the `N` columns' staging lanes, all operands in bounds. | statement or value |
| `iq1s_qmma_t(a, a_scales, qb, d, grid)` | same | As above, decoded per warp instead of staged once per CTA. | statement or value |

A statement-only built-in is valid only as the whole right-hand side of a tensor-slice assignment: `SCRATCH[:, pn * TN :+ TN] = iq1s_qdecode_t(...)`.

### Formats

| `<fmt>` | Bytes / 256 | Layout | `qdot_t` | `qdot_i8_t` | `qgemm_t` | `qdecode_t` | `qmma_staged_t` |
|---|---|---|---|---|---|---|---|
| `iq1s` | 50 | grouped | yes | yes | yes | yes | yes |
| `iq1m` | 56 | grouped | yes | yes | yes | yes | |
| `iq2xxs` | 66 | grouped | yes | yes | yes | yes | yes |
| `iq2xs` | 74 | grouped | yes | yes | yes | yes | yes |
| `iq2s` | 82 | grouped | yes | yes | yes | yes | yes |
| `iq3xxs` | 98 | grouped | yes | yes | yes | yes | yes |
| `iq3s` | 110 | grouped | yes | yes | yes | yes | yes |
| `iq4xs` | 136 | row-major | yes | | | | |
| `q2k` | 84 | row-major | yes | | | | |
| `q3k` | 110 | row-major | yes | | | | |
| `q4k` | 144 | grouped | | yes | yes | | |
| `q5k` | 176 | grouped | | yes | yes | | |
| `q6k` | 208 | grouped | | yes | yes | | |
| `ptq1` | 56 | grouped | | yes | yes | | |

`q6k` moves each block's trailing `d` into the scale plane on the device (208 of 210 bytes). `ptq1` is two 128-weight file blocks per 256, re-laid at upload. For `q4k`, `q5k` and `ptq1` the scales are inside the block: `d` is taken but not read.

### Table operands

Tables are `[1, n]` `i8` tensors, built on the host by `phobos-gguf`'s `quant` module.

| `<fmt>` | `qdot_t`, `qdecode_t`, `qmma_staged_t` | `qdot_i8_t`, `qgemm_t` |
|---|---|---|
| `iq1s` | `grid` = `iq1s_packed_grid`; `iq1s_signed_grid` for `qmma_staged_t` and `iq1s_qmma_t` | `grid` = `iq1s_grid4` (`qdot_i8_t`), `iq1s_grid2` (`qgemm_t`) |
| `iq1m` | `grid` = `iq1s_packed_grid` | as `iq1s` |
| `iq2xxs` | `grid, signs` = `iq2xxs_packed_grid`, `iq2xxs_packed_signs` | `grid, masks` = `iq2xxs_packed_grid`, `iq2xxs_sign_masks` |
| `iq2xs` | `iq2xs_packed_grid`, `iq2xxs_packed_signs` | packed grid, 0/-1 sign masks |
| `iq2s` | `iq2s_packed_grid`, `iq2s_packed_signs` | `iq2s_packed_grid`, `iq2s_sign_masks` |
| `iq3xxs` | `iq3xxs_packed_grid`, `iq2xxs_packed_signs` | packed grid, 0/-1 sign masks |
| `iq3s` | `iq3s_packed_grid`, `iq2s_packed_signs` | packed grid, 0/-1 sign masks |
| `iq4xs` | `codebook` = `iq4xs_flat_codebook` (`[1, 16]`) | |
| `q2k` | `dmin`, not a table: `[N, K/256]` `f16` | |
| `q3k`, `q4k`, `q5k`, `q6k`, `ptq1` | none | none |

### Grouped layout

For formats marked `grouped`, the device stores columns in groups of 8:

- `qb`, declared `[N, K/256 * bytes]`, is laid out `[N/8][K/256][8][bytes]`.
- `d`, declared `[N, K/256]`, is laid out `[N/8][K/256][8]`.
- A last group short of 8 columns is zero-padded.

The host produces this at upload (`Quant::grouped_rows`). Row-major formats are stored as declared.

## Attributes

| Attribute | Effect |
|---|---|
| `@autotune(X in [a, b], Y in [c, d, e])` | Search space for tile constants. Two values: inclusive bounds, searched in doubling steps (`[16, 256]` gives 16, 32, 64, 128, 256). Three or more: an explicit list. The first choice seeds the shape environment. |
| `@cluster(X in [..], ...)` | Super-tile dimensions and search space for cluster tuning. |
| `@aligned(DIM = tile, ...)` | Promises `DIM` is a whole number of `tile` (a literal or an `@autotune` constant). Removes masking on that dimension and allows 16-byte vector access into its rows. Unchecked. |
| `@launch(threads[, min_blocks[, max_regs]])` | CTA size (default 256), PTX `.maxntid`, `.minnctapersm`, `.maxnreg`. `max_regs` (16..255) is a hard cap; `min_blocks` is advisory. |
| `@tensorcore` | `dot`/`dot_t` and the GEMM run on tensor cores (sm_70+), operands rounded to f16, f32 accumulation. Uses `mma.sync` + `ldmatrix` with swizzled staging (sm_75+, 64-bit index), else WMMA m16n16k16. Falls back to the f32 path silently unless tile dims and the k-slice are multiples of 16 and the CTA's warps tile the 16x16 fragment grid. |
| `@tensorcore(wmma)` | Forces the WMMA m16n16k16 path. |
| `@tensorcore(sync)` | Same as `@tensorcore`. |
| `@pipeline` | Double-buffered staging is on by default for every eligible loop. The attribute asserts it: compilation fails if nothing is pipelined. It also enables double buffering for the GEMM's own operand pairs. |
| `@padstage` | Pads `var t = <tensor slice>` staging tiles whose row pitch is a multiple of 128 bytes, to avoid bank conflicts. Other tiles are unchanged. |
| `@dynshared` | Dynamic shared memory instead of the static 48 KB; the launch must supply the size. The allocator resets once every tile is released, so phases separated by a barrier share one footprint. |
| `@persistent` | The grid must be co-resident (required by `grid_barrier`). The launcher sizes it from the occupancy API (`phobos_kernels::launch::persistent_grid`). |
| Unknown | Parsed and ignored, with a note. |

GEMM tile sizes (`TILE_M`, `TILE_N`, `TILE_K`, `WARP_M`, `WARP_N`, `TILE_TM`, `TILE_TN`) come from `@autotune` or the shape environment.

Planned: `@readonly`, `@unroll` (parameter and loop level), `@fast_math`, `@assert_coalesced`.

## Examples

SGEMM
```plain
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
@launch(256)
@pipeline
kernel matmul(A: tensor<f32>[M, K],
              B: tensor<f32>[K, N],
              C: tensor<f32>[M, N],
              alpha: f32,
              beta: f32) {
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TILE_M, TILE_N] = 0.0
  for kt in range(0, K, TILE_K) {
    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
  }
  let c_old = C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N]
  C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = alpha * acc + beta * c_old
}

```

Flash Attention
```plain
@cluster(BR in [1024, 4096], BC in [1024, 4096])
@autotune(D in [64], BR in [32, 128], BC in [32, 128])
kernel flash_attention(Q: tensor<f32>[Nq, D],
                       K: tensor<f32>[Nk, D],
                       V: tensor<f32>[Nk, D],
                       O: tensor<f32>[Nq, D],
                       scale: f32) {
  let pid = program_id(0)
  let row = pid * BR

  // query tile stays resident; running softmax state for BR rows.
  let q = Q[row :+ BR, :]
  var acc: tile<f32>[BR, D] = 0.0              // unnormalized output sum p*v
  var m: tile<f32>[BR, 1] = -999999999.0       // running row max
  var l: tile<f32>[BR, 1] = 0.0                // running denominator sum p

  for kt in range(0, Nk, BC) {
    let k = K[kt :+ BC, :]
    let v = V[kt :+ BC, :]

    // scaled scores for this tile
    var s: tile<f32>[BR, BC] = dot_t(q, k)     // [BR, BC] = q @ k.T
    s = s * scale

    // online softmax update
    var mnew: tile<f32>[BR, 1] = rowmax(s)
    mnew = tmax(m, mnew)                       // new running max
    var p: tile<f32>[BR, BC] = exp(s - mnew)   // probabilities (broadcast subtraction)
    var corr: tile<f32>[BR, 1] = exp(m - mnew) // rescale factor for old state

    l = l * corr
    l += rowsum(p)

    acc = acc * corr                           // broadcast output
    acc += dot(p, v)                           // add this tile's contribution

    m = mnew
  }

  acc = acc / l                                // normalize (broadcast divide)
  O[row :+ BR, :] = acc
}
```

Gated Linear Attention, chunkwise, using `cumsum`, `tril` and `transpose`. The commented kernel is [`examples/kda_fp32.ph`](./examples/kda_fp32.ph).

```plain
@autotune(D in [64], C in [32, 128])
kernel kda(Q: tensor<f32>[N, D], K: tensor<f32>[N, D], V: tensor<f32>[N, D],
           G: tensor<f32>[N, 1], O: tensor<f32>[N, D], scale: f32) {
  var S: tile<f32>[D, D] = 0.0                 // recurrent state (keys x values)
  for c in range(0, N, C) {
    let q = Q[c :+ C, :]
    let k = K[c :+ C, :]
    let v = V[c :+ C, :]
    let g = G[c :+ C, :]                        // [C, 1] per-token log-gates

    var b: tile<f32>[C, 1] = cumsum(g)          // cumulative in-chunk decay
    var negb = b * -1.0
    var qd: tile<f32>[C, D] = q * exp(b)        // decay-folded queries
    qd = qd * scale
    var kd: tile<f32>[C, D] = k * exp(negb)     // decay-folded keys

    var p: tile<f32>[C, C] = dot_t(qd, kd)      // intra-chunk scores
    p = tril(p)                                 // causal mask
    var o: tile<f32>[C, D] = dot(p, v)
    o += dot(qd, S)                             // inter-chunk (carried state)
    O[c :+ C, :] = o

    var gt = transpose(g)
    var total: tile<f32>[1, 1] = rowsum(gt)     // chunk-total decay
    var kfin = k * exp(total - b)
    var kt = transpose(kfin)                    // [D, C]
    var kv: tile<f32>[D, D] = dot(kt, v)        // sum_j kfin_j^T v_j
    S = S * exp(total) + kv                     // decay state, add K^T V
  }
}
```
