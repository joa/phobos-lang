# Grammar (EBNF)

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

## Notes
- **Keywords:** `kernel let var if else for in while true false`.
- **Tensor size**: Assumed to be a multiple of `4`. A tensor dimension need not
  be a multiple of the tile: a slice that would run off the end is masked, so
  the out-of-bounds elements read as zero and their stores are skipped.
  - A compile-time constant size is masked in place, which falls back from the
    specialized register/tensor-core matmul paths to the generic tiled path.
  - A dynamic (runtime) size walked by a **loop** is handled by splitting that
    loop: it is trimmed to `(extent / tile) * tile`, so its slices are whole by
    construction and keep the vectorized, tensor-core and `cp.async` fast
    paths, and the ragged remainder replays the body once under a runtime mask
    against the tensor's own extent. Only spans with a static length split;
    loops carrying `mma.sync` fragment accumulators, or that the compiler
    pipelines (see `@pipeline` below), do not split yet, so their slices are
    masked unless `@aligned` covers the dimension they walk.
  - A dynamic size indexed by a **program id** cannot be trimmed the same way:
    the grid is the host's, and nothing inside the kernel bounds it. Such a
    slice is masked against the tensor's runtime extent, which costs the
    specialized paths, since their drains have no per-element store guard.
    `@aligned(DIM = tile)` is how a caller that knows better says so: it
    promises the extent is a whole number of tiles, so every program id
    addresses a whole tile and the mask drops. The promise is unchecked, and
    breaking it writes past the tensor -- through the end of one row and into
    the next, not merely off the end. A caller that cannot guarantee the shape
    should leave it off, or compile the kernel both ways and pick per launch.
  - Zero is the fill value, so a reduction that is not zero-identity (a softmax
    denominator, for instance) still needs its own masking in the kernel.
- **`f16`**: half precision (IEEE binary16). Float literals are written in f32 and
  rounded to f16 on store, and arithmetic that mixes f16 with a wider float widens
  to the wider type (so `f16 + f32 -> f32`). The heavy compute paths run f16 inputs
  through the tensor cores with f32 accumulation (see `@tensorcore`); without it,
  f16 tiles use the generic element/vector paths and accumulate in f16.
- **`bf16`**: brain float. Same width as `f16` but with f32's exponent range and 8
  fewer mantissa bits, so **neither 16-bit float contains the other**: mixing them
  widens to `f32` rather than picking a side, and a direct `f16`/`bf16` conversion
  round-trips through f32. The type is available on every target; what changes with
  the target is the instruction count. From sm_80 a conversion is one
  `cvt.rn.bf16.f32`, and below it the NVPTX backend emulates it with the
  shift-and-round-to-nearest-even integer sequence. Nothing in phobos gates on
  this, so a bf16 kernel compiles and runs everywhere; `GpuConfig::supports_bf16_native`
  reports which of the two a target gets.
- **`i8`**: signed byte, the quantized-weight element type. Loads sign-extend
  (`ld.global.s8`) and `f32(w)` converts, so a dequantizing weight load costs a
  quarter of the memory traffic of the same weights in f32. Integers are signed
  throughout; there is no `u8` yet.
- **Integer contraction**: `dot`/`dot_t` over `i8` operands accumulate in `i32`,
  not in the operand type, since a dot product of bytes overflows a byte almost
  at once. Both hardware paths hang off `dot_t` rather than `dot`, because both
  want the bytes of each operand contiguous, and `dot_t` contracts the last axis
  of both operands so both walk memory that way. From widest to narrowest:
  - the **integer tensor cores** (`mma.sync.m8n8k16.s8.s8.s32`) when the target
    has them (Turing onwards, `GpuConfig::supports_int8_mma`), the output tile is
    a whole number of 8x8 blocks, and the contraction is a multiple of 16. This
    needs no staging buffer and no `ldmatrix`: the fragment layout is already
    what `dot_t` holds, a lane reading four contiguous bytes of one row per
    operand. One warp owns each 8x8 output tile.
  - the **four-way byte dot product** (`dp4a`, one instruction for four
    multiplies and four adds) on Pascal onwards, `GpuConfig::supports_dp4a`,
    when the contraction is a multiple of 4. This is where a single-row
    contraction lands, since one row cannot fill an 8-row tensor-core tile.
  - the **generic integer path**, with the same result, for anything else: a
    ragged contraction, a pre-Pascal target, or a masked operand.
- **Grouped raw layout**: the seven IQ formats (`iq1s`, `iq1m`, `iq2xxs`, `iq2xs`, `iq2s`, `iq3xxs`, `iq3s`) and the three K-quants (`q4k`, `q5k`, `q6k`) are held on the device grouped by eight columns: the payload `qb`, declared `[N, K/256*bytes]`, is laid out `[N/8][K/256][8][bytes]`, eight columns' copies of each block side by side, and the scale plane `d`, declared `[N, K/256]`, likewise `[N/8][K/256][8]`; a last group short of eight columns is zero-padded. Every built-in that reads these formats addresses them so, through one helper, and the host lays them out at upload (`Quant::grouped_rows`). The point is the decode matvec: a warp reading eight columns' blocks touches three or four whole lines rather than eight scattered sectors, and the miss-tracking slots that bound its bandwidth go three times as far. The other formats stay row-major.
- **Quantized contraction**: `qdot_t` is the Q8_0 contraction with the block
  scales folded in, so the whole of `k` is one operation. `dot` and `dot_t`
  cannot be given enough of `k` at a time here: a Q8_0 block carries its own
  scale, so a plain dot has to stop every 32 elements to apply it, and with one
  thread per output walking `k`, a warp reads 32 rows four bytes apart and the
  block pays several barriers per 32 elements. Folding the scales in is what
  lets the mapping turn around: a **warp owns one output and its lanes divide
  `k`**, so 32 lanes read 512 contiguous bytes of one weight row, nothing is
  staged, and the only synchronization is the closing butterfly shuffle. Each
  lane takes 16 bytes, four `dp4a` under one scale pair, which is the widest
  chunk that still sits inside a single block.

  `qmma_t` is the same contraction batched over rows, on the integer tensor
  cores, and it exists for the same reason at the other end: writing it in the
  tile language puts the accumulator in shared memory, so a `[64, 64]` tile is
  16 KB of accumulator alone and cannot be built at all. Folding the scales in
  keeps the accumulators in registers across the whole of `k`, reads both
  operands straight from global memory in the layout the `m8n8k16` fragments
  already want, and leaves no barrier in the loop. A warp takes a square patch
  of tensor-core tiles, since `rm` by `rn` tiles issue `2 * rm * rn` tensor
  instructions against `2 * (rm + rn)` operand loads.
- **K-quant contraction**: `q4k`, `q5k` and `q6k` join `<fmt>_qgemm_t` and
  `<fmt>_qdot_i8_t` with no tables: a quant is a nibble plus, for Q5_K, one
  bit of the block's 32-byte `qh` plane, or, for Q6_K, two bits of its 64-byte
  one, and the scales are in the block, six-bit indices packed twelve bytes for
  Q4_K and Q5_K (both branches of the unpacking computed and one selected on
  the run) and sixteen signed bytes for Q6_K. Six-bit quants are the same
  bytes signed or unsigned, so they ride the signed `dp4a` and `mma` as they
  are, and a run's dot product stays under 2^22, within the exact
  integer-to-float conversion. Q4_K and Q5_K subtract a minimum,
  `d * sc * q - dmin * m`, so against an activation `sa * aq` the contraction
  is `sa * (d * sc * A - dmin * m * S)` with `S` the activation's sum over the
  run: `<fmt>_qgemm_t` sums each row's 32-element group at stage time (the two
  threads holding a group's halves join theirs with one xor shuffle) into an
  `[128, K/128*4]` i32 plane and keeps `dmin * m` a column beside the scales;
  `<fmt>_qdot_i8_t` sums each run in the lane that contracts it, eight `dp4a`
  against ones over the activation bytes it holds for the dot. Both read `d`
  and `dmin` out of the block header for Q4_K and Q5_K, so the `d` operand is
  taken and never read; Q6_K's `d` trails its block on disk, is dropped from
  the device block (208 bytes of 210), and is read from the plane. Q6_K has no
  minimum: its `q - 32` folds into the byte before the `dp4a` without a carry,
  as `ql | (((qh2 + 0x0E0E0E0E) & 0x0F0F0F0F) << 4)`, and its scales are per
  sixteen elements, so it takes the split epilogue. A Q6_K decode lane is not
  64 contiguous elements: the block is interleaved across two 128-element
  groups, so a lane takes sixteen elements of each of a group's four quarters,
  four runs of sixteen that are each one Q6_K scale run, with its activations
  four sixteen-byte loads 32 apart.
- **Grid-decode contraction**: `iq1s_qdot_t` is IQ1_S's matvec contraction with
  its grid-table decode folded in, the same **warp owns one output, register
  accumulator, one closing shuffle** shape `qdot_t` uses, for the same reason:
  ordinary `let`/`gather`/`dot_t` codegen stages every intermediate through
  shared memory with a barrier, and an IQ1_S block's per-lane decode has many
  intermediates (a byte read, a grid index, a table lookup, a sign correction)
  where Q8_0's bytes are already values. Unlike `qdot_t`, no hardware
  instruction backs this one; the win is deleting the staging and barriers,
  not a wider memory transaction. It works only because an IQ1_S block's
  thirty-two decode lanes are exactly a warp's width: warp lane `l` decodes
  format lane `l` outright, with no remainder to fold in.

  `iq2xxs_qdot_t` is the same shape for IQ2_XXS, which needs two table
  lookups a lane (a magnitude grid, a sign table) instead of one and has a
  genuine branch its geometry cannot avoid (lane `l % 4 == 0` reads one raw
  byte where the other three lanes read two and combine them). Folded
  branch-free with `arith.select` rather than control flow, since a warp's
  lanes already diverge in cost by reading different addresses and a real
  branch would only add a reconvergence point.

  `iq1m_qdot_t` shares IQ1_S's grid outright but has its own per-group
  scale-pair geometry, needing more `arith.select`s than either of the
  other two (`iq2xxs_qdot_t`'s one branch versus `iq1m_qdot_t`'s several).

  `iq2s_qdot_t` is IQ2_XXS's two-gather shape again with plainer fields:
  both offsets collapse the same branch-free way `iq1s_qdot_t`'s do, the
  divisor is a plain runtime shift with no `l == 0` special case, and the
  sign table is keyed by the raw byte outright rather than a parity index,
  so only the scale nibble needs a select.

  `iq2xs_qdot_t` is `iq2s_qdot_t`'s same shape and same one select, splitting
  a single 16-bit `qs` halfword by `% 512` / `/ 512` into the magnitude and
  sign indices instead of reading them from separate byte fields, and reuses
  IQ2_XXS's sign table outright rather than uploading its own.

  `iq3xxs_qdot_t` reuses `iq2xxs_qdot_t`'s scale/sign geometry outright (the
  two share an identical `aux32` field), but its magnitude grid holds
  four-byte entries where IQ2_XXS's hold eight, so a lane's eight elements
  come from two grid entries instead of one: two four-wide reductions
  against their own slice of `a` and `signs`, both adding into the same
  per-lane partial before the closing shuffle, rather than one eight-wide
  reduction.

  `iq3s_qdot_t` is `iq3xxs_qdot_t`'s same two-four-wide-entry shape, but with
  `iq2s_qdot_t`'s direct-byte sign table (so no `lo`/`hi` assembly at all: a
  single `qh` byte a half, not a two-byte word) and its scale nibble select,
  gated on the half rather than the lane's low bits.

  `iq4xs_qdot_t` is not grid-coded, and its warp mapping is structurally
  different from the rest: IQ4_XS decodes eight 32-element runs a block
  rather than thirty-two 8-element lanes, and a run is exactly a warp's
  width, so lane `y` is a run's own element position, not a format lane. A
  run's scale depends only on the run and the output column, not the lane,
  so it is computed once and read identically by all 32 lanes; only the
  16-entry codebook index (which nibble of `qs` lane `y` needs) varies by
  lane. The eight runs are a Rust-unrolled loop inside the same per-block
  body every other format's intrinsic has one decode step in, each run's
  product adding into the same per-lane partial before the closing shuffle.

  `q2k_qdot_t` and `q3k_qdot_t` need no table lookup at all -- Q2_K and
  Q3_K decode from static offsets -- but pay the same staging cost the
  grid-coded formats' old bodies did, since that cost comes from the tile
  language's execution model, not from `gather`. Their block shape is
  sixteen runs of sixteen elements rather than thirty-two of eight, so
  `RUN * 2 == WARP`: two runs process a warp-pass, lane `l < 16` on the
  first and `l >= 16` on the second, and the local k-offset collapses to
  the raw lane id plus a per-iteration constant outright. Q3_K's six-bit
  signed scale needs two `arith.select`s Q2_K's packed scale/min byte does
  not, for a genuine conditional in its cross-byte unpacking.
- **Packed tables**: every `<fmt>_qdot_t` and `<fmt>_qdecode_t` reads its
  lookup tables as one `i8` a slot, not one `i32`. A lane's whole entry is then
  eight contiguous bytes (four for IQ3_XXS and IQ3_S, whose grid entries are
  half as wide), so it takes one `vector.load` where the widened layout took
  one scalar load an element: eight table loads a lane become one, and the
  table is a quarter the size, which is what gets it into L1. The values fit
  exactly -- IQ1_S's grid holds `{-1, 0, 1}`, the IQ2/IQ3 magnitudes top out at
  62, and the sign multipliers are `+-1` -- and they are signed, so they widen
  with `arith.extsi`, not the zero-extend the unsigned block fields take.

  The `i32` tables stay uploaded beside the packed ones for the fallback
  `<fmt>_matvec`/`<fmt>_dequant` kernels, which reach them through `gather` a
  slot at a time. Two copies of every table is a few KB.
- **Vector activation reads**: in the `<fmt>_qdot_t` contractions whose lane
  owns a contiguous run of the activation row, the lane reads that run as
  `vector<4xf32>` rather than one scalar load an element. A warp's 32 lanes are
  then 1024 contiguous bytes covered by two instructions, where the scalar form
  issued eight, each spreading the warp across 32 separate sectors because
  neighbouring lanes sit eight elements apart.

  This does not apply to `iq4xs_qdot_t`, `q2k_qdot_t` or `q3k_qdot_t`: their
  lane index *is* an element position within a run, so consecutive lanes
  already read consecutive activations and the scalar form is coalesced.
- **Grid-decode expansion**: `<fmt>_qdecode_t` runs one of those same decodes
  and stores the weights into a `[K, N]` scratch instead of contracting them,
  which is what a prompt pass needs: a batched matmul reads a dequantized
  strip once for all its rows, where the matvec above reads a row of activation.
  Only a statement, never a value -- `SCRATCH[:, pn * TN :+ TN] =
  iq1s_qdecode_t(..)` -- because it writes the destination itself, the way
  `qmma_t` writes its accumulators into an output slice.

  The decode is shared with the matching `<fmt>_qdot_t` outright, so what
  differs is only the thread map, and the store sets it. `qdot_t` gives a warp
  one output column, making its lanes the format's 32 lanes. Here consecutive
  threads take consecutive output *columns* of the same rows, which puts a
  warp's 32 stores in four fully covered 32-byte sectors of the scratch, and a
  thread still owns eight elements of one column so the block bytes are still
  read once per eight decodes. Nothing is staged in shared memory and there is
  no barrier at all.
- **Mixed element types**: a binary op whose operands differ converts both to their
  join before computing. Between floats the join is the wider type, except that
  `f16` and `bf16` join at `f32`; an integer meeting a float joins at the float.
  Matching types keep the vectorized path, and a converting op falls back to the
  scalar element path.
- **Conversions**: every numeric scalar type names a conversion builtin that takes
  a tile or a scalar: `f16(x)`, `bf16(x)`, `f32(x)`, `f64(x)`, `i8(x)`, `i32(x)`,
  `i64(x)`. Float-to-float rounds, integer-to-float sign-converts, float-to-integer
  truncates toward zero. There is no `bool(x)`.
- **Tile buffers are pooled by liveness**: a tile lives in shared memory, and the
  buffer of a tile a block declares goes back to the pool after the last
  statement of that block mentioning its name, so a later declaration reuses it.
  A nested body counts as part of the statement containing it, so a name read
  inside a loop stays live until after the loop. This is what keeps a long chain
  of named intermediates from costing a static allocation each: static shared
  memory is capped at 48 KB on every architecture.

  Since a tile outlives the loops between its declaration and its last use, a
  tile declared ahead of one loop and read after another is **block-private
  storage carried across both**, and the barrier that trails every tile store is
  what makes one thread's write visible to the rest of the CTA.
- **`var` without an initializer**: `var t: tile<f32>[R, C]` declares a buffer and
  leaves it alone, for one a following loop overwrites element for element. The
  type is required, nothing being left to infer one from, and it must be a tile:
  a scalar with no value would name nothing. `let` always needs its initializer.
- **Statements end at newlines**: Similar to how Golang is doing it
- **`else` must follow `}` on the same line**: a newline after the `}` of the then-block ends the `if` statement.
- Tiles
  - **Slicing**: a tile is sliced and slice-assigned exactly as a tensor is, so a
    loop can fill one a few rows at a time rather than only as a whole value. The
    compiler's own staging buffers are the exception: the WMMA path's padded tile
    and the `ldmatrix` path's XOR-swizzled one hold their rows somewhere other
    than row-major says, and neither is reachable from source anyway.
  - **Ranges**:`A[start : end]` is the elements from `start` up to but not including `end`.
  - **Spans**: `A[start :+ length]` is `length` elements starting at `start`; same as `A[start : start + length]`.
  - **Full**: `A[:]` selects the entire dimension.
  - **Open-Ended**: `A[i:]`, `A[:j]` are not supported. A `:` after an expression requires an end, and a `:+` requires a length.
- **Unary minus on a tile**: `-t` negates elementwise, lowering as `0 - t`. `!` stays scalar-only.
- **Broadcasting**: binary tile ops broadcast a NumPy-style axis of extent 1 (so `[R, C] x [R, 1]` stretches the column vector), and `tile x scalar` (either order) broadcasts the scalar over the tile.
- **Contextual Identifiers:** `tensor`, `tile`, `range`, `program_id`, the tile builtins (`dot`, `dot_t`, `qdot_t`, `qmma_t`, `iq1s_qdot_t`, `iq2xxs_qdot_t`, `iq1m_qdot_t`, `iq2s_qdot_t`, `iq2xs_qdot_t`, `iq3xxs_qdot_t`, `iq3s_qdot_t`, `iq4xs_qdot_t`, `q2k_qdot_t`, `q3k_qdot_t`, `iq1s_qdecode_t`, `iq2xxs_qdecode_t`, `iq1m_qdecode_t`, `iq2s_qdecode_t`, `iq2xs_qdecode_t`, `iq3xxs_qdecode_t`, `iq3s_qdecode_t`, `iq1s_qgemm_t`, `iq1m_qgemm_t`, `iq2xxs_qgemm_t`, `iq2xs_qgemm_t`, `iq2s_qgemm_t`, `iq3xxs_qgemm_t`, `iq3s_qgemm_t`, `q4k_qgemm_t`, `q5k_qgemm_t`, `q6k_qgemm_t`, `iq1s_qdot_i8_t`, `iq1m_qdot_i8_t`, `iq2xxs_qdot_i8_t`, `iq2xs_qdot_i8_t`, `iq2s_qdot_i8_t`, `iq3xxs_qdot_i8_t`, `iq3s_qdot_i8_t`, `q4k_qdot_i8_t`, `q5k_qdot_i8_t`, `q6k_qdot_i8_t`, `rms_norm_q_t`, `exp`, `log`, `round`, `sqrt`, `tanh`, `rowmax`, `rowsum`, `tmax`, `argsel`, `cumsum`, `tril`, `transpose`, `flat`), the
  synchronization builtins (`grid_barrier`, `atomic_add`) and the
  conversion builtins named after the scalar types are ordinary
  identifiers, not keywords. `range` is recognized positionally inside `for ... in range(...)`.
- **Built-Ins**:
  - `dot(a, b)`: `a @ b` (contracts `a`'s last dim with `b`'s first).
  - `dot_t(a, b)`: `a @ b.t` (contracting the last dim of both: `[M, K] x [N, K] -> [M, N]`). Over `i8` operands this is the integer tensor-core and `dp4a` path; see **Integer contraction**.
  - `qdot_t(a, a_scales, w, w_scales)`: the Q8_0 contraction with its block scales, `[M, K] i8 x [N, K] i8 -> [M, N] f32`, where the scales are `[M, K/32]` and `[N, K/32]` f32 and element `[i, j]` is `sum_b (sum_{k in block b} a[i, k] * w[j, k]) * a_scales[i, b] * w_scales[j, b]`. The contraction axis may be dynamic, since it is not tiled. The scales are indexed `[row, block]` so a lane's scale load is contiguous with its neighbours'. See **Quantized contraction**.
  - `qmma_t(a, a_scales, w, w_scales)`: the same contraction batched over rows, `[M, K] i8 x [N, K] i8 -> [M, N] f32` with `M` and `N` multiples of 8, where `a_scales` is `[M, K/32]` and `w_scales` is `[K/32, N]`. The weight scales are indexed `[block, out]`, the opposite of `qdot_t`: a lane here holds two neighbouring output columns of one block, so that order puts its two scales next to each other. See **Quantized contraction**.
  - `<fmt>_qgemm_t(a, a_scales, qb, d, tables..)`: a raw format's prompt projection with both operands staged through shared memory, `[128, K] i8 x [64, K/256*bytes] i8 -> [128, 64] f32`, for `iq1s`, `iq1m`, `iq2xxs`, `iq2xs`, `iq2s`, `iq3xxs` and `iq3s`, and for the K-quants `q4k`, `q5k` and `q6k`, which take no tables (see **K-quant contraction**). The tile is fixed at 128 x 64 and the kernel at `@launch(256)`, eight warps; `K` is a whole number of 256-element blocks and `a_scales` is `[128, K/32]`. The tables are the format's packed magnitude grid and, where it carries signs apart, its 0/-1 sign masks; the two ternary formats take the grid at two bits a lane instead. Per 128 elements of `k` the CTA copies the activation tile in with 16-byte loads, decodes its 64 columns into an int8 tile, and contracts both with `ldmatrix` and `mma.m8n8k16`; nothing in the k loop reads global memory, and the next tile's bytes are fetched before the current one is contracted. Only a statement, like `qmma_t`. See **Quantized contraction**.
  - `<fmt>_qdot_i8_t(aq, a_scales, qb, d, tables..)`: a raw format's single-row contraction against an activation already quantized to Q8_0, in `dp4a`: `[1, K] i8 x [N, K/256*bytes] i8 -> [1, N] f32`, for the ten formats `<fmt>_qgemm_t` lists, with `a_scales` `[1, K/32]` f32, `d` `[N, K/256]` f16 and `K` a whole number of 256-element blocks (see **K-quant contraction** for the two formats with a minimum). The tables are the format's packed magnitude grid and, where it carries signs apart, its 0/-1 sign masks; the two ternary formats take the grid at a nibble a lane (`[1, 8192]`, `crate::quant::iq1s_grid4`), since that entry is already the selector a byte permute takes. Four lanes decode a column, a quarter of each block apiece, and a warp covers eight columns a block, so `N` is a whole number of eights and the tile a whole number of warps' worth; the block bytes ride a two-deep register pipeline and the 64 activations a lane needs come from L1 at decode time. The weight and its scale plane are read in the grouped layout below. See **Grid-decode contraction**.
  - `rms_norm_q_t(x, gain, eps, out, q, scales)`: one row's RMS normalization with its gain, written to `out`, and its Q8_0 copy, `q` int8 and `scales` f32 one a 32-element block, as a single statement over `[K/32, 32]` views of the row; `scales` is `[K/32, 1]`. A thread owns four contiguous elements of every `4 * threads`, so the row must be a whole number of those, and the reductions are warp shuffles rather than tile passes. The operands must be in-bounds slices of tensors, which `@aligned` promises; a masked slice would be staged and the stores would land in the copy.
  - `iq1s_qdot_t(a, qb, d, grid)`: IQ1_S's single-row matvec contraction with its grid-table decode folded in, `[1, K] f32 x [N, K/256*50] i8 -> [1, N] f32`, where `K` is a multiple of 256, `d` is `[N, K/256]` f16 (one block scale a row) and `grid` is the packed one-`i8`-per-slot grid (`[1, 2048*8]`, `crate::quant::iq1s_packed_grid`), whose eight bytes for a lane are one vector load. `N` must be a multiple of the CTA's warp count, one warp an output column. See **Grid-decode contraction**.
  - `iq2xxs_qdot_t(a, qb, d, grid, signs)`: IQ2_XXS's single-row matvec contraction with its magnitude-grid and sign-table decode folded in, `[1, K] f32 x [N, K/256*66] i8 -> [1, N] f32`, where `K` is a multiple of 256, `d` is `[N, K/256]` f16 and `grid`/`signs` are the packed one-`i8`-per-slot tables (`crate::quant::iq2xxs_packed_grid`/`iq2xxs_packed_signs`). Same `N` requirement as `iq1s_qdot_t`. See **Grid-decode contraction**.
  - `iq1m_qdot_t(a, qb, d, grid)`: IQ1_M's single-row matvec contraction with the decode folded in, `[1, K] f32 x [N, K/256*56] i8 -> [1, N] f32`, where `K` is a multiple of 256, `d` is `[N, K/256]` f16 and `grid` is IQ1_S's own packed grid (`crate::quant::iq1s_packed_grid`, shared outright). Same `N` requirement as `iq1s_qdot_t`. See **Grid-decode contraction**.
  - `iq2s_qdot_t(a, qb, d, grid, signs)`: IQ2_S's single-row matvec contraction with its magnitude-grid and sign-table decode folded in, `[1, K] f32 x [N, K/256*82] i8 -> [1, N] f32`, where `K` is a multiple of 256, `d` is `[N, K/256]` f16 and `grid`/`signs` are the packed tables (`crate::quant::iq2s_packed_grid`/`iq2s_packed_signs`). Same `N` requirement as `iq1s_qdot_t`. See **Grid-decode contraction**.
  - `iq2xs_qdot_t(a, qb, d, grid, signs)`: IQ2_XS's single-row matvec contraction with its magnitude-grid and sign-table decode folded in, `[1, K] f32 x [N, K/256*74] i8 -> [1, N] f32`, where `K` is a multiple of 256, `d` is `[N, K/256]` f16, `grid` is `crate::quant::iq2xs_packed_grid` and `signs` is IQ2_XXS's own packed sign table (shared outright). Same `N` requirement as `iq1s_qdot_t`. See **Grid-decode contraction**.
  - `iq3xxs_qdot_t(a, qb, d, grid, signs)`: IQ3_XXS's single-row matvec contraction with its two four-wide grid lookups and sign-table decode folded in, `[1, K] f32 x [N, K/256*98] i8 -> [1, N] f32`, where `K` is a multiple of 256, `d` is `[N, K/256]` f16, `grid` is `crate::quant::iq3xxs_packed_grid` (four `i8` lanes an entry, not eight) and `signs` is IQ2_XXS's own packed sign table (shared outright). Same `N` requirement as `iq1s_qdot_t`. See **Grid-decode contraction**.
  - `iq3s_qdot_t(a, qb, d, grid, signs)`: IQ3_S's single-row matvec contraction with its two four-wide grid lookups and sign-table decode folded in, `[1, K] f32 x [N, K/256*110] i8 -> [1, N] f32`, where `K` is a multiple of 256, `d` is `[N, K/256]` f16, `grid` is `crate::quant::iq3s_packed_grid` (four `i8` lanes an entry) and `signs` is IQ2_S's own packed sign table (shared outright). Same `N` requirement as `iq1s_qdot_t`. See **Grid-decode contraction**.
  - `iq4xs_qdot_t(a, qb, d, codebook)`: IQ4_XS's single-row matvec contraction with its fixed 16-entry codebook decode folded in, `[1, K] f32 x [N, K/256*136] i8 -> [1, N] f32`, where `K` is a multiple of 256, `d` is `[N, K/256]` f16 and `codebook` is the flattened codebook `gather` already uses (`[1, 16]`, `crate::quant::iq4xs_flat_codebook`). Same `N` requirement as `iq1s_qdot_t`. See **Grid-decode contraction**.
  - `q2k_qdot_t(a, qb, d, dmin)`: Q2_K's single-row matvec contraction with its static-offset decode folded in, `[1, K] f32 x [N, K/256*84] i8 -> [1, N] f32`, where `K` is a multiple of 256 and `d`/`dmin` are each `[N, K/256]` f16. `N` must be a multiple of the CTA's warp count, one warp an output column. See **Grid-decode contraction**.
  - `q3k_qdot_t(a, qb, d)`: Q3_K's single-row matvec contraction with its static-offset decode folded in, `[1, K] f32 x [N, K/256*110] i8 -> [1, N] f32`, where `K` is a multiple of 256 and `d` is `[N, K/256]` f16 (no minimum term). Same `N` requirement as `q2k_qdot_t`. See **Grid-decode contraction**.
  - `iq1s_qdecode_t(qb, d, grid)`: IQ1_S's decode written into a dequantized weight strip rather than contracted, `[N, K/256*50] i8 -> [K, N] f32` or `[K, N] f16`, with `d` and the packed `grid` exactly as `iq1s_qdot_t` takes them. The destination's element type picks the width: f16 halves the traffic to the matmul that reads the strip, and the tensor-core ladder truncates a weight operand to f16 before the WMMA regardless, so it costs no accuracy there. Only valid as the whole right-hand side of an assignment to a tensor slice, which it writes itself; there is no value form. `K` must be a multiple of 256 and `N` a multiple of the destination's tile width. See **Grid-decode expansion**.
  - `iq2xxs_qdecode_t(qb, d, grid, signs)`: the same for IQ2_XXS, `[N, K/256*66] i8 -> [K, N] f32`, with `d`, `grid` and `signs` exactly as `iq2xxs_qdot_t` takes them. See **Grid-decode expansion**.
  - `iq1m_qdecode_t(qb, d, grid)`, `iq2s_qdecode_t(qb, d, grid, signs)`, `iq2xs_qdecode_t(qb, d, grid, signs)`, `iq3xxs_qdecode_t(qb, d, grid, signs)`, `iq3s_qdecode_t(qb, d, grid, signs)`: the same for the remaining formats whose lane is eight elements of a 256-element block, each taking exactly the operands its `<fmt>_qdot_t` takes and producing `[K, N] f32`. IQ3_XXS and IQ3_S read their eight elements four at a time from two grid entries, which changes nothing outside the intrinsic. Q2_K and IQ4_XS have no expansion: their lane geometry is not this one (sixteen runs of sixteen, and eight runs of thirty-two), and a dense pass spends a fifth of a percent of itself in them. See **Grid-decode expansion**.
  - `exp(t)`: element-wise `e^x` (lowers to the hardware `ex2.approx`).
  - `log(t)`: element-wise natural logarithm (lowers to the hardware `lg2.approx`, with the change of base folded in).
  - `round(t)`: element-wise nearest integer, ties to even (lowers to the hardware `cvt.rni.f32.f32`). Rounding by biasing into a positive range and truncating instead costs the low mantissa bits, which is enough to move a value across a boundary at the top of a quantization range.
  - `sqrt(t)`: element-wise square root (lowers to the hardware `sqrt.approx.f32`).
  - `tanh(t)`: element-wise hyperbolic tangent (lowers to the hardware `tanh.approx.f32`).
  - `rowmax(t)` / `rowsum(t)`: reduce a rank-2 tile over its last (column) dim to a `[rows, 1]` column vector.
  - `tmax(a, b)`: elementwise maximum (broadcasting).
  - `argsel(va, vb, ia, ib)`: elementwise `select(va >= vb, ia, ib)` (broadcasting): whichever of two indexed candidates carries the winning value, `>=` so a caller that always passes the more-recent candidate as `(va, ia)` gets a reproducible winner on an exact tie. `va`/`vb` share a float element type and `ia`/`ib` share the result's; there is no reduction primitive of its own (no `rowargmax`), so a caller folds it alongside `tmax` across a loop or a halving tree to build one, carrying an index as a tile of the same float type as the value it accompanies (safe up to 2^24, i.e. any array the memory to hold fits well under).
  - `cumsum(t)`: inclusive prefix sum of a rank-2 tile down its first (row) dim, so `out[i, j] = sum_{r <= i} t[r, j]`. The scan runs along the sequence axis (the leading dim of a `[seq, feat]` tile), producing the running gate cumulant that chunkwise linear attention needs. Same shape as the input.
  - `tril(t)`: causal lower-triangular mask of a rank-2 tile, keeping `t[i, j]` when `j <= i` and zeroing the strict upper triangle. Same shape as the input.
  - `transpose(t)`: rank-2 tile transpose, `out[i, j] = t[j, i]` (a `[R, C]` tile becomes `[C, R]`). Lets a contraction run over the leading (sequence) axis, which `dot`/`dot_t` cannot reach on their own.
  - `flat(t)`: a declared rank-2 tile viewed as one row, `[1, R * C]`. A **view**, not a copy: a tile is contiguous in shared memory, so the row-major flattening is the same bytes under another type, read-only and allocating nothing. It is for a value whose computed and consumed shapes differ, as quantizing an activation is: `rowmax` reduces the last axis, so the blocks of 32 have to be rows, and the contraction that follows wants one row. Only a tile declared with `var` or `let ... : tile<..>` can be flattened, not a slice of one (whose offset the view would drop) and not a tensor slice (which is not in shared memory).
  - `f16(x)` / `bf16(x)` / `f32(x)` / `f64(x)` / `i8(x)` / `i32(x)` / `i64(x)`: element type conversion of a tile or a scalar (see **Conversions** above).
  - `grid_barrier(bar)`: every block of the grid waits for every other, so a kernel spanning several stages of a pass can order one against the next. `bar` is an `i32` tensor of at least two elements, slot 0 the arrival counter and slot 1 the release generation (`tensor<i32>[2]` and the rank-2 `tensor<i32>[2, 1]` column both spell it). The caller zeroes it once before the launch and leaves it alone while the kernel runs; the barrier restores both slots, so one pair serves every barrier of every launch. Two things are the caller's to guarantee and neither is checked: that the grid is co-resident, which is what `@persistent` is for, and that no two concurrent kernels share the tensor. The grid must be one-dimensional, since the arrival count comes from `gridDim.x`, and every block has to reach the call, so it must not sit under a branch that only some of them take. See `phobos-lang/src/codegen/sync.rs`.
  - `atomic_add(t, i, v)`: adds `v` to `t[i]` atomically across the device and returns the previous value. `t` is a named `i32` tensor parameter: a tile is block-private and a slice carries an offset the atomic would have to fold in, so neither is accepted.
- **Attributes**:
  - `@autotune(X in [..], ...)`: local search space; the first choice seeds the shape env. Two values are inclusive bounds searched in doubling steps (`X in [16, 256]` -> 16, 32, 64, 128, 256); three or more are an explicit list of choices. `[256, 16]` is two values (when x > y)
  - `@cluster(X in [..], ...)`: super tile dimensions and search space for cluster tuning.
  - `@aligned(DIM = tile, ...)`: promises that a symbolic tensor dimension is a whole number of `tile` elements, where `tile` is an integer or an `@autotune` constant. It also sets how far into a row a vector access may reach: the assumed multiple of 4 is 16 bytes of `f32` but only 8 of a 16-bit type, so a narrow tensor stages at half width unless a promise carries its row pitch to 16 bytes. 
  - `@launch(maxThreads[, minBlocks[, maxRegs]])`: specifies CTA thread assumption (default: 256); maps to PTX `.maxntid` / `.minnctapersm` / `.maxnreg` at codegen. `maxRegs` (16..255) hard-caps registers per thread, forcing ptxas to fit the budget (spilling if needed) where `minBlocks`'s `.minnctapersm` is only advisory.
  - `@padstage`: a `var name = <tensor slice>` staging tile (an unmasked slice copied into shared memory before use, e.g. flash attention's K/V tiles) allocates through the same padded-stride mechanism `@tensorcore`'s legacy WMMA path uses, but only when the tile's own row pitch is itself an exact multiple of the shared-memory bank period (128 bytes: 32 banks times the 4-byte word each addresses) -- the layout where every row of the tile lands on the same bank at a fixed column. A tile whose pitch misses that multiple is left alone, so this never grows a kernel's shared footprint for a tile that was not the defect it targets. Masked slices (an offset the compiler cannot prove in-bounds, materialized element-by-element) do not go through this path regardless of pitch; see `phobos-lang/src/codegen/tile/check.rs`'s `materialize_masked`. Opt-in per kernel: unlike `@pipeline`, this is not attempted by default, since it changes a tile's physical layout on kernels this pass did not audit.
  - `@pipeline`: double-buffered (ping-pong shared buffer) staging is now attempted on every eligible loop by default, with no attribute needed -- a leading run of `var name = <static tensor slice>` statements whose names are never written again, whose slices are not partial, and whose doubled footprint still fits shared memory (see `phobos-lang/src/codegen/pipeline.rs`'s `pipeline_candidate`). Writing `@pipeline` now asserts that promise instead of requesting it: the kernel fails to compile, naming why, if nothing in it ends up pipelined. It still separately selects double-buffered staging for the fused-GEMM backend's own operand pairs (`matmul`/`reg`/`wmma.rs`'s `pairs`), which has no legality check of its own and stays attribute-gated exactly as before, since flipping it on by default would double every such kernel's staging footprint unconditionally -- a separate, unaudited change this pass did not make.
  - `@tensorcore`: runs the matmul on tensor cores (sm_70+).
     Operands are rounded to **f16** when staged (accumulation stays f32).
     Tile dims and the k-slice must be multiples of 16 and the CTA's warps must tile the 16x16-fragment grid;
     otherwise the kernel silently uses the regular f32 path.
     The matmul lowers through thread-level `mma.sync` + `ldmatrix` (m16n8kK:
     k8 on sm_75, k16 on sm_80+) with unpadded, XOR-swizzled shared staging by
     default. This needs 64-bit `index` (the `nvgpu`
     ops are pointer-width; the bench widens it automatically), so below sm_75
     or when `index` lowers at 32 bits it falls back to the legacy
     warp-collective WMMA API (m16n16k16). Tile-level `dot`/`dot_t` (flash
     attention) take the same `mma.sync` path under the same gate. A dot
     operand that is a `let`-bound tensor slice defined outside an enclosing
     loop (the flash `q`) is staged into shared memory once before the loop
     instead of every iteration, as long as the loop body stores to no
     tensor.
  - `@tensorcore(wmma)`: forces the legacy warp-collective WMMA m16n16k16 matmul
     back on (the pre-`mma.sync` path), e.g. for comparison or rollback.
     `@tensorcore(sync)` is accepted as a now-redundant explicit opt-in to the
     default `mma.sync` path.
  - `@dynshared`: Use dynamically shared memory (instead of the static 48KB); must change launch config respectively. Tiles pool by (element type, shape) as usual, but when every live tile has been released the byte cursor also resets to 0, so a later, differently-shaped tile can reuse a dead region's bytes instead of appending past it -- this is what lets a barrier-separated kernel's two phases share one footprint sized to the wider phase rather than their sum.
  - `@persistent`: the kernel spans several stages of a pass and separates them with `grid_barrier`, so the whole grid must be resident at once or the barrier deadlocks. The compiler only records the intent; honouring it is the launcher's, which sizes the grid from the occupancy API (`phobos_kernels::launch::persistent_grid`).
  - tile sizes for the MLIR GEMM (`TILE_M/N/K`, `WARP_M/N`, `TILE_TM/TN`) come from `@autotune` / the shape env.
  - Unknown attributes parse but are ignored (with a note)
  - **TODO**: Param- and loop-level attributes (`@readonly`, `@unroll`)
  - **TODO**: `@fast_math`: enables fast-math flags at codegen.
  - **TODO**: `@assert_coalesced`: build fails if any global access is strided (a broadcast is fine).
- **Comments:** `// ...` to end of line.

## Example

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

Gated Linear Attention (the KDA backbone)

Chunkwise gated linear attention, the first building block for Kimi Delta
Attention. It streams a head's sequence in chunks of `C`, carrying an `[D, D]`
recurrent state across chunks, and exercises `cumsum` (the running gate), `tril`
(intra-chunk causal masking), and `transpose` (the `K^T V` state update). See
[`examples/kda_fp32.ph`](./examples/kda_fp32.ph) for the fully commented kernel.

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