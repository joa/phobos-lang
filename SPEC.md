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
var_stmt    = "var" ident [ ":" type ] "=" expr terminator ;
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
    loops carrying `mma.sync` fragment accumulators or driven by `@pipeline`
    do not split yet, so their slices are masked unless `@aligned` covers the
    dimension they walk.
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
- **Statements end at newlines**: Similar to how Golang is doing it
- **`else` must follow `}` on the same line**: a newline after the `}` of the then-block ends the `if` statement.
- Tiles
  - **Ranges**:`A[start : end]` is the elements from `start` up to but not including `end`.
  - **Spans**: `A[start :+ length]` is `length` elements starting at `start`; same as `A[start : start + length]`.
  - **Full**: `A[:]` selects the entire dimension.
  - **Open-Ended**: `A[i:]`, `A[:j]` are not supported. A `:` after an expression requires an end, and a `:+` requires a length.
- **Unary minus on a tile**: `-t` negates elementwise, lowering as `0 - t`. `!` stays scalar-only.
- **Broadcasting**: binary tile ops broadcast a NumPy-style axis of extent 1 (so `[R, C] x [R, 1]` stretches the column vector), and `tile x scalar` (either order) broadcasts the scalar over the tile.
- **Contextual Identifiers:** `tensor`, `tile`, `range`, `program_id`, the tile builtins (`dot`, `dot_t`, `qdot_t`, `qmma_t`, `exp`, `log`, `round`, `sqrt`, `tanh`, `rowmax`, `rowsum`, `tmax`, `cumsum`, `tril`, `transpose`) and the
  conversion builtins named after the scalar types are ordinary
  identifiers, not keywords. `range` is recognized positionally inside `for ... in range(...)`.
- **Built-Ins**:
  - `dot(a, b)`: `a @ b` (contracts `a`'s last dim with `b`'s first).
  - `dot_t(a, b)`: `a @ b.t` (contracting the last dim of both: `[M, K] x [N, K] -> [M, N]`). Over `i8` operands this is the integer tensor-core and `dp4a` path; see **Integer contraction**.
  - `qdot_t(a, a_scales, w, w_scales)`: the Q8_0 contraction with its block scales, `[M, K] i8 x [N, K] i8 -> [M, N] f32`, where the scales are `[M, K/32]` and `[N, K/32]` f32 and element `[i, j]` is `sum_b (sum_{k in block b} a[i, k] * w[j, k]) * a_scales[i, b] * w_scales[j, b]`. The contraction axis may be dynamic, since it is not tiled. The scales are indexed `[row, block]` so a lane's scale load is contiguous with its neighbours'. See **Quantized contraction**.
  - `qmma_t(a, a_scales, w, w_scales)`: the same contraction batched over rows, `[M, K] i8 x [N, K] i8 -> [M, N] f32` with `M` and `N` multiples of 8, where `a_scales` is `[M, K/32]` and `w_scales` is `[K/32, N]`. The weight scales are indexed `[block, out]`, the opposite of `qdot_t`: a lane here holds two neighbouring output columns of one block, so that order puts its two scales next to each other. See **Quantized contraction**.
  - `exp(t)`: element-wise `e^x` (lowers to the hardware `ex2.approx`).
  - `log(t)`: element-wise natural logarithm (lowers to the hardware `lg2.approx`, with the change of base folded in).
  - `round(t)`: element-wise nearest integer, ties to even (lowers to the hardware `cvt.rni.f32.f32`). Rounding by biasing into a positive range and truncating instead costs the low mantissa bits, which is enough to move a value across a boundary at the top of a quantization range.
  - `sqrt(t)`: element-wise square root (lowers to the hardware `sqrt.approx.f32`).
  - `tanh(t)`: element-wise hyperbolic tangent (lowers to the hardware `tanh.approx.f32`).
  - `rowmax(t)` / `rowsum(t)`: reduce a rank-2 tile over its last (column) dim to a `[rows, 1]` column vector.
  - `tmax(a, b)`: elementwise maximum (broadcasting).
  - `cumsum(t)`: inclusive prefix sum of a rank-2 tile down its first (row) dim, so `out[i, j] = sum_{r <= i} t[r, j]`. The scan runs along the sequence axis (the leading dim of a `[seq, feat]` tile), producing the running gate cumulant that chunkwise linear attention needs. Same shape as the input.
  - `tril(t)`: causal lower-triangular mask of a rank-2 tile, keeping `t[i, j]` when `j <= i` and zeroing the strict upper triangle. Same shape as the input.
  - `transpose(t)`: rank-2 tile transpose, `out[i, j] = t[j, i]` (a `[R, C]` tile becomes `[C, R]`). Lets a contraction run over the leading (sequence) axis, which `dot`/`dot_t` cannot reach on their own.
  - `f16(x)` / `bf16(x)` / `f32(x)` / `f64(x)` / `i8(x)` / `i32(x)` / `i64(x)`: element type conversion of a tile or a scalar (see **Conversions** above).
- **Attributes**:
  - `@autotune(X in [..], ...)`: local search space; the first choice seeds the shape env. Two values are inclusive bounds searched in doubling steps (`X in [16, 256]` -> 16, 32, 64, 128, 256); three or more are an explicit list of choices. `[256, 16]` is two values (when x > y)
  - `@cluster(X in [..], ...)`: super tile dimensions and search space for cluster tuning.
  - `@aligned(DIM = tile, ...)`: promises that a symbolic tensor dimension is a whole number of `tile` elements, where `tile` is an integer or an `@autotune` constant. 
  - `@launch(maxThreads[, minBlocks[, maxRegs]])`: specifies CTA thread assumption (default: 256); maps to PTX `.maxntid` / `.minnctapersm` / `.maxnreg` at codegen. `maxRegs` (16..255) hard-caps registers per thread, forcing ptxas to fit the budget (spilling if needed) where `minBlocks`'s `.minnctapersm` is only advisory.
  - `@pipeline`: selects the double-buffered (ping-pong shared buffer) MLIR GEMM backend.
  - `@tensorcore`: runs the matmul on tensor cores (sm_70+).
     Operands are rounded to **f16** when staged (accumulation stays f32).
     Tile dims and the k-slice must be multiples of 16 and the CTA's warps must tile the 16x16-fragment grid;
     otherwise the kernel silently uses the regular f32 path.
     The matmul lowers through thread-level `mma.sync` + `ldmatrix` (m16n8kK:
     k8 on sm_75, k16 on sm_80+) with unpadded, XOR-swizzled shared staging by
     default (see `docs/MMA_SYNC.md`). This needs 64-bit `index` (the `nvgpu`
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
  - `@dynshared`: Use dynamically shared memory (instead of the static 48KB); must change launch config respectively.
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