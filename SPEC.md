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
scalar      = "f16" | "f32" | "f64" | "i32" | "i64" | "bool" ;
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
- **Tensor size**: Assumed to be a multiple of `4`. A tensor dimension whose
  size is a compile-time constant need not be a multiple of the tile: a slice
  that would run off the end is masked, so the last (partial) tile reads
  out-of-bounds elements as zero and skips their stores. This falls back from
  the specialized register/tensor-core matmul paths to the generic tiled path.
  Dynamic (runtime) sizes are still assumed to tile evenly.
- **`f16`**: half precision (IEEE binary16). Float literals are written in f32 and
  rounded to f16 on store, and arithmetic that mixes f16 with a wider float widens
  to the wider type (so `f16 + f32 -> f32`). The heavy compute paths run f16 inputs
  through the tensor cores with f32 accumulation (see `@tensorcore`); without it,
  f16 tiles use the generic element/vector paths and accumulate in f16.
- **Statements end at newlines**: Similar to how Golang is doing it
- **`else` must follow `}` on the same line**: a newline after the `}` of the then-block ends the `if` statement.
- Tiles
  - **Ranges**:`A[start : end]` is the elements from `start` up to but not including `end`.
  - **Spans**: `A[start :+ length]` is `length` elements starting at `start`; same as `A[start : start + length]`.
  - **Full**: `A[:]` selects the entire dimension.
  - **Open-Ended**: `A[i:]`, `A[:j]` are not supported. A `:` after an expression requires an end, and a `:+` requires a length.
- **Broadcasting**: binary tile ops broadcast a NumPy-style axis of extent 1 (so `[R, C] x [R, 1]` stretches the column vector), and `tile x scalar` (either order) broadcasts the scalar over the tile.
- **Contextual Identifiers:** `tensor`, `tile`, `range`, `program_id` and the tile builtins (`dot`, `dot_t`, `exp`, `rowmax`, `rowsum`, `tmax`) are ordinary
  identifiers, not keywords. `range` is recognized positionally inside `for ... in range(...)`.
- **Built-Ins**:
  - `dot(a, b)`: `a @ b` (contracts `a`'s last dim with `b`'s first).
  - `dot_t(a, b)`: `a @ b.t` (contracting the last dim of both: `[M, K] x [N, K] -> [M, N]`).
  - `exp(t)`: element-wise `e^x` (lowers to the hardware `ex2.approx`).
  - `rowmax(t)` / `rowsum(t)`: reduce a rank-2 tile over its last (column) dim to a `[rows, 1]` column vector.
  - `tmax(a, b)`: elementwise maximum (broadcasting).
- **Attributes**:
  - `@autotune(X in [..], ...)`: local search space; the first choice seeds the shape env. Two values are inclusive bounds searched in doubling steps (`X in [16, 256]` -> 16, 32, 64, 128, 256); three or more are an explicit list of choices. `[256, 16]` is two values (when x > y)
  - `@cluster(X in [..], ...)`: super tile dimensions and search space for cluster tuning.
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
  - tile sizes for the MLIR GEMM (`TILE_M/N/K`, `WARP_M/N`, `TILE_TM/TN`) come from `@autotune` / the shape env.
  - Unknown attributes parse but are ignored (with a note)
  - **TODO**: Param- and loop-level attributes (`@align`, `@readonly`, `@unroll`)
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