// Attention kernel sources, the split and blocked decode variants,
// and rope.

use crate::backend::Attn;

/// Elements of one `[BC, head_dim]` tile in [`attention_src`]. Shared memory
/// bounds it, not arithmetic: key and value tiles are `[BC, head_dim]` each and
/// the codegen allocates a second pair for the masked replay of the loop body,
/// so four have to fit under the 48K [`ATTN_BLOCK_ELEMS`] explains.
///
/// The caches being f16 halves what those four cost, so the budget doubles and
/// a head dimension of 256 reaches [`ATTN_TILE_ROWS`] where it used to stop at
/// half of it.
pub(crate) const ATTN_TILE_ELEMS: usize = 4096;

/// The other bound on that tile, and usually the tighter one.
///
/// Rows are what the scan's remainder costs, which is why this is capped apart
/// from the budget above: the tiled loop walks whole tiles and then picks up
/// what is left one key at a time, up to `BC - 1` single-key steps. Going to 32
/// measured 7% faster in `attndecode` back when it swept powers of two, where no
/// remainder ever runs, and 45% slower on a model.
pub(crate) const ATTN_TILE_ROWS: usize = 16;

/// Rows of the cache one pass of the scan covers, and the cap on the single-key
/// remainder that follows it.
pub(crate) fn attention_tile(head_dim: usize) -> usize {
    (ATTN_TILE_ELEMS / head_dim).clamp(1, ATTN_TILE_ROWS)
}

/// Query heads one program of the decode split covers, capped rather than taken
/// from the group. Two is where both models land: at a head dimension of 128
/// over a group of eight, and at 256 over a group of four, the cache lengths a
/// decode reaches are flat between two and four and worse at one and at the
/// whole group. The query, the accumulator and its rescale are `[QG, head_dim]`
/// in shared memory, so the block's footprint grows with `QG` while the bytes it
/// saves only halve each time: the first halving is most of the win and the rest
/// is paid for in occupancy.
pub(crate) const ATTN_QGROUP: usize = 2;

/// Query heads one program of the decode split actually carries: the cap above,
/// or the largest divisor of it the group divides by. The heads a program
/// carries read one key head between them, so a program straddling two groups
/// would hand the second one the first one's keys.
pub(crate) fn attention_qgroup(group: usize) -> usize {
    (1..=ATTN_QGROUP)
        .rev()
        .find(|qgroup| group.is_multiple_of(*qgroup))
        .unwrap_or(1)
}

/// Elements of one `[BR, head_dim]` tile in [`attention_block_src`]. The blocked
/// kernel carries about ten of them against the row kernel's four, the codegen
/// staging each rescale step in shared memory rather than registers. Ten at 4 KB
/// is 40 KB, as far as this can go: these are `memref.global`s in the shared
/// address space, so they become statically declared arrays, which cap at 48 KB
/// on every architecture rather than Turing's 64.
pub(crate) const ATTN_BLOCK_ELEMS: usize = 1024;

/// Queries one program of [`attention_block_src`] covers: four at a head
/// dimension of 256, which is what caps it.
pub(crate) fn attention_block_tile(head_dim: usize) -> usize {
    (ATTN_BLOCK_ELEMS / head_dim).clamp(1, 16)
}

/// Query rows, keys and contraction depth of one [`attn_gemm_src`] tile. The
/// square tile is what puts the causal mask on `tril`: with both sides 64, the
/// one tile straddling the diagonal has key `j` paired with query `i` at the
/// same tile coordinates, so keeping `j <= i` is the mask.
pub const ATTN_GEMM_TILE: usize = 64;

pub(crate) const ATTN_GEMM_STEP: usize = 32;

/// The same for the softmax and the mix, which shared memory bounds rather than
/// throughput. The softmax's tile has to be square for the same reason and holds
/// four at once, so 64 would be 66 KB. The mix takes the same row tile, which is
/// not a tuning choice: the softmax leaves a row block zeroed only as far as its
/// own last query reaches, so a mix contracting deeper would sum raw scores for
/// the rows above. Its step is shallower to fit the `[32, 64]` accumulator.
pub const ATTN_SOFT_TILE: usize = 32;

pub(crate) const ATTN_MIX_STEP: usize = 16;

/// Rows one program of the key transpose carries.
pub(crate) const ATTN_KT_ROWS: usize = 8;

/// Whether a call tiles evenly enough for [`DeviceBackend::attention_gemm`]. The
/// kernels it falls back to need no alignment at all.
pub(crate) fn attn_gemm_fits(spec: Attn) -> bool {
    spec.rows >= ATTN_GEMM_TILE
        && spec.rows.is_multiple_of(ATTN_GEMM_TILE)
        && spec.start_pos.is_multiple_of(ATTN_GEMM_TILE)
        && spec.head_dim.is_multiple_of(ATTN_GEMM_TILE)
        && spec.total().is_multiple_of(ATTN_KT_ROWS)
}

/// Causal attention for a prompt, as two matmuls with the scores in between.
///
/// [`attention_block_src`] keeps one query block's whole attention inside one
/// program, the right shape at a small head dimension and the wrong one here. At
/// 256 the query tile, the accumulator and the key and value tiles are all
/// `[BR, 256]` f32, so 48 KB of shared memory caps `BR` at four and the score
/// tile is `[4, 4]`: sixteen output elements on a 256-thread CTA, 0.1 TFLOP/s.
/// Materializing the scores costs a `[rows, keys]` buffer per head and removes
/// the cap, leaving two ordinary tiled matmuls at 3.3 and 2.6 TFLOP/s, 26 ms of
/// a 512-token pass. The tensor cores would roughly double that again, but only
/// by rounding both operands to f16, and attention is off the critical path.
///
/// Two things have to be arranged for those matmuls to be clean. `dot_t` cannot
/// accumulate in place and `acc = acc + dot_t(..)` builds a whole tile per step,
/// 0.33 against 3.3 TFLOP/s, so the keys are transposed once a head and the
/// scores are a plain `dot`. And a head is a column window at an offset no
/// promise can bound, so slicing one inside the kernel puts a mask on every
/// operand and loses the pipelined path; each head is gathered into its own
/// buffer first. That gather is where the f16 cache widens, once per head
/// rather than once per tile the matmuls stage.
///
/// The three kernels split at the two points a row of scores has to be whole:
/// the softmax needs its row's maximum before it can exponentiate and its sum
/// before it can normalize. The sum is left for the mix to divide by, so the
/// scores are read three times rather than four.
///
/// All three skip the blocks past the diagonal rather than masking them, which
/// is half the rectangle.
pub fn attn_gemm_src(head_dim: usize, tile: usize) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    let (step, soft, mix_step) = (ATTN_GEMM_STEP, ATTN_SOFT_TILE, ATTN_MIX_STEP);
    format!(
        "@launch(256)
@autotune(TM in [{tile}], TN in [{tile}], TK in [{step}], TB in [{ATTN_KT_ROWS}], HD in [{head_dim}])
@aligned(NK = TN, KW = HD, DK = TB)
kernel attn_kt(K: tensor<f16>[NK, KW], T: tensor<f32>[DK, NK]) {{
  let p = program_id(0)
  var t = K[p * TB :+ TB, 0 :+ HD]
  T[0 :+ HD, p * TB :+ TB] = f32(transpose(t))
}}

@launch(256)
@autotune(TM in [{tile}], TN in [{tile}], TK in [{step}], HD in [{head_dim}])
@aligned(R = TM, NK = TN, DQ = TK, DK = TK)
kernel attn_scores(Q: tensor<f32>[R, DQ], KT: tensor<f32>[DK, NK], S: tensor<f32>[R, NK]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  if pn * TN < NK - R + pm * TM + TM {{
    var acc: tile<f32>[TM, TN] = 0.0
    for kt in range(0, DQ, TK) {{
      var a = Q[pm * TM :+ TM, kt :+ TK]
      var b = KT[kt :+ TK, pn * TN :+ TN]
      acc += dot(a, b)
    }}
    S[pm * TM :+ TM, pn * TN :+ TN] = acc * {scale:.9}
  }}
}}

@launch(256)
@autotune(TM in [{soft}], TN in [{soft}])
@aligned(R = TM, NK = TN)
kernel attn_softmax(S: tensor<f32>[R, NK], L: tensor<f32>[R, 1]) {{
  let p = program_id(0)
  let diag = NK - R + p * TM
  var m: tile<f32>[TM, 1] = -300000000.0
  for j in range(0, diag, TN) {{
    m = tmax(m, rowmax(S[p * TM :+ TM, j :+ TN]))
  }}
  var d = S[p * TM :+ TM, diag :+ TN]
  m = tmax(m, rowmax(d))

  var l: tile<f32>[TM, 1] = 0.0
  for j in range(0, diag, TN) {{
    var e = exp(S[p * TM :+ TM, j :+ TN] - m)
    l = l + rowsum(e)
    S[p * TM :+ TM, j :+ TN] = e
  }}
  var de = tril(exp(d - m))
  l = l + rowsum(de)
  S[p * TM :+ TM, diag :+ TN] = de
  L[p * TM :+ TM, 0 :+ 1] = l
}}

@launch(256)
@autotune(TM in [{soft}], TN in [{tile}], TK in [{mix_step}])
@aligned(R = TM, DV = TN, NK = TK)
kernel attn_mix(P: tensor<f32>[R, NK], V: tensor<f32>[NK, DV], L: tensor<f32>[R, 1],
                O: tensor<f32>[R, DV]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, NK - R + pm * TM + TM, TK) {{
    var a = P[pm * TM :+ TM, kt :+ TK]
    var b = V[kt :+ TK, pn * TN :+ TN]
    acc += dot(a, b)
  }}
  O[pm * TM :+ TM, pn * TN :+ TN] = acc / L[pm * TM :+ TM, 0 :+ 1]
}}
"
    )
}

/// Causal attention over a block of queries at once.
///
/// [`attention_src`] gives each program one query row, which re-reads the key
/// and value tiles per query. Here a program owns `BR` consecutive positions of
/// one head, so one pass over the cache serves all of them. The caches are f16
/// and widen as the tiles are read, for the reason [`attention_split_src`]
/// gives.
///
/// The query block needs no rearranging: `Q` is `[rows, n_head * head_dim]`, the
/// same memory as the `[rows * n_head, head_dim]` the norm and the rotary want.
///
/// Causality is `tril` on the one tile straddling the diagonal, which is why the
/// key tile is the query block's size: at both `BR`, tile column `j` is key
/// `base + j` and tile row `i` is query `base + i`. It cleans up the end of the
/// cache too, since a column past the last key is only paired with a query row
/// that is also past it, and that row is not stored.
///
/// The mask goes on the probabilities rather than the scores, which is the same
/// thing: a masked score's `exp` is what softmax would have driven to zero. The
/// running maximum takes in masked entries too, which only shifts the
/// exponentials down and cannot overflow.
pub(crate) fn attention_block_src(
    n_head: usize,
    group: usize,
    head_dim: usize,
    tile: usize,
) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    format!(
        "@launch(256)
@autotune(NH in [{n_head}], G in [{group}], D in [{head_dim}], BR in [{tile}])
kernel attention_block(Q: tensor<f32>[R, QW], K: tensor<f16>[NK, KW],
                       V: tensor<f16>[NK, KW], O: tensor<f32>[R, QW]) {{
  let qt = program_id(0)
  let h = program_id(1)
  let qcol = h * D
  let kcol = h / G * D
  var q = Q[qt * BR :+ BR, qcol :+ D]

  var acc: tile<f32>[BR, D] = 0.0
  var m: tile<f32>[BR, 1] = -300000000.0
  var l: tile<f32>[BR, 1] = 0.0

  let base = NK - R + qt * BR
  for kt in range(0, base, BR) {{
    let k = K[kt :+ BR, kcol :+ D]
    let v = V[kt :+ BR, kcol :+ D]
    var s: tile<f32>[BR, BR] = dot_t(q, k)
    s = s * {scale:.9}
    var mn: tile<f32>[BR, 1] = rowmax(s)
    mn = tmax(m, mn)
    s = exp(s - mn)
    var corr: tile<f32>[BR, 1] = exp(m - mn)
    l = l * corr + rowsum(s)
    acc = acc * corr + dot(s, v)
    m = mn
  }}

  let dk = K[base :+ BR, kcol :+ D]
  let dv = V[base :+ BR, kcol :+ D]
  var ds: tile<f32>[BR, BR] = dot_t(q, dk)
  ds = ds * {scale:.9}
  var dm: tile<f32>[BR, 1] = rowmax(ds)
  dm = tmax(m, dm)
  ds = exp(ds - dm)
  ds = tril(ds)
  var dcorr: tile<f32>[BR, 1] = exp(m - dm)
  l = l * dcorr + rowsum(ds)
  acc = acc * dcorr + dot(ds, dv)

  O[qt * BR :+ BR, qcol :+ D] = acc / l
}}
"
    )
}

/// The rotary embedding, in place.
///
/// The angles arrive as a table rather than being computed: the language has no
/// sine, and the hardware's approximate one loses accuracy across the range an
/// absolute position covers. The caller offsets `T` to the first row's
/// position, so `r / H` is the row's position within it.
///
/// Reading both halves into tiles before either store is what makes the update
/// safe in place; the second store still needs the pre-rotation values.
pub(crate) fn rope_src(heads: usize, half: usize) -> String {
    format!(
        "@launch(256)
@autotune(H in [{heads}])
kernel rope(X: tensor<f32>[R, D], T: tensor<f32>[P, RD]) {{
  let r = program_id(0)
  let p = r / H
  var a = X[r :+ 1, 0 :+ {half}]
  var b = X[r :+ 1, {half} :+ {half}]
  var c = T[p :+ 1, 0 :+ {half}]
  var s = T[p :+ 1, {half} :+ {half}]
  X[r :+ 1, 0 :+ {half}] = a * c - b * s
  X[r :+ 1, {half} :+ {half}] = a * s + b * c
}}
"
    )
}

/// Causal softmax attention over the whole cache, one group of query rows per
/// program, with the key axis split across the grid and a pass folding the
/// pieces back together.
///
/// FlashAttention's online softmax at one query row: the running maximum and
/// denominator rescale the carried output as each key tile arrives, so the
/// scores are never materialized and the cache is read once. Causality is a loop
/// bound rather than a mask, since the keys a row may see are a prefix: the tiled
/// loop runs over that prefix's whole tiles and picks up the last few keys one at
/// a time, at most `BC - 1` narrow steps and no triangular mask or padding.
///
/// `SP` is not passed: the key extent is `start_pos + rows` and the query extent
/// is `rows * n_head`, so the kernel recovers the position from the two. The
/// caches keep every head of one position together, so `K` and `V` are indexed
/// by `col` rather than sliced by row.
///
/// Unsplit, one query row leaves a grid of `n_head` blocks, a sixth of this card,
/// each walking the whole cache: eight per cent of a decode step for a few
/// megabytes of reading. A split block covers a slice of the keys and writes the
/// running maximum, sum and unnormalized accumulator it reached; merging those is
/// the same rescaling the online softmax already does between tiles.
///
/// A program carries `QG` query rows, the ones sharing a key head: grouped-query
/// attention gives `G` queries one key head, so single-row programs read that
/// head once per query and the L2 does not absorb it. `group 1, same grid` in
/// the `attndecode` example is the measurement, eight times the distinct bytes
/// at the same block count for ten per cent more time. Carrying the group reads
/// a key tile once and contracts it against `QG` queries, the same tiled loop
/// with a taller first operand. It divides the grid, so the split count
/// multiplies by `QG` to leave the block count the tile size was tuned against.
///
/// The keys are cut into pieces of whole tiles rather than into equal parts, so
/// the single-key steps that finish a piece run once for the whole cache instead
/// of once per piece: cutting evenly costs about `BC / 2` of them per piece at a
/// length that does not divide, which is around 64 against the 128 tiled steps a
/// cache of 2048 takes. The price is that a cache with fewer tiles than pieces
/// idles the last blocks, and those are the caches whose attention is too cheap
/// to matter.
///
/// The merge folds a head's pieces in one shot rather than walking them, which
/// would cost a dependent global load each, about 0.3 us with nothing to overlap
/// it. The two buffers are one memory under two shapes: the partials are written
/// `[S * NH, D]` at row `s * NH + h` and read `[S, NH * D]` sliced at column
/// `h * D`, so a head's pieces are a plain `[S, D]` tile. The maxima and sums are
/// `[NH, 2 * S]` for the same reason, a head's contiguous, written as a column
/// and reduced as a row.
///
/// Both decode kernels below say `@aligned(KW = D)` and stage their key and
/// value slices with `var` rather than binding them with `let`, and the two go
/// together. A head is the column window `[col :+ D]` at `col = h / G * D`, and
/// without the promise that the cache width is a whole number of head widths,
/// which it is, `KW` being `n_kv * head_dim`, that window may run past the row
/// and every element of it is loaded under a bounds compare, one scalar
/// `ld.global.b32` at a time. The promise alone is worse than the mask: an
/// unmasked slice is read where it lies instead of being staged, so the
/// contraction walks global memory in a serial dependent chain and the register
/// count goes from 75 to 166, which is three blocks per SM down to one. `var`
/// asks for the staging the mask used to force, and the copy then vectorizes.
/// Worth 37% of a decode step's attention at a cache of 1024, and 44% of what
/// each further cached position costs, which is the number that matters.
///
/// `K` and `V` are f16 and the query is not. A cached element widens where the
/// staged tile is read, so the contraction is the same f32 one and only the
/// halves cross the bus; the staged tiles are f16 as well, which is the shared
/// memory the tile size above is measured against. Nothing else narrows: the
/// query, the running maximum and the accumulator all stay f32, and a decode
/// step is bound by what it reads rather than by what it holds.
pub(crate) fn attention_split_src(
    n_head: usize,
    group: usize,
    head_dim: usize,
    tile: usize,
    qgroup: usize,
    splits: usize,
) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    format!(
        "@launch(256)
@autotune(NH in [{n_head}], G in [{group}], QG in [{qgroup}], D in [{head_dim}], BC in [{tile}],
          S in [{splits}])
@aligned(KW = D)
kernel attention_split(Q: tensor<f32>[R, D], K: tensor<f16>[NK, KW],
                       V: tensor<f16>[NK, KW],
                       P: tensor<f32>[SH, D], ML: tensor<f32>[NH, MW]) {{
  let g = program_id(0)
  let s = program_id(1)
  let h = g * QG
  let col = h / G * D
  var q = Q[h :+ QG, 0 :+ D]

  var acc: tile<f32>[QG, D] = 0.0
  var m: tile<f32>[QG, 1] = -300000000.0
  var l: tile<f32>[QG, 1] = 0.0

  let tiles = (NK + BC - 1) / BC
  let per = (tiles + S - 1) / S * BC
  var lo = s * per
  if lo > NK {{
    lo = NK
  }}
  var hi = lo + per
  if hi > NK {{
    hi = NK
  }}
  let full = lo + (hi - lo) / BC * BC
  for kt in range(lo, full, BC) {{
    var k = K[kt :+ BC, col :+ D]
    var v = V[kt :+ BC, col :+ D]
    var sc: tile<f32>[QG, BC] = dot_t(q, k)
    sc = sc * {scale:.9}
    var mn: tile<f32>[QG, 1] = rowmax(sc)
    mn = tmax(m, mn)
    sc = exp(sc - mn)
    var corr: tile<f32>[QG, 1] = exp(m - mn)
    l = l * corr + rowsum(sc)
    acc = acc * corr + dot(sc, v)
    m = mn
  }}
  for j in range(full, hi, 1) {{
    var k1 = K[j :+ 1, col :+ D]
    var v1 = V[j :+ 1, col :+ D]
    var s1: tile<f32>[QG, 1] = dot_t(q, k1)
    s1 = s1 * {scale:.9}
    var mn: tile<f32>[QG, 1] = tmax(m, s1)
    var p: tile<f32>[QG, 1] = exp(s1 - mn)
    var corr: tile<f32>[QG, 1] = exp(m - mn)
    l = l * corr + p
    acc = acc * corr + dot(p, v1)
    m = mn
  }}
  P[s * NH + h :+ QG, 0 :+ D] = acc
  ML[h :+ QG, s :+ 1] = m
  ML[h :+ QG, S + s :+ 1] = l
}}

@launch(256)
@autotune(NH in [{n_head}], D in [{head_dim}], S in [{splits}])
kernel attention_merge(P: tensor<f32>[S, PW], ML: tensor<f32>[NH, MW],
                       O: tensor<f32>[R, D]) {{
  let h = program_id(0)
  var mv = ML[h :+ 1, 0 :+ S]
  var m: tile<f32>[1, 1] = rowmax(mv)
  var c: tile<f32>[1, S] = exp(mv - m)
  var l: tile<f32>[1, 1] = rowsum(c * ML[h :+ 1, S :+ S])
  var acc: tile<f32>[1, D] = dot(c, P[0 :+ S, h * D :+ D])
  O[h :+ 1, 0 :+ D] = acc / l
}}
"
    )
}

/// [`attention_split_src`]'s two kernels, `attention_split` and
/// `attention_merge`, folded into one `@persistent` kernel via
/// `grid_barrier()` instead of a launch and a scratch round-trip. Phase one is
/// the split kernel's body verbatim, over a grid-strided range of `(group,
/// split)` units instead of one per block; phase two is the merge kernel's
/// body, over a grid-strided range of heads. The two phases share nothing but
/// the barrier and the scratch, so neither changed a line of the
/// online-softmax math both already had.
///
/// The scratch buffer the two phases round-trip through is a parameter
/// twice, `P` and `PM`, one pointer under the two shapes the two phases read
/// and write, exactly as [`DeviceBackend::attention_decode`] (see
/// `phobos-gguf/src/backend/device/attn.rs`) already passes it to two
/// separate kernels; a persistent kernel can bind the same device pointer to
/// two parameter names in one launch just as easily as two launches could.
///
/// `IT1` and `IT2` are the grid-strided loop counts, compiled in rather than
/// derived from a dynamic tensor extent: `docs/megakernel.md`'s own step 1
/// found that the natural spelling of a strided loop over a dynamic bound
/// gets split, and the masked remainder then wants a static shape neither
/// loop has. Compiling the trip count in and guarding with `if unit < total`
/// sidesteps that, the same fix the fused MLP's own unit loops use.
///
/// Both phases assume every block of `BLOCKS` is resident at once, since
/// `grid_barrier` deadlocks otherwise; the caller settles `BLOCKS` from
/// `cuOccupancyMaxActiveBlocksPerMultiprocessor` before compiling this, the
/// same precondition `docs/megakernel.md` names for the wider megakernel.
pub(crate) fn attention_persist_src(
    n_head: usize,
    group: usize,
    head_dim: usize,
    tile: usize,
    qgroup: usize,
    splits: usize,
    blocks: u32,
) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    let groups = n_head / qgroup;
    let units1 = groups * splits;
    let it1 = (units1 as u32).div_ceil(blocks);
    let it2 = (n_head as u32).div_ceil(blocks);
    format!(
        "@launch(256)
@persistent
@autotune(NH in [{n_head}], G in [{group}], QG in [{qgroup}], D in [{head_dim}], BC in [{tile}],
          S in [{splits}], U1 in [{units1}], IT1 in [{it1}], IT2 in [{it2}], BLOCKS in [{blocks}])
@aligned(KW = D)
kernel attention_persist(Q: tensor<f32>[R, D], K: tensor<f16>[NK, KW],
                       V: tensor<f16>[NK, KW],
                       P: tensor<f32>[SH, D], PM: tensor<f32>[S, PW],
                       ML: tensor<f32>[NH, MW], O: tensor<f32>[R, D],
                       BAR: tensor<i32>[2, 1]) {{
  let pid = program_id(0)

  for i1 in range(0, IT1) {{
    let u = pid + i1 * BLOCKS
    if u < U1 {{
      let g = u / S
      let s = u % S
      let h = g * QG
      let col = h / G * D
      var q = Q[h :+ QG, 0 :+ D]

      var acc: tile<f32>[QG, D] = 0.0
      var m: tile<f32>[QG, 1] = -300000000.0
      var l: tile<f32>[QG, 1] = 0.0

      let tiles = (NK + BC - 1) / BC
      let per = (tiles + S - 1) / S * BC
      var lo = s * per
      if lo > NK {{
        lo = NK
      }}
      var hi = lo + per
      if hi > NK {{
        hi = NK
      }}
      let full = lo + (hi - lo) / BC * BC
      for kt in range(lo, full, BC) {{
        var k = K[kt :+ BC, col :+ D]
        var v = V[kt :+ BC, col :+ D]
        var sc: tile<f32>[QG, BC] = dot_t(q, k)
        sc = sc * {scale:.9}
        var mn: tile<f32>[QG, 1] = rowmax(sc)
        mn = tmax(m, mn)
        sc = exp(sc - mn)
        var corr: tile<f32>[QG, 1] = exp(m - mn)
        l = l * corr + rowsum(sc)
        acc = acc * corr + dot(sc, v)
        m = mn
      }}
      for j in range(full, hi, 1) {{
        var k1 = K[j :+ 1, col :+ D]
        var v1 = V[j :+ 1, col :+ D]
        var s1: tile<f32>[QG, 1] = dot_t(q, k1)
        s1 = s1 * {scale:.9}
        var mn: tile<f32>[QG, 1] = tmax(m, s1)
        var prob: tile<f32>[QG, 1] = exp(s1 - mn)
        var corr: tile<f32>[QG, 1] = exp(m - mn)
        l = l * corr + prob
        acc = acc * corr + dot(prob, v1)
        m = mn
      }}
      P[s * NH + h :+ QG, 0 :+ D] = acc
      ML[h :+ QG, s :+ 1] = m
      ML[h :+ QG, S + s :+ 1] = l
    }}
  }}

  grid_barrier(BAR)

  for i2 in range(0, IT2) {{
    let u2 = pid + i2 * BLOCKS
    if u2 < NH {{
      var mv = ML[u2 :+ 1, 0 :+ S]
      var mm: tile<f32>[1, 1] = rowmax(mv)
      var c: tile<f32>[1, S] = exp(mv - mm)
      var ll: tile<f32>[1, 1] = rowsum(c * ML[u2 :+ 1, S :+ S])
      var oacc: tile<f32>[1, D] = dot(c, PM[0 :+ S, u2 * D :+ D])
      O[u2 :+ 1, 0 :+ D] = oacc / ll
    }}
  }}
}}
"
    )
}

/// Pieces the key axis is cut into while decoding, per query head a program
/// carries. Fixed rather than chosen from the cache length, which is what it
/// wants to be: a count that grows with the cache reshapes the pass every few
/// dozen tokens, and each change costs a graph rebuild. A piece with no keys
/// only costs a block that exits immediately.
///
/// The merge folds a head's pieces as one `[S, head_dim]` tile, so this times
/// [`ATTN_QGROUP`] is bounded by shared memory: 16 pieces at a head dimension of
/// 256 is a 16 KB tile, and 64 would not compile.
pub(crate) const ATTN_SPLITS: usize = 8;

pub(crate) fn attention_src(n_head: usize, group: usize, head_dim: usize, tile: usize) -> String {
    let scale = (head_dim as f32).sqrt().recip();
    format!(
        "@launch(256)
@autotune(NH in [{n_head}], G in [{group}], D in [{head_dim}], BC in [{tile}])
@aligned(KW = D)
kernel attention(Q: tensor<f32>[R, D], K: tensor<f16>[NK, KW],
                 V: tensor<f16>[NK, KW], O: tensor<f32>[R, D]) {{
  let t = program_id(0)
  let h = program_id(1)
  let col = h / G * D
  let row = t * NH + h
  var q = Q[row :+ 1, 0 :+ D]

  var acc: tile<f32>[1, D] = 0.0
  var m: tile<f32>[1, 1] = -300000000.0
  var l: tile<f32>[1, 1] = 0.0

  let visible = NK - R / NH + t + 1
  let full = visible / BC * BC
  for kt in range(0, full, BC) {{
    var k = K[kt :+ BC, col :+ D]
    var v = V[kt :+ BC, col :+ D]
    var s: tile<f32>[1, BC] = dot_t(q, k)
    s = s * {scale:.9}
    var mn: tile<f32>[1, 1] = rowmax(s)
    mn = tmax(m, mn)
    s = exp(s - mn)
    var corr: tile<f32>[1, 1] = exp(m - mn)
    l = l * corr + rowsum(s)
    acc = acc * corr + dot(s, v)
    m = mn
  }}
  for j in range(full, visible, 1) {{
    var k1 = K[j :+ 1, col :+ D]
    var v1 = V[j :+ 1, col :+ D]
    var s1: tile<f32>[1, 1] = dot_t(q, k1)
    s1 = s1 * {scale:.9}
    var mn: tile<f32>[1, 1] = tmax(m, s1)
    var p: tile<f32>[1, 1] = exp(s1 - mn)
    var corr: tile<f32>[1, 1] = exp(m - mn)
    l = l * corr + p
    acc = acc * corr + p * v1
    m = mn
  }}
  O[row :+ 1, 0 :+ D] = acc / l
}}
"
    )
}
