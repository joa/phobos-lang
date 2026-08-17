// Greedy decode's fast path: a device-side argmax over the logits row, so a
// caller that only wants the winning token id reads a couple of floats back
// instead of the whole vocab. See `autoresearch/beams/greedy-argmax-readback.md`.
//
// Two kernels. `argmax_reduce` grid-strides the row in chunks of `W`, folding
// a running `[1, W]` (value, index) pair per lane with `tmax`/`argsel`, then
// collapses that tile to one winner with an unrolled halving tree and leaves
// one partial per block in `P`. `argmax_finish` folds the (small) partial
// list the same way, serially, down to the single answer.
//
// The index side of the fold carries the winning column as an f32: safe up to
// 2^24, far past either model's vocab, and it lets `argsel` stay a plain
// float select with no integer reduction path to add alongside it.

/// Columns one program of [`argmax_reduce_src`] folds per grid-stride step.
/// Picked to divide both this session's vocabularies (130,560 and 248,320)
/// exactly; see [`argmax_chunk_width`] for what a vocabulary that does not
/// falls back to.
pub(crate) const ARGMAX_CHUNK: usize = 512;

/// Blocks [`argmax_reduce_src`] launches with, and so partials
/// [`argmax_finish_src`] folds. Comfortably below the card's SM count is not
/// the goal here: the whole reduction moves at most a couple of megabytes
/// already resident on the device, so the grid only needs to be wide enough
/// that `argmax_finish`'s serial fold over it stays cheap.
pub(crate) const ARGMAX_SPLITS: usize = 128;

/// The widest power of two, at most [`ARGMAX_CHUNK`], that divides `n`.
///
/// A chunk that does not divide the vocab would need a masked tail read, and
/// the mask's zero fill is not `tmax`'s identity: a row whose real values are
/// all negative would lose to a fake 0.0 in the ragged remainder. Every
/// candidate here is a power of two by construction (the loop only ever
/// halves), which is also what [`argmax_reduce_src`]'s halving tree needs to
/// bottom out exactly at one element.
pub(crate) fn argmax_chunk_width(n: usize) -> usize {
    let mut w = ARGMAX_CHUNK;
    while w > n.max(1) {
        w /= 2;
    }
    while w > 1 && !n.is_multiple_of(w) {
        w /= 2;
    }
    w.max(1)
}

/// Blocks to launch [`argmax_reduce_src`] with for a vocabulary of `n`, so a
/// short row never launches more blocks than it has chunks for (every block
/// would otherwise cross zero real chunks and contribute nothing but its own
/// identity to the finish pass).
pub(crate) fn argmax_splits(n: usize, w: usize) -> usize {
    ARGMAX_SPLITS.min((n / w).max(1))
}

/// The unrolled `[1, W]` -> `[1, 1]` halving tree: fold the upper half into
/// the lower with `argsel` (the index side) then `tmax` (the value side), in
/// that order, since `argsel` needs both halves' values intact. The upper
/// half goes in `argsel`/`tmax`'s first operand throughout, so a value tie
/// resolves toward the higher lane deterministically -- not a proof of
/// agreement with the host's own "last of equal maxima" rule (the lane a
/// given index lands in does not correlate with index order once the
/// grid-stride loop has folded several chunks into it), only a fixed,
/// reproducible choice for the rare case for real logits are exactly tied.
fn halving_tree(width: usize) -> String {
    let mut tree = String::new();
    let mut half = width / 2;
    while half >= 1 {
        tree.push_str(&format!(
            "  i[0 :+ 1, 0 :+ {half}] = argsel(v[0 :+ 1, {half} :+ {half}], v[0 :+ 1, 0 :+ {half}], \
             i[0 :+ 1, {half} :+ {half}], i[0 :+ 1, 0 :+ {half}])\n  \
             v[0 :+ 1, 0 :+ {half}] = tmax(v[0 :+ 1, {half} :+ {half}], v[0 :+ 1, 0 :+ {half}])\n"
        ));
        half /= 2;
    }
    tree
}

/// `A[0, :]`'s argmax, grid-strided in chunks of `w`, one partial `(value,
/// index)` pair left per block in `P[:, pid]`. `IO` is `[0, 1, .., w - 1]`,
/// uploaded once and reused for every launch at this width; adding the
/// chunk's own base index turns it into that chunk's column-to-vocab-index
/// map, which is what the fold's index side tracks.
pub(crate) fn argmax_reduce_src(w: usize) -> String {
    let tree = halving_tree(w);
    format!(
        "@launch(256)
@autotune(W in [{w}])
kernel argmax_reduce(A: tensor<f32>[M, N], IO: tensor<f32>[M, W], P: tensor<f32>[2, S]) {{
  let pid = program_id(0)
  var v: tile<f32>[1, W] = -300000000.0
  var i: tile<f32>[1, W] = -1.0
  let io = IO[0 :+ 1, :]
  for kt in range(pid * W, N, S * W) {{
    let chunk = A[0 :+ 1, kt :+ W]
    var cand: tile<f32>[1, W] = io + f32(kt)
    i = argsel(chunk, v, cand, i)
    v = tmax(chunk, v)
  }}
{tree}  P[0 :+ 1, pid :+ 1] = v[0 :+ 1, 0 :+ 1]
  P[1 :+ 1, pid :+ 1] = i[0 :+ 1, 0 :+ 1]
}}
"
    )
}

/// [`argmax_reduce_src`]'s partials, folded down to the single winner. `S` is
/// small (at most [`ARGMAX_SPLITS`]) and this runs once a step, so a plain
/// serial loop over it costs nothing worth a second block-level tree for.
pub(crate) const ARGMAX_FINISH_SRC: &str = "\
@launch(256)
kernel argmax_finish(P: tensor<f32>[2, S], OUT: tensor<f32>[1, 2]) {
  var v: tile<f32>[1, 1] = -300000000.0
  var i: tile<f32>[1, 1] = -1.0
  for s in range(0, S, 1) {
    let cv = P[0 :+ 1, s :+ 1]
    let ci = P[1 :+ 1, s :+ 1]
    i = argsel(cv, v, ci, i)
    v = tmax(cv, v)
  }
  OUT[0 :+ 1, 0 :+ 1] = v
  OUT[0 :+ 1, 1 :+ 1] = i
}
";
