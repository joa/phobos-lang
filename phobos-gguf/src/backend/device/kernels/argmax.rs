// Greedy decode's fast path: a device-side argmax over the logits row, so a
// caller that only wants the winning token id reads a couple of floats back
// instead of the whole vocab.
//
// Two kernels: `argmax_reduce` grid-strides the row in chunks of `W`,
// folding a running `[1, W]` (value, index) pair per lane with
// `tmax`/`argsel`, and leaves one partial per block in `P`. Both kernels
// collapse a row the same way: its largest value, then the largest index
// holding it, two row reductions.
//
// The index side of the fold carries the winning column as an f32, safe up
// to 2^24 and far past a real vocabulary, so `argsel` stays a plain float
// select with no integer reduction path to add alongside it.

/// Columns one program of [`argmax_reduce_src`] folds per grid-stride step,
/// picked to divide real vocabularies exactly; [`argmax_chunk_width`]
/// handles one that does not.
pub(crate) const ARGMAX_CHUNK: usize = 512;

/// Blocks [`argmax_reduce_src`] launches with, and so partials
/// [`argmax_finish_src`] folds. Matching the SM count is not the goal: the
/// whole reduction moves at most a couple of megabytes already resident on
/// the device. Always this many, whatever the vocab: a block with no chunk
/// leaves the identity, and `argmax_finish` reduces a row this wide.
pub(crate) const ARGMAX_SPLITS: usize = 128;

/// The widest power of two, at most [`ARGMAX_CHUNK`], that divides `n`.
///
/// A chunk that does not divide the vocab would need a masked tail read,
/// and the mask's zero fill is not `tmax`'s identity: a row whose real
/// values are all negative would lose to a fake 0.0 in the ragged
/// remainder.
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

/// `A[0, :]`'s argmax, grid-strided in chunks of `w`, one partial `(value,
/// index)` pair left per block in `P[:, pid]`. `IO` is `[0, 1, .., w - 1]`,
/// uploaded once and reused for every launch at this width; adding the
/// chunk's own base index turns it into that chunk's column-to-vocab-index
/// map, which is what the fold's index side tracks.
pub(crate) fn argmax_reduce_src(w: usize) -> String {
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
  let vm = rowmax(v)
  var none: tile<f32>[1, W] = -1.0
  P[0 :+ 1, pid :+ 1] = vm
  P[1 :+ 1, pid :+ 1] = rowmax(argsel(v, vm, i, none))
}}
"
    )
}

/// [`argmax_reduce_src`]'s [`ARGMAX_SPLITS`] partials folded to the single
/// winner: the largest value, then the largest index holding it, the host's
/// last-of-equal-maxima rule.
pub(crate) fn argmax_finish_src() -> String {
    format!(
        "@launch(256)
@autotune(S in [{ARGMAX_SPLITS}])
kernel argmax_finish(P: tensor<f32>[2, S], OUT: tensor<f32>[1, 2]) {{
  let v = P[0 :+ 1, 0 :+ S]
  let vm = rowmax(v)
  var none: tile<f32>[1, S] = -1.0
  OUT[0 :+ 1, 0 :+ 1] = vm
  OUT[0 :+ 1, 1 :+ 1] = rowmax(argsel(v, vm, P[1 :+ 1, 0 :+ S], none))
}}
"
    )
}
