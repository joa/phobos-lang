// Device-side argmax over the logits row, for greedy decode. The caller
// reads back two floats instead of the whole vocab.
//
// `argmax_reduce` grid-strides the row in chunks of `W`, keeping a running
// (value, index) pair per lane, and writes one partial per block to `P`.
// `argmax_finish` folds the partials. Both pick the largest value, then the
// largest index holding it.
//
// Indices are carried as f32, which is exact up to 2^24. That keeps
// `argsel` a plain float select.

/// Columns one program of [`argmax_reduce_src`] folds per grid-stride step.
/// [`argmax_chunk_width`] narrows it for a vocab it does not divide.
pub(crate) const ARGMAX_CHUNK: usize = 512;

/// Blocks [`argmax_reduce_src`] launches, and so the partials
/// [`argmax_finish_src`] folds. Fixed whatever the vocab: a block with no
/// chunk leaves the identity.
pub(crate) const ARGMAX_SPLITS: usize = 128;

/// The widest power of two, at most [`ARGMAX_CHUNK`], that divides `n`.
///
/// The chunk must divide the vocab. A masked tail would fill with zero,
/// which is not `tmax`'s identity, so an all-negative row would lose to a
/// fake 0.0.
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

/// Argmax of `A[0, :]`, grid-strided in chunks of `w`. Each block leaves
/// one `(value, index)` partial in `P[:, pid]`.
///
/// `IO` is `[0, 1, .., w - 1]`, uploaded once per width. Adding a chunk's
/// base turns it into that chunk's vocab indices.
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

/// Folds [`argmax_reduce_src`]'s [`ARGMAX_SPLITS`] partials to one winner.
/// Ties go to the largest index, matching the host.
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
