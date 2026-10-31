// Grid and sign tables the IQ formats index into, transcribed from
// `ggml-common.h`, and the packed forms the device kernels read.

pub(crate) mod iq1;
pub(crate) mod iq2;
pub(crate) mod iq3;

/// A grid as raw signed bytes, one a lane: byte `j` of entry `i` lands at
/// `i * W + j`. A lane's values are contiguous, so one load fetches them,
/// and every magnitude fits an `i8`.
#[cfg(feature = "cuda")]
pub(crate) fn packed<T: Copy, const W: usize>(grid: &[T], le_bytes: fn(T) -> [u8; W]) -> Vec<i8> {
    grid.iter().flat_map(|&entry| le_bytes(entry).map(|b| b as i8)).collect()
}

/// A sign table as the device reads it, eight entries a sign byte by
/// [`iq2::KMASK_IQ2XS`]: the +-1 multipliers the float decode applies, then
/// the same signs as 0/-1 masks for the `dp4a` decode, which applies them
/// with `and` since there is no elementwise `i8` multiply.
///
/// IQ2_XXS's sign bytes are [`iq2::KSIGNS_IQ2XS`], indexed by parity, and
/// IQ2_S's every byte value, since it stores the byte itself.
#[cfg(feature = "cuda")]
pub(crate) fn packed_signs(bytes: impl Iterator<Item = u8> + Clone) -> Vec<i8> {
    let expand = |neg: i8, pos: i8| {
        bytes.clone().flat_map(move |b| iq2::KMASK_IQ2XS.map(|bit| if b & bit != 0 { neg } else { pos }))
    };
    expand(-1, 1).chain(expand(-1, 0)).collect()
}
