// The grouped row layout the raw-format device kernels read.

/// Rows a group holds: a warp covers eight columns a block, so eight rows'
/// blocks sit side by side.
pub const RAW_GROUP: usize = 8;

/// Columns a grouped upload is padded to: the widest decode tile.
pub const RAW_GROUP_PAD: usize = 64;

/// `[n][nb][unit]` rows as `[n' / 8][nb][8][unit]`, `n'` the next multiple
/// of [`RAW_GROUP_PAD`], zero-padded.
pub fn group_rows<T: Copy + Default>(rows: &[T], n: usize, nb: usize, unit: usize) -> Vec<T> {
    let mut out = vec![T::default(); grouped_len(n, nb, unit)];
    group_rows_into(rows, n, nb, unit, &mut out);
    out
}

/// Elements [`group_rows`] produces for `n` rows of `nb` units.
pub fn grouped_len(n: usize, nb: usize, unit: usize) -> usize {
    n.div_ceil(RAW_GROUP_PAD) * RAW_GROUP_PAD * nb * unit
}

/// [`group_rows`] into a buffer the caller owns, of [`grouped_len`]
/// elements. Padding rows are left as they are, so a reused buffer wants
/// zeroing once.
pub fn group_rows_into<T: Copy>(rows: &[T], n: usize, nb: usize, unit: usize, out: &mut [T]) {
    assert_eq!(out.len(), grouped_len(n, nb, unit), "grouped buffer is the wrong size");
    for j in 0..n {
        let (group, in_group) = (j / RAW_GROUP, j % RAW_GROUP);
        for b in 0..nb {
            let src = (j * nb + b) * unit;
            let dst = ((group * nb + b) * RAW_GROUP + in_group) * unit;
            out[dst..dst + unit].copy_from_slice(&rows[src..src + unit]);
        }
    }
}
