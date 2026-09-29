// The grouped row layout the raw-format device kernels read.

/// Rows per group. A warp covers eight columns per block, so eight rows'
/// blocks sit side by side.
pub const RAW_GROUP: usize = 8;

/// Columns a grouped upload is padded to: the widest decode tile.
pub const RAW_GROUP_PAD: usize = 64;

/// Reorders `[n][nb][unit]` rows into `[n' / 8][nb][8][unit]`, where `n'` is
/// `n` rounded up to a multiple of [`RAW_GROUP_PAD`]. Padding rows are zero.
pub fn group_rows<T: Copy + Default>(rows: &[T], n: usize, nb: usize, unit: usize) -> Vec<T> {
    let mut out = vec![T::default(); grouped_len(n, nb, unit)];
    group_rows_into(rows, n, nb, unit, &mut out);
    out
}

/// Elements [`group_rows`] produces for `n` rows of `nb` units.
pub fn grouped_len(n: usize, nb: usize, unit: usize) -> usize {
    n.div_ceil(RAW_GROUP_PAD) * RAW_GROUP_PAD * nb * unit
}

/// [`group_rows`] into a caller-owned buffer of [`grouped_len`] elements.
/// Padding rows are never written, so zero the buffer once before first use.
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
