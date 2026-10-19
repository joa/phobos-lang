// A tile operand's index words in phobos-mlir's exploded-memref ABI, which
// the pod launches at 32-bit indices.

use anyhow::{Context, Result};

/// The offset, sizes and row-major strides of a tile of `shape`. Fails on a
/// tile a 32-bit index cannot address, rather than letting a size or an
/// offset wrap.
pub(crate) fn memref_words(shape: &[u64]) -> Result<Vec<i32>> {
    let dims = shape
        .iter()
        .map(|&s| i64::try_from(s))
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("a tile of shape {shape:?} is past a 64-bit extent"))?;
    phobos_base::index::check_extent(&dims)?;
    let mut words = Vec::with_capacity(1 + 2 * dims.len());
    words.push(0); // offset
    // In range: every size and stride is at most the element count.
    words.extend(dims.iter().map(|&d| d as i32));
    let mut strides = vec![0i32; dims.len()];
    let mut stride = 1i64;
    for (i, &d) in dims.iter().enumerate().rev() {
        strides[i] = stride as i32;
        stride *= d;
    }
    words.extend(strides);
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::memref_words;

    #[test]
    fn a_tile_packs_its_offset_sizes_and_strides() {
        assert_eq!(memref_words(&[2, 3, 4]).unwrap(), vec![0, 2, 3, 4, 12, 4, 1]);
    }

    #[test]
    fn a_tile_past_a_32_bit_index_is_refused() {
        assert!(memref_words(&[1 << 16, 1 << 15]).is_err());
        assert!(memref_words(&[u64::MAX]).is_err());
    }
}
