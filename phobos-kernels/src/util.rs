use std::path::PathBuf;

/// Where compiled PTX is cached: `PHOBOS_KERNEL_CACHE_DIR`, or
/// `~/.phobos/kernel-cache`. `None` when the variable is set but empty, which
/// is how caching is turned off. Lives here rather than in `cache` so the
/// `cache` example can find the directory without a CUDA build.
pub fn kernel_cache_dir() -> Option<PathBuf> {
    match std::env::var("PHOBOS_KERNEL_CACHE_DIR") {
        Ok(dir) if dir.is_empty() => None,
        Ok(dir) => Some(PathBuf::from(dir)),
        Err(_) => std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(|home| PathBuf::from(home).join(".phobos").join("kernel-cache")),
    }
}

/// `x` rounded up to a whole number of `tile`s.
pub fn round_up(x: usize, tile: usize) -> usize {
    x.div_ceil(tile) * tile
}

/// Zero-pad a row-major `[r, c]` matrix into `[rp, cp]`.
///
/// Zero-padding the contraction axis contributes zero terms to the dot, and the
/// padded output rows and columns are discarded, so a kernel that only handles
/// whole tiles still computes the right `[r, c]`.
pub fn pad(src: &[f32], r: usize, c: usize, rp: usize, cp: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rp * cp];
    for i in 0..r {
        out[i * cp..i * cp + c].copy_from_slice(&src[i * c..i * c + c]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_up_to_whole_tiles() {
        assert_eq!(round_up(0, 32), 0);
        assert_eq!(round_up(1, 32), 32);
        assert_eq!(round_up(32, 32), 32);
        assert_eq!(round_up(33, 32), 64);
    }

    #[test]
    fn pad_keeps_rows_and_zeroes_the_rest() {
        // [2, 2] into [3, 4]: each row keeps its two values, the rest is zero.
        let out = pad(&[1.0, 2.0, 3.0, 4.0], 2, 2, 3, 4);
        assert_eq!(
            out,
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }
}
