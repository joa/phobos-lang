// The bound a 32-bit index puts on a tensor, shared by every launcher.

/// Fails unless a tensor of extents `dims` is addressable at a 32-bit index:
/// every extent non-negative and the element count at most `i32::MAX`, so no
/// size, stride or linear offset a kernel forms wraps.
///
/// A kernel compiled at 64-bit indices could address more, but the launch
/// ABIs pack every size as a 32-bit word, so the bound holds for both.
pub fn check_extent(dims: &[i64]) -> anyhow::Result<()> {
    let elements = dims
        .iter()
        .try_fold(1i64, |n, &d| if d < 0 { None } else { n.checked_mul(d) })
        .filter(|&n| n <= i64::from(i32::MAX));
    anyhow::ensure!(
        elements.is_some(),
        "a tensor of extents {dims:?} is past the {} elements a 32-bit index addresses",
        i32::MAX
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_extent;

    #[test]
    fn the_bound_is_the_element_count_not_each_extent() {
        // An LM head of 2^31 elements, each extent well inside i32.
        assert!(check_extent(&[1 << 18, 1 << 13]).is_err());
        assert!(check_extent(&[1 << 18, (1 << 13) - 1]).is_ok());
        assert!(check_extent(&[i64::from(i32::MAX)]).is_ok());
        assert!(check_extent(&[i64::from(i32::MAX) + 1]).is_err());
        assert!(check_extent(&[-1, 4]).is_err());
        assert!(check_extent(&[i64::MAX, i64::MAX]).is_err());
    }
}
