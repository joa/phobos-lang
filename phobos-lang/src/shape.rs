// Shape facts and tiling decisions the AST-to-IR build and the emitter both
// make. One copy, so the two can never disagree about how a matmul tiles.

/// MLIR's ShapedType::kDynamic.
pub(crate) const DYN: i64 = i64::MIN;

/// The (WM, WN) warp block of the matmul's 16x16-fragment grid, or None when
/// the shape doesn't split into whole fragments owned by whole warps. Picks
/// the factorization with the fewest fragments loaded per k-step, breaking
/// ties toward wider warp blocks for contiguous b reads and epilogue stores.
pub(crate) fn wmma_plan(m: i64, n: i64, kk: i64, cta_threads: i64) -> Option<(i64, i64)> {
    if m == DYN || n == DYN || kk == DYN || m % 16 != 0 || n % 16 != 0 || kk % 16 != 0 {
        return None;
    }

    let warps = cta_threads / 32;
    let (gm, gn) = (m / 16, n / 16);

    (1..=warps)
        .filter(|wm| warps % wm == 0)
        .map(|wm| (wm, warps / wm))
        .filter(|&(wm, wn)| gm % wm == 0 && gn % wn == 0)
        .min_by_key(|&(wm, wn)| (gm / wm + gn / wn, gm / wm))
}

/// Register sub-tile extents (TM, TN) for an mxn matmul output: the largest
/// of 8x8 or 8x4 whose sub-tile grid keeps at least one sub-tile per CTA
/// thread, else the largest of 4, 2 or 1 dividing each extent. 8x8 needs an
/// m*n >= 128x128 output: its shared accumulator tile only fits the CTA
/// budget through the register-accumulator fusion.
pub(crate) fn sub_tile(m: i64, n: i64, cta_threads: i64) -> (i64, i64) {
    for (tm, tn) in [(8, 8), (8, 4)] {
        if m % tm == 0 && n % tn == 0 && (m / tm) * (n / tn) >= cta_threads {
            return (tm, tn);
        }
    }
    (sub_extent(m), sub_extent(n))
}

/// Largest register sub-tile extent that divides d.
fn sub_extent(d: i64) -> i64 {
    [4, 2].into_iter().find(|c| d % c == 0).unwrap_or(1)
}

/// Lane grid (lm x ln, lm*ln = 32) for warp tiling: each warp owns an
/// (lm*TM)x(ln*TN) tile, lanes row-major inside it. Picks the factorization
/// with the fewest distinct shared reads per k-step (WM + WN), breaking ties
/// toward wider WN so the warp's b reads stay contiguous. None when no
/// factorization divides the sub-tile grid; the caller falls back to a flat
/// per-thread distribution.
pub(crate) fn lane_grid(tiles_m: i64, tiles_n: i64, tm: i64, tn: i64) -> Option<(i64, i64)> {
    [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]
        .into_iter()
        .filter(|&(lm, ln)| tiles_m % lm == 0 && tiles_n % ln == 0)
        .min_by_key(|&(lm, ln)| (lm * tm + ln * tn, lm))
}
