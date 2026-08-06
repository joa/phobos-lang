/// SplitMix64, which keeps a run reproducible without an RNG dependency.
///
/// [`Lcg`] is the older of the two and fills test tensors; this one has the
/// better distribution and is what sampling draws from.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> SplitMix64 {
        // An all-zero state degenerates.
        SplitMix64(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The next value uniformly in `[0.0, 1.0)`.
    pub fn next_f32(&mut self) -> f32 {
        // The top 24 bits give a uniform float without bias.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// A linear congruential generator, used to fill tensors with reproducible
/// values. See [`SplitMix64`] for the better-distributed one.
#[derive(Clone, Debug)]
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Lcg {
        Lcg(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// The next value uniformly in `[-1.0, 1.0)`.
    pub fn next_unit_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32 - 1.0
    }

    /// `n` values in `[-1.0, 1.0)`.
    pub fn unit_f32s(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_unit_f32()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Lcg, SplitMix64};

    #[test]
    fn splitmix_is_seed_reproducible() {
        let draw = |seed| {
            let mut rng = SplitMix64::new(seed);
            (0..16).map(|_| rng.next_u64()).collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8));
    }

    #[test]
    fn splitmix_floats_are_unit_interval() {
        let mut rng = SplitMix64::new(0);
        for _ in 0..4096 {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x), "out of range: {x}");
        }
    }

    #[test]
    fn equal_seeds_are_reproducible() {
        assert_eq!(Lcg::new(1).unit_f32s(64), Lcg::new(1).unit_f32s(64));
    }

    #[test]
    fn different_seeds_diverge() {
        assert_ne!(Lcg::new(1).unit_f32s(16), Lcg::new(2).unit_f32s(16));
    }

    #[test]
    fn stays_in_unit_range() {
        let mut lcg = Lcg::new(7);
        for x in lcg.unit_f32s(4096) {
            assert!((-1.0..1.0).contains(&x), "out of range: {x}");
        }
    }

    #[test]
    fn matches_the_hand_rolled_recurrence() {
        // The exact expression every tool inlined, kept as a guard so the
        // shared generator stays byte-compatible with old outputs.
        let mut seed = 1u64;
        let mut lcg = Lcg::new(1);
        for _ in 0..8 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let expected = (seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
            assert_eq!(lcg.next_unit_f32(), expected);
        }
    }
}
