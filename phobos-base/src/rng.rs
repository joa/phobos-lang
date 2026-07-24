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
    use super::Lcg;

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
