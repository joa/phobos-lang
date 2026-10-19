// The rotary embedding's angle table.

use std::cell::RefCell;

use anyhow::{Result, ensure};

use super::Uploads;
use crate::backend::{Backend, Buf};

/// Rotary cosines and sines by absolute position, `[positions, rope_dim]`: a
/// row's cosines, then its sines. Built on the host because the language has
/// no sine, and the hardware's approximate one loses accuracy at large
/// positions. Precomputing also makes it a constant at decode.
pub(crate) struct RopeTable {
    rope_dim: usize,
    freq_base: f32,
    angles: RefCell<Vec<f32>>,
}

impl RopeTable {
    pub(crate) fn new(rope_dim: usize, freq_base: f32) -> RopeTable {
        RopeTable {
            rope_dim,
            freq_base,
            angles: RefCell::new(Vec::new()),
        }
    }

    /// The table, extended to cover `positions` and uploaded. The key includes
    /// the length, so growth makes a new constant instead of mutating a cached
    /// one. Superseded copies stay resident; doubling bounds them at about one
    /// extra table.
    pub(crate) fn buf(&self, backend: &dyn Backend, positions: usize) -> Result<Buf> {
        let (rope_dim, half) = (self.rope_dim, self.rope_dim / 2);
        let mut table = self.angles.borrow_mut();
        let have = table.len() / rope_dim;
        if have < positions {
            let want = positions.next_power_of_two().max(512);
            table.resize(want * rope_dim, 0.0);
            for p in have..want {
                let row = &mut table[p * rope_dim..][..rope_dim];
                for i in 0..half {
                    let inv_freq = self.freq_base.powf(-(2.0 * i as f32) / rope_dim as f32);
                    let (sin, cos) = (p as f32 * inv_freq).sin_cos();
                    row[i] = cos;
                    row[half + i] = sin;
                }
            }
        }
        let key = format!("rope.{rope_dim}.{}.{}", self.freq_base, table.len());
        backend.constant(&key, &table)
    }

    /// What the table costs once a sequence reaches `positions`. Superseded
    /// copies stay resident, so a run holds about twice the final table.
    pub(crate) fn footprint(&self, into: &mut Uploads, positions: usize) {
        let rows = positions.next_power_of_two().max(512);
        let key = format!("rope.{}.{}", self.rope_dim, self.freq_base);
        into.add(&key, 2 * rows * self.rope_dim * size_of::<f32>(), true);
    }
}

/// Refuses interleaved multimodal rotary sections that a text-only pass does
/// not reproduce.
///
/// qwen35 files declare `rope.dimension_sections`, the rotary pairs given to
/// the time, height, width and extra positions, interleaved as ggml's
/// `IMROPE`. A text token's four positions are `[p, p, p, 0]`, so a pair taken
/// by time, height or width rotates by `p`, which is the plain NeoX rotation
/// [`RopeTable`] builds, while a pair left to the extra position never
/// rotates. Only sections that leave no pair to it are the same pass.
pub(crate) fn check_text_sections(sections: &[i64], rope_dim: usize) -> Result<()> {
    let pairs = rope_dim / 2;
    let &[t, h, w, e] = sections else {
        anyhow::bail!("rope.dimension_sections has {} entries, expected 4", sections.len());
    };
    ensure!(
        sections.iter().all(|&s| s >= 0),
        "rope.dimension_sections {sections:?} has a negative entry"
    );
    let span = t + h + w + e;
    ensure!(
        span > 0 && span <= rope_dim as i64,
        "rope.dimension_sections {sections:?} spans {span} pairs of a {rope_dim}-wide rotation"
    );
    // ggml's interleaved assignment of pair `k`, by its sector in the span.
    let unrotated = (0..pairs as i64).find(|k| {
        let sector = k % span;
        let taken = match sector % 3 {
            0 => sector < 3 * t,
            1 => sector < 3 * h,
            _ => sector < 3 * w,
        };
        !taken
    });
    ensure!(
        unrotated.is_none(),
        "rope.dimension_sections {sections:?} leaves rotary pair {} to the extra position,          which only multimodal input moves; text-only decoding does not model it",
        unrotated.unwrap_or_default()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_text_sections;

    #[test]
    fn the_sections_every_qwen35_file_ships_are_plain_rotation_for_text() {
        check_text_sections(&[11, 11, 10, 0], 64).unwrap();
    }

    #[test]
    fn sections_that_leave_a_pair_unrotated_are_refused() {
        // A pair given to the extra position, and a span short of the pairs
        // with too few width sectors to cover its tail.
        for sections in [[11, 11, 9, 1], [8, 12, 12, 0], [16, 16, 0, 0]] {
            let err = check_text_sections(&sections, 64).unwrap_err().to_string();
            assert!(err.contains("extra position"), "{sections:?}: {err}");
        }
    }

    #[test]
    fn malformed_sections_are_refused() {
        assert!(check_text_sections(&[11, 11, 10], 64).is_err());
        assert!(check_text_sections(&[0, 0, 0, 0], 64).is_err());
        assert!(check_text_sections(&[-1, 11, 10, 0], 64).is_err());
        assert!(check_text_sections(&[40, 40, 40, 0], 64).is_err());
    }
}
