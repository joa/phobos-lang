// The rotary embedding's angle table.

use std::cell::RefCell;

use anyhow::Result;

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
