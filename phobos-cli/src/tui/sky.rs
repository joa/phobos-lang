// The night sky behind the banner and the splash: stars that twinkle and
// drift, nearer ones faster, and now and then a meteor.
//
// Like the rest of the animation it is a function of the frame counter and a
// fixed seed, so a frame is deterministic and testable.

use ratatui::style::Color;

use super::theme;

/// One star in this many cells. Sparse, so the sky stays behind what is drawn
/// over it.
const SPARSITY: u32 = 26;

/// Frames between a meteor's starts, and how many it is in flight.
const METEOR_EVERY: u64 = 260;
const METEOR_FLIGHT: u64 = 26;

/// Cells a meteor's tail reaches behind its head.
const TAIL: i32 = 7;

/// The glyphs a star is drawn in, faintest first.
const GLYPHS: [char; 4] = ['.', '\u{00b7}', '+', '*'];

pub struct Sky {
    width: u16,
    height: u16,
    stars: Vec<Star>,
    /// Columns the sky stays out of, so what is drawn there reads cleanly.
    clear: std::ops::Range<u16>,
}

#[derive(Clone, Copy)]
struct Star {
    /// Position at frame zero, in eighths of a cell so drift is smooth.
    x: u32,
    y: u16,
    /// Brightness, 0.0 to 1.0. Brighter stars are nearer, so they drift
    /// faster.
    magnitude: f32,
    /// Twinkle period and phase, in frames.
    period: u64,
    phase: u64,
    tint: Color,
}

impl Sky {
    pub fn new() -> Sky {
        Sky {
            width: 0,
            height: 0,
            stars: Vec::new(),
            clear: 0..0,
        }
    }

    /// Keep the stars out of `columns`, where something opaque will be drawn.
    pub fn clear_lane(&mut self, columns: std::ops::Range<u16>) {
        self.clear = columns;
    }

    /// Fit the sky to `width` by `height`, rebuilding only on a change.
    pub fn resize(&mut self, width: u16, height: u16) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        let mut rng = Xorshift::new(0x9e37_79b9_7f4a_7c15);
        let count = (width as u32 * height as u32) / SPARSITY;
        self.stars = (0..count)
            .map(|_| {
                // Most stars faint, a few bright.
                let magnitude = (rng.unit() * rng.unit()).sqrt() * 0.9 + 0.1 * rng.unit();
                let tint = match rng.below(9) {
                    0 => theme::BRAND,
                    1 => theme::NEBULA,
                    2 => theme::GOLD,
                    _ => theme::TEXT,
                };
                Star {
                    x: rng.below(width.max(1) as u32) * 8,
                    y: rng.below(height.max(1) as u32) as u16,
                    magnitude,
                    period: 70 + rng.below(110) as u64,
                    phase: rng.below(180) as u64,
                    tint,
                }
            })
            .collect();
    }

    /// Everything lit on `frame`: each star and the meteor's cells, as a
    /// position, a glyph and a colour. The meteor comes last, so it draws over
    /// a star it crosses.
    pub fn lights(&self, frame: u64) -> Vec<(u16, u16, char, Color)> {
        let span = self.width as u64 * 8;
        if span == 0 {
            return Vec::new();
        }
        let mut out: Vec<_> = self
            .stars
            .iter()
            .map(|star| {
                // Leftward drift, nearer stars faster.
                let drift = frame * (1 + (star.magnitude * 3.0) as u64) / 6;
                let x = ((star.x as u64 + span - drift % span) % span / 8) as u16;
                let twinkle = smooth_pulse(frame + star.phase, star.period);
                let light = star.magnitude * (0.55 + 0.45 * twinkle);
                let glyph = GLYPHS[((light * GLYPHS.len() as f32) as usize).min(GLYPHS.len() - 1)];
                let color = theme::mix(theme::BG, star.tint, 0.25 + 0.75 * light);
                (x, star.y, glyph, color)
            })
            .collect();
        out.extend(self.meteor(frame));
        out.retain(|&(x, y, _, _)| !self.clear.contains(&x) && x < self.width && y < self.height);
        out
    }

    /// The meteor's head and tail, while one is in flight.
    ///
    /// It falls from the upper right towards the lower left, two columns for
    /// every row, the slant a `/` draws.
    fn meteor(&self, frame: u64) -> Vec<(u16, u16, char, Color)> {
        let flight = frame % METEOR_EVERY;
        if flight >= METEOR_FLIGHT || self.width < 24 {
            return Vec::new();
        }
        // Where this one starts, fixed per meteor.
        let mut rng = Xorshift::new(frame / METEOR_EVERY + 7);
        let start_x = (self.width / 3) as i32 + rng.below(self.width as u32 * 2 / 3) as i32;
        let start_y = -(rng.below(2) as i32);
        let step = flight as i32;
        let (hx, hy) = (start_x - step * 2, start_y + step);
        (0..=TAIL)
            .filter_map(|behind| {
                let (x, y) = (hx + behind * 2, hy - behind);
                if x < 0 || y < 0 {
                    return None;
                }
                let lit = if behind == 0 {
                    ('*', theme::BRAND_HI)
                } else {
                    let fade = behind as f32 / (TAIL + 1) as f32;
                    ('/', theme::mix(theme::BRAND, theme::BG, fade))
                };
                Some((x as u16, y as u16, lit.0, lit.1))
            })
            .collect()
    }
}

impl Default for Sky {
    fn default() -> Sky {
        Sky::new()
    }
}

/// A 0.0 to 1.0 wave of `period` frames that eases at both ends, for a
/// twinkle rather than a blink.
pub fn smooth_pulse(frame: u64, period: u64) -> f32 {
    let phase = (frame % period.max(1)) as f32 / period.max(1) as f32;
    0.5 - 0.5 * (phase * std::f32::consts::TAU).cos()
}

/// A 64-bit xorshift. Fixed-seeded, so the sky looks the same every run.
pub(super) struct Xorshift(u64);

impl Xorshift {
    pub(super) fn new(seed: u64) -> Xorshift {
        // Spread small seeds, which xorshift takes many steps to mix.
        Xorshift(seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1)
    }

    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(super) fn below(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            0
        } else {
            (self.next() % bound as u64) as u32
        }
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sky_is_sparse_and_stays_lit() {
        let mut sky = Sky::new();
        sky.resize(120, 7);
        for frame in [0, 100, 1000, 100_000] {
            let n = sky.lights(frame).len();
            assert!((10..=60).contains(&n), "{n} stars on frame {frame}");
        }
    }

    #[test]
    fn the_cleared_lane_stays_dark() {
        let mut sky = Sky::new();
        sky.resize(120, 7);
        sky.clear_lane(40..80);
        for frame in 0..400 {
            assert!(
                sky.lights(frame)
                    .iter()
                    .all(|&(x, _, _, _)| !(40..80).contains(&x))
            );
        }
    }

    #[test]
    fn a_meteor_crosses_now_and_then() {
        let mut sky = Sky::new();
        sky.resize(120, 20);
        let heads = |frame| sky.meteor(frame).iter().filter(|l| l.2 == '*').count();
        assert_eq!(heads(5), 1);
        assert_eq!(heads(METEOR_FLIGHT + 5), 0);
    }
}
