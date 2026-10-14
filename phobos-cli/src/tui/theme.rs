// The dashboard's palette and its glyph vocabulary.
//
// The palette is the website's night sky, `phobos_base::palette`: a
// near-black blue sky, starlight for text, a pale blue brand, a violet nebula
// and Mars for the warm end. Colors are RGB rather than the sixteen named
// ones, so the dashboard looks the same under any terminal theme. Non-ASCII
// glyphs are written as escapes to keep the source ASCII.

use phobos_base::palette::{self, Rgb};
use ratatui::style::{Color, Modifier, Style};

const fn rgb((r, g, b): Rgb) -> Color {
    Color::Rgb(r, g, b)
}

/// The night sky, so the colors do not depend on the terminal's background.
pub const BG: Color = rgb(palette::SKY);

pub const BRAND: Color = rgb(palette::BRAND);
/// The primary for figures and keys.
pub const BRAND_HI: Color = rgb(palette::BRAND_HI);
pub const BRAND_DIM: Color = rgb(palette::BRAND_DIM);
pub const FAINT: Color = rgb(palette::FAINT);

pub const NEBULA: Color = rgb(palette::NEBULA);
pub const GOLD: Color = rgb(palette::GOLD);
pub const MARS: Color = rgb(palette::MARS);

/// Body text.
pub const TEXT: Color = rgb(palette::STAR);
/// Labels and units, quieter than the figures beside them.
pub const MUTED: Color = rgb(palette::DUST);
/// A panel's edge when nothing is happening in it.
pub const BORDER: Color = rgb(palette::SKY_LINE);

pub fn text() -> Style {
    Style::default().fg(TEXT)
}

pub fn muted() -> Style {
    Style::default().fg(MUTED)
}

pub fn accent(color: Color) -> Style {
    Style::default().fg(color).add_modifier(Modifier::BOLD)
}

/// Blend two colors, `t` running 0.0 to 1.0 from `from` to `to`. `t` is
/// clamped, so a caller may pass a raw ratio.
pub fn mix(from: Color, to: Color, t: f32) -> Color {
    let (Color::Rgb(r1, g1, b1), Color::Rgb(r2, g2, b2)) = (from, to) else {
        return to;
    };
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
    Color::Rgb(lerp(r1, r2), lerp(g1, g2), lerp(b1, b2))
}

/// The brand while there is room, gold as it runs out, Mars once it has.
///
/// The thresholds sit below full, since paging starts well before the last
/// byte is used.
pub fn pressure(ratio: f64) -> Color {
    match ratio {
        r if r >= 0.92 => MARS,
        r if r >= 0.78 => GOLD,
        _ => BRAND_HI,
    }
}

/// Solid block, for bars and strips.
pub const FULL: char = '\u{2588}';

/// Eighth-height blocks, shortest first, for the spark plots.
pub const BARS: [char; 8] = [
    '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}',
];

/// Eighth-width blocks, narrowest first, for a horizontal gauge whose end
/// falls inside a cell.
pub const SLICES: [char; 8] = [
    '\u{258f}', '\u{258e}', '\u{258d}', '\u{258c}', '\u{258b}', '\u{258a}', '\u{2589}', '\u{2588}',
];

/// Shaded blocks, lightest first, for the unfilled part of a gauge.
pub const SHADES: [char; 3] = ['\u{2591}', '\u{2592}', '\u{2593}'];

/// Filled and hollow circles, for the status light and the legend dots.
pub const DOT: char = '\u{25cf}';
pub const DOT_HOLLOW: char = '\u{25cb}';

/// Right-pointing triangle, marking the active row.
pub const CARET: char = '\u{25b6}';

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixing_lands_on_the_ends() {
        assert_eq!(mix(BRAND_HI, MARS, 0.0), BRAND_HI);
        assert_eq!(mix(BRAND_HI, MARS, 1.0), MARS);
        // Out of range is clamped rather than extrapolated.
        assert_eq!(mix(BRAND_HI, MARS, -3.0), BRAND_HI);
        assert_eq!(mix(BRAND_HI, MARS, 9.0), MARS);
    }

    #[test]
    fn pressure_climbs_through_the_three_colors() {
        assert_eq!(pressure(0.10), BRAND_HI);
        assert_eq!(pressure(0.80), GOLD);
        assert_eq!(pressure(0.99), MARS);
    }
}
