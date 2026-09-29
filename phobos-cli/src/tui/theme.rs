// The dashboard's palette and its glyph vocabulary.
//
// Colors are RGB rather than the sixteen named ones, so the dashboard looks
// the same under any terminal theme. Non-ASCII glyphs are written as escapes
// to keep the source ASCII.

use ratatui::style::{Color, Modifier, Style};

/// Near black, faintly green, so the colors do not depend on the terminal's
/// background.
pub const BG: Color = Color::Rgb(8, 12, 10);

/// The primary: a bright phosphor green.
pub const GREEN: Color = Color::Rgb(57, 255, 106);
/// The primary, at rest.
pub const GREEN_DIM: Color = Color::Rgb(31, 138, 76);
/// The primary, barely lit. What the digital rain fades into.
pub const GREEN_FAINT: Color = Color::Rgb(18, 58, 34);

pub const CYAN: Color = Color::Rgb(34, 211, 238);
pub const MAGENTA: Color = Color::Rgb(255, 47, 191);
pub const AMBER: Color = Color::Rgb(255, 182, 66);
pub const RED: Color = Color::Rgb(255, 77, 94);

/// Body text.
pub const TEXT: Color = Color::Rgb(200, 247, 212);
/// Labels and units, quieter than the figures beside them.
pub const MUTED: Color = Color::Rgb(90, 130, 105);
/// A panel's edge when nothing is happening in it.
pub const BORDER: Color = Color::Rgb(31, 59, 42);

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

/// Green while there is room, amber as it runs out, red once it has.
///
/// The thresholds sit below full, since paging starts well before the last
/// byte is used.
pub fn pressure(ratio: f64) -> Color {
    match ratio {
        r if r >= 0.92 => RED,
        r if r >= 0.78 => AMBER,
        _ => GREEN,
    }
}

/// Solid block, used by both bars and the block font.
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
        assert_eq!(mix(GREEN, RED, 0.0), GREEN);
        assert_eq!(mix(GREEN, RED, 1.0), RED);
        // Out of range is clamped rather than extrapolated.
        assert_eq!(mix(GREEN, RED, -3.0), GREEN);
        assert_eq!(mix(GREEN, RED, 9.0), RED);
    }

    #[test]
    fn pressure_climbs_through_the_three_colors() {
        assert_eq!(pressure(0.10), GREEN);
        assert_eq!(pressure(0.80), AMBER);
        assert_eq!(pressure(0.99), RED);
    }
}
