// The splash shown at startup: the mark drawn as if by a pen over a night
// sky coming out, then the wordmark and the line the website leads with.
//
// Shown once at startup. `p` replays it.

use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::font;
use super::mark::Part;
use super::theme;
use super::view::View;

/// Frames the splash lasts, four seconds.
pub(super) const FRAMES: u64 = 120;

/// The line under the wordmark, the website's headline.
const HEADLINE: &str = "Tile-based GPU kernel language & LLM inference engine";

/// Rows the mark is drawn in, at least and at most.
const MARK_ROWS: std::ops::RangeInclusive<u16> = 9..=24;

/// Rows below the mark: air, the wordmark, air, the headline, the version.
const BELOW: u16 = font::HEIGHT as u16 + 4;

/// The rows the mark gets in `area`, or `None` if the splash does not fit.
/// The mark is a little under three times as wide as it is tall.
fn mark_rows(area: Rect) -> Option<u16> {
    let by_height = area.height.saturating_sub(BELOW + 1);
    let by_width = area.width.saturating_sub(4) * 10 / 28;
    let rows = by_height.min(by_width).min(*MARK_ROWS.end());
    let wide_enough = area.width as usize >= HEADLINE.len() + 4;
    (rows >= *MARK_ROWS.start() && wide_enough).then_some(rows)
}

/// Whether the splash fits. It is never drawn cropped.
pub(super) fn fits(area: Rect) -> bool {
    mark_rows(area).is_some()
}

/// Smoothstep of `t`, clamped to 0.0 to 1.0.
fn ease(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Progress of a stage running from frame `from` for `frames`.
fn stage(elapsed: u64, from: u64, frames: u64) -> f32 {
    ease((elapsed as f32 - from as f32) / frames as f32)
}

pub(super) fn draw(frame: &mut Frame, view: &mut View, area: Rect) {
    let Some(rows) = mark_rows(area) else {
        return;
    };
    // Counts up while `view.splash` counts down, so the drawing runs forwards.
    let elapsed = FRAMES.saturating_sub(view.splash);
    view.sky.clear_lane(0..0);
    view.sky.resize(area.width, area.height);
    let lights = view.sky.lights(view.frame);
    let mark = view.mark(rows);
    let word_width = font::width("Phobos") as u16;

    let total = mark.height + BELOW;
    let top = area.y + area.height.saturating_sub(total) / 2;
    let mark_left = area.x + area.width.saturating_sub(mark.width) / 2;
    let word_top = top + mark.height + 1;
    let word_left = area.x + area.width.saturating_sub(word_width) / 2;
    let headline_y = word_top + font::HEIGHT as u16 + 1;
    let headline_left = area.x + area.width.saturating_sub(HEADLINE.len() as u16) / 2;
    let buf = frame.buffer_mut();

    // The sky, coming out over the first two thirds of a second. The moon
    // covers the stars behind it, and the lettering's rows stay clear.
    let dusk = stage(elapsed, 0, 20);
    for (x, y, glyph, color) in lights {
        let (x, y) = (area.x + x, area.y + y);
        let on_moon = x >= mark_left
            && y >= top
            && (mark.inside(x - mark_left, y - top)
                || mark
                    .strokes
                    .iter()
                    .any(|s| s.x == x - mark_left && s.y == y - top));
        let lettering = (word_top..=headline_y + 1).contains(&y)
            && x + 2 >= headline_left
            && x < headline_left + HEADLINE.len() as u16 + 2;
        if !on_moon && !lettering {
            put(buf, x, y, glyph, theme::mix(theme::BG, color, dusk));
        }
    }

    // The moon's face, a faint wash of the brand inside the outline, as the
    // logo's fill is.
    let wash = theme::mix(theme::BG, theme::BRAND, 0.08 * stage(elapsed, 50, 30));
    for y in 0..mark.height {
        for x in 0..mark.width {
            if mark.inside(x, y) {
                bg(buf, mark_left + x, top + y, wash);
            }
        }
    }

    // The outlines, each drawn along its length: the body, then the crater,
    // then the pit and the dot.
    let outline = theme::mix(theme::BRAND_DIM, theme::TEXT, 0.6);
    let body = stage(elapsed, 8, 42);
    let crater = stage(elapsed, 38, 28);
    let pits = stage(elapsed, 62, 10);
    for stroke in &mark.strokes {
        let (drawn, rest) = match stroke.part {
            Part::Body => (body, outline),
            Part::Crater => (crater, theme::BRAND),
            Part::Pit => (pits, outline),
        };
        if drawn <= 0.0 || stroke.at > drawn {
            continue;
        }
        // The pen's tip glows, and the line settles behind it.
        let tip = 1.0 - ((drawn - stroke.at) / 0.12).clamp(0.0, 1.0);
        let lit = if drawn >= 1.0 {
            rest
        } else {
            theme::mix(rest, theme::BRAND_HI, tip)
        };
        put(buf, mark_left + stroke.x, top + stroke.y, stroke.glyph, lit);
    }

    // The wordmark, revealed left to right, with a highlight passing over it
    // once it is all out.
    let reveal = stage(elapsed, 62, 18) * (word_width as f32 + 1.0);
    let sweep = (elapsed as f32 - 82.0) * 1.6 - 6.0;
    for (row, line) in font::render("Phobos").iter().enumerate() {
        for (col, cell) in line.chars().enumerate() {
            if cell == ' ' || col as f32 >= reveal {
                continue;
            }
            let heat = (1.0 - (col as f32 - sweep).abs() / 8.0).clamp(0.0, 1.0);
            let color = theme::mix(theme::BRAND, theme::BRAND_HI, heat);
            put(
                buf,
                word_left + col as u16,
                word_top + row as u16,
                cell,
                color,
            );
        }
    }

    // The headline and the version fade in last.
    let words = stage(elapsed, 80, 20);
    let version = format!("v{}", env!("CARGO_PKG_VERSION"));
    let version_left = area.x + area.width.saturating_sub(version.len() as u16) / 2;
    write(
        buf,
        headline_left,
        headline_y,
        HEADLINE,
        theme::mix(theme::BG, theme::TEXT, words),
    );
    write(
        buf,
        version_left,
        headline_y + 1,
        &version,
        theme::mix(theme::BG, theme::MUTED, words),
    );
}

pub(super) fn put(buf: &mut Buffer, x: u16, y: u16, glyph: char, color: Color) {
    if buf.area.contains((x, y).into()) {
        buf[(x, y)].set_char(glyph).set_fg(color);
    }
}

pub(super) fn bg(buf: &mut Buffer, x: u16, y: u16, color: Color) {
    if buf.area.contains((x, y).into()) {
        buf[(x, y)].set_bg(color);
    }
}

fn write(buf: &mut Buffer, x: u16, y: u16, text: &str, color: Color) {
    for (i, ch) in text.chars().enumerate() {
        put(buf, x + i as u16, y, ch, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::mark::Mark;

    #[test]
    fn the_common_terminal_gets_the_splash() {
        assert!(fits(Rect::new(0, 0, 80, 24)));
        assert!(fits(Rect::new(0, 0, 240, 70)));
    }

    #[test]
    fn a_short_or_narrow_terminal_gets_none() {
        assert!(!fits(Rect::new(0, 0, 80, 17)));
        assert!(!fits(Rect::new(0, 0, 50, 40)));
    }

    #[test]
    fn the_mark_fits_where_the_splash_says_it_does() {
        for (width, height) in [(60, 22), (80, 24), (120, 40), (240, 70)] {
            let area = Rect::new(0, 0, width, height);
            let Some(rows) = mark_rows(area) else {
                continue;
            };
            let mark = Mark::new(rows);
            assert!(mark.width <= width, "{width}x{height}: {} wide", mark.width);
            assert!(mark.height + BELOW <= height, "{width}x{height}");
        }
    }
}
