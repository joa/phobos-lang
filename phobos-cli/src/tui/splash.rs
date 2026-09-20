// The moon, on the way in.
//
// The same image the README carries, kept beside this module as a text file
// so it stays a picture rather than becoming a string literal nobody can
// read. It is drawn in the density characters ASCII art has always used,
// `.:-=+*#%@` from faintest to solidest, which is already a brightness ramp:
// colouring each character by where it sits on that ramp is the whole effect.
//
// Shown once on the way in and then given up, because a cold start has
// minutes of compiling to report and a picture is not what a watcher needs
// for those. `p` puts it back for anyone who wants another look.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::theme;
use super::view::View;

/// The image, as the README prints it.
const MOON: &str = include_str!("phobos.txt");

/// Frames the splash lasts, at the dashboard's frame rate: about two and a
/// half seconds, which is long enough to read and short enough not to be in
/// the way of a load.
pub(super) const FRAMES: u64 = 75;

/// Frames the reveal takes, the rest being the image held lit.
const REVEAL: u64 = 45;

/// How far ahead of the revealed edge the bright band reaches.
const BAND: f32 = 2.5;

/// The characters ASCII art shades with, faintest first. Anything outside
/// this, which is the lettering and the signature worked into the image, is
/// drawn at full brightness.
const RAMP: [char; 9] = ['.', ':', '-', '=', '+', '*', '#', '%', '@'];

/// Where `ch` sits on the ramp, 0.0 to 1.0, or nothing for a space.
fn weight(ch: char) -> Option<f32> {
    if ch == ' ' {
        return None;
    }
    match RAMP.iter().position(|&c| c == ch) {
        Some(at) => Some((at + 1) as f32 / RAMP.len() as f32),
        // Lettering: the wordmark along the bottom, and the initials someone
        // left in the surface.
        None => Some(1.0),
    }
}

/// Rows and columns the image needs, plus a line of air under it.
pub(super) fn size() -> (u16, u16) {
    let rows = MOON.lines().count() as u16 + 2;
    let cols = MOON.lines().map(str::len).max().unwrap_or(0) as u16;
    (cols, rows)
}

/// Whether there is room to draw it at all. A terminal too short for the
/// whole moon gets no splash rather than a cropped one.
pub(super) fn fits(area: Rect) -> bool {
    let (cols, rows) = size();
    area.width >= cols && area.height >= rows
}

/// Draw the image, revealed from the top down.
pub(super) fn draw(frame: &mut Frame, view: &View, area: Rect) {
    let rows: Vec<&str> = MOON.lines().collect();
    let (cols, _) = size();
    let left = area.width.saturating_sub(cols) / 2;
    let top = area.height.saturating_sub(rows.len() as u16 + 2) / 2;

    // Counts up while `view.splash` counts down, so the reveal runs forwards.
    let elapsed = FRAMES.saturating_sub(view.splash);
    let edge = (elapsed as f32 / REVEAL as f32).min(1.0) * rows.len() as f32;

    let mut lines = vec![Line::raw(""); top as usize];
    for (y, row) in rows.iter().enumerate() {
        // How far this row is behind the edge: negative is not yet reached.
        let behind = edge - y as f32;
        if behind <= 0.0 {
            lines.push(Line::raw(""));
            continue;
        }
        let mut spans = vec![Span::raw(" ".repeat(left as usize))];
        for ch in row.chars() {
            let Some(weight) = weight(ch) else {
                spans.push(Span::raw(" "));
                continue;
            };
            // Lit by the ramp, and brighter still just behind the edge, so
            // the reveal reads as a band passing over the surface.
            let heat = ((BAND - behind) / BAND).clamp(0.0, 1.0);
            let lit = theme::mix(theme::GREEN_FAINT, theme::GREEN, weight);
            spans.push(Span::styled(
                ch.to_string(),
                Style::default().fg(theme::mix(lit, theme::CYAN, heat)),
            ));
        }
        lines.push(Line::from(spans));
    }

    // The name and the version, once the moon is all the way out.
    if edge >= rows.len() as f32 {
        let tag = format!(
            "tile-based GPU kernel language   v{}",
            env!("CARGO_PKG_VERSION")
        );
        let pad = area.width.saturating_sub(tag.len() as u16) / 2;
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::raw(" ".repeat(pad as usize)),
            Span::styled(
                tag,
                Style::default()
                    .fg(theme::GREEN_DIM)
                    .add_modifier(Modifier::ITALIC),
            ),
        ]));
    }
    frame.render_widget(Paragraph::new(lines), area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_image_is_ascii_and_rectangular_enough_to_place() {
        let (cols, rows) = size();
        assert!(rows > 20 && cols > 60, "{cols}x{rows} is not the moon");
        assert!(
            MOON.is_ascii(),
            "the image has to stay ASCII to render in any terminal"
        );
    }

    #[test]
    fn the_ramp_runs_from_faint_to_solid() {
        assert_eq!(weight(' '), None);
        let faint = weight('.').unwrap();
        let solid = weight('@').unwrap();
        assert!(faint < solid, "{faint} should be fainter than {solid}");
        // Lettering is off the ramp, at full brightness.
        assert_eq!(weight('P'), Some(1.0));
    }

    /// The image lives in two places, here and in the README, and neither is
    /// generated from the other. This is what keeps them the same picture.
    #[test]
    fn the_readme_carries_the_same_moon() {
        const README: &str = include_str!("../../../README.md");
        let fences: Vec<usize> = README
            .lines()
            .enumerate()
            .filter(|(_, line)| line.starts_with("```"))
            .map(|(at, _)| at)
            .collect();
        let (open, close) = (fences[fences.len() - 2], fences[fences.len() - 1]);
        let all: Vec<&str> = README.lines().collect();
        let in_readme = &all[open + 1..close];

        assert_eq!(
            in_readme,
            MOON.lines().collect::<Vec<_>>(),
            "the README's moon and this one have drifted apart"
        );
    }

    #[test]
    fn a_short_terminal_gets_no_splash() {
        let (cols, rows) = size();
        assert!(fits(Rect::new(0, 0, cols, rows)));
        assert!(!fits(Rect::new(0, 0, cols, rows - 1)));
        assert!(
            !fits(Rect::new(0, 0, 80, 24)),
            "the common size is too short"
        );
    }
}
