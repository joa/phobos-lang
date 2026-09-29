// The moon splash shown at startup.
//
// The image is the README's, kept as a text file beside this module. It uses
// the ASCII density ramp `.:-=+*#%@`, faintest to solidest, and each character
// is coloured by its place on that ramp.
//
// Shown once at startup. `p` replays it.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::theme;
use super::view::View;

/// The image, as the README prints it.
const MOON: &str = include_str!("phobos.txt");

/// Frames the splash lasts, about two and a half seconds.
pub(super) const FRAMES: u64 = 75;

/// Frames the reveal takes, the rest being the image held lit.
const REVEAL: u64 = 45;

/// How far ahead of the revealed edge the bright band reaches.
const BAND: f32 = 2.5;

/// The shading characters, faintest first. Anything else, such as lettering
/// in the image, is drawn at full brightness.
const RAMP: [char; 9] = ['.', ':', '-', '=', '+', '*', '#', '%', '@'];

/// Where `ch` sits on the ramp, 0.0 to 1.0, or `None` for a space.
fn weight(ch: char) -> Option<f32> {
    if ch == ' ' {
        return None;
    }
    match RAMP.iter().position(|&c| c == ch) {
        Some(at) => Some((at + 1) as f32 / RAMP.len() as f32),
        // Lettering in the image.
        None => Some(1.0),
    }
}

/// Rows and columns the image needs, plus a line of air under it.
pub(super) fn size() -> (u16, u16) {
    let rows = MOON.lines().count() as u16 + 2;
    let cols = MOON.lines().map(str::len).max().unwrap_or(0) as u16;
    (cols, rows)
}

/// Whether the whole image fits. It is never drawn cropped.
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
            // Lit by the ramp, and brighter in a band just behind the edge.
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
        // Lettering is not on the ramp and is drawn at full brightness.
        assert_eq!(weight('P'), Some(1.0));
    }

    /// The image lives both here and in the README, and neither is generated
    /// from the other. This keeps them in sync.
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
