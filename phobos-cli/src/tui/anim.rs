// The moving parts: bars, plots, and easing.
//
// Nothing here reads the engine. Everything is a function of the frame
// counter and the values the panels pass in, so rendering is deterministic
// and testable.

use ratatui::style::Color;
use ratatui::text::Span;

use super::theme;

/// Move `current` a fraction of the way to `target`.
///
/// `rate` is the fraction of the gap closed per frame.
pub fn ease(current: f64, target: f64, rate: f64) -> f64 {
    if !current.is_finite() {
        return target;
    }
    let next = current + (target - current) * rate.clamp(0.0, 1.0);
    // Snap when close, or a decaying value never quite reaches its target.
    if (target - next).abs() < 1e-3 {
        target
    } else {
        next
    }
}

/// A 0.0 to 1.0 triangle wave of `period` frames, for anything that breathes.
pub fn pulse(frame: u64, period: u64) -> f32 {
    let period = period.max(1);
    let phase = (frame % period) as f32 / period as f32;
    if phase < 0.5 {
        phase * 2.0
    } else {
        (1.0 - phase) * 2.0
    }
}

/// A one-line bar plot of `values`, newest at the right.
///
/// Scaled to the largest value shown rather than a fixed ceiling, since it
/// shows shape and the absolute figure is printed beside it.
pub fn spark(values: &[f64], width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let start = values.len().saturating_sub(width);
    let window = &values[start..];
    let max = window.iter().copied().fold(0.0f64, f64::max);
    let mut out = " ".repeat(width.saturating_sub(window.len()));
    for &value in window {
        out.push(if max <= 0.0 {
            theme::BARS[0]
        } else {
            let step = (value / max * (theme::BARS.len() - 1) as f64).round() as usize;
            theme::BARS[step.min(theme::BARS.len() - 1)]
        });
    }
    out
}

/// A horizontal gauge `width` cells wide, `ratio` of it filled.
///
/// The last filled cell is a part-width block, so the gauge moves smoothly.
/// The unfilled track is shaded so it is visible at any fill.
pub fn bar(ratio: f64, width: usize) -> String {
    let ratio = ratio.clamp(0.0, 1.0);
    let eighths = (ratio * width as f64 * 8.0).round() as usize;
    // A nonzero share that rounds to nothing still gets the narrowest slice.
    let eighths = if eighths == 0 && ratio > 0.0 {
        1
    } else {
        eighths
    };
    let full = eighths / 8;
    let part = eighths % 8;
    let mut out = String::new();
    for _ in 0..full.min(width) {
        out.push(theme::FULL);
    }
    if full < width && part > 0 {
        out.push(theme::SLICES[part - 1]);
    }
    while out.chars().count() < width {
        out.push(theme::SHADES[0]);
    }
    out
}

/// A bar of `width` cells split into `parts`, each a share of the whole in
/// its own colour. The rest is shaded as the track.
///
/// A nonzero share too small to fill a cell still gets one, so it stays
/// visible. The legend beside it carries the exact figures.
pub fn stacked(parts: &[(f64, Color)], width: usize) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut filled = 0usize;
    for &(share, color) in parts {
        if filled >= width {
            break;
        }
        let want = (share.clamp(0.0, 1.0) * width as f64).round() as usize;
        let cells = want.max(usize::from(share > 0.0)).min(width - filled);
        if cells == 0 {
            continue;
        }
        spans.push(Span::styled(
            theme::FULL.to_string().repeat(cells),
            ratatui::style::Style::default().fg(color),
        ));
        filled += cells;
    }
    if filled < width {
        spans.push(Span::styled(
            theme::SHADES[0].to_string().repeat(width - filled),
            ratatui::style::Style::default().fg(theme::BORDER),
        ));
    }
    spans
}

/// Format a rate for the display font, with at most one decimal place and none
/// from 100 up.
pub fn rate_text(rate: f64) -> String {
    if !rate.is_finite() || rate <= 0.0 {
        return "0".to_string();
    }
    match rate {
        r if r >= 1000.0 => format!("{}", r.round() as u64),
        r if r >= 100.0 => format!("{r:.0}"),
        r => format!("{r:.1}"),
    }
}

/// A color for a figure relative to the run's best: bright at the best, dim
/// as it falls away.
pub fn relative(value: f64, best: f64, color: Color) -> Color {
    if best <= 0.0 || value <= 0.0 {
        return theme::BRAND_DIM;
    }
    theme::mix(theme::BRAND_DIM, color, (value / best) as f32)
}
