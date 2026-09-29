// The moving parts: falling glyphs, a block font, bars, and easing.
//
// Nothing here reads the engine. Everything is a function of the frame
// counter and the values the panels pass in, so rendering is deterministic
// and testable.

use ratatui::style::Color;
use ratatui::text::Span;

use super::theme;

/// The glyphs the rain falls in: single width and present in any terminal
/// font.
const RAIN_GLYPHS: &[u8] = b"0123456789ABCDEF<>[]{}/\\|=+*-";

/// Frames a drop waits between steps, at its slowest and at its fastest. Each
/// column picks one and keeps it, which gives the field depth.
const SLOWEST: u64 = 9;
const FASTEST: u64 = 3;

/// How far behind the head a drop stays lit.
const TAIL: i32 = 5;

/// How many columns in sixteen carry a drop. Kept sparse so the backdrop does
/// not compete with what is drawn over it.
const DENSITY: u32 = 5;

/// A field of falling glyphs, one drop per column.
///
/// Rebuilt when its area changes, so a resize leaves no drops outside the
/// panel.
pub struct Rain {
    width: u16,
    height: u16,
    /// One entry per column, `None` where no drop falls.
    columns: Vec<Option<Column>>,
    /// Columns the rain stays out of, so text drawn there reads cleanly.
    clear: std::ops::Range<u16>,
}

#[derive(Clone, Copy)]
struct Column {
    /// The head's row. Starts above the top, at a random offset, so columns
    /// do not all fall together.
    head: i32,
    period: u64,
    /// The column's starting glyph. Each row offsets from it, so a column's
    /// pattern stays stable as the head moves.
    seed: u32,
}

impl Rain {
    pub fn new() -> Rain {
        Rain {
            width: 0,
            height: 0,
            columns: Vec::new(),
            clear: 0..0,
        }
    }

    /// Keep the drops out of `columns`, where something opaque will be drawn.
    pub fn clear_lane(&mut self, columns: std::ops::Range<u16>) {
        self.clear = columns;
    }

    /// Fit the field to `width` by `height`, rebuilding only on a change.
    pub fn resize(&mut self, width: u16, height: u16) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        let mut rng = Xorshift::new(0x9e3779b97f4a7c15);
        self.columns = (0..width as usize)
            .map(|_| {
                (rng.below(16) < DENSITY).then(|| Column {
                    head: -(rng.below(height as u32 * 2 + 1) as i32),
                    period: FASTEST + rng.below((SLOWEST - FASTEST + 1) as u32) as u64,
                    seed: rng.next() as u32,
                })
            })
            .collect();
    }

    /// Advance every column that is due on this frame.
    pub fn tick(&mut self, frame: u64) {
        let height = self.height as i32;
        for column in self.columns.iter_mut().flatten() {
            if !frame.is_multiple_of(column.period) {
                continue;
            }
            column.head += 1;
            if column.head - TAIL > height {
                column.head = -((column.seed % 12) as i32);
            }
        }
    }

    /// What to draw at `(x, y)` within the field, if anything.
    ///
    /// The head is brightest and the tail fades to the faintest green.
    pub fn cell(&self, x: u16, y: u16) -> Option<Span<'static>> {
        if self.clear.contains(&x) {
            return None;
        }
        let column = self.columns.get(x as usize)?.as_ref()?;
        let behind = column.head - y as i32;
        if !(0..=TAIL).contains(&behind) {
            return None;
        }
        let glyph = RAIN_GLYPHS[(column.seed as usize + y as usize * 7) % RAIN_GLYPHS.len()];
        let color = if behind == 0 {
            theme::GREEN_DIM
        } else {
            theme::mix(theme::GREEN_FAINT, theme::BG, behind as f32 / TAIL as f32)
        };
        Some(Span::styled(
            (glyph as char).to_string(),
            ratatui::style::Style::default().fg(color),
        ))
    }
}

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

/// Rows the block font is drawn in.
pub const FONT_HEIGHT: usize = 5;

/// `text` in the block font, as [`FONT_HEIGHT`] rows of equal length.
///
/// Only the characters for throughput figures and the wordmark exist. Any
/// other character renders as a three-cell blank.
pub fn block_text(text: &str) -> Vec<String> {
    let mut rows = vec![String::new(); FONT_HEIGHT];
    for ch in text.chars() {
        let glyph = glyph(ch);
        for (row, cut) in rows.iter_mut().zip(glyph) {
            if !row.is_empty() {
                row.push(' ');
            }
            for cell in cut.chars() {
                row.push(if cell == '#' { theme::FULL } else { ' ' });
            }
        }
    }
    rows
}

/// Width [`block_text`] will produce for `text`, in cells.
pub fn block_width(text: &str) -> usize {
    let count = text.chars().count();
    if count == 0 {
        return 0;
    }
    text.chars().map(|c| glyph(c)[0].len()).sum::<usize>() + count - 1
}

/// One character of the block font, [`FONT_HEIGHT`] rows of `#` and space.
///
/// Three cells wide, the narrowest that keeps digits distinct. A period is
/// one cell wide.
fn glyph(ch: char) -> [&'static str; FONT_HEIGHT] {
    match ch.to_ascii_uppercase() {
        '0' => ["###", "# #", "# #", "# #", "###"],
        '1' => ["  #", "  #", "  #", "  #", "  #"],
        '2' => ["###", "  #", "###", "#  ", "###"],
        '3' => ["###", "  #", "###", "  #", "###"],
        '4' => ["# #", "# #", "###", "  #", "  #"],
        '5' => ["###", "#  ", "###", "  #", "###"],
        '6' => ["###", "#  ", "###", "# #", "###"],
        '7' => ["###", "  #", "  #", "  #", "  #"],
        '8' => ["###", "# #", "###", "# #", "###"],
        '9' => ["###", "# #", "###", "  #", "###"],
        '.' => [" ", " ", " ", " ", "#"],
        'B' => ["## ", "# #", "## ", "# #", "## "],
        'H' => ["# #", "# #", "###", "# #", "# #"],
        'O' => ["###", "# #", "# #", "# #", "###"],
        'P' => ["###", "# #", "###", "#  ", "#  "],
        'S' => ["###", "#  ", "###", "  #", "###"],
        _ => ["   ", "   ", "   ", "   ", "   "],
    }
}

/// Format a rate for the block font, with at most one decimal place and none
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

/// A 64-bit xorshift for laying out the rain. Fixed-seeded, so the field
/// looks the same every run.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Xorshift {
        Xorshift(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            0
        } else {
            (self.next() % bound as u64) as u32
        }
    }
}

/// A color for a figure relative to the run's best: bright at the best, dim
/// as it falls away.
pub fn relative(value: f64, best: f64, color: Color) -> Color {
    if best <= 0.0 || value <= 0.0 {
        return theme::GREEN_DIM;
    }
    theme::mix(theme::GREEN_DIM, color, (value / best) as f32)
}
