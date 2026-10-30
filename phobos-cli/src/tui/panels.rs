// The panels, each a pure function of a snapshot and the view state.
//
// A panel never queries the engine. It gets a [`Snapshot`] taken before the
// frame, so every figure is from the same instant. Anything the front end did
// not report is drawn as missing, not as zero.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

use phobos_base::log::Level;
use phobos_inference::telemetry::{Phase, Snapshot};

use super::anim;
use super::cinema;
use super::font;
use super::mark::Part;
use super::meters;
use super::splash;
use super::theme;
use super::view::View;

/// Rows the banner needs: the font with a line of air above and below, which
/// is the height the mark beside it is drawn at.
const HEADER_ROWS: u16 = font::HEIGHT as u16 + 2;

/// Below this many rows the banner becomes a single title line.
const TALL_ENOUGH: u16 = 26;

/// Below this many columns the panels stack instead of sitting side by side.
const WIDE_ENOUGH: u16 = 100;

/// Rows for the network row, set by what the network and card panels need.
const NETWORK_ROWS: u16 = 6;

/// The card and sensor panels' widths, each its longest line and the border.
/// The network strip takes what is left, at least [`NETWORK_MIN_WIDTH`].
const CARD_WIDTH: u16 = 46;
const SENSORS_WIDTH: u16 = 38;
const NETWORK_MIN_WIDTH: u16 = 56;

pub fn render(frame: &mut Frame, view: &mut View, snap: &Snapshot) {
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(theme::BG)), area);

    // The splash takes the whole screen while it runs, if it fits.
    if view.splash > 0 && splash::fits(area) {
        splash::draw(frame, view, area);
        return;
    }

    let banner = if area.height >= TALL_ENOUGH {
        HEADER_ROWS
    } else {
        1
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(banner),
            Constraint::Length(2),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(area);

    if banner > 1 {
        header(frame, view, rows[0]);
    } else {
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("Phobos", theme::accent(theme::BRAND_HI)),
                Span::styled("  inference dashboard", theme::muted()),
            ])),
            rows[0],
        );
    }
    // Until there is a model, the loading screen takes the whole body.
    if snap.fixed.label.is_empty() {
        loading(frame, view, snap, rows[1].union(rows[2]));
    } else {
        status(frame, view, snap, rows[1]);
        body(frame, view, snap, rows[2]);
    }
    footer(frame, view, snap, rows[3]);
}

/// The screen shown before there is a model: load progress, the kernel being
/// compiled, and compile times.
fn loading(frame: &mut Frame, view: &View, snap: &Snapshot, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(6)])
        .split(area);
    meters::loading(frame, view, snap, rows[0]);
    if rows[1].height == 0 {
        return;
    }
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(64), Constraint::Percentage(36)])
        .split(rows[1]);
    cinema::compiling(frame, view, snap, columns[0]);
    cinema::histogram(frame, snap, columns[1]);
}

/// The mark and the wordmark under the night sky, with a highlight sweeping
/// across both now and then.
fn header(frame: &mut Frame, view: &mut View, area: Rect) {
    const GAP: u16 = 4;
    let word_width = font::width("Phobos") as u16;
    view.sky.resize(area.width, area.height);
    let frame_no = view.frame;
    // A row short of the banner, the size at which the crater still reads.
    let mark_rows = area.height.saturating_sub(1);
    let mark = view.mark(mark_rows);
    // The mark goes first when there is no room for both.
    let (mark_width, mark_height) = (mark.width, mark.height);
    let with_mark = area.width >= mark_width + GAP + word_width + 8;
    let group = if with_mark {
        mark_width + GAP + word_width
    } else {
        word_width
    };
    let left = area.width.saturating_sub(group) / 2;
    let word_left = if with_mark {
        left + mark_width + GAP
    } else {
        left
    };
    // The sweep runs a little past both edges, once every eight seconds.
    let sweep = (frame_no as i64 % 240) * 2 - 30;
    let heat = |x: u16| {
        let distance = (x as i64 - sweep).abs() as f32;
        (1.0 - distance / 10.0).clamp(0.0, 1.0).powi(2)
    };

    view.sky
        .clear_lane(left.saturating_sub(3)..left + group + 3);
    let lights = view.sky.lights(frame_no);
    let mark = view.mark(mark_rows);
    let buf = frame.buffer_mut();
    for (x, y, glyph, color) in lights {
        splash::put(buf, area.x + x, area.y + y, glyph, color);
    }

    if with_mark {
        let top = area.y + area.height.saturating_sub(mark_height) / 2;
        let wash = theme::mix(theme::BG, theme::BRAND, 0.08);
        let outline = theme::mix(theme::BRAND_DIM, theme::TEXT, 0.6);
        for y in 0..mark_height {
            for x in 0..mark_width {
                if mark.inside(x, y) {
                    splash::bg(buf, area.x + left + x, top + y, wash);
                }
            }
        }
        for stroke in &mark.strokes {
            let rest = match stroke.part {
                Part::Crater => theme::BRAND,
                Part::Body | Part::Pit => outline,
            };
            let x = left + stroke.x;
            let color = theme::mix(rest, theme::BRAND_HI, heat(x));
            splash::put(buf, area.x + x, top + stroke.y, stroke.glyph, color);
        }
    }

    let word_top = area.y + area.height.saturating_sub(font::HEIGHT as u16) / 2;
    for (row, line) in font::render("Phobos").iter().enumerate() {
        for (col, cell) in line.chars().enumerate() {
            if cell == ' ' {
                continue;
            }
            let x = word_left + col as u16;
            let color = theme::mix(theme::BRAND, theme::BRAND_HI, heat(x));
            splash::put(buf, area.x + x, word_top + row as u16, cell, color);
        }
    }
}

/// The status line: phase, uptime, and totals.
fn status(frame: &mut Frame, view: &View, snap: &Snapshot, area: Rect) {
    let label = snap.phase.label();
    let color = match snap.phase {
        Phase::Idle => theme::MUTED,
        Phase::Prefill => theme::BRAND,
        Phase::Decode => theme::NEBULA,
    };
    // The lamp pulses only while there is work.
    let lamp = if snap.phase == Phase::Idle {
        Style::default().fg(theme::MUTED)
    } else {
        Style::default().fg(theme::mix(theme::FAINT, color, anim::pulse(view.frame, 24)))
    };

    let mut spans = vec![
        Span::raw(" "),
        Span::styled(theme::DOT.to_string(), lamp),
        Span::raw(" "),
        Span::styled(format!("{label:<7}"), theme::accent(color)),
        Span::styled(" | ", theme::muted()),
        Span::styled("up ", theme::muted()),
        Span::styled(duration(snap.uptime), theme::text()),
        Span::styled("  requests ", theme::muted()),
        Span::styled(snap.requests.to_string(), theme::text()),
        Span::styled("  in ", theme::muted()),
        Span::styled(count(snap.prompt_tokens), theme::text()),
        Span::styled("  out ", theme::muted()),
        Span::styled(count(snap.completion_tokens), theme::text()),
    ];
    if let Some(listen) = snap.fixed.listen.as_deref() {
        spans.push(Span::styled("  | ", theme::muted()));
        spans.push(Span::styled(
            format!("http://{listen}"),
            theme::accent(theme::GOLD),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn body(frame: &mut Frame, view: &mut View, snap: &Snapshot, area: Rect) {
    if area.width < WIDE_ENOUGH {
        // Narrow: only throughput and memory.
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(area);
        meters::throughput(frame, view, snap, rows[0]);
        meters::memory(frame, view, snap, rows[1]);
        return;
    }

    // Three rows: gauges, then network and card, then activity and log. As
    // the window shortens the bottom row goes first, then the middle.
    let network = if area.height >= NETWORK_ROWS + 16 {
        NETWORK_ROWS
    } else {
        0
    };
    let rings = if area.height >= network + 22 {
        Constraint::Percentage(40)
    } else {
        Constraint::Length(0)
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(network), rings])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(32),
            Constraint::Percentage(36),
            Constraint::Percentage(32),
        ])
        .split(rows[0]);
    // Model and caches share the left column.
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Min(4)])
        .split(top[0]);
    model(frame, snap, left[0]);
    meters::caches(frame, snap, left[1]);
    meters::memory(frame, view, snap, top[1]);
    meters::throughput(frame, view, snap, top[2]);

    if rows[1].height > 0 {
        // The card's fixed description gives way first, the live sensors
        // stay.
        let card = if rows[1].width >= NETWORK_MIN_WIDTH + CARD_WIDTH + SENSORS_WIDTH {
            CARD_WIDTH
        } else {
            0
        };
        let middle = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(0),
                Constraint::Length(card),
                Constraint::Length(SENSORS_WIDTH),
            ])
            .split(rows[1]);
        meters::layers(frame, view, snap, middle[0]);
        if card > 0 {
            meters::card(frame, snap, middle[1]);
        }
        meters::sensors(frame, view, snap, middle[2]);
    }
    if rows[2].height == 0 {
        return;
    }
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(rows[2]);
    activity(frame, snap, bottom[0]);
    log(frame, snap, bottom[1]);
}

fn model(frame: &mut Frame, snap: &Snapshot, area: Rect) {
    let fixed = &snap.fixed;
    let mut lines = vec![
        field("engine", &fixed.label, theme::BRAND_HI),
        field("backend", &fixed.backend, theme::BRAND),
        field("vocab", &count(fixed.vocab_size as u64), theme::TEXT),
        field("context", &count(fixed.context_limit as u64), theme::TEXT),
    ];
    match fixed.footprint {
        Some(footprint) => {
            lines.push(field(
                "weights",
                &bytes(footprint.weight_bytes),
                theme::GOLD,
            ));
            if footprint.dense_bytes > 0 {
                lines.push(field(
                    "widened",
                    &format!("{} of that, held f32", bytes(footprint.dense_bytes)),
                    theme::MUTED,
                ));
            }
            if footprint.streamed_bytes > 0 {
                lines.push(field(
                    "streamed",
                    &format!("{} of experts, on the host", bytes(footprint.streamed_bytes)),
                    theme::MUTED,
                ));
            }
        }
        None => lines.push(field(
            "weights",
            "not reported by this front end",
            theme::MUTED,
        )),
    }
    frame.render_widget(
        Paragraph::new(lines).block(panel("MODEL", theme::BRAND_DIM, false)),
        area,
    );
}

/// What the engine has been asked to do, newest first.
fn activity(frame: &mut Frame, snap: &Snapshot, area: Rect) {
    let mut lines = Vec::new();
    if let Some(active) = snap.active {
        lines.push(Line::from(vec![
            Span::styled(format!("{} ", theme::CARET), theme::accent(theme::NEBULA)),
            Span::styled(
                format!("{} ", count(active.prompt_tokens as u64)),
                theme::text(),
            ),
            Span::styled("prompt, ", theme::muted()),
            Span::styled(format!("{} ", count(active.produced as u64)), theme::text()),
            Span::styled("out, ", theme::muted()),
            Span::styled(duration(active.elapsed), theme::text()),
        ]));
    }
    if !snap.recent.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("   prompt    out", theme::muted()),
            Span::styled("       pp", Style::default().fg(theme::BRAND)),
            Span::styled("       tg", Style::default().fg(theme::NEBULA)),
            Span::styled("   stopped on", theme::muted()),
        ]));
    }
    let rows = area
        .height
        .saturating_sub(if snap.active.is_some() { 4 } else { 3 });
    for request in snap.recent.iter().take(rows as usize) {
        lines.push(Line::from(vec![
            Span::styled(format!("{} ", theme::DOT_HOLLOW), theme::muted()),
            Span::styled(
                format!("{:>7}", count(request.prompt_tokens as u64)),
                theme::text(),
            ),
            Span::styled(
                format!("{:>7}", count(request.completion_tokens as u64)),
                theme::text(),
            ),
            Span::styled(
                format!("{:>9}", anim::rate_text(request.prefill_rate)),
                Style::default().fg(theme::BRAND),
            ),
            Span::styled(
                format!("{:>9}", anim::rate_text(request.decode_rate)),
                Style::default().fg(theme::NEBULA),
            ),
            Span::styled(format!("   {}", request.reason), theme::muted()),
        ]));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "waiting for a request",
            theme::muted(),
        )));
    }
    frame.render_widget(
        Paragraph::new(lines).block(panel("ACTIVITY", theme::NEBULA, snap.active.is_some())),
        area,
    );
}

/// The runtime's log lines, newest first.
fn log(frame: &mut Frame, snap: &Snapshot, area: Rect) {
    let rows = area.height.saturating_sub(2) as usize;
    let lines: Vec<Line> = snap
        .log
        .iter()
        .take(rows)
        .map(|line| {
            let color = match line.level {
                Level::Off | Level::Info => theme::TEXT,
                Level::Debug => theme::MUTED,
                Level::Trace => theme::FAINT,
            };
            Line::from(vec![
                Span::styled(
                    format!("{:>7.2}s ", line.at.as_secs_f64()),
                    Style::default().fg(theme::FAINT),
                ),
                Span::styled(line.text.clone(), Style::default().fg(color)),
            ])
        })
        .collect();
    frame.render_widget(
        Paragraph::new(lines).block(panel("LOG", theme::BRAND_DIM, false)),
        area,
    );
}

fn footer(frame: &mut Frame, view: &View, snap: &Snapshot, area: Rect) {
    let keys = Line::from(vec![
        Span::styled(" q ", theme::accent(theme::BRAND_HI)),
        Span::styled("quit    ", theme::muted()),
        Span::styled(" c ", theme::accent(theme::BRAND_HI)),
        Span::styled("clear the log    ", theme::muted()),
        Span::styled(" r ", theme::accent(theme::BRAND_HI)),
        Span::styled("reset the peaks    ", theme::muted()),
        Span::styled(" p ", theme::accent(theme::BRAND_HI)),
        Span::styled("splash", theme::muted()),
    ]);
    frame.render_widget(Paragraph::new(keys), area);

    let heat = anim::pulse(view.frame, 40);
    let tail = Line::from(vec![
        Span::styled(snap.fixed.backend.clone(), theme::muted()),
        Span::styled(
            format!(" {} ", theme::DOT),
            Style::default().fg(theme::mix(theme::FAINT, theme::BRAND_HI, heat)),
        ),
    ]);
    frame.render_widget(Paragraph::new(tail).alignment(Alignment::Right), area);
}

/// A bordered panel, its edge lit while something is happening inside it.
pub(super) fn panel(title: &str, color: ratatui::style::Color, active: bool) -> Block<'static> {
    let edge = if active { color } else { theme::BORDER };
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(edge))
        .title(Line::from(vec![
            Span::raw(" "),
            Span::styled(title.to_string(), theme::accent(color)),
            Span::raw(" "),
        ]))
}

fn field(label: &str, value: &str, color: ratatui::style::Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<9}"), theme::muted()),
        Span::styled(value.to_string(), Style::default().fg(color)),
    ])
}

/// Bytes in the largest binary unit that keeps the figure at least one.
pub fn bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// A count with thousands separated by spaces.
pub fn count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(ch);
    }
    out
}

pub fn duration(d: std::time::Duration) -> String {
    let seconds = d.as_secs();
    match seconds {
        s if s >= 3600 => format!("{}h{:02}m", s / 3600, (s % 3600) / 60),
        s if s >= 60 => format!("{}m{:02}s", s / 60, s % 60),
        _ => format!("{:.1}s", d.as_secs_f64()),
    }
}
