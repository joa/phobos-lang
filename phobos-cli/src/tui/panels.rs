// The panels, each one a pure function of a snapshot and the view state.
//
// A panel never queries the engine and never blocks; it is handed a
// [`Snapshot`] taken before the frame started, so every figure on screen is
// from the same instant. Anything the front end could not report is drawn as
// such rather than as a zero, since a host build has no card to read and an
// ONNX model does not account for its own weights.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

use phobos_base::log::Level;
use phobos_inference::telemetry::{Phase, Snapshot};

use super::anim::{self, FONT_HEIGHT};
use super::cinema;
use super::meters;
use super::splash;
use super::theme;
use super::view::View;

/// Rows the wordmark banner needs: the font plus a line of air under it.
const HEADER_ROWS: u16 = FONT_HEIGHT as u16 + 2;

/// Below this many rows the banner is dropped for a single title line: the
/// panels carry the information and the wordmark does not.
const TALL_ENOUGH: u16 = 26;

/// Below this many columns the panels stack instead of sitting side by side.
const WIDE_ENOUGH: u16 = 100;

/// Rows the block strip needs: a line of numbers, the strip, and a legend.
/// The card sits beside it and wants the same, which is what sets the floor.
const NETWORK_ROWS: u16 = 6;

pub fn render(frame: &mut Frame, view: &mut View, snap: &Snapshot) {
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(theme::BG)), area);

    // The moon gets the whole screen while it is up, and only on a terminal
    // with room for it. A load has minutes of compiling to report and the
    // picture is not what a watcher needs for those, so it is brief.
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
                Span::styled("PHOBOS", theme::accent(theme::GREEN)),
                Span::styled("  inference dashboard", theme::muted()),
            ])),
            rows[0],
        );
    }
    // Until there is a model, the panels have nothing in them and the load
    // is the only thing happening, so it gets the whole screen.
    if snap.fixed.label.is_empty() {
        loading(frame, view, snap, rows[1].union(rows[2]));
    } else {
        status(frame, view, snap, rows[1]);
        body(frame, view, snap, rows[2]);
    }
    footer(frame, view, snap, rows[3]);
}

/// The screen that is up before there is a model: what the load has got
/// through, the kernel going past, and where the time went.
fn loading(frame: &mut Frame, view: &View, snap: &Snapshot, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(6)])
        .split(area);
    meters::loading(frame, view, snap, rows[0]);
    if rows[1].height == 0 {
        return;
    }
    // The cinema wants width and the histogram is a short list, so they sit
    // beside each other rather than stacked.
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(64), Constraint::Percentage(36)])
        .split(rows[1]);
    cinema::compiling(frame, view, snap, columns[0]);
    cinema::histogram(frame, snap, columns[1]);
}

/// The wordmark, with the rain falling around it and a highlight sweeping
/// across it.
fn header(frame: &mut Frame, view: &mut View, area: Rect) {
    view.rain.resize(area.width, area.height);
    let glyphs = anim::block_text("PHOBOS");
    let mark_width = anim::block_width("PHOBOS") as u16;
    let left = area.width.saturating_sub(mark_width) / 2;
    // A margin either side, so a drop never lands against a stroke.
    view.rain
        .clear_lane(left.saturating_sub(3)..(left + mark_width + 3));
    // Travels a little past both edges, so the sweep leaves and arrives
    // rather than appearing at the margin.
    let span = area.width as i64 + 24;
    let sweep = (view.frame as i64 / 2) % span - 12;

    let mut lines = Vec::with_capacity(area.height as usize);
    for y in 0..area.height {
        let mut spans = Vec::with_capacity(area.width as usize);
        let row = (y as usize).checked_sub(1).and_then(|r| glyphs.get(r));
        for x in 0..area.width {
            let lit = row
                .and_then(|row| {
                    x.checked_sub(left)
                        .and_then(|i| row.chars().nth(i as usize))
                })
                .is_some_and(|cell| cell == theme::FULL);
            if lit {
                // Brightest where the sweep is, falling away over a few cells
                // on either side of it.
                let distance = (x as i64 - sweep).abs() as f32;
                let heat = (1.0 - distance / 14.0).clamp(0.0, 1.0);
                let color = theme::mix(theme::GREEN_DIM, theme::CYAN, heat.powi(2));
                spans.push(Span::styled(
                    theme::FULL.to_string(),
                    Style::default().fg(color),
                ));
            } else if let Some(drop) = view.rain.cell(x, y) {
                spans.push(drop);
            } else {
                spans.push(Span::raw(" "));
            }
        }
        lines.push(Line::from(spans));
    }
    frame.render_widget(Paragraph::new(lines), area);
}

/// The one line that says what the engine is doing right now.
fn status(frame: &mut Frame, view: &View, snap: &Snapshot, area: Rect) {
    let (label, color) = match snap.phase {
        Phase::Idle => ("IDLE", theme::MUTED),
        Phase::Prefill => ("PREFILL", theme::CYAN),
        Phase::Decode => ("DECODE", theme::MAGENTA),
    };
    // The light breathes only while there is work; a steady dot when idle is
    // easier to read past than one that never stops moving.
    let lamp = if snap.phase == Phase::Idle {
        Style::default().fg(theme::MUTED)
    } else {
        Style::default().fg(theme::mix(
            theme::GREEN_FAINT,
            color,
            anim::pulse(view.frame, 24),
        ))
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
            theme::accent(theme::AMBER),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn body(frame: &mut Frame, view: &mut View, snap: &Snapshot, area: Rect) {
    if area.width < WIDE_ENOUGH {
        // Narrow: the two that answer "is it working" go first.
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(area);
        meters::throughput(frame, view, snap, rows[0]);
        meters::memory(frame, view, snap, rows[1]);
        return;
    }

    // Three rows, each given up in turn as the window shortens: the gauges
    // answer "is it working", the strip says what it is, and the rings are
    // the history, which is what a short window can most afford to lose.
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
    // The two that describe the engine rather than the run share a column.
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Min(4)])
        .split(top[0]);
    model(frame, snap, left[0]);
    meters::caches(frame, snap, left[1]);
    meters::memory(frame, view, snap, top[1]);
    meters::throughput(frame, view, snap, top[2]);

    if rows[1].height > 0 {
        let middle = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(rows[1]);
        meters::layers(frame, view, snap, middle[0]);
        meters::card(frame, snap, middle[1]);
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
        field("engine", &fixed.label, theme::GREEN),
        field("backend", &fixed.backend, theme::CYAN),
        field("vocab", &count(fixed.vocab_size as u64), theme::TEXT),
        field("context", &count(fixed.context_limit as u64), theme::TEXT),
    ];
    match fixed.footprint {
        Some(footprint) => {
            lines.push(field(
                "weights",
                &bytes(footprint.weight_bytes),
                theme::AMBER,
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
        Paragraph::new(lines).block(panel("MODEL", theme::GREEN_DIM, false)),
        area,
    );
}

/// What the engine has been asked to do, newest first.
fn activity(frame: &mut Frame, snap: &Snapshot, area: Rect) {
    let mut lines = Vec::new();
    if let Some(active) = snap.active {
        lines.push(Line::from(vec![
            Span::styled(format!("{} ", theme::CARET), theme::accent(theme::MAGENTA)),
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
            Span::styled("       pp", Style::default().fg(theme::CYAN)),
            Span::styled("       tg", Style::default().fg(theme::MAGENTA)),
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
                Style::default().fg(theme::CYAN),
            ),
            Span::styled(
                format!("{:>9}", anim::rate_text(request.decode_rate)),
                Style::default().fg(theme::MAGENTA),
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
        Paragraph::new(lines).block(panel("ACTIVITY", theme::MAGENTA, snap.active.is_some())),
        area,
    );
}

/// Whatever the runtime wanted to say, which on a full-screen front end has
/// nowhere else to go.
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
                Level::Trace => theme::GREEN_FAINT,
            };
            Line::from(vec![
                Span::styled(
                    format!("{:>7.2}s ", line.at.as_secs_f64()),
                    Style::default().fg(theme::GREEN_FAINT),
                ),
                Span::styled(line.text.clone(), Style::default().fg(color)),
            ])
        })
        .collect();
    frame.render_widget(
        Paragraph::new(lines).block(panel("LOG", theme::GREEN_DIM, false)),
        area,
    );
}

fn footer(frame: &mut Frame, view: &View, snap: &Snapshot, area: Rect) {
    let keys = Line::from(vec![
        Span::styled(" q ", theme::accent(theme::GREEN)),
        Span::styled("quit    ", theme::muted()),
        Span::styled(" c ", theme::accent(theme::GREEN)),
        Span::styled("clear the log    ", theme::muted()),
        Span::styled(" r ", theme::accent(theme::GREEN)),
        Span::styled("reset the peaks    ", theme::muted()),
        Span::styled(" p ", theme::accent(theme::GREEN)),
        Span::styled("splash", theme::muted()),
    ]);
    frame.render_widget(Paragraph::new(keys), area);

    let heat = anim::pulse(view.frame, 40);
    let tail = Line::from(vec![
        Span::styled(snap.fixed.backend.clone(), theme::muted()),
        Span::styled(
            format!(" {} ", theme::DOT),
            Style::default().fg(theme::mix(theme::GREEN_FAINT, theme::GREEN, heat)),
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

/// Bytes at the largest unit that leaves a figure above one, so a reader
/// compares two numbers rather than two exponents.
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

/// A token count, thousands separated, since these run to six figures.
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
