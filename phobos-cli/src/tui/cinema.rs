// The compile screen: a kernel's source on one side, the PTX it became on the
// other, and that PTX's bytes as hex.
//
// The source, PTX and hex are real compiler input and output. The middle
// column is not: its stages are lit in turn on a timer, since the compiler
// does not report which pass is running.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use phobos_inference::telemetry::{Loading, Snapshot};

use super::anim;
use super::panels::{bytes, count, panel};
use super::theme;
use super::view::View;

/// The passes a kernel goes through, named for the display. Lit in order on a
/// timer, not by compiler progress.
const STAGES: [&str; 5] = ["PARSE", "IR", "MLIR", "LLVM", "PTX"];

/// Frames each stage stays lit.
const STAGE_FRAMES: u64 = 9;

/// Columns the stage column needs, the widest label plus room either side.
const STAGE_WIDTH: u16 = 11;

/// The whole compile panel: source, stages, PTX and hex.
pub(super) fn compiling(frame: &mut Frame, view: &View, snap: &Snapshot, area: Rect) {
    let block = panel("COMPILING", theme::CYAN, true);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(load) = snap.loading.as_ref() else {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "waiting for the first kernel",
                theme::muted(),
            ))),
            inner,
        );
        return;
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(3)])
        .split(inner);
    frame.render_widget(Paragraph::new(headline(load)), rows[0]);

    if rows[1].height == 0 {
        return;
    }
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Length(STAGE_WIDTH),
            Constraint::Percentage(34),
            Constraint::Percentage(32),
        ])
        .split(rows[1]);

    let height = rows[1].height as usize;
    // All three text columns scroll together.
    let scroll = (view.frame / 4) as usize;
    text_column(
        frame,
        columns[0],
        &load.source,
        scroll,
        height,
        theme::GREEN,
    );
    stages(frame, view, columns[1]);
    text_column(frame, columns[2], &load.ptx, scroll, height, theme::CYAN);
    hex_column(frame, columns[3], &load.ptx, scroll, height);
}

/// The last kernel built, its sizes, and how long it took.
fn headline(load: &Loading) -> Line<'static> {
    // Only a finished kernel has source and PTX to show.
    let mut spans = vec![
        Span::styled("built ", theme::muted()),
        Span::styled(load.item.clone(), theme::accent(theme::CYAN)),
        Span::styled("   ", theme::muted()),
        Span::styled(bytes(load.source.len() as u64), theme::text()),
        Span::styled(" source", theme::muted()),
        Span::styled("  ->  ", Style::default().fg(theme::GREEN_FAINT)),
        Span::styled(bytes(load.ptx.len() as u64), theme::text()),
        Span::styled(" ptx", theme::muted()),
    ];
    if load.took > std::time::Duration::ZERO {
        spans.push(Span::styled("   ", theme::muted()));
        spans.push(Span::styled(
            format!("{:.2}s", load.took.as_secs_f64()),
            theme::accent(theme::MAGENTA),
        ));
    }
    Line::from(spans)
}

/// Lines of `text`, scrolled, wrapped to the column and cut to it.
fn text_column(
    frame: &mut Frame,
    area: Rect,
    text: &str,
    scroll: usize,
    height: usize,
    color: ratatui::style::Color,
) {
    if area.width < 2 {
        return;
    }
    let width = area.width as usize - 1;
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return;
    }
    let rendered: Vec<Line> = (0..height)
        .map(|row| {
            // Wraps around when the text runs out.
            let line = lines[(scroll + row) % lines.len()];
            let cut: String = line.chars().take(width).collect();
            // Brightest in the middle of the column, fading at both edges.
            let edge = (row as f32 / height as f32 - 0.5).abs() * 2.0;
            Line::from(Span::styled(
                cut,
                Style::default().fg(theme::mix(color, theme::BG, edge * 0.7)),
            ))
        })
        .collect();
    frame.render_widget(Paragraph::new(rendered), area);
}

/// The PTX's bytes as hex.
fn hex_column(frame: &mut Frame, area: Rect, ptx: &str, scroll: usize, height: usize) {
    if area.width < 8 {
        return;
    }
    // Three columns per byte, whole bytes only.
    let per_row = ((area.width as usize - 1) / 3).max(1);
    let data = ptx.as_bytes();
    if data.is_empty() {
        return;
    }
    let rendered: Vec<Line> = (0..height)
        .map(|row| {
            let at = ((scroll + row) * per_row) % data.len();
            let hex: String = (0..per_row)
                .map(|i| format!("{:02x} ", data[(at + i) % data.len()]))
                .collect();
            let edge = (row as f32 / height as f32 - 0.5).abs() * 2.0;
            Line::from(Span::styled(
                hex,
                Style::default().fg(theme::mix(theme::MAGENTA, theme::BG, edge * 0.7)),
            ))
        })
        .collect();
    frame.render_widget(Paragraph::new(rendered), area);
}

/// The middle column: the passes, lit in turn.
fn stages(frame: &mut Frame, view: &View, area: Rect) {
    let height = area.height as usize;
    let lit = (view.frame / STAGE_FRAMES) as usize % STAGES.len();
    // Centred vertically in the column.
    let top = height.saturating_sub(STAGES.len()) / 2;

    let rendered: Vec<Line> = (0..height)
        .map(
            |row| match row.checked_sub(top).and_then(|i| STAGES.get(i)) {
                Some(stage) => {
                    let i = row - top;
                    let style = if i == lit {
                        theme::accent(theme::CYAN)
                    } else if i < lit {
                        Style::default().fg(theme::GREEN_DIM)
                    } else {
                        Style::default().fg(theme::GREEN_FAINT)
                    };
                    Line::from(Span::styled(format!("{stage:^11}"), style))
                }
                // A drifting shaded band above and below the labels.
                None => {
                    let shade =
                        theme::SHADES[(row + (view.frame / 6) as usize) % theme::SHADES.len()];
                    Line::from(Span::styled(
                        shade.to_string().repeat(area.width as usize),
                        Style::default().fg(theme::GREEN_FAINT),
                    ))
                }
            },
        )
        .collect();
    frame.render_widget(Paragraph::new(rendered), area);
}

/// How long the kernels took, in buckets.
///
/// Shows whether a cold start's time went to many fast kernels or a few slow
/// ones.
pub(super) fn histogram(frame: &mut Frame, snap: &Snapshot, area: Rect) {
    let block = panel("COMPILE TIMES", theme::MAGENTA, false);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(load) = snap.loading.as_ref().filter(|l| l.built > 0) else {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "nothing built yet",
                theme::muted(),
            ))),
            inner,
        );
        return;
    };

    let rows = load.histogram();
    let peak = rows.iter().map(|&(_, n)| n).max().unwrap_or(1).max(1);
    let track = (inner.width as usize).saturating_sub(20).clamp(6, 40);
    let mut lines: Vec<Line> = rows
        .into_iter()
        .map(|(label, n)| {
            Line::from(vec![
                Span::styled(format!("{label:>9} "), theme::muted()),
                Span::styled(
                    anim::bar(n as f64 / peak as f64, track),
                    Style::default().fg(theme::mix(theme::GREEN_DIM, theme::MAGENTA, 0.6)),
                ),
                Span::styled(format!(" {n}"), theme::text()),
            ])
        })
        .collect();

    lines.push(Line::raw(""));
    lines.push(Line::from(vec![
        Span::styled(bytes(load.source_bytes), theme::text()),
        Span::styled(" of kernel text became ", theme::muted()),
        Span::styled(bytes(load.ptx_bytes), theme::text()),
        Span::styled(" of ptx", theme::muted()),
    ]));
    if let Some(expansion) = load.expansion() {
        lines.push(Line::from(vec![
            Span::styled(format!("{expansion:.1}x"), theme::accent(theme::GREEN)),
            Span::styled(" ptx to source", theme::muted()),
        ]));
    }
    lines.push(Line::from(vec![
        Span::styled(
            format!("{:.0}s", load.compile_time.as_secs_f64()),
            theme::text(),
        ),
        // Summed across threads, so it can exceed the wall clock.
        Span::styled(" lowering, ", theme::muted()),
        Span::styled(count(load.built), theme::text()),
        Span::styled(" kernels at once", theme::muted()),
    ]));
    if !load.slowest.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("slowest  ", theme::muted()),
            Span::styled(
                load.slowest.clone(),
                Style::default()
                    .fg(theme::AMBER)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {:.2}s", load.slowest_took.as_secs_f64()),
                theme::text(),
            ),
        ]));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}
