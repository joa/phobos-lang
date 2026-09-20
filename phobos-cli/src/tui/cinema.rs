// The compile, shown as what it is: a kernel's own text going in one side,
// the PTX it became coming out the other, and the bytes of that PTX after it.
//
// Every character on this screen is real. The source is the text the compiler
// was handed, the PTX is what it returned, and the hex is that PTX's bytes.
// Nothing is generated to fill space, which is what makes it worth watching:
// a kernel that takes a minute is a minute of its own code going past.
//
// What is not real is the middle column. The stages there are the passes a
// lowering runs, lit in turn on a timer rather than by anything the compiler
// reports, so it says work is happening and never which pass is running.

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
/// timer: the compiler does not report which it is in, and a marker that
/// claimed to know would be inventing it.
const STAGES: [&str; 5] = ["PARSE", "IR", "MLIR", "LLVM", "PTX"];

/// Frames each stage stays lit.
const STAGE_FRAMES: u64 = 9;

/// Columns the stage column needs, the widest label plus room either side.
const STAGE_WIDTH: u16 = 11;

/// The whole cinema: the text going in, the machine, the text coming out.
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
    // All three text columns scroll together, so a line of source and the PTX
    // beside it move as one thing rather than three.
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

/// What is being compiled, how big it is, and what it cost.
fn headline(load: &Loading) -> Line<'static> {
    // The last one built, which is what there is source and PTX for. What is
    // still running has produced neither yet.
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
            // Wraps around rather than running out: a kernel compiles for
            // longer than its own text takes to go past.
            let line = lines[(scroll + row) % lines.len()];
            let cut: String = line.chars().take(width).collect();
            // Brightest in the middle of the column, falling away at both
            // edges, so the text reads as moving through rather than sitting.
            let edge = (row as f32 / height as f32 - 0.5).abs() * 2.0;
            Line::from(Span::styled(
                cut,
                Style::default().fg(theme::mix(color, theme::BG, edge * 0.7)),
            ))
        })
        .collect();
    frame.render_widget(Paragraph::new(rendered), area);
}

/// The same PTX again, as the bytes it is.
fn hex_column(frame: &mut Frame, area: Rect, ptx: &str, scroll: usize, height: usize) {
    if area.width < 8 {
        return;
    }
    // Three columns a byte, and the row is only worth drawing whole.
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

/// The machine in the middle: the passes, lit in turn.
fn stages(frame: &mut Frame, view: &View, area: Rect) {
    let height = area.height as usize;
    let lit = (view.frame / STAGE_FRAMES) as usize % STAGES.len();
    // Centred in the column, so the labels sit against the two text columns
    // rather than at the top of a tall panel.
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
                // The shaded band the text appears to pass through, drifting so
                // the column is never still.
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
/// The shape is the point: a handful of slow kernels account for most of a
/// cold start, and this says which end the time went to.
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
        // Summed across threads, so it runs ahead of the clock on the wall.
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
