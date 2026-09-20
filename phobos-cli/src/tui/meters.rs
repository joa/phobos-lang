// The gauges: what memory holds, how fast it is going, what the network is
// made of, and what the backend's caches gave back.
//
// Split from the panels around them because these four are the ones that
// draw rather than list, and because a panel file that renders every bar as
// well ends up longer than anything should be.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use phobos_inference::BlockKind;
use phobos_inference::telemetry::{Phase, Snapshot};

use super::anim::{self, FONT_HEIGHT};
use super::panels::{bytes, count, panel};
use super::theme;
use super::view::View;

/// The card, and what is standing on it.
///
/// One stacked bar rather than three separate ones: the parts are shares of
/// the same total, and a reader wants to see them add up. What is left over
/// after the weights and the caches is real memory belonging to someone, so
/// it gets a segment rather than being left out.
pub(super) fn memory(frame: &mut Frame, view: &mut View, snap: &Snapshot, area: Rect) {
    let block = panel("MEMORY", theme::AMBER, false);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let track = (inner.width as usize).saturating_sub(28).clamp(10, 44);
    let mut lines = Vec::new();

    match snap.device {
        Some(device) if device.total_bytes > 0 => {
            let total = device.total_bytes as f64;
            view.vram = anim::ease(view.vram, device.used_bytes() as f64 / total, 0.2);

            // What the model will occupy once every weight is up, which
            // before the first pass it is not: the weights go to the card
            // lazily. The driver's figure is the truth, so the shares are
            // clamped to it and can never add up to more than is in use.
            let claimed = snap
                .fixed
                .footprint
                .map(|f| f.weight_bytes as f64 / total)
                .unwrap_or(0.0);
            let weights = claimed.min(view.vram);
            let cache =
                (snap.cache_bytes.unwrap_or(0) as f64 / total).min((view.vram - weights).max(0.0));
            // Whatever the two named shares do not account for: a pass's
            // intermediates, the driver's own context, and anything else on
            // the card, this process or not.
            let elsewhere = (view.vram - weights - cache).max(0.0);
            // A tenth of a percent of slack: the reading and the footprint are
            // taken at different moments and need not agree to the byte.
            let resident = claimed <= view.vram + 0.001;

            lines.push(Line::from({
                let mut spans = vec![Span::styled("card      ", theme::muted())];
                spans.extend(anim::stacked(
                    &[
                        (weights, theme::AMBER),
                        (cache, theme::MAGENTA),
                        (elsewhere, theme::CYAN),
                    ],
                    track,
                ));
                spans.push(Span::styled(
                    format!(" {:>5.1}%", view.vram * 100.0),
                    theme::accent(theme::pressure(view.vram)),
                ));
                spans
            }));
            lines.push(Line::from(vec![
                Span::raw("          "),
                Span::styled(bytes(device.used_bytes()), theme::text()),
                Span::styled(" of ", theme::muted()),
                Span::styled(bytes(device.total_bytes), theme::text()),
            ]));
            lines.push(Line::raw(""));

            let held = snap.fixed.footprint.map(|f| f.weight_bytes).unwrap_or(0);
            lines.push(key(
                theme::AMBER,
                "weights",
                if resident {
                    bytes(held)
                } else {
                    // Less on the card than the model accounts for, so the
                    // upload is still going: saying the whole figure here
                    // would draw memory that is not there yet.
                    format!("{} of {} up", bytes((weights * total) as u64), bytes(held))
                },
            ));
            lines.push(key(
                theme::MAGENTA,
                "kv cache",
                snap.cache_bytes.map(bytes).unwrap_or("idle".to_string()),
            ));
            lines.push(key(
                theme::CYAN,
                "elsewhere",
                bytes((elsewhere * total) as u64),
            ));
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(theme::DOT_HOLLOW.to_string(), theme::muted()),
                Span::styled(format!(" {:<10}", "free"), theme::muted()),
                Span::styled(bytes(device.free_bytes), theme::accent(theme::GREEN)),
            ]));
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled(
                    "scratch, driver, other processes",
                    Style::default().fg(theme::GREEN_FAINT),
                ),
            ]));
        }
        _ => {
            lines.push(Line::from(Span::styled(
                "no device to read",
                theme::accent(theme::AMBER),
            )));
            lines.push(Line::from(Span::styled(
                "a host backend competes with the whole machine",
                theme::muted(),
            )));
            lines.push(Line::from(Span::styled(
                "rather than with a fixed budget",
                theme::muted(),
            )));
        }
    }

    lines.push(Line::raw(""));
    lines.extend(context(view, snap, track));
    frame.render_widget(Paragraph::new(lines), inner);
}

/// One legend row: a dot in the segment's colour, its name, its size.
fn key(color: Color, label: &str, value: String) -> Line<'static> {
    Line::from(vec![
        Span::raw("  "),
        Span::styled(theme::DOT.to_string(), Style::default().fg(color)),
        Span::styled(format!(" {label:<10}"), theme::muted()),
        Span::styled(value, Style::default().fg(color)),
    ])
}

/// How much context is in use, and what the rest of it would cost.
///
/// The cost at the full context is a projection and says so: it is what the
/// caches would take if a conversation ever ran that long, which for a large
/// model is usually more than the card has. The figure worth acting on is the
/// last one, which is how much context actually fits in what is free.
fn context(view: &mut View, snap: &Snapshot, track: usize) -> Vec<Line<'static>> {
    let limit = snap.fixed.context_limit;
    let ratio = if limit == 0 {
        0.0
    } else {
        snap.cache_tokens as f64 / limit as f64
    };
    view.context = anim::ease(view.context, ratio, 0.2);
    let color = theme::pressure(view.context);

    let mut lines = vec![
        Line::from(vec![
            Span::styled("context   ", theme::muted()),
            Span::styled(anim::bar(view.context, track), Style::default().fg(color)),
            Span::styled(
                format!(" {}", count(snap.cache_tokens as u64)),
                theme::accent(color),
            ),
        ]),
        Line::from(vec![
            Span::raw("          "),
            Span::styled("of ", theme::muted()),
            Span::styled(count(limit as u64), theme::text()),
            Span::styled(" positions", theme::muted()),
        ]),
    ];

    let Some(footprint) = snap.fixed.footprint else {
        return lines;
    };
    let per_token = footprint.kv_bytes_per_token;
    lines.push(Line::from(vec![
        Span::raw("          "),
        Span::styled(bytes(per_token), theme::text()),
        Span::styled(" a position", theme::muted()),
    ]));
    if per_token > 0 {
        lines.push(Line::from(vec![
            Span::raw("          "),
            Span::styled(bytes(per_token * limit as u64), theme::text()),
            Span::styled(" if the context ever filled", theme::muted()),
        ]));
        if let Some(device) = snap.device {
            lines.push(Line::from(vec![
                Span::raw("          "),
                Span::styled(
                    count(device.free_bytes / per_token),
                    theme::accent(theme::GREEN),
                ),
                Span::styled(" more fit in what is free", theme::muted()),
            ]));
        }
    }
    lines
}

/// The two rates, each as a block-font figure over a plot of its history.
pub(super) fn throughput(frame: &mut Frame, view: &mut View, snap: &Snapshot, area: Rect) {
    let block = panel("THROUGHPUT", theme::CYAN, snap.phase != Phase::Idle);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    view.prefill = anim::ease(view.prefill, snap.prefill_rate, 0.25);
    view.decode = anim::ease(view.decode, snap.decode_rate, 0.25);
    view.best_prefill = view.best_prefill.max(snap.prefill_rate);
    view.best_decode = view.best_decode.max(snap.decode_rate);

    let halves = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(inner);

    rate(
        frame,
        halves[0],
        "PROMPT",
        view.prefill,
        view.best_prefill,
        &snap.prefill_history,
        theme::CYAN,
    );
    rate(
        frame,
        halves[1],
        "DECODE",
        view.decode,
        view.best_decode,
        &snap.decode_history,
        theme::MAGENTA,
    );
}

/// One rate: the figure, its units, its best so far, and its recent shape.
fn rate(
    frame: &mut Frame,
    area: Rect,
    label: &str,
    value: f64,
    best: f64,
    history: &[f64],
    color: Color,
) {
    if area.height == 0 {
        return;
    }
    let text = anim::rate_text(value);
    let lit = anim::relative(value, best, color);
    let mut lines = vec![Line::from(vec![
        Span::styled(format!("{label} "), theme::muted()),
        Span::styled("tok/s", theme::muted()),
        Span::styled(
            format!("   peak {}", anim::rate_text(best)),
            Style::default().fg(theme::GREEN_FAINT),
        ),
    ])];

    // The block font only when there are rows for it; below that the figure
    // is still the point, just at one line.
    if area.height as usize >= FONT_HEIGHT + 2 && anim::block_width(&text) <= area.width as usize {
        for row in anim::block_text(&text) {
            lines.push(Line::from(Span::styled(row, Style::default().fg(lit))));
        }
    } else {
        lines.push(Line::from(Span::styled(
            text,
            Style::default().fg(lit).add_modifier(Modifier::BOLD),
        )));
    }

    if lines.len() < area.height as usize {
        lines.push(Line::from(Span::styled(
            anim::spark(history, area.width as usize),
            Style::default().fg(theme::mix(theme::GREEN_FAINT, color, 0.6)),
        )));
    }
    frame.render_widget(Paragraph::new(lines), area);
}

/// What the network is made of, one cell a block.
///
/// The interleave is the point on a model that has one: attention blocks are
/// the only ones whose cost grows with the conversation, so the ratio between
/// the two colours is what decides whether a long context fits.
pub(super) fn layers(frame: &mut Frame, view: &View, snap: &Snapshot, area: Rect) {
    let working = snap.phase != Phase::Idle;
    let block = panel("NETWORK", theme::GREEN, working);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(arch) = snap.fixed.architecture.as_ref() else {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "not reported by this front end",
                theme::muted(),
            ))),
            inner,
        );
        return;
    };

    let attention = arch.count(BlockKind::Attention);
    let recurrent = arch.count(BlockKind::Recurrent);
    let mut lines = vec![Line::from(vec![
        Span::styled(count(arch.blocks.len() as u64), theme::accent(theme::GREEN)),
        Span::styled(" blocks    ", theme::muted()),
        Span::styled("d_model ", theme::muted()),
        Span::styled(count(arch.d_model as u64), theme::text()),
        Span::styled("    ffn ", theme::muted()),
        Span::styled(count(arch.d_ff as u64), theme::text()),
        Span::styled("    heads ", theme::muted()),
        Span::styled(
            format!("{} q / {} kv", arch.n_head, arch.n_head_kv),
            theme::text(),
        ),
        Span::styled(" x ", theme::muted()),
        Span::styled(count(arch.head_dim as u64), theme::text()),
    ])];

    // Lit while a pass is running, so the strip reads as the live part of the
    // panel. It says the network is working, not which block is: nothing here
    // measures a block, and a marker crawling along it would claim otherwise.
    let wash = if working {
        0.75 + 0.25 * anim::pulse(view.frame, 30)
    } else {
        0.45
    };
    lines.push(Line::from(strip(&arch.blocks, inner.width as usize, wash)));

    lines.push(Line::from(vec![
        Span::styled(theme::DOT.to_string(), Style::default().fg(theme::CYAN)),
        Span::styled(format!(" attention {attention:<6}"), theme::muted()),
        Span::styled(theme::DOT.to_string(), Style::default().fg(theme::MAGENTA)),
        Span::styled(format!(" recurrent {recurrent:<6}"), theme::muted()),
        Span::styled(
            if recurrent > 0 {
                "only the attention blocks cost anything per position"
            } else {
                "every block caches keys and values per position"
            },
            Style::default().fg(theme::GREEN_FAINT),
        ),
    ]));
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The blocks across the whole width, each one as many cells as it gets.
///
/// Stretched rather than drawn one cell a block: at any usual width that
/// leaves a short bar in a wide box, and the pattern is what this is for. A
/// window too narrow for one cell each samples instead, which keeps the
/// pattern even though it loses the count.
fn strip(blocks: &[BlockKind], width: usize, wash: f32) -> Vec<Span<'static>> {
    if blocks.is_empty() || width == 0 {
        return Vec::new();
    }
    (0..width)
        .map(|i| {
            let block = blocks[i * blocks.len() / width];
            let color = match block {
                BlockKind::Attention => theme::CYAN,
                BlockKind::Recurrent => theme::MAGENTA,
            };
            Span::styled(
                theme::FULL.to_string(),
                Style::default().fg(theme::mix(theme::BG, color, wash)),
            )
        })
        .collect()
}

/// The card, as the driver describes it.
///
/// The clocks are the card's maxima and are labelled as such: an idle card
/// sits at a fraction of them, and nothing here samples a running one. The
/// bandwidth is the figure a decode is worth judging against, since decoding
/// reads every weight once a token.
pub(super) fn card(frame: &mut Frame, snap: &Snapshot, area: Rect) {
    let block = panel("CARD", theme::CYAN, false);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(card) = snap.fixed.card.as_ref() else {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("no card", theme::accent(theme::AMBER))),
                Line::from(Span::styled(
                    "this backend runs on the host",
                    theme::muted(),
                )),
            ]),
            inner,
        );
        return;
    };

    let (major, minor) = card.capability;
    let mut driver = vec![
        Span::styled("driver ", theme::muted()),
        Span::styled(
            card.driver.clone().unwrap_or_else(|| "unknown".to_string()),
            theme::text(),
        ),
        Span::styled("   CUDA ", theme::muted()),
        Span::styled(format!("{}.{}", card.cuda.0, card.cuda.1), theme::text()),
    ];
    if major > 0 {
        driver.push(Span::styled("   ", theme::muted()));
        driver.push(Span::styled(
            format!("sm_{major}{minor}"),
            theme::accent(theme::GREEN),
        ));
    }

    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(card.name.clone(), theme::accent(theme::CYAN))),
            Line::from(driver),
            Line::from(vec![
                Span::styled(card.multiprocessors.to_string(), theme::text()),
                Span::styled(" SMs   ", theme::muted()),
                Span::styled(mhz(card.core_clock_khz), theme::text()),
                Span::styled(" core   ", theme::muted()),
                Span::styled(mhz(card.memory_clock_khz), theme::text()),
                Span::styled(" mem", theme::muted()),
            ]),
            Line::from(vec![
                Span::styled(
                    format!("{}-bit bus   ", card.memory_bus_bits),
                    theme::muted(),
                ),
                Span::styled(
                    format!("{:.0} GB/s", card.peak_bandwidth() as f64 / 1e9),
                    theme::accent(theme::MAGENTA),
                ),
                Span::styled(" peak, at its rated clocks", theme::muted()),
            ]),
        ]),
        inner,
    );
}

/// A share, or nothing before anything was asked for, which is not a rate of
/// zero.
fn rate_of(hits: u64, total: u64) -> Option<f64> {
    (total > 0).then(|| hits as f64 / total as f64)
}

/// A clock in megahertz, from the kilohertz the driver reports.
fn mhz(khz: u32) -> String {
    format!("{} MHz", khz / 1000)
}

/// What the backend got back out of its caches.
///
/// Both settle: the kernel shapes a model uses are fixed once it is loaded,
/// and a decode asks for the same buffers every step. A number still climbing
/// after a few tokens is the interesting case, not a high one.
pub(super) fn caches(frame: &mut Frame, snap: &Snapshot, area: Rect) {
    let block = panel("CACHES", theme::GREEN_DIM, false);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Prompt positions a kept session already held. Unlike the two below it
    // this is not a property of the backend, so it is shown whether or not
    // the backend keeps caches of its own.
    let prompt = hit_rate(
        "prompt",
        rate_of(snap.prompt_reused, snap.prompt_tokens),
        snap.prompt_reused,
        snap.prompt_tokens - snap.prompt_reused.min(snap.prompt_tokens),
        (inner.width as usize).saturating_sub(38).clamp(5, 18),
        theme::GREEN,
    );

    let Some(stats) = snap.caches else {
        frame.render_widget(
            Paragraph::new(vec![
                prompt,
                Line::from(Span::styled("this backend keeps none", theme::muted())),
                Line::from(Span::styled(
                    "it compiles nothing and pools nothing",
                    Style::default().fg(theme::GREEN_FAINT),
                )),
            ]),
            inner,
        );
        return;
    };

    let track = (inner.width as usize).saturating_sub(38).clamp(5, 18);
    let lines = vec![
        prompt,
        hit_rate(
            "kernels",
            stats.kernel_hit_rate(),
            stats.kernels_reused,
            stats.kernels_compiled,
            track,
            theme::CYAN,
        ),
        hit_rate(
            "buffers",
            stats.buffer_hit_rate(),
            stats.buffers_reused,
            stats.buffers_allocated,
            track,
            theme::MAGENTA,
        ),
        Line::from(vec![
            Span::raw("          "),
            Span::styled(
                "kept / made from scratch",
                Style::default().fg(theme::GREEN_FAINT),
            ),
        ]),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

fn hit_rate(
    label: &str,
    rate: Option<f64>,
    hits: u64,
    misses: u64,
    track: usize,
    color: Color,
) -> Line<'static> {
    let Some(rate) = rate else {
        return Line::from(vec![
            Span::styled(format!("{label:<10}"), theme::muted()),
            Span::styled("nothing asked for yet", theme::muted()),
        ]);
    };
    Line::from(vec![
        Span::styled(format!("{label:<10}"), theme::muted()),
        Span::styled(anim::bar(rate, track), Style::default().fg(color)),
        Span::styled(format!(" {:>5.1}%", rate * 100.0), theme::accent(color)),
        Span::styled(format!("  {}", count(hits)), Style::default().fg(color)),
        Span::styled(" / ", theme::muted()),
        Span::styled(count(misses), theme::muted()),
    ])
}

/// What a load is doing, for the screen that is up before there is a model.
///
/// A cold start compiles every kernel from source and takes minutes; a warm
/// one finds them all in the on-disk cache and takes seconds. Which of the
/// two is happening is the thing a watcher most wants to know, so the counts
/// are split rather than summed.
pub(super) fn loading(frame: &mut Frame, view: &View, snap: &Snapshot, area: Rect) {
    let block = panel("LOADING", theme::CYAN, true);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(load) = snap.loading.as_ref() else {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    "reading the model",
                    theme::accent(theme::CYAN),
                )),
                Line::from(Span::styled(
                    "nothing has been compiled yet",
                    theme::muted(),
                )),
            ]),
            inner,
        );
        return;
    };

    let track = (inner.width as usize).saturating_sub(24).clamp(10, 60);
    // Names what is running, not what last finished. During a batch those
    // are different kernels, and the one still going is the answer to why
    // the screen has not moved.
    let mut lines = vec![
        match load.in_flight.as_slice() {
            [] => Line::from(Span::styled("loading", theme::accent(theme::CYAN))),
            [(name, _)] => Line::from(vec![
                Span::styled("compiling ", theme::muted()),
                Span::styled(name.clone(), theme::accent(theme::CYAN)),
            ]),
            many => Line::from(vec![
                Span::styled("lowering ", theme::muted()),
                Span::styled(count(many.len() as u64), theme::accent(theme::CYAN)),
                Span::styled(" kernels at once", theme::muted()),
            ]),
        },
        Line::raw(""),
    ];

    // A bar only where there is a batch to measure against. A kernel asked
    // for on its own is a batch of one, and a bar that is always full says
    // nothing.
    match load.ratio() {
        Some(ratio) => lines.push(Line::from(vec![
            Span::styled(anim::bar(ratio, track), Style::default().fg(theme::CYAN)),
            Span::styled(
                format!("  {:>6.2}%", ratio * 100.0),
                theme::accent(theme::CYAN),
            ),
            Span::styled(
                format!("   {} of {}", load.done, load.total),
                theme::muted(),
            ),
        ])),
        None => lines.push(Line::from(vec![
            Span::styled(
                anim::bar(anim::pulse(view.frame, 40) as f64, track),
                Style::default().fg(theme::GREEN_DIM),
            ),
            Span::styled("   one at a time", theme::muted()),
        ])),
    }

    // What has not come back yet, and which of those has been going longest.
    // A batch starts everything at once and cannot finish before its slowest
    // member, so that kernel is what the wait is actually for; the name of
    // the last one to finish says nothing about it.
    if let Some((name, waiting)) = load.longest() {
        lines.push(Line::from(vec![
            Span::styled("waiting on ", theme::muted()),
            Span::styled(name.clone(), theme::accent(theme::AMBER)),
            Span::styled(
                format!("  {} and counting", super::panels::duration(*waiting)),
                theme::text(),
            ),
        ]));
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(vec![
        Span::styled(count(load.finished()), theme::accent(theme::GREEN)),
        Span::styled(" kernels ready", theme::muted()),
        Span::styled(
            format!("   {} ", count(load.built)),
            Style::default().fg(theme::AMBER),
        ),
        Span::styled("built", theme::muted()),
        Span::styled(format!("   {} ", count(load.cached)), theme::text()),
        Span::styled("already cached", theme::muted()),
    ]));
    lines.push(Line::from(vec![
        Span::styled(super::panels::duration(load.elapsed), theme::text()),
        Span::styled(" so far", theme::muted()),
    ]));
    if load.built > 0 {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            "a cold start builds every kernel from source; the next one reads them back",
            Style::default().fg(theme::GREEN_FAINT),
        )));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}
