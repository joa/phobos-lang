// A full-screen dashboard for the inference engine.
//
// It runs on its own thread and owns the terminal. The engine stays on the
// thread that loaded the model, where its device context is.
//
// The two share only a [`Meter`]. The engine writes events into it and the
// dashboard reads snapshots on its own clock, so a slow redraw cannot slow a
// decode.
//
// The dashboard owns the keyboard, so it signals quitting: `q` clears the
// meter's running flag and the engine returns at its next idle tick.

mod anim;
mod cinema;
mod meters;
mod panels;
mod splash;
mod theme;
mod view;

#[cfg(test)]
mod tests;

use std::io::{self, IsTerminal, Stdout};
use std::sync::Arc;
use std::sync::mpsc::SyncSender;
use std::time::Duration;

use anyhow::Result;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::{
    ExecutableCommand,
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};

use phobos_inference::ModelInfo;
use phobos_inference::telemetry::Meter;

use view::View;

/// Frame interval, about thirty frames a second.
const FRAME: Duration = Duration::from_millis(33);

/// How long to wait for the dashboard to take the terminal before giving up.
const START: Duration = Duration::from_secs(5);

/// A running dashboard, and the meter it is drawing.
///
/// Started before the model is loaded, so kernel compilation is visible.
/// Until [`Dashboard::loaded`] it draws the loading screen, then the panels.
pub struct Dashboard {
    meter: Arc<Meter>,
    drawing: std::thread::JoinHandle<Result<()>>,
}

impl Dashboard {
    pub fn meter(&self) -> Arc<Meter> {
        self.meter.clone()
    }

    /// Logs that the model has loaded.
    pub fn loaded(&self, info: &ModelInfo) {
        self.meter.log(
            phobos_base::log::Level::Info,
            format!("loaded {} on {}", info.label, info.backend),
        );
    }

    /// Wait for the dashboard to finish, which it does once the meter stops.
    pub fn join(self) -> Result<()> {
        match self.drawing.join() {
            Ok(drawn) => drawn,
            // The panic already printed and `Screen::drop` restored the
            // terminal.
            Err(_) => Ok(()),
        }
    }
}

/// Take the terminal and start drawing, before there is anything to draw.
///
/// Fails softly. If the terminal cannot be taken over, this says why and
/// switches the meter to echo on stderr, so serving carries on without a
/// dashboard.
pub fn start() -> Result<Dashboard> {
    let meter = Arc::new(Meter::new());

    // Route runtime logs and progress to the meter, so nothing is printed
    // over the dashboard. The sinks stay installed, so from here on the ring
    // is the only copy of those lines.
    let logging = meter.clone();
    phobos_base::log::set_sink(move |level, text| logging.log(level, text));
    let stepping = meter.clone();
    phobos_base::progress::set_sink(move |event| match event {
        phobos_base::progress::Event::Started { item, .. } => stepping.starting(item),
        phobos_base::progress::Event::Finished(step) => stepping.stepped(step),
    });

    // Taking over the terminal needs a console on both stdin and stdout, which
    // is not always there. Find out before the model loads, so a failure is
    // reported up front.
    let (ready, started) = std::sync::mpsc::sync_channel(1);
    let screen = meter.clone();
    let drawing = std::thread::spawn(move || run(screen, ready));
    match started.recv_timeout(START) {
        Ok(None) => {}
        Ok(Some(why)) => {
            meter.echo_to_stderr();
            eprintln!("no dashboard: {why}. Carrying on without one.");
        }
        Err(_) => {
            meter.echo_to_stderr();
            eprintln!("no dashboard: it did not open within {START:?}. Carrying on without one.");
        }
    }
    Ok(Dashboard { meter, drawing })
}

/// Variables that enable reports written straight to stderr, bypassing the
/// logger. They would draw over the dashboard.
///
/// Presence turns them on, not the value, so `PHOBOS_VRAM=0` counts too.
const RAW_REPORTS: &[&str] = &["PHOBOS_VRAM", "PHOBOS_PASS_REPORT"];

/// The first of [`RAW_REPORTS`] that is set, if any.
pub fn raw_report() -> Option<&'static str> {
    RAW_REPORTS
        .iter()
        .copied()
        .find(|name| std::env::var_os(name).is_some())
}

/// Why a dashboard would not be drawn here, if it would not.
///
/// Returns a reason rather than a flag, so the caller can tell the user.
pub fn unavailable() -> Option<String> {
    if let Some(name) = raw_report() {
        return Some(format!(
            "{name} is set, and its report goes straight to stderr, over the top of anything drawn"
        ));
    }
    // Redirected output must not get escape codes.
    if !io::stdout().is_terminal() {
        return Some("stdout is not a terminal".to_string());
    }
    None
}

/// Draw until the user quits or the engine stops.
///
/// Returns once either happens, with the terminal restored.
///
/// `ready` receives exactly one message before anything is drawn: `None` if
/// the terminal was taken over, or the reason it could not be.
pub fn run(meter: Arc<Meter>, ready: SyncSender<Option<String>>) -> Result<()> {
    let mut screen = match Screen::enter(meter.clone()) {
        Ok(screen) => {
            let _ = ready.send(None);
            screen
        }
        Err(e) => {
            // Not fatal: serving goes on without a dashboard.
            let _ = ready.send(Some(format!("{e:#}")));
            return Ok(());
        }
    };
    let mut view = View::new();

    while meter.running() {
        let snapshot = meter.snapshot();
        screen
            .terminal
            .draw(|frame| panels::render(frame, &mut view, &snapshot))?;
        view.tick();

        // Wait up to one frame for a keypress.
        if !event::poll(FRAME)? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                // Raw mode swallows the interrupt, so handle Ctrl-C here.
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                KeyCode::Char('c') => meter.clear_log(),
                KeyCode::Char('r') => view.reset_peaks(),
                KeyCode::Char('p') => view.replay_splash(),
                _ => {}
            },
            _ => {}
        }
    }

    Ok(())
}

/// The terminal in raw mode on the alternate screen, for as long as this
/// lives.
///
/// Both settings are process-wide and outlive a panic, so they must be
/// restored on every exit path. [`Drop`] covers normal exits and the panic
/// hook covers the rest.
///
/// Dropping also stops the engine, since without the dashboard nothing can
/// reach it from the keyboard.
struct Screen {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    meter: Arc<Meter>,
}

impl Screen {
    fn enter(meter: Arc<Meter>) -> Result<Screen> {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // Restore first, or the message prints onto the alternate screen.
            restore();
            previous(info);
        }));

        enable_raw_mode()?;
        io::stdout().execute(EnterAlternateScreen)?;
        let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        terminal.hide_cursor()?;
        terminal.clear()?;
        Ok(Screen { terminal, meter })
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        self.meter.stop();
        restore();
    }
}

/// Put the terminal back. Every step is best effort, since this often runs
/// while unwinding and a second panic there would abort.
fn restore() {
    let _ = disable_raw_mode();
    let _ = io::stdout().execute(LeaveAlternateScreen);
    let _ = io::stdout().execute(ratatui::crossterm::cursor::Show);
}
