// A full-screen dashboard for the inference engine.
//
// It runs on its own thread and owns the terminal; the engine stays on the
// thread that loaded the model, since that is where its device context is.
// The two share nothing but a [`Meter`]: the engine writes events into it as
// they happen and the dashboard reads whole snapshots on its own clock, so
// neither waits for the other and a slow redraw cannot slow a decode.
//
// Quitting is the dashboard's to signal, because it has the keyboard: `q`
// clears the meter's running flag, the engine notices at its next idle tick
// and returns, and the model drops on the way out.

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

/// Frame interval. Thirty a second is enough for the rain to fall smoothly
/// and cheap enough that an idle dashboard does not show up in a profile.
const FRAME: Duration = Duration::from_millis(33);

/// How long to wait for the dashboard to take the terminal before giving up
/// on it. It either works immediately or not at all.
const START: Duration = Duration::from_secs(5);

/// A running dashboard, and the meter it is drawing.
///
/// Started before the model is loaded, so a cold start has something to watch
/// while every kernel is compiled. Until [`Dashboard::loaded`] it draws the
/// loading screen; after it, the panels.
pub struct Dashboard {
    meter: Arc<Meter>,
    drawing: std::thread::JoinHandle<Result<()>>,
}

impl Dashboard {
    pub fn meter(&self) -> Arc<Meter> {
        self.meter.clone()
    }

    /// There is a model now, so the panels have something to show.
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
            // The panic already printed and the terminal is already back; see
            // `Screen::drop`.
            Err(_) => Ok(()),
        }
    }
}

/// Take the terminal and start drawing, before there is anything to draw.
///
/// Fails softly: a terminal that cannot be taken over is a reason to serve
/// plainly, not a reason not to serve, so this reports and hands back nothing
/// rather than an error.
pub fn start() -> Result<Dashboard> {
    let meter = Arc::new(Meter::new());

    // Anything the runtime would have written to stderr goes to the meter
    // instead: a pass that logs mid-frame would otherwise land on top of the
    // dashboard. Installed before the first pass and never uninstalled, so
    // from here on the ring is the only copy of those lines.
    let logging = meter.clone();
    phobos_base::log::set_sink(move |level, text| logging.log(level, text));
    let stepping = meter.clone();
    phobos_base::progress::set_sink(move |event| match event {
        phobos_base::progress::Event::Started { item, .. } => stepping.starting(item),
        phobos_base::progress::Event::Finished(step) => stepping.stepped(step),
    });

    // Taking over the terminal needs a console on stdin as well as stdout,
    // which not every way of starting a process provides. Whether it worked
    // has to be known before the caller goes off to load a model, or a
    // dashboard that never opened would go unreported for as long as it runs.
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

/// Reports that a pass writes straight to stderr, and the variable that asks
/// for each. They do not go through the logger, so no sink can catch them and
/// they would land on top of whatever the dashboard has drawn.
///
/// Presence is what turns these on, not the value, so `PHOBOS_VRAM=0` counts
/// as asking for one.
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
/// A reason rather than a flag, because a server that quietly prints lines
/// when it was expected to draw is a thing to spend an afternoon on. The
/// caller says this out loud.
pub fn unavailable() -> Option<String> {
    if let Some(name) = raw_report() {
        return Some(format!(
            "{name} is set, and its report goes straight to stderr, over the top of anything drawn"
        ));
    }
    // Redirected output must not get escape codes: a server being piped to a
    // file wants the lines.
    if !io::stdout().is_terminal() {
        return Some("stdout is not a terminal".to_string());
    }
    None
}

/// Draw until the user quits or the engine stops.
///
/// Returns once either happens, with the terminal put back the way it was.
///
/// `ready` carries the one thing the caller cannot find out any other way:
/// whether the terminal could be taken over at all. Taking it over needs a
/// console on stdin as well as stdout, which not every way of starting a
/// process provides, and the caller is about to block serving. It is sent
/// exactly once, before anything is drawn.
pub fn run(meter: Arc<Meter>, ready: SyncSender<Option<String>>) -> Result<()> {
    let mut screen = match Screen::enter(meter.clone()) {
        Ok(screen) => {
            let _ = ready.send(None);
            screen
        }
        Err(e) => {
            // Not an error the process should end on: serving without a
            // dashboard is worth more than not serving.
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

        // One poll per frame, so a keypress is acted on within a frame and
        // the loop is otherwise asleep.
        if !event::poll(FRAME)? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                // Raw mode swallows the interrupt, so the dashboard has to
                // honour it itself or the only way out is to kill the process.
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

/// The terminal, in the state the dashboard needs it, for as long as it is
/// alive.
///
/// Raw mode and the alternate screen are process-wide and outlive a panic, so
/// putting them back has to happen on every exit path. [`Drop`] covers the
/// ordinary ones and the hook covers the rest.
///
/// Dropping also stops the engine. However the dashboard ends, quitting or
/// panicking, it was the only thing holding the keyboard, so a process that
/// kept serving after it went would have no way left to reach it.
struct Screen {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    meter: Arc<Meter>,
}

impl Screen {
    fn enter(meter: Arc<Meter>) -> Result<Screen> {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // Before the default hook, or the message prints onto a screen
            // that is about to be thrown away.
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

/// Put the terminal back. Every step is best effort: this runs while
/// unwinding as often as not, and a second failure there would abort.
fn restore() {
    let _ = disable_raw_mode();
    let _ = io::stdout().execute(LeaveAlternateScreen);
    let _ = io::stdout().execute(ratatui::crossterm::cursor::Show);
}
