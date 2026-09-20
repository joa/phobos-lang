use std::fmt;
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    Off,
    Info,
    Debug,
    Trace,
}

impl Level {
    fn tag(self) -> &'static str {
        match self {
            Level::Off => "OFF",
            Level::Info => "INFO",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
        }
    }
}

fn configured() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("PHOBOS_LOG").ok().as_deref() {
        Some("off") | Some("none") | Some("0") => Level::Off,
        Some("debug") => Level::Debug,
        Some("trace") => Level::Trace,
        _ => Level::Info,
    })
}

fn started() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

pub fn enabled(level: Level) -> bool {
    level != Level::Off && configured() >= level
}

/// Where a line goes instead of stderr.
type Sink = Box<dyn Fn(Level, &str) + Send + Sync>;

static SINK: OnceLock<Sink> = OnceLock::new();

/// Route every line from here on to `sink` rather than to stderr.
///
/// A full-screen caller owns the terminal, so a stray `eprintln!` from inside
/// a pass would land on top of whatever it has drawn. Installing a sink is how
/// such a caller collects those lines instead. First one wins, and the return
/// says whether this call was it; a second caller is a bug rather than a
/// fallback, since the two would disagree about who owns the screen.
pub fn set_sink(sink: impl Fn(Level, &str) + Send + Sync + 'static) -> bool {
    SINK.set(Box::new(sink)).is_ok()
}

pub fn emit(level: Level, args: fmt::Arguments) {
    if let Some(sink) = SINK.get() {
        // The sink renders its own prefix: it knows how wide it can draw.
        sink(level, &args.to_string());
        return;
    }
    eprintln!(
        "[{:>7.3}s {:>5}] {}",
        started().elapsed().as_secs_f64(),
        level.tag(),
        args
    );
}

#[macro_export]
macro_rules! phlog {
    ($lvl:expr, $($arg:tt)*) => {
        if $crate::log::enabled($lvl) {
            $crate::log::emit($lvl, format_args!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! phinfo {
    ($($a:tt)*) => { $crate::phlog!($crate::log::Level::Info, $($a)*) };
}

#[macro_export]
macro_rules! phdebug {
    ($($a:tt)*) => { $crate::phlog!($crate::log::Level::Debug, $($a)*) };
}

#[macro_export]
macro_rules! phtrace {
    ($($a:tt)*) => { $crate::phlog!($crate::log::Level::Trace, $($a)*) };
}
