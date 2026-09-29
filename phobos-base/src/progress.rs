// Progress reports for long startup steps, mainly kernel compilation.
//
// Like the logger's sink, a library reports and a front end decides whether
// to show it. Nothing here formats, and reporting costs nothing when no sink
// is installed.

use std::sync::OnceLock;
use std::time::Duration;

/// One unit of startup work that has just finished.
#[derive(Clone, Copy, Debug)]
pub struct Step<'a> {
    /// The kind of work, for grouping or labelling.
    pub stage: &'a str,
    /// What was finished, such as a kernel's name.
    pub item: &'a str,
    /// How many of this batch are done, counting this one.
    pub done: usize,
    /// How many the batch holds. Never zero: a single piece of work is a
    /// batch of one.
    pub total: usize,
    /// Whether it was loaded from the cache rather than built.
    pub cached: bool,
    /// How long this kernel's lowering took, measured in the thread that did
    /// it, not the whole batch. Zero for a cached step.
    pub took: Duration,
    /// The kernel's source and the PTX it became, borrowed for the call only.
    /// A display copies just the part it has room for.
    pub source: &'a str,
    pub ptx: &'a str,
}

impl Step<'_> {
    /// Characters of kernel text this step put through the compiler.
    pub fn source_bytes(&self) -> usize {
        self.source.len()
    }

    /// Characters of PTX it produced.
    pub fn ptx_bytes(&self) -> usize {
        self.ptx.len()
    }
}

/// A progress event from a long task.
///
/// Both start and finish are reported. A batch runs its jobs concurrently but
/// joins them in order, so only a start says what is being worked on right
/// now.
pub enum Event<'a> {
    /// Work has begun. Not reported for a cache hit, which only finishes.
    Started { stage: &'a str, item: &'a str },
    /// Work has finished, with what it cost and what it produced.
    Finished(Step<'a>),
}

type Sink = Box<dyn for<'a> Fn(Event<'a>) + Send + Sync>;

static SINK: OnceLock<Sink> = OnceLock::new();

/// Send every step from here on to `sink`.
///
/// The first sink wins, and the return says whether this call installed it.
/// A second caller is a bug, not a fallback.
pub fn set_sink(sink: impl for<'a> Fn(Event<'a>) + Send + Sync + 'static) -> bool {
    SINK.set(Box::new(sink)).is_ok()
}

pub fn started(stage: &str, item: &str) {
    emit(Event::Started { stage, item });
}

pub fn report(step: Step<'_>) {
    emit(Event::Finished(step));
}

/// A no-op unless a sink is installed.
fn emit(event: Event<'_>) {
    if let Some(sink) = SINK.get() {
        sink(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn a_step_reaches_the_sink_that_was_installed() {
        static SEEN: AtomicUsize = AtomicUsize::new(0);
        static LAST: Mutex<String> = Mutex::new(String::new());

        // Reporting before a sink exists is silent.
        report(Step {
            stage: "kernels",
            item: "before",
            done: 1,
            total: 1,
            cached: true,
            took: Duration::ZERO,
            source: "",
            ptx: "",
        });
        assert_eq!(SEEN.load(Ordering::Relaxed), 0);

        static STARTED: AtomicUsize = AtomicUsize::new(0);
        assert!(set_sink(|event| match event {
            Event::Started { item, .. } => {
                STARTED.fetch_add(1, Ordering::Relaxed);
                *LAST.lock().unwrap() = item.to_string();
            }
            Event::Finished(step) => {
                SEEN.fetch_add(1, Ordering::Relaxed);
                *LAST.lock().unwrap() = step.item.to_string();
                assert_eq!(step.source_bytes(), step.source.len());
            }
        }));
        // The sink is set once, so a second attempt fails.
        assert!(!set_sink(|_| {}));

        report(Step {
            stage: "kernels",
            item: "q8_qmma",
            done: 3,
            total: 9,
            cached: false,
            took: Duration::from_millis(1500),
            source: "kernel q8_qmma() {}",
            ptx: ".visible .entry q8_qmma",
        });
        assert_eq!(SEEN.load(Ordering::Relaxed), 1);
        assert_eq!(*LAST.lock().unwrap(), "q8_qmma");

        // A start is reported before any finish.
        started("kernels", "iq1s_matvec");
        assert_eq!(STARTED.load(Ordering::Relaxed), 1);
        assert_eq!(*LAST.lock().unwrap(), "iq1s_matvec");
    }
}
