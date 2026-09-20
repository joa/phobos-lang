// What a long startup step is doing, for something that shows it.
//
// The same shape as the logger's sink and for the same reason: a library
// reports, and a front end decides whether any of it reaches a screen. Nothing
// here formats, and nothing here costs anything when no one is listening.
//
// The thing worth reporting is kernel compilation. A warm start loads every
// kernel from the on-disk cache and is over in seconds; a cold one lowers each
// of them through MLIR to PTX and takes minutes, during which a process that
// says nothing looks like a process that has hung.

use std::sync::OnceLock;
use std::time::Duration;

/// One unit of startup work that has just finished.
#[derive(Clone, Copy, Debug)]
pub struct Step<'a> {
    /// The kind of work, for a display that groups or labels it.
    pub stage: &'a str,
    /// What was finished, such as a kernel's name.
    pub item: &'a str,
    /// How many of this batch are done, counting this one.
    pub done: usize,
    /// How many the batch holds. Work that arrives one piece at a time is a
    /// batch of one, so this is never zero and a caller never has to guess.
    pub total: usize,
    /// Whether it was already built and only had to be loaded. This is the
    /// difference between a start that takes seconds and one that takes
    /// minutes, so it is worth saying which happened.
    pub cached: bool,
    /// How long the lowering itself took, measured inside the thread that did
    /// it, so it is this kernel's cost and not the batch's. Zero for a step
    /// that was read back from the cache rather than built.
    pub took: Duration,
    /// The kernel's own text, and the PTX it became. Borrowed for the length
    /// of the call: a display keeps whatever slice of them it has room for
    /// rather than the whole of something measured in tens of kilobytes.
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

/// Something a long task wants to say.
///
/// Both ends are reported, not just the finish. A batch starts everything at
/// once and joins in the order the jobs were given, so a finish tells a
/// watcher how far down that order it has got, and only a start tells it what
/// is actually being worked on. A kernel that takes ten minutes would
/// otherwise go unnamed for ten minutes while the name of the last one to
/// finish sat on the screen.
pub enum Event<'a> {
    /// Work has begun. Reported for work that is really done: something read
    /// back from a cache never starts, it only finishes.
    Started { stage: &'a str, item: &'a str },
    /// Work has finished, with what it cost and what it produced.
    Finished(Step<'a>),
}

type Sink = Box<dyn for<'a> Fn(Event<'a>) + Send + Sync>;

static SINK: OnceLock<Sink> = OnceLock::new();

/// Send every step from here on to `sink`.
///
/// First one wins, and the return says whether this call was it. A second
/// caller is a bug rather than a fallback: the two would disagree about who
/// is drawing.
pub fn set_sink(sink: impl for<'a> Fn(Event<'a>) + Send + Sync + 'static) -> bool {
    SINK.set(Box::new(sink)).is_ok()
}

/// Say that `item` has been started.
pub fn started(stage: &str, item: &str) {
    emit(Event::Started { stage, item });
}

/// Report one finished step.
pub fn report(step: Step<'_>) {
    emit(Event::Finished(step));
}

/// Nothing at all unless a front end has asked to hear about it.
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

        // Reporting before a sink exists is the ordinary case and is silent.
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
        // The sink is process-wide and set once, so a second attempt fails
        // rather than replacing it.
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

        // A start is reported on its own, which is the only way a watcher
        // learns the name of something that has not finished yet.
        started("kernels", "iq1s_matvec");
        assert_eq!(STARTED.load(Ordering::Relaxed), 1);
        assert_eq!(*LAST.lock().unwrap(), "iq1s_matvec");
    }
}
