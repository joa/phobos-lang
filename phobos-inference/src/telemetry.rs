// What the runtime is doing, for something that displays it.
//
// The engine writes events as they happen and a viewer reads whole snapshots
// on its own clock, so the two never have to run at the same rate. Nothing
// here draws, and nothing here names a model format: a viewer is one more
// reader of the same traits the runtime already speaks.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use phobos_base::log::Level;
use phobos_base::progress::Step;

use crate::model::{Architecture, CacheStats, DeviceInfo, DeviceMemory, Footprint, Model};

/// Samples each rate history keeps. At one per token a decode of any length
/// fills it, and the viewer plots however much of it fits.
const HISTORY: usize = 256;

/// Lines the log ring holds. Older ones are dropped: a viewer that is behind
/// wants the recent end, not the start.
const LOG_LINES: usize = 256;

/// Finished requests the ring holds.
const RECENT: usize = 32;

/// Weight of the newest interval in the decode rate's moving average. Low
/// enough that one slow token does not make the figure jump, high enough that
/// the figure still follows a real change within a few tokens.
const DECODE_SMOOTHING: f64 = 0.2;

/// What the engine is doing right now.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Phase {
    #[default]
    Idle,
    /// Running a prompt, which is one call however many tokens it carries.
    Prefill,
    /// Producing tokens, one call each.
    Decode,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Phase::Idle => "IDLE",
            Phase::Prefill => "PREFILL",
            Phase::Decode => "DECODE",
        }
    }
}

/// One finished request, as the log of them keeps it.
#[derive(Clone, Debug)]
pub struct Request {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub reason: String,
    /// Prompt positions that were already in the session and did not have to
    /// be run again.
    pub reused: usize,
    /// Prompt tokens a second, over the prompt pass alone.
    pub prefill_rate: f64,
    /// Generated tokens a second, over the decode alone. The prompt pass is
    /// excluded on purpose: the two differ by an order of magnitude and an
    /// average over both describes neither.
    pub decode_rate: f64,
    pub at: Duration,
}

/// One line something in the runtime wanted to say.
#[derive(Clone, Debug)]
pub struct LogLine {
    pub level: Level,
    pub text: String,
    pub at: Duration,
}

/// Upper edge of each compile-time bucket, in milliseconds, the last one
/// standing for everything slower.
pub const BUCKET_EDGES_MILLIS: [u64; 7] = [250, 500, 1_000, 2_000, 4_000, 8_000, u64::MAX];

/// How much of a kernel's text and of its PTX to keep for a display.
///
/// A few screens' worth. The whole of either runs to tens of kilobytes, and
/// nothing can show that much before the next kernel replaces it.
const EXCERPT_BYTES: usize = 8 << 10;

/// How far a load has got.
///
/// Kept separate from the caches a running engine reports: this is the work
/// done before there is a model at all, and on a cold start it is nearly all
/// of the time a user waits.
#[derive(Clone, Debug, Default)]
pub struct Loading {
    pub stage: String,
    /// What was finished most recently.
    pub item: String,
    /// Position in the batch the last item belonged to. Work that arrives one
    /// piece at a time reports a batch of one, so a bar is only worth drawing
    /// when `total` is more than that.
    pub done: usize,
    pub total: usize,
    /// Built from source, and found already built. The first number is what
    /// makes a cold start slow.
    pub built: u64,
    pub cached: u64,
    pub elapsed: Duration,
    /// Kernels built, by how long they took. See [`BUCKET_EDGES_MILLIS`].
    pub buckets: [u64; BUCKET_EDGES_MILLIS.len()],
    /// Characters put through the compiler, and characters of PTX that came
    /// out. Counted over the kernels built, since a cached one compiled
    /// nothing this run.
    pub source_bytes: u64,
    pub ptx_bytes: u64,
    /// Summed lowering time. Larger than the wall clock, because a batch
    /// lowers on every core at once.
    pub compile_time: Duration,
    /// The slowest kernel so far, which is the one worth knowing about.
    pub slowest: String,
    pub slowest_took: Duration,
    /// The last kernel built: its text, the PTX it became, and what it cost.
    /// Both are cut to [`EXCERPT_BYTES`].
    pub source: String,
    pub ptx: String,
    pub took: Duration,
    /// Kernels whose lowering has begun and not finished, longest first, with
    /// how long each has been going.
    ///
    /// A batch starts all of them at once, so for most of a cold start this
    /// is most of the batch. The head of it is the one to know about: the
    /// batch cannot finish before its slowest member does, so that kernel is
    /// what the wait is actually for.
    pub in_flight: Vec<(String, Duration)>,
}

impl Loading {
    /// Everything finished so far, however many batches it took.
    pub fn finished(&self) -> u64 {
        self.built + self.cached
    }

    /// How far through the current batch, or nothing when the batch is one
    /// item and a bar would say nothing.
    pub fn ratio(&self) -> Option<f64> {
        (self.total > 1).then(|| self.done as f64 / self.total as f64)
    }

    /// The kernel that has been lowering the longest, which is the one the
    /// rest of the batch is waiting on.
    pub fn longest(&self) -> Option<&(String, Duration)> {
        self.in_flight.first()
    }

    /// The buckets with their labels, for a histogram.
    pub fn histogram(&self) -> Vec<(String, u64)> {
        let mut out = Vec::with_capacity(self.buckets.len());
        let mut low = 0u64;
        for (i, &count) in self.buckets.iter().enumerate() {
            let high = BUCKET_EDGES_MILLIS[i];
            out.push((
                if high == u64::MAX {
                    format!("{}s +", low / 1000)
                } else if high < 1000 {
                    format!("< {high} ms")
                } else {
                    format!("< {} s", high / 1000)
                },
                count,
            ));
            low = high;
        }
        out
    }

    /// PTX characters produced per character of kernel text, over what has
    /// been built. One line of this language becomes a great many of PTX,
    /// and how many is the interesting part.
    pub fn expansion(&self) -> Option<f64> {
        (self.source_bytes > 0).then(|| self.ptx_bytes as f64 / self.source_bytes as f64)
    }
}

/// A request that has started and not finished.
#[derive(Clone, Copy, Debug)]
pub struct Active {
    pub prompt_tokens: usize,
    pub reused: usize,
    pub produced: usize,
    pub elapsed: Duration,
}

/// Everything fixed at load: named once so a viewer need not hold the model.
#[derive(Clone, Debug, Default)]
pub struct Fixed {
    pub label: String,
    pub backend: String,
    pub vocab_size: usize,
    pub context_limit: usize,
    pub listen: Option<String>,
    pub footprint: Option<Footprint>,
    pub architecture: Option<Architecture>,
    pub card: Option<DeviceInfo>,
}

impl Fixed {
    /// Everything a model can say about itself that will not change again.
    ///
    /// Asked once, on the way into serving: finding the card's driver version
    /// may cost a subprocess, and none of this moves afterwards.
    pub fn of(model: &dyn Model, listen: Option<&str>) -> Fixed {
        let info = model.info();
        Fixed {
            label: info.label.clone(),
            backend: info.backend.to_string(),
            vocab_size: info.vocab_size,
            context_limit: info.context_limit,
            listen: listen.map(str::to_string),
            footprint: model.footprint(),
            architecture: model.architecture(),
            card: model.device_info(),
        }
    }
}

/// A consistent read of the whole meter.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub uptime: Duration,
    pub phase: Phase,
    pub fixed: Fixed,
    pub device: Option<DeviceMemory>,
    pub caches: Option<CacheStats>,
    /// What the load is doing, until there is a model to describe.
    pub loading: Option<Loading>,
    /// Positions the live session holds, and what its caches reserved for them.
    pub cache_tokens: usize,
    pub cache_bytes: Option<u64>,
    pub prefill_rate: f64,
    pub decode_rate: f64,
    pub prefill_history: Vec<f64>,
    pub decode_history: Vec<f64>,
    pub requests: u64,
    pub prompt_tokens: u64,
    /// Of [`Snapshot::prompt_tokens`], those a kept session already held.
    pub prompt_reused: u64,
    pub completion_tokens: u64,
    pub active: Option<Active>,
    pub recent: Vec<Request>,
    pub log: Vec<LogLine>,
}

struct Live {
    prompt_tokens: usize,
    reused: usize,
    produced: usize,
    started: Instant,
    /// When the prompt pass ended, so the decode rate measures decoding only.
    decode_started: Option<Instant>,
    last_token: Option<Instant>,
    prefill_rate: f64,
}

struct Inner {
    started: Instant,
    /// Lowerings begun and not finished, in the order they began.
    in_flight: Vec<(String, Instant)>,
    phase: Phase,
    fixed: Fixed,
    device: Option<DeviceMemory>,
    caches: Option<CacheStats>,
    loading: Option<Loading>,
    cache_tokens: usize,
    cache_bytes: Option<u64>,
    prefill_rate: f64,
    decode_rate: f64,
    prefill_history: VecDeque<f64>,
    decode_history: VecDeque<f64>,
    requests: u64,
    prompt_tokens: u64,
    prompt_reused: u64,
    completion_tokens: u64,
    live: Option<Live>,
    recent: VecDeque<Request>,
    log: VecDeque<LogLine>,
}

/// The shared record of what the engine is doing.
///
/// Every method takes `&self` so the engine can hold one behind an [`Arc`] and
/// write to it from wherever the work happens. The lock is held for a field
/// update and never across a pass, so a decode at any rate a card can reach
/// does not contend on it.
///
/// [`Arc`]: std::sync::Arc
pub struct Meter {
    running: AtomicBool,
    /// Whether a line written here should also reach stderr. Off while a
    /// full-screen viewer owns the terminal, on when nothing else would ever
    /// show the line.
    echo: AtomicBool,
    inner: Mutex<Inner>,
}

impl Default for Meter {
    fn default() -> Meter {
        Meter::new()
    }
}

impl Meter {
    pub fn new() -> Meter {
        Meter {
            running: AtomicBool::new(true),
            echo: AtomicBool::new(false),
            inner: Mutex::new(Inner {
                started: Instant::now(),
                in_flight: Vec::new(),
                phase: Phase::Idle,
                fixed: Fixed::default(),
                device: None,
                caches: None,
                loading: None,
                cache_tokens: 0,
                cache_bytes: None,
                prefill_rate: 0.0,
                decode_rate: 0.0,
                prefill_history: VecDeque::new(),
                decode_history: VecDeque::new(),
                requests: 0,
                prompt_tokens: 0,
                prompt_reused: 0,
                completion_tokens: 0,
                live: None,
                recent: VecDeque::new(),
                log: VecDeque::new(),
            }),
        }
    }

    /// A poisoned meter is a display fault and not a reason to stop serving,
    /// so every path recovers the guard rather than unwrapping it.
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether the engine should keep serving. A viewer clears this when the
    /// user asks to quit, which is the only way a full-screen front end can
    /// say so: it has the keyboard.
    pub fn running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Print every line to stderr as well as keeping it.
    ///
    /// For a caller with no viewer: the ring alone would mean a server that
    /// says nothing at all, and the engine's lines are the only account of
    /// what it did.
    pub fn echo_to_stderr(&self) {
        self.echo.store(true, Ordering::Relaxed);
    }

    pub fn describe(&self, fixed: Fixed) {
        let mut inner = self.lock();
        inner.fixed = fixed;
        // Loading is over the moment there is a model to describe. Kernels
        // compiled later, on the first pass that wants a shape nothing has
        // built yet, keep counting but no longer hold up a screen.
        if let Some(loading) = inner.loading.as_mut() {
            loading.done = 0;
            loading.total = 0;
        }
    }

    /// One piece of startup work has begun. See [`phobos_base::progress`].
    pub fn starting(&self, item: &str) {
        let mut inner = self.lock();
        let at = Instant::now();
        inner.in_flight.push((item.to_string(), at));
    }

    /// One piece of startup work finished. See [`phobos_base::progress`].
    pub fn stepped(&self, step: Step<'_>) {
        let mut inner = self.lock();
        let started = inner.started;
        let loading = inner.loading.get_or_insert_with(Loading::default);
        loading.stage = step.stage.to_string();
        loading.item = step.item.to_string();
        loading.done = step.done;
        loading.total = step.total;
        loading.elapsed = started.elapsed();
        if step.cached {
            loading.cached += 1;
            return;
        }

        // By name and only the first: the sequential phase compiles several
        // kernels under one name, but never two at the same time.
        if let Some(at) = inner
            .in_flight
            .iter()
            .position(|(name, _)| name == step.item)
        {
            inner.in_flight.remove(at);
        }
        let loading = inner.loading.get_or_insert_with(Loading::default);
        loading.built += 1;
        loading.took = step.took;
        loading.compile_time += step.took;
        loading.source_bytes += step.source_bytes() as u64;
        loading.ptx_bytes += step.ptx_bytes() as u64;
        let millis = step.took.as_millis() as u64;
        let bucket = BUCKET_EDGES_MILLIS
            .iter()
            .position(|&edge| millis < edge)
            .unwrap_or(BUCKET_EDGES_MILLIS.len() - 1);
        loading.buckets[bucket] += 1;
        if step.took > loading.slowest_took {
            loading.slowest_took = step.took;
            loading.slowest = step.item.to_string();
        }
        loading.source = excerpt(step.source);
        loading.ptx = excerpt(step.ptx);
    }

    /// What a kept session holds now that a request is over, or nothing when
    /// it was dropped.
    pub fn set_cache(&self, tokens: usize, bytes: Option<u64>) {
        let mut inner = self.lock();
        inner.cache_tokens = tokens;
        inner.cache_bytes = bytes;
    }

    pub fn set_cache_stats(&self, caches: Option<CacheStats>) {
        if caches.is_some() {
            self.lock().caches = caches;
        }
    }

    pub fn set_device_memory(&self, memory: Option<DeviceMemory>) {
        if let Some(memory) = memory {
            self.lock().device = Some(memory);
        }
    }

    pub fn log(&self, level: Level, text: impl Into<String>) {
        let text = text.into();
        let mut inner = self.lock();
        let at = inner.started.elapsed();
        // The ring keeps everything, since a viewer renders it with its own
        // styling and can be asked for more detail. Only stderr is filtered,
        // because that is the stream PHOBOS_LOG was set to configure.
        if self.echo.load(Ordering::Relaxed) && phobos_base::log::enabled(level) {
            eprintln!("[{:>7.3}s] {text}", at.as_secs_f64());
        }
        push_capped(&mut inner.log, LogLine { level, text, at }, LOG_LINES);
    }

    /// Drop every line held. A viewer offers this because the ring is the
    /// only place these lines exist once a full-screen front end owns the
    /// terminal.
    pub fn clear_log(&self) {
        self.lock().log.clear();
    }

    /// A request has begun, `reused` of whose `prompt_tokens` a kept session
    /// already held and which therefore never reach a pass.
    pub fn request_started(&self, prompt_tokens: usize, reused: usize) {
        let mut inner = self.lock();
        inner.phase = Phase::Prefill;
        inner.requests += 1;
        inner.prompt_tokens += prompt_tokens as u64;
        inner.prompt_reused += reused as u64;
        inner.decode_rate = 0.0;
        inner.live = Some(Live {
            prompt_tokens,
            reused,
            produced: 0,
            started: Instant::now(),
            decode_started: None,
            last_token: None,
            prefill_rate: 0.0,
        });
    }

    /// The prompt pass finished, having run `tokens` positions in `took`.
    pub fn prefilled(&self, tokens: usize, took: Duration) {
        let rate = rate_of(tokens, took);
        let mut inner = self.lock();
        inner.phase = Phase::Decode;
        inner.prefill_rate = rate;
        push_capped(&mut inner.prefill_history, rate, HISTORY);
        if let Some(live) = inner.live.as_mut() {
            live.prefill_rate = rate;
            live.decode_started = Some(Instant::now());
        }
    }

    /// One token was produced, leaving the session holding `cache_tokens`
    /// positions in `cache_bytes` of cache.
    pub fn token(&self, cache_tokens: usize, cache_bytes: Option<u64>) {
        let now = Instant::now();
        let mut inner = self.lock();
        inner.phase = Phase::Decode;
        inner.cache_tokens = cache_tokens;
        inner.cache_bytes = cache_bytes;
        inner.completion_tokens += 1;

        let Some(live) = inner.live.as_mut() else {
            return;
        };
        live.produced += 1;
        // For the moving average, and only for it: the gap back to the prompt
        // pass is the prompt's cost rather than this token's.
        let interval = live.last_token.map(|last| now.duration_since(last));
        live.last_token = Some(now);
        let Some(rate) = interval.map(|d| rate_of(1, d)) else {
            return;
        };
        inner.decode_rate = if inner.decode_rate > 0.0 {
            inner.decode_rate + DECODE_SMOOTHING * (rate - inner.decode_rate)
        } else {
            rate
        };
        let smoothed = inner.decode_rate;
        push_capped(&mut inner.decode_history, smoothed, HISTORY);
    }

    /// Close the live request and hand back what was recorded, so a caller
    /// can report it without reading the whole snapshot back.
    pub fn request_finished(&self, reason: &str) -> Option<Request> {
        let mut inner = self.lock();
        inner.phase = Phase::Idle;
        let at = inner.started.elapsed();
        let live = inner.live.take()?;
        // Over the decode alone, so it is comparable with the live figure and
        // with what a decode benchmark reports. Every token is followed by one
        // more step, so a span that produced n of them covers n steps: the
        // first token came free with the prompt pass and the last step's token
        // was not emitted, which cancel.
        let decode_rate = match live.decode_started {
            Some(from) if live.produced > 0 => rate_of(live.produced, from.elapsed()),
            _ => 0.0,
        };
        let done = Request {
            prompt_tokens: live.prompt_tokens,
            completion_tokens: live.produced,
            reason: reason.to_string(),
            prefill_rate: live.prefill_rate,
            decode_rate,
            reused: live.reused,
            at,
        };
        push_capped(&mut inner.recent, done.clone(), RECENT);
        // The caches are not cleared here: a session may be kept for the next
        // request, and whether it was is the caller's to report.
        Some(done)
    }

    pub fn snapshot(&self) -> Snapshot {
        let inner = self.lock();
        // Longest first: the head is the one the batch is waiting on.
        let mut in_flight: Vec<(String, Duration)> = inner
            .in_flight
            .iter()
            .map(|(name, at)| (name.clone(), at.elapsed()))
            .collect();
        in_flight.sort_unstable_by_key(|(_, waiting)| std::cmp::Reverse(*waiting));
        Snapshot {
            uptime: inner.started.elapsed(),
            phase: inner.phase,
            fixed: inner.fixed.clone(),
            device: inner.device,
            caches: inner.caches,
            loading: inner.loading.clone().map(|mut loading| {
                loading.in_flight = in_flight;
                loading
            }),
            cache_tokens: inner.cache_tokens,
            cache_bytes: inner.cache_bytes,
            prefill_rate: inner.prefill_rate,
            decode_rate: inner.decode_rate,
            prefill_history: inner.prefill_history.iter().copied().collect(),
            decode_history: inner.decode_history.iter().copied().collect(),
            requests: inner.requests,
            prompt_tokens: inner.prompt_tokens,
            prompt_reused: inner.prompt_reused,
            completion_tokens: inner.completion_tokens,
            active: inner.live.as_ref().map(|live| Active {
                prompt_tokens: live.prompt_tokens,
                reused: live.reused,
                produced: live.produced,
                elapsed: live.started.elapsed(),
            }),
            recent: inner.recent.iter().rev().cloned().collect(),
            log: inner.log.iter().rev().cloned().collect(),
        }
    }
}

/// As much of `text` as a display could use, cut on a character boundary so
/// what is kept is still a string.
fn excerpt(text: &str) -> String {
    match text.char_indices().nth(EXCERPT_BYTES) {
        Some((at, _)) => text[..at].to_string(),
        None => text.to_string(),
    }
}

/// Tokens a second, with a zero-length interval reported as no rate at all
/// rather than as an infinite one.
fn rate_of(tokens: usize, took: Duration) -> f64 {
    let seconds = took.as_secs_f64();
    if seconds <= 0.0 {
        return 0.0;
    }
    tokens as f64 / seconds
}

fn push_capped<T>(ring: &mut VecDeque<T>, item: T, cap: usize) {
    if ring.len() == cap {
        ring.pop_front();
    }
    ring.push_back(item);
}

#[cfg(test)]
mod tests;
