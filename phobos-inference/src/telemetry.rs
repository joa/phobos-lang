// A record of what the runtime is doing, for a viewer to display.
//
// The engine writes events as they happen. A viewer reads whole snapshots on
// its own clock. Nothing here draws or names a model format.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use phobos_base::log::Level;
use phobos_base::progress::Step;

use crate::model::{Architecture, CacheStats, DeviceInfo, DeviceMemory, Footprint, Model};

/// Samples each rate history keeps.
const HISTORY: usize = 256;

/// Lines the log ring holds. Older ones are dropped.
const LOG_LINES: usize = 256;

/// Finished requests the ring holds.
const RECENT: usize = 32;

/// Weight of the newest interval in the decode rate's moving average.
const DECODE_SMOOTHING: f64 = 0.2;

/// What the engine is doing right now.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Phase {
    #[default]
    Idle,
    /// Running a prompt.
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

/// One finished request.
#[derive(Clone, Debug)]
pub struct Request {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub reason: String,
    /// Prompt positions already in the session, which were not run again.
    pub reused: usize,
    /// Prompt tokens per second, over the prompt pass alone.
    pub prefill_rate: f64,
    /// Generated tokens per second, over the decode alone.
    pub decode_rate: f64,
    pub at: Duration,
}

/// One log line from the runtime.
#[derive(Clone, Debug)]
pub struct LogLine {
    pub level: Level,
    pub text: String,
    pub at: Duration,
}

/// Upper edge of each compile-time bucket, in milliseconds. The last one
/// catches everything slower.
pub const BUCKET_EDGES_MILLIS: [u64; 7] = [250, 500, 1_000, 2_000, 4_000, 8_000, u64::MAX];

/// How much of a kernel's text and of its PTX to keep for display, a few
/// screens' worth.
const EXCERPT_BYTES: usize = 8 << 10;

/// Progress of the startup work done before a model exists, mostly kernel
/// compilation.
#[derive(Clone, Debug, Default)]
pub struct Loading {
    pub stage: String,
    /// What was finished most recently.
    pub item: String,
    /// Position in the batch of the last item. Work that arrives one piece at
    /// a time reports a batch of one.
    pub done: usize,
    pub total: usize,
    /// Kernels built from source, and kernels found already built.
    pub built: u64,
    pub cached: u64,
    pub elapsed: Duration,
    /// Kernels built, by how long they took. See [`BUCKET_EDGES_MILLIS`].
    pub buckets: [u64; BUCKET_EDGES_MILLIS.len()],
    /// Kernel source and PTX sizes, counted over built kernels only.
    pub source_bytes: u64,
    pub ptx_bytes: u64,
    /// Summed lowering time. Can exceed the wall clock, since a batch lowers
    /// on every core at once.
    pub compile_time: Duration,
    /// The slowest kernel so far.
    pub slowest: String,
    pub slowest_took: Duration,
    /// The last kernel built: its text, its PTX, and how long it took. Both
    /// texts are cut to [`EXCERPT_BYTES`].
    pub source: String,
    pub ptx: String,
    pub took: Duration,
    /// Kernels still lowering, longest first, with how long each has run.
    ///
    /// The head is the one the batch is waiting on.
    pub in_flight: Vec<(String, Duration)>,
}

impl Loading {
    /// Everything finished so far, across all batches.
    pub fn finished(&self) -> u64 {
        self.built + self.cached
    }

    /// Progress through the current batch, or `None` for a batch of one.
    pub fn ratio(&self) -> Option<f64> {
        (self.total > 1).then(|| self.done as f64 / self.total as f64)
    }

    /// The kernel that has been lowering the longest.
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

    /// PTX bytes produced per byte of kernel source, over what was built.
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

/// Everything fixed at load, copied so a viewer need not hold the model.
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
    /// Everything a model reports about itself that never changes.
    ///
    /// Call once. Finding the driver version may spawn a subprocess.
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
    /// Startup progress.
    pub loading: Option<Loading>,
    /// Positions the live session holds, and the cache bytes reserved for them.
    pub cache_tokens: usize,
    pub cache_bytes: Option<u64>,
    pub prefill_rate: f64,
    pub decode_rate: f64,
    pub prefill_history: Vec<f64>,
    pub decode_history: Vec<f64>,
    pub requests: u64,
    pub prompt_tokens: u64,
    /// The part of [`Snapshot::prompt_tokens`] a kept session already held.
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
/// Every method takes `&self`, so it can be shared behind an [`Arc`]. The
/// lock is held only for field updates, never across a pass.
///
/// [`Arc`]: std::sync::Arc
pub struct Meter {
    running: AtomicBool,
    /// Whether log lines also go to stderr. Off while a full-screen viewer
    /// owns the terminal.
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

    /// Recovers from poisoning, since a display fault should not stop serving.
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether the engine should keep serving. A viewer clears this when the
    /// user asks to quit.
    pub fn running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Print every log line to stderr as well as keeping it. For a caller
    /// with no viewer.
    pub fn echo_to_stderr(&self) {
        self.echo.store(true, Ordering::Relaxed);
    }

    pub fn describe(&self, fixed: Fixed) {
        let mut inner = self.lock();
        inner.fixed = fixed;
        // Loading ends once there is a model. Kernels compiled later still
        // count, but no longer show a loading bar.
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

        // Remove only the first match by name. Several kernels can share a
        // name, but never compile at the same time.
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

    /// What the kept session holds after a request, or zero and `None` if it
    /// was dropped.
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
        // The ring keeps every level. Only the stderr echo is filtered by
        // PHOBOS_LOG.
        if self.echo.load(Ordering::Relaxed) && phobos_base::log::enabled(level) {
            eprintln!("[{:>7.3}s] {text}", at.as_secs_f64());
        }
        push_capped(&mut inner.log, LogLine { level, text, at }, LOG_LINES);
    }

    /// Drop every log line held.
    pub fn clear_log(&self) {
        self.lock().log.clear();
    }

    /// A request has begun. `reused` of its `prompt_tokens` were already in
    /// the kept session and are not run.
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

    /// The prompt pass is under way, `tokens` positions run in `took`. Only
    /// [`Meter::prefilled`] records the rate in the history.
    pub fn prefilling(&self, tokens: usize, took: Duration) {
        let rate = rate_of(tokens, took);
        let mut inner = self.lock();
        inner.prefill_rate = rate;
        if let Some(live) = inner.live.as_mut() {
            live.prefill_rate = rate;
        }
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
        // The first token has no interval, since the gap back to the prompt
        // pass is not a decode step.
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

    /// Close the live request and return its record.
    pub fn request_finished(&self, reason: &str) -> Option<Request> {
        let mut inner = self.lock();
        inner.phase = Phase::Idle;
        let at = inner.started.elapsed();
        let live = inner.live.take()?;
        // Over the decode alone. A span that produced n tokens covers n
        // steps: the first token came with the prompt pass, and the last
        // step's token was not emitted.
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
        // Cache figures are left alone. The caller reports whether the
        // session was kept.
        Some(done)
    }

    pub fn snapshot(&self) -> Snapshot {
        let inner = self.lock();
        // Longest first.
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

/// `text` cut to [`EXCERPT_BYTES`] characters.
fn excerpt(text: &str) -> String {
    match text.char_indices().nth(EXCERPT_BYTES) {
        Some((at, _)) => text[..at].to_string(),
        None => text.to_string(),
    }
}

/// Tokens per second, or zero for a zero-length interval.
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
