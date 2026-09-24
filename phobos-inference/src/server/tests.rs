// The server's two responsibilities that no model file is needed to check:
// that it tells the meter what it loaded, and that it stops when asked.
//
// Stopping is the one worth a test. A full-screen viewer has no other way out
// than clearing the running flag, so a worker that never looks at it would
// leave the user with a dead dashboard and a live process.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::model::{DeviceMemory, Footprint, Model, ModelInfo, Session, Tokenizer};
use crate::sampling::SampleConfig;
use crate::telemetry::Meter;

use super::{Defaults, serve};

/// A model that reports a footprint and produces one token forever, so a test
/// exercises the worker without a file, a tokenizer or a device.
struct Fake {
    info: ModelInfo,
    /// How many times the card has been read, so a test can show that a
    /// generation asks at all.
    probes: std::sync::atomic::AtomicUsize,
}

struct FakeSession {
    len: usize,
}

impl Model for Fake {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        self
    }

    fn session(&self) -> Result<Box<dyn Session + '_>> {
        Ok(Box::new(FakeSession { len: 0 }))
    }

    fn footprint(&self) -> Option<Footprint> {
        Some(Footprint {
            weight_bytes: 1 << 30,
            dense_bytes: 0,
            streamed_bytes: 0,
            kv_bytes_per_token: 1 << 10,
        })
    }

    fn device_memory(&self) -> Option<DeviceMemory> {
        self.probes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(DeviceMemory {
            free_bytes: 2 << 30,
            total_bytes: 8 << 30,
        })
    }
}

impl Tokenizer for Fake {
    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        Ok(text.bytes().map(|b| i64::from(b) % 4 + 1).collect())
    }

    fn decode_bytes(&self, _ids: &[i64]) -> Vec<u8> {
        b"x".to_vec()
    }

    fn is_eog(&self, id: i64) -> bool {
        id == 0
    }
}

impl Session for FakeSession {
    fn extend(&mut self, ids: &[i64]) -> Result<Vec<f32>> {
        self.len += ids.len();
        // Token 1 always wins, so nothing ever reaches the end-of-turn id and
        // a generation is bounded only by its token limit.
        Ok(vec![0.0, 1.0, 0.0, 0.0, 0.0])
    }

    fn len(&self) -> usize {
        self.len
    }

    fn cache_bytes(&self) -> Option<u64> {
        Some(self.len as u64 * (1 << 10))
    }
}

fn fake() -> Fake {
    Fake {
        probes: std::sync::atomic::AtomicUsize::new(0),
        info: ModelInfo {
            label: "fake".to_string(),
            backend: "test",
            vocab_size: 5,
            context_limit: 128,
            chat_template: None,
        },
    }
}

fn model() -> Box<dyn Model> {
    Box::new(fake())
}

fn defaults() -> Defaults {
    Defaults {
        sample: SampleConfig::greedy(),
        seed: 0,
        max_tokens: 4,
        prefix_cache: true,
    }
}

/// What a session holds has to be exactly what the outcome reports, or a
/// kept session is rewound against the wrong tokens and the next request
/// silently attends to somebody else's prompt.
///
/// The two differ whenever a generation stops without feeding its last token
/// back, which is every stop reason except the token limit.
#[test]
fn what_a_generation_reports_holding_is_what_the_session_holds() {
    use crate::generate::{self, Config, Flow};
    use crate::sampling::Rng;

    let model = model();
    for (limit, label) in [(4, "the token limit"), (64, "a cancelled sink")] {
        let mut session = model.session().unwrap();
        let prompt = vec![1i64, 2, 3];
        let config = Config::new(SampleConfig::greedy(), limit);
        // The second run stops from the sink rather than the limit, which is
        // the case that emits a token it never feeds back.
        let mut emitted = 0usize;
        let mut sink = |_: &str| {
            emitted += 1;
            if emitted >= 2 && limit > 4 {
                Flow::Stop
            } else {
                Flow::Continue
            }
        };
        let outcome = generate::generate(
            model.as_ref(),
            session.as_mut(),
            &prompt,
            &config,
            &mut Rng::new(0),
            &mut sink,
        )
        .unwrap();

        assert_eq!(
            outcome.held.len(),
            session.len(),
            "{label}: reported {} held, session has {}",
            outcome.held.len(),
            session.len()
        );
        assert!(
            outcome.held.starts_with(&prompt),
            "{label}: the prompt is not at the front of what is held"
        );
    }
}

/// Rewinding is what makes a kept session safe to reuse, and a backend that
/// cannot rewind has to say so rather than quietly keeping stale positions.
#[test]
fn a_session_rewinds_only_as_far_as_its_backend_allows() {
    let model = model();
    let mut session = model.session().unwrap();
    session.extend(&[1, 2, 3, 4]).unwrap();
    assert_eq!(session.len(), 4);

    // The default accepts only the request that drops nothing, which is how a
    // session is extended rather than rewound.
    assert!(session.truncate(4));
    assert_eq!(session.len(), 4);
    assert!(!session.truncate(2));
    assert_eq!(session.len(), 4, "a refused rewind changes nothing");
}

/// Only the thread running the pass can ask the driver, and it is inside the
/// generation for the whole of a request. Without asking from in there, a
/// dashboard's memory figures sit at whatever they were before the request
/// and catch up only once it is over, which is when they stop mattering.
#[test]
fn a_generation_reads_the_card_while_it_runs() {
    use crate::generate::{self, Config, Flow};
    use crate::sampling::Rng;
    use std::sync::atomic::Ordering;

    let model = fake();
    let mut session = model.session().unwrap();
    let meter = Arc::new(Meter::new());
    let config = Config {
        sample: SampleConfig::greedy(),
        max_tokens: 4,
        meter: Some(meter.clone()),
    };

    assert_eq!(model.probes.load(Ordering::Relaxed), 0);
    let mut sink = |_: &str| Flow::Continue;
    generate::generate(
        &model,
        session.as_mut(),
        &[1, 2, 3],
        &config,
        &mut Rng::new(0),
        &mut sink,
    )
    .unwrap();

    // At least either side of the prompt pass, which is the one that puts the
    // weights on the card. A long decode adds more, on a timer.
    assert!(
        model.probes.load(Ordering::Relaxed) >= 2,
        "the card was read {} times during a generation",
        model.probes.load(Ordering::Relaxed)
    );
    assert!(meter.snapshot().device.is_some(), "and the reading landed");
}

#[test]
fn the_worker_stops_when_the_meter_says_to() {
    let meter = Arc::new(Meter::new());
    let worker = meter.clone();
    // Port zero, so the test never collides with a port something else holds.
    let serving =
        std::thread::spawn(move || serve("127.0.0.1:0".to_string(), model(), defaults(), worker));

    // Long enough for the worker to have reached its wait.
    std::thread::sleep(Duration::from_millis(300));
    let snapshot = meter.snapshot();
    assert_eq!(snapshot.fixed.label, "fake");
    assert_eq!(snapshot.fixed.context_limit, 128);
    assert_eq!(
        snapshot.fixed.footprint.map(|f| f.weight_bytes),
        Some(1 << 30)
    );
    // Read on the way in, and again at every idle tick.
    assert_eq!(snapshot.device.map(|d| d.total_bytes), Some(8 << 30));

    let asked = Instant::now();
    meter.stop();
    let result = serving.join().expect("the worker panicked");
    assert!(result.is_ok(), "{result:?}");
    // It waits for a request at most one tick at a time, so it must notice
    // well inside a second.
    assert!(
        asked.elapsed() < Duration::from_secs(2),
        "took {:?} to stop",
        asked.elapsed()
    );
}

/// Token ids that are their own bytes, so a test reads the text a log line
/// quotes.
struct Bytes;

impl Tokenizer for Bytes {
    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        Ok(text.bytes().map(i64::from).collect())
    }

    fn decode_bytes(&self, ids: &[i64]) -> Vec<u8> {
        ids.iter().map(|&id| id as u8).collect()
    }

    fn is_eog(&self, _id: i64) -> bool {
        false
    }
}

#[test]
fn a_prompt_that_extends_the_session_is_not_a_divergence() {
    let held = Bytes.encode("<user>hi<assistant>hello").unwrap();
    let ids = Bytes.encode("<user>hi<assistant>hello<user>more").unwrap();
    assert_eq!(super::divergence(&Bytes, &held, &ids), None);

    let edited = Bytes.encode("<user>hi<assistant>HELLO<user>more").unwrap();
    let line = super::divergence(&Bytes, &held, &edited).unwrap();
    assert!(line.contains("token 19 of 24"), "{line}");
    assert!(line.contains("hello") && line.contains("HELLO"), "{line}");
}

#[test]
fn a_request_reports_decode_and_prompt_lookups_apart() {
    use crate::model::CacheStats;
    let before = CacheStats {
        expert_hits: 100,
        expert_misses: 100,
        ..Default::default()
    };
    assert_eq!(super::describe_experts(&before, &before), None);
    let after = CacheStats {
        expert_hits: 175,
        expert_misses: 125,
        expert_prompt_hits: 10,
        expert_prompt_misses: 30,
        expert_bytes: 2_000_000_000,
        ..before
    };
    assert_eq!(
        super::describe_experts(&before, &after).unwrap(),
        "experts resident: decode 75.0% of 100, prompt 25.0% of 40; 2.00 GB copied"
    );
}
