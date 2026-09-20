use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use crate::model::{Model, Session};
use crate::sampling::{Rng, SampleConfig, Sequence, choose};
use crate::telemetry::Meter;

/// How often a generation stops to re-read the card.
///
/// Only the thread running the pass can ask the driver, and it is inside this
/// loop for the whole of a request. Without this the memory a dashboard shows
/// would sit at whatever it was before the request started and only catch up
/// once the request was over, which is exactly when it stops being
/// interesting. The call is a driver query costing microseconds, and once a
/// second against a forward pass is nothing.
const DEVICE_REFRESH: Duration = Duration::from_secs(1);

pub enum Stop {
    Eos,
    Context(usize),
    Limit,
    Cancelled,
}

impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stop::Eos => write!(f, "end-of-sequence token"),
            Stop::Context(limit) => write!(f, "{limit}-token context limit"),
            Stop::Limit => write!(f, "token limit"),
            Stop::Cancelled => write!(f, "client disconnected"),
        }
    }
}

impl Stop {
    pub fn finish_reason(&self) -> &'static str {
        match self {
            Stop::Eos => "stop",
            Stop::Context(_) | Stop::Limit => "length",
            Stop::Cancelled => "client_disconnect",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

pub struct Config {
    pub sample: SampleConfig,
    pub max_tokens: usize,
    /// Where to report progress, for a caller displaying it. Not a setting
    /// the generation reads: nothing below branches on it.
    pub meter: Option<Arc<Meter>>,
}

impl Config {
    /// The two fields a generation actually reads, with no meter attached.
    pub fn new(sample: SampleConfig, max_tokens: usize) -> Config {
        Config {
            sample,
            max_tokens,
            meter: None,
        }
    }
}

pub struct Outcome {
    pub stop: Stop,
    pub tokens: usize,
    /// Every token the session now holds, in order.
    ///
    /// Not the same as the prompt plus what was emitted: a generation stopped
    /// by its sink emits a token it never feeds back, and one stopped by an
    /// end-of-turn token never feeds that either. A caller keeping the session
    /// has to know which, so it is reported rather than reconstructed.
    pub held: Vec<i64>,
}

pub fn prefill(session: &mut dyn Session, prompt: &[i64]) -> Result<Vec<f32>> {
    session.extend(prompt)
}

/// [`prefill`] then [`continue_from`].
///
/// Whatever `session` already holds must be a prefix of `prompt`; only the
/// remainder is run. A fresh session holds nothing and so runs all of it,
/// which is the ordinary case. Sampling still sees the whole sequence, so a
/// penalty does not depend on how much of the prompt was already cached.
pub fn generate(
    model: &dyn Model,
    session: &mut dyn Session,
    prompt: &[i64],
    config: &Config,
    rng: &mut Rng,
    sink: &mut dyn FnMut(&str) -> Flow,
) -> Result<Outcome> {
    // Before anything: the first pass of a run is what puts the weights on
    // the card, so the reading either side of it is the interesting pair.
    probe(model, config);
    let cached = session.len().min(prompt.len());
    let fresh = &prompt[cached..];
    if fresh.is_empty() {
        // The decode's first token is chosen from the logits of the last
        // prompt position, and those have to come from a pass run now: what
        // the session last returned belongs to whatever it was doing before.
        // A caller reusing a session leaves it one position short for this.
        bail!("the session already holds the whole prompt, leaving no position to run");
    }

    let started = Instant::now();
    let logits = prefill(session, fresh)?;
    if let Some(meter) = config.meter.as_ref() {
        meter.prefilled(fresh.len(), started.elapsed());
    }
    probe(model, config);
    let mut sequence = Sequence::new(prompt.to_vec());
    continue_from(model, session, &mut sequence, &logits, config, rng, sink)
}

/// Continue a generation, `logits` being those for the position after
/// `sequence`.
///
/// Split from [`generate`] so a caller that wants to show the prompt pass's own
/// logits, as the CLI's top-candidates listing does, can look at them first.
pub fn continue_from(
    model: &dyn Model,
    session: &mut dyn Session,
    sequence: &mut Sequence,
    logits: &[f32],
    config: &Config,
    rng: &mut Rng,
    sink: &mut dyn FnMut(&str) -> Flow,
) -> Result<Outcome> {
    let tokenizer = model.tokenizer();
    let context_limit = model.info().context_limit;

    // Greedy with no penalty active needs nothing from a step but the
    // winning token id, which a backend may be able to produce without
    // moving the whole logits vector; see `Session::extend_greedy`.
    let fast_greedy = config.sample.is_greedy_unpenalized();

    // Bytes short of a complete UTF-8 character; a token can split one.
    let mut pending: Vec<u8> = Vec::new();
    let mut probed = Instant::now();
    let mut next = choose(logits, &config.sample, sequence.history(), rng);
    let mut produced = 0usize;
    let mut emitted = 0usize;

    let stop = loop {
        if tokenizer.is_eog(next) {
            break Stop::Eos;
        }
        if session.len() >= context_limit {
            break Stop::Context(context_limit);
        }

        pending.extend(tokenizer.decode_bytes(&[next]));
        emitted += 1;
        if let Some(meter) = config.meter.as_ref() {
            // Counted here rather than at the sink: a token whose bytes only
            // half-finish a character does not reach the sink, and it is
            // still a token the card produced.
            meter.token(session.len(), session.cache_bytes());
            if probed.elapsed() >= DEVICE_REFRESH {
                probe(model, config);
                probed = Instant::now();
            }
        }
        if let Some(text) = take_complete(&mut pending)
            && sink(&text) == Flow::Stop
        {
            break Stop::Cancelled;
        }

        sequence.push(next);
        next = if fast_greedy {
            session.extend_greedy(&[next])?
        } else {
            let logits = session.extend(&[next])?;
            choose(&logits, &config.sample, sequence.history(), rng)
        };
        produced += 1;
        if produced >= config.max_tokens {
            break Stop::Limit;
        }
    };

    // Trailing incomplete bytes, rendered lossy: there is nothing left to
    // complete them with.
    if !pending.is_empty() {
        sink(&String::from_utf8_lossy(&pending));
    }
    Ok(Outcome {
        stop,
        tokens: emitted,
        held: sequence.tokens().to_vec(),
    })
}

/// Ask the model what the card looks like now, for whoever is watching.
///
/// Costs nothing when nobody is: without a meter there is no one to tell, and
/// a front end with no device to read reports nothing either way.
fn probe(model: &dyn Model, config: &Config) {
    let Some(meter) = config.meter.as_ref() else {
        return;
    };
    meter.set_device_memory(model.device_memory());
    meter.set_cache_stats(model.cache_stats());
}

/// Take the longest valid UTF-8 prefix out of `pending`, leaving whatever is
/// still mid-character.
fn take_complete(pending: &mut Vec<u8>) -> Option<String> {
    let valid = match std::str::from_utf8(pending) {
        Ok(s) => s.len(),
        Err(e) => e.valid_up_to(),
    };
    if valid == 0 {
        return None;
    }
    // Valid UTF-8 by construction.
    let text = std::str::from_utf8(&pending[..valid]).unwrap().to_string();
    pending.drain(..valid);
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_back_a_split_character() {
        // The two halves of a two-byte character arrive separately.
        let mut pending = vec![0xC3];
        assert!(take_complete(&mut pending).is_none());
        pending.push(0xA9);
        assert_eq!(take_complete(&mut pending).as_deref(), Some("é"));
        assert!(pending.is_empty());
    }

    #[test]
    fn emits_the_complete_prefix_only() {
        let mut pending = b"ok".to_vec();
        pending.push(0xE2);
        assert_eq!(take_complete(&mut pending).as_deref(), Some("ok"));
        assert_eq!(pending, vec![0xE2]);
    }
}
