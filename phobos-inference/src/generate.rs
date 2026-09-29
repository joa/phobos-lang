use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use crate::model::{Model, Session};
use crate::sampling::{Rng, SampleConfig, Sequence, choose};
use crate::telemetry::Meter;

/// How often a generation stops to re-read the card's memory.
///
/// Only the thread running the pass can query the driver, and it stays in
/// this loop for the whole request. Without a periodic refresh a dashboard
/// would show stale memory until the request ends. The query is cheap.
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
    /// Where to report progress. The generation never branches on it.
    pub meter: Option<Arc<Meter>>,
    /// The prompt position to leave a [`Session::checkpoint`] at, typically
    /// where the next prompt is likely to diverge from this one. `None`, or a
    /// position the pass does not reach, means the end of the prompt.
    pub checkpoint_at: Option<usize>,
}

impl Config {
    /// The two fields a generation actually reads, with no meter attached.
    pub fn new(sample: SampleConfig, max_tokens: usize) -> Config {
        Config {
            sample,
            max_tokens,
            meter: None,
            checkpoint_at: None,
        }
    }
}

pub struct Outcome {
    pub stop: Stop,
    pub tokens: usize,
    /// Every token the session now holds, in order.
    ///
    /// This is not always the prompt plus what was emitted. A token that
    /// stopped the sink, or an end-of-turn token, is never fed back. A caller
    /// keeping the session needs the exact list.
    pub held: Vec<i64>,
}

pub fn prefill(session: &mut dyn Session, prompt: &[i64]) -> Result<Vec<f32>> {
    session.extend(prompt)
}

/// [`prefill`] then [`continue_from`].
///
/// Whatever `session` already holds must be a prefix of `prompt`, and only
/// the remainder is run. Sampling still sees the whole sequence, so a penalty
/// does not depend on how much of the prompt was cached.
pub fn generate(
    model: &dyn Model,
    session: &mut dyn Session,
    prompt: &[i64],
    config: &Config,
    rng: &mut Rng,
    sink: &mut dyn FnMut(&str) -> Flow,
) -> Result<Outcome> {
    // The first pass puts the weights on the card, so read memory before it.
    probe(model, config);
    let cached = session.len().min(prompt.len());
    let fresh = &prompt[cached..];
    if fresh.is_empty() {
        // The first token needs fresh logits for the last prompt position.
        // A caller reusing a session must leave it one position short.
        bail!("the session already holds the whole prompt, leaving no position to run");
    }

    let started = Instant::now();
    // The checkpoint goes where the prompt's last turn opens, if that is in
    // the fresh part, and at its end otherwise.
    let split = config
        .checkpoint_at
        .filter(|&at| at > cached && at < prompt.len())
        .map_or(fresh.len(), |at| at - cached);
    let logits = match session.prompt_batch() {
        Some(batch) => feed_in_batches(session, fresh, split, batch, config.meter.as_deref(), started)?,
        None => {
            let logits = prefill(session, fresh)?;
            session.checkpoint()?;
            logits
        }
    };
    if let Some(meter) = config.meter.as_ref() {
        meter.prefilled(fresh.len(), started.elapsed());
    }
    probe(model, config);
    let mut sequence = Sequence::new(prompt.to_vec());
    continue_from(model, session, &mut sequence, &logits, config, rng, sink)
}

/// Feeds `fresh` a batch at a time, reporting progress between batches, and
/// checkpoints after the first `split` positions. Returns the last position's
/// logits.
fn feed_in_batches(session: &mut dyn Session, fresh: &[i64], split: usize, batch: usize, meter: Option<&Meter>, started: Instant) -> Result<Vec<f32>> {
    let mut logits = Vec::new();
    let mut done = 0;
    for (i, part) in [&fresh[..split], &fresh[split..]].into_iter().enumerate() {
        for chunk in part.chunks(batch.max(1)) {
            logits = session.extend(chunk)?;
            done += chunk.len();
            if let Some(meter) = meter {
                meter.prefilling(done, started.elapsed());
            }
        }
        if i == 0 {
            session.checkpoint()?;
        }
    }
    Ok(logits)
}

/// Continue a generation, `logits` being those for the position after
/// `sequence`.
///
/// Separate from [`generate`] so a caller can inspect the prompt pass's
/// logits first, as the CLI's top-candidates listing does.
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

    // Unpenalized greedy only needs the winning token id, which a backend
    // may produce without copying out the logits. See `Session::extend_greedy`.
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
            // Counted here, not at the sink, because a token that ends
            // mid-character never reaches the sink.
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

    // Nothing will complete the trailing bytes, so render them lossy.
    if !pending.is_empty() {
        sink(&String::from_utf8_lossy(&pending));
    }
    Ok(Outcome {
        stop,
        tokens: emitted,
        held: sequence.tokens().to_vec(),
    })
}

/// Copies the model's current device memory and cache stats into the meter.
/// Does nothing without a meter.
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
