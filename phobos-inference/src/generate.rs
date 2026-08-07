use anyhow::Result;

use crate::model::{Model, Session};
use crate::sampling::{Rng, SampleConfig, Sequence, choose};

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
}

pub struct Outcome {
    pub stop: Stop,
    pub tokens: usize,
}

pub fn prefill(session: &mut dyn Session, prompt: &[i64]) -> Result<Vec<f32>> {
    session.extend(prompt)
}

/// [`prefill`] then [`continue_from`].
pub fn generate(
    model: &dyn Model,
    session: &mut dyn Session,
    prompt: &[i64],
    config: &Config,
    rng: &mut Rng,
    sink: &mut dyn FnMut(&str) -> Flow,
) -> Result<Outcome> {
    let logits = prefill(session, prompt)?;
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

    // Bytes short of a complete UTF-8 character; a token can split one.
    let mut pending: Vec<u8> = Vec::new();
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
        if let Some(text) = take_complete(&mut pending)
            && sink(&text) == Flow::Stop
        {
            break Stop::Cancelled;
        }

        sequence.push(next);
        let logits = session.extend(&[next])?;
        next = choose(&logits, &config.sample, sequence.history(), rng);
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
    })
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
