use std::collections::HashSet;

pub struct Sequence {
    tokens: Vec<i64>,
    prompt_len: usize,
}

impl Sequence {
    pub fn new(prompt: Vec<i64>) -> Sequence {
        Sequence {
            prompt_len: prompt.len(),
            tokens: prompt,
        }
    }

    pub fn push(&mut self, id: i64) {
        self.tokens.push(id);
    }

    pub fn history(&self) -> History<'_> {
        let (prompt, generated) = self.tokens.split_at(self.prompt_len);
        History::new(prompt, generated)
    }
}

#[derive(Clone, Copy, Default)]
pub struct History<'a> {
    prompt: &'a [i64],
    generated: &'a [i64],
}

impl<'a> History<'a> {
    pub fn new(prompt: &'a [i64], generated: &'a [i64]) -> History<'a> {
        History { prompt, generated }
    }

    fn is_empty(&self) -> bool {
        self.prompt.is_empty() && self.generated.is_empty()
    }
}

#[derive(Clone, Copy)]
pub struct SampleConfig {
    /// Softmax temperature.
    ///
    /// At or below 0 this is greedy argmax.
    pub temperature: f32,

    /// Keep only the `k` highest-logit tokens; 0 disables.
    pub top_k: usize,

    /// Keep the smallest set of tokens whose probability sums to `top_p`; 1.0
    /// and above disables.
    pub top_p: f32,

    /// Keep the tokens at least `min_p` as likely as the best candidate; 0.0
    /// and below disables. Unlike top-p the cut adapts to how peaked the
    /// distribution is: a confident step keeps few tokens, a flat one many.
    pub min_p: f32,

    /// Subtracted from the logit of every token the model has generated, the
    /// prompt excluded; 0.0 disables. Flat, so ten occurrences cost what one
    /// does.
    pub presence_penalty: f32,

    /// Scales the logit of every token already in the sequence, the prompt
    /// included, towards zero: positive logits divide by it and negative ones
    /// multiply, so it pushes the same direction either way. 1.0 disables.
    pub repetition_penalty: f32,
}

impl SampleConfig {
    pub fn greedy() -> SampleConfig {
        SampleConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Whether the penalties would move any logit. Checked per half so the first
    /// generated token does not pay for a vocab-sized copy just because a
    /// presence penalty is set.
    fn penalizes(&self, history: History) -> bool {
        (self.repetition_penalty != 1.0 && !history.is_empty())
            || (self.presence_penalty != 0.0 && !history.generated.is_empty())
    }
}

/// The generator a sampled run draws from.
pub use phobos_base::rng::SplitMix64 as Rng;

pub fn choose(logits: &[f32], cfg: &SampleConfig, history: History, rng: &mut Rng) -> i64 {
    // The penalties rewrite the logits, so they need a copy of the vocab.
    // Both the greedy and the sampled path see the rewritten values.
    let penalized = cfg.penalizes(history).then(|| {
        let mut scratch = logits.to_vec();
        penalize(&mut scratch, cfg, history);
        scratch
    });

    let logits = penalized.as_deref().unwrap_or(logits);

    if cfg.is_greedy() {
        return argmax(logits);
    }

    // Ranked by logit, then cut to the top-k.
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    if cfg.top_k > 0 {
        ranked.truncate(cfg.top_k);
    }

    // Temperature-scaled softmax over the survivors.
    let max = ranked[0].1;
    let mut probs: Vec<f32> = ranked
        .iter()
        .map(|&(_, l)| ((l - max) / cfg.temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Nucleus: the shortest prefix reaching cumulative probability top_p.
    let mut keep = probs.len();
    if cfg.top_p < 1.0 {
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= cfg.top_p {
                keep = i + 1;
                break;
            }
        }
    }

    // min-p: drop whatever the leader outclasses by more than min_p.
    if cfg.min_p > 0.0 {
        let cutoff = cfg.min_p * probs[0];
        let bound = probs
            .iter()
            .position(|&p| p < cutoff)
            .unwrap_or(probs.len());
        keep = keep.min(bound.max(1));
    }

    if keep < probs.len() {
        ranked.truncate(keep);
        probs.truncate(keep);
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }
    }

    // Inverse-CDF sample.
    let r = rng.next_f32();
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return ranked[i].0 as i64;
        }
    }

    ranked.last().map(|&(i, _)| i as i64).unwrap_or(0)
}

/// Demote the tokens already in the sequence. Both penalties are one-shot: a
/// token occurring ten times is hit as hard as one occurring once.
fn penalize(logits: &mut [f32], cfg: &SampleConfig, history: History) {
    let vocab = logits.len();
    if cfg.repetition_penalty != 1.0 {
        let seen = history.prompt.iter().chain(history.generated).copied();
        for_each_unique(seen, vocab, |i| {
            let logit = &mut logits[i];
            *logit = if *logit > 0.0 {
                *logit / cfg.repetition_penalty
            } else {
                *logit * cfg.repetition_penalty
            };
        });
    }
    if cfg.presence_penalty != 0.0 {
        for_each_unique(history.generated.iter().copied(), vocab, |i| {
            logits[i] -= cfg.presence_penalty;
        });
    }
}

/// Call `f` once per distinct in-range token id.
fn for_each_unique(ids: impl Iterator<Item = i64>, vocab: usize, mut f: impl FnMut(usize)) {
    let mut seen = HashSet::new();
    for id in ids {
        let Ok(i) = usize::try_from(id) else { continue };
        if i < vocab && seen.insert(i) {
            f(i);
        }
    }
}

pub fn argmax(logits: &[f32]) -> i64 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests;
