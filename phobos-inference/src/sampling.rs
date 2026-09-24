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

    /// Every token in order: the prompt followed by what has been generated
    /// and fed back. This is exactly what the session that produced it holds,
    /// which is what lets a caller keep the session for the next request.
    pub fn tokens(&self) -> &[i64] {
        &self.tokens
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

    /// Whether [`choose`]'s result depends on nothing but which logit is
    /// largest: greedy, with both penalties off so [`penalize`] is a no-op
    /// whatever the history. Exactly when a caller may substitute a
    /// device-side argmax for [`choose`]; penalized greedy still needs the
    /// full rewritten vector and stays on the ordinary path.
    pub fn is_greedy_unpenalized(&self) -> bool {
        self.is_greedy() && self.repetition_penalty == 1.0 && self.presence_penalty == 0.0
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
    // A top-k cut applies the penalties as it passes over the vocab; every
    // other path rewrites a copy of it. Both see the same values.
    if cfg.top_k > 0 && cfg.top_k < logits.len() && !cfg.is_greedy() {
        return sample(top_k(logits, cfg, history), cfg, rng);
    }
    let penalized = cfg.penalizes(history).then(|| {
        let mut scratch = logits.to_vec();
        penalize(&mut scratch, cfg, history);
        scratch
    });

    let logits = penalized.as_deref().unwrap_or(logits);

    if cfg.is_greedy() {
        return argmax(logits);
    }

    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    if cfg.top_k > 0 {
        ranked.truncate(cfg.top_k);
    }
    sample(ranked, cfg, rng)
}

/// One draw from `ranked`, the candidates highest logit first, after the
/// temperature, the nucleus and the min-p cut.
fn sample(mut ranked: Vec<(usize, f32)>, cfg: &SampleConfig, rng: &mut Rng) -> i64 {

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

/// The `cfg.top_k` highest logits once penalized, highest first, in one pass
/// over the vocab: a quarter million of them, where copying the vocab to
/// penalize it or ranking all of it costs a millisecond a token. A logit
/// only enters the buffer by beating its lowest, which almost none do.
fn top_k(logits: &[f32], cfg: &SampleConfig, history: History) -> Vec<(usize, f32)> {
    let k = cfg.top_k;
    // Which penalties touch which ids: bit 0 repetition, bit 1 presence.
    let mut marks = Vec::new();
    if cfg.penalizes(history) {
        marks = vec![0u8; logits.len()];
        if cfg.repetition_penalty != 1.0 {
            for id in history.prompt.iter().chain(history.generated) {
                if let Some(m) = usize::try_from(*id).ok().and_then(|i| marks.get_mut(i)) {
                    *m |= 1;
                }
            }
        }
        if cfg.presence_penalty != 0.0 {
            for id in history.generated {
                if let Some(m) = usize::try_from(*id).ok().and_then(|i| marks.get_mut(i)) {
                    *m |= 2;
                }
            }
        }
    }
    let mut best: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
    for (i, &raw) in logits.iter().enumerate() {
        let mut logit = raw;
        if let Some(&m) = marks.get(i) {
            if m & 1 != 0 {
                logit = if logit > 0.0 { logit / cfg.repetition_penalty } else { logit * cfg.repetition_penalty };
            }
            if m & 2 != 0 {
                logit -= cfg.presence_penalty;
            }
        }
        if best.len() == k && best[k - 1].1.total_cmp(&logit).is_ge() {
            continue;
        }
        let at = best.partition_point(|&(_, l)| l.total_cmp(&logit).is_ge());
        best.insert(at, (i, logit));
        best.truncate(k);
    }
    best
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
