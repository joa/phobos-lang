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

    /// Whether the penalties would move any logit.
    ///
    /// Checked per half so the first generated token does not pay for a vocab-sized copy just because
    /// a presence penalty is set.
    fn penalizes(&self, history: History) -> bool {
        (self.repetition_penalty != 1.0 && !history.is_empty())
            || (self.presence_penalty != 0.0 && !history.generated.is_empty())
    }
}

/// SplitMix64, which keeps runs reproducible without an RNG dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // An all-zero state degenerates.
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        // The top 24 bits give a uniform float without bias.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

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
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let logits = [0.1, 3.0, 0.2, 2.9];
        let mut rng = Rng::new(0);
        assert_eq!(
            choose(
                &logits,
                &SampleConfig::greedy(),
                History::default(),
                &mut rng
            ),
            1
        );
    }

    #[test]
    fn top_k_one_is_deterministic() {
        // With only the best token surviving, sampling has to return it.
        let logits = [0.1, 3.0, 0.2, 2.9];
        let cfg = SampleConfig {
            temperature: 1.0,
            top_k: 1,
            ..SampleConfig::greedy()
        };
        let mut rng = Rng::new(42);
        for _ in 0..16 {
            assert_eq!(choose(&logits, &cfg, History::default(), &mut rng), 1);
        }
    }

    #[test]
    fn sampling_is_seed_reproducible() {
        let logits = [1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = SampleConfig {
            temperature: 1.0,
            ..SampleConfig::greedy()
        };
        let draw = |seed| {
            let mut rng = Rng::new(seed);
            (0..8)
                .map(|_| choose(&logits, &cfg, History::default(), &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
    }

    #[test]
    fn min_p_keeps_only_the_contenders() {
        // At temperature 1 the top two are within a factor of 1.1 of each
        // other and the rest are orders of magnitude behind.
        let logits = [5.0, 4.9, 0.0, -5.0];
        let cfg = SampleConfig {
            temperature: 1.0,
            min_p: 0.5,
            ..SampleConfig::greedy()
        };
        let mut rng = Rng::new(3);
        for _ in 0..64 {
            assert!(choose(&logits, &cfg, History::default(), &mut rng) < 2);
        }
    }

    #[test]
    fn min_p_of_one_leaves_only_the_best() {
        let logits = [0.1, 3.0, 0.2, 2.9];
        let cfg = SampleConfig {
            temperature: 1.0,
            min_p: 1.0,
            ..SampleConfig::greedy()
        };
        let mut rng = Rng::new(11);
        for _ in 0..16 {
            assert_eq!(choose(&logits, &cfg, History::default(), &mut rng), 1);
        }
    }

    #[test]
    fn presence_penalty_demotes_a_generated_token() {
        let logits = [0.1, 3.0, 0.2, 2.9];
        let cfg = SampleConfig {
            presence_penalty: 0.5,
            ..SampleConfig::greedy()
        };
        let mut rng = Rng::new(0);
        assert_eq!(choose(&logits, &cfg, History::default(), &mut rng), 1);
        assert_eq!(choose(&logits, &cfg, History::new(&[], &[1]), &mut rng), 3);
    }

    #[test]
    fn presence_penalty_leaves_the_prompt_alone() {
        // vLLM and the OpenAI API both count only what was sampled, so a
        // token the prompt happens to contain keeps its full logit.
        let logits = [0.1, 3.0, 0.2, 2.9];
        let cfg = SampleConfig {
            presence_penalty: 2.0,
            ..SampleConfig::greedy()
        };
        let mut rng = Rng::new(0);
        assert_eq!(choose(&logits, &cfg, History::new(&[1], &[]), &mut rng), 1);
    }

    #[test]
    fn repetition_penalty_counts_the_prompt() {
        // Halving the leader's 3.0 puts it behind the runner-up's 2.9.
        let logits = [0.1, 3.0, 0.2, 2.9];
        let cfg = SampleConfig {
            repetition_penalty: 2.0,
            ..SampleConfig::greedy()
        };
        let mut rng = Rng::new(0);
        assert_eq!(choose(&logits, &cfg, History::new(&[1], &[]), &mut rng), 3);
    }

    #[test]
    fn repetition_penalty_pushes_both_signs_towards_zero() {
        let mut logits = [2.0, -2.0];
        let cfg = SampleConfig {
            repetition_penalty: 2.0,
            ..SampleConfig::greedy()
        };
        penalize(&mut logits, &cfg, History::new(&[0], &[1]));
        assert_eq!(logits, [1.0, -4.0]);
    }

    #[test]
    fn penalties_ignore_repeat_count() {
        let logits = [2.0, 1.0, 1.0];
        let cfg = SampleConfig {
            presence_penalty: 0.25,
            repetition_penalty: 1.5,
            ..SampleConfig::greedy()
        };
        let once = {
            let mut scratch = logits;
            penalize(&mut scratch, &cfg, History::new(&[0], &[0]));
            scratch
        };
        let many = {
            let mut scratch = logits;
            penalize(&mut scratch, &cfg, History::new(&[0, 0], &[0, 0, 0]));
            scratch
        };
        assert_eq!(once, many);
    }

    #[test]
    fn penalties_skip_ids_outside_the_vocab() {
        let logits = [1.0, 2.0];
        let cfg = SampleConfig {
            presence_penalty: 1.0,
            ..SampleConfig::greedy()
        };
        let mut rng = Rng::new(0);
        assert_eq!(
            choose(&logits, &cfg, History::new(&[], &[-1, 7]), &mut rng),
            1
        );
    }
}
