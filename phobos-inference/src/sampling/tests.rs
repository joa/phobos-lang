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
