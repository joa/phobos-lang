use super::*;

/// Feeds `prompts` prompts of `len` tokens, each from position 0, timing
/// every pass with `cost`, seconds per token at a size and a middle position.
fn feed(tuner: &mut Tuner, prompts: usize, len: usize, cost: &dyn Fn(usize, f64) -> f64, room: &dyn Fn(usize) -> bool) {
    for _ in 0..prompts {
        let mut at = 0;
        while at < len {
            let size = tuner.size(at, len - at);
            let rows = size.min(len - at);
            if rows == size {
                let secs = cost(size, at as f64 + size as f64 / 2.0) * size as f64;
                tuner.observe(size, at, secs, room);
            }
            at += rows;
        }
    }
}

fn anywhere(_: usize) -> bool {
    true
}

/// Per-token cost with its minimum at `best`, rising 10% per doubling away.
fn bowl(best: usize) -> impl Fn(usize, f64) -> f64 {
    move |size, _| 1e-4 * (1.0 + 0.1 * (size.ilog2() as f64 - best.ilog2() as f64).abs())
}

#[test]
fn buckets_double_from_2048() {
    assert_eq!(bucket(0), 0);
    assert_eq!(bucket(2047), 0);
    assert_eq!(bucket(2048), 1);
    assert_eq!(bucket(4095), 1);
    assert_eq!(bucket(4096), 2);
    assert_eq!(bucket(1 << 20), 10);
}

#[test]
fn climbs_up_to_the_fastest_size() {
    let mut tuner = Tuner::new(MAX_BATCH);
    feed(&mut tuner, 40, 4096, &bowl(2048), &anywhere);
    assert_eq!(tuner.size(0, 4096), 2048);
    assert!(tuner.buckets[0].settled);
    assert!(tuner.take_dirty());
}

#[test]
fn a_trial_wider_than_the_prompt_waits() {
    let mut tuner = Tuner::new(MAX_BATCH);
    feed(&mut tuner, 40, 2048, &bowl(4096), &anywhere);
    // Only 4096 is left, and no 2048-token prompt can measure it, so they
    // run at the pick instead.
    assert_eq!(tuner.buckets[0].queue, Some(vec![4096]));
    assert_eq!(tuner.size(0, 2048), 2048);
    assert_eq!(tuner.size(0, 5000), 4096);
}

#[test]
fn climbs_down_when_narrower_wins() {
    let mut tuner = Tuner::new(MAX_BATCH);
    feed(&mut tuner, 40, 4096, &bowl(128), &anywhere);
    assert_eq!(tuner.size(0, 4096), 128);
    assert!(tuner.buckets[0].settled);
}

#[test]
fn stays_put_on_a_tie() {
    let mut tuner = Tuner::new(MAX_BATCH);
    feed(&mut tuner, 40, 4096, &|_, _| 1e-4, &anywhere);
    assert_eq!(tuner.size(0, 4096), DEFAULT_BATCH);
    assert!(tuner.buckets[0].settled);
}

#[test]
fn finds_a_size_past_a_losing_neighbour() {
    // A kernel cliff: 256 loses to 512, and 128 beats both.
    let cost = |size: usize, _: f64| match size {
        128 => 0.5e-4,
        512 => 1e-4,
        _ => 1.2e-4,
    };
    let mut tuner = Tuner::new(MAX_BATCH);
    feed(&mut tuner, 40, 4096, &cost, &anywhere);
    assert_eq!(tuner.size(0, 4096), 128);
    assert!(tuner.buckets[0].settled);
}

#[test]
fn a_size_the_prompt_cannot_fill_does_not_hold_up_the_rest() {
    // 1024 is nearest the pick, but these prompts fill no pass of it.
    let mut tuner = Tuner::new(MAX_BATCH);
    feed(&mut tuner, 40, 1000, &bowl(128), &anywhere);
    assert_eq!(tuner.buckets[0].pick, 128);
    assert_eq!(tuner.buckets[0].queue, Some(vec![1024, 2048, 4096]));
}

#[test]
fn never_passes_the_limit() {
    // A mixture-of-experts pass takes at most 1024 rows.
    let mut tuner = Tuner::new(1024);
    feed(&mut tuner, 40, 4096, &bowl(4096), &anywhere);
    assert_eq!(tuner.buckets[0].pick, 1024);
    assert!(tuner.buckets[0].settled);
}

#[test]
fn steps_up_only_with_room() {
    let mut tuner = Tuner::new(MAX_BATCH);
    feed(&mut tuner, 40, 2048, &bowl(2048), &|size| size <= 512);
    assert_eq!(tuner.size(0, 4096), 512);
}

#[test]
fn drops_a_size_off_a_cliff_at_once() {
    let mut tuner = Tuner::new(MAX_BATCH);
    let paging = |size: usize, _: f64| if size == 1024 { 1e-3 } else { 1e-4 };
    feed(&mut tuner, 1, 2048, &paging, &anywhere);
    // Three passes at 512 bring the trial to 1024, whose first pass pages.
    let seen = std::cell::RefCell::new(Vec::new());
    let watched = |size, mid| {
        seen.borrow_mut().push(size);
        paging(size, mid)
    };
    feed(&mut tuner, 3, 2048, &watched, &anywhere);
    assert_eq!(seen.borrow().iter().filter(|&&s| s == 1024).count(), 1);
    assert!(tuner.buckets[0].lost.contains(&1024));
}

#[test]
fn a_deeper_trial_is_not_marked_down_for_its_depth() {
    // 1024 is 5% faster per token at equal depth, but the cost per token
    // doubles across the bucket, and the trial is measured deeper than the
    // pick: the raw medians would say the opposite.
    let cost = |size: usize, mid: f64| {
        let base = if size == 1024 { 0.95e-4 } else { 1e-4 };
        base + 1e-4 * (mid - 2048.0) / 2048.0
    };
    let mut tuner = Tuner::new(MAX_BATCH);
    // One long prompt from 2048 to 4096, bucket 1, then more of the same.
    for _ in 0..6 {
        let mut at = 2048;
        while at < 4096 {
            let size = tuner.size(at, 4096 - at);
            let rows = size.min(4096 - at);
            if rows == size {
                tuner.observe(size, at, cost(size, at as f64 + size as f64 / 2.0) * size as f64, &anywhere);
            }
            at += rows;
        }
    }
    assert_eq!(tuner.buckets[1].pick, 1024);
}

#[test]
fn a_new_bucket_starts_from_the_shallower_pick() {
    let mut tuner = Tuner::new(MAX_BATCH);
    feed(&mut tuner, 40, 2048, &bowl(1024), &anywhere);
    assert_eq!(tuner.buckets[0].pick, 1024);
    assert_eq!(tuner.size(8192, 4096), 1024);
}

#[test]
fn a_record_round_trips() {
    let mut tuner = Tuner::new(MAX_BATCH);
    feed(&mut tuner, 40, 4096, &bowl(1024), &anywhere);
    let text = tuner.to_text();
    assert!(text.starts_with("bucket=0 pick=1024 settled=1 lost=128,256,512,2048,4096 us="), "{text}");

    let mut loaded = Tuner::new(MAX_BATCH);
    loaded.load_text(&text);
    assert_eq!(loaded.size(0, 4096), 1024);
    assert!(loaded.buckets[0].settled);
    // A settled bucket takes no more samples and writes its medians back.
    assert!(!loaded.observe(1024, 0, 1.0, &anywhere));
    assert_eq!(loaded.to_text(), text);
}

#[test]
fn a_record_past_the_limit_is_dropped() {
    let mut tuner = Tuner::new(1024);
    tuner.load_text("bucket=0 pick=4096 settled=1 lost= us=\nbucket=1 pick=256 settled=0 lost=8192,512 us=\ngarbage\n");
    assert_eq!(tuner.size(0, 4096), DEFAULT_BATCH);
    assert_eq!(tuner.size(2048, 4096), 256);
    assert!(!tuner.buckets[1].settled);
    assert_eq!(tuner.buckets[1].lost, BTreeSet::from([512]));
}
