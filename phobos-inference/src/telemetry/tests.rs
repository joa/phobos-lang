// Tests for the meter, beside the module they cover: the inline block
// outgrew the cap `phobos-base/tests/source_size.rs` keeps on them.

use super::*;

#[test]
fn a_request_reports_its_two_rates_separately() {
    let meter = Meter::new();
    meter.request_started(128, 0);
    meter.prefilled(128, Duration::from_millis(100));
    for i in 0..4 {
        meter.token(129 + i, Some(4096));
    }
    meter.request_finished("stop");

    let snap = meter.snapshot();
    // 128 tokens in 100ms.
    assert!(
        (snap.prefill_rate - 1280.0).abs() < 1.0,
        "{}",
        snap.prefill_rate
    );
    assert_eq!(snap.requests, 1);
    assert_eq!(snap.prompt_tokens, 128);
    assert_eq!(snap.completion_tokens, 4);
    assert_eq!(snap.phase, Phase::Idle);
    assert!(snap.active.is_none());

    let recent = &snap.recent[0];
    assert_eq!(recent.completion_tokens, 4);
    assert_eq!(recent.reason, "stop");
    assert!(recent.decode_rate > 0.0);
    // A second close has no live request to report.
    assert!(meter.request_finished("stop").is_none());
}

#[test]
fn a_decode_is_rated_over_every_step_it_took() {
    // Two tokens over a span of two sleeps. Counting steps rather than
    // intervals is the whole difference: over ~80ms, two tokens is ~25 a
    // second and one is ~12.5, so the two answers do not overlap.
    let step = Duration::from_millis(40);
    let meter = Meter::new();
    meter.request_started(4, 0);
    meter.prefilled(4, Duration::from_millis(10));
    for i in 0..2 {
        std::thread::sleep(step);
        meter.token(5 + i, None);
    }
    let done = meter.request_finished("stop").expect("a request was live");

    assert_eq!(done.completion_tokens, 2);
    let span = 2.0 / done.decode_rate;
    // Loose, since a sleep only promises a lower bound, but far tighter
    // than the factor of two an off-by-one step would cost.
    assert!(
        (0.06..0.13).contains(&span),
        "two tokens rated at {:.1}/s implies a {span:.3}s span",
        done.decode_rate
    );
}

fn step(item: &'static str, millis: u64, cached: bool) -> Step<'static> {
    Step {
        stage: "kernels",
        item,
        done: 1,
        total: 3,
        cached,
        took: Duration::from_millis(millis),
        source: "kernel x() { }",
        ptx: ".visible .entry x  .param .u64 p0",
    }
}

#[test]
fn compile_times_land_in_the_bucket_they_belong_to() {
    let meter = Meter::new();
    // One in each of the first, third and last buckets.
    meter.stepped(step("fast", 10, false));
    meter.stepped(step("middling", 900, false));
    meter.stepped(step("slow", 30_000, false));
    // A cached kernel compiled nothing, so it belongs in no bucket.
    meter.stepped(step("free", 0, true));

    let loading = meter.snapshot().loading.unwrap();
    let counts: Vec<u64> = loading.buckets.to_vec();
    assert_eq!(counts, vec![1, 0, 1, 0, 0, 0, 1]);
    assert_eq!(loading.built, 3);
    assert_eq!(loading.cached, 1);
    assert_eq!(loading.slowest, "slow");
    assert_eq!(loading.slowest_took, Duration::from_secs(30));
    // Summed lowering time, which a parallel batch makes exceed the clock.
    assert_eq!(loading.compile_time, Duration::from_millis(30_910));

    // Only what was built put anything through the compiler.
    assert_eq!(loading.source_bytes, 3 * "kernel x() { }".len() as u64);
    assert!(loading.expansion().unwrap() > 1.0, "PTX is the longer text");

    let labels: Vec<String> = loading.histogram().into_iter().map(|(l, _)| l).collect();
    assert_eq!(labels[0], "< 250 ms");
    assert_eq!(labels[2], "< 1 s");
    assert_eq!(labels[6], "8s +");
}

#[test]
fn loading_counts_what_was_built_apart_from_what_was_cached() {
    let meter = Meter::new();
    assert!(meter.snapshot().loading.is_none());

    for (i, cached) in [false, true, true].into_iter().enumerate() {
        meter.stepped(Step {
            stage: "kernels",
            item: "q8_qmma",
            done: i + 1,
            total: 3,
            cached,
            took: Duration::from_millis(100),
            source: "kernel q8_qmma() {}",
            ptx: ".visible .entry q8_qmma",
        });
    }
    let loading = meter.snapshot().loading.expect("a step was reported");
    assert_eq!(loading.built, 1);
    assert_eq!(loading.cached, 2);
    assert_eq!(loading.finished(), 3);
    assert_eq!(loading.ratio(), Some(1.0));

    // A batch of one says nothing a bar could draw.
    meter.stepped(Step {
        stage: "kernels",
        item: "alone",
        done: 1,
        total: 1,
        cached: false,
        took: Duration::from_millis(5),
        source: "kernel alone() {}",
        ptx: ".visible .entry alone",
    });
    assert_eq!(meter.snapshot().loading.unwrap().ratio(), None);
}

#[test]
fn a_reused_prefix_is_counted_against_the_prompt() {
    let meter = Meter::new();
    meter.request_started(500, 400);
    meter.prefilled(100, Duration::from_millis(50));
    meter.token(501, Some(1 << 20));
    let done = meter.request_finished("stop").expect("a request was live");

    assert_eq!(done.reused, 400);
    let snap = meter.snapshot();
    assert_eq!(snap.prompt_tokens, 500);
    assert_eq!(snap.prompt_reused, 400);
}

#[test]
fn the_cache_empties_with_the_session() {
    let meter = Meter::new();
    meter.request_started(8, 0);
    meter.prefilled(8, Duration::from_millis(10));
    meter.token(9, Some(1 << 20));
    assert_eq!(meter.snapshot().cache_bytes, Some(1 << 20));
    // Finishing no longer clears it: the session may be kept, and only
    // the caller knows whether it was.
    meter.request_finished("stop");
    assert_eq!(meter.snapshot().cache_bytes, Some(1 << 20));
    meter.set_cache(0, None);
    assert_eq!(meter.snapshot().cache_bytes, None);
    assert_eq!(meter.snapshot().cache_tokens, 0);
}

#[test]
fn rings_keep_the_recent_end() {
    let meter = Meter::new();
    for i in 0..LOG_LINES + 10 {
        meter.log(Level::Info, format!("line {i}"));
    }
    let snap = meter.snapshot();
    assert_eq!(snap.log.len(), LOG_LINES);
    // Newest first.
    assert_eq!(snap.log[0].text, format!("line {}", LOG_LINES + 9));
}

#[test]
fn a_zero_interval_is_not_an_infinite_rate() {
    assert_eq!(rate_of(10, Duration::ZERO), 0.0);
    assert!(rate_of(10, Duration::from_secs(1)).is_finite());
}

#[test]
fn stopping_is_what_a_viewer_says_to_the_engine() {
    let meter = Meter::new();
    assert!(meter.running());
    meter.stop();
    assert!(!meter.running());
}
