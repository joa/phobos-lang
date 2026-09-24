// Rendering tests against a backend that draws into a buffer.
//
// They cannot say whether the dashboard looks right, which is a judgement for
// somebody with a terminal. They can say that every size renders without
// panicking and that the figures reach the screen, which is what a layout
// change is most likely to break: a panel one row too tall to fit does not
// fail a type check.

use std::time::Duration;

use ratatui::Terminal;
use ratatui::backend::TestBackend;

use phobos_base::log::Level;
use phobos_inference::telemetry::{Fixed, Meter, Snapshot};
use phobos_inference::{Architecture, BlockKind, CacheStats, DeviceInfo, DeviceMemory, Footprint};

use super::panels;
use super::view::View;

/// Everything the panels read, filled the way a loaded model would fill it.
fn loaded() -> Snapshot {
    let meter = Meter::new();
    // A qwen-shaped interleave: every fourth block is attention and the rest
    // carry recurrent state, which is the case worth drawing.
    let blocks = (0..48)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                BlockKind::Attention
            } else {
                BlockKind::Recurrent
            }
        })
        .collect();
    meter.describe(Fixed {
        label: "GGUF, qwen35".to_string(),
        backend: "phobos GPU".to_string(),
        vocab_size: 151_936,
        context_limit: 32_768,
        listen: Some("127.0.0.1:8080".to_string()),
        footprint: Some(Footprint {
            weight_bytes: 7 << 30,
            dense_bytes: 1 << 30,
            streamed_bytes: 0,
            kv_bytes_per_token: 256 << 10,
        }),
        architecture: Some(Architecture {
            d_model: 4096,
            d_ff: 12288,
            n_head: 32,
            n_head_kv: 8,
            head_dim: 128,
            blocks,
        }),
        card: Some(DeviceInfo {
            name: "NVIDIA GeForce RTX 2080 SUPER".to_string(),
            capability: (7, 5),
            multiprocessors: 48,
            core_clock_khz: 1_815_000,
            memory_clock_khz: 7_751_000,
            memory_bus_bits: 256,
            cuda: (13, 0),
            driver: Some("610.88".to_string()),
        }),
    });
    meter.set_cache_stats(Some(CacheStats {
        kernels_reused: 3512,
        kernels_compiled: 64,
        buffers_reused: 18204,
        buffers_allocated: 1801,
        buffer_live_bytes: 3 << 30,
        buffer_idle_bytes: 512 << 20,
        expert_hits: 0,
        expert_misses: 0,
        expert_bytes: 0,
        expert_prefetches: 0,
        expert_prefetch_hits: 0,
        expert_cpu_misses: 0,
        expert_cpu_nanos: 0,
    }));
    meter.set_device_memory(Some(DeviceMemory {
        free_bytes: 5 << 30,
        total_bytes: 24 << 30,
    }));
    meter.log(Level::Info, "listening on http://127.0.0.1:8080");
    meter.log(Level::Debug, "POST /v1/chat/completions 412");

    meter.request_started(512, 0);
    meter.prefilled(512, Duration::from_millis(400));
    for i in 0..8 {
        meter.token(513 + i, Some(8 << 20));
    }
    meter.request_finished("stop");

    // The second request of a conversation: most of its prompt was already
    // in the session the first one left behind.
    meter.request_started(128, 96);
    meter.prefilled(128, Duration::from_millis(90));
    meter.token(129, Some(4 << 20));
    meter.snapshot()
}

/// Draw one frame and hand back the screen as lines of text.
fn draw(width: u16, height: u16, snap: &Snapshot) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    let mut view = View::new();
    // Past the splash: these are about the panels behind it.
    view.splash = 0;
    // Several frames, so the eased gauges have moved off their starting
    // values and the rain has fallen: a panic that only happens once the
    // animations are running is still a panic.
    for _ in 0..30 {
        terminal
            .draw(|frame| panels::render(frame, &mut view, snap))
            .unwrap();
        view.tick();
    }
    let buffer = terminal.backend().buffer().clone();
    (0..height)
        .map(|y| {
            (0..width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        })
        .collect()
}

fn screen(width: u16, height: u16, snap: &Snapshot) -> String {
    draw(width, height, snap).join("\n")
}

/// The moon, on the way in. Drawn over everything while it is up, so the
/// panels behind it are not what a test of it should be looking at.
#[test]
fn the_splash_draws_the_moon_and_then_gets_out_of_the_way() {
    let snap = loaded();
    let mut terminal = Terminal::new(TestBackend::new(100, 46)).unwrap();
    let mut view = View::new();

    // Far enough in for the reveal to have reached the bottom.
    for _ in 0..60 {
        terminal
            .draw(|frame| panels::render(frame, &mut view, &snap))
            .unwrap();
        view.tick();
    }
    let buffer = terminal.backend().buffer().clone();
    let text: String = (0..46)
        .map(|y| {
            (0..100)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join(
            "
",
        );
    // The wordmark worked into the bottom of the image.
    assert!(
        text.contains("P          H                O"),
        "the moon is not on screen:
{text}"
    );
    assert!(text.contains("tile-based GPU kernel language"), "{text}");
    assert!(
        !text.contains("THROUGHPUT"),
        "the panels drew under the splash"
    );

    // It gives up the screen on its own.
    while view.splash > 0 {
        view.tick();
    }
    terminal
        .draw(|frame| panels::render(frame, &mut view, &snap))
        .unwrap();
    let after = terminal.backend().buffer().clone();
    let text: String = (0..46)
        .map(|y| (0..100).map(|x| after[(x, y)].symbol()).collect::<String>())
        .collect::<Vec<_>>()
        .join(
            "
",
        );
    assert!(
        text.contains("MEMORY"),
        "the panels never came back:
{text}"
    );
}

/// A terminal too short for the whole moon gets none of it rather than a
/// cropped one, and goes straight to the panels.
#[test]
fn a_small_terminal_skips_the_splash_entirely() {
    let text = screen(80, 24, &loaded());
    assert!(text.contains("THROUGHPUT"), "{text}");
}

#[test]
fn renders_at_every_size_it_may_be_given() {
    let snap = loaded();
    // A tall wide terminal, the ordinary one, the default 80x24, and sizes
    // small enough that a panel has no room left at all.
    for (width, height) in [(200, 60), (120, 40), (80, 24), (60, 20), (40, 10), (20, 5)] {
        let lines = draw(width, height, &snap);
        assert_eq!(lines.len(), height as usize, "{width}x{height}");
        assert!(
            lines
                .iter()
                .all(|line| line.chars().count() == width as usize),
            "{width}x{height} produced a ragged frame"
        );
    }
}

#[test]
fn an_empty_meter_renders() {
    // What the first frame draws, before a request has arrived: every
    // optional field is absent at once.
    let snap = Meter::new().snapshot();
    for (width, height) in [(120, 40), (80, 24), (30, 8)] {
        draw(width, height, &snap);
    }
}

#[test]
fn the_panels_are_on_screen() {
    let text = screen(140, 44, &loaded());
    for title in [
        "MODEL",
        "MEMORY",
        "THROUGHPUT",
        "NETWORK",
        "CARD",
        "CACHES",
        "ACTIVITY",
        "LOG",
    ] {
        assert!(text.contains(title), "{title} is missing:\n{text}");
    }
    for key in ["quit", "clear the log", "reset the peaks"] {
        assert!(text.contains(key), "{key} is missing from the footer");
    }
}

#[test]
fn the_figures_reach_the_screen() {
    let text = screen(140, 44, &loaded());
    assert!(text.contains("GGUF, qwen35"), "the engine label is missing");
    assert!(text.contains("phobos GPU"), "the backend is missing");
    assert!(text.contains("24.00 GiB"), "the card total is missing");
    assert!(text.contains("7.00 GiB"), "the weight figure is missing");
    assert!(text.contains("127.0.0.1:8080"), "the address is missing");
    // Mid-request, so the phase is the decode it is in.
    assert!(text.contains("DECODE"), "the phase is missing");
}

#[test]
fn the_context_cost_reads_as_a_projection_and_not_as_usage() {
    let text = screen(150, 46, &loaded());
    // What the caches would take if the conversation ever ran the whole
    // context, which is not what they take now, and used to be labelled as
    // though it were.
    assert!(
        text.contains("if the context ever filled"),
        "the projection is unlabelled"
    );
    assert!(
        text.contains("more fit in what is free"),
        "the figure worth acting on is missing"
    );
    // What they actually take, which is the gauge beside it.
    assert!(text.contains("4.00 MiB"), "the live cache size is missing");
}

#[test]
fn the_card_is_described_including_both_version_numbers() {
    let text = screen(150, 46, &loaded());
    assert!(text.contains("RTX 2080 SUPER"), "the card name is missing");
    // The display driver's version and the CUDA API's are different numbers
    // and both are worth having; only the second is one CUDA can report.
    assert!(text.contains("610.88"), "the driver version is missing");
    assert!(text.contains("CUDA 13.0"), "the CUDA version is missing");
    assert!(text.contains("sm_75"), "the compute capability is missing");
    assert!(text.contains("48 SMs"), "the SM count is missing");
    // 2 x 32 bytes x 7.751 GHz.
    assert!(text.contains("496 GB/s"), "the peak bandwidth is missing");
}

#[test]
fn the_block_strip_counts_both_kinds() {
    let text = screen(150, 46, &loaded());
    // 48 blocks, every fourth one attention.
    assert!(text.contains("attention 12"), "{text}");
    assert!(text.contains("recurrent 36"), "{text}");
    assert!(text.contains("d_model"), "{text}");
}

#[test]
fn the_caches_say_what_they_count() {
    let text = screen(150, 46, &loaded());
    // 96 of 640 prompt positions came out of a kept session.
    for words in [
        "prefix hit",
        "15.0% of 640 prompt tokens",
        "3.00 GiB in use, 512.00 MiB idle",
        "1 801 allocated, 18 204 reused",
    ] {
        assert!(
            text.contains(words),
            "{words:?} is missing:
{text}"
        );
    }
    // Buffers are a count, not a rate, and kernels are not shown.
    assert!(!text.contains("91.0%"), "the buffer hit rate is back");
    assert!(!text.contains("launches"), "the kernel row is back");
}

#[test]
fn memory_accounts_for_what_is_neither_weights_nor_cache() {
    let text = screen(150, 46, &loaded());
    for label in ["weights", "kv cache", "elsewhere", "free"] {
        assert!(text.contains(label), "{label} is missing from the legend");
    }
}

/// Weights reach the card lazily, so between loading a model and its first
/// pass the footprint is larger than anything the driver says is in use.
/// Drawing the whole footprint then puts memory on the screen that is not on
/// the card.
#[test]
fn weights_not_yet_uploaded_are_not_drawn_as_resident() {
    let meter = Meter::new();
    meter.describe(Fixed {
        label: "GGUF, llama".to_string(),
        backend: "phobos GPU".to_string(),
        context_limit: 4096,
        // 5.6 GiB of weights, of which none is up yet.
        footprint: Some(Footprint {
            weight_bytes: 6_012_954_214,
            dense_bytes: 0,
            streamed_bytes: 0,
            kv_bytes_per_token: 1 << 10,
        }),
        ..Default::default()
    });
    // The card holds 1.30 GiB, all of it somebody else's.
    meter.set_device_memory(Some(DeviceMemory {
        free_bytes: 7_390_069_064,
        total_bytes: 8_589_934_592,
    }));

    let text = screen(150, 46, &meter.snapshot());
    assert!(
        text.contains("of 5.60 GiB up"),
        "the weights are drawn as resident when they are not:
{text}"
    );
    // The card's own figure is what the headline reports.
    assert!(
        text.contains("1.12 GiB of 8.00 GiB"),
        "
{text}"
    );
}

#[test]
fn a_host_build_says_it_has_no_card_to_read() {
    // No device and no footprint: what an ONNX model on the host reports.
    let meter = Meter::new();
    meter.describe(Fixed {
        label: "ONNX, KV-cached".to_string(),
        backend: "host reference".to_string(),
        vocab_size: 50_257,
        context_limit: 1024,
        ..Default::default()
    });
    let text = screen(140, 44, &meter.snapshot());
    assert!(text.contains("no device to read"), "\n{text}");
    assert!(text.contains("not reported by this front end"), "\n{text}");
    assert!(text.contains("this backend keeps none"), "{text}");
    assert!(text.contains("this backend runs on the host"), "{text}");
}

#[test]
fn formatting_is_readable_at_the_scales_these_reach() {
    assert_eq!(panels::bytes(512), "512 B");
    assert_eq!(panels::bytes(1536), "1.50 KiB");
    assert_eq!(panels::bytes(7 << 30), "7.00 GiB");
    assert_eq!(panels::count(0), "0");
    assert_eq!(panels::count(151_936), "151 936");
    assert_eq!(panels::duration(Duration::from_millis(1500)), "1.5s");
    assert_eq!(panels::duration(Duration::from_secs(90)), "1m30s");
    assert_eq!(panels::duration(Duration::from_secs(7265)), "2h01m");
}

/// Before there is a model, the load is the only thing happening and gets
/// the screen. A cold start spends minutes here.
fn mid_load(cached: bool) -> Snapshot {
    // Real shapes: a kernel's text is short and its PTX is not.
    const SOURCE: &str = "kernel iq2s_matvec(A: tensor<f32>[M, K]) {\n\
                          let pm = program_id(0)\n\
                          var acc: tile<f32>[TM, TN] = 0.0\n\
                          for kt in range(0, K, TK) { acc += dot(a, b) }\n\
                          }\n";
    const PTX: &str = ".visible .entry iq2s_matvec(\n\
                       .param .u64 iq2s_matvec_param_0\n\
                       ) {\n\
                       ld.param.u64 %rd1, [iq2s_matvec_param_0];\n\
                       mov.u32 %r1, %ctaid.x;\n\
                       ret;\n\
                       }\n";
    let meter = Meter::new();
    for i in 0..12 {
        meter.starting("iq2s_matvec");
        meter.stepped(phobos_base::progress::Step {
            stage: "kernels",
            item: "iq2s_matvec",
            done: i + 1,
            total: 37,
            cached,
            // A spread, so the histogram has more than one bar.
            took: std::time::Duration::from_millis(if cached { 0 } else { 120 * (i as u64 + 1) }),
            source: SOURCE,
            ptx: PTX,
        });
    }
    meter.snapshot()
}

#[test]
fn a_load_reports_what_it_is_building() {
    let text = screen(150, 46, &mid_load(false));
    assert!(text.contains("LOADING"), "{text}");
    assert!(text.contains("iq2s_matvec"), "the kernel name is missing");
    assert!(
        text.contains("12 of 37"),
        "the position is missing:\n{text}"
    );
    assert!(
        text.contains("32.43%"),
        "the percentage is missing:\n{text}"
    );
    assert!(text.contains("12 built"), "the built count is missing");
    // The panels have nothing to show yet, so they are not drawn.
    assert!(!text.contains("THROUGHPUT"), "the panels drew too early");
}

/// The case that prompted this: a batch starts 71 kernels at once, one of
/// them takes ten minutes, and reporting only completions leaves its name off
/// the screen for the whole of that wait while a finished kernel's name sits
/// there instead.
#[test]
fn the_kernel_the_batch_is_waiting_on_is_named() {
    let meter = Meter::new();
    // The whole batch starts together, as a real one does.
    for name in ["q2k_matvec", "q3k_matvec", "iq1s_matvec"] {
        meter.starting(name);
    }
    // Two come back; the third does not.
    for (i, name) in ["q2k_matvec", "q3k_matvec"].into_iter().enumerate() {
        meter.stepped(phobos_base::progress::Step {
            stage: "kernels",
            item: name,
            done: i + 1,
            total: 3,
            cached: false,
            took: std::time::Duration::from_millis(400),
            source: "kernel x() {}",
            ptx: ".visible .entry x",
        });
    }

    let snap = meter.snapshot();
    let load = snap.loading.as_ref().unwrap();
    assert_eq!(load.in_flight.len(), 1, "two of three came back");
    assert_eq!(load.longest().unwrap().0, "iq1s_matvec");

    let text = screen(150, 46, &snap);
    assert!(text.contains("waiting on"), "{text}");
    assert!(
        text.contains("iq1s_matvec"),
        "the kernel holding up the batch is not on screen:
{text}"
    );
}

#[test]
fn the_compile_screen_shows_the_real_source_and_the_real_ptx() {
    let text = screen(150, 46, &mid_load(false));
    assert!(text.contains("COMPILING"), "{text}");
    // The kernel's own text, going in.
    assert!(
        text.contains("program_id"),
        "the source is missing:\n{text}"
    );
    // The PTX it became, coming out.
    assert!(text.contains("ctaid"), "the ptx is missing:\n{text}");
    // And that PTX as bytes. `.` is 2e, the first character of `.visible`.
    assert!(text.contains("2e "), "the bytes are missing:\n{text}");
    // The stages the text appears to pass through.
    for stage in ["PARSE", "MLIR", "LLVM", "PTX"] {
        assert!(text.contains(stage), "{stage} is missing");
    }
}

#[test]
fn the_histogram_says_where_the_compile_time_went() {
    let text = screen(150, 46, &mid_load(false));
    assert!(text.contains("COMPILE TIMES"), "{text}");
    assert!(
        text.contains("< 250 ms"),
        "the buckets are missing:\n{text}"
    );
    assert!(text.contains("slowest"), "the slowest kernel is missing");
    // 12 kernels at 120ms .. 1440ms.
    assert!(
        text.contains("1.44s"),
        "the slowest time is missing:\n{text}"
    );
    assert!(text.contains("ptx to source"), "the expansion is missing");
}

#[test]
fn a_warm_load_says_the_kernels_were_already_there() {
    let text = screen(150, 46, &mid_load(true));
    assert!(text.contains("12 already cached"), "{text}");
    // Nothing was built, so the note about a cold start is not shown.
    assert!(!text.contains("builds every kernel from source"), "{text}");
}
