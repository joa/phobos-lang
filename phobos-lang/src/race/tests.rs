use super::*;

fn check(src: &str) -> Vec<Report> {
    let base = phobos_base::context::Context::default();
    check_source(&base, src, &Config::default()).expect("the kernel compiles and runs")
}

fn one(src: &str) -> Report {
    check(src).into_iter().next().expect("one kernel")
}

/// Each thread writes its own element of a tile and reads its mirror's,
/// which another warp wrote: a race until a `barrier()` separates them.
#[test]
fn a_mirrored_read_races_without_a_barrier() {
    let racy = one("@launch(64)
        kernel k(X: tensor<f32>[1, 64], Y: tensor<f32>[1, 64]) {
            var t: tile<f32>[1, 64] = 0.0
            t[0, ctx.thread] = X[0, ctx.thread]
            Y[0, ctx.thread] = t[0, 63 - ctx.thread]
        }");
    assert!(racy.stopped.is_none(), "{racy:#?}");
    assert_eq!(racy.races.len(), 1, "{racy:#?}");
    let race = &racy.races[0];
    assert!(race.first.write && !race.second.write, "{race:#?}");

    let ordered = one("@launch(64)
        kernel k(X: tensor<f32>[1, 64], Y: tensor<f32>[1, 64]) {
            var t: tile<f32>[1, 64] = 0.0
            t[0, ctx.thread] = X[0, ctx.thread]
            barrier()
            Y[0, ctx.thread] = t[0, 63 - ctx.thread]
        }");
    assert!(ordered.clean(), "{ordered:#?}");
}

/// The tile staged each turn and read by lane loads, which the compiler
/// once left unordered across turns.
#[test]
fn a_tile_restaged_after_lane_reads_is_ordered() {
    let reports = check(
        "@autotune(K in [256])
        @launch(128)
        kernel k(W: tensor<f32>[N, K], x: tensor<f32>[1, K], y: tensor<f32>[1, N]) {
            let row = ctx.block.x * ctx.warps + ctx.warp
            var acc: f32 = 0.0
            for kt in range(0, K, ctx.lanes * 4) {
                var xs = x[0 :+ 1, kt :+ ctx.lanes * 4]
                let at = ctx.lane * 4
                if row < N {
                    acc += sum(W[row, kt + at :+ 4] * xs[0, at :+ 4])
                }
            }
            let total = sum<over ctx.lane>(acc)
            if ctx.lane == 0 && row < N {
                y[0, row] = total
            }
        }",
    );
    for r in &reports {
        assert!(r.clean(), "{r:#?}");
        assert!(
            r.bytes > 0 && r.barriers >= 2,
            "the loop's accesses were not seen: {r:#?}"
        );
    }
}

/// A tile kernel with a loop, a reduction and a dot: every barrier the
/// compiler keeps or elides is checked against what each thread touches.
#[test]
fn a_tile_kernel_is_clean() {
    let r = one(
        "@aligned(M = 32, N = 32, K = 16)
        @launch(128)
        kernel k(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N], S: tensor<f32>[M, 1]) {
            var acc: tile<f32>[32, 32] = 0.0
            for kt in range(0, K, 16) {
                let a = A[ctx.block.x * 32 :+ 32, kt :+ 16]
                let b = B[kt :+ 16, ctx.block.y * 32 :+ 32]
                acc += dot(a, b)
            }
            var s: tile<f32>[32, 1] = rowsum(acc)
            C[ctx.block.x * 32 :+ 32, ctx.block.y * 32 :+ 32] = acc
            S[ctx.block.x * 32 :+ 32, 0 :+ 1] = s
        }",
    );
    assert!(r.clean(), "{r:#?}");
    assert!(r.possible.is_empty() && r.warp_races.is_empty(), "{r:#?}");
}

/// A transpose through shared memory in f16: each scalar access moves one
/// two-byte element, so neighbours written by other threads do not overlap.
#[test]
fn half_precision_elements_do_not_overlap() {
    let r = one("@launch(256)
        @autotune(TB in [8], HD in [256])
        @aligned(NK = 64, KW = HD, DK = TB)
        kernel attn_kt(K: tensor<f16>[NK, KW], T: tensor<f32>[DK, NK]) {
            let p = ctx.block.x
            var t = K[p * TB :+ TB, 0 :+ HD]
            T[0 :+ HD, p * TB :+ TB] = f32(transpose(t))
        }");
    assert!(r.clean(), "{r:#?}");
    assert!(r.possible.is_empty() && r.warp_races.is_empty(), "{r:#?}");
}

/// Every kernel source in the tree, on each target `emit` is diffed under,
/// runs with no race between warps, no divergent barrier and no access
/// outside a buffer.
#[test]
fn every_kernel_in_the_tree_is_race_free() {
    use phobos_base::context::{GpuConfig, NvidiaGpuConfig};

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut sources = Vec::new();
    for dir in ["examples", "phobos-lang/examples"] {
        for entry in std::fs::read_dir(root.join(dir)).unwrap().flatten() {
            if entry.path().extension().is_some_and(|e| e == "ph") {
                sources.push(entry.path());
            }
        }
    }
    assert!(sources.len() >= 10, "found {} sources", sources.len());
    let mut failures = Vec::new();
    for (chip, bits) in [("sm_75", 32), ("sm_75", 64), ("sm_80", 64)] {
        let base = phobos_base::context::Context {
            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip.to_string())),
            index_bitwidth: bits,
            ..Default::default()
        };
        for path in &sources {
            let src = std::fs::read_to_string(path).unwrap();
            let reports = check_source(&base, &src, &Config::default())
                .unwrap_or_else(|e| panic!("{} on {chip}/{bits}: {e:#}", path.display()));
            for r in reports.iter().filter(|r| !r.clean()) {
                failures.push(format!("{} on {chip}/{bits}: {r:#?}", path.display()));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A claimed value every thread reads from a shared slot, then a staged
/// tile over the slot's bytes: the copy waits for every warp's read.
#[test]
fn a_claimed_slot_is_read_before_its_bytes_are_reused() {
    let r = one(
        "kernel k(L: tensor<i32>[1], X: tensor<f32>[16, 64]) {
            if atomic_cas(L, 0, 0, 1) == 0 {
                var x = X[0 :+ 16, 0 :+ 64]
                var m: tile<f32>[16, 1] = rowmax(x)
                X[0 :+ 16, 0 :+ 1] = m
            }
        }",
    );
    assert!(r.clean(), "{r:#?}");
    assert!(r.warp_races.is_empty(), "{r:#?}");
}

const MIRRORED: &str = "@launch(64)
kernel k(X: tensor<f32>[1, 64], Y: tensor<f32>[1, 64]) {
    var t: tile<f32>[1, 64] = 0.0
    t[0, ctx.thread] = X[0, ctx.thread]
    Y[0, ctx.thread] = t[0, 63 - ctx.thread]
}
";

fn mirrored() -> (Vec<crate::ast::Kernel>, Vec<Report>) {
    let file = crate::diag::SourceFile::new("race.ph", MIRRORED);
    let kernels = crate::parse_source(&file).expect("the kernel parses");
    let base = phobos_base::context::Context::default();
    let reports = check_kernels(&base, &kernels, &Config::default()).expect("the kernel runs");
    (kernels, reports)
}

#[test]
fn a_race_points_at_both_accesses_and_suggests_a_barrier() {
    let (kernels, reports) = mirrored();
    let found = diagnostics(&kernels, &reports).expect("the race is reported");
    // Both CTAs see it; it is reported once.
    assert_eq!(found.errors(), 1, "{found}");
    let out = found.render(crate::diag::Colors::Off);
    assert!(out.starts_with("error: a thread reads shared memory another warp writes, with no `barrier()` between\n --> race.ph:5:"), "{out}");
    assert!(out.contains("4 |     t[0, ctx.thread] = X[0, ctx.thread]\n"), "{out}");
    assert!(out.contains("writes here"), "{out}");
    assert!(out.contains("reads here"), "{out}");
    assert!(out.contains("help: a barrier between the two orders them\n  |\n5 +     barrier()\n6 |     Y[0, ctx.thread]"), "{out}");
}

#[test]
fn a_clean_kernel_has_nothing_to_report() {
    let src = MIRRORED.replace("    Y[", "    barrier()\n    Y[");
    let file = crate::diag::SourceFile::new("race.ph", src);
    let kernels = crate::parse_source(&file).unwrap();
    let base = phobos_base::context::Context::default();
    let reports = check_kernels(&base, &kernels, &Config::default()).unwrap();
    assert!(diagnostics(&kernels, &reports).is_none(), "{reports:#?}");
}

/// MLIR printed with its locations, as `emit --loc` prints it, parses back
/// with them.
#[test]
fn printed_locations_come_back_from_the_text() {
    use melior::ir::operation::OperationPrintingFlags;

    let (kernels, emitted) = mirrored();
    let context = context();
    let module = Module::new(Location::unknown(&context));
    let base = phobos_base::context::Context::default();
    crate::codegen::emit(&base, &kernels, &context, &module).unwrap();
    let flags = OperationPrintingFlags::new().enable_debug_info(true, false);
    let text = module.as_operation().to_string_with_flags(flags).unwrap();
    assert!(text.contains(r#"loc("race.ph":5:"#), "{text}");
    let parsed = check_text(&text, &Config::default()).unwrap();
    let (a, b) = (&emitted[0].races[0], &parsed[0].races[0]);
    assert_eq!((&a.first.loc, &a.second.loc), (&b.first.loc, &b.second.loc));
    let second = b.second.loc.as_ref().expect("a location");
    assert_eq!((second.file.as_str(), second.line), ("race.ph", 5));
}

#[test]
fn an_async_group_parses_back_with_its_location() {
    assert_eq!(
        reparsable("  %1 = nvgpu.device_async_create_group loc(#loc3)\n  %2 = nvgpu.device_async_create_group\n"),
        concat!(
            r#"  %1 = "nvgpu.device_async_create_group"() : () -> !nvgpu.device.async.token loc(#loc3)"#,
            "\n",
            r#"  %2 = "nvgpu.device_async_create_group"() : () -> !nvgpu.device.async.token"#,
            "\n",
        )
    );
}
