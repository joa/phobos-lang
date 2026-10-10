use super::*;
use crate::backend::{DeltaMix, FusedMix, ProjRun, ProjWeight};

/// Qwen3.5-0.8B's MLP at 192 blocks.
fn qwen_plan() -> Plan {
    let chain = mlp_chain(Buf(0), Buf(1), QBuf(0), QBuf(1), 1024, 3584, 1e-6);
    chain
        .key(192)
        .plan()
        .expect("a well-formed chain")
        .expect("a shape the pass fuses")
}

/// Qwen3.5-0.8B's delta-net input: normalization plus the stacked qkv,
/// output gate, decay and write strength projection. `mix` adds the
/// convolution and gates.
fn qwen_project_plan(mix: bool) -> Plan {
    let (channels, carried, gates) = (6144, 3 * 6144, 2048 + 2 * 16);
    let runs = [
        ProjRun {
            weight: 0,
            row_off: 0,
            width: channels,
            dst: Buf(2),
            dst_off: carried,
        },
        ProjRun {
            weight: 0,
            row_off: channels,
            width: gates,
            dst: Buf(3),
            dst_off: channels,
        },
    ];
    let spec = DeltaMix {
        rows: 1,
        heads: 16,
        head_dim: 128,
        kv_heads: 16,
        kernel: 4,
        planes: [0, 2048, 4096],
        head_stride: 128,
        normalize: true,
        query_scale: (128.0f32).sqrt().recip(),
    };
    let project = FusedProject {
        x: Buf(0),
        d_model: 1024,
        gain: Buf(1),
        eps: 1e-6,
        weights: &[(ProjWeight::Q8(QBuf(0)), 8448)],
        runs: &runs,
        mix: mix.then_some(FusedMix {
            spec,
            history: Buf(2),
            taps: Buf(4),
            // The decay and write strength follow the 2048-wide output gate
            // in the second run.
            decay: (Buf(3), channels + 2048),
            beta: (Buf(3), channels + 2048 + 16),
            rate: Buf(5),
            dt_bias: Buf(6),
            packed: Buf(7),
        }),
    };
    project_chain(&project)
        .expect("a shape the pass records")
        .key(192)
        .plan()
        .expect("a well-formed chain")
        .expect("a shape the pass fuses")
}

/// Prints the emitted kernels, for feeding to
/// `cargo run -p phobos-lang --example emit`.
///
///     cargo test -p phobos-gguf fused_source -- --nocapture --ignored
#[test]
#[ignore = "prints the emitted source rather than checking anything"]
fn fused_source() {
    let plans = [
        ("fused_qwen08_mlp", qwen_plan()),
        ("fused_qwen08_project", qwen_project_plan(false)),
        ("fused_qwen08_project_mix", qwen_project_plan(true)),
        ("fused_qwen4b_mlp", qwen_4b_plan(64)),
        ("fused_qwen4b_project", qwen_4b_project_plan()),
    ];
    // With PHOBOS_DUMP_DIR set, each is also written there as a `.ph`.
    let dir = std::env::var("PHOBOS_DUMP_DIR").ok().map(std::path::PathBuf::from);
    for (name, plan) in plans {
        println!("{}", plan.source);
        if let Some(dir) = &dir {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join(format!("{name}.ph")), &plan.source).unwrap();
        }
    }
}

#[test]
fn the_mlp_chain_needs_exactly_one_barrier() {
    let plan = qwen_plan();
    assert_eq!(plan.barriers, 1);
    assert_eq!(plan.source.matches("grid_barrier").count(), 1);
    // It sits after the SwiGLU's quantization, since the down projection
    // reads the whole hidden row. The normalization is redundant, so it
    // needs none.
    let (before, after) = plan.source.split_once("grid_barrier").expect("a barrier");
    assert!(before.contains("rms_norm_q_t("), "the normalization comes first");
    assert!(before.contains("exp(-v1)"), "the SwiGLU comes first");
    assert!(after.contains("+= qdot_t"), "the accumulation comes after");
}

#[test]
fn only_a_value_crossing_a_barrier_reaches_global_memory() {
    let plan = qwen_plan();
    // The SwiGLU's quantized row, read whole by the down projection.
    assert_eq!(plan.scratch.len(), 1);
    assert_eq!(plan.scratch[0].bytes, 3584);
    assert_eq!(plan.scratch[0].scales, 112);
    // The normalized row is held in shared memory per block.
    assert_eq!(plan.held, 1);
    assert_eq!(plan.source.matches("flat(").count(), 2);
    assert!(
        plan.source.contains("var Aq2: tile<i8>[32, 32]"),
        "the activation should be a tile, not an operand"
    );
}

/// A held value lives in shared memory, so its size does not depend on the
/// block count.
#[test]
fn the_held_row_does_not_scale_with_the_grid() {
    let at = |blocks: u32| {
        let chain = mlp_chain(Buf(0), Buf(1), QBuf(0), QBuf(1), 1024, 3584, 1e-6);
        let plan = chain
            .key(blocks)
            .plan()
            .expect("well-formed")
            .expect("fusable");
        let held = plan
            .source
            .lines()
            .filter(|l| l.contains("tile<i8>") || l.contains("flat("))
            .map(str::trim)
            .map(str::to_string)
            .collect::<Vec<_>>();
        (plan.scratch, held)
    };
    let (small, small_held) = at(48);
    let (large, large_held) = at(192);
    assert_eq!(small_held, large_held);
    assert_eq!(small[0].bytes, large[0].bytes);
}

/// The residual is read folded and accumulated flat, so it is bound twice.
#[test]
fn the_residual_is_bound_under_both_views() {
    let plan = qwen_plan();
    let x: Vec<[i64; 2]> = plan
        .slots
        .iter()
        .filter(|s| matches!(s.bound, Bound::Given(v) if v == Val(0)))
        .map(|s| s.dims)
        .collect();
    assert_eq!(x, vec![[32, 32], [1, 1024]]);
}

/// A mixer's projection needs no barrier. The normalization is redundant
/// and the runs write disjoint windows that nothing in the chain reads back.
#[test]
fn the_mixer_projection_needs_no_barrier() {
    let plan = qwen_project_plan(false);
    assert_eq!(plan.barriers, 0);
    assert!(!plan.source.contains("grid_barrier"));
    // Nothing is published. The only value passed between nests is the
    // normalized row, held in shared memory.
    assert!(plan.scratch.is_empty());
    assert_eq!(plan.held, 1);
}

#[test]
fn a_run_is_bound_only_as_far_as_it_writes() {
    let plan = qwen_project_plan(false);
    let given: Vec<[i64; 2]> = plan
        .slots
        .iter()
        .filter(|s| matches!(s.bound, Bound::Given(v) if v.0 > 1))
        .map(|s| s.dims)
        .collect();
    // The stream up to the end of the new position, and the stacked
    // projection up to the end of the write strength. Neither is the
    // caller's full buffer size.
    assert_eq!(given, vec![[1, 4 * 6144], [1, 6144 + 2080]]);
    assert!(plan.source.contains("DO1 in [18432]"));
    assert!(plan.source.contains("OF2 in [6144], DO2 in [6144]"));
}

/// The gates share the convolution's nest, so the convolution's barrier
/// covers them too.
#[test]
fn the_convolution_costs_one_barrier_and_the_gates_none() {
    let plan = qwen_project_plan(true);
    assert_eq!(plan.barriers, 1);
    let (before, after) = plan.source.split_once("grid_barrier").expect("a barrier");
    assert!(before.contains("qdot_t"), "the projection comes first");
    assert!(!before.contains("rowsum(y"), "the convolution comes after");
    assert!(after.contains("rowsum(y3"), "the convolution");
    assert!(after.contains("exp(X6["), "the gates");
    // Both in one grid-strided loop.
    assert_eq!(after.matches("in range(0, IT").count(), 1);
    assert!(after.contains("UN3 in [48]") || plan.source.contains("UN3 in [48]"));
}

/// The convolution reads what the projection wrote to the stream. The pass
/// sees that dependency only if both name one value.
#[test]
fn the_stream_is_one_value_under_two_shapes() {
    let plan = qwen_project_plan(true);
    let stream: Vec<[i64; 2]> = plan
        .slots
        .iter()
        .filter(|s| matches!(s.bound, Bound::Given(v) if v == Val(2)))
        .map(|s| s.dims)
        .collect();
    // Flat for the projection's write, by position for the convolution's
    // taps.
    assert_eq!(stream, vec![[1, 24576], [4, 6144]]);
}

/// A convolution without gates still needs the barrier, because a head's
/// row spans several of the projection's units.
#[test]
fn the_convolution_alone_still_costs_the_barrier() {
    let (channels, carried) = (6144, 3 * 6144);
    let mut chain = Chain::default();
    let x = chain.given(Buf(0), 1024);
    let gain = chain.given(Buf(1), 1024);
    let stream = chain.given(Buf(2), carried + channels);
    let taps = chain.given(Buf(3), 4 * channels);
    let packed = chain.given(Buf(4), 6176);
    let act = chain.quant(1024);
    chain.push(Stage::norm_q(x, gain, act, 1024, 1e-6));
    let w = chain.weight(QBuf(0), channels, 1024);
    chain.push(Stage::ProjF {
        a: act,
        w,
        out: stream,
        out_off: carried,
        units: channels / Q8_BLOCK,
        row_off: 0,
    });
    chain.push(Stage::Conv {
        history: stream,
        taps,
        out: packed,
        planes: 3,
        heads: 16,
        head_dim: 128,
        kernel: 4,
        channels,
        plane_base: 0,
        plane_stride: 2048,
        head_stride: 128,
        normalize: true,
        scale_bits: 1.0f32.to_bits(),
    });
    let plan = chain
        .key(192)
        .plan()
        .expect("well-formed")
        .expect("fusable");
    assert_eq!(plan.barriers, 1);
}

/// The gates go where the delta rule reads them: after the three planes,
/// the decay then the write strength.
#[test]
fn the_gates_land_behind_the_planes() {
    let plan = qwen_project_plan(true);
    let (span, heads) = (2048, 16);
    assert!(
        plan.source
            .contains(&format!("X5[0 :+ 1, {} + u3 :+ 1] = exp(", 3 * span))
    );
    assert!(plan.source.contains(&format!(
        "X5[0 :+ 1, {} + u3 :+ 1] = 1.0 /",
        3 * span + heads
    )));
    // At one position a plane is exactly `heads` rows, so the unit index is
    // the packed row and the planes need no offset.
    assert!(plan.source.contains("X5[0 :+ 1, u3 * 128 :+ HD3]"));
    assert_eq!(span, heads * 128);
}

/// Only the query carries the readout scale, and only the query and key are
/// normalized. The gain branches on the plane, which is uniform across the
/// CTA.
#[test]
fn only_the_query_and_key_are_normalized() {
    let plan = qwen_project_plan(true);
    let gains: Vec<&str> = plan
        .source
        .lines()
        .skip_while(|l| !l.contains("var g3"))
        .filter(|l| l.contains("g3 = "))
        .map(str::trim)
        .collect();
    assert_eq!(gains.len(), 2, "the value plane keeps the default gain");
    assert_eq!(
        gains[0],
        "g3 = 0.088388346 / sqrt(rowsum(y3 * y3) + 0.000000000001)"
    );
    assert_eq!(
        gains[1],
        "g3 = 1.0 / sqrt(rowsum(y3 * y3) + 0.000000000001)"
    );
    // Guarded on the plane, with 1.0 as the value plane's default.
    assert!(plan.source.contains("if pl3 == 0 {"));
    assert!(plan.source.contains("if pl3 == 1 {"));
    assert!(plan.source.contains("var g3: tile<f32>[1, 1] = 1.0"));
}

/// More than one position is declined; the pass only emits the decode
/// shape.
#[test]
fn a_prompt_pass_is_not_recorded() {
    let (channels, carried) = (6144, 3 * 6144);
    let runs = [ProjRun {
        weight: 0,
        row_off: 0,
        width: channels,
        dst: Buf(2),
        dst_off: carried,
    }];
    let spec = DeltaMix {
        rows: 4,
        heads: 16,
        head_dim: 128,
        kv_heads: 16,
        kernel: 4,
        planes: [0, 2048, 4096],
        head_stride: 128,
        normalize: true,
        query_scale: 1.0,
    };
    let project = FusedProject {
        x: Buf(0),
        d_model: 1024,
        gain: Buf(1),
        eps: 1e-6,
        weights: &[(ProjWeight::Q8(QBuf(0)), 6144)],
        runs: &runs,
        mix: Some(FusedMix {
            spec,
            history: Buf(2),
            taps: Buf(4),
            decay: (Buf(3), 0),
            beta: (Buf(3), 16),
            rate: Buf(5),
            dt_bias: Buf(6),
            packed: Buf(7),
        }),
    };
    assert!(project_chain(&project).is_none());
}

/// A run that is not whole Q8_0 output blocks is not recorded.
#[test]
fn a_ragged_run_is_not_recorded() {
    let runs = [ProjRun {
        weight: 0,
        row_off: 0,
        width: 48,
        dst: Buf(2),
        dst_off: 0,
    }];
    let project = FusedProject {
        x: Buf(0),
        d_model: 1024,
        gain: Buf(1),
        eps: 1e-6,
        weights: &[(ProjWeight::Q8(QBuf(0)), 48)],
        runs: &runs,
        mix: None,
    };
    assert!(project_chain(&project).is_none());
}

/// A row that is not whole Q8_0 blocks is declined.
#[test]
fn an_unfoldable_width_is_declined() {
    let chain = mlp_chain(Buf(0), Buf(1), QBuf(0), QBuf(1), 1040, 3584, 1e-6);
    assert!(chain.key(192).plan().expect("well-formed").is_none());
}

/// A register value read outside the nest that wrote it is declined.
#[test]
fn a_register_crossing_a_nest_is_declined() {
    let mut chain = Chain::default();
    let x = chain.given(Buf(0), 1024);
    let gain = chain.given(Buf(1), 1024);
    let act = chain.quant(1024);
    chain.push(Stage::norm_q(x, gain, act, 1024, 1e-6));
    let w = chain.weight(QBuf(0), 7168, 1024);
    let gate = chain.temp(Q8_BLOCK);
    chain.push(Stage::ProjQ {
        a: act,
        w,
        out: gate,
        units: 112,
        row_off: 0,
    });
    // A different unit count starts a second nest.
    let up = chain.temp(Q8_BLOCK);
    chain.push(Stage::ProjQ {
        a: act,
        w,
        out: up,
        units: 64,
        row_off: 3584,
    });
    let out = chain.temp(Q8_BLOCK);
    chain.push(Stage::Swiglu {
        g: gate,
        u: up,
        out,
    });
    assert!(chain.key(192).plan().expect("well-formed").is_none());
}

/// Qwen3.5-4B-Q4_K_M's MLP: gate and up as separate Q4_K weights, the down
/// projection Q6_K, at `blocks` blocks.
fn qwen_4b_plan(blocks: u32) -> Plan {
    let chain = mlp_chain_raw(
        Buf(0),
        Buf(1),
        (RawBuf(0), Quant::Q4_K),
        (RawBuf(1), Quant::Q4_K),
        (RawBuf(2), Quant::Q6_K),
        2560,
        9216,
        1e-6,
    )
    .expect("three formats with a fused decode");
    chain
        .key(blocks)
        .plan()
        .expect("a well-formed chain")
        .expect("a shape the pass fuses")
}

#[test]
fn the_raw_mlp_decodes_each_weight_with_its_own_intrinsic() {
    // At 64 blocks the down projection's 40 tiles keep the grid busy.
    let plan = qwen_4b_plan(64);
    assert_eq!(plan.barriers, 1);
    let src = &plan.source;
    assert_eq!(src.matches("= q4k_qdot_i8_t(").count(), 2, "gate and up");
    assert_eq!(src.matches("+= q6k_qdot_i8_t(").count(), 1, "the down projection");
    // Block bytes and `d` plane. A 2560-wide row is ten Q4_K blocks of 144
    // bytes; a 9216-wide row is 36 Q6_K blocks of 208.
    assert!(src.contains("Rq3: tensor<i8>[9216, 1440]"), "{src}");
    assert!(src.contains("Rd3: tensor<f16>[9216, 10]"), "{src}");
    assert!(src.contains("Rq9: tensor<i8>[2560, 7488]"), "{src}");
    assert!(src.contains("Rd9: tensor<f16>[2560, 36]"), "{src}");
    // A unit is 64 hidden values, two Q8_0 blocks, each quantized and
    // stored on its own row.
    assert!(src.contains("[0 :+ 1, 0 :+ 32]"), "{src}");
    assert!(src.contains("[0 :+ 1, 32 :+ 32]"), "{src}");
    assert!(src.contains("* 2 + 1 :+ 1, 0 :+ 32]"), "{src}");
    // Two CTAs per SM with a raw decode.
    assert!(src.starts_with("@launch(256, 2)"), "{src}");
}

#[test]
fn a_wide_grid_splits_the_down_projection_along_k() {
    // 340 blocks against 40 tiles: k's 36 blocks split six ways into
    // partials, and after a second barrier the tiles add them onto x.
    let plan = qwen_4b_plan(340);
    assert_eq!(plan.barriers, 2);
    let src = &plan.source;
    assert_eq!(src.matches("= q6k_qdot_i8_t(").count(), 1, "{src}");
    assert!(!src.contains("+= q6k_qdot_i8_t("), "{src}");
    assert!(src.contains("P5: tensor<f32>[1, 15360]"), "{src}");
    assert!(src.contains("if u5 < 240"), "{src}");
    // The sum reads all six slices.
    assert!(src.contains("P5[0 :+ 1, 12800 + r5 :+ RT]"), "{src}");
    assert!(plan.scratch.iter().any(|s| s.scales == 15360), "{:?}", plan.scratch);
}

#[test]
fn the_raw_mlp_is_declined_for_a_format_without_a_fused_decode() {
    let chain = mlp_chain_raw(
        Buf(0),
        Buf(1),
        (RawBuf(0), Quant::IQ1_M),
        (RawBuf(1), Quant::Q4_K),
        (RawBuf(2), Quant::Q6_K),
        2560,
        9216,
        1e-6,
    );
    assert!(chain.is_none());
    let ragged = mlp_chain_raw(
        Buf(0),
        Buf(1),
        (RawBuf(0), Quant::Q4_K),
        (RawBuf(1), Quant::Q4_K),
        (RawBuf(2), Quant::Q6_K),
        2560,
        9216 + 32,
        1e-6,
    );
    assert!(ragged.is_none());
}

/// The 4B's delta-net projection: `attn_qkv` Q5_K and `attn_gate` Q4_K as
/// raw weights, `ssm_alpha` and `ssm_beta` as Q8_0, into the history and
/// one stacked buffer of gate operands. The projection only: with 16 key
/// heads and 32 value heads the convolution and gates stay separate
/// launches.
fn qwen_4b_project_plan() -> Plan {
    let (channels, carried) = (8192, 3 * 8192);
    let weights = [
        (ProjWeight::Raw(RawBuf(0), Quant::Q5_K), channels),
        (ProjWeight::Raw(RawBuf(1), Quant::Q4_K), 4096),
        (ProjWeight::Q8(QBuf(0)), 32),
        (ProjWeight::Q8(QBuf(1)), 32),
    ];
    let runs = [
        ProjRun { weight: 0, row_off: 0, width: channels, dst: Buf(2), dst_off: carried },
        ProjRun { weight: 1, row_off: 0, width: 4096, dst: Buf(3), dst_off: 0 },
        ProjRun { weight: 2, row_off: 0, width: 32, dst: Buf(3), dst_off: 4096 },
        ProjRun { weight: 3, row_off: 0, width: 32, dst: Buf(3), dst_off: 4128 },
    ];
    let project = FusedProject {
        x: Buf(0),
        d_model: 2560,
        gain: Buf(1),
        eps: 1e-6,
        weights: &weights,
        runs: &runs,
        mix: None,
    };
    project_chain(&project)
        .expect("a chain")
        .key(192)
        .plan()
        .expect("a well-formed chain")
        .expect("a shape the pass fuses")
}

#[test]
fn the_split_projection_mixes_raw_and_q8_weights() {
    let plan = qwen_4b_project_plan();
    assert_eq!(plan.barriers, 0);
    let src = &plan.source;
    assert_eq!(src.matches("= q5k_qdot_i8_t(").count(), 1, "{src}");
    assert_eq!(src.matches("= q4k_qdot_i8_t(").count(), 1, "{src}");
    assert_eq!(src.matches("= qdot_t(").count(), 2, "{src}");
    assert!(src.starts_with("@launch(256, 2)"), "{src}");
    // 128 and 64 units of the raw weights, one block each of the Q8_0 ones.
    assert!(src.contains("UN1 in [128]"), "{src}");
    assert!(src.contains("UN2 in [64]"), "{src}");
}

/// A raw run not divisible by 64 ends with a remainder unit. It decodes a
/// whole unit, reading the upload's row padding, and stores only its head.
#[test]
fn a_ragged_raw_run_stores_its_remainder() {
    let weights = [(ProjWeight::Raw(RawBuf(0), Quant::Q4_K), 96)];
    let runs = [ProjRun { weight: 0, row_off: 0, width: 96, dst: Buf(2), dst_off: 0 }];
    let project = FusedProject {
        x: Buf(0),
        d_model: 1024,
        gain: Buf(1),
        eps: 1e-6,
        weights: &weights,
        runs: &runs,
        mix: None,
    };
    let plan = project_chain(&project)
        .expect("a chain")
        .key(192)
        .plan()
        .expect("a well-formed chain")
        .expect("a shape the pass fuses");
    let src = &plan.source;
    assert!(src.contains("Rq4: tensor<i8>[128, 576]"), "{src}");
    assert_eq!(src.matches("= q4k_qdot_i8_t(").count(), 2, "{src}");
    assert!(src.contains(":+ 32] = v2[0 :+ 1, 0 :+ 32]"), "{src}");
}
