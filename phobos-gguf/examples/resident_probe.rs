// A decode matvec on synthetic data, on its own:
//
//   cargo run --release --features cuda -p phobos-gguf --example resident_probe
//   cargo run --release --features cuda -p phobos-gguf --example resident_probe -- 6144
//
// Compiles and launches one kernel directly, without a model, to tell a slow
// kernel from an evicted one. The same kernel at the same shape reads much
// slower once VRAM is tight. The optional argument is VRAM ballast in MiB,
// held and written between launches, to reproduce that pressure.

use anyhow::Result;
use cust::prelude::*;
use phobos_kernels::{compile, cuda_ok, push_descriptor};

thread_local! {
    /// Free VRAM before the ballast, so its real cost can be read off.
    static START: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Set while the warm-up runs, so the rows it runs print nothing.
    static WARMING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// One format's decode matvec: the production source, its device block
/// stride, its output tile, and how wide a lookup table it wants.
struct Probe {
    name: &'static str,
    kernel: &'static str,
    src: &'static str,
    block_bytes: usize,
    tn: usize,
    table_bytes: usize,
    /// The activation arrives already quantized, with a scale every 32.
    int8_act: bool,
    /// A second table, for the formats whose decode carries signs.
    signs_bytes: usize,
}

const Q3K: Probe = Probe {
    name: "q3k",
    kernel: "q3k_qdot_matvec",
    int8_act: false,
    signs_bytes: 0,
    block_bytes: 112,
    tn: 32,
    table_bytes: 0,
    src: "\
@launch(256)
@autotune(TN in [{TN}])
@aligned(N = TN)
kernel q3k_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                       D: tensor<f16>[N, NB], C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = q3k_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :])
}
",
};

const IQ1S: Probe = Probe {
    name: "iq1s",
    kernel: "iq1s_qdot_matvec",
    int8_act: false,
    signs_bytes: 0,
    block_bytes: 48,
    tn: 8,
    table_bytes: 2048 * 8,
    src: "\
@launch(256)
@autotune(TN in [{TN}])
@aligned(N = TN)
kernel iq1s_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                        D: tensor<f16>[N, NB], GRID: tensor<i8>[1, 16384],
                        C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq1s_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                         D[pn * TN :+ TN, :], GRID[0 :+ 1, :])
}
",
};

const IQ1S_I8: Probe = Probe {
    name: "iq1s-i8",
    kernel: "iq1s_qdot_i8_matvec",
    int8_act: true,
    signs_bytes: 0,
    block_bytes: 48,
    tn: 64,
    table_bytes: 2048 * 4,
    src: "@launch(256, 4)
@autotune(TN in [{TN}])
@aligned(N = TN)
kernel iq1s_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                           QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                           GRID: tensor<i8>[1, 8192], C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq1s_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                            QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                            GRID[0 :+ 1, :])
}
",
};

const IQ2XXS: Probe = Probe {
    name: "iq2xxs",
    kernel: "iq2xxs_qdot_matvec",
    int8_act: false,
    block_bytes: 64,
    tn: 8,
    table_bytes: 256 * 8,
    signs_bytes: 128 * 8,
    src: "@launch(256)
@autotune(TN in [{TN}])
@aligned(N = TN)
kernel iq2xxs_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                          D: tensor<f16>[N, NB], GRID: tensor<i8>[1, 2048],
                          SIGNS: tensor<i8>[1, 1024], C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq2xxs_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                           D[pn * TN :+ TN, :], GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}
",
};

const IQ2XXS_I8: Probe = Probe {
    name: "iq2xxs-i8",
    kernel: "iq2xxs_qdot_i8_matvec",
    int8_act: true,
    block_bytes: 64,
    tn: 64,
    table_bytes: 256 * 8,
    signs_bytes: 128 * 8,
    src: "@launch(256, 4)
@autotune(TN in [{TN}])
@aligned(N = TN)
kernel iq2xxs_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                             QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                             GRID: tensor<i8>[1, 2048], SIGNS: tensor<i8>[1, 1024],
                             C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq2xxs_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                              QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                              GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}
",
};

const IQ3XXS: Probe = Probe {
    name: "iq3xxs",
    kernel: "iq3xxs_qdot_matvec",
    int8_act: false,
    block_bytes: 96,
    tn: 8,
    table_bytes: 256 * 4,
    signs_bytes: 128 * 8,
    src: "@launch(256, 4)
@autotune(TN in [{TN}])
@aligned(N = TN)
kernel iq3xxs_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                          D: tensor<f16>[N, NB], GRID: tensor<i8>[1, 1024],
                          SIGNS: tensor<i8>[1, 1024], C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq3xxs_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                           D[pn * TN :+ TN, :], GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}
",
};

const IQ3XXS_I8: Probe = Probe {
    name: "iq3xxs-i8",
    kernel: "iq3xxs_qdot_i8_matvec",
    int8_act: true,
    block_bytes: 96,
    tn: 64,
    table_bytes: 256 * 4,
    signs_bytes: 128 * 8,
    src: "@launch(256, 4)
@autotune(TN in [{TN}])
@aligned(N = TN)
kernel iq3xxs_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                             QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                             GRID: tensor<i8>[1, 1024], SIGNS: tensor<i8>[1, 1024],
                             C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq3xxs_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                              QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                              GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}
",
};

const REPS: usize = 10;

/// One scale for the whole synthetic activation, so the two paths agree.
const ACT_SCALE: f32 = 0.01;

/// The 27B's output head, and a representative IQ1_S projection.
const HEAD_K: usize = 5120;
const HEAD_N: usize = 248320;
const FFN_K: usize = 5120;
const FFN_N: usize = 17408;

/// A K-quant decode matvec, `<fmt>_qdot_i8_t` with no tables, as
/// `kernels/kquant.rs` builds it, at each format's launch bound.
const fn kquant_probe(name: &'static str, kernel: &'static str, block_bytes: usize, src: &'static str) -> Probe {
    Probe {
        name,
        kernel,
        int8_act: true,
        signs_bytes: 0,
        block_bytes,
        tn: 64,
        table_bytes: 0,
        src,
    }
}

const Q4K_I8: Probe = kquant_probe(
    "q4k-i8",
    "q4k_qdot_i8_matvec",
    144,
    "@launch(256, 4)
@autotune(TN in [{TN}])
@aligned(N = TN)
kernel q4k_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                          QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                          C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = q4k_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                           QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :])
}
",
);

const Q5K_I8: Probe = kquant_probe(
    "q5k-i8",
    "q5k_qdot_i8_matvec",
    176,
    "@launch(256, 3)
@autotune(TN in [{TN}])
@aligned(N = TN)
kernel q5k_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                          QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                          C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = q5k_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                           QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :])
}
",
);

const Q6K_I8: Probe = kquant_probe(
    "q6k-i8",
    "q6k_qdot_i8_matvec",
    208,
    "@launch(256, 3)
@autotune(TN in [{TN}])
@aligned(N = TN)
kernel q6k_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                          QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                          C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = q6k_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                           QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :])
}
",
);

/// The 4B Q4_K_M's decode shapes, `k` by `n`, each with the format the
/// model stores it in.
const KQUANT_SHAPES: [(&Probe, usize, usize, &str); 6] = [
    (&Q4K_I8, 2560, 9216, "ffn gate/up"),
    (&Q4K_I8, 9216, 2560, "ffn down (half)"),
    (&Q6K_I8, 9216, 2560, "ffn down (half)"),
    (&Q5K_I8, 2560, 8192, "attn_qkv"),
    (&Q5K_I8, 4096, 2560, "ssm_out"),
    (&Q6K_I8, 2560, 248320, "the head"),
];

/// A K-quant staged projection, `<fmt>_qgemm_t` with no tables, as
/// `kernels/qgemm.rs` builds it, at its shipped launch bound.
fn kquant_gemm_src(name: &str) -> String {
    format!(
        "@launch(256, 2)
@autotune(TM in [128], TN in [64])
@aligned(M = TM, N = TN, K = 256)
kernel {name}_qgemm(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                  QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                  C: tensor<f32>[M, N]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm * TM :+ TM, pn * TN :+ TN] = {name}_qgemm_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :])
}}
"
    )
}

/// Two seconds of `work` before anything is timed: an idle card sits at its
/// lowest clock, and a short probe would measure the ramp.
fn warm(stream: &Stream, mut work: impl FnMut() -> Result<()>) -> Result<()> {
    let start = std::time::Instant::now();
    let mut n = 0;
    WARMING.with(|w| w.set(true));
    while start.elapsed().as_secs_f32() < 2.0 {
        work()?;
        n += 1;
    }
    WARMING.with(|w| w.set(false));
    stream.synchronize()?;
    println!("warmed the card with {n} runs of the first kernel");
    Ok(())
}

/// Times one staged projection at `m` rows. Prints TOPS from the MACs next
/// to GB/s from the weight bytes.
fn run_gemm(stream: &Stream, name: &str, block_bytes: usize, m: usize, k: usize, n: usize) -> Result<f64> {
    let nb = k / 256;
    let rb = nb * block_bytes;
    let module = compile(&kquant_gemm_src(name), &[("TM", 128), ("TN", 64)], &format!("{name}_gemm"))?;
    let function = module.get_function(format!("{name}_qgemm"))?.to_raw();
    let bytes: Vec<i8> = (0..n * rb).map(|i| (i.wrapping_mul(2654435761) >> 13) as i8).collect();
    let d: Vec<u16> = vec![0x3400u16; n * nb];
    let aq: Vec<i8> = (0..m * k).map(|i| ((i % 17) as i32 - 8) as i8).collect();
    let asc: Vec<f32> = vec![ACT_SCALE; m * k / 32];
    let qb = DeviceBuffer::from_slice(&bytes)?;
    let db = DeviceBuffer::from_slice(&d)?;
    let aqb = DeviceBuffer::from_slice(&aq)?;
    let ascb = DeviceBuffer::from_slice(&asc)?;
    let cb = DeviceBuffer::from_slice(&vec![0.0f32; m * n])?;
    let mut slots = Vec::new();
    push_descriptor(&mut slots, aqb.as_device_ptr().as_raw(), [m as i64, k as i64]);
    push_descriptor(&mut slots, ascb.as_device_ptr().as_raw(), [m as i64, (k / 32) as i64]);
    push_descriptor(&mut slots, qb.as_device_ptr().as_raw(), [n as i64, rb as i64]);
    push_descriptor(&mut slots, db.as_device_ptr().as_raw(), [n as i64, nb as i64]);
    push_descriptor(&mut slots, cb.as_device_ptr().as_raw(), [m as i64, n as i64]);
    let grid = ((m / 128) as u32, (n / 64) as u32);
    let millis = time_grid(stream, function, grid, 256, &mut slots, &[])?;
    if !WARMING.with(|w| w.get()) {
        println!(
            "{:>6} {m:>5} {k:>8} {n:>8} {millis:>9.3} {:>7.1} {:>7.1}",
            name,
            2.0 * (m * n * k) as f64 / (millis / 1000.0) / 1e12,
            (n * rb) as f64 / (millis / 1000.0) / 1e9,
        );
    }
    Ok(millis)
}

/// `resident_probe --prompt`: the three K-quant staged projections at the
/// 4B's shapes and at the 27B's FFN shape. Comparing the two `k`s shows what
/// a short k loop costs. Comparing formats at one shape shows what the
/// decode and the minimum term cost.
fn run_prompt(stream: &Stream) -> Result<()> {
    warm(stream, || run_gemm(stream, "q4k", 144, 512, 5120, 17408).map(|_| ()))?;
    println!("{:>6} {:>5} {:>8} {:>8} {:>9} {:>7} {:>7}", "fmt", "m", "k", "n", "ms", "TOPS", "GB/s");
    for (name, block_bytes, m, k, n) in [
        ("q4k", 144usize, 128usize, 2560usize, 9216usize),
        ("q4k", 144, 512, 2560, 9216),
        ("q4k", 144, 512, 9216, 2560),
        ("q4k", 144, 128, 5120, 17408),
        ("q4k", 144, 512, 5120, 17408),
        ("q5k", 176, 512, 2560, 8192),
        ("q5k", 176, 128, 5120, 17408),
        ("q6k", 208, 512, 9216, 2560),
        ("q6k", 208, 128, 5120, 17408),
    ] {
        run_gemm(stream, name, block_bytes, m, k, n)?;
    }
    Ok(())
}

/// `resident_probe --kquant [BALLAST_MIB [CHUNKS]]`: the three K-quant
/// decode matvecs at the 4B's shapes, and nothing else.
fn run_kquant(stream: &Stream, ballast: &[DeviceBuffer<f32>]) -> Result<()> {
    warm(stream, || run(stream, &Q6K_I8, 2560, 248320, Q6K_I8.tn, &[]).map(|_| ()))?;
    println!(
        "{:>6} {:>8} {:>8} {:>5} {:>9} {:>9} {:>9}",
        "fmt", "k", "n", "TN", "ms", "GB/s", "GMAC/s"
    );
    for (probe, k, n, role) in KQUANT_SHAPES {
        run(stream, probe, k, n, probe.tn, ballast)?;
        println!("      ^ {role}");
    }
    if !ballast.is_empty() {
        return Ok(());
    }
    // The narrow-n shapes at thinner CTAs. A thinner CTA gives each warp
    // fewer columns and puts more CTAs on the card.
    println!();
    for (probe, k, n) in [(&Q4K_I8, 9216usize, 2560usize), (&Q5K_I8, 4096, 2560), (&Q4K_I8, 2560, 1024)] {
        for (tn, threads) in [(64usize, 256u32), (32, 128), (16, 64)] {
            let _ = run_at(stream, probe, k, n, tn, threads, &[])?;
            println!("      ^ TN {tn} @launch({threads}), {} CTAs", n / tn);
        }
        println!();
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ctx = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    START.with(|s| s.set(cust::memory::mem_get_info().map_or(0, |(free, _)| free)));

    // VRAM ballast in MiB, written between launches so the driver has to keep
    // it resident and cannot simply evict what nobody touches.
    let positional: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with("--")).collect();
    let ballast_mib: usize = positional.first().and_then(|a| a.parse().ok()).unwrap_or(0);
    // How many allocations the ballast is split across. One matches
    // llama.cpp's few large buffers; several hundred matches phobos's one
    // `cuMemAlloc` per tensor. WDDM manages residency per allocation, so
    // equal totals are not equal pressure.
    let chunks: usize = positional.get(1).and_then(|a| a.parse().ok()).unwrap_or(1);
    let mut ballast = Vec::new();
    if ballast_mib > 0 {
        let per = ballast_mib * 1024 * 1024 / 4 / chunks;
        for _ in 0..chunks {
            ballast.push(DeviceBuffer::from_slice(&vec![0.0f32; per])?);
        }
        println!(
            "holding {ballast_mib} MiB of ballast across {chunks} allocation(s),              written between launches"
        );
    }
    // What the ballast really took off the card. Each `cuMemAlloc` rounds up
    // to the driver's page, so many small ones cost more than the bytes asked
    // for, and the process cannot see that otherwise.
    if let Ok((free, total)) = cust::memory::mem_get_info() {
        let mib = |b: usize| b as f64 / (1 << 20) as f64;
        println!(
            "free {:.0} of {:.0} MiB; the ballast cost {:.0} MiB for {ballast_mib} asked",
            mib(free),
            mib(total),
            mib(START.with(|s| s.get()).saturating_sub(free)),
        );
    }

    if std::env::args().any(|a| a == "--kquant") {
        return run_kquant(&stream, &ballast);
    }
    if std::env::args().any(|a| a == "--prompt") {
        return run_prompt(&stream);
    }
    println!(
        "{:>6} {:>8} {:>8} {:>5} {:>9} {:>9} {:>9}",
        "fmt", "k", "n", "TN", "ms", "GB/s", "GMAC/s"
    );
    // The 27B's two heaviest decode kernels.
    run(&stream, &Q3K, HEAD_K, HEAD_N, Q3K.tn, &ballast)?;
    run(&stream, &IQ1S, FFN_K, FFN_N, IQ1S.tn, &ballast)?;
    run(&stream, &IQ1S, FFN_N, FFN_K, IQ1S.tn, &ballast)?;
    run(&stream, &IQ1S_I8, FFN_K, FFN_N, IQ1S_I8.tn, &ballast)?;
    run(&stream, &IQ1S_I8, FFN_N, FFN_K, IQ1S_I8.tn, &ballast)?;

    // The dp4a path must agree with the float path it replaces. Both read the
    // same weights and activation values, so only accumulation order differs.
    let want = run(&stream, &IQ1S, FFN_K, FFN_N, IQ1S.tn, &[])?;
    let got = run(&stream, &IQ1S_I8, FFN_K, FFN_N, IQ1S_I8.tn, &[])?;
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = want
        .iter()
        .zip(&got)
        .map(|(w, g)| (w - g).abs() / scale)
        .fold(0.0f32, f32::max);
    println!("
  iq1s  dp4a against float, worst relative error {worst:.3e}");
    anyhow::ensure!(worst < 1e-3, "the iq1s dp4a path disagrees with the float path");

    let want = run(&stream, &IQ2XXS, FFN_K, FFN_N, IQ2XXS.tn, &[])?;
    let got = run(&stream, &IQ2XXS_I8, FFN_K, FFN_N, IQ2XXS_I8.tn, &[])?;
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = want
        .iter()
        .zip(&got)
        .map(|(w, g)| (w - g).abs() / scale)
        .fold(0.0f32, f32::max);
    println!("  iq2xxs dp4a against float, worst relative error {worst:.3e}");
    anyhow::ensure!(worst < 1e-3, "the iq2xxs dp4a path disagrees with the float path");

    let want = run(&stream, &IQ3XXS, FFN_K, FFN_N, IQ3XXS.tn, &[])?;
    let got = run(&stream, &IQ3XXS_I8, FFN_K, FFN_N, IQ3XXS_I8.tn, &[])?;
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = want.iter().zip(&got).map(|(w, g)| (w - g).abs() / scale).fold(0.0f32, f32::max);
    println!("  iq3xxs dp4a against float, worst relative error {worst:.3e}");
    anyhow::ensure!(worst < 1e-3, "the iq3xxs dp4a path disagrees with the float path");
    if ballast_mib > 0 {
        return Ok(());
    }
    println!();
    // Linear in n, or the cost is not in the contraction.
    for n in [HEAD_N / 8, HEAD_N / 4, HEAD_N / 2, HEAD_N] {
        run(&stream, &Q3K, HEAD_K, n, Q3K.tn, &[])?;
    }
    println!();
    // The CTA size sets occupancy and also how many columns a warp owns.
    for threads in [256u32, 512, 1024] {
        let got = run_at(&stream, &IQ1S_I8, FFN_K, FFN_N, 64, threads, &[])?;
        let _ = got;
        println!("      ^ @launch({threads})");
    }
    println!();
    // Columns a CTA covers: every column re-reads the activation vector, so a
    // wider tile shares more of it inside one L1.
    for tn in [8, 16, 32] {
        run(&stream, &IQ1S, FFN_K, FFN_N, tn, &[])?;
    }
    println!();
    // The i8 kernel gives a warp two columns, so it needs twice the tile to
    // keep a 256-thread CTA busy. Past 64 the grid runs short of blocks to
    // fill the card.
    for tn in [8, 16, 32, 64, 128, 256] {
        run(&stream, &IQ1S_I8, FFN_K, FFN_N, tn, &[])?;
    }
    println!();
    // Tile and CTA together, at both model shapes and for IQ2_XXS too. A warp
    // owns `tn * 32 / threads` columns, and that ratio is what is swept.
    for probe in [&IQ1S_I8, &IQ2XXS_I8] {
        for (k, n) in [(FFN_K, FFN_N), (FFN_N, FFN_K)] {
            for tn in [64usize, 128, 256] {
                for threads in [256u32, 512, 1024] {
                    // The tile must fill the CTA a whole number of times.
                    if threads as usize > tn * 32 || !(tn * 32).is_multiple_of(threads as usize) {
                        continue;
                    }
                    let _ = run_at(&stream, probe, k, n, tn, threads, &[])?;
                    println!("      ^ TN {tn} @launch({threads})");
                }
            }
        }
        println!();
    }
    Ok(())
}

fn run(
    stream: &Stream,
    probe: &Probe,
    k: usize,
    n: usize,
    tn: usize,
    ballast: &[DeviceBuffer<f32>],
) -> Result<Vec<f32>> {
    run_at(stream, probe, k, n, tn, 256, ballast)
}

fn run_at(
    stream: &Stream,
    probe: &Probe,
    k: usize,
    n: usize,
    tn: usize,
    threads: u32,
    ballast: &[DeviceBuffer<f32>],
) -> Result<Vec<f32>> {
    let nb = k / 256;
    let rb = nb * probe.block_bytes;
    let src = probe
        .src
        .replace("{TN}", &tn.to_string())
        .replace("@launch(256)", &format!("@launch({threads})"))
        .replace("@launch(256, 4)", &format!("@launch({threads}, {})", 1024 / threads))
        .replace("@launch(256, 3)", &format!("@launch({threads}, {})", (1024 / threads * 3 / 4).max(1)));
    let module = compile(&src, &[("TN", tn)], probe.name)?;
    let function = module.get_function(probe.kernel)?.to_raw();

    // Values do not affect timing, since every lane decodes the same number
    // of weights. They vary so that a constant page cannot stand in for real
    // traffic.
    let bytes: Vec<i8> = (0..n * rb).map(|i| (i.wrapping_mul(2654435761) >> 13) as i8).collect();
    let d: Vec<u16> = vec![0x3400u16; n * nb];
    // The f32 activation is exactly the dequantized int8 one, so the float
    // and dp4a kernels contract identical values and their outputs compare.
    let aq: Vec<i8> = (0..k).map(|i| ((i % 17) as i32 - 8) as i8).collect();
    let asc: Vec<f32> = vec![ACT_SCALE; k / 32];
    let a: Vec<f32> = aq.iter().map(|&q| f32::from(q) * ACT_SCALE).collect();

    let qb = DeviceBuffer::from_slice(&bytes)?;
    let db = DeviceBuffer::from_slice(&d)?;
    let ab = DeviceBuffer::from_slice(&a)?;
    let aqb = DeviceBuffer::from_slice(&aq)?;
    let ascb = DeviceBuffer::from_slice(&asc)?;
    let cb = DeviceBuffer::from_slice(&vec![0.0f32; n])?;
    // The tables both paths read, built from one random grid so they agree.
    // IQ1_S gets a ternary grid: bytes `g` in -1..1 for the float path, or
    // nibbles `g + 1`, eight per word, for the int8 path. The other formats
    // get nonzero magnitudes like a real grid's. Zero is avoided because the
    // int8 path negates a masked byte as `(m ^ 0xff) + 1`.
    let ternary = |i: usize| ((i.wrapping_mul(2654435761) >> 7) % 3) as i8 - 1;
    let table: Vec<i8> = if probe.name.starts_with("iq1s") {
        if probe.int8_act {
            (0..probe.table_bytes / 4)
                .flat_map(|e| {
                    let word = (0..8).fold(0u32, |acc, j| acc | (u32::from((ternary(e * 8 + j) + 1) as u8) << (4 * j)));
                    word.to_le_bytes().map(|b| b as i8)
                })
                .collect()
        } else {
            (0..probe.table_bytes).map(ternary).collect()
        }
    } else {
        (0..probe.table_bytes.max(1)).map(|i| (i % 7 + 1) as i8).collect()
    };
    let table = DeviceBuffer::from_slice(&table)?;
    // Signs are never zero, so a disagreement cannot hide behind a vanished
    // weight. The float path multiplies by +/-1 and the dp4a path masks with
    // 0/-1, which encode the same signs.
    let signs: Vec<i8> = (0..probe.signs_bytes.max(1))
        .map(|i| match (i % 3 == 0, probe.int8_act) {
            (true, _) => -1,
            (false, true) => 0,
            (false, false) => 1,
        })
        .collect();
    let signs = DeviceBuffer::from_slice(&signs)?;

    let mut slots = Vec::new();
    if probe.int8_act {
        push_descriptor(&mut slots, aqb.as_device_ptr().as_raw(), [1, k as i64]);
        push_descriptor(&mut slots, ascb.as_device_ptr().as_raw(), [1, (k / 32) as i64]);
    } else {
        push_descriptor(&mut slots, ab.as_device_ptr().as_raw(), [1, k as i64]);
    }
    push_descriptor(&mut slots, qb.as_device_ptr().as_raw(), [n as i64, rb as i64]);
    push_descriptor(&mut slots, db.as_device_ptr().as_raw(), [n as i64, nb as i64]);
    if probe.table_bytes > 0 {
        push_descriptor(&mut slots, table.as_device_ptr().as_raw(), [1, probe.table_bytes as i64]);
    }
    if probe.signs_bytes > 0 {
        push_descriptor(&mut slots, signs.as_device_ptr().as_raw(), [1, probe.signs_bytes as i64]);
    }
    push_descriptor(&mut slots, cb.as_device_ptr().as_raw(), [1, n as i64]);

    let millis = time(stream, function, n.div_ceil(tn) as u32, threads, &mut slots, ballast)?;
    let mut got = vec![0.0f32; n];
    cb.copy_to(&mut got)?;
    if !WARMING.with(|w| w.get()) {
        println!(
            "{:>6} {k:>8} {n:>8} {tn:>5} {millis:>9.3} {:>9.1} {:>9.1}",
            probe.name,
            (n * rb) as f64 / (millis / 1000.0) / 1e9,
            (n * k) as f64 / (millis / 1000.0) / 1e9,
        );
    }
    Ok(got)
}

fn time(
    stream: &Stream,
    function: cust::sys::CUfunction,
    blocks: u32,
    threads: u32,
    slots: &mut [u64],
    ballast: &[DeviceBuffer<f32>],
) -> Result<f64> {
    time_grid(stream, function, (blocks, 1), threads, slots, ballast)
}

fn time_grid(
    stream: &Stream,
    function: cust::sys::CUfunction,
    blocks: (u32, u32),
    threads: u32,
    slots: &mut [u64],
    ballast: &[DeviceBuffer<f32>],
) -> Result<f64> {
    let mut params: Vec<*mut std::ffi::c_void> =
        slots.iter_mut().map(|s| (s as *mut u64).cast()).collect();
    // SAFETY: the function comes from a module held by the caller, and the
    // descriptor slots outlive the launch, which is synchronized below.
    let mut launch = || -> Result<()> {
        unsafe {
            cuda_ok(
                cust::sys::cuLaunchKernel(
                    function,
                    blocks.0,
                    blocks.1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    stream.as_inner(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "probe launch",
            )?;
        }
        Ok(())
    };
    launch()?;
    stream.synchronize()?;

    // Timed by events, not a host clock, so the ballast writes between
    // launches are not counted.
    use cust::event::{Event, EventFlags};
    let (begin, end) = (Event::new(EventFlags::DEFAULT)?, Event::new(EventFlags::DEFAULT)?);
    let mut total = 0.0f32;
    for _ in 0..REPS {
        for b in ballast {
            // SAFETY: the buffer is live and the stream is synchronized below.
            unsafe {
                cuda_ok(
                    cust::sys::cuMemsetD8_v2(b.as_device_ptr().as_raw(), 1, b.len() * 4),
                    "ballast touch",
                )?;
            }
        }
        begin.record(stream)?;
        launch()?;
        end.record(stream)?;
        end.synchronize()?;
        total += end.elapsed_time_f32(&begin)?;
    }
    Ok(f64::from(total) / REPS as f64)
}
