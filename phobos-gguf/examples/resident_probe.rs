// A decode matvec on synthetic data, on its own:
//
//   cargo run --release --features cuda -p phobos-gguf --example resident_probe
//   cargo run --release --features cuda -p phobos-gguf --example resident_probe -- 6144
//
// The model path compiles two hundred kernels before it reaches one of these,
// so asking it a question used to cost twenty minutes. This compiles the one
// kernel and launches it directly, which makes a question cost seconds.
//
// It exists to tell a slow kernel from an evicted one. The same kernel at the
// same shape runs at 170 GB/s on a card with room and 15 GB/s on one without,
// so a number measured inside a model that does not fit says nothing about
// the kernel. The optional argument is VRAM ballast in MiB, held and written
// between launches, which is what reproduces the second case.

use anyhow::Result;
use cust::prelude::*;
use phobos_kernels::{compile, cuda_ok, push_descriptor};

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
    block_bytes: 50,
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
    block_bytes: 50,
    tn: 64,
    table_bytes: 2048 * 8,
    src: "@launch(256)
@autotune(TN in [{TN}])
@aligned(N = TN)
kernel iq1s_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                           QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                           GRID: tensor<i8>[1, 16384], C: tensor<f32>[M, N]) {
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
    block_bytes: 66,
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
    block_bytes: 66,
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
    src: "@launch(256)
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

fn main() -> Result<()> {
    let _ctx = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // VRAM ballast in MiB, written between launches so the driver has to keep
    // it resident and cannot simply evict what nobody touches.
    let ballast_mib: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(0);
    let ballast = if ballast_mib > 0 {
        let held = DeviceBuffer::from_slice(&vec![0.0f32; ballast_mib * 1024 * 1024 / 4])?;
        println!("holding {ballast_mib} MiB of ballast, written between launches");
        Some(held)
    } else {
        None
    };

    println!(
        "{:>6} {:>8} {:>8} {:>5} {:>9} {:>9} {:>9}",
        "fmt", "k", "n", "TN", "ms", "GB/s", "GMAC/s"
    );
    // The model's two heaviest decode kernels, with room to breathe.
    run(&stream, &Q3K, HEAD_K, HEAD_N, Q3K.tn, ballast.as_ref())?;
    run(&stream, &IQ1S, FFN_K, FFN_N, IQ1S.tn, ballast.as_ref())?;
    run(&stream, &IQ1S, FFN_N, FFN_K, IQ1S.tn, ballast.as_ref())?;
    run(&stream, &IQ1S_I8, FFN_K, FFN_N, IQ1S_I8.tn, ballast.as_ref())?;
    run(&stream, &IQ1S_I8, FFN_N, FFN_K, IQ1S_I8.tn, ballast.as_ref())?;

    // The dp4a path has to agree with the float one it replaces. Both read
    // the same weights and the same activation values, so the only difference
    // left is the order the sums are accumulated in.
    let want = run(&stream, &IQ1S, FFN_K, FFN_N, IQ1S.tn, None)?;
    let got = run(&stream, &IQ1S_I8, FFN_K, FFN_N, IQ1S_I8.tn, None)?;
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = want
        .iter()
        .zip(&got)
        .map(|(w, g)| (w - g).abs() / scale)
        .fold(0.0f32, f32::max);
    println!("
  iq1s  dp4a against float, worst relative error {worst:.3e}");
    anyhow::ensure!(worst < 1e-3, "the iq1s dp4a path disagrees with the float path");

    let want = run(&stream, &IQ2XXS, FFN_K, FFN_N, IQ2XXS.tn, None)?;
    let got = run(&stream, &IQ2XXS_I8, FFN_K, FFN_N, IQ2XXS_I8.tn, None)?;
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = want
        .iter()
        .zip(&got)
        .map(|(w, g)| (w - g).abs() / scale)
        .fold(0.0f32, f32::max);
    println!("  iq2xxs dp4a against float, worst relative error {worst:.3e}");
    anyhow::ensure!(worst < 1e-3, "the iq2xxs dp4a path disagrees with the float path");

    let want = run(&stream, &IQ3XXS, FFN_K, FFN_N, IQ3XXS.tn, None)?;
    let got = run(&stream, &IQ3XXS_I8, FFN_K, FFN_N, IQ3XXS_I8.tn, None)?;
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
        run(&stream, &Q3K, HEAD_K, n, Q3K.tn, None)?;
    }
    println!();
    // The CTA size sets both occupancy and, since it is derived, how many
    // columns a warp owns. 256 threads leaves only 24 of 32 warps resident.
    for threads in [256u32, 512, 1024] {
        let got = run_at(&stream, &IQ1S_I8, FFN_K, FFN_N, 64, threads, None)?;
        let _ = got;
        println!("      ^ @launch({threads})");
    }
    println!();
    // Columns a CTA covers: every column re-reads the activation vector, so a
    // wider tile shares more of it inside one L1.
    for tn in [8, 16, 32] {
        run(&stream, &IQ1S, FFN_K, FFN_N, tn, None)?;
    }
    println!();
    // The i8 kernel gives a warp two columns, so it needs twice the tile to
    // keep a 256-thread CTA busy.
    for tn in [8, 16, 32, 64] {
        run(&stream, &IQ1S_I8, FFN_K, FFN_N, tn, None)?;
    }
    Ok(())
}

fn run(
    stream: &Stream,
    probe: &Probe,
    k: usize,
    n: usize,
    tn: usize,
    ballast: Option<&DeviceBuffer<f32>>,
) -> Result<Vec<f32>> {
    run_at(stream, probe, k, n, tn, 256, ballast)
}

#[allow(clippy::too_many_arguments)]
fn run_at(
    stream: &Stream,
    probe: &Probe,
    k: usize,
    n: usize,
    tn: usize,
    threads: u32,
    ballast: Option<&DeviceBuffer<f32>>,
) -> Result<Vec<f32>> {
    let nb = k / 256;
    let rb = nb * probe.block_bytes;
    let src = probe.src.replace("{TN}", &tn.to_string()).replace("@launch(256)", &format!("@launch({threads})"));
    let module = compile(&src, &[("TN", tn)], probe.name)?;
    let function = module.get_function(probe.kernel)?.to_raw();

    // Values do not reach the timing: every lane decodes the same count of
    // weights whatever the bytes say. They vary only so that a constant page
    // cannot stand in for the traffic.
    let bytes: Vec<i8> = (0..n * rb).map(|i| (i.wrapping_mul(2654435761) >> 13) as i8).collect();
    let d: Vec<u16> = vec![0x3400u16; n * nb];
    // The f32 activation is exactly the dequantization of the int8 one, so
    // the float and dp4a kernels contract identical values and their outputs
    // are comparable rather than merely both plausible.
    let aq: Vec<i8> = (0..k).map(|i| ((i % 17) as i32 - 8) as i8).collect();
    let asc: Vec<f32> = vec![ACT_SCALE; k / 32];
    let a: Vec<f32> = aq.iter().map(|&q| f32::from(q) * ACT_SCALE).collect();

    let qb = DeviceBuffer::from_slice(&bytes)?;
    let db = DeviceBuffer::from_slice(&d)?;
    let ab = DeviceBuffer::from_slice(&a)?;
    let aqb = DeviceBuffer::from_slice(&aq)?;
    let ascb = DeviceBuffer::from_slice(&asc)?;
    let cb = DeviceBuffer::from_slice(&vec![0.0f32; n])?;
    let table = DeviceBuffer::from_slice(&vec![0i8; probe.table_bytes.max(1)])?;
    // Signs are +/-1; a table of zeroes would make every weight vanish and
    // hide a disagreement between the two paths.
    // The float path multiplies by +/-1; the dp4a path masks with 0/-1. Both
    // spell the same signs, so the two kernels stay comparable.
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
    println!(
        "{:>6} {k:>8} {n:>8} {tn:>5} {millis:>9.3} {:>9.1} {:>9.1}",
        probe.name,
        (n * rb) as f64 / (millis / 1000.0) / 1e9,
        (n * k) as f64 / (millis / 1000.0) / 1e9,
    );
    Ok(got)
}

#[allow(clippy::too_many_arguments)]
fn time(
    stream: &Stream,
    function: cust::sys::CUfunction,
    blocks: u32,
    threads: u32,
    slots: &mut [u64],
    ballast: Option<&DeviceBuffer<f32>>,
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
                    blocks,
                    1,
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

    // Events rather than a host timer: with a ballast held, the number wanted
    // is what the kernel costs, not what writing the ballast costs beside it.
    use cust::event::{Event, EventFlags};
    let (begin, end) = (Event::new(EventFlags::DEFAULT)?, Event::new(EventFlags::DEFAULT)?);
    let mut total = 0.0f32;
    for _ in 0..REPS {
        if let Some(b) = ballast {
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
