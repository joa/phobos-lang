// A quantized format's prompt projection, timed through the device backend:
//
//   cargo run --release --features cuda -p phobos-gguf --example qgemm_time -- IQ2_XXS
//
// Random blocks at a 512-token prompt's shapes, each projection run `REPS`
// times after a warm-up, with the integer tensor-core path on. Build it at
// two commits and interleave the runs to compare their kernels; a figure is
// only comparable to one from the same session on the same card. Each
// result is also compared with the weight dequantized on the host and
// multiplied in f32 on the device, and a 128-row projection, whose grid is
// small enough to split `k`, is checked too.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use phobos_base::half::f32_to_f16;
use phobos_gguf::backend::{Backend, device, read_vec};
use phobos_gguf::quant::{Packed, Quant};

const REPS: usize = 20;
const ROWS: usize = 512;

/// `(rows, k, n)` of the 4B's projections: attention in, FFN gate/up, FFN
/// down, then FFN down at a prompt short enough to split.
const SHAPES: [(usize, usize, usize); 4] = [(ROWS, 2560, 4096), (ROWS, 2560, 9216), (ROWS, 9216, 2560), (128, 9216, 2560)];

/// Largest difference from the f32 path, relative to its largest value,
/// the fused path's activation quantization allows.
const SPREAD_MAX: f64 = 2e-2;

/// Where a format's block keeps its f16 scales, which random bytes would
/// make infinite or NaN. IQ1_M has none and is handled apart.
fn scale_offsets(quant: Quant) -> &'static [usize] {
    match quant {
        Quant::Q4_K | Quant::Q5_K => &[0, 2],
        Quant::Q6_K => &[208],
        Quant::PTQ1_0 => &[26, 54],
        _ => &[0],
    }
}

fn main() -> Result<()> {
    let name = std::env::args().nth(1).context("usage: qgemm_time FORMAT")?;
    let Some(quant) = Quant::ALL.iter().copied().find(|q| q.spec().name == name) else {
        bail!("no format {name}");
    };
    let block_bytes = quant.spec().block_bytes;
    let gpu = &device::DeviceBackend::new(&[quant])?;
    gpu.set_raw_qmma(true);
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 8388608.0 - 1.0
    };
    let mut total_micros = 0.0;
    let mut failed = false;
    for (rows, k, n) in SHAPES {
        // Random payload bytes with small finite f16 scales. The values do
        // not change the time.
        let mut blocks = vec![0u8; n * (k / quant.spec().block) * block_bytes];
        for chunk in blocks.chunks_exact_mut(block_bytes) {
            for byte in chunk.iter_mut() {
                *byte = (((next() + 1.0) * 127.5) as i32).clamp(0, 255) as u8;
            }
            let d = f32_to_f16(0.02 * (next().abs() + 0.1));
            if quant == Quant::IQ1_M {
                // No f16 field: `d` is the top nibbles of the four scale
                // halfwords that end the block.
                for i in 0..4 {
                    let at = block_bytes - 8 + 2 * i + 1;
                    chunk[at] = (chunk[at] & 0x0F) | (((d >> (4 * i)) & 15) as u8) << 4;
                }
            } else {
                for &at in scale_offsets(quant) {
                    chunk[at..at + 2].copy_from_slice(&d.to_le_bytes());
                }
            }
        }
        let packed = Packed::new(quant, blocks, k, n)?;
        // The reference: the weight dequantized here, `[k, n]`.
        let mut wt = vec![0.0f32; k * n];
        let mut row = vec![0.0f32; k];
        for j in 0..n {
            packed.row_into(j, &mut row)?;
            for (i, &v) in row.iter().enumerate() {
                wt[i * n + j] = v;
            }
        }
        let a: Vec<f32> = (0..rows * k).map(|_| next()).collect();
        let ab = gpu.upload(&a)?;
        let out = gpu.alloc(rows * n)?;
        let wd = gpu.upload(&wt)?;
        gpu.matmul(ab, rows, k, wd, n, out)?;
        let dense = read_vec(gpu, out, rows * n)?;
        // Q5_0 is no raw format: it uploads as a quantized weight and
        // projects through `q50_qmma`.
        let project: Box<dyn Fn() -> Result<()>> = if quant == Quant::Q5_0 {
            let wq = gpu.constant_quant(&format!("time{rows}x{k}x{n}"), &packed)?;
            Box::new(move || gpu.matmul_quant(ab, rows, k, wq, n, out))
        } else {
            let wb = gpu.constant_raw(&format!("time{rows}x{k}x{n}"), &packed)?;
            Box::new(move || gpu.matmul_raw(ab, rows, k, wb, n, out))
        };
        project()?;
        let fused = read_vec(gpu, out, rows * n)?;
        let scale = dense.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()));
        let spread = fused.iter().zip(&dense).fold(0.0f64, |m, (&f, &d)| m.max((f as f64 - d as f64).abs())) / scale;
        let verdict = if spread <= SPREAD_MAX { "ok" } else { "FAIL" };
        failed |= spread.is_nan() || spread > SPREAD_MAX;
        let start = Instant::now();
        for _ in 0..REPS {
            project()?;
        }
        read_vec(gpu, out, 1)?;
        let micros = start.elapsed().as_secs_f64() * 1e6 / REPS as f64;
        if rows == ROWS {
            total_micros += micros;
        }
        println!("{verdict:4} {name} [{rows} x {k} x {n}]  {micros:9.1} us  vs f32 {spread:.3e}");
    }
    println!("{name} total {total_micros:.1} us at {ROWS} rows");
    if failed {
        bail!("a projection strayed from the f32 path");
    }
    Ok(())
}
