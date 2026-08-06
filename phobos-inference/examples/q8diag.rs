// Accuracy of the Q8_0 matvec against an f64 evaluation of the same sum:
//
//   cargo run --release -p phobos-inference --features cuda --example q8diag
//
// The device kernel and the host reference disagree by more than plain f32
// noise would suggest, and comparing both to f64 shows the kernel is the more
// accurate of the two: it sums each 32-element block before scaling and
// accumulating, which grows error more slowly than the host's flat running sum
// over all of k. So the disagreement is summation order, and `backend_check`'s
// tolerance has to allow for it rather than the kernel matching bit for bit.
use anyhow::Result;
use phobos_gguf::compute::{Backend, HostBackend, quantize_row, read_vec};

#[path = "../src/device.rs"]
mod device;

fn main() -> Result<()> {
    let gpu = device::DeviceBackend::new()?;
    let host = HostBackend::new();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 8388608.0 - 1.0
    };

    for (k, n) in [(1024usize, 6144usize), (3584, 1024)] {
        let qs: Vec<i8> = (0..k * n).map(|_| (next() * 127.0) as i8).collect();
        let scales: Vec<f32> = (0..(k / 32) * n).map(|_| next().abs() + 0.01).collect();
        let a: Vec<f32> = (0..k).map(|_| next()).collect();

        // The f64 truth of the sum both backends are actually computing, which
        // means the quantized activation rather than the original: quantizing is
        // part of the operation's definition, and including its error here would
        // swamp the difference in accumulation order this exists to measure.
        //
        // `qs` is [n, k], the order the file stores and the contraction wants,
        // so an output's weights are one contiguous row.
        let mut qa = vec![0i8; k];
        let mut da = vec![0.0f64; k / 32];
        for (b, chunk) in a.chunks_exact(32).enumerate() {
            da[b] = quantize_row(chunk, &mut qa[b * 32..(b + 1) * 32]) as f64;
        }
        let mut truth = vec![0.0f64; n];
        for (j, t) in truth.iter_mut().enumerate() {
            let row = &qs[j * k..(j + 1) * k];
            for p in 0..k {
                let b = p / 32;
                *t += qa[p] as f64 * da[b] * row[p] as f64 * scales[b * n + j] as f64;
            }
        }

        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let ab = b.upload(&a)?;
            let wb = b.constant_q8(&format!("q{k}x{n}"), &qs, &scales, k, n)?;
            let out = b.alloc(n)?;
            b.matmul_q8(ab, 1, k, wb, n, out)?;
            read_vec(b, out, n)
        };
        let h = run(&host)?;
        let g = run(&gpu)?;

        let rel = |v: &[f32]| {
            v.iter()
                .zip(&truth)
                .map(|(&x, &t)| (x as f64 - t).abs() / t.abs().max(1.0))
                .fold(0.0f64, f64::max)
        };
        let mag = truth.iter().fold(0.0f64, |m, &t| m.max(t.abs()));
        println!(
            "k={k} n={n}  |value| up to {mag:.1}\n  host vs f64: {:.3e}\n  gpu  vs f64: {:.3e}",
            rel(&h),
            rel(&g)
        );
    }
    Ok(())
}
