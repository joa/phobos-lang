// The GPU matmul backend against the host reference:
//
//   cargo run --release -p phobos-onnx --features cuda --example mm_check
//
// Covers the shapes a decode step produces, the ones that do not divide the
// tile included: an unaligned M from a short prompt, an unaligned N from the
// 16-wide delta-net gate projections, and the single-row matvec case.

use anyhow::Result;
use phobos_onnx::interp::{HostBackend, MatmulBackend};
use phobos_onnx::runner::GpuBackend;

fn main() -> Result<()> {
    let gpu = GpuBackend::new()?;
    let host = HostBackend;

    // (m, k, n) with a note on what makes each interesting.
    let cases = [
        (1usize, 1024usize, 2048usize, "decode, aligned"),
        (1, 1024, 16, "decode, N below one tile"),
        (1, 1024, 3584, "decode, FFN width"),
        (1, 1024, 248320, "decode, vocab"),
        (5, 1024, 2048, "prompt, M below one tile"),
        (5, 1024, 16, "prompt, both M and N ragged"),
        (32, 1024, 2048, "aligned both ways"),
        (33, 1024, 96, "M just past a tile"),
        (2, 1000, 100, "K not a multiple of the k-slice"),
        (5, 1024, 48, "M ragged, N ragged above one tile"),
        (64, 1024, 16, "M aligned, N below one tile"),
        (32, 1024, 16, "M exactly one tile, N ragged"),
        (5, 1024, 32, "M ragged, N exactly one tile"),
        (8, 1024, 16, "both ragged, smaller"),
        (5, 1024, 64, "M ragged, N two whole tiles"),
    ];

    let mut worst = 0.0f32;
    for (m, k, n, note) in cases {
        // Pseudorandom rather than periodic: a periodic pattern can hide an
        // index or layout mistake by making the wrong elements sum to the right
        // answer.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / 8388608.0 - 1.0
        };
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| next()).collect();

        let want = host.matmul(&a, m, k, &b, n)?;
        let got = gpu.matmul(&a, m, k, &b, n)?;

        let error = want
            .iter()
            .zip(&got)
            .map(|(w, g)| (w - g).abs() / w.abs().max(1.0))
            .fold(0.0f32, f32::max);
        worst = worst.max(error);
        // f32 reductions in a different order will not match bit for bit; this
        // bound is far tighter than any layout mistake could survive.
        let verdict = if error < 1e-5 { "ok  " } else { "FAIL" };
        println!("{verdict} [{m:>3} x {k:>4} x {n:>6}]  rel err {error:>10.3e}   {note}");
    }

    println!("\nworst relative error {worst:.3e}");
    if worst >= 1e-5 {
        anyhow::bail!("GPU matmul disagrees with the host reference");
    }
    Ok(())
}
