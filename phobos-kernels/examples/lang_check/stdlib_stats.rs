// The standard library's statistics on the device: column reductions and
// batch norm, the losses, and the random numbers, each against the host.

use super::*;

/// The worst of `got` against `want`, absolute below 1 and relative above.
fn worst_of(got: &[f32], want: &[f64]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| ((g as f64 - w).abs() / w.abs().max(1.0)) as f32)
        .fold(0.0f32, f32::max)
}

pub(crate) fn columns(failures: &mut u32) -> Result<()> {
    let x: Vec<f32> = noise(16 * 64, 51).iter().map(|v| v * 3.0).collect();
    let (w, b) = (noise(64, 52), noise(64, 53));
    let src = "kernel k(X: tensor<f32>[16, 64], S: tensor<f32>[4, 64], W: tensor<f32>[1, 64], B: tensor<f32>[1, 64],
                    Y: tensor<f32>[16, 64]) {
            var x = X[0 :+ 16, 0 :+ 64]
            S[0 :+ 1, :] = colsum(x)
            S[1 :+ 1, :] = colmax(x)
            S[2 :+ 1, :] = colmin(x)
            var mu: tile<f32>[1, 64] = 0.0
            var va: tile<f32>[1, 64] = 0.0
            batch_norm_stats(mu, va, x)
            S[3 :+ 1, :] = va
            Y[0 :+ 16, :] = batch_norm(x, mu, va, W[0 :+ 1, :], B[0 :+ 1, :], 0.00001)
        }";
    let (gx, gs, gw, gb, gy) = (
        Arg::new(&x, 16, 64)?,
        Arg::<f32>::zeroed(4, 64)?,
        Arg::new(&w, 1, 64)?,
        Arg::new(&b, 1, 64)?,
        Arg::<f32>::zeroed(16, 64)?,
    );
    let mut slots = Vec::new();
    for arg in [&gx, &gs, &gw, &gb, &gy] {
        arg.push(&mut slots);
    }
    run(src, "k", 1, &mut slots)?;
    let x = &x;
    let col = |c: usize| (0..16).map(move |r| x[r * 64 + c] as f64);
    let mut stats = vec![0.0f64; 4 * 64];
    let mut normed = vec![0.0f64; 16 * 64];
    for c in 0..64 {
        let mean = col(c).sum::<f64>() / 16.0;
        let var = col(c).map(|v| (v - mean).powi(2)).sum::<f64>() / 16.0;
        stats[c] = col(c).sum();
        stats[64 + c] = col(c).fold(f64::MIN, f64::max);
        stats[128 + c] = col(c).fold(f64::MAX, f64::min);
        stats[192 + c] = var;
        for r in 0..16 {
            normed[r * 64 + c] = (x[r * 64 + c] as f64 - mean) / (var + 1e-5).sqrt() * w[c] as f64 + b[c] as f64;
        }
    }
    let worst = worst_of(&gs.read()?, &stats);
    report("std colsum, colmax, colmin", worst <= 1e-6, format!("err {worst:.2e}"), failures);
    let worst = worst_of(&gy.read()?, &normed);
    report("std batch_norm_stats, batch_norm", worst <= 2e-5, format!("err {worst:.2e}"), failures);
    Ok(())
}

pub(crate) fn losses(failures: &mut u32) -> Result<()> {
    let x: Vec<f32> = noise(16 * 64, 54).iter().map(|v| v * 4.0).collect();
    // In [0, 1), every seventh exactly 0, where `kl_div` adds nothing.
    let p: Vec<f32> = noise(16 * 64, 55)
        .iter()
        .enumerate()
        .map(|(i, v)| if i % 7 == 0 { 0.0 } else { v.abs() })
        .collect();
    let target: Vec<i32> = (0..16).map(|r| r * 37 % 64).collect();
    let src = "kernel k(X: tensor<f32>[16, 64], P: tensor<f32>[16, 64], T: tensor<i32>[16, 1], L: tensor<f32>[16, 8]) {
            var x = X[0 :+ 16, 0 :+ 64]
            var p = P[0 :+ 16, 0 :+ 64]
            cross_entropy(L[0 :+ 16, 0 :+ 1], x, T[0 :+ 16, :])
            L[0 :+ 16, 1 :+ 1] = nll_loss(x, T[0 :+ 16, :])
            L[0 :+ 16, 2 :+ 1] = mse_loss(x, p)
            L[0 :+ 16, 3 :+ 1] = l1_loss(x, p)
            L[0 :+ 16, 4 :+ 1] = smooth_l1_loss(x, p, 1.5)
            L[0 :+ 16, 5 :+ 1] = kl_div(x, p)
            L[0 :+ 16, 6 :+ 1] = binary_cross_entropy_with_logits(x, p)
        }";
    let (gx, gp, gt, gl) = (Arg::new(&x, 16, 64)?, Arg::new(&p, 16, 64)?, Arg::new(&target, 16, 1)?, Arg::<f32>::zeroed(16, 8)?);
    let mut slots = Vec::new();
    gx.push(&mut slots);
    gp.push(&mut slots);
    gt.push(&mut slots);
    gl.push(&mut slots);
    run(src, "k", 1, &mut slots)?;
    let got = gl.read()?;
    let names = ["cross_entropy", "nll_loss", "mse_loss", "l1_loss", "smooth_l1_loss", "kl_div", "binary_cross_entropy_with_logits"];
    let mut worst = [0.0f32; 7];
    for r in 0..16 {
        let xs: Vec<f64> = x[r * 64..(r + 1) * 64].iter().map(|&v| v as f64).collect();
        let ps: Vec<f64> = p[r * 64..(r + 1) * 64].iter().map(|&v| v as f64).collect();
        let t = target[r] as usize;
        let top = xs.iter().cloned().fold(f64::MIN, f64::max);
        let lse = top + xs.iter().map(|v| (v - top).exp()).sum::<f64>().ln();
        let mean = |f: &dyn Fn(f64, f64) -> f64| xs.iter().zip(&ps).map(|(&a, &b)| f(a, b)).sum::<f64>() / 64.0;
        let want = [
            lse - xs[t],
            -xs[t],
            mean(&|a, b| (a - b).powi(2)),
            mean(&|a, b| (a - b).abs()),
            mean(&|a, b| {
                let d = (a - b).abs();
                if d < 1.5 { 0.5 * d * d / 1.5 } else { d - 0.75 }
            }),
            xs.iter().zip(&ps).map(|(&l, &q)| if q > 0.0 { q * (q.ln() - l) } else { 0.0 }).sum(),
            mean(&|a, b| a.max(0.0) - a * b + (1.0 + (-a.abs()).exp()).ln()),
        ];
        for (j, w) in want.iter().enumerate() {
            let g = got[r * 8 + j] as f64;
            worst[j] = worst[j].max(((g - w).abs() / w.abs().max(1.0)) as f32);
        }
    }
    for (name, worst) in names.iter().zip(worst) {
        report(&format!("std {name}"), worst <= 2e-5, format!("err {worst:.2e}"), failures);
    }
    Ok(())
}

/// Chris Wellons' `lowbias32`, which `hash32` is.
fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}

fn uniform(seed: u32, counter: u32) -> f32 {
    (lowbias32(counter ^ lowbias32(seed)) & 0x7f_ffff) as f32 * (2.0f32).powi(-23)
}

pub(crate) fn random(failures: &mut u32) -> Result<()> {
    // 256 rows of 64, 16 blocks of 16 rows: a counter a sample.
    let (rows, seed) = (256, -12345i32);
    let x: Vec<f32> = noise(rows * 64, 56).iter().map(|v| v + 2.0).collect();
    let src = "kernel k(X: tensor<f32>[N, 64], U: tensor<f32>[N, 64], G: tensor<f32>[N, 64], D: tensor<f32>[N, 64],
                    H: tensor<i32>[N, 64], seed: i32) {
            let r = ctx.block.x * 16
            var x = X[r :+ 16, :]
            U[r :+ 16, :] = uniform(seed, (r + iota(0)) * 64 + iota(1))
            G[r :+ 16, :] = randn(seed, (r + iota(0)) * 64 + iota(1))
            D[r :+ 16, :] = dropout(x, 0.25, seed, (r + iota(0)) * 64 + iota(1))
            H[r :+ 16, :] = hash32((r + iota(0)) * 64 + iota(1))
        }";
    let gx = Arg::new(&x, rows, 64)?;
    let (gu, gg, gd) = (Arg::<f32>::zeroed(rows, 64)?, Arg::<f32>::zeroed(rows, 64)?, Arg::<f32>::zeroed(rows, 64)?);
    let gh = Arg::<i32>::zeroed(rows, 64)?;
    let mut slots = Vec::new();
    for arg in [&gx, &gu, &gg, &gd] {
        arg.push(&mut slots);
    }
    gh.push(&mut slots);
    slots.push(seed as u32 as u64);
    run(src, "k", (rows / 16) as u32, &mut slots)?;
    let (u, g, d, h) = (gu.read()?, gg.read()?, gd.read()?, gh.read()?);
    let n = rows * 64;
    let seed = seed as u32;

    let hashed = (0..n).all(|c| h[c] as u32 == lowbias32(c as u32));
    report("std hash32", hashed, "bit-exact".into(), failures);
    let exact = (0..n).all(|c| u[c] == uniform(seed, c as u32));
    report("std uniform, bit-exact", exact, format!("{n} counters"), failures);
    // The moments, and 16 buckets against a flat histogram: chi-square at
    // 15 degrees of freedom stays under 37.7 but one time in a thousand.
    let mean = u.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var = u.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    let mut buckets = [0usize; 16];
    for &v in &u {
        buckets[(v * 16.0) as usize] += 1;
    }
    let expected = n as f64 / 16.0;
    let chi2: f64 = buckets.iter().map(|&b| (b as f64 - expected).powi(2) / expected).sum();
    let flat = (mean - 0.5).abs() < 0.01 && (var - 1.0 / 12.0).abs() < 0.003 && chi2 < 37.7;
    report("std uniform, statistics", flat, format!("mean {mean:.4} var {var:.4} chi2 {chi2:.1}"), failures);

    let normal = |c: u32| {
        let (u1, u2) = (uniform(seed, 2 * c) as f64, uniform(seed, 2 * c + 1) as f64);
        (-2.0 * (1.0 - u1).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let close = (0..n).map(|c| (g[c] as f64 - normal(c as u32)).abs()).fold(0.0f64, f64::max);
    let mean = g.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var = g.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    let ok = close < 1e-3 && mean.abs() < 0.03 && (var - 1.0).abs() < 0.04;
    report("std randn", ok, format!("err {close:.1e} mean {mean:.4} var {var:.4}"), failures);

    // Kept where the uniform clears p, scaled by 1 / (1 - p).
    let mut ok = true;
    let mut kept = 0;
    for c in 0..n {
        let want = if uniform(seed, c as u32) >= 0.25 { x[c] / 0.75 } else { 0.0 };
        ok &= (d[c] - want).abs() <= 1e-6 * want.abs().max(1.0);
        kept += usize::from(d[c] != 0.0);
    }
    let rate = kept as f64 / n as f64;
    report("std dropout", ok && (rate - 0.75).abs() < 0.01, format!("kept {rate:.4}"), failures);
    Ok(())
}
