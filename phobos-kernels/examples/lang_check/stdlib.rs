// The standard library on the device: each function as a kernel calls it,
// against the closed form on the host in f64.

use super::*;

/// `[16, 64]` of `X` through `body`, which writes `Y`, against `want`.
fn elementwise(name: &str, body: &str, x: &[f32], want: impl Fn(f64) -> f64, tolerance: f32, failures: &mut u32) -> Result<()> {
    let src = format!(
        "kernel k(X: tensor<f32>[16, 64], Y: tensor<f32>[16, 64]) {{
            var x = X[0 :+ 16, 0 :+ 64]
            {body}
        }}"
    );
    let a = Arg::new(x, 16, 64)?;
    let y = Arg::<f32>::zeroed(16, 64)?;
    let mut slots = Vec::new();
    a.push(&mut slots);
    y.push(&mut slots);
    run(&src, "k", 1, &mut slots)?;
    // Absolute below 1, relative above.
    let worst = y
        .read()?
        .iter()
        .zip(x)
        .map(|(&g, &v)| {
            let w = want(v as f64);
            ((g as f64 - w).abs() / w.abs().max(1.0)) as f32
        })
        .fold(0.0f32, f32::max);
    report(name, worst <= tolerance, format!("err {worst:.2e}"), failures);
    Ok(())
}

fn erf(x: f64) -> f64 {
    // The series, exact to double precision for |x| <= 4.
    let (mut term, mut total) = (x, x);
    for n in 1..120 {
        term *= -x * x / n as f64;
        total += term / (2 * n + 1) as f64;
    }
    2.0 / std::f64::consts::PI.sqrt() * total
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}

pub(crate) fn activations(failures: &mut u32) -> Result<()> {
    let x: Vec<f32> = noise(16 * 64, 31).iter().map(|v| v * 4.0).collect();
    // ex2.approx and lg2.approx: a few ulp; tanh.approx: about 2^-11.
    const APPROX: f32 = 2e-5;
    const TANH: f32 = 2e-3;
    let elu = |x: f64, a: f64| if x > 0.0 { x } else { a * (x.exp() - 1.0) };
    let gelu_tanh = |x: f64| 0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh());
    type Case = (&'static str, f32, Box<dyn Fn(f64) -> f64>);
    let cases: Vec<Case> = vec![
        ("relu(x)", 0.0, Box::new(|x| x.max(0.0))),
        ("relu6(x)", 0.0, Box::new(|x| x.clamp(0.0, 6.0))),
        ("leaky_relu(x, 0.1)", 1e-7, Box::new(|x| if x > 0.0 { x } else { 0.1 * x })),
        ("hardtanh(x)", 0.0, Box::new(|x| x.clamp(-1.0, 1.0))),
        ("elu(x, 1.0)", APPROX, Box::new(move |x| elu(x, 1.0))),
        ("selu(x)", APPROX, Box::new(move |x| 1.0507009873554805 * elu(x, 1.6732632423543772))),
        ("celu(x, 2.0)", APPROX, Box::new(move |x| elu(x / 2.0, 1.0) * 2.0)),
        ("squared_relu(x)", 1e-6, Box::new(|x| x.max(0.0).powi(2))),
        ("sigmoid(x)", APPROX, Box::new(sigmoid)),
        ("log_sigmoid(x)", APPROX, Box::new(|x| -softplus(-x))),
        ("hardsigmoid(x)", 1e-7, Box::new(|x| (x / 6.0 + 0.5).clamp(0.0, 1.0))),
        ("softsign(x)", 1e-6, Box::new(|x| x / (1.0 + x.abs()))),
        ("silu(x)", APPROX, Box::new(|x| x * sigmoid(x))),
        ("swish(x, 1.5)", APPROX, Box::new(|x| x * sigmoid(1.5 * x))),
        ("hardswish(x)", 1e-6, Box::new(|x| x * (x / 6.0 + 0.5).clamp(0.0, 1.0))),
        ("quick_gelu(x)", APPROX, Box::new(|x| x * sigmoid(1.702 * x))),
        ("gelu(x)", 2e-6 + APPROX, Box::new(|x| 0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2)))),
        ("gelu_tanh(x)", TANH, Box::new(gelu_tanh)),
        ("softplus(x)", APPROX, Box::new(softplus)),
        ("mish(x)", TANH, Box::new(|x| x * softplus(x).tanh())),
        ("erf(x)", 2e-6 + APPROX, Box::new(erf)),
        ("sign(x)", 0.0, Box::new(|x: f64| x.signum())),
        ("clamp(x, -1.5, 2.5)", 0.0, Box::new(|x| x.clamp(-1.5, 2.5))),
        ("lerp(x, 1.0, 0.25)", 1e-6, Box::new(|x| x + 0.25 * (1.0 - x))),
        ("exp2(x)", APPROX, Box::new(|x: f64| x.exp2())),
        ("square(x) + rcp(x)", 1e-5, Box::new(|x| x * x + 1.0 / x)),
        ("swiglu(x, x)", APPROX, Box::new(|x| x * sigmoid(x) * x)),
        ("geglu(x, 2.0)", 2e-6 + APPROX, Box::new(|x| 0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2)) * 2.0)),
        ("reglu(x, x)", 1e-6, Box::new(|x| x.max(0.0) * x)),
    ];
    for (call, tolerance, want) in &cases {
        let body = format!("Y[0 :+ 16, 0 :+ 64] = {call}");
        elementwise(&format!("std {call}"), &body, &x, want, *tolerance, failures)?;
    }
    // log2 for a positive x.
    let pos: Vec<f32> = x.iter().map(|v| v.abs() + 0.05).collect();
    elementwise("std log2(x)", "Y[0 :+ 16, 0 :+ 64] = log2(x)", &pos, |x| x.log2(), APPROX, failures)?;
    Ok(())
}

/// `[16, 64]` of `X` through `body`, which writes `Y` (`[16, cols]`), against
/// `want` row by row.
fn rows(name: &str, body: &str, cols: usize, x: &[f32], want: impl Fn(&[f64]) -> Vec<f64>, tolerance: f32, failures: &mut u32) -> Result<()> {
    let src = format!(
        "kernel k(X: tensor<f32>[16, 64], Y: tensor<f32>[16, {cols}], W: tensor<f32>[1, 64], B: tensor<f32>[1, 64]) {{
            var x = X[0 :+ 16, 0 :+ 64]
            {body}
        }}"
    );
    let (w, b) = (weight(), bias());
    let (a, y, gw, gb) = (Arg::new(x, 16, 64)?, Arg::<f32>::zeroed(16, cols)?, Arg::new(&w, 1, 64)?, Arg::new(&b, 1, 64)?);
    let mut slots = Vec::new();
    for arg in [&a, &y, &gw, &gb] {
        arg.push(&mut slots);
    }
    run(&src, "k", 1, &mut slots)?;
    let got = y.read()?;
    let mut worst = 0.0f32;
    for r in 0..16 {
        let row: Vec<f64> = x[r * 64..(r + 1) * 64].iter().map(|&v| v as f64).collect();
        for (c, w) in want(&row).iter().enumerate() {
            let g = got[r * cols + c] as f64;
            worst = worst.max(((g - w).abs() / w.abs().max(1.0)) as f32);
        }
    }
    report(name, worst <= tolerance, format!("err {worst:.2e}"), failures);
    Ok(())
}

fn weight() -> Vec<f32> {
    noise(64, 41).iter().map(|v| 1.0 + 0.5 * v).collect()
}

fn bias() -> Vec<f32> {
    noise(64, 42)
}

fn softmax_of(row: &[f64]) -> Vec<f64> {
    let top = row.iter().cloned().fold(f64::MIN, f64::max);
    let e: Vec<f64> = row.iter().map(|v| (v - top).exp()).collect();
    let total: f64 = e.iter().sum();
    e.iter().map(|v| v / total).collect()
}

pub(crate) fn row_functions(failures: &mut u32) -> Result<()> {
    let x: Vec<f32> = noise(16 * 64, 32).iter().map(|v| v * 6.0).collect();
    let mean = |r: &[f64]| r.iter().sum::<f64>() / r.len() as f64;
    let lse = |r: &[f64]| {
        let top = r.iter().cloned().fold(f64::MIN, f64::max);
        top + r.iter().map(|v| (v - top).exp()).sum::<f64>().ln()
    };
    rows("std softmax", "softmax(x)\n Y[0 :+ 16, :] = x", 64, &x, softmax_of, 2e-6, failures)?;
    rows("std log_softmax", "log_softmax(x)\n Y[0 :+ 16, :] = x", 64, &x, |r| r.iter().map(|v| v - lse(r)).collect(), 2e-5, failures)?;
    rows("std logsumexp", "logsumexp(Y[0 :+ 16, :], x)", 1, &x, |r| vec![lse(r)], 2e-5, failures)?;
    rows("std rowmean", "Y[0 :+ 16, :] = rowmean(x * x)", 1, &x, |r| vec![r.iter().map(|v| v * v).sum::<f64>() / 64.0], 1e-5, failures)?;
    rows("std rowmin", "Y[0 :+ 16, :] = rowmin(x)", 1, &x, |r| vec![r.iter().cloned().fold(f64::MAX, f64::min)], 0.0, failures)?;
    rows(
        "std mask_cols, softmax",
        "mask_cols(x, 37, -1.0e30)\n softmax(x)\n Y[0 :+ 16, :] = x",
        64,
        &x,
        |r| {
            let mut p = softmax_of(&r[..37]);
            p.resize(64, 0.0);
            p
        },
        2e-6,
        failures,
    )?;
    causal(&x, failures)?;
    topk(&x, failures)?;

    let w: Vec<f64> = weight().iter().map(|&v| v as f64).collect();
    let b: Vec<f64> = bias().iter().map(|&v| v as f64).collect();
    let (w1, w2, b2) = (w.clone(), w.clone(), b.clone());
    rows(
        "std rms_norm",
        "rms_norm(x, W[0 :+ 1, :], 0.00001)\n Y[0 :+ 16, :] = x",
        64,
        &x,
        move |r| {
            let s = 1.0 / (r.iter().map(|v| v * v).sum::<f64>() / 64.0 + 1e-5).sqrt();
            r.iter().zip(&w1).map(|(v, g)| v * s * g).collect()
        },
        2e-5,
        failures,
    )?;
    rows(
        "std layer_norm",
        "layer_norm(x, W[0 :+ 1, :], B[0 :+ 1, :], 0.00001)\n Y[0 :+ 16, :] = x",
        64,
        &x,
        move |r| {
            let mu = mean(r);
            let var = r.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / 64.0;
            let s = 1.0 / (var + 1e-5).sqrt();
            r.iter().zip(&w2).zip(&b2).map(|((v, g), b)| (v - mu) * s * g + b).collect()
        },
        2e-5,
        failures,
    )?;
    rows(
        "std l2_normalize",
        "l2_normalize(x, 0.000001)\n Y[0 :+ 16, :] = x",
        64,
        &x,
        |r| {
            let n = r.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-6);
            r.iter().map(|v| v / n).collect()
        },
        2e-6,
        failures,
    )?;
    Ok(())
}

/// `causal_mask` row by row: row `i` keeps keys `8 + j <= 40 + i`.
fn causal(x: &[f32], failures: &mut u32) -> Result<()> {
    let src = "kernel k(X: tensor<f32>[16, 64], Y: tensor<f32>[16, 64]) {
            var x = X[0 :+ 16, 0 :+ 64]
            causal_mask(x, 40, 8)
            softmax(x)
            Y[0 :+ 16, :] = x
        }";
    let (a, y) = (Arg::new(x, 16, 64)?, Arg::<f32>::zeroed(16, 64)?);
    let mut slots = Vec::new();
    a.push(&mut slots);
    y.push(&mut slots);
    run(src, "k", 1, &mut slots)?;
    let got = y.read()?;
    let mut worst = 0.0f32;
    for i in 0..16 {
        let keep = (32 + i + 1).min(64);
        let row: Vec<f64> = x[i * 64..i * 64 + keep].iter().map(|&v| v as f64).collect();
        let mut p = softmax_of(&row);
        p.resize(64, 0.0);
        for j in 0..64 {
            worst = worst.max((got[i * 64 + j] as f64 - p[j]).abs() as f32);
        }
    }
    report("std causal_mask", worst < 2e-6, format!("err {worst:.2e}"), failures);
    Ok(())
}

fn topk(x: &[f32], failures: &mut u32) -> Result<()> {
    let src = "kernel k(X: tensor<f32>[16, 64], V: tensor<f32>[16, 4], I: tensor<i32>[16, 4]) {
            var t = X[0 :+ 16, 0 :+ 64]
            var v: tile<f32>[16, 4]
            var i: tile<i32>[16, 4]
            topk(v, i, t)
            V[0 :+ 16, :] = v
            I[0 :+ 16, :] = i
        }";
    let (a, v, i) = (Arg::new(x, 16, 64)?, Arg::<f32>::zeroed(16, 4)?, Arg::<i32>::zeroed(16, 4)?);
    let mut slots = Vec::new();
    a.push(&mut slots);
    v.push(&mut slots);
    i.push(&mut slots);
    run(src, "k", 1, &mut slots)?;
    let (vals, idx) = (v.read()?, i.read()?);
    let mut ok = true;
    for r in 0..16 {
        let mut order: Vec<usize> = (0..64).collect();
        order.sort_by(|&a, &b| x[r * 64 + b].total_cmp(&x[r * 64 + a]).then(a.cmp(&b)));
        for k in 0..4 {
            ok &= idx[r * 4 + k] as usize == order[k] && vals[r * 4 + k] == x[r * 64 + order[k]];
        }
    }
    report("std topk", ok, "4 of 64 a row".into(), failures);
    Ok(())
}

pub(crate) fn rope_and_quant(failures: &mut u32) -> Result<()> {
    let x = noise(16 * 64, 33);
    let angle: Vec<f32> = noise(16 * 32, 34).iter().map(|v| v * 3.0).collect();
    let (cos, sin): (Vec<f32>, Vec<f32>) = angle.iter().map(|a| (a.cos(), a.sin())).unzip();
    let src = "kernel k(X: tensor<f32>[16, 64], C: tensor<f32>[16, 32], S: tensor<f32>[16, 32], Y: tensor<f32>[16, 64]) {
            var x = X[0 :+ 16, 0 :+ 64]
            rope(x, C[0 :+ 16, :], S[0 :+ 16, :])
            Y[0 :+ 16, :] = x
        }";
    let (a, c, s, y) = (Arg::new(&x, 16, 64)?, Arg::new(&cos, 16, 32)?, Arg::new(&sin, 16, 32)?, Arg::<f32>::zeroed(16, 64)?);
    let mut slots = Vec::new();
    for arg in [&a, &c, &s, &y] {
        arg.push(&mut slots);
    }
    run(src, "k", 1, &mut slots)?;
    let got = y.read()?;
    let mut worst = 0.0f32;
    for r in 0..16 {
        for j in 0..32 {
            let (p, q) = (x[r * 64 + j], x[r * 64 + 32 + j]);
            let (cs, sn) = (cos[r * 32 + j], sin[r * 32 + j]);
            worst = worst.max((got[r * 64 + j] - (p * cs - q * sn)).abs());
            worst = worst.max((got[r * 64 + 32 + j] - (q * cs + p * sn)).abs());
        }
    }
    report("std rope", worst < 1e-6, format!("err {worst:.2e}"), failures);

    // The round trip: quantized within half a step of the row's scale.
    let src = "kernel k(X: tensor<f32>[16, 64], Q: tensor<i8>[16, 64], S: tensor<f32>[16, 1], Y: tensor<f32>[16, 64]) {
            var x = X[0 :+ 16, 0 :+ 64]
            var q: tile<i8>[16, 64]
            quantize_i8(q, S[0 :+ 16, :], x)
            Q[0 :+ 16, :] = q
            dequantize_i8(Y[0 :+ 16, :], q, S[0 :+ 16, :])
        }";
    let (a, q, s, y) = (Arg::new(&x, 16, 64)?, Arg::<i8>::zeroed(16, 64)?, Arg::<f32>::zeroed(16, 1)?, Arg::<f32>::zeroed(16, 64)?);
    let mut slots = Vec::new();
    a.push(&mut slots);
    q.push(&mut slots);
    s.push(&mut slots);
    y.push(&mut slots);
    run(src, "k", 1, &mut slots)?;
    let (qs, ss, ys) = (q.read()?, s.read()?, y.read()?);
    let mut ok = true;
    let mut worst = 0.0f32;
    for r in 0..16 {
        let top = x[r * 64..(r + 1) * 64].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        ok &= (ss[r] - top / 127.0).abs() <= 1e-6 * top;
        ok &= qs[r * 64..(r + 1) * 64].iter().any(|&v| v.unsigned_abs() == 127);
        for j in 0..64 {
            worst = worst.max((ys[r * 64 + j] - x[r * 64 + j]).abs() / ss[r]);
        }
    }
    report("std quantize_i8 round trip", ok && worst <= 0.5 + 1e-3, format!("worst {worst:.3} of a step"), failures);
    Ok(())
}

pub(crate) fn matmuls(failures: &mut u32) -> Result<()> {
    let (m, n, k) = (64, 64, 96);
    let (a, b) = (noise(m * k, 35), noise(k * n, 36));
    let bt: Vec<f32> = (0..n * k).map(|i| b[(i % k) * n + i / k]).collect();
    let mut want = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            want[i * n + j] = (0..k).map(|x| a[i * k + x] as f64 * b[x * n + j] as f64).sum();
        }
    }
    for (name, call, rhs, rhs_dims) in [
        ("std matmul", "matmul(acc, A, B, m0, n0, 16)", &b, (k, n)),
        ("std matmul_t", "matmul_t(acc, A, B, m0, n0, 16)", &bt, (n, k)),
    ] {
        let src = format!(
            "@autotune(BM in [32], BN in [32])
            kernel k(A: tensor<f32>[M, K], B: tensor<f32>[{}, {}], C: tensor<f32>[M, N]) {{
                let m0 = ctx.block.x / 2 * BM
                let n0 = ctx.block.x % 2 * BN
                var acc: tile<f32>[BM, BN] = 0.0
                {call}
                C[m0 :+ BM, n0 :+ BN] = acc
            }}",
            if rhs_dims.0 == k { "K" } else { "N" },
            if rhs_dims.1 == k { "K" } else { "N" },
        );
        let (ga, gb, gc) = (Arg::new(&a, m, k)?, Arg::new(rhs, rhs_dims.0, rhs_dims.1)?, Arg::<f32>::zeroed(m, n)?);
        let mut slots = Vec::new();
        for arg in [&ga, &gb, &gc] {
            arg.push(&mut slots);
        }
        run(&src, "k", 4, &mut slots)?;
        let worst = gc.read()?.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0f64, f64::max) as f32;
        report(name, worst < 1e-4, format!("abs err {worst:.2e}"), failures);
    }
    Ok(())
}

pub(crate) fn attention(failures: &mut u32) -> Result<()> {
    // 100 keys in blocks of 32: the last block is ragged and masked. Two
    // blocks of 32 queries.
    let (r, keys, d) = (64, 100, 64);
    let (q, k, v) = (noise(r * d, 37), noise(keys * d, 38), noise(keys * d, 39));
    let src = "@tensorcore
        @launch(256)
        @autotune(D in [64], BR in [32], BC in [32])
        kernel fa(Q: tensor<f16>[Nq, D], K: tensor<f16>[Nk, D], V: tensor<f16>[Nk, D], O: tensor<f32>[Nq, D]) {
            let row = ctx.block.x * BR
            let q = Q[row :+ BR, :]
            var acc: tile<f32>[BR, D] = 0.0
            attention(acc, q, K, V, 0.125, BR, BC)
            O[row :+ BR, :] = acc
        }";
    let half = |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| phobos_base::half::f16_to_f32(phobos_base::half::f32_to_f16(v))).collect() };
    let (hq, hk, hv) = (half(&q), half(&k), half(&v));
    let mut want = vec![0.0f32; r * d];
    for i in 0..r {
        let s: Vec<f64> = (0..keys)
            .map(|j| 0.125 * (0..d).map(|x| hq[i * d + x] as f64 * hk[j * d + x] as f64).sum::<f64>())
            .collect();
        let p = softmax_of(&s);
        for n in 0..d {
            want[i * d + n] = (0..keys).map(|j| p[j] * hv[j * d + n] as f64).sum::<f64>() as f32;
        }
    }
    let h = |x: &[f32]| -> Vec<u16> { x.iter().map(|&v| phobos_base::half::f32_to_f16(v)).collect() };
    for bits in [32, 64] {
        let (gq, gk, gv) = (Arg::new(&h(&q), r, d)?, Arg::new(&h(&k), keys, d)?, Arg::new(&h(&v), keys, d)?);
        let out = Arg::<f32>::zeroed(r, d)?;
        let mut slots = Vec::new();
        for a in [&gq, &gk, &gv] {
            a.push(&mut slots);
        }
        out.push(&mut slots);
        run_at(bits, src, "fa", 2, &mut slots)?;
        let got = out.read()?;
        let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        let worst = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max) / scale;
        // The probabilities are rounded to f16 for the second dot.
        report(&format!("std attention, ragged i{bits}"), worst < 2e-2, format!("rel err {worst:.2e}"), failures);
    }
    Ok(())
}
