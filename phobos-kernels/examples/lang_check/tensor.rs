// The contractions on the tensor cores and their fused updates.

use super::*;

pub(crate) fn dot_epilogue(failures: &mut u32) -> Result<()> {
    // Attention's two updates of a product: without @tensorcore both run as
    // the product is written out, with it the dots take the tensor cores and
    // the updates their own sweeps.
    let (r, d) = (32, 64);
    let (q, k, v, c) = (noise(r * d, 15), noise(r * d, 16), noise(r * d, 17), noise(r, 18));
    for (attrs, elem) in [("", "f32"), ("@tensorcore", "f16")] {
        let src = format!(
            "{attrs}
            @launch(256)
            kernel k(Q: tensor<{elem}>[32, 64], K: tensor<{elem}>[32, 64], V: tensor<{elem}>[32, 64],
                     C: tensor<f32>[32, 1], O: tensor<f32>[32, 64]) {{
                var q = Q[0 :+ 32, 0 :+ 64]
                var k = K[0 :+ 32, 0 :+ 64]
                var v = V[0 :+ 32, 0 :+ 64]
                var c = C[0 :+ 32, 0 :+ 1]
                var acc: tile<f32>[32, 64] = 1.0
                var s: tile<f32>[32, 32] = dot_t(q, k)
                s = s * 0.125
                acc = acc * c + dot(s, v)
                O[0 :+ 32, 0 :+ 64] = acc
            }}"
        );
        let half = elem == "f16";
        let round = |x: &[f32]| -> Vec<f32> {
            x.iter()
                .map(|&v| if half { phobos_base::half::f16_to_f32(phobos_base::half::f32_to_f16(v)) } else { v })
                .collect()
        };
        let (hq, hk, hv) = (round(&q), round(&k), round(&v));
        let mut want = vec![0.0f32; r * d];
        for i in 0..r {
            let s: Vec<f32> = (0..r)
                .map(|j| 0.125 * (0..d).map(|x| hq[i * d + x] * hk[j * d + x]).sum::<f32>())
                .collect();
            for n in 0..d {
                want[i * d + n] = c[i] + (0..r).map(|j| s[j] * hv[j * d + n]).sum::<f32>();
            }
        }
        let gc = Arg::new(&c, r, 1)?;
        let out = Arg::<f32>::zeroed(r, d)?;
        let mut slots = Vec::new();
        if half {
            let h = |x: &[f32]| -> Vec<u16> { x.iter().map(|&v| phobos_base::half::f32_to_f16(v)).collect() };
            let (gq, gk, gv) = (Arg::new(&h(&q), r, d)?, Arg::new(&h(&k), r, d)?, Arg::new(&h(&v), r, d)?);
            for a in [&gq, &gk, &gv] {
                a.push(&mut slots);
            }
            gc.push(&mut slots);
            out.push(&mut slots);
            run(&src, "k", 1, &mut slots)?;
        } else {
            let (gq, gk, gv) = (Arg::new(&q, r, d)?, Arg::new(&k, r, d)?, Arg::new(&v, r, d)?);
            for a in [&gq, &gk, &gv] {
                a.push(&mut slots);
            }
            gc.push(&mut slots);
            out.push(&mut slots);
            run(&src, "k", 1, &mut slots)?;
        }
        let got = out.read()?;
        let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        let worst = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max) / scale;
        // The tensor cores round the scores to f16 before the second dot.
        let tol = if half { 1e-2 } else { 1e-5 };
        report(&format!("dot epilogue {elem}"), worst < tol, format!("rel err {worst:.2e}"), failures);
    }
    Ok(())
}

pub(crate) fn flash_softmax(failures: &mut u32) -> Result<()> {
    // The online softmax of flash attention over two key blocks, at 64-bit
    // indices: the scores stay in mma.sync fragments, the scale, the row max,
    // the exponent and the row sum run in the lanes, and the scores reach
    // the second dot through one shared tile. The shared-tile form of the
    // same source needs more than the 48 KB of static shared memory.
    let (r, keys, d) = (128, 64, 32);
    let (q, k, v) = (noise(r * d, 21), noise(keys * d, 22), noise(keys * d, 23));
    let src = "@tensorcore
        @launch(256)
        kernel fa(Q: tensor<f16>[128, 32], K: tensor<f16>[64, 32], V: tensor<f16>[64, 32], O: tensor<f32>[128, 32]) {
            let q = Q[0 :+ 128, 0 :+ 32]
            var acc: tile<f32>[128, 32] = 0.0
            var m: tile<f32>[128, 1] = -65504.0
            var l: tile<f32>[128, 1] = 0.0
            for kt in range(0, 64, 32) {
                let k = K[kt :+ 32, 0 :+ 32]
                let v = V[kt :+ 32, 0 :+ 32]
                var s: tile<f32>[128, 32] = dot_t(q, k)
                s = s * 0.125
                var mnew: tile<f32>[128, 1] = rowmax(s)
                mnew = tmax(m, mnew)
                s = exp(s - mnew)
                var corr: tile<f32>[128, 1] = exp(m - mnew)
                l = l * corr
                l += rowsum(s)
                acc = acc * corr
                acc += dot(s, v)
                m = mnew
            }
            acc = acc / l
            O[0 :+ 128, 0 :+ 32] = acc
        }";
    let half = |x: &[f32]| -> Vec<f32> {
        x.iter().map(|&v| phobos_base::half::f16_to_f32(phobos_base::half::f32_to_f16(v))).collect()
    };
    let (hq, hk, hv) = (half(&q), half(&k), half(&v));
    // softmax(q k^T / 8) v, row by row, in f64.
    let mut want = vec![0.0f32; r * d];
    for i in 0..r {
        let s: Vec<f64> = (0..keys)
            .map(|j| 0.125 * (0..d).map(|x| hq[i * d + x] as f64 * hk[j * d + x] as f64).sum::<f64>())
            .collect();
        let top = s.iter().cloned().fold(f64::MIN, f64::max);
        let p: Vec<f64> = s.iter().map(|&x| (x - top).exp()).collect();
        let total: f64 = p.iter().sum();
        for n in 0..d {
            want[i * d + n] = ((0..keys).map(|j| p[j] * hv[j * d + n] as f64).sum::<f64>() / total) as f32;
        }
    }
    let h = |x: &[f32]| -> Vec<u16> { x.iter().map(|&v| phobos_base::half::f32_to_f16(v)).collect() };
    {
        let (gq, gk, gv) = (Arg::new(&h(&q), r, d)?, Arg::new(&h(&k), keys, d)?, Arg::new(&h(&v), keys, d)?);
        let out = Arg::<f32>::zeroed(r, d)?;
        let mut slots = Vec::new();
        for a in [&gq, &gk, &gv] {
            a.push(&mut slots);
        }
        out.push(&mut slots);
        run_at(64, src, "fa", 1, &mut slots)?;
        let got = out.read()?;
        let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        let worst = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max) / scale;
        // The probabilities are rounded to f16 for the second dot.
        report("flash softmax, mma.sync", worst < 2e-2, format!("rel err {worst:.2e}"), failures);
    }
    Ok(())
}

pub(crate) fn scaled_int8_dot(failures: &mut u32) -> Result<()> {
    // acc += f32(dot_t(a, w)) * row * col held in the integer tensor cores'
    // registers across k, with w read from global memory and from a shared
    // tile the loop rewrites.
    let (m, n, k) = (64, 128, 128);
    let kb = k / 32;
    let byte = |seed| -> Vec<i8> { noise(m.max(n) * k, seed).iter().map(|v| (v * 127.0) as i8).collect() };
    let (a, w) = (byte(21)[..m * k].to_vec(), byte(22)[..n * k].to_vec());
    let (asc, wsc) = (noise(m * kb, 23), noise(kb * n, 24));
    let mut want = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            for b in 0..kb {
                let s: i32 = (b * 32..(b + 1) * 32).map(|x| a[i * k + x] as i32 * w[j * k + x] as i32).sum();
                want[i * n + j] += s as f64 * asc[i * kb + b] as f64 * wsc[b * n + j] as f64;
            }
        }
    }
    let mut kernels = Vec::new();
    for (what, w_decl) in [
        ("global", "let w = W[pn * 64 :+ 64, kt :+ 32]"),
        ("shared", "var w: tile<i8>[64, 32] = W[pn * 64 :+ 64, kt :+ 32]"),
    ] {
        kernels.push((what, format!(
            "@launch(256)
            @aligned(K = 32)
            kernel k(A: tensor<i8>[64, K], AS: tensor<f32>[64, KB],
                     W: tensor<i8>[128, K], WS: tensor<f32>[KB, 128],
                     C: tensor<f32>[64, 128]) {{
                let pn = ctx.block.x
                var acc: tile<f32>[64, 64] = 0.0
                for kt in range(0, K, 32) {{
                    let b = kt / 32
                    let a = A[0 :+ 64, kt :+ 32]
                    {w_decl}
                    let ws = WS[b :+ 1, pn * 64 :+ 64]
                    let da = AS[0 :+ 64, b :+ 1]
                    acc += f32(dot_t(a, w)) * ws * da
                }}
                C[0 :+ 64, pn * 64 :+ 64] = acc
            }}"
        )));
    }
    // The activation and its scales as a pipelined loop's stages, read
    // through slices: prefetched through registers where there is no
    // cp.async. The scales are 128 chunks over 256 threads.
    kernels.push((
        "pipelined",
        "@launch(256)
        @pipeline
        @aligned(K = 64, KB = 2)
        kernel k(A: tensor<i8>[64, K], AS: tensor<f32>[64, KB],
                 W: tensor<i8>[128, K], WS: tensor<f32>[KB, 128],
                 C: tensor<f32>[64, 128]) {
            let pn = ctx.block.x
            var acc: tile<f32>[64, 64] = 0.0
            for b in range(0, KB, 2) {
                var at = A[0 :+ 64, b * 32 :+ 64]
                var das = AS[0 :+ 64, b :+ 2]
                let a0 = at[0 :+ 64, 0 :+ 32]
                let a1 = at[0 :+ 64, 32 :+ 32]
                let w0 = W[pn * 64 :+ 64, b * 32 :+ 32]
                let w1 = W[pn * 64 :+ 64, b * 32 + 32 :+ 32]
                let da0 = das[0 :+ 64, 0 :+ 1]
                let da1 = das[0 :+ 64, 1 :+ 1]
                let ws0 = WS[b :+ 1, pn * 64 :+ 64]
                let ws1 = WS[b + 1 :+ 1, pn * 64 :+ 64]
                acc += f32(dot_t(a0, w0)) * ws0 * da0
                acc += f32(dot_t(a1, w1)) * ws1 * da1
            }
            C[0 :+ 64, pn * 64 :+ 64] = acc
        }"
        .to_string(),
    ));
    // Both 128-byte-row tiles swizzled: the stage refilled by its copy, the
    // weight written eight bytes a thread by store, and both read back as
    // fragments by ldmatrix.
    kernels.push((
        "swizzled",
        "@launch(256)
        @swizzle
        @pipeline
        @aligned(K = 128, KB = 4)
        kernel k(A: tensor<i8>[64, K], AS: tensor<f32>[64, KB],
                 W: tensor<i8>[128, K], WS: tensor<f32>[KB, 128],
                 C: tensor<f32>[64, 128]) {
            let pn = ctx.block.x
            var w: tile<i8>[64, 128]
            var acc: tile<f32>[64, 64] = 0.0
            for b in range(0, KB, 4) {
                var at = A[0 :+ 64, b * 32 :+ 128]
                var das = AS[0 :+ 64, b :+ 4]
                for<unroll 4> i in range(0, 4, 1) {
                    let o = ctx.thread + 256 * i
                    let r = o / 16
                    let c = o % 16 * 8
                    store(w[r, c :+ 8], word(W[pn * 64 + r, b * 32 + c :+ 8], 0), word(W[pn * 64 + r, b * 32 + c :+ 8], 1))
                }
                barrier()
                let a0 = at[0 :+ 64, 0 :+ 32]
                let a1 = at[0 :+ 64, 32 :+ 32]
                let a2 = at[0 :+ 64, 64 :+ 32]
                let a3 = at[0 :+ 64, 96 :+ 32]
                let w0 = w[0 :+ 64, 0 :+ 32]
                let w1 = w[0 :+ 64, 32 :+ 32]
                let w2 = w[0 :+ 64, 64 :+ 32]
                let w3 = w[0 :+ 64, 96 :+ 32]
                let ws0 = WS[b :+ 1, pn * 64 :+ 64]
                let da0 = das[0 :+ 64, 0 :+ 1]
                acc += f32(dot_t(a0, w0)) * ws0 * da0
                let ws1 = WS[b + 1 :+ 1, pn * 64 :+ 64]
                let da1 = das[0 :+ 64, 1 :+ 1]
                acc += f32(dot_t(a1, w1)) * ws1 * da1
                let ws2 = WS[b + 2 :+ 1, pn * 64 :+ 64]
                let da2 = das[0 :+ 64, 2 :+ 1]
                acc += f32(dot_t(a2, w2)) * ws2 * da2
                let ws3 = WS[b + 3 :+ 1, pn * 64 :+ 64]
                let da3 = das[0 :+ 64, 3 :+ 1]
                acc += f32(dot_t(a3, w3)) * ws3 * da3
            }
            C[0 :+ 64, pn * 64 :+ 64] = acc
        }"
        .to_string(),
    ));
    // The same unpipelined: the stage's loads issue at the copy and its
    // stores wait for the barrier, past the weight's stores.
    let deferred = kernels.last().expect("the swizzled kernel").1.replace("@pipeline\n        ", "");
    kernels.push(("deferred", deferred));
    for (what, src) in kernels {
        for index_bitwidth in [32, 64] {
            let (ga, gw) = (Arg::new(&a, m, k)?, Arg::new(&w, n, k)?);
            let (gas, gws) = (Arg::new(&asc, m, kb)?, Arg::new(&wsc, kb, n)?);
            let out = Arg::<f32>::zeroed(m, n)?;
            let mut slots = Vec::new();
            ga.push(&mut slots);
            gas.push(&mut slots);
            gw.push(&mut slots);
            gws.push(&mut slots);
            out.push(&mut slots);
            run_at(index_bitwidth, &src, "k", 2, &mut slots)?;
            let got = out.read()?;
            let scale = want.iter().fold(0.0f64, |s, v| s.max(v.abs()));
            let worst = got.iter().zip(&want).map(|(&g, w)| (g as f64 - w).abs()).fold(0.0, f64::max) / scale;
            let name = format!("scaled int8 dot {what} i{index_bitwidth}");
            report(&name, worst < 1e-5, format!("rel err {worst:.2e}"), failures);
        }
    }
    // A minimum subtracted in the registers: `acc -= rs * mw` per block,
    // the activation scale's column times the weight scale's row.
    let mut want_min = want.clone();
    for i in 0..m {
        for j in 0..n {
            for b in 0..kb {
                want_min[i * n + j] -= asc[i * kb + b] as f64 * wsc[b * n + j] as f64;
            }
        }
    }
    let src = "@launch(256)
        @aligned(K = 32)
        kernel k(A: tensor<i8>[64, K], AS: tensor<f32>[64, KB],
                 W: tensor<i8>[128, K], WS: tensor<f32>[KB, 128],
                 C: tensor<f32>[64, 128]) {
            let pn = ctx.block.x
            var acc: tile<f32>[64, 64] = 0.0
            for kt in range(0, K, 32) {
                let b = kt / 32
                let a = A[0 :+ 64, kt :+ 32]
                let w = W[pn * 64 :+ 64, kt :+ 32]
                let ws = WS[b :+ 1, pn * 64 :+ 64]
                let da = AS[0 :+ 64, b :+ 1]
                acc += f32(dot_t(a, w)) * ws * da
                acc -= da * ws
            }
            C[0 :+ 64, pn * 64 :+ 64] = acc
        }";
    for index_bitwidth in [32, 64] {
        let (ga, gw) = (Arg::new(&a, m, k)?, Arg::new(&w, n, k)?);
        let (gas, gws) = (Arg::new(&asc, m, kb)?, Arg::new(&wsc, kb, n)?);
        let out = Arg::<f32>::zeroed(m, n)?;
        let mut slots = Vec::new();
        ga.push(&mut slots);
        gas.push(&mut slots);
        gw.push(&mut slots);
        gws.push(&mut slots);
        out.push(&mut slots);
        run_at(index_bitwidth, src, "k", 2, &mut slots)?;
        let got = out.read()?;
        let scale = want_min.iter().fold(0.0f64, |s, v| s.max(v.abs()));
        let worst = got.iter().zip(&want_min).map(|(&g, w)| (g as f64 - w).abs()).fold(0.0, f64::max) / scale;
        let name = format!("int8 accumulator minus a column times a row i{index_bitwidth}");
        report(&name, worst < 1e-5, format!("rel err {worst:.2e}"), failures);
    }
    // The weight generated a word at a time from `W` itself: as the scaled
    // dot's weight each lane computes its own fragments, and read through a
    // slice it is materialized first. Both are the plain product.
    let generated = |reads: &str| {
        format!(
            "fn wq(j, q) {{
            let v = word(W[pn * 64 + j, kt + 4 * q :+ 4], 0)
            v
        }}

        fn wsf(i, j) {{
            WS[b, pn * 64 + j]
        }}

        fn daf(i, j) {{
            AS[i, b]
        }}

        @launch(256)
        @aligned(K = 32)
        kernel k(A: tensor<i8>[64, K], AS: tensor<f32>[64, KB],
                 W: tensor<i8>[128, K], WS: tensor<f32>[KB, 128],
                 C: tensor<f32>[64, 128]) {{
            let pn = ctx.block.x
            var acc: tile<f32>[64, 64] = 0.0
            for kt in range(0, K, 32) {{
                let b = kt / 32
                let a = A[0 :+ 64, kt :+ 32]
                let w: tile<i8>[64, 32] = generate(wq)
                let ws = WS[b :+ 1, pn * 64 :+ 64]
                let da = AS[0 :+ 64, b :+ 1]
                {reads}
            }}
            C[0 :+ 64, pn * 64 :+ 64] = acc
        }}"
        )
    };
    let forms = [
        ("into its fragments", "acc += f32(dot_t(a, w)) * ws * da", &want),
        ("materialized", "let w0 = w[0 :+ 64, 0 :+ 32]\n                acc += f32(dot_t(a, w0)) * ws * da", &want),
        (
            "with generated scales, less a column times a row",
            "let wsg: tile<f32>[1, 64] = generate(wsf)
                let dag: tile<f32>[64, 1] = generate(daf)
                acc += f32(dot_t(a, w)) * wsg * dag
                acc -= dag * wsg",
            &want_min,
        ),
    ];
    for (what, reads, want) in forms {
        let src = generated(reads);
        for index_bitwidth in [32, 64] {
            let (ga, gw) = (Arg::new(&a, m, k)?, Arg::new(&w, n, k)?);
            let (gas, gws) = (Arg::new(&asc, m, kb)?, Arg::new(&wsc, kb, n)?);
            let out = Arg::<f32>::zeroed(m, n)?;
            let mut slots = Vec::new();
            ga.push(&mut slots);
            gas.push(&mut slots);
            gw.push(&mut slots);
            gws.push(&mut slots);
            out.push(&mut slots);
            run_at(index_bitwidth, &src, "k", 2, &mut slots)?;
            let got = out.read()?;
            let scale = want.iter().fold(0.0f64, |s, v| s.max(v.abs()));
            let worst = got.iter().zip(want).map(|(&g, w)| (g as f64 - w).abs()).fold(0.0, f64::max) / scale;
            let name = format!("generated int8 weight {what} i{index_bitwidth}");
            report(&name, worst < 1e-5, format!("rel err {worst:.2e}"), failures);
        }
    }
    Ok(())
}
