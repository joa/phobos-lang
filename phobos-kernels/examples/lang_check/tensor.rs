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
    for (what, w_decl) in [
        ("global", "let w = W[pn * 64 :+ 64, kt :+ 32]"),
        ("shared", "var w: tile<i8>[64, 32] = W[pn * 64 :+ 64, kt :+ 32]"),
    ] {
        let src = format!(
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
        );
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
    Ok(())
}
