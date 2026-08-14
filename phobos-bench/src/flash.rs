// The fp32 and fp16 flash-attention benchmarks.

use crate::harness::*;
use crate::*;

/// Flash attention: O = softmax(scale * Q @ K.T) @ V, with the softmax taken
/// row-wise over the Nk keys. A full reference is Nq*Nk*D work, so (as with
/// the large matmul) spot-check a sample of output elements against an f64
/// reference instead.
///
/// The reference replays the kernel's own online-softmax recurrence in f64, so
/// the two agree on algorithm and differ only in precision. Each output is a
/// convex combination of the V rows, hence bounded by max|V| ~ 1; the f32
/// kernel's absolute error is dominated by the Nk-term accumulation
/// (~Nk*eps ~ 2.4e-4) plus the f32 exp. Errors are normalized by
/// max(|want|, 0.1) (an absolute floor, since near-zero averages should not
/// blow up a plain relative test). The @tensorcore path rounds both the
/// scores (Q @ K.T) and P @ V through fp16 fragments, so the tolerance is the
/// looser fp16-grade 2e-2 (the f32 fallback configs clear it comfortably).
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_flash_attention(
    o: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    scale: f64,
    Nq: usize,
    Nk: usize,
    D: usize,
) -> anyhow::Result<()> {
    let (tol, floor) = (2e-2, 1e-1);
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut sample = |limit: usize| {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as usize) % limit
    };
    for s in 0..64 {
        let (i, d) = match s {
            0 => (0, 0),
            1 => (Nq - 1, D - 1),
            _ => (sample(Nq), sample(D)),
        };
        // Online softmax over the keys, mirroring the kernel's recurrence.
        let mut m = f64::NEG_INFINITY;
        let mut l = 0.0f64;
        let mut acc = 0.0f64;
        for j in 0..Nk {
            let mut score = 0.0f64;
            for e in 0..D {
                score += q[i * D + e] as f64 * k[j * D + e] as f64;
            }
            score *= scale;
            let mnew = m.max(score);
            let corr = (m - mnew).exp();
            let p = (score - mnew).exp();
            l = l * corr + p;
            acc = acc * corr + p * v[j * D + d] as f64;
            m = mnew;
        }
        let want = acc / l;
        let got = o[i * D + d] as f64;
        let rel = (got - want).abs() / want.abs().max(floor);
        anyhow::ensure!(
            rel < tol,
            "O[{i},{d}] = {got:e}, want ~ {want:e} (rel err {rel:.2e})"
        );
    }
    Ok(())
}

#[allow(non_snake_case)]
pub(crate) fn bench_flash_attention_fp32(
    stream: &cust::stream::Stream,
    pins: &HashMap<String, i64>,
    results: &mut Results,
) -> anyhow::Result<()> {
    let kernels = phobos_lang::parse(CODE_FLASH)?;
    let space = autotune::pin(phobos_lang::search_space(&kernels[0]), pins)?;

    // D is pinned to 64 by the kernel's @autotune(D in [64]) and must match
    // the static head-dim of the tensors.
    let Nq: i32 = 4096;
    let Nk: i32 = 4096;
    let D: i32 = 64;
    let scale: f32 = 1.0 / (D as f32).sqrt();

    let mut rng = SmallRng::seed_from_u64(42);
    let q: Vec<f32> = (0..(Nq * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();
    let k: Vec<f32> = (0..(Nk * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();
    let v: Vec<f32> = (0..(Nk * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();
    let mut o = vec![0.0f32; (Nq * D) as usize];

    let q_dev = q.as_slice().as_dbuf()?;
    let k_dev = k.as_slice().as_dbuf()?;
    let v_dev = v.as_slice().as_dbuf()?;
    let o_dev = o.as_slice().as_dbuf()?;

    let (q_ptr, k_ptr, v_ptr, o_ptr) = (
        q_dev.as_device_ptr(),
        k_dev.as_device_ptr(),
        v_dev.as_device_ptr(),
        o_dev.as_device_ptr(),
    );

    // Launch the block size the kernel was compiled for: @launch sets a
    // PTX .maxntid, and launching more threads than that is a hard error.
    let block: u32 = kernels[0].cta_threads().map_err(anyhow::Error::msg)? as u32;

    // @tensorcore compiles at 64-bit index (the default mma.sync path), widening
    // the memref descriptor's offset/size/stride params from i32 to i64; the
    // host metadata must match. (Flash attention's dots still run on legacy WMMA
    // via wmma_dot, but the wide index is forced by the kernel attr regardless.)
    let wide = phobos_lang::requires_wide_index(&kernels);

    let mut tuner = autotune::Autotuner {
        code: CODE_FLASH,
        grid_for: |cfg: &[autotune::Setting]| {
            let br = autotune::cfg_value(cfg, "BR")? as u32;
            anyhow::ensure!(
                (Nq as u32).is_multiple_of(br),
                "BR does not divide the query length"
            );
            Ok(autotune::Grid(Nq as u32 / br, 1))
        },
        launch: |module: &cust::module::Module, grid: autotune::Grid| {
            // Each tensor is a memref<?x64>; the kernel writes all of O, so no
            // memset.
            let func = module.get_function("flash_attention")?;
            launch_flash(
                &func,
                grid.0,
                block,
                stream,
                q_ptr.as_raw(),
                k_ptr.as_raw(),
                v_ptr.as_raw(),
                o_ptr.as_raw(),
                Nq,
                Nk,
                D,
                scale,
                wide,
            )
        },
        verify: || {
            o_dev.copy_to(&mut o)?;
            verify_flash_attention(
                &o,
                &q,
                &k,
                &v,
                scale as f64,
                Nq as usize,
                Nk as usize,
                D as usize,
            )
        },
        short_probes: PROBES_SHORT,
        long_probes: PROBES_LONG,
        finalists: 4,
    };

    let winner = tuner.run(&space)?;
    let module = cust::module::Module::from_ptx(winner.ptx.as_str(), &[])?;
    let grid = (tuner.grid_for)(&winner.config)?;

    let (phobos_avg, _) = bench("phobos flash", || {
        (tuner.launch)(&module, grid)?;
        Ok(())
    })?;
    (tuner.verify)()?;

    phinfo!("check: 64 probes, correct");
    // Two Nq*Nk*D matmuls (Q@K.T and P@V), 2 flops each.
    let gflop = 4.0 * Nq as f64 * Nk as f64 * D as f64 / 1e9;
    phinfo!(
        "phobos flash_fp32: {:.1} GFLOP/s",
        gflop / phobos_avg.as_secs_f64()
    );

    results.push(
        "flash_fp32",
        "phobos",
        Precision::F32,
        gflop / phobos_avg.as_secs_f64(),
    );

    Ok(())
}

/// Half-precision flash attention benchmark (examples/flash_attention_fp16.ph):
/// fp16 Q/K/V/O with an f32 online-softmax state, both matmuls on the tensor
/// cores. Inputs/outputs live on the device as fp16 (u16 bit patterns). The
/// reference replays the recurrence in f64 over the fp16-rounded inputs, so the
/// fp16-grade 2e-2 tolerance from the f32 tensor-core path applies.
#[allow(non_snake_case)]
pub(crate) fn bench_flash_attention_fp16(
    stream: &cust::stream::Stream,
    pins: &HashMap<String, i64>,
    results: &mut Results,
) -> anyhow::Result<()> {
    let kernels = phobos_lang::parse(CODE_FLASH_F16)?;
    let space = autotune::pin(phobos_lang::search_space(&kernels[0]), pins)?;

    let Nq: i32 = 4096;
    let Nk: i32 = 4096;
    let D: i32 = 64;
    let scale: f32 = 1.0 / (D as f32).sqrt();

    let mut rng = SmallRng::seed_from_u64(42);
    let q: Vec<f32> = (0..(Nq * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();
    let k: Vec<f32> = (0..(Nk * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();
    let v: Vec<f32> = (0..(Nk * D))
        .map(|_| rng.gen_range(-1.0f32..=1.0))
        .collect();

    // The reference must see the same fp16-rounded inputs the kernel does.
    let qf: Vec<f32> = q.iter().map(|&x| fp16_round(x)).collect();
    let kf: Vec<f32> = k.iter().map(|&x| fp16_round(x)).collect();
    let vf: Vec<f32> = v.iter().map(|&x| fp16_round(x)).collect();

    let q_dev = to_fp16_bits(&q).as_slice().as_dbuf()?;
    let k_dev = to_fp16_bits(&k).as_slice().as_dbuf()?;
    let v_dev = to_fp16_bits(&v).as_slice().as_dbuf()?;
    let o_dev = vec![0u16; (Nq * D) as usize].as_slice().as_dbuf()?;
    let mut o_bits = vec![0u16; (Nq * D) as usize];

    let (q_ptr, k_ptr, v_ptr, o_ptr) = (
        q_dev.as_device_ptr(),
        k_dev.as_device_ptr(),
        v_dev.as_device_ptr(),
        o_dev.as_device_ptr(),
    );

    let block: u32 = kernels[0].cta_threads().map_err(anyhow::Error::msg)? as u32;

    // @tensorcore compiles at 64-bit index (the default mma.sync path); the host
    // descriptor metadata must match (see bench_flash_attention_fp32).
    let wide = phobos_lang::requires_wide_index(&kernels);

    let mut tuner = autotune::Autotuner {
        code: CODE_FLASH_F16,
        grid_for: |cfg: &[autotune::Setting]| {
            let br = autotune::cfg_value(cfg, "BR")? as u32;
            anyhow::ensure!(
                (Nq as u32).is_multiple_of(br),
                "BR does not divide the query length"
            );
            Ok(autotune::Grid(Nq as u32 / br, 1))
        },
        launch: |module: &cust::module::Module, grid: autotune::Grid| {
            let func = module.get_function("flash_attention")?;
            launch_flash(
                &func,
                grid.0,
                block,
                stream,
                q_ptr.as_raw(),
                k_ptr.as_raw(),
                v_ptr.as_raw(),
                o_ptr.as_raw(),
                Nq,
                Nk,
                D,
                scale,
                wide,
            )
        },
        verify: || {
            o_dev.copy_to(&mut o_bits)?;
            let o = from_fp16_bits(&o_bits);
            verify_flash_attention(
                &o,
                &qf,
                &kf,
                &vf,
                scale as f64,
                Nq as usize,
                Nk as usize,
                D as usize,
            )
        },
        short_probes: PROBES_SHORT,
        long_probes: PROBES_LONG,
        finalists: 4,
    };

    let winner = tuner.run(&space)?;
    let module = cust::module::Module::from_ptx(winner.ptx.as_str(), &[])?;
    let grid = (tuner.grid_for)(&winner.config)?;

    let (phobos_avg, _) = bench("phobos flash_fp16", || {
        (tuner.launch)(&module, grid)?;
        Ok(())
    })?;
    (tuner.verify)()?;

    phinfo!("check: 64 probes, correct");
    let gflop = 4.0 * Nq as f64 * Nk as f64 * D as f64 / 1e9;
    phinfo!(
        "phobos flash_fp16: {:.1} GFLOP/s",
        gflop / phobos_avg.as_secs_f64()
    );

    results.push(
        "flash_fp16",
        "phobos",
        Precision::F16TcF32,
        gflop / phobos_avg.as_secs_f64(),
    );

    Ok(())
}
