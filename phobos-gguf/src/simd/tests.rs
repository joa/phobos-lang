use phobos_base::half::f32_to_f16;

use super::*;
use crate::Gguf;
use crate::quant::grouped::{group_rows, grouped_len};
use crate::tests::Builder;

/// A deterministic byte stream.
fn xorshift(seed: u64) -> impl FnMut() -> u8 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 56) as u8
    }
}

/// `n` rows of `k` in `quant`, random quants and scales with small finite
/// headers, which any byte pattern of a K-quant block is otherwise.
fn blocks(quant: Quant, n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut next = xorshift(seed);
    let spec = quant.spec();
    let mut out = Vec::new();
    for _ in 0..n * k / spec.block {
        let mut block: Vec<u8> = (0..spec.block_bytes).map(|_| next()).collect();
        let d = f32_to_f16(0.001 + f32::from(next()) / 4096.0).to_le_bytes();
        let dmin = f32_to_f16(0.0005 + f32::from(next()) / 8192.0).to_le_bytes();
        match quant {
            Quant::Q6_K => block[208..210].copy_from_slice(&d),
            _ => {
                block[..2].copy_from_slice(&d);
                block[2..4].copy_from_slice(&dmin);
            }
        }
        out.extend(block);
    }
    out
}

fn activation(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut next = xorshift(seed);
    (0..rows * k).map(|_| f32::from(next()) / 128.0 - 1.0).collect()
}

/// `[n, rows]` of the dequantized weight against the dequantized Q8
/// activation, in f64: what the integer kernels compute exactly.
fn reference(quant: Quant, weight: &[u8], n: usize, act: &Q8Act) -> Vec<f64> {
    let k = act.k;
    let mut dense = vec![0.0f32; n * k];
    (quant.spec().dequantize)(weight, &mut dense);
    let mut out = vec![0.0; n * act.rows];
    for j in 0..n {
        for r in 0..act.rows {
            out[j * act.rows + r] = (0..k)
                .map(|i| {
                    let (d, qs, _) = act.block(r, i / BLOCK);
                    f64::from(dense[j * k + i]) * f64::from(d[i % BLOCK / RUN]) * f64::from(qs[i % BLOCK])
                })
                .sum();
        }
    }
    out
}

fn check(quant: Quant, wide: bool) {
    let (n, k, rows) = (5, 512, 3);
    let weight = blocks(quant, n, k, 7);
    let act = Q8Act::quantize(&activation(rows, k, 11), rows, k);
    let want = reference(quant, &weight, n, &act);
    let flat = Weight::Flat { bytes: &weight, block_bytes: quant.spec().block_bytes };
    let mut got = vec![0.0f32; n * rows];
    gemm_raw(quant, wide, &flat, n, &act, &mut got, true).unwrap();
    let mut serial = vec![0.0f32; n * rows];
    gemm_raw(quant, wide, &flat, n, &act, &mut serial, false).unwrap();
    assert_eq!(got, serial, "{} threaded against serial", quant.name());
    // The device's layout, with the trailing scale in its plane, reads
    // the same.
    let nb = k / BLOCK;
    let (skip, unit) = quant.device_block();
    let trimmed: Vec<u8> = weight.chunks_exact(quant.spec().block_bytes).flat_map(|b| b[skip..skip + unit].to_vec()).collect();
    let bytes = group_rows(&trimmed, n, nb, unit);
    let planes: Vec<u16> = match quant.spec().raw_scales {
        Some(split) => group_rows(&split(&weight, k, n).d, n, nb, 1),
        None => vec![0; grouped_len(n, nb, 1)],
    };
    let grouped = Weight::Grouped { bytes: &bytes, unit, nb, scales: &planes };
    let mut via_grouped = vec![0.0f32; n * rows];
    gemm_raw(quant, wide, &grouped, n, &act, &mut via_grouped, false).unwrap();
    assert_eq!(got, via_grouped, "{} flat against grouped", quant.name());
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for (j, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!((f64::from(g) - w).abs() <= 1e-5 * scale, "{} output {j}: {g} against {w}", quant.name());
    }
}

#[test]
fn scalar_kernels_match_the_decoders() {
    for quant in [Quant::Q4_K, Quant::Q5_K, Quant::Q6_K] {
        check(quant, false);
    }
}

#[test]
fn avx2_kernels_match_the_decoders() {
    if !avx2() {
        return;
    }
    for quant in [Quant::Q4_K, Quant::Q5_K, Quant::Q6_K] {
        check(quant, true);
    }
}

#[test]
fn the_activation_keeps_its_sums() {
    let (rows, k) = (2, 512);
    let x = activation(rows, k, 3);
    let act = Q8Act::quantize(&x, rows, k);
    for r in 0..rows {
        for b in 0..k / BLOCK {
            let (d, qs, sums) = act.block(r, b);
            for (sum, run) in sums.iter().zip(qs.chunks_exact(SUM_RUN)) {
                assert_eq!(*sum, run.iter().map(|&q| i16::from(q)).sum::<i16>());
            }
            for (i, &q) in qs.iter().enumerate() {
                let (v, d) = (x[r * k + b * BLOCK + i], d[i / RUN]);
                assert!((d * f32::from(q) - v).abs() <= d * 0.5 + 1e-6, "row {r} element {i}: {v} quantized to {q} at {d}");
            }
        }
    }
}

/// Q4_K, Q5_K and Q6_K's ggml type codes.
const CODES: [(Quant, u32); 3] = [(Quant::Q4_K, 12), (Quant::Q5_K, 13), (Quant::Q6_K, 14)];

/// A file holding one block's three expert stacks, gate and up in `gate_up`
/// and down in `down`.
fn expert_file(count: usize, d: usize, d_ff: usize, gate_up: Quant, down: Quant) -> Gguf {
    let code = |quant| CODES.iter().find(|(q, _)| *q == quant).unwrap().1;
    let mut builder = Builder::default();
    builder.kv_string("general.architecture", "qwen35moe");
    for (name, n, k, quant, seed) in [("gate", d_ff, d, gate_up, 1), ("up", d_ff, d, gate_up, 2), ("down", d, d_ff, down, 3)] {
        let bytes = blocks(quant, count * n, k, seed);
        builder.tensor_raw(&format!("blk.0.ffn_{name}_exps.weight"), &[k as u64, n as u64, count as u64], code(quant), &bytes);
    }
    Gguf::from_bytes(builder.build()).unwrap()
}

/// The host reference's arithmetic for one expert: dense weights, f32 dots.
fn reference_ffn(set: &ExpertSet, e: usize, x: &[f32], rows: usize, weights: &[f32]) -> Vec<f32> {
    let (d, d_ff) = (set.down.n(), set.gate.n());
    let mut gate = vec![0.0; d_ff * d];
    let mut up = vec![0.0; d_ff * d];
    let mut down = vec![0.0; d * d_ff];
    set.gate.dequantize(e, &mut gate).unwrap();
    set.up.dequantize(e, &mut up).unwrap();
    set.down.dequantize(e, &mut down).unwrap();
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f32>();
    let mut out = vec![0.0; rows * d];
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let h: Vec<f32> = (0..d_ff)
            .map(|j| {
                let g = dot(&gate[j * d..(j + 1) * d], xr);
                g / (1.0 + (-g).exp()) * dot(&up[j * d..(j + 1) * d], xr)
            })
            .collect();
        for i in 0..d {
            out[r * d + i] = weights[r] * dot(&down[i * d_ff..(i + 1) * d_ff], &h);
        }
    }
    out
}

#[test]
fn an_expert_matches_the_reference_within_the_activation_quantization() {
    for (gate_up, down) in [(Quant::Q4_K, Quant::Q5_K), (Quant::Q4_K, Quant::Q6_K)] {
        let (count, d, d_ff, rows) = (3, 512, 256, 4);
        let gguf = expert_file(count, d, d_ff, gate_up, down);
        let set = ExpertSet::load(&gguf, "blk.0", count, d, d_ff).unwrap();
        assert!(supports(&*set));
        let x = activation(rows, d, 5);
        let weights = [0.5, 1.0, 0.25, 2.0];
        let act = Q8Act::quantize(&x, rows, d);
        let mut scratch = Scratch::default();
        let mut got = vec![0.0; rows * d];
        expert_ffn(&*set, 2, &act, &weights, &mut scratch, &mut got).unwrap();
        let want = reference_ffn(&set, 2, &x, rows, &weights);
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 0.02 * scale, "{}/{} output {i}: {g} against {w}", gate_up.name(), down.name());
        }
        // Accumulates: a second call doubles it.
        expert_ffn(&*set, 2, &act, &weights, &mut scratch, &mut got).unwrap();
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!((g - 2.0 * w).abs() <= 0.04 * scale, "output {i} after two calls: {g} against {}", 2.0 * w);
        }
    }
}

#[test]
fn jobs_over_the_pool_sum_each_experts_rows() {
    let (count, d, d_ff, rows) = (3, 512, 256, 4);
    let gguf = expert_file(count, d, d_ff, Quant::Q4_K, Quant::Q5_K);
    let set = ExpertSet::load(&gguf, "blk.0", count, d, d_ff).unwrap();
    let x = activation(rows, d, 9);
    let jobs = [
        Job { expert: 0, rows: vec![0, 2], weights: vec![0.5, 1.5] },
        Job { expert: 2, rows: vec![1, 2, 3], weights: vec![1.0, 0.25, 2.0] },
        Job { expert: 1, rows: vec![3], weights: vec![0.75] },
    ];
    let mut got = vec![0.0; rows * d];
    experts_ffn(&*set, &jobs, &x, rows, &mut got).unwrap();
    let mut want = vec![0.0f32; rows * d];
    for job in &jobs {
        let xs: Vec<f32> = job.rows.iter().flat_map(|&r| x[r * d..(r + 1) * d].to_vec()).collect();
        let y = reference_ffn(&set, job.expert, &xs, job.rows.len(), &job.weights);
        for (&r, y) in job.rows.iter().zip(y.chunks_exact(d)) {
            for (w, &v) in want[r * d..(r + 1) * d].iter_mut().zip(y) {
                *w += v;
            }
        }
    }
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 0.02 * scale, "output {i}: {g} against {w}");
    }
}
