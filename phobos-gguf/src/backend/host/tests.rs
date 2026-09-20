use phobos_base::half::f32_to_f16;

use super::super::read_vec;
use super::*;
use crate::quant::pack_q8_0;

#[test]
fn host_matmul_matches_by_hand() {
    let backend = HostBackend::new();
    let a = backend.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = backend.upload(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let c = backend.alloc(4).unwrap();
    backend.matmul(a, 2, 3, b, 2, c).unwrap();
    assert_eq!(
        read_vec(&backend, c, 4).unwrap(),
        vec![4.0, 5.0, 10.0, 11.0]
    );
}

#[test]
fn q8_matmul_matches_the_dequantized_matmul() {
    let (k, n) = (Q8_BLOCK * 3, 5usize);
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 8388608.0 - 1.0
    };

    let qs: Vec<i8> = (0..k * n).map(|_| (next() * 127.0) as i8).collect();
    // Rounded through a half, which is how a scale is stored, so the dense
    // reference below is built from the value the packed weight really holds.
    let scales: Vec<f32> = (0..(k / Q8_BLOCK) * n)
        .map(|_| f16_to_f32(f32_to_f16(next().abs() + 0.01)))
        .collect();
    // qs is [n, k]; the dense equivalent is [k, n], so this transposes.
    let mut dense = vec![0.0f32; k * n];
    for j in 0..n {
        for p in 0..k {
            dense[p * n + j] = qs[j * k + p] as f32 * scales[(p / Q8_BLOCK) * n + j];
        }
    }
    // Pre-quantize the activation: matmul_quant requantizes internally, and
    // requantizing an already quantized row is exact, so this isolates the
    // weight layout from the activation's precision loss.
    let mut a: Vec<f32> = (0..2 * k).map(|_| next()).collect();
    let mut scratch = vec![0i8; Q8_BLOCK];
    for block in a.chunks_exact_mut(Q8_BLOCK) {
        let d = quantize_row(block, &mut scratch);
        for (v, &q) in block.iter_mut().zip(&scratch) {
            *v = q as f32 * d;
        }
    }

    let backend = HostBackend::new();
    let x = backend.upload(&a).unwrap();

    let dense_buf = backend.constant("dense", &dense).unwrap();
    let want = backend.alloc(2 * n).unwrap();
    backend.matmul(x, 2, k, dense_buf, n, want).unwrap();

    let packed = pack_q8_0(&qs, &scales, k, n).unwrap();
    let q = backend.constant_quant("q", &packed).unwrap();
    let got = backend.alloc(2 * n).unwrap();
    backend.matmul_quant(x, 2, k, q, n, got).unwrap();

    for (w, g) in read_vec(&backend, want, 2 * n)
        .unwrap()
        .iter()
        .zip(&read_vec(&backend, got, 2 * n).unwrap())
    {
        assert!((w - g).abs() / w.abs().max(1.0) < 1e-5, "{w} vs {g}");
    }
}

#[test]
fn q8_constants_upload_once_and_check_their_shape() {
    let backend = HostBackend::new();
    let qs = vec![1i8; Q8_BLOCK * 2];
    let scales = vec![0.5f32; 2];
    let packed = pack_q8_0(&qs, &scales, Q8_BLOCK, 2).unwrap();
    let a = backend.constant_quant("w", &packed).unwrap();
    let b = backend.constant_quant("w", &packed).unwrap();
    assert_eq!(a, b);

    // k must tile into whole blocks, and the scale count follows from it.
    assert!(pack_q8_0(&qs, &scales, 7, 2).is_err());
    assert!(pack_q8_0(&qs, &[0.5], Q8_BLOCK, 2).is_err());
}

#[test]
fn released_handles_are_reused() {
    let backend = HostBackend::new();
    let first = backend.alloc(8).unwrap();
    backend.release(first);
    let second = backend.alloc(8).unwrap();
    assert_eq!(first, second);
}

#[test]
fn constants_upload_once() {
    let backend = HostBackend::new();
    let a = backend.constant("w", &[1.0, 2.0]).unwrap();
    let b = backend.constant("w", &[9.9, 9.9]).unwrap();
    assert_eq!(a, b);
    assert_eq!(read_vec(&backend, a, 2).unwrap(), vec![1.0, 2.0]);
}

#[test]
fn rms_norm_scales_rows_independently() {
    let backend = HostBackend::new();
    let x = backend.upload(&[3.0, 4.0, 30.0, 40.0]).unwrap();
    let g = backend.upload(&[1.0, 1.0]).unwrap();
    let out = backend.alloc(4).unwrap();
    backend.rms_norm(x, 2, 2, g, 0.0, out).unwrap();
    let got = read_vec(&backend, out, 4).unwrap();
    let expected = 3.0 / (12.5f32).sqrt();
    assert!((got[0] - expected).abs() < 1e-6);
    assert!((got[2] - expected).abs() < 1e-6);
}

#[test]
fn swiglu_and_residual() {
    let backend = HostBackend::new();
    // The gate and the up half as one buffer, as the fused projection
    // hands them over.
    let both = backend.upload(&[0.0, 1.0, 2.0, 3.0]).unwrap();
    let u = backend.upload(&[2.0, 3.0]).unwrap();
    let out = backend.alloc(2).unwrap();
    backend.swiglu(both, 0, both, 2, out, 2).unwrap();
    let got = read_vec(&backend, out, 2).unwrap();
    assert_eq!(got[0], 0.0);
    assert!((got[1] - silu(1.0) * 3.0).abs() < 1e-6);

    backend.add_into(out, u).unwrap();
    assert_eq!(read_vec(&backend, out, 2).unwrap()[0], 2.0);
}

#[test]
fn copy_moves_a_window() {
    let backend = HostBackend::new();
    let src = backend.upload(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let dst = backend.alloc(4).unwrap();
    backend.copy(src, 1, dst, 2, 2).unwrap();
    assert_eq!(
        read_vec(&backend, dst, 4).unwrap(),
        vec![0.0, 0.0, 2.0, 3.0]
    );
}

#[test]
fn delta_conv_leaves_an_all_zero_head_at_zero() {
    // One position and one tap, so the convolution is a multiply by one and
    // the query plane is its input through SiLU and the normalization. A
    // silent head has no norm to divide by.
    let backend = HostBackend::new();
    let mix = DeltaMix {
        rows: 1,
        heads: 2,
        head_dim: 2,
        kv_heads: 2,
        kernel: 1,
        planes: [0, 4, 8],
        head_stride: 2,
        normalize: true,
        query_scale: 1.0,
    };
    let mut stream = vec![0.0f32; mix.channels()];
    stream[0..2].copy_from_slice(&[3.0, 4.0]);
    let history = backend.upload(&stream).unwrap();
    let taps = backend.upload(&vec![1.0; mix.channels()]).unwrap();
    let packed = backend.alloc(mix.packed_len()).unwrap();
    backend.delta_conv(history, taps, mix, packed).unwrap();

    let out = read_vec(&backend, packed, mix.packed_len()).unwrap();
    let norm = (out[0] * out[0] + out[1] * out[1]).sqrt();
    assert!((norm - 1.0).abs() < 1e-6, "loud head normalized to {norm}");
    assert_eq!(&out[2..4], &[0.0, 0.0]);
}

#[test]
fn delta_conv_expands_query_and_key_over_kv_heads() {
    // Four value heads sharing two key/query heads, the grouped-query
    // deltanet shape UD-IQ1_M uses. One position, one tap, no normalization,
    // so this tests only which input column each destination head reads.
    let backend = HostBackend::new();
    let mix = DeltaMix {
        rows: 1,
        heads: 4,
        head_dim: 2,
        kv_heads: 2,
        kernel: 1,
        planes: [0, 4, 8],
        head_stride: 2,
        normalize: false,
        query_scale: 1.0,
    };
    // [q_h0, q_h1, k_h0, k_h1, v_h0, v_h1, v_h2, v_h3], two elements apiece.
    let stream: Vec<f32> = (1..=8).flat_map(|h| [h as f32, h as f32]).collect();
    let history = backend.upload(&stream).unwrap();
    let taps = backend.upload(&vec![1.0; mix.channels()]).unwrap();
    let packed = backend.alloc(mix.packed_len()).unwrap();
    backend.delta_conv(history, taps, mix, packed).unwrap();

    let out = read_vec(&backend, packed, mix.packed_len()).unwrap();
    let head = |plane: usize, h: usize| &out[plane * mix.span() + h * mix.head_dim..][..2];

    // Upstream's `ggml_repeat_4d` tiles rather than block-repeats, so packed
    // head `h` of query/key reads physical head `h % kv_heads` back: heads 0
    // and 2 both read kv-head 0, heads 1 and 3 both read kv-head 1.
    for plane in [0usize, 1] {
        assert_eq!(head(plane, 0), head(plane, 2), "plane {plane} head 0 vs 2");
        assert_eq!(head(plane, 1), head(plane, 3), "plane {plane} head 1 vs 3");
        assert_ne!(head(plane, 0), head(plane, 1), "plane {plane} head 0 vs 1");
    }

    // The value plane has a real head for every one of the four destination
    // slots, so all four stay distinct.
    let v: Vec<&[f32]> = (0..4).map(|h| head(2, h)).collect();
    for (i, a) in v.iter().enumerate() {
        for b in &v[i + 1..] {
            assert_ne!(a, b, "value heads collapsed that should not have");
        }
    }
}

#[test]
fn delta_gates_hold_up_where_the_direct_softplus_overflows() {
    let backend = HostBackend::new();
    let mix = DeltaMix {
        rows: 2,
        heads: 1,
        head_dim: 2,
        kv_heads: 1,
        kernel: 1,
        planes: [0, 2, 4],
        head_stride: 2,
        normalize: false,
        query_scale: 1.0,
    };
    // exp(200) is infinite in f32, so a softplus written as log(1 + exp(x))
    // returns infinity here and the decay comes out as zero or a NaN.
    let decay_in = backend.upload(&[200.0, 0.0]).unwrap();
    let beta_in = backend.upload(&[0.0, 0.0]).unwrap();
    let rate = backend.upload(&[-1.0]).unwrap();
    let bias = backend.upload(&[0.0]).unwrap();
    let packed = backend.alloc(mix.packed_len()).unwrap();
    backend
        .delta_gates(decay_in, 0, beta_in, 0, rate, bias, mix, packed)
        .unwrap();

    let out = read_vec(&backend, packed, mix.packed_len()).unwrap();
    let at = 3 * mix.span();
    // softplus(200) is 200, so the decay is exp(-200), which underflows to
    // zero rather than NaN.
    assert_eq!(out[at], 0.0);
    // softplus(0) is ln(2), so the decay is a half.
    assert!((out[at + 1] - 0.5).abs() < 1e-6);
    assert_eq!(&out[at + 2..at + 4], &[0.5, 0.5]);
}

#[test]
fn softmax_sums_to_one() {
    let mut row = vec![1.0, 2.0, 3.0];
    softmax(&mut row);
    assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(row[2] > row[1] && row[1] > row[0]);
}

#[test]
fn softplus_stays_finite_for_large_inputs() {
    assert!((softplus(0.0) - 2.0f32.ln()).abs() < 1e-6);
    assert_eq!(softplus(100.0), 100.0);
    assert!(softplus(-100.0).abs() < 1e-6);
}

/// The routed feed-forward against the same arithmetic done densely: every
/// expert decoded through `Packed`, each row's choice and weights from
/// `route`, the shared expert scaled by its gate's sigmoid.
#[test]
fn moe_matches_a_dense_reference_and_reports_its_routes() {
    use std::sync::Arc;

    use crate::experts::tests::{Q4_K, q4k_stack};
    use crate::experts::ExpertSet;
    use crate::quant::{Packed, Quant};
    use crate::tests::Builder;

    let (count, used, d, d_ff, rows) = (4usize, 2usize, 256usize, 256usize, 3usize);
    let stacks = [
        ("gate", d_ff, d, q4k_stack(count, d_ff, d, 21)),
        ("up", d_ff, d, q4k_stack(count, d_ff, d, 22)),
        ("down", d, d_ff, q4k_stack(count, d, d_ff, 23)),
    ];
    let mut builder = Builder::default();
    builder.kv_string("general.architecture", "qwen35moe");
    for (name, n, k, bytes) in &stacks {
        builder.tensor_raw(
            &format!("blk.0.ffn_{name}_exps.weight"),
            &[*k as u64, *n as u64, count as u64],
            Q4_K,
            bytes,
        );
    }
    let gguf = crate::Gguf::from_bytes(builder.build()).unwrap();
    let set: Arc<ExpertSet> = ExpertSet::load(&gguf, "blk.0", count, d, d_ff).unwrap();

    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 8388608.0 - 1.0
    };
    let x: Vec<f32> = (0..rows * d).map(|_| next()).collect();
    let logits: Vec<f32> = (0..rows * count).map(|_| 3.0 * next()).collect();
    let shared: Vec<f32> = (0..rows * d).map(|_| next()).collect();
    let gate: Vec<f32> = (0..rows).map(|_| 2.0 * next()).collect();
    let resid: Vec<f32> = (0..rows * d).map(|_| next()).collect();

    // The reference: dense matmuls over `Packed`'s [k, n] decoding.
    let dense = |bytes: &[u8], n: usize, k: usize, e: usize| {
        let per = n * k / 256 * 144;
        Packed::from_bytes(Quant::Q4_K, &bytes[e * per..(e + 1) * per], k, n).unwrap().dense()
    };
    let matvec = |w: &[f32], k: usize, n: usize, v: &[f32]| -> Vec<f32> {
        (0..n).map(|j| (0..k).map(|i| w[i * n + j] * v[i]).sum()).collect()
    };
    let mut want = resid.clone();
    let mut want_routes = Vec::new();
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        for (e, w) in super::super::route(&logits[r * count..(r + 1) * count], used) {
            want_routes.push(e as f32);
            let g = matvec(&dense(&stacks[0].3, d_ff, d, e), d, d_ff, xr);
            let u = matvec(&dense(&stacks[1].3, d_ff, d, e), d, d_ff, xr);
            let h: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
            let y = matvec(&dense(&stacks[2].3, d, d_ff, e), d_ff, d, &h);
            for (acc, v) in want[r * d..(r + 1) * d].iter_mut().zip(y) {
                *acc += w * v;
            }
        }
        let s = 1.0 / (1.0 + (-gate[r]).exp());
        for (acc, &v) in want[r * d..(r + 1) * d].iter_mut().zip(&shared[r * d..(r + 1) * d]) {
            *acc += s * v;
        }
    }

    let backend = HostBackend::new();
    let dest = backend.upload(&resid).unwrap();
    let routes = backend.alloc(rows * used).unwrap();
    let experts = backend.constant_experts("blk.0.experts", &set).unwrap();
    assert_eq!(backend.constant_experts("blk.0.experts", &set).unwrap(), experts);
    backend
        .moe(super::super::Moe {
            x: backend.upload(&x).unwrap(),
            act: None,
            rows,
            d_model: d,
            d_ff,
            logits: backend.upload(&logits).unwrap(),
            n_expert: count,
            n_used: used,
            experts,
            shared: Some((backend.upload(&shared).unwrap(), backend.upload(&gate).unwrap())),
            dest,
            routes: Some(routes),
        })
        .unwrap();
    let got = read_vec(&backend, dest, rows * d).unwrap();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 1e-4 * (1.0 + w.abs()), "element {i}: {g} vs {w}");
    }
    assert_eq!(read_vec(&backend, routes, rows * used).unwrap(), want_routes);
}
