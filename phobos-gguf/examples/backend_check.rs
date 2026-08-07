// The device backend against the host reference, op by op:
//
//   cargo run --release -p phobos-gguf --features cuda --example backend_check
//
// The host implementation defines the semantics, so anything the device does
// differently by more than f32 reduction noise is a bug.

use anyhow::Result;
use phobos_gguf::backend::{Attn, Backend, Buf, DeltaMix, HostBackend, Plane, Rope, read_vec};

use phobos_gguf::backend::device;

fn main() -> Result<()> {
    let gpu = device::DeviceBackend::new()?;
    let host = HostBackend::new();
    // Cells rather than plain locals so the two checkers below can both hold
    // them; a closure that captured them by mutable reference would lock the
    // other one out.
    let worst = std::cell::Cell::new(0.0f32);
    let failures = std::cell::Cell::new(0u32);

    let check_within = |name: &str, tolerance: f32, want: &[f32], got: &[f32]| {
        let error = want
            .iter()
            .zip(got)
            .map(|(w, g)| (w - g).abs() / w.abs().max(1.0))
            .fold(0.0f32, f32::max);
        worst.set(worst.get().max(error));
        // f32 reductions in a different order will not match bit for bit; a layout
        // mistake would be orders of magnitude worse than this.
        let ok = error < tolerance && want.len() == got.len();
        if !ok {
            failures.set(failures.get() + 1);
        }
        println!(
            "{} {name:<34} rel err {error:>10.3e}",
            if ok { "ok  " } else { "FAIL" }
        );
        if !ok {
            let first = want
                .iter()
                .zip(got)
                .position(|(w, g)| (w - g).abs() / w.abs().max(1.0) > tolerance);
            println!("      first bad index {first:?} of {}", want.len());
        }
    };
    // The everyday tolerance: anything that is not a reordered f32 reduction
    // fails by orders of magnitude more than this.
    let check = |name: &str, want: &[f32], got: &[f32]| check_within(name, 1e-4, want, got);

    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 8388608.0 - 1.0
    };

    // Sizes the model actually uses, plus a ragged one.
    let (rows, width) = (5usize, 1024usize);
    let x: Vec<f32> = (0..rows * width).map(|_| next()).collect();
    let gain: Vec<f32> = (0..width).map(|_| next().abs() + 0.5).collect();

    // upload / read round trip
    let on_gpu = gpu.upload(&x)?;
    check("upload+read", &x, &read_vec(&gpu, on_gpu, x.len())?);

    // rms_norm
    let eps = 1e-6f32;
    let run_rms = |b: &dyn Backend| -> Result<Vec<f32>> {
        let xb = b.upload(&x)?;
        let gb = b.upload(&gain)?;
        let out = b.alloc(rows * width)?;
        b.rms_norm(xb, rows, width, gb, eps, out)?;
        read_vec(b, out, rows * width)
    };
    let (rms_want, rms_got) = (run_rms(&host)?, run_rms(&gpu)?);
    check("rms_norm [5 x 1024]", &rms_want, &rms_got);
    let bad_rows: Vec<usize> = (0..rows)
        .filter(|&r| {
            rms_want[r * width..(r + 1) * width]
                .iter()
                .zip(&rms_got[r * width..(r + 1) * width])
                .any(|(w, g)| (w - g).abs() / w.abs().max(1.0) > 1e-4)
        })
        .collect();
    println!("     rms_norm rows disagreeing: {bad_rows:?}");

    // add_into
    let y: Vec<f32> = (0..rows * width).map(|_| next()).collect();
    let run_add = |b: &dyn Backend| -> Result<Vec<f32>> {
        let acc = b.upload(&x)?;
        let add = b.upload(&y)?;
        b.add_into(acc, add)?;
        read_vec(b, acc, x.len())
    };
    check("add_into [3072]", &run_add(&host)?, &run_add(&gpu)?);

    // swiglu, at the FFN width and at a length that is not a tile multiple.
    // The offsets are how the fused gate-and-up projection is read back: the
    // two halves are one buffer, so `at` walks the second one forward.
    for (len, at) in [(rows * 3584, 0), (1000, 0), (1024, 1024)] {
        let g: Vec<f32> = (0..len + at).map(|_| next()).collect();
        let u: Vec<f32> = (0..len + at).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let gb = b.upload(&g)?;
            let ub = b.upload(&u)?;
            let out = b.alloc(len)?;
            b.swiglu(gb, 0, ub, at, out, len)?;
            read_vec(b, out, len)
        };
        check(&format!("swiglu [{len} @ {at}]"), &run(&host)?, &run(&gpu)?);
    }

    // swiglu_planes, which is what a prompt pass takes: the gate and the up half
    // interleave in the fused projection's output, so both are strided. 3584 is
    // a whole row per program and 4608 is not, since three tiles of it overrun
    // the static shared memory a kernel gets.
    for ffn in [3584usize, 4608] {
        let both: Vec<f32> = (0..rows * 2 * ffn).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let src = b.upload(&both)?;
            let out = b.alloc(rows * ffn)?;
            let half = |offset| Plane {
                buf: src,
                offset,
                pitch: 2 * ffn,
            };
            b.swiglu_planes(half(0), half(ffn), out, rows, ffn)?;
            read_vec(b, out, rows * ffn)
        };
        check(
            &format!("swiglu_planes [{rows} x {ffn}]"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // copy, offset on both sides
    let run_copy = |b: &dyn Backend| -> Result<Vec<f32>> {
        let src = b.upload(&x)?;
        let dst = b.alloc(x.len())?;
        b.copy(src, width, dst, 0, width)?;
        read_vec(b, dst, width)
    };
    check("copy [1024 @ offset]", &run_copy(&host)?, &run_copy(&gpu)?);

    // The LM head reads the last row of a [rows, d] buffer.
    let run_last = |b: &dyn Backend| -> Result<Vec<f32>> {
        let src = b.upload(&x)?;
        let dst = b.alloc(width)?;
        b.copy(src, (rows - 1) * width, dst, 0, width)?;
        read_vec(b, dst, width)
    };
    check("copy [last row of 5]", &run_last(&host)?, &run_last(&gpu)?);

    // matmul at the shapes decoding and prefill produce
    // Every (m, k, n) a prefill and a decode step produce.
    for (m, k, n) in [
        (1usize, 1024usize, 2048usize),
        (1, 1024, 16),
        (1, 1024, 4096),
        (1, 1024, 6144),
        (1, 1024, 512),
        (1, 1024, 3584),
        (1, 3584, 1024),
        (1, 2048, 1024),
        (5, 1024, 2048),
        (5, 1024, 16),
        (5, 1024, 4096),
        (5, 1024, 6144),
        (5, 1024, 512),
        (5, 1024, 3584),
        (5, 3584, 1024),
        (5, 2048, 1024),
        // Ragged N and ragged M, which leave the last tile partially outside
        // the tensor in each direction.
        (5, 1024, 40),
        (5, 1024, 2049),
        (5, 1024, 33),
        (33, 1024, 64),
        (33, 1024, 40),
    ] {
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let w: Vec<f32> = (0..k * n).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let ab = b.upload(&a)?;
            let wb = b.upload(&w)?;
            let out = b.alloc(m * n)?;
            b.matmul(ab, m, k, wb, n, out)?;
            read_vec(b, out, m * n)
        };
        check(
            &format!("matmul [{m} x {k} x {n}]"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // The delta rule, which carries eighteen of the model's twenty-four blocks.
    // The state it leaves behind matters as much as the output it returns, so
    // both are compared; a decay that is slightly wrong shows up in the state
    // long before it shows up in one position's readout.
    for (rows, heads, head_dim) in [
        (1usize, 16usize, 128usize),
        (7, 16, 128),
        (64, 4, 32),
        // Prompt depths, where the recurrence runs far enough for the f32
        // accumulation to separate the two backends.
        (48, 16, 128),
        (47, 16, 128),
    ] {
        let n = rows * heads * head_dim;
        let vecs: Vec<Vec<f32>> = (0..3).map(|_| (0..n).map(|_| next()).collect()).collect();
        // Decays live in (0, 1) and betas in (0, 1), as the gates produce them.
        let decay: Vec<f32> = (0..rows * heads).map(|_| next().abs().min(1.0)).collect();
        let beta: Vec<f32> = (0..rows * heads).map(|_| next().abs().min(1.0)).collect();
        let plane = heads * head_dim * head_dim;
        let state0: Vec<f32> = (0..plane).map(|_| next()).collect();

        let mut staged = Vec::new();
        for part in &vecs {
            staged.extend_from_slice(part);
        }
        staged.extend_from_slice(&decay);
        staged.extend_from_slice(&beta);
        let run = |b: &dyn Backend| -> Result<(Vec<f32>, Vec<f32>)> {
            let packed = b.upload(&staged)?;
            let state = b.upload(&state0)?;
            let out = b.alloc(n)?;
            b.delta_rule(packed, rows, heads, head_dim, state, out)?;
            Ok((read_vec(b, out, n)?, read_vec(b, state, plane)?))
        };
        let (host_out, host_state) = run(&host)?;
        let (gpu_out, gpu_state) = run(&gpu)?;
        // The recurrence accumulates over every row in f32 and the device
        // associates it differently from the host, which at 47 rows of 16
        // heads by 128 measures 6.0e-5: the everyday 1e-4 is tighter than the
        // arithmetic at that depth. A layout mistake is orders of magnitude
        // worse than either.
        check_within(
            &format!("delta_rule [{rows} x {heads} x {head_dim}]"),
            3e-4,
            &host_out,
            &gpu_out,
        );
        check_within(
            &format!("delta_rule state [{rows} x {heads} x {head_dim}]"),
            3e-4,
            &host_state,
            &gpu_state,
        );
    }

    // matmul_q8 at the same shapes, against the host reference. The quantized
    // kernel is the one decoding actually runs, so it carries the model.
    for (m, k, n) in [
        (1usize, 1024usize, 2048usize),
        (1, 1024, 16),
        (1, 1024, 6144),
        (1, 1024, 3584),
        (1, 3584, 1024),
        (1, 2048, 1024),
        (2, 1024, 512),
        // From 8 rows up the tensor-core kernel takes over, so these cover the
        // seam: an exact tile, one row short of two, one row past, and an n
        // that does not tile.
        (8, 1024, 512),
        (15, 1024, 512),
        (17, 1024, 512),
        (128, 1024, 2048),
        (8, 1024, 40),
        (33, 1024, 33),
        // From 64 rows up the fused tensor-core kernel takes over, and it hands
        // what it cannot tile to the two below it. These cover that seam: an
        // exact tile, one short, and one that leaves all three kernels
        // something (64 + 32 + 5). The last two have an n the 64-wide column
        // tile does not divide, so no row of them may reach the fused kernel.
        (64, 1024, 512),
        (63, 1024, 512),
        (101, 1024, 512),
        (128, 1024, 2049),
        (101, 1024, 40),
    ] {
        let qs: Vec<i8> = (0..k * n).map(|_| (next() * 127.0) as i8).collect();
        let scales: Vec<f32> = (0..(k / 32) * n).map(|_| next().abs() + 0.01).collect();
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let ab = b.upload(&a)?;
            let wb = b.constant_q8(&format!("q{m}x{k}x{n}"), &qs, &scales, k, n)?;
            let out = b.alloc(m * n)?;
            b.matmul_q8(ab, m, k, wb, n, out)?;
            read_vec(b, out, m * n)
        };
        // Looser than the dense ops on purpose. Quantized values reach 127, so
        // these sums are of magnitude a few thousand, and the kernel groups
        // its additions by 32-element block while the reference runs one flat
        // accumulator over all of k. `q8diag` compares both to f64 and shows
        // the kernel is the more accurate of the two, so matching the
        // reference more tightly than this would be the wrong thing to ask.
        check_within(
            &format!("matmul_q8 [{m} x {k} x {n}]"),
            1e-3,
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // The convolution feeding the delta rule, in both fused layouts. The decode
    // shape is the one to watch: at a single row the carried positions are most
    // of the input, so an off-by-one in the padding still produces a plausible
    // number.
    for (rows, heads, head_dim, interleaved, normalize) in [
        (1usize, 16usize, 128usize, false, true),
        (7, 16, 128, false, true),
        (5, 4, 32, true, true),
        (3, 4, 32, false, false),
        // A prompt, where a program carries eight positions at once rather than
        // one, and a length that only reaches half of that.
        (16, 16, 128, false, true),
        (12, 4, 32, true, true),
        (8, 4, 32, false, false),
    ] {
        let inner = heads * head_dim;
        let (planes, head_stride) = if interleaved {
            ([0, head_dim, 2 * head_dim], 3 * head_dim)
        } else {
            ([0, inner, 2 * inner], head_dim)
        };
        let mix = DeltaMix {
            rows,
            heads,
            head_dim,
            kernel: 4,
            planes,
            head_stride,
            normalize,
            query_scale: (head_dim as f32).sqrt().recip(),
        };
        let history: Vec<f32> = (0..mix.history_len()).map(|_| next()).collect();
        let taps: Vec<f32> = (0..mix.kernel * mix.channels()).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let stream = b.upload(&history)?;
            let weights = b.upload(&taps)?;
            let packed = b.alloc(mix.packed_len())?;
            b.delta_conv(stream, weights, mix, packed)?;
            read_vec(b, packed, 3 * mix.span())
        };
        let layout = if interleaved { "interleaved" } else { "planar" };
        check(
            &format!("delta_conv [{rows} x {heads} x {head_dim}] {layout}"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // The gates. The decay input reaches far enough that a softplus written the
    // obvious way overflows, which is the case worth having a test for.
    for (rows, heads) in [(1usize, 16usize), (9, 16), (64, 4)] {
        let mix = DeltaMix {
            rows,
            heads,
            head_dim: 32,
            kernel: 4,
            planes: [0, 0, 0],
            head_stride: 32,
            normalize: false,
            query_scale: 1.0,
        };
        let mut alpha: Vec<f32> = (0..mix.gates()).map(|_| next() * 8.0).collect();
        alpha[0] = 120.0;
        let beta: Vec<f32> = (0..mix.gates()).map(|_| next() * 8.0).collect();
        let rate: Vec<f32> = (0..heads).map(|_| -next().abs()).collect();
        let bias: Vec<f32> = (0..heads).map(|_| next()).collect();
        let zeros = vec![0.0f32; mix.packed_len()];
        // Both gates come out of one stacked projection, so they arrive as
        // windows of a buffer rather than buffers; the decay sits second here so
        // its offset is not zero either.
        let stacked: Vec<f32> = beta.iter().chain(&alpha).copied().collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let both = b.upload(&stacked)?;
            let (r, d) = (b.upload(&rate)?, b.upload(&bias)?);
            let packed = b.upload(&zeros)?;
            b.delta_gates(both, mix.gates(), both, 0, r, d, mix, packed)?;
            let all = read_vec(b, packed, mix.packed_len())?;
            Ok(all[3 * mix.span()..].to_vec())
        };
        check(
            &format!("delta_gates [{rows} x {heads}]"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // Causal attention against the caches, at the shapes a decode step and a
    // prefill produce. The ragged totals matter: the scan covers whole key tiles
    // and then picks up the last few one at a time, so a total that is a
    // multiple of the tile exercises a different path from one that is not.
    for (rows, start_pos, n_head, n_kv, head_dim) in [
        (1usize, 0usize, 16usize, 8usize, 128usize),
        (1, 63, 16, 8, 128),
        (1, 64, 16, 8, 128),
        (1, 100, 16, 8, 128),
        (29, 0, 16, 8, 128),
        (5, 31, 16, 2, 64),
        (7, 0, 4, 4, 32),
        // The model's own shape, and the one the blocked kernel runs on: a
        // prompt whose length is a whole number of query blocks, and one that
        // leaves a partial block at the end.
        (64, 0, 8, 4, 256),
        (29, 0, 8, 4, 256),
        (1, 40, 8, 4, 256),
        // Decoding deep enough that the key axis is split several ways and one
        // piece has a partial key tile, which is what the merge pass has to
        // rescale correctly. The last one is shorter than the split count, so
        // most pieces see no keys at all.
        (1, 300, 8, 4, 256),
        (1, 511, 8, 4, 256),
        (1, 3, 8, 4, 256),
        // A prompt whose rows, cache and head dimension all tile by 64, which
        // is what the matmul path needs: into an empty cache, onto one that is
        // already a whole number of tiles deep, and a block that is exactly one
        // tile. The 96-row case tiles by 32 but not 64 and must not take it.
        (128, 0, 8, 4, 256),
        (128, 64, 8, 4, 256),
        (64, 0, 8, 4, 256),
        (96, 0, 8, 4, 256),
        // A llama block's shape at eight query heads per key head, decoding and
        // continuing a prompt, against a cache several hundred positions deep.
        // Nothing above reaches both a wide group and a deep cache at once.
        (1, 300, 16, 2, 128),
        (1, 512, 16, 2, 128),
        (1, 600, 16, 2, 128),
        (8, 512, 16, 2, 128),
        (88, 512, 16, 2, 128),
    ] {
        let spec = Attn {
            rows,
            start_pos,
            n_head,
            n_kv,
            head_dim,
        };
        let cached = spec.total() * spec.kv_width();
        let q: Vec<f32> = (0..rows * n_head * head_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..cached).map(|_| next()).collect();
        let v: Vec<f32> = (0..cached).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let (qb, kb, vb) = (b.upload(&q)?, b.upload(&k)?, b.upload(&v)?);
            let out = b.alloc(q.len())?;
            b.attention(qb, kb, vb, spec, out)?;
            read_vec(b, out, q.len())
        };
        check(
            &format!("attention [{rows} @ {start_pos} x {n_head}/{n_kv} x {head_dim}]"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    // The rotary embedding, the strided split, and the output gate.
    for (rows, heads, head_dim, rope_dim, start_pos) in [
        (1usize, 16usize, 128usize, 128usize, 40usize),
        (9, 4, 64, 32, 0),
    ] {
        let table: Vec<f32> = (0..(start_pos + rows) * rope_dim).map(|_| next()).collect();
        let x: Vec<f32> = (0..rows * heads * head_dim).map(|_| next()).collect();
        let spec = Rope {
            heads,
            head_dim,
            rope_dim,
            start_pos,
        };
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let (xb, tb) = (b.upload(&x)?, b.upload(&table)?);
            b.rope(xb, rows, tb, spec)?;
            read_vec(b, xb, x.len())
        };
        check(
            &format!("rope [{rows} x {heads} x {head_dim}/{rope_dim}]"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    for (rows, width, pitch, offset) in
        [(64usize, 128usize, 256usize, 128usize), (5, 2048, 4096, 0)]
    {
        let src: Vec<f32> = (0..rows * pitch).map(|_| next()).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let sb = b.upload(&src)?;
            let db = b.alloc(rows * width)?;
            b.copy_2d(
                Plane {
                    buf: sb,
                    offset,
                    pitch,
                },
                Plane {
                    buf: db,
                    offset: 0,
                    pitch: width,
                },
                rows,
                width,
            )?;
            read_vec(b, db, rows * width)
        };
        check(
            &format!("copy_2d [{rows} x {width} of {pitch}]"),
            &run(&host)?,
            &run(&gpu)?,
        );
    }

    {
        let n = 3000;
        let x: Vec<f32> = (0..n).map(|_| next() * 4.0).collect();
        let g: Vec<f32> = (0..n).map(|_| next() * 4.0).collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>> {
            let (xb, gb) = (b.upload(&x)?, b.upload(&g)?);
            b.gate_into(xb, gb)?;
            read_vec(b, xb, n)
        };
        check("gate_into", &run(&host)?, &run(&gpu)?);
    }

    let _ = Buf(0);
    println!("\nworst relative error {:.3e}", worst.get());
    if failures.get() > 0 {
        anyhow::bail!(
            "{} device ops disagree with the host reference",
            failures.get()
        );
    }
    Ok(())
}
