// The language's semantics on the device, one probe kernel per feature,
// each against a host reference:
//
//   cargo run --release -p phobos-kernels --features cuda --example lang_check
//
// The codegen tests check what a construct emits; only a run checks what it
// computes. Every tensor is rank 2, the launch ABI's descriptor.

use anyhow::{Result, ensure};
use cust::memory::DeviceCopy;
use cust::prelude::*;
use phobos_kernels::compile::context_for;
use phobos_kernels::{compile_in, cuda_ok, push_descriptor};

mod stdlib;
mod stdlib_stats;
mod tensor;
use tensor::{dot_epilogue, flash_softmax, scaled_int8_dot};

const THREADS: u32 = 256;

/// A device buffer laid out as `[rows, cols]`.
struct Arg<T: DeviceCopy> {
    buf: DeviceBuffer<T>,
    dims: [i64; 2],
}

impl<T: DeviceCopy + Default + Clone> Arg<T> {
    fn new(host: &[T], rows: usize, cols: usize) -> Result<Self> {
        assert_eq!(host.len(), rows * cols);
        Ok(Arg {
            buf: DeviceBuffer::from_slice(host)?,
            dims: [rows as i64, cols as i64],
        })
    }

    fn zeroed(rows: usize, cols: usize) -> Result<Self> {
        Self::new(&vec![T::default(); rows * cols], rows, cols)
    }

    fn read(&self) -> Result<Vec<T>> {
        let mut out = vec![T::default(); self.buf.len()];
        self.buf.copy_to(&mut out)?;
        Ok(out)
    }

    fn push(&self, slots: &mut Vec<u64>) {
        push_descriptor(slots, self.buf.as_device_ptr().as_raw(), self.dims);
    }
}

/// Compiles `src` and launches its kernel `name` over `blocks` blocks.
fn run(src: &str, name: &str, blocks: u32, slots: &mut [u64]) -> Result<()> {
    run_at(32, src, name, blocks, slots)
}

/// [`run`] at an index width: 64 bits is what reaches mma.sync and cp.async.
fn run_at(index_bitwidth: u32, src: &str, name: &str, blocks: u32, slots: &mut [u64]) -> Result<()> {
    let mut ctx = context_for(&[]);
    ctx.index_bitwidth = index_bitwidth;
    let module = compile_in(&ctx, src, name)?.0;
    let func = module.get_function(name)?.to_raw();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let mut argv: Vec<*mut std::ffi::c_void> = slots
        .iter_mut()
        .map(|s| s as *mut u64 as *mut std::ffi::c_void)
        .collect();
    // SAFETY: argv points into slots, which outlives the launch, in the
    // exploded-memref ABI the kernel was compiled for.
    cuda_ok(
        unsafe {
            cust::sys::cuLaunchKernel(
                func,
                blocks,
                1,
                1,
                THREADS,
                1,
                1,
                0,
                stream.as_inner(),
                argv.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        name,
    )?;
    stream.synchronize()?;
    Ok(())
}

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn report(name: &str, ok: bool, detail: String, failures: &mut u32) {
    if !ok {
        *failures += 1;
    }
    println!("{} {name:<28} {detail}", if ok { "ok  " } else { "FAIL" });
}

fn main() -> Result<()> {
    let _ctx = cust::quick_init()?;
    let mut failures = 0;
    argmax(&mut failures)?;
    integer_rows(&mut failures)?;
    wide_rows(&mut failures)?;
    atomics(&mut failures)?;
    jumps(&mut failures)?;
    pow(&mut failures)?;
    unsigned(&mut failures)?;
    flat_write(&mut failures)?;
    ragged_k(&mut failures)?;
    handoff(&mut failures)?;
    pipelined_tail(&mut failures)?;
    register_tile(&mut failures)?;
    register_rows(&mut failures)?;
    dot_epilogue(&mut failures)?;
    flash_softmax(&mut failures)?;
    scaled_int8_dot(&mut failures)?;
    lanes(&mut failures)?;
    lane_matvec(&mut failures)?;
    restaged_lanes(&mut failures)?;
    lane_runs(&mut failures)?;
    unrolled_sum(&mut failures)?;
    stdlib::activations(&mut failures)?;
    stdlib::row_functions(&mut failures)?;
    stdlib::rope_and_quant(&mut failures)?;
    stdlib::matmuls(&mut failures)?;
    stdlib::attention(&mut failures)?;
    stdlib_stats::columns(&mut failures)?;
    stdlib_stats::losses(&mut failures)?;
    stdlib_stats::random(&mut failures)?;
    // Last: a trap leaves the context unusable.
    aligned_promise(&mut failures)?;
    ensure!(failures == 0, "{failures} probe(s) failed");
    println!("every probe agrees with the host");
    Ok(())
}

fn argmax(failures: &mut u32) -> Result<()> {
    let (rows, cols) = (16, 64);
    // f32 on the warp path; i32 with planted ties, where the first must win.
    let x = noise(rows * cols, 1);
    let ties: Vec<i32> = (0..rows * cols).map(|k| ((k * 7) % 13) as i32).collect();
    for (elem, name) in [("f32", "argmax_f32"), ("i32", "argmax_i32_ties"), ("f16", "argmax_f16")] {
        let src = format!(
            "@launch(256)
            kernel k(X: tensor<{elem}>[16, 64], O: tensor<i32>[16, 1]) {{
                var x = X[0 :+ 16, 0 :+ 64]
                var m: tile<i32>[16, 1] = rowargmax(x)
                O[0 :+ 16, 0 :+ 1] = m
            }}"
        );
        let out = Arg::<i32>::zeroed(rows, 1)?;
        let mut slots = Vec::new();
        let want: Vec<i32> = match elem {
            "i32" => {
                let a = Arg::new(&ties, rows, cols)?;
                a.push(&mut slots);
                out.push(&mut slots);
                run(&src, "k", 1, &mut slots)?;
                ties.chunks(cols).map(|r| first_max(r.iter().map(|&v| v as f64))).collect()
            }
            "f16" => {
                let h: Vec<u16> = x.iter().map(|&v| phobos_base::half::f32_to_f16(v)).collect();
                let a = Arg::new(&h, rows, cols)?;
                a.push(&mut slots);
                out.push(&mut slots);
                run(&src, "k", 1, &mut slots)?;
                h.chunks(cols)
                    .map(|r| first_max(r.iter().map(|&v| phobos_base::half::f16_to_f32(v) as f64)))
                    .collect()
            }
            _ => {
                let a = Arg::new(&x, rows, cols)?;
                a.push(&mut slots);
                out.push(&mut slots);
                run(&src, "k", 1, &mut slots)?;
                x.chunks(cols).map(|r| first_max(r.iter().map(|&v| v as f64))).collect()
            }
        };
        let got = out.read()?;
        report(name, got == want, format!("{got:?}"), failures);
    }
    Ok(())
}

/// One value per thread from `expr`, over `x` (f32) and `i` (an integer
/// with planted ties), against `want(thread)`.
fn lanes(failures: &mut u32) -> Result<()> {
    let n = THREADS as usize;
    let x = noise(n, 7);
    let ints: Vec<i32> = (0..n).map(|t| ((t * 37) % 11) as i32).collect();
    let lane = |t: usize| t % 32;
    let group = |t: usize, w: usize| (t / w) * w..(t / w) * w + w;
    let fold = |r: std::ops::Range<usize>, f: &dyn Fn(f32, f32) -> f32| r.map(|k| x[k]).reduce(f).unwrap();
    let first_max_in = |r: std::ops::Range<usize>, v: &dyn Fn(usize) -> f64| {
        let start = r.start;
        start + first_max(r.map(v)) as usize
    };
    #[allow(clippy::type_complexity)]
    let probes: Vec<(&str, &str, u32, Box<dyn Fn(usize) -> f32>)> = vec![
        ("lane_sum", "sum<over ctx.lane>(x)", 32, Box::new(|t| fold(group(t, 32), &|a, b| a + b))),
        ("lane_max_w8", "max<over ctx.lane, width 8>(x)", 32, Box::new(|t| fold(group(t, 8), &f32::max))),
        ("lane_reduce_fn", "reduce<over ctx.lane>(x, mx)", 32, Box::new(|t| fold(group(t, 32), &f32::max))),
        (
            "lane_argmax",
            "argmax<over ctx.lane>(x)",
            32,
            Box::new(|t| lane(first_max_in(group(t, 32), &|k| x[k] as f64)) as f32),
        ),
        (
            "cta_argmax_ties",
            "argmax<over ctx.thread>(i)",
            32,
            Box::new(|_| first_max_in(0..n, &|k| ints[k] as f64) as f32),
        ),
        (
            "cta_argmax_ties_64",
            "argmax<over ctx.thread>(i)",
            64,
            Box::new(|_| first_max_in(0..n, &|k| ints[k] as f64) as f32),
        ),
        ("cta_sum", "sum<over ctx.thread>(x)", 32, Box::new(|_| fold(0..n, &|a, b| a + b))),
        (
            "lane_any",
            "where(any<over ctx.lane>(x > 0.9), 1.0, 0.0)",
            32,
            Box::new(|t| if group(t, 32).any(|k| x[k] > 0.9) { 1.0 } else { 0.0 }),
        ),
        ("shfl_from", "shfl<from 3>(x)", 32, Box::new(|t| x[t - lane(t) + 3])),
        ("shfl_xor", "shfl<xor 5>(x)", 32, Box::new(|t| x[t ^ 5])),
        // A source past the 16-lane piece reads the lane's own value.
        ("shfl_down_w16", "shfl<down 1, width 16>(x)", 32, Box::new(|t| x[if t % 16 == 15 { t } else { t + 1 }])),
        ("shfl_up", "shfl<up 2>(x)", 32, Box::new(|t| x[if lane(t) < 2 { t } else { t - 2 }])),
    ];
    for (name, expr, bits, want) in probes {
        let src = format!(
            "fn mx(a, b) = max(a, b)

            @launch({THREADS})
            kernel k(X: tensor<f32>[1, {n}], I: tensor<i32>[1, {n}], O: tensor<f32>[1, {n}]) {{
                let x = X[0, ctx.thread]
                let i = I[0, ctx.thread]
                O[0, ctx.thread] = f32({expr})
            }}"
        );
        let (xa, ia, out) = (Arg::new(&x, 1, n)?, Arg::new(&ints, 1, n)?, Arg::<f32>::zeroed(1, n)?);
        let mut slots = Vec::new();
        xa.push(&mut slots);
        ia.push(&mut slots);
        out.push(&mut slots);
        run_at(bits, &src, "k", 1, &mut slots)?;
        let got = out.read()?;
        let worst = (0..n).map(|t| (got[t] - want(t)).abs()).fold(0.0f32, f32::max);
        report(name, worst <= 1e-5, format!("worst {worst:.2e}"), failures);
    }
    Ok(())
}

/// A warp per output column, its lanes striding k, summed with a shuffle:
/// per-thread loads, a per-thread `var` and a per-lane store.
fn lane_matvec(failures: &mut u32) -> Result<()> {
    let (n, k) = (40usize, 1003usize);
    let a = noise(k, 11);
    let w = noise(n * k, 12);
    let src = format!(
        "@launch({THREADS})
        kernel mv(A: tensor<f32>[1, K], W: tensor<f32>[N, K], C: tensor<f32>[1, N]) {{
            let col = ctx.block.x * ctx.warps + ctx.warp
            if col < N {{
                var acc: f32 = 0.0
                for j in range(ctx.lane, K, ctx.lanes) {{
                    acc += W[col, j] * A[0, j]
                }}
                let total = sum<over ctx.lane>(acc)
                if ctx.lane == 0 {{
                    C[0, col] = total
                }}
            }}
        }}"
    );
    let (aa, wa, out) = (Arg::new(&a, 1, k)?, Arg::new(&w, n, k)?, Arg::<f32>::zeroed(1, n)?);
    let mut slots = Vec::new();
    aa.push(&mut slots);
    wa.push(&mut slots);
    out.push(&mut slots);
    let warps = THREADS as usize / 32;
    run(&src, "mv", n.div_ceil(warps) as u32, &mut slots)?;
    let got = out.read()?;
    let worst = (0..n)
        .map(|c| {
            let want: f64 = (0..k).map(|j| w[c * k + j] as f64 * a[j] as f64).sum();
            (got[c] as f64 - want).abs()
        })
        .fold(0.0, f64::max);
    report("lane_matvec", worst < 1e-4, format!("worst {worst:.2e}"), failures);
    Ok(())
}

/// A tile staged once per turn and read by lane loads from other threads than
/// the copy that wrote it. Without a barrier ahead of the next turn's copy,
/// a warp could overwrite the tile while another still reads it.
fn restaged_lanes(failures: &mut u32) -> Result<()> {
    let (n, k) = (1000usize, 4096usize);
    let x = noise(k, 21);
    let w = noise(n * k, 22);
    let src = format!(
        "@launch({THREADS})
        @aligned(K = 128)
        kernel mv(W: tensor<f32>[N, K], X: tensor<f32>[1, K], Y: tensor<f32>[1, N]) {{
            let row = ctx.block.x * ctx.warps + ctx.warp
            var acc: f32 = 0.0
            for kt in range(0, K, 128) {{
                var xs = X[0 :+ 1, kt :+ 128]
                let at = ctx.lane * 4
                if row < N {{
                    acc += sum(W[row, kt + at :+ 4] * xs[0, at :+ 4])
                }}
            }}
            let total = sum<over ctx.lane>(acc)
            if ctx.lane == 0 && row < N {{
                Y[0, row] = total
            }}
        }}"
    );
    let (wa, xa, out) = (Arg::new(&w, n, k)?, Arg::new(&x, 1, k)?, Arg::<f32>::zeroed(1, n)?);
    let mut slots = Vec::new();
    wa.push(&mut slots);
    xa.push(&mut slots);
    out.push(&mut slots);
    let warps = THREADS as usize / 32;
    run(&src, "mv", n.div_ceil(warps) as u32, &mut slots)?;
    let got = out.read()?;
    let worst = (0..n)
        .map(|r| {
            let want: f64 = (0..k).map(|j| w[r * k + j] as f64 * x[j] as f64).sum();
            (got[r] as f64 - want).abs()
        })
        .fold(0.0, f64::max);
    report("restaged_lanes", worst < 1e-3, format!("worst {worst:.2e}"), failures);
    Ok(())
}

/// Each thread doubles its run of four into a shared tile, and past a
/// `barrier()` reads its neighbour's: the barrier is what makes the
/// neighbour's stores visible.
fn lane_runs(failures: &mut u32) -> Result<()> {
    let n = THREADS as usize * 4;
    let x = noise(n, 13);
    let src = format!(
        "@launch({THREADS})
        @aligned(N = {n})
        kernel runs(X: tensor<f32>[1, N], Y: tensor<f32>[1, N]) {{
            var t: tile<f32>[1, {n}] = 0.0
            let e = 4 * ctx.thread
            store(t[0, e :+ 4], 2.0 * elem(X[0, e :+ 4], 0), 2.0 * elem(X[0, e :+ 4], 1),
                  2.0 * elem(X[0, e :+ 4], 2), 2.0 * elem(X[0, e :+ 4], 3))
            barrier()
            let o = (e + 4) % N
            store(Y[0, e :+ 4], elem(t[0, o :+ 4], 0), elem(t[0, o :+ 4], 1), elem(t[0, o :+ 4], 2),
                  elem(t[0, o :+ 4], 3))
        }}"
    );
    let (xa, out) = (Arg::new(&x, 1, n)?, Arg::<f32>::zeroed(1, n)?);
    let mut slots = Vec::new();
    xa.push(&mut slots);
    out.push(&mut slots);
    run(&src, "runs", 1, &mut slots)?;
    let got = out.read()?;
    let bad = (0..n).filter(|&i| got[i] != 2.0 * x[(i + 4) % n]).count();
    report("lane_runs", bad == 0, format!("{bad} of {n} wrong"), failures);
    Ok(())
}

/// Each thread sums its strided elements with `for<unroll 3>`: a dynamic
/// length that leaves every thread a remainder after the whole turns.
fn unrolled_sum(failures: &mut u32) -> Result<()> {
    let n = 1000;
    let x = noise(n, 14);
    let src = format!(
        "@launch({THREADS})
        kernel unrolled(X: tensor<f32>[1, N], O: tensor<f32>[1, T]) {{
            var acc: f32 = 0.0
            for<unroll 3> i in range(ctx.thread, N, {THREADS}) {{
                acc += X[0, i]
            }}
            O[0, ctx.thread] = acc
        }}"
    );
    let threads = THREADS as usize;
    let (xa, out) = (Arg::new(&x, 1, n)?, Arg::<f32>::zeroed(1, threads)?);
    let mut slots = Vec::new();
    xa.push(&mut slots);
    out.push(&mut slots);
    run(&src, "unrolled", 1, &mut slots)?;
    let got = out.read()?;
    let worst = (0..threads)
        .map(|t| {
            let want: f32 = (t..n).step_by(threads).map(|i| x[i]).sum();
            (got[t] - want).abs()
        })
        .fold(0.0f32, f32::max);
    report("unrolled_sum", worst < 1e-5, format!("worst {worst:.2e}"), failures);
    Ok(())
}

fn first_max(row: impl Iterator<Item = f64>) -> i32 {
    let mut best = (f64::NEG_INFINITY, 0);
    for (j, v) in row.enumerate() {
        if v > best.0 {
            best = (v, j as i32);
        }
    }
    best.1
}

fn integer_rows(failures: &mut u32) -> Result<()> {
    let (rows, cols) = (16, 64);
    let x: Vec<i32> = (0..rows * cols).map(|k| ((k * 37) % 101) as i32 - 50).collect();
    let src = "@launch(256)
        kernel k(X: tensor<i32>[16, 64], S: tensor<i32>[16, 1], M: tensor<i32>[16, 1]) {
            var x = X[0 :+ 16, 0 :+ 64]
            var s: tile<i32>[16, 1] = rowsum(x)
            var m: tile<i32>[16, 1] = rowmax(x)
            S[0 :+ 16, 0 :+ 1] = s
            M[0 :+ 16, 0 :+ 1] = m
        }";
    let a = Arg::new(&x, rows, cols)?;
    let (s, m) = (Arg::<i32>::zeroed(rows, 1)?, Arg::<i32>::zeroed(rows, 1)?);
    let mut slots = Vec::new();
    a.push(&mut slots);
    s.push(&mut slots);
    m.push(&mut slots);
    run(src, "k", 1, &mut slots)?;
    let want_s: Vec<i32> = x.chunks(cols).map(|r| r.iter().sum()).collect();
    let want_m: Vec<i32> = x.chunks(cols).map(|r| *r.iter().max().unwrap()).collect();
    report("rowsum i32 (warp)", s.read()? == want_s, String::new(), failures);
    report("rowmax i32 (warp)", m.read()? == want_m, String::new(), failures);
    Ok(())
}

fn wide_rows(failures: &mut u32) -> Result<()> {
    // Few rows, many columns: several warps share each row.
    for (rows, cols) in [(1usize, 2560usize), (2, 1024)] {
        let x = noise(rows * cols, 11);
        let xi: Vec<i32> = (0..rows * cols).map(|k| ((k * 37) % 101) as i32 - 50).collect();
        let src = format!(
            "@launch(256)
            kernel k(X: tensor<f32>[{rows}, {cols}], I: tensor<i32>[{rows}, {cols}], S: tensor<f32>[{rows}, 1], M: tensor<f32>[{rows}, 1], T: tensor<i32>[{rows}, 1]) {{
                var x = X[0 :+ {rows}, 0 :+ {cols}]
                var i = I[0 :+ {rows}, 0 :+ {cols}]
                var s: tile<f32>[{rows}, 1] = rowsum(x)
                var m: tile<f32>[{rows}, 1] = rowmax(abs(x))
                var t: tile<i32>[{rows}, 1] = rowsum(i)
                S[0 :+ {rows}, 0 :+ 1] = s
                M[0 :+ {rows}, 0 :+ 1] = m
                T[0 :+ {rows}, 0 :+ 1] = t
            }}"
        );
        let (gx, gi) = (Arg::new(&x, rows, cols)?, Arg::new(&xi, rows, cols)?);
        let (s, m, t) = (Arg::<f32>::zeroed(rows, 1)?, Arg::<f32>::zeroed(rows, 1)?, Arg::<i32>::zeroed(rows, 1)?);
        let mut slots = Vec::new();
        for push in [&gx as &dyn Push, &gi, &s, &m, &t] {
            push.push_to(&mut slots);
        }
        run(&src, "k", 1, &mut slots)?;
        let want_s: Vec<f32> = x.chunks(cols).map(|r| r.iter().sum()).collect();
        let want_m: Vec<f32> = x.chunks(cols).map(|r| r.iter().fold(0.0f32, |a, v| a.max(v.abs()))).collect();
        let want_t: Vec<i32> = xi.chunks(cols).map(|r| r.iter().sum()).collect();
        let sum_err = s.read()?.iter().zip(&want_s).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        let ok = sum_err < 1e-3 && m.read()? == want_m && t.read()? == want_t;
        report(&format!("{rows} x {cols} rows, warps"), ok, format!("sum err {sum_err:.2e}"), failures);
    }
    Ok(())
}

fn atomics(failures: &mut u32) -> Result<()> {
    // Every block takes a ticket and races for one lock.
    let blocks = 8;
    let src = "@launch(256)
        kernel k(C: tensor<i32>[2, 1], L: tensor<i32>[1, 1], O: tensor<i32>[8, 2]) {
            let p = ctx.block.x
            O[p, 0] = atomic_add(C, 0, 1)
            O[p, 1] = atomic_cas(L, 0, 0, p + 1)
        }";
    let c = Arg::<i32>::zeroed(2, 1)?;
    let l = Arg::<i32>::zeroed(1, 1)?;
    let o = Arg::<i32>::zeroed(blocks, 2)?;
    let mut slots = Vec::new();
    c.push(&mut slots);
    l.push(&mut slots);
    o.push(&mut slots);
    run(src, "k", blocks as u32, &mut slots)?;
    let (c, l, o) = (c.read()?, l.read()?, o.read()?);
    let mut tickets: Vec<i32> = o.chunks(2).map(|r| r[0]).collect();
    tickets.sort();
    let once = c[0] == blocks as i32 && tickets == (0..blocks as i32).collect::<Vec<_>>();
    report("atomic_add once per CTA", once, format!("count {} tickets {tickets:?}", c[0]), failures);
    let winners: Vec<usize> = o.chunks(2).enumerate().filter(|(_, r)| r[1] == 0).map(|(p, _)| p).collect();
    let won = winners.len() == 1 && l[0] == winners[0] as i32 + 1;
    let losers_saw = o.chunks(2).all(|r| r[1] == 0 || r[1] == l[0]);
    report("atomic_cas one winner", won && losers_saw, format!("lock {} winners {winners:?}", l[0]), failures);
    Ok(())
}

fn jumps(failures: &mut u32) -> Result<()> {
    let cols = 64;
    let x = noise(cols, 3);
    // The first j with x > 0.5, the count of non-negatives, and a block that
    // returns before its write.
    let src = "@launch(256)
        kernel k(X: tensor<f32>[1, 64], O: tensor<i32>[2, 3]) {
            let p = ctx.block.x
            var found = -1
            var j = 0
            while j < 64 {
                if X[0, j] > 0.5 {
                    found = j
                    break
                }
                j = j + 1
            }
            var count = 0
            for k in range(0, 64) {
                if X[0, k] < 0.0 {
                    continue
                }
                count = count + 1
            }
            O[p, 0] = found
            O[p, 1] = count
            if p == 1 {
                return
            }
            O[p, 2] = 7
        }";
    let a = Arg::new(&x, 1, cols)?;
    let o = Arg::new(&[-9i32; 6], 2, 3)?;
    let mut slots = Vec::new();
    a.push(&mut slots);
    o.push(&mut slots);
    run(src, "k", 2, &mut slots)?;
    let got = o.read()?;
    let found = x.iter().position(|&v| v > 0.5).map_or(-1, |j| j as i32);
    let count = x.iter().filter(|&&v| v >= 0.0).count() as i32;
    let want = vec![found, count, 7, found, count, -9];
    report("break, continue, return", got == want, format!("{got:?} want {want:?}"), failures);
    Ok(())
}

fn pow(failures: &mut u32) -> Result<()> {
    let x: Vec<f32> = noise(16 * 64, 4).iter().map(|v| v.abs() * 4.0 + 0.01).collect();
    let src = "@launch(256)
        kernel k(X: tensor<f32>[16, 64], Y: tensor<f32>[16, 64]) {
            var x = X[0 :+ 16, 0 :+ 64]
            var y: tile<f32>[16, 64] = pow(x, 2.5)
            Y[0 :+ 16, 0 :+ 64] = y
        }";
    let a = Arg::new(&x, 16, 64)?;
    let y = Arg::<f32>::zeroed(16, 64)?;
    let mut slots = Vec::new();
    a.push(&mut slots);
    y.push(&mut slots);
    run(src, "k", 1, &mut slots)?;
    let worst = y
        .read()?
        .iter()
        .zip(&x)
        .map(|(g, v)| (g - v.powf(2.5)).abs() / v.powf(2.5))
        .fold(0.0f32, f32::max);
    report("pow", worst < 1e-4, format!("rel err {worst:.3e}"), failures);
    Ok(())
}

fn unsigned(failures: &mut u32) -> Result<()> {
    let b: Vec<u8> = (0..16 * 64).map(|k| (k * 29 % 256) as u8).collect();
    let src = "@launch(256)
        kernel k(B: tensor<u8>[16, 64], Q: tensor<i32>[16, 64], F: tensor<f32>[16, 64], S: tensor<i32>[1, 64],
                 G: tensor<f32>[16, 64]) {
            var b = B[0 :+ 16, 0 :+ 64]
            var q: tile<i32>[16, 64] = (i32(b) >> 4) & 15
            var f: tile<f32>[16, 64] = f32(b)
            Q[0 :+ 16, 0 :+ 64] = q
            F[0 :+ 16, 0 :+ 64] = f
            G[0 :+ 16, 0 :+ 64] = f32(i32(b) / 16 % 4) * 0.5
            for j in range(0, 64) {
                S[0, j] = B[15, j] + 1
            }
        }";
    let a = Arg::new(&b, 16, 64)?;
    let q = Arg::<i32>::zeroed(16, 64)?;
    let f = Arg::<f32>::zeroed(16, 64)?;
    let s = Arg::<i32>::zeroed(1, 64)?;
    let g = Arg::<f32>::zeroed(16, 64)?;
    let mut slots = Vec::new();
    for push in [&a as &dyn Push, &q, &f, &s, &g] {
        push.push_to(&mut slots);
    }
    run(src, "k", 1, &mut slots)?;
    let want_q: Vec<i32> = b.iter().map(|&v| i32::from(v >> 4)).collect();
    let want_f: Vec<f32> = b.iter().map(|&v| f32::from(v)).collect();
    let want_s: Vec<i32> = b[15 * 64..].iter().map(|&v| i32::from(v) + 1).collect();
    report("u8 nibble decode", q.read()? == want_q, String::new(), failures);
    report("u8 to f32", f.read()? == want_f, String::new(), failures);
    report("u8 scalar load", s.read()? == want_s, String::new(), failures);
    // Integer arithmetic under a float conversion stays integer.
    let want_g: Vec<f32> = b.iter().map(|&v| f32::from(v / 16 % 4) * 0.5).collect();
    report("integer under f32()", g.read()? == want_g, String::new(), failures);
    Ok(())
}

fn flat_write(failures: &mut u32) -> Result<()> {
    let x = noise(16 * 16, 5);
    let src = "@launch(256)
        kernel k(X: tensor<f32>[16, 16], O: tensor<f32>[16, 16]) {
            var t: tile<f32>[16, 16] = X[0 :+ 16, 0 :+ 16]
            let f = flat(t)
            f[0 :+ 1, 0 :+ 128] = 0.0
            O[0 :+ 16, 0 :+ 16] = t
        }";
    let a = Arg::new(&x, 16, 16)?;
    let o = Arg::<f32>::zeroed(16, 16)?;
    let mut slots = Vec::new();
    a.push(&mut slots);
    o.push(&mut slots);
    run(src, "k", 1, &mut slots)?;
    let want: Vec<f32> = x.iter().enumerate().map(|(k, &v)| if k < 128 { 0.0 } else { v }).collect();
    report("write through flat()", o.read()? == want, String::new(), failures);
    Ok(())
}

fn ragged_k(failures: &mut u32) -> Result<()> {
    // K = 40 against a 16-wide k-slice: the last slice is partial, on the
    // tensor cores and off them.
    let (m, k, n) = (64, 40, 64);
    let a = noise(m * k, 6);
    let b = noise(k * n, 7);
    let round = |v: f32| phobos_base::half::f16_to_f32(phobos_base::half::f32_to_f16(v));
    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            want[i * n + j] = (0..k).map(|p| round(a[i * k + p]) * round(b[p * n + j])).sum();
        }
    }
    // Static K, then K known only at launch, with and without @pipeline.
    for (attrs, kdim, name, bits) in [
        ("@tensorcore", "40", "ragged k, tensor cores", 32),
        ("", "40", "ragged k, fma", 32),
        ("@tensorcore", "K", "ragged dynamic k, tc", 32),
        ("@tensorcore @pipeline", "K", "ragged k, tc pipelined", 32),
        ("@tensorcore", "40", "ragged k, mma.sync", 64),
        ("@tensorcore @pipeline", "K", "ragged k, mma.sync pipelined", 64),
    ] {
        let src = format!(
            "@launch(256)
            {attrs}
            kernel k(A: tensor<f32>[64, {kdim}], B: tensor<f32>[{kdim}, 64], C: tensor<f32>[64, 64]) {{
                var acc: tile<f32>[64, 64] = 0.0
                for kt in range(0, {kdim}, 16) {{
                    var a = A[0 :+ 64, kt :+ 16]
                    var b = B[kt :+ 16, 0 :+ 64]
                    acc += dot(a, b)
                }}
                C[0 :+ 64, 0 :+ 64] = acc
            }}"
        );
        let (ga, gb) = (Arg::new(&a, m, k)?, Arg::new(&b, k, n)?);
        let c = Arg::<f32>::zeroed(m, n)?;
        let mut slots = Vec::new();
        ga.push(&mut slots);
        gb.push(&mut slots);
        c.push(&mut slots);
        run_at(bits, &src, "k", 1, &mut slots)?;
        let worst = c.read()?.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        report(name, worst < 1e-2, format!("max abs err {worst:.3e}"), failures);
    }
    Ok(())
}

fn register_tile(failures: &mut u32) -> Result<()> {
    // A sum of squares over k held in registers, then stored and reduced.
    let (rows, k) = (16, 192);
    let x = noise(rows * k, 14);
    let src = "@launch(256)
        @registers(acc)
        kernel k(X: tensor<f32>[16, K], O: tensor<f32>[16, 64], S: tensor<f32>[16, 1]) {
            var acc: tile<f32>[16, 64] = 0.0
            for kt in range(0, K, 64) {
                var x = X[0 :+ 16, kt :+ 64]
                acc += x * x
            }
            O[0 :+ 16, 0 :+ 64] = acc
            var s: tile<f32>[16, 1] = rowsum(acc)
            S[0 :+ 16, 0 :+ 1] = s
        }";
    let mut want = vec![0.0f32; rows * 64];
    for r in 0..rows {
        for j in 0..k {
            want[r * 64 + j % 64] += x[r * k + j] * x[r * k + j];
        }
    }
    let want_s: Vec<f32> = want.chunks(64).map(|r| r.iter().sum()).collect();
    let gx = Arg::new(&x, rows, k)?;
    let (o, s) = (Arg::<f32>::zeroed(rows, 64)?, Arg::<f32>::zeroed(rows, 1)?);
    let mut slots = Vec::new();
    gx.push(&mut slots);
    o.push(&mut slots);
    s.push(&mut slots);
    run(src, "k", 1, &mut slots)?;
    let err = |got: Vec<f32>, want: &[f32]| got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
    let (eo, es) = (err(o.read()?, &want), err(s.read()?, &want_s));
    report("@registers accumulator", eo < 1e-5 && es < 1e-4, format!("err {eo:.1e} / {es:.1e}"), failures);
    Ok(())
}

/// Row sums and maxima of a register tile whose rows are 16 lanes of a
/// warp, folded with shuffles where the elements live.
fn register_rows(failures: &mut u32) -> Result<()> {
    let (rows, k) = (64, 48);
    let x = noise(rows * k, 15);
    for (how, name) in [("rowsum", "@registers rowsum, shuffles"), ("rowmax", "@registers rowmax, shuffles")] {
        let src = format!(
            "@launch(256)
            @registers(acc)
            kernel k(X: tensor<f32>[64, K], S: tensor<f32>[64, 1]) {{
                var acc: tile<f32>[64, 16] = 0.0
                for kt in range(0, K, 16) {{
                    var x = X[0 :+ 64, kt :+ 16]
                    acc += x * x
                }}
                var s: tile<f32>[64, 1] = {how}(acc)
                S[0 :+ 64, 0 :+ 1] = s
            }}"
        );
        let mut acc = vec![0.0f32; rows * 16];
        for r in 0..rows {
            for j in 0..k {
                acc[r * 16 + j % 16] += x[r * k + j] * x[r * k + j];
            }
        }
        let want: Vec<f32> = acc
            .chunks(16)
            .map(|r| if how == "rowsum" { r.iter().sum() } else { r.iter().copied().fold(f32::MIN, f32::max) })
            .collect();
        let gx = Arg::new(&x, rows, k)?;
        let s = Arg::<f32>::zeroed(rows, 1)?;
        let mut slots = Vec::new();
        gx.push(&mut slots);
        s.push(&mut slots);
        run(&src, "k", 1, &mut slots)?;
        let err = s.read()?.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        report(name, err < 1e-4, format!("err {err:.1e}"), failures);
    }
    Ok(())
}

fn pipelined_tail(failures: &mut u32) -> Result<()> {
    // K = 72 against 32-wide stages: two whole tiles pipelined, then 8
    // columns of remainder, masked.
    let (rows, k) = (64, 72);
    let (a, b) = (noise(rows * k, 12), noise(rows * k, 13));
    let src = "@launch(256)
        @pipeline
        kernel k(A: tensor<f32>[64, K], B: tensor<f32>[64, K], C: tensor<f32>[64, 32]) {
            var acc: tile<f32>[64, 32] = 0.0
            for kt in range(0, K, 32) {
                var a = A[0 :+ 64, kt :+ 32]
                var b = B[0 :+ 64, kt :+ 32]
                acc += a * b
            }
            C[0 :+ 64, 0 :+ 32] = acc
        }";
    let mut want = vec![0.0f32; rows * 32];
    for r in 0..rows {
        for j in 0..k {
            want[r * 32 + j % 32] += a[r * k + j] * b[r * k + j];
        }
    }
    for bits in [32, 64] {
        let (ga, gb) = (Arg::new(&a, rows, k)?, Arg::new(&b, rows, k)?);
        let c = Arg::<f32>::zeroed(rows, 32)?;
        let mut slots = Vec::new();
        ga.push(&mut slots);
        gb.push(&mut slots);
        c.push(&mut slots);
        run_at(bits, src, "k", 1, &mut slots)?;
        let worst = c.read()?.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        report(&format!("pipelined ragged k, {bits}-bit"), worst < 1e-5, format!("max abs err {worst:.2e}"), failures);
    }
    Ok(())
}

fn handoff(failures: &mut u32) -> Result<()> {
    // The flag is raised before the launch, so the wait passes at once; the
    // kernel adds the row and lowers the flag.
    let src = "@launch(256)
        kernel k(F: tensor<i32>[1, 1], Y: tensor<f32>[1, 64], D: tensor<f32>[1, 64]) {
            while load_acquire(F, 0) == 0 {
            }
            var y = Y[0 :+ 1, 0 :+ 64]
            var d = D[0 :+ 1, 0 :+ 64]
            D[0 :+ 1, 0 :+ 64] = d + y
            store_release(F, 0, 0)
        }";
    let (y, d0) = (noise(64, 9), noise(64, 10));
    let f = Arg::new(&[1i32], 1, 1)?;
    let (gy, gd) = (Arg::new(&y, 1, 64)?, Arg::new(&d0, 1, 64)?);
    let mut slots = Vec::new();
    f.push(&mut slots);
    gy.push(&mut slots);
    gd.push(&mut slots);
    run(src, "k", 1, &mut slots)?;
    let want: Vec<f32> = d0.iter().zip(&y).map(|(a, b)| a + b).collect();
    let ok = gd.read()? == want && f.read()? == [0];
    report("acquire, add, release", ok, String::new(), failures);
    Ok(())
}

fn aligned_promise(failures: &mut u32) -> Result<()> {
    let src = "@launch(256)
        @aligned(N = 32)
        kernel k(A: tensor<f32>[1, N], B: tensor<f32>[1, N]) {
            let p = ctx.block.x
            B[0 :+ 1, p * 32 :+ 32] = A[0 :+ 1, p * 32 :+ 32]
        }";
    let launch = |cols: usize| -> Result<Vec<f32>> {
        let a = Arg::new(&noise(cols, 8), 1, cols)?;
        let b = Arg::<f32>::zeroed(1, cols)?;
        let mut slots = Vec::new();
        a.push(&mut slots);
        b.push(&mut slots);
        run(src, "k", (cols / 32) as u32, &mut slots)?;
        b.read()
    };
    let kept = launch(64)?;
    report("@aligned kept runs", kept == noise(64, 8), String::new(), failures);
    let broken = launch(48);
    let trapped = broken.is_err();
    report("@aligned broken traps", trapped, format!("{:?}", broken.err()), failures);
    Ok(())
}

/// Lets buffers of different element types push in one list.
trait Push {
    fn push_to(&self, slots: &mut Vec<u64>);
}

impl<T: DeviceCopy + Default + Clone> Push for Arg<T> {
    fn push_to(&self, slots: &mut Vec<u64>) {
        self.push(slots);
    }
}
