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
            let p = program_id(0)
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
            let p = program_id(0)
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
        kernel k(B: tensor<u8>[16, 64], Q: tensor<i32>[16, 64], F: tensor<f32>[16, 64], S: tensor<i32>[1, 64]) {
            var b = B[0 :+ 16, 0 :+ 64]
            var q: tile<i32>[16, 64] = (i32(b) >> 4) & 15
            var f: tile<f32>[16, 64] = f32(b)
            Q[0 :+ 16, 0 :+ 64] = q
            F[0 :+ 16, 0 :+ 64] = f
            for j in range(0, 64) {
                S[0, j] = B[15, j] + 1
            }
        }";
    let a = Arg::new(&b, 16, 64)?;
    let q = Arg::<i32>::zeroed(16, 64)?;
    let f = Arg::<f32>::zeroed(16, 64)?;
    let s = Arg::<i32>::zeroed(1, 64)?;
    let mut slots = Vec::new();
    for push in [&a as &dyn Push, &q, &f, &s] {
        push.push_to(&mut slots);
    }
    run(src, "k", 1, &mut slots)?;
    let want_q: Vec<i32> = b.iter().map(|&v| i32::from(v >> 4)).collect();
    let want_f: Vec<f32> = b.iter().map(|&v| f32::from(v)).collect();
    let want_s: Vec<i32> = b[15 * 64..].iter().map(|&v| i32::from(v) + 1).collect();
    report("u8 nibble decode", q.read()? == want_q, String::new(), failures);
    report("u8 to f32", f.read()? == want_f, String::new(), failures);
    report("u8 scalar load", s.read()? == want_s, String::new(), failures);
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
            let p = program_id(0)
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
