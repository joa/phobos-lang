// What one decode step spends inside attention, against cache length:
//
//   cargo run --release -p phobos-gguf --features cuda --example attndecode
//
// Calls the decode attention path alone, one block at a time through a pass,
// over a working set the size of a real model's caches, next to the card's
// measured copy bandwidth. Lengths are primes so none lines up with a cache
// split or tile boundary, and the last three shapes vary query/key-head
// grouping to isolate whether repeated key reads cost anything.

use std::time::Instant;

use anyhow::Result;

use phobos_gguf::backend::device::DeviceBackend;
use phobos_gguf::backend::{Attn, Backend, Buf, HBuf};

struct Shape {
    name: &'static str,
    blocks: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
}

/// Both benchmarked models, counting only the blocks that keep a growing cache:
/// qwen35 interleaves full attention every fourth block and the rest carry a
/// fixed-size recurrent state instead.
const SHAPES: [Shape; 5] = [
    Shape {
        name: "minicpm5-1b",
        blocks: 24,
        n_head: 16,
        n_kv: 2,
        head_dim: 128,
    },
    Shape {
        name: "Qwen3.5-0.8B",
        blocks: 6,
        n_head: 8,
        n_kv: 2,
        head_dim: 256,
    },
    Shape {
        name: "group 1, same grid",
        blocks: 24,
        n_head: 16,
        n_kv: 16,
        head_dim: 128,
    },
    Shape {
        name: "group 1, same cache",
        blocks: 24,
        n_head: 2,
        n_kv: 2,
        head_dim: 128,
    },
    Shape {
        name: "group 2",
        blocks: 24,
        n_head: 16,
        n_kv: 8,
        head_dim: 128,
    },
];

/// Primes, spaced like the powers of two they sit next to, so none lines up
/// with a cache split or tile boundary.
const LENGTHS: [usize; 8] = [37, 67, 131, 257, 521, 1031, 2053, 4099];

/// Repeat `batch` until `secs` have passed and return the mean seconds an
/// iteration took. `batch` reports how many iterations it ran, and has to leave
/// the device drained: the calls are asynchronous, so a batch that has not been
/// read back is not work the clock has seen.
fn repeat_for(secs: f64, mut batch: impl FnMut() -> Result<usize>) -> Result<f64> {
    let start = Instant::now();
    let mut iters = 0;
    while start.elapsed().as_secs_f64() < secs {
        iters += batch()?;
    }
    Ok(start.elapsed().as_secs_f64() / iters as f64)
}

/// Device-to-device copy bandwidth, measured rather than assumed: the point of
/// the comparison is what this card does, not what the box claims.
///
/// The copy runs for `warm_secs` before anything is timed. A card sitting at
/// its idle clock needs seconds of work to reach boost, and a measurement
/// taken across the ramp says more about the clock than about the kernel.
fn copy_bandwidth(backend: &dyn Backend, warm_secs: f64) -> Result<f64> {
    let elems = 32 << 20;
    let src = backend.zeroed(elems)?;
    let dst = backend.alloc(elems)?;
    let mut sink = [0.0f32; 1];
    let mut copies = |passes: usize| -> Result<usize> {
        for _ in 0..passes {
            backend.copy(src, 0, dst, 0, elems)?;
        }
        backend.read(dst, &mut sink)?;
        Ok(passes)
    };
    repeat_for(warm_secs, || copies(64))?;
    let start = Instant::now();
    let passes = copies(200)?;
    let secs = start.elapsed().as_secs_f64();
    for buf in [src, dst] {
        backend.release(buf);
    }
    Ok(2.0 * passes as f64 * (elems * size_of::<f32>()) as f64 / secs)
}

/// One decode step's worth of attention: every block's call, over that block's
/// own cache, so nothing is served out of L2 that would not be.
fn step(
    backend: &dyn Backend,
    caches: &[(HBuf, HBuf)],
    q: Buf,
    out: Buf,
    spec: Attn,
) -> Result<()> {
    backend.begin_pass(spec.rows)?;
    for &(keys, values) in caches {
        backend.attention(q, keys, values, spec, out)?;
    }
    backend.end_pass()
}

/// Restricts the sweep to one cache length, for an external profiler (ncu)
/// that needs to isolate a single kernel launch. Unset by default, so a
/// normal run sweeps every length.
fn length_wanted(length: usize) -> bool {
    std::env::var("PHOBOS_ATTNDECODE_LENGTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .is_none_or(|want| want == length)
}

fn main() -> Result<()> {
    let backend = DeviceBackend::new()?;
    let bandwidth = copy_bandwidth(&backend, 6.0)?;
    println!("device copy bandwidth {:.0} GB/s\n", bandwidth / 1e9);

    for shape in &SHAPES {
        // Same restriction, by shape name substring: an external profiler
        // isolating one kernel launch wants one shape as well as one length.
        if std::env::var("PHOBOS_ATTNDECODE_SHAPE").is_ok_and(|s| !shape.name.contains(&s)) {
            continue;
        }
        let group = shape.n_head / shape.n_kv;
        let kv_width = shape.n_kv * shape.head_dim;
        println!(
            "{} : {} blocks, {} heads over {} kv heads (group {}), head dim {}",
            shape.name, shape.blocks, shape.n_head, shape.n_kv, group, shape.head_dim,
        );
        println!(
            "{:>6}{:>12}{:>11}{:>9}{:>10}{:>10}",
            "cache", "attn/step", "distinct", "floor", "vs floor", "us/1k pos",
        );

        let q = backend.zeroed(shape.n_head * shape.head_dim)?;
        let out = backend.alloc(shape.n_head * shape.head_dim)?;
        let mut sink = [0.0f32; 1];
        let mut previous: Option<(usize, f64)> = None;

        for &length in &LENGTHS {
            if !length_wanted(length) {
                continue;
            }
            let caches = (0..shape.blocks)
                .map(|_| {
                    Ok((
                        backend.zeroed_h(length * kv_width)?,
                        backend.zeroed_h(length * kv_width)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let spec = Attn {
                rows: 1,
                start_pos: length - 1,
                n_head: shape.n_head,
                n_kv: shape.n_kv,
                head_dim: shape.head_dim,
            };

            // Timed by wall clock rather than by a fixed count, so a cheap
            // shape gets as many passes as an expensive one gets seconds.
            let mut steps = |count: usize| -> Result<usize> {
                for _ in 0..count {
                    step(&backend, &caches, q, out, spec)?;
                }
                backend.read(out, &mut sink)?;
                Ok(count)
            };
            repeat_for(0.25, || steps(1))?;
            let micros = repeat_for(0.5, || steps(32))? * 1e6;

            // Every cached key and value of every block, counted once. The
            // caches are f16, so a position is half what it was.
            let bytes = shape.blocks * length * kv_width * 2 * size_of::<u16>();
            let floor = bytes as f64 / bandwidth * 1e6;
            // The slope against the row above, which drops whatever fixed cost
            // the launches carry and leaves what a position itself is worth.
            let slope = previous.map_or(f64::NAN, |(was, took)| {
                (micros - took) / (length - was) as f64 * 1e3
            });
            previous = Some((length, micros));

            println!(
                "{length:>6}{micros:>11.1}u{:>10.2}M{floor:>8.1}u{:>9.1}x{slope:>10.1}",
                bytes as f64 / (1 << 20) as f64,
                micros / floor,
            );

            for (keys, values) in caches {
                backend.release_h(keys);
                backend.release_h(values);
            }
        }
        for buf in [q, out] {
            backend.release(buf);
        }
        println!();
    }
    Ok(())
}
