// Where the quantized matvec's time goes, and what moves it:
//
//   cargo run --release -p phobos-inference --features cuda --example q8sweep
//
// Profiling a decode step once said `q8_dp4a` was 84% of its device time, and
// three plausible reasons turned out to be wrong. What the numbers say:
//
// - Achieved bandwidth tracks the grid, which is `n / TN` blocks: 38 GB/s at 32
//   blocks, 111 at 112, 159 at 7760, against roughly 427 the card can sustain.
//   Below a couple of hundred blocks the kernel is starved of parallelism, and
//   most of a decode step's projections are down there.
// - Widening the tile to do more work per barrier is worse everywhere, since it
//   shrinks the grid further.
// - Rearranging the weight so a program's tile is contiguous rather than `TN`
//   scattered 32-byte pieces does not help, so the ceiling above a few hundred
//   blocks is not the row scatter.
// - Pairing the tile with the CTA so no thread idles does not help either, even
//   with the k-split restoring the grid: 256 outputs on 256 threads measures 2
//   to 3 times slower than 32 on 128. Idle threads were not the constraint.
//
// Splitting the contraction buys blocks without shrinking the tile, and that
// does help: each program takes a slice of `k` and a second pass sums the
// partials.
//
// `qdot_t` helps far more. It folds the Q8_0 block scales into the contraction
// so the whole of `k` goes in at once, which lets a warp own an output and its
// lanes divide `k` rather than a thread owning an output and walking `k`: the
// weight read becomes 512 contiguous bytes a warp, nothing is staged, and the
// k-split stops being worth anything. It is 2.4x to 7.4x here, and split 1 wins
// on every shape.
//
// Launches the kernels directly rather than through `DeviceBackend`, since the
// block size is baked into both the kernel attribute and the launch.

use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result};
use cust::function::Function;
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

use phobos_gguf::compute::quantize_row;
use phobos_onnx::abi::{self, KernelArg};

/// The kernel as it ships, with the whole contraction in one program.
const SRC_WHOLE: &str = "\
@launch({BLOCK})
@autotune(TN in [{TN}])
{ALIGNED}
kernel q8_dp4a(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
               C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  var acc: tile<f32>[1, TN] = 0.0
  for kt in range(0, K, 32) {
    let b = kt / 32
    let a = A[0 :+ 1, kt :+ 32]
    let w = W[pn * TN :+ TN, kt :+ 32]
    let ws = WS[b :+ 1, pn * TN :+ TN]
    let da = AS[0 :+ 1, b :+ 1]
    acc += f32(dot_t(a, w)) * ws * da
  }
  C[0 :+ 1, pn * TN :+ TN] = acc
}
";

/// The same arithmetic with the contraction split across the grid's second axis.
///
/// Program `(pn, ps)` covers outputs `pn * TN` and the `k` slice starting at
/// `ps * SLICE`, and writes its partial sum to its own row of `P`. The grid is
/// `SPLITS` times larger for the same tile, which is the point: the tile stays
/// wide enough to be efficient and the block count no longer follows from `n`.
const SRC_SPLIT: &str = "\
@launch({BLOCK})
@autotune(TN in [{TN}], SLICE in [{SLICE}])
{ALIGNED}
kernel q8_split(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
                P: tensor<f32>[S, N]) {
  let pn = program_id(0)
  let ps = program_id(1)
  let from = ps * SLICE
  var acc: tile<f32>[1, TN] = 0.0
  for kt in range(from, from + SLICE, 32) {
    let b = kt / 32
    let a = A[0 :+ 1, kt :+ 32]
    let w = W[pn * TN :+ TN, kt :+ 32]
    let ws = WS[b :+ 1, pn * TN :+ TN]
    let da = AS[0 :+ 1, b :+ 1]
    acc += f32(dot_t(a, w)) * ws * da
  }
  P[ps :+ 1, pn * TN :+ TN] = acc
}

@launch(256)
@autotune(RT in [128])
kernel q8_reduce(P: tensor<f32>[S, N], C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  var acc: tile<f32>[1, RT] = 0.0
  for s in range(0, S, 1) {
    acc = acc + P[s :+ 1, pn * RT :+ RT]
  }
  C[0 :+ 1, pn * RT :+ RT] = acc
}
";

/// The same contraction as one `qdot_t`, which folds the block scales in so
/// the whole of `k` can be handed over at once.
///
/// The mapping turns around: a warp owns one output and its lanes divide `k`,
/// instead of a thread owning one output and walking `k`. That is what makes
/// the weight read contiguous, and it leaves nothing to stage.
const SRC_QDOT: &str = "\
@launch({BLOCK})
@autotune(TN in [{TN}])
@aligned(N = TN, K = 32)
kernel q8_qdot(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[N, KB],
               C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = qdot_t(A[0 :+ 1, :], AS[0 :+ 1, :],
                                    W[pn * TN :+ TN, :], WS[pn * TN :+ TN, :])
}
";

/// `qdot_t` over a slice of `k`, for the narrow projections whose output alone
/// does not fill the grid.
const SRC_QDOT_SPLIT: &str = "\
@launch({BLOCK})
@autotune(TN in [{TN}], SLICE in [{SLICE}], SB in [{SB}])
@aligned(N = TN, K = SLICE, KB = SB)
kernel q8_qdot_split(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                     W: tensor<i8>[N, K], WS: tensor<f32>[N, KB],
                     P: tensor<f32>[S, N]) {
  let pn = program_id(0)
  let ps = program_id(1)
  let from = ps * SLICE
  let fb = ps * SB
  P[ps :+ 1, pn * TN :+ TN] = qdot_t(A[0 :+ 1, from :+ SLICE], AS[0 :+ 1, fb :+ SB],
                                     W[pn * TN :+ TN, from :+ SLICE], WS[pn * TN :+ TN, fb :+ SB])
}

@launch(256)
@autotune(RT in [128])
kernel q8_reduce(P: tensor<f32>[S, N], C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  var acc: tile<f32>[1, RT] = 0.0
  for s in range(0, S, 1) {
    acc = acc + P[s :+ 1, pn * RT :+ RT]
  }
  C[0 :+ 1, pn * RT :+ RT] = acc
}
";

/// The projections a decode step runs, and how many of each per token.
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (3584, 1024, 24, "ffn down"),
    (2048, 1024, 24, "mixer out"),
    (1024, 3584, 48, "ffn gate/up"),
    (1024, 6144, 18, "delta qkv"),
    (1024, 2048, 18, "delta gate"),
    (1024, 16, 36, "delta alpha/beta"),
    (1024, 248320, 1, "lm head"),
];

/// Output tile and the CTA that carries it.
///
/// `dot_t` puts one thread on each output column, so a CTA wider than the tile
/// runs the contraction on `TN` of its threads and idles the rest. The shipped
/// 32-wide tile on 128 threads uses a quarter of them. Pairing the two is the
/// obvious thing to try and it was not tried before, because the earlier sweep
/// widened the tile without splitting `k` and so kept shrinking the grid.
const TILES: &[(usize, usize)] = &[(32, 128), (64, 64), (128, 128), (256, 256), (512, 512)];
const SPLITS: &[usize] = &[1, 2, 4, 8, 16, 32];

/// The same for `qdot_t`, where the tile is warps rather than threads: a CTA
/// of 256 carries eight outputs at a time, so a 16-wide tile gives each warp
/// two.
const QTILES: &[(usize, usize)] = &[(8, 256), (16, 256), (32, 256), (8, 128), (4, 128)];

fn compile(source: &str, subs: &[(&str, usize)], tiles_evenly: bool) -> Result<Module> {
    let claim = if tiles_evenly {
        "@aligned(N = TN, K = 32)"
    } else {
        "@aligned(K = 32)"
    };
    compile_claimed(source, subs, claim)
}

fn compile_claimed(source: &str, subs: &[(&str, usize)], claim: &str) -> Result<Module> {
    let mut text = source.replace("{ALIGNED}", claim);
    for (name, value) in subs {
        text = text.replace(&format!("{{{name}}}"), &value.to_string());
    }
    let ctx = phobos_base::context::Context::default();
    let ptx = phobos_lang::compile(&ctx, &text).context("compiling the quantized matvec")?;
    Module::from_ptx(&ptx, &[]).context("loading PTX")
}

/// One launch of a kernel over tensor operands given as (pointer, extents).
///
/// # Safety
/// Every operand must outlive the launch and match the kernel's expected shape.
unsafe fn launch(
    stream: &Stream,
    func: &Function,
    operands: &[(u64, [i64; 2])],
    grid: (u32, u32, u32),
    block: usize,
) -> Result<()> {
    let mut args: Vec<KernelArg> = Vec::new();
    for (ptr, dims) in operands {
        abi::push_tensor_descriptor(&mut args, *ptr, dims);
    }
    let mut slots: Vec<u64> = args.iter().map(|a| a.slot()).collect();
    let raw: Vec<*mut c_void> = slots
        .iter_mut()
        .map(|s| s as *mut u64 as *mut c_void)
        .collect();
    unsafe {
        stream.launch(func, grid, (block as u32, 1, 1), 0, &raw)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ctx = cust::quick_init().context("initializing CUDA")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 8388608.0 - 1.0
    };

    println!("microseconds per projection, by output tile / CTA and k-split");
    println!("(* fails the host reference; split 1 is the shipped shape)\n");
    let mut shipped_micros = 0.0;
    let mut best_micros = 0.0;

    for &(k, n, per_token, what) in SHAPES {
        let blocks = k / 32;
        let qs: Vec<i8> = (0..k * n).map(|_| (next() * 127.0) as i8).collect();
        let scales: Vec<f32> = (0..blocks * n).map(|_| next().abs() + 0.01).collect();
        let activation: Vec<f32> = (0..k).map(|_| next()).collect();

        // Quantized here rather than by launching the quantize kernel, so this
        // measures the projection alone.
        let mut qa = vec![0i8; k];
        let mut das = vec![0.0f32; blocks];
        for (b, chunk) in activation.chunks_exact(32).enumerate() {
            das[b] = quantize_row(chunk, &mut qa[b * 32..(b + 1) * 32]);
        }

        let mut truth = vec![0.0f32; n];
        for (j, t) in truth.iter_mut().enumerate() {
            let row = &qs[j * k..(j + 1) * k];
            let mut total = 0.0f32;
            for (b, (a_block, w_block)) in qa.chunks_exact(32).zip(row.chunks_exact(32)).enumerate()
            {
                let partial: i32 = a_block
                    .iter()
                    .zip(w_block)
                    .map(|(&x, &q)| x as i32 * q as i32)
                    .sum();
                total += partial as f32 * scales[b * n + j] * das[b];
            }
            *t = total;
        }

        let w_dev = DeviceBuffer::from_slice(&qs)?;
        let s_dev = DeviceBuffer::from_slice(&scales)?;
        let qa_dev = DeviceBuffer::from_slice(&qa)?;
        let das_dev = DeviceBuffer::from_slice(&das)?;
        let out_dev = DeviceBuffer::<f32>::zeroed(n)?;
        let part_dev = DeviceBuffer::<f32>::zeroed(n * SPLITS[SPLITS.len() - 1])?;
        let (a_ptr, das_ptr) = (
            qa_dev.as_device_ptr().as_raw(),
            das_dev.as_device_ptr().as_raw(),
        );
        let (w_ptr, s_ptr) = (
            w_dev.as_device_ptr().as_raw(),
            s_dev.as_device_ptr().as_raw(),
        );
        let (out_ptr, part_ptr) = (
            out_dev.as_device_ptr().as_raw(),
            part_dev.as_device_ptr().as_raw(),
        );
        // What the kernel has to read however it is tiled: the weight bytes plus
        // one scale per block of 32.
        let weight_bytes = (k * n + blocks * n * 4) as f64;
        let repeats = if k * n > 1 << 24 { 20 } else { 200 };

        println!("{what}  [k={k} n={n}]  x{per_token}/token");
        print!("{:>10}", "");
        for &s in SPLITS {
            print!("{:>12}", format!("split {s}"));
        }
        println!();

        let mut shipped = f64::MAX;
        let mut best = f64::MAX;
        let mut best_at = (0, 0, 0);
        for &(tile, block) in TILES {
            print!("{:>10}", format!("{tile}/{block}"));
            for &splits in SPLITS {
                let slice = k / splits;
                // A split has to land on Q8_0 block boundaries, and there is no
                // point splitting past one block per program.
                if splits > 1 && (!slice.is_multiple_of(32) || slice == 0) {
                    print!("{:>12}", "-");
                    continue;
                }
                let tiles_evenly = n.is_multiple_of(tile);
                let elapsed = if splits == 1 {
                    let module =
                        compile(SRC_WHOLE, &[("BLOCK", block), ("TN", tile)], tiles_evenly)?;
                    let func = module.get_function("q8_dp4a")?;
                    let operands = [
                        (a_ptr, [1i64, k as i64]),
                        (das_ptr, [1, blocks as i64]),
                        (w_ptr, [n as i64, k as i64]),
                        (s_ptr, [blocks as i64, n as i64]),
                        (out_ptr, [1, n as i64]),
                    ];
                    let grid = (n.div_ceil(tile) as u32, 1, 1);
                    let go = |times: usize| -> Result<()> {
                        for _ in 0..times {
                            // SAFETY: every buffer outlives this call.
                            unsafe { launch(&stream, &func, &operands, grid, block)? };
                        }
                        stream.synchronize().map_err(Into::into)
                    };
                    go(4)?;
                    let start = Instant::now();
                    go(repeats)?;
                    start.elapsed().as_secs_f64() * 1e6 / repeats as f64
                } else {
                    let module = compile(
                        SRC_SPLIT,
                        &[("BLOCK", block), ("TN", tile), ("SLICE", slice)],
                        tiles_evenly,
                    )?;
                    let split_fn = module.get_function("q8_split")?;
                    let reduce_fn = module.get_function("q8_reduce")?;
                    let split_ops = [
                        (a_ptr, [1i64, k as i64]),
                        (das_ptr, [1, blocks as i64]),
                        (w_ptr, [n as i64, k as i64]),
                        (s_ptr, [blocks as i64, n as i64]),
                        (part_ptr, [splits as i64, n as i64]),
                    ];
                    let reduce_ops = [
                        (part_ptr, [splits as i64, n as i64]),
                        (out_ptr, [1, n as i64]),
                    ];
                    let split_grid = (n.div_ceil(tile) as u32, splits as u32, 1);
                    let reduce_grid = (n.div_ceil(128) as u32, 1, 1);
                    let go = |times: usize| -> Result<()> {
                        for _ in 0..times {
                            // SAFETY: every buffer outlives this call.
                            unsafe {
                                launch(&stream, &split_fn, &split_ops, split_grid, block)?;
                                launch(&stream, &reduce_fn, &reduce_ops, reduce_grid, 256)?;
                            }
                        }
                        stream.synchronize().map_err(Into::into)
                    };
                    go(4)?;
                    let start = Instant::now();
                    go(repeats)?;
                    start.elapsed().as_secs_f64() * 1e6 / repeats as f64
                };

                let mut got = vec![0.0f32; n];
                out_dev.copy_to(&mut got)?;
                let error = truth
                    .iter()
                    .zip(&got)
                    .map(|(&t, &g)| (t - g).abs() / t.abs().max(1.0))
                    .fold(0.0f32, f32::max);
                let ok = error < 1e-3;
                print!(
                    "{:>12}",
                    format!("{elapsed:.1}{}", if ok { "" } else { "*" })
                );
                if ok && elapsed < best {
                    best = elapsed;
                    best_at = (tile, block, splits);
                }
                if ok && tile == 32 && block == 128 && splits == 1 {
                    shipped = elapsed;
                }
            }
            println!();
        }
        // qdot_t wants the weight scales row-major, so a lane's scale load sits
        // next to its neighbours'; the shipped kernels want them block-major.
        let mut row_scales = vec![0.0f32; blocks * n];
        for b in 0..blocks {
            for j in 0..n {
                row_scales[j * blocks + b] = scales[b * n + j];
            }
        }
        let rs_dev = DeviceBuffer::from_slice(&row_scales)?;
        let rs_ptr = rs_dev.as_device_ptr().as_raw();

        let mut qbest = f64::MAX;
        let mut qbest_at = (0usize, 0usize, 0usize);
        for &(tile, block) in QTILES {
            print!("{:>10}", format!("{tile}/{block}"));
            for &splits in SPLITS {
                let slice = k / splits;
                if !n.is_multiple_of(tile) || (splits > 1 && !slice.is_multiple_of(32)) {
                    print!("{:>12}", "-");
                    continue;
                }
                let elapsed = if splits == 1 {
                    let module = compile_claimed(
                        SRC_QDOT,
                        &[("BLOCK", block), ("TN", tile)],
                        "@aligned(N = TN, K = 32)",
                    )?;
                    let func = module.get_function("q8_qdot")?;
                    let operands = [
                        (a_ptr, [1i64, k as i64]),
                        (das_ptr, [1, blocks as i64]),
                        (w_ptr, [n as i64, k as i64]),
                        (rs_ptr, [n as i64, blocks as i64]),
                        (out_ptr, [1, n as i64]),
                    ];
                    let grid = (n.div_ceil(tile) as u32, 1, 1);
                    let go = |times: usize| -> Result<()> {
                        for _ in 0..times {
                            // SAFETY: every buffer outlives this call.
                            unsafe { launch(&stream, &func, &operands, grid, block)? };
                        }
                        stream.synchronize().map_err(Into::into)
                    };
                    go(4)?;
                    let start = Instant::now();
                    go(repeats)?;
                    start.elapsed().as_secs_f64() * 1e6 / repeats as f64
                } else {
                    let module = compile_claimed(
                        SRC_QDOT_SPLIT,
                        &[
                            ("BLOCK", block),
                            ("TN", tile),
                            ("SLICE", slice),
                            ("SB", slice / 32),
                        ],
                        "@aligned(N = TN, K = SLICE, KB = SB)",
                    )?;
                    let split_fn = module.get_function("q8_qdot_split")?;
                    let reduce_fn = module.get_function("q8_reduce")?;
                    let split_ops = [
                        (a_ptr, [1i64, k as i64]),
                        (das_ptr, [1, blocks as i64]),
                        (w_ptr, [n as i64, k as i64]),
                        (rs_ptr, [n as i64, blocks as i64]),
                        (part_ptr, [splits as i64, n as i64]),
                    ];
                    let reduce_ops = [
                        (part_ptr, [splits as i64, n as i64]),
                        (out_ptr, [1, n as i64]),
                    ];
                    let split_grid = (n.div_ceil(tile) as u32, splits as u32, 1);
                    let reduce_grid = (n.div_ceil(128) as u32, 1, 1);
                    let go = |times: usize| -> Result<()> {
                        for _ in 0..times {
                            // SAFETY: every buffer outlives this call.
                            unsafe {
                                launch(&stream, &split_fn, &split_ops, split_grid, block)?;
                                launch(&stream, &reduce_fn, &reduce_ops, reduce_grid, 256)?;
                            }
                        }
                        stream.synchronize().map_err(Into::into)
                    };
                    go(4)?;
                    let start = Instant::now();
                    go(repeats)?;
                    start.elapsed().as_secs_f64() * 1e6 / repeats as f64
                };

                let mut got = vec![0.0f32; n];
                out_dev.copy_to(&mut got)?;
                let error = truth
                    .iter()
                    .zip(&got)
                    .map(|(&t, &g)| (t - g).abs() / t.abs().max(1.0))
                    .fold(0.0f32, f32::max);
                let ok = error < 1e-3;
                print!(
                    "{:>12}",
                    format!("{elapsed:.1}{}", if ok { "" } else { "*" })
                );
                if ok && elapsed < qbest {
                    qbest = elapsed;
                    qbest_at = (tile, block, splits);
                }
            }
            println!();
        }

        let gb_per_s = |micros: f64| weight_bytes / (micros * 1e3);
        println!(
            "  shipped {shipped:.1} us ({:.0} GB/s)  ->  dot_t {}/{} split {}: {best:.1} us \
             ({:.0} GB/s)  ->  qdot_t {}/{} split {}: {qbest:.1} us ({:.0} GB/s), {:.2}x\n",
            best_at.0,
            best_at.1,
            best_at.2,
            gb_per_s(shipped),
            gb_per_s(best),
            qbest_at.0,
            qbest_at.1,
            qbest_at.2,
            gb_per_s(qbest),
            shipped / qbest,
        );
        shipped_micros += shipped * per_token as f64;
        best_micros += best.min(qbest) * per_token as f64;
    }

    println!(
        "per token over these projections: {:.2} ms shipped, {:.2} ms best ({:.2}x)",
        shipped_micros / 1e3,
        best_micros / 1e3,
        shipped_micros / best_micros,
    );
    Ok(())
}
