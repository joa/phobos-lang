// Where a prompt pass's time goes, and what moves it:
//
//   cargo run --release -p phobos-inference --features cuda --example ppsweep
//
// Decoding was fixed by turning the projection's mapping around. Prompt
// processing runs a different kernel, `q8_mma`, measured here at the shapes a
// 512-token prompt asks for.
//
// Reports achieved integer throughput, since a batched projection is compute
// bound where the matvec was bandwidth bound: the weight is read once per row
// tile, so what matters is how much of the tensor core's 89 TOPS the tiling
// reaches.

use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result};
use cust::function::Function;
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

use phobos_gguf::compute::quantize_row;
use phobos_onnx::abi::{self, KernelArg};

/// The batched projection as it ships: an output tile in both directions, with
/// the block scales applied to a shared-memory accumulator every 32 elements.
const SRC_MMA: &str = "\
@launch({BLOCK})
@autotune(TM in [{TM}], TN in [{TN}])
{ALIGNED}
kernel q8_mma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
              W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
              C: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, K, 32) {
    let b = kt / 32
    let a = A[pm * TM :+ TM, kt :+ 32]
    let w = W[pn * TN :+ TN, kt :+ 32]
    let ws = WS[b :+ 1, pn * TN :+ TN]
    let da = AS[pm * TM :+ TM, b :+ 1]
    acc += f32(dot_t(a, w)) * ws * da
  }
  C[pm * TM :+ TM, pn * TN :+ TN] = acc
}
";

/// The same contraction as one `qmma_t`, which folds the Q8_0 block scales in
/// so the whole of `k` can be handed over at once.
///
/// The accumulators then live in registers across all of `k` rather than in a
/// shared-memory tile rewritten every 32 elements, and neither operand is
/// staged: the `m8n8k16` fragment layout is what they are already in.
const SRC_QMMA: &str = "\
@launch({BLOCK})
@autotune(TM in [{TM}], TN in [{TN}])
{ALIGNED}
kernel q8_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
               W: tensor<i8>[N, K], WS: tensor<f32>[KB, N],
               C: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm * TM :+ TM, pn * TN :+ TN] = qmma_t(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                           W[pn * TN :+ TN, :], WS[:, pn * TN :+ TN])
}
";

/// The projections a 512-token prompt runs, with how many of each a pass does.
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (1024, 7168, 24, "ffn gate/up"),
    (3584, 1024, 24, "ffn down"),
    (1024, 8448, 18, "delta stacked"),
    (2048, 1024, 24, "mixer out"),
];

/// Row tile, column tile and the CTA carrying them.
const TILES: &[(usize, usize, usize)] = &[
    (8, 64, 256),
    (16, 64, 256),
    (32, 64, 256),
    (64, 64, 256),
    (32, 32, 256),
    (64, 32, 256),
    (16, 128, 256),
    (32, 128, 256),
];

/// The same for `qmma_t`, where the tile is only bounded by the register file
/// and the shared-memory result, so it can be much wider.
const QTILES: &[(usize, usize, usize)] = &[
    (128, 256, 256),
    (128, 128, 256),
    (64, 256, 256),
    // The same tiles on a smaller CTA. A patch is capped by the register file
    // either way, so these hold the warps per multiprocessor and change only
    // how many blocks the grid has to spread over it.
    (128, 256, 128),
    (128, 128, 128),
    (64, 128, 128),
    (256, 128, 128),
    (128, 128, 64),
    (128, 64, 64),
    (256, 256, 128),
];

/// Rows in the batch, which is what `-p 512` hands the projection.
const ROWS: usize = 512;

fn compile(source: &str, subs: &[(&str, usize)], aligned: &str) -> Result<Module> {
    let mut text = source.replace("{ALIGNED}", aligned);
    for (name, value) in subs {
        text = text.replace(&format!("{{{name}}}"), &value.to_string());
    }
    let ctx = phobos_base::context::Context::default();
    let ptx = phobos_lang::compile(&ctx, &text).context("compiling the batched projection")?;
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

    println!("a {ROWS}-row projection: microseconds, and the integer throughput it reaches");
    println!("(* fails the host reference)\n");

    let mut shipped_millis = 0.0;
    let mut best_millis = 0.0;

    for &(k, n, per_pass, what) in SHAPES {
        let blocks = k / 32;
        let qs: Vec<i8> = (0..k * n).map(|_| (next() * 127.0) as i8).collect();
        let scales: Vec<f32> = (0..blocks * n).map(|_| next().abs() + 0.01).collect();
        let activation: Vec<f32> = (0..ROWS * k).map(|_| next()).collect();

        let mut qa = vec![0i8; ROWS * k];
        let mut das = vec![0.0f32; ROWS * blocks];
        for r in 0..ROWS {
            for b in 0..blocks {
                let from = r * k + b * 32;
                das[r * blocks + b] =
                    quantize_row(&activation[from..from + 32], &mut qa[from..from + 32]);
            }
        }

        // Only the first row is checked: it exercises the same arithmetic as
        // every other and a full reference at these sizes costs minutes.
        let mut truth = vec![0.0f32; n];
        for (j, t) in truth.iter_mut().enumerate() {
            let row = &qs[j * k..(j + 1) * k];
            let mut total = 0.0f32;
            for (b, (a_block, w_block)) in qa[..k]
                .chunks_exact(32)
                .zip(row.chunks_exact(32))
                .enumerate()
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
        let out_dev = DeviceBuffer::<f32>::zeroed(ROWS * n)?;
        let (a_ptr, das_ptr) = (
            qa_dev.as_device_ptr().as_raw(),
            das_dev.as_device_ptr().as_raw(),
        );
        let (w_ptr, s_ptr) = (
            w_dev.as_device_ptr().as_raw(),
            s_dev.as_device_ptr().as_raw(),
        );
        let out_ptr = out_dev.as_device_ptr().as_raw();
        let macs = (ROWS * k * n) as f64;

        println!("{what}  [k={k} n={n}]  x{per_pass}/pass");
        let mut shipped = f64::MAX;
        let mut best = f64::MAX;
        let mut best_at = ("", 0, 0, 0);
        for (name, source, tiles) in [("q8_mma", SRC_MMA, TILES), ("q8_qmma", SRC_QMMA, QTILES)] {
            for &(tm, tn, block) in tiles {
                let aligned = format!("@aligned(M = {tm}, N = {tn})");
                // A tile whose accumulator does not fit in shared memory fails
                // to build, which is itself a result: it is why `q8_mma`'s tile
                // has to stay small.
                let Ok(module) = compile(
                    source,
                    &[("BLOCK", block), ("TM", tm), ("TN", tn)],
                    &aligned,
                ) else {
                    println!(
                        "{:>18}  {:>9}",
                        format!("{name} {tm}x{tn}/{block}"),
                        "no fit"
                    );
                    continue;
                };
                let func = module.get_function(name)?;
                let operands = [
                    (a_ptr, [ROWS as i64, k as i64]),
                    (das_ptr, [ROWS as i64, blocks as i64]),
                    (w_ptr, [n as i64, k as i64]),
                    (s_ptr, [blocks as i64, n as i64]),
                    (out_ptr, [ROWS as i64, n as i64]),
                ];
                let grid = (ROWS.div_ceil(tm) as u32, n.div_ceil(tn) as u32, 1);
                let go = |times: usize| -> Result<()> {
                    for _ in 0..times {
                        // SAFETY: every buffer outlives this call.
                        unsafe { launch(&stream, &func, &operands, grid, block)? };
                    }
                    stream.synchronize().map_err(Into::into)
                };
                go(2)?;
                let start = Instant::now();
                go(20)?;
                let micros = start.elapsed().as_secs_f64() * 1e6 / 20.0;

                let mut got = vec![0.0f32; ROWS * n];
                out_dev.copy_to(&mut got)?;
                let error = truth
                    .iter()
                    .zip(&got)
                    .map(|(&t, &g)| (t - g).abs() / t.abs().max(1.0))
                    .fold(0.0f32, f32::max);
                let ok = error < 1e-3;
                println!(
                    "{:>18}  {micros:>9.1} us  {:>6.2} TOPS{}",
                    format!("{name} {tm}x{tn}/{block}"),
                    2.0 * macs / (micros * 1e6),
                    if ok { "" } else { "  *" }
                );
                if ok && micros < best {
                    best = micros;
                    best_at = (name, tm, tn, block);
                }
                if ok && name == "q8_mma" && (tm, tn, block) == (8, 64, 256) {
                    shipped = micros;
                }
            }
        }
        let (name, tm, tn, block) = best_at;
        println!("  best {name} {tm}x{tn}/{block} at {best:.1} us, shipped {shipped:.1}\n");
        shipped_millis += shipped * per_pass as f64 / 1e3;
        best_millis += best * per_pass as f64 / 1e3;
    }

    println!(
        "projection time in a {ROWS}-row pass: shipped {shipped_millis:.1} ms, best {best_millis:.1} ms"
    );
    Ok(())
}
