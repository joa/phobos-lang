// What a small tile matmul costs, written two ways:
//
//   cargo run --release -p phobos-inference --features cuda --example dotform
//
// The delta rule's chunk scan is 18 GFLOP a pass in 22 ms, 0.8 TFLOP/s where
// the same card's tiled matmul reaches three to five, so the obvious suspicion
// is that its dots are on a slower path: they are single `dot(a, b)` calls over
// the whole contraction, where the fast ones in the attention rewrite are a
// `+=` inside a loop over slices of it.
//
// They are not. Both forms measure 1.25 TFLOP/s at the shape the scan uses, so
// the contraction being in one piece costs nothing and there is no rewrite to
// do. What is left is the shape: `[16, 128]` by `[128, 32]` is 512 output
// elements on a 256-thread CTA, and a matmul that small does not reach a large
// one's rate however it is spelled.
//
// Which puts it back on the chunk size, and so on shared memory: the scan holds
// `[C, head_dim]` and `[head_dim, TN]` tiles, and a chunk of 32 over two column
// slices would double the tile and halve the count, but it is 64 KB against the
// 48 static shared memory allows.

use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result};
use cust::function::Function;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

use phobos_onnx::abi::{self, KernelArg};

/// The scan's shape: a chunk of 16 positions, a head of 128, a state slice of
/// 32 columns, and 32 chunks to a 512-token pass.
const C: usize = 16;
const D: usize = 128;
const TN: usize = 32;
const CHUNKS: usize = 32;
/// Programs the scan launches, and delta-net blocks a pass runs.
const PROGRAMS: usize = 64;
const BLOCKS_PER_PASS: usize = 18;

/// One `dot` over the whole contraction, which is what the scan does today.
const WHOLE: &str = "\
@launch(256)
@autotune(C in [16], D in [128], TN in [32])
@aligned(N = C, QW = D, SD = D, OW = TN)
kernel dots(K: tensor<f32>[N, QW], S: tensor<f32>[SD, TN], O: tensor<f32>[N, OW]) {
  var st: tile<f32>[D, TN] = S[0 :+ D, 0 :+ TN]
  var acc: tile<f32>[C, TN] = 0.0
  for c in range(0, N, C) {
    var kst: tile<f32>[C, TN] = dot(K[c :+ C, 0 :+ D], st)
    acc = acc + kst
  }
  O[0 :+ C, 0 :+ TN] = acc
}
";

/// The same contraction as an accumulating loop over slices of it, which is the
/// shape the register-blocked path recognizes.
const SLICED: &str = "\
@launch(256)
@autotune(C in [16], D in [128], TN in [32], DT in [16])
@aligned(N = C, QW = DT, SD = DT, OW = TN)
kernel dots(K: tensor<f32>[N, QW], S: tensor<f32>[SD, TN], O: tensor<f32>[N, OW]) {
  var acc: tile<f32>[C, TN] = 0.0
  for c in range(0, N, C) {
    var kst: tile<f32>[C, TN] = 0.0
    for d in range(0, D, DT) {
      kst += dot(K[c :+ C, d :+ DT], S[d :+ DT, 0 :+ TN])
    }
    acc = acc + kst
  }
  O[0 :+ C, 0 :+ TN] = acc
}
";

fn compile(source: &str) -> Result<Module> {
    let ctx = phobos_base::context::Context::default();
    let ptx = phobos_lang::compile(&ctx, source).context("compiling")?;
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
    grid: u32,
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
        stream.launch(func, (grid, 1, 1), (256, 1, 1), 0, &raw)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ctx = cust::quick_init().context("initializing CUDA")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let rows = CHUNKS * C;
    let k: Vec<f32> = (0..rows * D).map(|i| (i % 13) as f32 * 0.1).collect();
    let s: Vec<f32> = (0..D * TN).map(|i| (i % 7) as f32 * 0.1).collect();
    let k_dev = DeviceBuffer::from_slice(&k)?;
    let s_dev = DeviceBuffer::from_slice(&s)?;
    let o_dev = DeviceBuffer::<f32>::zeroed(C * TN)?;
    let operands = [
        (k_dev.as_device_ptr().as_raw(), [rows as i64, D as i64]),
        (s_dev.as_device_ptr().as_raw(), [D as i64, TN as i64]),
        (o_dev.as_device_ptr().as_raw(), [C as i64, TN as i64]),
    ];
    // One contraction per chunk, over every program the scan launches.
    let flops = 2.0 * (C * D * TN * CHUNKS * PROGRAMS) as f64;

    println!("a [{C}, {D}] by [{D}, {TN}] contraction, {CHUNKS} chunks on {PROGRAMS} programs\n");
    for (name, source) in [("one dot", WHOLE), ("sliced and accumulated", SLICED)] {
        let module = compile(source)?;
        let func = module.get_function("dots")?;
        let go = |times: usize| -> Result<()> {
            for _ in 0..times {
                // SAFETY: every buffer outlives this call.
                unsafe { launch(&stream, &func, &operands, PROGRAMS as u32)? };
            }
            stream.synchronize().map_err(Into::into)
        };
        go(4)?;
        let start = Instant::now();
        go(50)?;
        let micros = start.elapsed().as_secs_f64() * 1e6 / 50.0;
        println!(
            "{name:>24}  {micros:>8.1} us  {:>5.2} TFLOP/s  {:>5.1} ms/pass",
            flops / (micros * 1e6),
            micros * BLOCKS_PER_PASS as f64 / 1e3
        );
    }
    Ok(())
}
