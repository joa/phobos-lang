// What a prompt pass's attention could cost as two matmuls:
//
//   cargo run --release -p phobos-gguf --features cuda --example attnsweep
//
// A block of queries at head dimension 256 cannot hold much: the query tile,
// the accumulator and the key and value tiles are all `[BR, 256]` f32, so 48 KB
// of shared memory caps `BR` at eight and a `[8, 8]` score tile puts 64 output
// elements on a 256-thread CTA. Materializing the scores makes both halves
// ordinary matmuls with no such cap.
//
// This measures the two halves alone, to see whether they are worth the pass
// over memory the fused form does not pay.

use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result};
use cust::function::Function;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

use phobos_kernels::abi::{self, KernelArg};

/// `S = Q K^T` for one head: the query block against the whole cache.
const SRC_SCORES: &str = "\
@launch({BLOCK})
@autotune(TM in [{TM}], TN in [{TN}], TK in [{TK}])
{TC}
@aligned(M = TM, N = TN, K = TK)
kernel scores(Q: tensor<f32>[M, K], W: tensor<f32>[N, K], S: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, K, TK) {
    var a = Q[pm * TM :+ TM, kt :+ TK]
    var b = W[pn * TN :+ TN, kt :+ TK]
    acc = acc + dot_t(a, b)
  }
  S[pm * TM :+ TM, pn * TN :+ TN] = acc
}
";

/// The same scores against a key block already transposed to `[D, NK]`.
///
/// `dot_t` cannot accumulate in place, so the loop above builds a fresh tile
/// every step and adds it; `dot` can, which is what reaches the pipelined and
/// tensor-core paths. Transposing the cache once a layer is the price.
const SRC_SCORES_T: &str = "@launch({BLOCK})
@autotune(TM in [{TM}], TN in [{TN}], TK in [{TK}])
{TC}
@aligned(M = TM, N = TN, K = TK)
kernel scores_t(Q: tensor<f32>[M, K], W: tensor<f32>[K, N], S: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, K, TK) {
    var a = Q[pm * TM :+ TM, kt :+ TK]
    var b = W[kt :+ TK, pn * TN :+ TN]
    acc += dot(a, b)
  }
  S[pm * TM :+ TM, pn * TN :+ TN] = acc
}
";

/// The same again with the head on the grid's third axis.
///
/// Both operands are then column windows of a wider tensor at an offset the
/// compiler cannot bound, so this measures what the bounds mask costs against
/// gathering each head into a buffer of its own first.
const SRC_SCORES_H: &str = "@launch({BLOCK})
@autotune(TM in [{TM}], TN in [{TN}], TK in [{TK}], D in [256], NH in [8])
{TC}
@aligned(M = TM, N = TN, K = TK)
kernel scores_h(Q: tensor<f32>[M, QW], W: tensor<f32>[KW, N], S: tensor<f32>[HM, N]) {
  let pm = program_id(0)
  let pn = program_id(1)
  let h = program_id(2)
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, D, TK) {
    var a = Q[pm * TM :+ TM, h * D + kt :+ TK]
    var b = W[h * D + kt :+ TK, pn * TN :+ TN]
    acc += dot(a, b)
  }
  S[h * M + pm * TM :+ TM, pn * TN :+ TN] = acc
}
";

/// `O = P V`: the probabilities against the values, an ordinary matmul.
const SRC_MIX: &str = "@launch({BLOCK})
@autotune(TM in [{TM}], TN in [{TN}], TK in [{TK}])
{TC}
@aligned(M = TM, N = TN, K = TK)
kernel mix(P: tensor<f32>[M, K], V: tensor<f32>[K, N], O: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[TM, TN] = 0.0
  for kt in range(0, K, TK) {
    var a = P[pm * TM :+ TM, kt :+ TK]
    var b = V[kt :+ TK, pn * TN :+ TN]
    acc += dot(a, b)
  }
  O[pm * TM :+ TM, pn * TN :+ TN] = acc
}
";

/// Rows in the batch, keys in the cache, and the head dimension.
const ROWS: usize = 512;
const HEAD_DIM: usize = 256;
/// Heads of one attention block, and how many blocks a pass runs.
const HEADS: usize = 8;
const BLOCKS_PER_PASS: usize = 6;

/// Output tile, contraction tile and CTA, with and without the tensor cores.
const TILES: &[(usize, usize, usize, usize)] = &[
    (32, 32, 16, 256),
    (64, 64, 16, 256),
    (64, 64, 32, 256),
    (128, 64, 16, 256),
    (64, 128, 16, 256),
    (128, 128, 16, 256),
];

fn compile(source: &str, subs: &[(&str, usize)], tensorcore: bool) -> Result<Module> {
    let mut text = source.replace("{TC}", if tensorcore { "@tensorcore" } else { "" });
    for (name, value) in subs {
        text = text.replace(&format!("{{{name}}}"), &value.to_string());
    }
    phobos_kernels::compile(&text, &[], "compiling the attention matmul")
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

    // Wide enough for every head, so the head-on-the-grid kernel reads the
    // same layout the model has rather than a flattered one.
    let wide = HEADS * HEAD_DIM;
    let q: Vec<f32> = (0..ROWS * wide).map(|_| next()).collect();
    let q_dev = DeviceBuffer::from_slice(&q)?;
    let k_dev = DeviceBuffer::from_slice(&q)?;
    let s_dev = DeviceBuffer::<f32>::zeroed(HEADS * ROWS * ROWS)?;
    let (q_ptr, k_ptr) = (
        q_dev.as_device_ptr().as_raw(),
        k_dev.as_device_ptr().as_raw(),
    );
    let s_ptr = s_dev.as_device_ptr().as_raw();

    println!("one head of {ROWS} queries at head dimension {HEAD_DIM}");
    println!("({HEADS} heads a block, {BLOCKS_PER_PASS} blocks a pass)");
    println!(
        "the fused kernel measures 5.0 ms a block, 30 ms a pass
"
    );

    // Each half on its own: they are different shapes, only one of them can be
    // written to accumulate in place, and the head is either a grid axis or a
    // gather into a buffer of its own.
    for tensorcore in [false, true] {
        println!("{}", if tensorcore { "@tensorcore" } else { "f32" });
        for &(tm, tn, tk, block) in TILES {
            let subs = &[("TM", tm), ("TN", tn), ("TK", tk), ("BLOCK", block)];
            let mut cell = String::new();
            let mut pass_millis = 0.0;
            for (name, source, contract, across) in [
                ("scores", SRC_SCORES, HEAD_DIM, ROWS),
                ("scores_t", SRC_SCORES_T, HEAD_DIM, ROWS),
                ("scores_h", SRC_SCORES_H, HEAD_DIM, ROWS),
                ("mix", SRC_MIX, ROWS, HEAD_DIM),
            ] {
                let Ok(module) = compile(source, subs, tensorcore) else {
                    cell.push_str(&format!("{:>16}", "no fit"));
                    continue;
                };
                let func = module.get_function(name)?;
                // The head kernel sees the whole width and picks its own
                // window; the others are handed one head's buffer.
                let head_on_grid = name == "scores_h";
                let (qw, kw, sw) = if head_on_grid {
                    (wide as i64, wide as i64, (HEADS * ROWS) as i64)
                } else {
                    (contract as i64, contract as i64, ROWS as i64)
                };
                let operands = [
                    (q_ptr, [ROWS as i64, qw]),
                    (k_ptr, [kw, across as i64]),
                    (s_ptr, [sw, across as i64]),
                ];
                let grid = ((ROWS / tm) as u32, (across / tn) as u32, 1);
                let go = |times: usize| -> Result<()> {
                    for _ in 0..times {
                        // SAFETY: every buffer outlives this call.
                        unsafe { launch(&stream, &func, &operands, grid, block)? };
                    }
                    stream.synchronize().map_err(Into::into)
                };
                go(4)?;
                let start = Instant::now();
                go(50)?;
                let micros = start.elapsed().as_secs_f64() * 1e6 / 50.0;
                let flops = 2.0 * (ROWS * contract * across) as f64;
                cell.push_str(&format!("{micros:>9.1} us{:>6.2}T", flops / (micros * 1e6)));
                // A pass is the head-on-the-grid scores plus the mix.
                if head_on_grid || name == "mix" {
                    pass_millis += micros * (HEADS * BLOCKS_PER_PASS) as f64 / 1e3;
                }
            }
            println!(
                "{:>12} {cell}  {pass_millis:>6.1} ms/pass",
                format!("{tm}x{tn}x{tk}")
            );
        }
        println!();
    }
    Ok(())
}
