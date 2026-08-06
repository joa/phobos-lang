// Where the delta rule's per-position cost goes:
//
//   cargo run --release -p phobos-inference --features cuda --example deltasweep
//
// At a 512-token prompt the recurrence is 43% of the pass and each position
// costs about six microseconds, some ten thousand cycles for two 128-long dot
// products. That is far too much for the arithmetic, so this takes the step
// apart: each variant drops one piece and keeps the rest, and the difference is
// what that piece costs.
//
// The variants are not all correct as recurrences. They keep the same
// loop-carried dependency so the measurement stays honest, but only `full`
// computes the delta rule.

use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result};
use cust::function::Function;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

use phobos_onnx::abi::{self, KernelArg};

/// Positions in the pass, and the shape of one delta-net block.
const ROWS: usize = 512;
const HEADS: usize = 16;
const DIM: usize = 128;
const TN: usize = 16;
/// Delta-net blocks a pass runs.
const BLOCKS_PER_PASS: usize = 18;

/// The step as it ships, and the same with one piece taken out at a time.
///
/// `body` replaces the loop body; every variant reads the same operands and
/// carries `st` across positions.
const VARIANTS: &[(&str, &str)] = &[
    (
        "full",
        "    st = st * dec
    var e: tile<f32>[1, TN] = dot(k, st)
    e = (v - e) * bet
    var kt: tile<f32>[D, 1] = transpose(k)
    st = st + dot(kt, e)
    O[r :+ 1, jn * TN :+ TN] = dot(q, st)",
    ),
    (
        "outer as product",
        "    st = st * dec
    var e: tile<f32>[1, TN] = dot(k, st)
    e = (v - e) * bet
    var kt: tile<f32>[D, 1] = transpose(k)
    st = st + kt * e
    O[r :+ 1, jn * TN :+ TN] = dot(q, st)",
    ),
    (
        "decay folded",
        "    var e: tile<f32>[1, TN] = dot(k, st) * dec
    e = (v - e) * bet
    var kt: tile<f32>[D, 1] = transpose(k)
    st = st * dec + kt * e
    O[r :+ 1, jn * TN :+ TN] = dot(q, st)",
    ),
    (
        "no readout",
        "    st = st * dec
    var e: tile<f32>[1, TN] = dot(k, st)
    e = (v - e) * bet
    var kt: tile<f32>[D, 1] = transpose(k)
    st = st + dot(kt, e)
    O[r :+ 1, jn * TN :+ TN] = e",
    ),
    (
        "no contraction",
        "    st = st * dec
    var e: tile<f32>[1, TN] = v * bet
    var kt: tile<f32>[D, 1] = transpose(k)
    st = st + dot(kt, e)
    O[r :+ 1, jn * TN :+ TN] = e",
    ),
    (
        "no rank-one write",
        "    st = st * dec
    var e: tile<f32>[1, TN] = dot(k, st)
    e = (v - e) * bet
    O[r :+ 1, jn * TN :+ TN] = e",
    ),
    (
        "decay only",
        "    st = st * dec
    O[r :+ 1, jn * TN :+ TN] = v * bet",
    ),
    (
        "loop only",
        "    st = st + dec
    O[r :+ 1, jn * TN :+ TN] = v",
    ),
];

fn source(body: &str) -> String {
    format!(
        "@launch(256)
@autotune(H in [{HEADS}], D in [{DIM}], TN in [{TN}])
@aligned(R = H, D = TN, SD = D)
kernel delta_rule(Q:   tensor<f32>[R, D],
                  K:   tensor<f32>[R, D],
                  V:   tensor<f32>[R, D],
                  DEC: tensor<f32>[R, 1],
                  BET: tensor<f32>[R, 1],
                  S:   tensor<f32>[SD, D],
                  O:   tensor<f32>[R, D]) {{
  let h = program_id(0)
  let jn = program_id(1)
  var st: tile<f32>[D, TN] = S[h * D :+ D, jn * TN :+ TN]

  for t in range(0, R, H) {{
    let r = t + h
    var k = K[r :+ 1, 0 :+ D]
    var q = Q[r :+ 1, 0 :+ D]
    var v = V[r :+ 1, jn * TN :+ TN]
    var dec = DEC[r :+ 1, 0 :+ 1]
    var bet = BET[r :+ 1, 0 :+ 1]

{body}
  }}

  S[h * D :+ D, jn * TN :+ TN] = st
}}
"
    )
}

fn compile(text: &str) -> Result<Module> {
    let ctx = phobos_base::context::Context::default();
    let ptx = phobos_lang::compile(&ctx, text).context("compiling the delta rule")?;
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
        stream.launch(func, grid, (256, 1, 1), 0, &raw)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ctx = cust::quick_init().context("initializing CUDA")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let span = ROWS * HEADS * DIM;
    let gates = ROWS * HEADS;
    let plane = HEADS * DIM * DIM;
    let ones = vec![0.5f32; span.max(plane)];
    let qb = DeviceBuffer::from_slice(&ones[..span])?;
    let sb = DeviceBuffer::from_slice(&ones[..plane])?;
    let gb = DeviceBuffer::from_slice(&ones[..gates])?;
    let ob = DeviceBuffer::<f32>::zeroed(span)?;
    let (q, s, g, o) = (
        qb.as_device_ptr().as_raw(),
        sb.as_device_ptr().as_raw(),
        gb.as_device_ptr().as_raw(),
        ob.as_device_ptr().as_raw(),
    );
    let (r, d) = ((ROWS * HEADS) as i64, DIM as i64);
    let operands = [
        (q, [r, d]),
        (q, [r, d]),
        (q, [r, d]),
        (g, [r, 1]),
        (g, [r, 1]),
        (s, [(HEADS * DIM) as i64, d]),
        (o, [r, d]),
    ];
    let grid = (HEADS as u32, (DIM / TN) as u32, 1);

    println!("a {ROWS}-position delta-net block, {HEADS} heads of {DIM}\n");
    let mut previous = 0.0;
    for (name, body) in VARIANTS {
        let module = compile(&source(body))?;
        let func = module.get_function("delta_rule")?;
        let go = |times: usize| -> Result<()> {
            for _ in 0..times {
                // SAFETY: every buffer outlives this call.
                unsafe { launch(&stream, &func, &operands, grid)? };
            }
            stream.synchronize().map_err(Into::into)
        };
        go(2)?;
        let start = Instant::now();
        go(10)?;
        let micros = start.elapsed().as_secs_f64() * 1e6 / 10.0;
        let step_nanos = micros * 1e3 / ROWS as f64;
        let delta = if previous == 0.0 {
            String::new()
        } else {
            format!("  ({:+.0} us from the one above)", micros - previous)
        };
        previous = micros;
        println!(
            "{name:>18}  {micros:>8.1} us  {step_nanos:>6.0} ns/position  \
             {:>5.1} ms/pass{delta}",
            micros * BLOCKS_PER_PASS as f64 / 1e3
        );
    }
    Ok(())
}
