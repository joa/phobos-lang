// What a grid barrier costs against what a launch costs:
//
//   cargo run --release -p phobos-kernels --features cuda --example barrier_bench
//
// This is the measurement docs/megakernel.md turns on. A decode step is a deep,
// narrow chain of tiny kernels, so the question is whether a kernel spanning
// several stages and separating them with `grid_barrier` beats the same stages
// launched one at a time. Both rows below do the same arithmetic on the same
// buffer the same number of times; only the boundary differs.
//
// The launch row goes through a CUDA graph, not plain launches, because that is
// what the GGUF backend records a pass as.

use anyhow::Result;
use cust::prelude::*;
use phobos_kernels::launch::persistent_grid;
use phobos_kernels::{compile, cuda_ok, push_descriptor};

/// One stage of a pass: scale a slice. Deliberately trivial, so what the rows
/// measure is the boundary and not the body.
const STAGE_SRC: &str = "\
@launch(256)
@autotune(TILE in [1024])
kernel stage(X: tensor<f32>[M, N]) {
  let p = program_id(0)
  var x = X[0 :+ 1, p * TILE :+ TILE]
  X[0 :+ 1, p * TILE :+ TILE] = x * 1.0000001 + 0.000000001
}
";

/// The same stage STEPS times in one launch, with a grid barrier between them.
/// Every block strides the whole buffer, since a persistent grid is sized by the
/// card and not by the data.
const PERSISTENT_SRC: &str = "\
@launch(256)
@persistent
@autotune(TILE in [1024], STEPS in [484], BLOCKS in [48])
kernel staged(X: tensor<f32>[M, N], BAR: tensor<i32>[2]) {
  let p = program_id(0)
  for s in range(0, STEPS) {
    for t in range(p * TILE, N, BLOCKS * TILE) {
      var x = X[0 :+ 1, t :+ TILE]
      X[0 :+ 1, t :+ TILE] = x * 1.0000001 + 0.000000001
    }
    grid_barrier(BAR)
  }
}
";

const STEPS: usize = 484;
const TILE: usize = 1024;
const THREADS: u32 = 256;

fn main() -> Result<()> {
    let _ctx = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let elems = 8 * 1024;
    let x = DeviceBuffer::from_slice(&vec![0.0f32; elems])?;
    let bar = DeviceBuffer::from_slice(&[0i32, 0])?;

    // ---- the launch row: STEPS nodes in a graph, as a pass is recorded ----
    let stage = compile(STAGE_SRC, &[("TILE", TILE)], "stage")?;
    let stage_fn = stage.get_function("stage")?.to_raw();
    let blocks = (elems / TILE) as u32;
    let mut slots = Vec::new();
    push_descriptor(&mut slots, x.as_device_ptr().as_raw(), [1, elems as i64]);
    let graph_millis = time_graph(&stream, stage_fn, blocks, &mut slots)?;
    println!(
        "graph chain, {STEPS} nodes of {blocks} blocks: {graph_millis:7.3} ms -> {:5.2} us per node",
        graph_millis / STEPS as f64 * 1000.0
    );

    // ---- the barrier row: one persistent launch, STEPS barriers ----
    // The grid has to be known at compile time here, since the kernel strides by
    // it, so ask the driver first with a throwaway compile at the same shape.
    let probe = compile(
        PERSISTENT_SRC,
        &[("TILE", TILE), ("STEPS", 1), ("BLOCKS", 48)],
        "staged probe",
    )?;
    // SAFETY: the function comes from the module just compiled, which is held
    // in scope for the rest of main.
    let (grid, per_sm) =
        unsafe { persistent_grid(probe.get_function("staged")?.to_raw(), THREADS, 0)? };
    println!("persistent grid: {grid} blocks ({per_sm} per SM)");

    let staged = compile(
        PERSISTENT_SRC,
        &[("TILE", TILE), ("STEPS", STEPS), ("BLOCKS", grid as usize)],
        "staged",
    )?;
    let staged_fn = staged.get_function("staged")?.to_raw();
    let mut slots = Vec::new();
    push_descriptor(&mut slots, x.as_device_ptr().as_raw(), [1, elems as i64]);
    push_descriptor(&mut slots, bar.as_device_ptr().as_raw(), [1, 2]);
    let barrier_millis = time_launch(&stream, staged_fn, grid, &mut slots)?;
    println!(
        "persistent, {STEPS} barriers:            {barrier_millis:7.3} ms -> {:5.2} us per barrier",
        barrier_millis / STEPS as f64 * 1000.0
    );

    // The barrier tensor must come back to rest, or a later launch inherits a
    // partial arrival count and hangs.
    let mut rest = [0i32; 2];
    bar.copy_to(&mut rest)?;
    println!(
        "barrier state after the run: count {}, generation {}",
        rest[0], rest[1]
    );
    anyhow::ensure!(rest[0] == 0, "the arrival counter did not return to zero");

    let mut out = [0.0f32; 1];
    x.index(0).copy_to(&mut out[..])?;
    println!("x[0] = {} (both rows touched it)", out[0]);
    Ok(())
}

const REPS: usize = 20;

fn time_graph(
    stream: &Stream,
    func: cust::sys::CUfunction,
    blocks: u32,
    slots: &mut [u64],
) -> Result<f64> {
    let mut graph: cust::sys::CUgraph = std::ptr::null_mut();
    // SAFETY: graph is written on success and destroyed below.
    cuda_ok(
        unsafe { cust::sys::cuGraphCreate(&mut graph, 0) },
        "creating a graph",
    )?;
    let mut argv: Vec<*mut std::ffi::c_void> = slots
        .iter()
        .map(|s| s as *const u64 as *mut u64 as *mut std::ffi::c_void)
        .collect();
    let params = cust::sys::CUDA_KERNEL_NODE_PARAMS {
        func,
        gridDimX: blocks,
        gridDimY: 1,
        gridDimZ: 1,
        blockDimX: THREADS,
        blockDimY: 1,
        blockDimZ: 1,
        sharedMemBytes: 0,
        kernelParams: argv.as_mut_ptr(),
        extra: std::ptr::null_mut(),
    };
    let mut nodes: Vec<cust::sys::CUgraphNode> = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let deps = nodes.last().copied();
        let mut node: cust::sys::CUgraphNode = std::ptr::null_mut();
        // SAFETY: deps points at the previous node of the same graph.
        cuda_ok(
            unsafe {
                cust::sys::cuGraphAddKernelNode(
                    &mut node,
                    graph,
                    deps.as_ref().map_or(std::ptr::null(), |d| d as *const _),
                    usize::from(deps.is_some()),
                    &params,
                )
            },
            "adding a graph node",
        )?;
        nodes.push(node);
    }
    let mut exec: cust::sys::CUgraphExec = std::ptr::null_mut();
    // SAFETY: exec is written on success and destroyed below.
    cuda_ok(
        unsafe {
            cust::sys::cuGraphInstantiate_v2(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        },
        "instantiating a graph",
    )?;

    let launch = |stream: &Stream| -> Result<()> {
        // SAFETY: exec was instantiated above and the stream outlives the call.
        cuda_ok(
            unsafe { cust::sys::cuGraphLaunch(exec, stream.as_inner()) },
            "launching a graph",
        )
    };
    launch(stream)?;
    stream.synchronize()?;
    let start = std::time::Instant::now();
    for _ in 0..REPS {
        launch(stream)?;
    }
    stream.synchronize()?;
    let millis = start.elapsed().as_secs_f64() * 1e3 / REPS as f64;

    // SAFETY: both handles were created here and are destroyed once.
    unsafe {
        cust::sys::cuGraphExecDestroy(exec);
        cust::sys::cuGraphDestroy(graph);
    }
    Ok(millis)
}

fn time_launch(
    stream: &Stream,
    func: cust::sys::CUfunction,
    blocks: u32,
    slots: &mut [u64],
) -> Result<f64> {
    let mut argv: Vec<*mut std::ffi::c_void> = slots
        .iter()
        .map(|s| s as *const u64 as *mut u64 as *mut std::ffi::c_void)
        .collect();
    let launch = |stream: &Stream, argv: &mut Vec<*mut std::ffi::c_void>| -> Result<()> {
        // SAFETY: argv points into slots, which outlives the call, and matches
        // phobos-mlir's exploded-memref ABI.
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
            "launching a persistent kernel",
        )
    };
    launch(stream, &mut argv)?;
    stream.synchronize()?;
    let start = std::time::Instant::now();
    for _ in 0..REPS {
        launch(stream, &mut argv)?;
    }
    stream.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / REPS as f64)
}
